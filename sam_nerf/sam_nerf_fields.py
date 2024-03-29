"""
Field for compound nerf model, adds scene contraction and image embeddings to model
"""
from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn
import tinycudann as tcnn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames    
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions

from sam_nerf.sam_nerf_fieldheadname import SAMFieldHeadNames
from sam_nerf.sam_nerf_fieldhead import SAMFieldHead


class SAMNerfField(Field):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        base_res: base resolution of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        features_per_level: number of features per level for the hashgrid
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_sam: whether to use sam
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        pass_semantic_gradients: bool = False,
        pass_sam_gradients: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        use_sam: bool = True,
        sam_n_dims: int = 256,
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        # EMBEDDING 4 IMAGES
        self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        # SAM 
        self.use_sam = use_sam
        self.pass_semantic_gradients = pass_semantic_gradients
        self.pass_sam_gradients = pass_sam_gradients
        self.base_res = base_res

        # ENCODDING FOR IMAGES
        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2 - 1, implementation=implementation
        )

        self.mlp_base_grid = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation, 
        )
        self.mlp_base_mlp = MLP(
            in_dim=self.mlp_base_grid.get_out_dim(),
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1+self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )
        self.mlp_base = torch.nn.Sequential(self.mlp_base_grid, self.mlp_base_mlp)

        # SAM
        # per pixel  

        if self.use_sam:    

            self.mlp_sam = MLP(
                in_dim=self.geo_feat_dim, # [bs, 15]
                num_layers=4, #2, #4, #4, #2,
                layer_width= 256, #128, #256, #256, #128,
                out_dim=hidden_dim_transient, # [bs, 64]
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_sam = SAMFieldHead(
                in_dim=self.mlp_sam.get_out_dim(), # [bs, 64]
                sam_n_dims=sam_n_dims # [bs, 256] 
            )
        
        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
            
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)  
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions

        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
            
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        
        outputs = {}

        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        # outputs_shape:  torch.Size([4096, 48]) 
        # ray_samples.frustums.directions.shape:  torch.Size([4096, 48, 3]) 

        outputs_shape = ray_samples.frustums.directions.shape[:-1]
        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                )

        # USE IT IN CASE CAMERA -> NERF -> SAM FEATURES
        #  # clip
        # if self.use_clip:
        #     clips_input = density_embedding.view(-1, self.geo_feat_dim)
            
        #     if not self.pass_clip_gradients:
        #         clips_input = clips_input.detach()

        #     x = self.mlp_clip(clips_input).view(*outputs_shape, -1).to(directions)
        #     outputs[ClipFieldHeadNames.CLIP] = self.field_head_clip(x)

        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
                embedded_appearance.view(-1, self.appearance_embedding_dim),
            ],
            dim=-1,
        )
        
        # rgb:  torch.Size([4096, 48, 3]) 
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})
    
        #  Semantic block
        if self.use_sam:
            sem_block_input = density_embedding.view(-1, self.geo_feat_dim)
            
            if not self.pass_sam_gradients:
                sem_block_input = sem_block_input.detach()

            x = self.mlp_sam(sem_block_input).view(*outputs_shape, -1).to(directions)
            outputs[SAMFieldHeadNames.SAM] = self.field_head_sam(x)

        return outputs
 