"""
SAM dataset.
"""

from typing import Dict
import numpy as np
import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset

from sam_nerf.data.base_sam_dataparser import SAM_features

class SAMDataset(InputDataset):
    """Dataset that returns images and sam embeddings.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """
    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["sam_features_array"]
    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert "sam" in dataparser_outputs.metadata.keys() and isinstance(self.metadata["sam"], SAM_features)
        self.sam_features = self.metadata["sam"]
        self.device = torch.device
        
    def get_metadata(self, data: Dict) -> Dict:
        filepath_array = self.sam_features.filenames_array[data["image_idx"]]
        filepath_emb = self.sam_features.filenames_emb[data["image_idx"]]
        filepath_emb = str(filepath_emb.stem)        
        sam_features_array = torch.from_numpy(np.load(filepath_array)).cuda() # torch.Size(image.shape)
        return {"sam_features_array": sam_features_array} 