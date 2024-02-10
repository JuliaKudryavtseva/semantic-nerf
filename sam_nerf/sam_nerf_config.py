"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
SAM-NERF configuration file.

"""
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from sam_nerf.data.sam_nerf_dataparser import SAMDataParserConfig
from sam_nerf.data.sam_dataset import SAMDataset
from sam_nerf.sam_nerf import SAMNerfModelConfig


sam_nerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="sam-nerf",  # TODO: rename to your own model
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=DynamicBatchPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[SAMDataset],
                dataparser=SAMDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=SAMNerfModelConfig(
                eval_num_rays_per_chunk=8192,
            ),
        ),
        optimizers={
            # TODO: consider changing optimizers depending on your custom method
            # "proposal_networks": {
            #     "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            #     "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            # },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
            # "camera_opt": {
            #     "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            #     "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            # },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for SAM Nerf",
)