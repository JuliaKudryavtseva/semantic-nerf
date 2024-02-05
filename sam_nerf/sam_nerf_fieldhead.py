from typing import Optional

from nerfstudio.field_components.field_heads import FieldHead
from sam_nerf.sam_nerf_fieldheadname import SAMFieldHeadNames

class SAMFieldHead(FieldHead):
    """Clip output

    Args:
        sam_n_dims: embed dimention 
        in_dim: input dimension. If not defined in constructor, it must be set later.
        activation: output head activation
    """

    def __init__(self, sam_n_dims: int, in_dim: Optional[int] = None) -> None:
        super().__init__(in_dim=in_dim, out_dim=sam_n_dims, field_head_name=SAMFieldHeadNames.SAM, activation=None)