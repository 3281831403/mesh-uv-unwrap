"""mesh-uv-unwrap: 批量网格 UV 展开工具包（出题用初始失败环境）。

公开契约见 :mod:`unwrap.atlas`。
"""

from .atlas import (
    UnwrapError,
    normalize_island,
    pack_islands,
    project_island,
    rotate_to_fit,
    split_islands,
    texel_density,
    unwrap_batch,
    validate_uv,
)

__all__ = [
    "UnwrapError",
    "split_islands",
    "project_island",
    "normalize_island",
    "pack_islands",
    "rotate_to_fit",
    "texel_density",
    "validate_uv",
    "unwrap_batch",
]
