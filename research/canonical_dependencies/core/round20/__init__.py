"""Round20：已有股票因子诊断 × 模型对照 × seed 稳健性。不改 O01/B03/90-10。不自动升 CURRENT。1.0/1.0。"""

from .contract import (
    AUTO_PROMOTE_CURRENT,
    CURRENT_BASELINE,
    KEEP_BASELINE_V5,
    PARENT_V4,
    RECOMMEND_NEW_ALPHA_CANDIDATE,
)

__all__ = [
    "AUTO_PROMOTE_CURRENT",
    "CURRENT_BASELINE",
    "KEEP_BASELINE_V5",
    "PARENT_V4",
    "RECOMMEND_NEW_ALPHA_CANDIDATE",
]
