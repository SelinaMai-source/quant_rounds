"""Round21：First-Principles Alpha Discovery Factory。

锁定 baseline_v5=R16_O01 与 Round7 5bp。不改 O01/B03/90-10。不自动升 CURRENT。
subsample=colsample=1.0。Parity FAIL 只停正式晋级。
"""

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
