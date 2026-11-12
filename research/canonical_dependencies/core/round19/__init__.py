"""Round19：期权确认 × Round18 管线取证。两条线独立。不自动升 CURRENT。1.0/1.0。"""

from .contract import AUTO_PROMOTE_V5, CURRENT_BASELINE, KEEP_BASELINE_V4, OPTION_CANDIDATE

__all__ = [
    "AUTO_PROMOTE_V5",
    "CURRENT_BASELINE",
    "KEEP_BASELINE_V4",
    "OPTION_CANDIDATE",
]
