"""Round9 几何：直接复用 Round8。分歧状态是 A/B/C，不是 HIGH/MEDIUM/LOW。"""

from __future__ import annotations

from ..round8.geometry import (
    build_geometry,
    eligible_heads,
    load_fused_alpha,
    load_head_ensembles,
    xs_hard_half,
    xs_tiers,
)
from .contract import DISAGREE_ALIAS, DISAGREE_LABEL, DISAGREE_MULT

__all__ = [
    "DISAGREE_ALIAS",
    "DISAGREE_LABEL",
    "DISAGREE_MULT",
    "build_geometry",
    "eligible_heads",
    "load_fused_alpha",
    "load_head_ensembles",
    "xs_hard_half",
    "xs_tiers",
]
