"""Track B Round18 取证。独立 namespace。B01 必须 force_retrain 原 14/13 列。

Round18 `train_both_heads` 在 five_extras/ten_extras 为空时直接用 frozen 预测，不会重训。
本 Track 的 R19_B01 走 `train.train_both_heads(..., force_retrain=True)`，不复用该短路。
不改 result/round18 原件。Stage 0 不跑本目录实验。
"""

from __future__ import annotations

from typing import Any

from ..contract import TRACK_B_STAGES
from .audit import run_feature_order, run_folds, run_join_audit, run_labels, run_missing, run_parity
from .allocation import run_allocation
from .heads import run_heads, run_top_change
from .neighbors import run_neighbors
from .repair import run_repair
from .retrain import run_fresh_retrain
from .verdict import run_bug_verdict

HANDLERS = {
    "b-fresh-retrain": run_fresh_retrain,
    "b-parity": run_parity,
    "b-join-audit": run_join_audit,
    "b-labels": run_labels,
    "b-folds": run_folds,
    "b-feature-order": run_feature_order,
    "b-missing": run_missing,
    "b-heads": run_heads,
    "b-top-change": run_top_change,
    "b-allocation": run_allocation,
    "b-neighbors": run_neighbors,
    "b-bug-verdict": run_bug_verdict,
    "b-repair": run_repair,
}


def run_track_stage(stage: str, **kwargs: Any) -> dict[str, Any]:
    if stage not in HANDLERS:
        raise RuntimeError(f"unknown track B stage: {stage}")
    return HANDLERS[stage](**kwargs)


__all__ = ["HANDLERS", "TRACK_B_STAGES", "run_track_stage"]
