"""Track C 综合决策。两条线独立，禁止 Option × New Factors。Stage 0 不写主报告。"""

from __future__ import annotations

from typing import Any

from ..contract import TRACK_C_STAGES
from .decision import run_decision
from .forward import run_forward
from .report import run_report

HANDLERS = {
    "c-forward": run_forward,
    "c-decision": run_decision,
    "c-report": run_report,
}


def run_track_stage(stage: str, **kwargs: Any) -> dict[str, Any]:
    if stage not in HANDLERS:
        raise RuntimeError(f"unknown track C stage: {stage}")
    return HANDLERS[stage](**kwargs)


__all__ = ["HANDLERS", "TRACK_C_STAGES", "run_track_stage"]
