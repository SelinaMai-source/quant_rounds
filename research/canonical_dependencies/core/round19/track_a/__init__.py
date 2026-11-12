"""Track A 期权确认。独立 namespace。Stage 0 不跑本目录实验。"""

from __future__ import annotations

from typing import Any

from ..contract import TRACK_A_STAGES


def run_track_stage(stage: str, **kwargs: Any) -> dict[str, Any]:
    if stage not in TRACK_A_STAGES:
        raise RuntimeError(f"unknown track A stage: {stage}")
    if stage == "a-replay":
        from .replay import run_replay

        return run_replay(**kwargs)
    if stage == "a-events":
        from .events import run_events

        return run_events(**kwargs)
    if stage == "a-attribution":
        from .attribution import run_attribution

        return run_attribution(**kwargs)
    if stage == "a-concentration":
        from .concentration import run_concentration

        return run_concentration(**kwargs)
    if stage == "a-equity-control":
        from .equity import run_equity

        return run_equity(**kwargs)
    if stage == "a-match":
        from .match import run_match

        return run_match(**kwargs)
    if stage == "a-placebo":
        from .placebo import run_placebo

        return run_placebo(**kwargs)
    if stage == "a-pit":
        from .pit import run_pit

        return run_pit(**kwargs)
    if stage == "a-quotes":
        from .quotes import run_quotes

        return run_quotes(**kwargs)
    if stage == "a-prediction":
        from .prediction import run_prediction

        return run_prediction(**kwargs)
    if stage == "a-stats":
        from .stats import run_stats

        return run_stats(**kwargs)
    if stage == "a-decision":
        from .decision import run_decision

        return run_decision(**kwargs)
    raise RuntimeError(f"unwired track A stage: {stage}")


__all__ = ["run_track_stage"]
