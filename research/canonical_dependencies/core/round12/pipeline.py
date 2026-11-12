"""Round12 stage 编排。禁止 --stage all。intake+freeze 必须先过。"""

from __future__ import annotations

from typing import Any

from .contract import STAGES
from .freeze import require_frozen

IMPLEMENTED = frozenset(STAGES)


def require_parity() -> dict[str, Any]:
    from .parity import load_parity_status

    parity = load_parity_status()
    if not parity.get("passed"):
        raise RuntimeError("REPRODUCIBILITY_FAIL_OR_MISSING STOP")
    return parity


def run_stage(
    stage: str,
    *,
    resume: bool = True,
    jobs: int = 1,
    experiment: str = "",
) -> dict[str, Any]:
    if stage not in IMPLEMENTED:
        raise RuntimeError(f"unknown stage: {stage}")
    if stage == "intake":
        from .intake import run_intake

        return run_intake(resume=resume)
    require_frozen()
    if stage == "parity":
        from .parity import run_parity

        return run_parity(resume=resume)
    require_parity()
    kwargs = {"resume": resume, "jobs": jobs, "experiment": experiment}
    if stage == "universe":
        from .universe import run_universe

        return run_universe(**kwargs)
    if stage == "execution":
        from .execution import run_execution

        return run_execution(**kwargs)
    if stage == "deploy":
        from .deploy import run_deploy

        return run_deploy(**kwargs)
    if stage == "trials":
        from .trials import run_trials

        return run_trials(**kwargs)
    if stage == "dsr":
        from .dsr import run_dsr

        return run_dsr(**kwargs)
    if stage == "pbo":
        from .stats import run_pbo

        return run_pbo(**kwargs)
    if stage == "alphaforge":
        from .alphaforge import run_alphaforge

        return run_alphaforge(**kwargs)
    if stage == "placebo":
        from .placebo import run_placebo

        return run_placebo(**kwargs)
    if stage == "timing-placebo":
        from .timing_placebo import run_timing_placebo

        return run_timing_placebo(**kwargs)
    if stage == "staircase":
        from .stats import run_staircase

        return run_staircase(**kwargs)
    if stage == "capacity":
        from .deploy import run_capacity

        return run_capacity(**kwargs)
    if stage == "shadow":
        from .shadow import run_shadow

        return run_shadow(**kwargs)
    if stage == "controls":
        from .docs import run_controls

        return run_controls(**kwargs)
    if stage == "stats":
        from .stats import run_stats

        return run_stats(**kwargs)
    if stage == "report":
        from .docs import run_report

        return run_report(**kwargs)
    raise RuntimeError(f"unwired stage: {stage}")


__all__ = ["IMPLEMENTED", "require_parity", "run_stage"]
