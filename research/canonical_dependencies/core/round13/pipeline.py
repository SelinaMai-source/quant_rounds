"""Round13 stage 编排。freeze 必须先过。parity 失败 STOP。禁止 --stage all 作为隐式搜参。"""

from __future__ import annotations

from typing import Any

from .contract import STAGES
from .freeze import require_frozen

IMPLEMENTED = frozenset(STAGES)


def require_parity() -> dict[str, Any]:
    from .candidates import load_parity_status

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
    if stage == "freeze":
        from .contract import copy_plan
        from .freeze import freeze_manifest

        copy_plan()
        return freeze_manifest(overwrite=False)
    require_frozen()
    kwargs = {"resume": resume, "jobs": jobs, "experiment": experiment}
    if stage == "parity":
        from .candidates import run_parity

        return run_parity(**kwargs)
    require_parity()
    if stage == "weights":
        from .weights import run_weights

        return run_weights(**kwargs)
    if stage == "candidates":
        from .candidates import run_candidates

        return run_candidates(**kwargs)
    if stage == "scenarios":
        from .scenarios import run_scenarios

        return run_scenarios(**kwargs)
    if stage == "bootstrap":
        from .bootstrap import run_bootstrap

        return run_bootstrap(**kwargs)
    if stage == "oob":
        from .oob import run_oob

        return run_oob(**kwargs)
    if stage == "cpcv":
        from .bootstrap import run_cpcv

        return run_cpcv(**kwargs)
    if stage == "spa":
        from .bootstrap import run_spa

        return run_spa(**kwargs)
    if stage == "placebo":
        from .oob import run_placebo

        return run_placebo(**kwargs)
    if stage == "timing":
        from .oob import run_timing

        return run_timing(**kwargs)
    if stage == "execution":
        from .docs import run_execution

        return run_execution(**kwargs)
    if stage == "capacity":
        from .docs import run_capacity

        return run_capacity(**kwargs)
    if stage == "forward":
        from .docs import run_forward

        return run_forward(**kwargs)
    if stage == "promotion":
        from .docs import run_promotion

        return run_promotion(**kwargs)
    if stage == "report":
        from .docs import run_report

        return run_report(**kwargs)
    raise RuntimeError(f"unwired stage: {stage}")


__all__ = ["IMPLEMENTED", "require_parity", "run_stage"]
