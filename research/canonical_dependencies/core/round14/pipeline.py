"""Round14 stage 编排。parity 失败 STOP。freeze 必须在风险诊断之后、候选之前。"""

from __future__ import annotations

from typing import Any

from .contract import STAGES

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
    kwargs = {"resume": resume, "jobs": jobs, "experiment": experiment}
    if stage == "parity":
        from .parity import run_parity

        return run_parity(**kwargs)
    require_parity()
    if stage == "risk":
        from .risk import run_risk

        return run_risk(**kwargs)
    if stage == "freeze":
        from .contract import copy_plan
        from .freeze import freeze_manifest
        from .risk import require_risk

        require_risk()
        copy_plan()
        return freeze_manifest(overwrite=False)
    from .freeze import require_frozen

    require_frozen()
    if stage == "candidates":
        from .candidates import run_candidates

        return run_candidates(**kwargs)
    if stage == "controls":
        from .candidates import run_controls

        return run_controls(**kwargs)
    if stage == "scenarios":
        from .scenarios import run_scenarios

        return run_scenarios(**kwargs)
    if stage == "execution":
        from .docs import run_execution

        return run_execution(**kwargs)
    if stage == "capacity":
        from .docs import run_capacity

        return run_capacity(**kwargs)
    if stage == "tail":
        from .docs import run_tail

        return run_tail(**kwargs)
    if stage == "bootstrap":
        from .stats import run_bootstrap

        return run_bootstrap(**kwargs)
    if stage == "matched_bootstrap":
        from .stats import run_matched_bootstrap

        return run_matched_bootstrap(**kwargs)
    if stage == "cpcv":
        from .stats import run_cpcv

        return run_cpcv(**kwargs)
    if stage == "spa":
        from .stats import run_spa

        return run_spa(**kwargs)
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
