"""Round15 stage 编排。parity 失败 STOP。freeze 必须在任何新策略 NAV 之前。"""

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
    if stage == "inventory":
        from .inventory import run_inventory

        return run_inventory(**kwargs)
    if stage == "sdk":
        from .sdk import run_sdk

        return run_sdk(**kwargs)
    if stage == "freeze":
        from .contract import copy_plan
        from .freeze import freeze_manifest
        from .inventory import require_inventory
        from .sdk import require_sdk

        require_inventory()
        require_sdk()
        copy_plan()
        return freeze_manifest(overwrite=False)
    from .freeze import require_frozen

    require_frozen()
    if stage == "factors":
        from .factors import run_factors

        return run_factors(**kwargs)
    if stage == "residual":
        from .residual import run_residual

        return run_residual(**kwargs)
    if stage == "options":
        from .options import run_options

        return run_options(**kwargs)
    if stage == "top1":
        from .top1 import run_top1

        return run_top1(**kwargs)
    if stage == "execution":
        from .execution import run_execution

        return run_execution(**kwargs)
    if stage == "survivorship":
        from .survivorship import run_survivorship

        return run_survivorship(**kwargs)
    if stage == "stats":
        from .stats import run_stats

        return run_stats(**kwargs)
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
