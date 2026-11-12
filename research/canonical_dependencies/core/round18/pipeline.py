"""Round18 stage 编排。parity 失败 STOP。freeze 前不得看候选 NAV。"""

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


def run_stage(stage: str, *, resume: bool = True, jobs: int = 8, **_: Any) -> dict[str, Any]:
    if stage not in IMPLEMENTED:
        raise RuntimeError(f"unknown stage: {stage}")
    if stage == "read":
        from .contract import copy_plan
        from .read import run_read

        copy_plan()
        return run_read()
    if stage == "inventory":
        from .inventory import run_inventory
        from .overlap import run_overlap
        from .read import run_read

        run_read()
        run_overlap()
        return run_inventory()
    if stage == "parity":
        from .parity import run_parity
        from .read import run_read

        try:
            run_read()
        except Exception:
            pass
        return run_parity(resume=resume)
    require_parity()
    if stage == "recover":
        from .download import run_recover

        return run_recover(resume=resume)
    if stage == "build":
        from .factors import build_factors

        return build_factors(resume=resume)
    if stage == "pit":
        from .factors import build_factors
        from .pit import run_pit

        build_factors(resume=True)
        return run_pit()
    if stage == "freeze":
        from .factors import build_factors
        from .freeze import freeze_manifest
        from .inventory import run_inventory
        from .pit import run_pit

        run_inventory()
        build_factors(resume=True)
        run_pit()
        return freeze_manifest(overwrite=False)
    from .freeze import require_frozen

    require_frozen()
    if stage == "train":
        from .runner import prepare_panels

        return prepare_panels(resume=resume)
    if stage == "backtest":
        from .runner import run_matrix

        return run_matrix(resume=resume, jobs=jobs)
    if stage == "stats":
        from .stats import run_stats

        return run_stats(resume=resume)
    if stage == "forward":
        from .stats import run_forward

        return run_forward()
    if stage == "report":
        from .decision import run_decision
        from .docs import run_report

        run_decision()
        return run_report()
    raise RuntimeError(f"unwired stage: {stage}")
