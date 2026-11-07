"""Round17 stage 编排。parity 失败 STOP。manifest 冻结前不得看发现收益。"""

from __future__ import annotations

from typing import Any

from .contract import N_TRIALS, STAGES

IMPLEMENTED = frozenset(STAGES)


def require_parity() -> dict[str, Any]:
    from .parity import load_parity_status

    parity = load_parity_status()
    if not parity.get("passed"):
        raise RuntimeError("REPRODUCIBILITY_FAIL_OR_MISSING STOP")
    return parity


def require_search_complete() -> dict[str, Any]:
    from .common import load_json
    from ..paths import ROUND17_ROOT

    search = load_json(ROUND17_ROOT / "search" / "search_status.json")
    n = int(search.get("n_completed") or 0)
    if n < N_TRIALS:
        cmd = search.get("resume_command") or "python tools/run_round17.py --stage search --resume --jobs 32"
        raise RuntimeError(f"search incomplete {n}/{N_TRIALS}. resume: {cmd}")
    return search


def run_stage(
    stage: str,
    *,
    resume: bool = True,
    jobs: int = 8,
    trial_id: str = "",
    trial_start: int = 0,
    trial_end: int = 0,
    dry_run: bool = False,
    **_: Any,
) -> dict[str, Any]:
    if stage not in IMPLEMENTED:
        raise RuntimeError(f"unknown stage: {stage}")
    kwargs = {
        "resume": resume, "jobs": jobs, "trial_id": trial_id,
        "trial_start": trial_start, "trial_end": trial_end, "dry_run": dry_run,
    }
    if stage == "read":
        from .read import run_read
        from .contract import copy_plan

        copy_plan()
        return run_read(**kwargs)
    if stage == "parity":
        from .parity import run_parity
        from .read import run_read

        try:
            run_read(**kwargs)
        except Exception:
            pass
        return run_parity(**kwargs)
    require_parity()
    if stage == "preflight":
        from .diversity import run_preflight

        return run_preflight(**kwargs)
    if stage == "inventory":
        from .inventory import run_inventory

        return run_inventory(**kwargs)
    if stage == "factors":
        from .factors import build_new_factors

        return build_new_factors(resume=resume)
    if stage == "pit":
        from .factors import build_new_factors
        from .pit import run_pit

        build_new_factors(resume=True)
        return run_pit(**kwargs)
    if stage == "freeze":
        from .freeze import freeze_manifest
        from .inventory import run_inventory
        from .factors import build_new_factors
        from .pit import run_pit

        run_inventory()
        build_new_factors(resume=True)
        run_pit()
        return freeze_manifest(overwrite=False)
    from .freeze import require_frozen

    if stage == "budget":
        from .freeze import write_budget

        require_frozen()
        return write_budget()
    require_frozen()
    if stage == "search":
        from .search import run_search

        return run_search(**kwargs)
    if stage == "select":
        require_search_complete()
        from .select import run_select

        return run_select(**kwargs)
    if stage == "validate":
        require_search_complete()
        from .select import run_select, run_validate

        run_select(resume=True)
        return run_validate(**kwargs)
    if stage == "fourway":
        from .select import run_validate
        from .fourway import run_fourway

        run_validate(resume=True, jobs=jobs)
        return run_fourway(**kwargs)
    if stage == "stats":
        from .stats import run_stats

        return run_stats(**kwargs)
    if stage == "forward":
        from .stats import run_forward

        return run_forward(**kwargs)
    if stage == "report":
        from .decision import run_decision
        from .docs import run_report

        run_decision()
        return run_report(**kwargs)
    raise RuntimeError(f"unwired stage: {stage}")


__all__ = ["IMPLEMENTED", "require_parity", "run_stage"]
