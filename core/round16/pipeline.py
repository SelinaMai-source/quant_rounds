"""Round16 stage 编排。parity 失败 STOP。P0 未完成前不做正式策略回测。"""

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


def require_audit() -> dict[str, Any]:
    from .audit import load_audit

    audit = load_audit()
    if audit.get("status") != "DONE":
        raise RuntimeError("AUDIT_MISSING STOP")
    return audit


def run_stage(
    stage: str,
    *,
    resume: bool = True,
    jobs: int = 1,
    experiment: str = "",
    start: int | None = None,
    end: int | None = None,
    underlying: str | None = None,
    dry_run: bool = False,
    verify: bool = False,
) -> dict[str, Any]:
    if stage not in IMPLEMENTED:
        raise RuntimeError(f"unknown stage: {stage}")
    kwargs = {
        "resume": resume,
        "jobs": jobs,
        "experiment": experiment,
        "start": start,
        "end": end,
        "underlying": underlying,
        "dry_run": dry_run,
        "verify": verify,
    }
    if stage == "audit":
        from .audit import run_audit
        from .contract import copy_plan
        from .parity import run_parity

        copy_plan()
        parity = run_parity(**kwargs)
        audit = run_audit(**kwargs)
        audit["parity"] = {k: parity.get(k) for k in ("status", "passed")}
        return audit
    require_parity()
    require_audit()
    if stage == "download":
        from .download import run_download

        return run_download(**kwargs)
    # 后续 stage 在 freeze 前可跑数据，正式回测必须在 freeze 后。
    if stage == "coverage":
        from .coverage import run_coverage

        return run_coverage(**kwargs)
    if stage == "chain":
        from .chain import run_chain

        return run_chain(**kwargs)
    if stage == "quality":
        from .quality import run_quality

        return run_quality(**kwargs)
    if stage == "iv":
        from .iv import run_iv

        return run_iv(**kwargs)
    if stage == "factors":
        from .factors import run_factors

        return run_factors(**kwargs)
    from .freeze import require_frozen

    require_frozen()
    if stage == "incremental":
        from .incremental import run_incremental

        return run_incremental(**kwargs)
    if stage == "redundancy":
        from .redundancy import run_redundancy

        return run_redundancy(**kwargs)
    if stage == "backtest":
        from .backtest import run_backtest

        return run_backtest(**kwargs)
    if stage == "controls":
        from .controls import run_controls

        return run_controls(**kwargs)
    if stage == "stress":
        from .stress import run_stress

        return run_stress(**kwargs)
    if stage == "stats":
        from .stats import run_stats

        return run_stats(**kwargs)
    if stage == "forward":
        from .docs import run_forward

        return run_forward(**kwargs)
    if stage == "report":
        from .docs import run_report

        return run_report(**kwargs)
    raise RuntimeError(f"unwired stage: {stage}")


__all__ = ["IMPLEMENTED", "require_audit", "require_parity", "run_stage"]
