"""Round21 stage 编排。Phase A + construct/evidence + 重组/WF/回测 + zoo/report。"""

from __future__ import annotations

from typing import Any

from .common import StageNotImplemented
from .contract import CONSTRUCT_STAGES, LATER_STAGES, PHASE_A_STAGES, RECOMBINE_STAGES, REPORT_STAGES, STAGES

IMPLEMENTED = frozenset(PHASE_A_STAGES + CONSTRUCT_STAGES + RECOMBINE_STAGES + REPORT_STAGES)  # zoo+report 已落地
ALL_STAGES = frozenset(STAGES)


def expand_stages(raw: str) -> list[str]:
    if raw == "all":
        raise RuntimeError("禁止 --stage all。请用 --stage phase_a / sequence / 逗号列表。")
    if raw in {"phase_a", "stage0", "a"}:
        return list(PHASE_A_STAGES)
    if raw in {"recombine", "phase_b"}:
        return list(RECOMBINE_STAGES)
    if raw == "sequence":
        return list(STAGES)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    bad = [p for p in parts if p not in ALL_STAGES]
    if bad:
        raise RuntimeError(f"unknown stage: {bad}")
    return parts


def require_frozen() -> dict[str, Any]:
    from .freeze import require_frozen as _req

    return _req()


def run_stage(stage: str, *, resume: bool = True, jobs: int = 8, dry_run: bool = False, **kwargs: Any) -> dict[str, Any]:
    if stage not in ALL_STAGES:
        raise RuntimeError(f"unknown stage: {stage}")
    kw = {"resume": resume, "jobs": jobs, "dry_run": dry_run, **kwargs}
    if dry_run:
        return {"status": "DRY_RUN", "stage": stage}
    if stage == "read":
        from .contract import copy_plan
        from .read import run_read

        copy_plan()
        return run_read(**kw)
    if stage == "parity":
        from .contract import copy_plan
        from .parity import run_parity
        from .read import run_read

        copy_plan()
        try:
            run_read(**kw)
        except Exception:
            pass
        return run_parity(**kw)
    if stage == "github":
        from .github import run_github

        return run_github(**kw)
    if stage == "audit":
        from .audit import run_audit

        return run_audit(**kw)
    if stage == "hypotheses":
        from .hypotheses import freeze_hypotheses

        return freeze_hypotheses(overwrite=False)
    if stage == "freeze":
        from .audit import run_audit
        from .freeze import freeze_manifest
        from .github import run_github
        from .hypotheses import freeze_hypotheses
        from .parity import run_parity
        from .read import run_read

        try:
            run_read(**kw)
        except Exception:
            pass
        try:
            run_github(**kw)
        except Exception:
            pass
        try:
            run_audit(**kw)
        except Exception:
            pass
        try:
            run_parity(**kw)
        except Exception:
            pass
        freeze_hypotheses(overwrite=False)
        return freeze_manifest(overwrite=False)
    if stage == "construct":
        from .construct import run_construct

        return run_construct(**kw)
    if stage == "pit":
        from .pit import run_pit

        return run_pit(**kw)
    if stage == "evidence":
        from .evidence import run_evidence

        return run_evidence(**kw)
    if stage == "redundancy":
        from .redundancy import run_redundancy

        return run_redundancy(**kw)
    if stage == "conditional":
        from .conditional import run_conditional

        return run_conditional(**kw)
    if stage == "combo":
        from .combo import run_combo

        return run_combo(**kw)
    if stage == "model":
        from .model import run_model

        return run_model(**kw)
    if stage == "seed":
        from .seed import run_seed

        return run_seed(**kw)
    if stage == "backtest":
        from .backtest import run_backtest

        return run_backtest(**kw)
    if stage == "stats":
        from .stats import run_stats

        return run_stats(**kw)
    if stage == "zoo":
        from .zoo import run_zoo

        return run_zoo(**kw)
    if stage == "report":
        from .report import run_report

        return run_report(**kw)
    if stage in LATER_STAGES:
        raise StageNotImplemented(stage)
    raise RuntimeError(f"unwired stage: {stage}")


__all__ = [
    "ALL_STAGES",
    "IMPLEMENTED",
    "expand_stages",
    "require_frozen",
    "run_stage",
]
