"""Round 21.2 stage 编排。"""

from __future__ import annotations

from typing import Any

from .contract import STAGES

ALL_STAGES = frozenset(STAGES)


def expand_stages(raw: str) -> list[str]:
    if raw in {"all", "sequence"}:
        return list(STAGES)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    bad = [p for p in parts if p not in ALL_STAGES]
    if bad:
        raise RuntimeError(f"unknown stage: {bad}")
    return parts


def run_stage(stage: str, *, resume: bool = True, jobs: int = 8, dry_run: bool = False, **kwargs: Any) -> dict[str, Any]:
    if stage not in ALL_STAGES:
        raise RuntimeError(f"unknown stage: {stage}")
    kw = {"resume": resume, "jobs": jobs, "dry_run": dry_run, **kwargs}
    if dry_run:
        return {"status": "DRY_RUN", "stage": stage}
    if stage == "parity":
        from .parity import run_parity

        return run_parity(**kw)
    if stage == "audit":
        from .audit import run_audit

        return run_audit(**kw)
    if stage == "lineage":
        from .lineage import run_lineage

        return run_lineage(**kw)
    if stage == "data_audit":
        from .data_audit import run_data_audit

        return run_data_audit(**kw)
    if stage == "hypotheses":
        from .hypotheses import run_hypotheses

        return run_hypotheses(**kw)
    if stage == "construct":
        from .construct import run_construct

        return run_construct(**kw)
    if stage == "pit":
        from .pit import run_pit

        return run_pit(**kw)
    if stage == "evidence":
        from .evidence import run_evidence

        return run_evidence(**kw)
    if stage == "github":
        from .github import run_github

        return run_github(**kw)
    if stage == "correlate":
        from .correlate import run_correlate

        return run_correlate(**kw)
    if stage == "train":
        from .train import run_train

        return run_train(**kw)
    if stage == "interact":
        from .interact import run_interact

        return run_interact(**kw)
    if stage == "backtest":
        from .backtest import run_backtest

        return run_backtest(**kw)
    if stage == "attribution":
        from .attribution import run_attribution

        return run_attribution(**kw)
    if stage == "stats":
        from .stats import run_stats

        return run_stats(**kw)
    if stage == "seed":
        from .seed import run_seed

        return run_seed(**kw)
    if stage == "zoo":
        from .zoo import run_zoo

        return run_zoo(**kw)
    if stage == "report":
        from .report import run_report

        return run_report(**kw)
    raise RuntimeError(f"unwired stage: {stage}")
