"""Round 22.2 stage 编排。"""

from __future__ import annotations

from typing import Any

from .contract import STAGES

ALL_STAGES = frozenset(STAGES)


def expand_stages(raw: str) -> list[str]:
    if raw in {"all", "sequence"}:
        return list(STAGES)
    if raw in {"stage0", "parity_contract"}:
        return ["parity", "freeze"]
    if raw in {"tracks"}:
        return ["interpret", "pit_c", "discover", "evidence_c"]
    if raw in {"heads"}:
        return ["train", "backtest", "ledger"]
    if raw in {"close"}:
        return ["stats", "factory", "zoo", "report"]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    bad = [p for p in parts if p not in ALL_STAGES]
    if bad:
        raise RuntimeError(f"unknown stage: {bad}")
    return parts


def run_stage(stage: str, *, resume: bool = True, jobs: int = 4, dry_run: bool = False, **kwargs: Any) -> dict[str, Any]:
    if stage not in ALL_STAGES:
        raise RuntimeError(f"unknown stage: {stage}")
    if dry_run:
        return {"status": "DRY_RUN", "stage": stage}
    kw = {"resume": resume, "jobs": jobs, **kwargs}
    if stage == "parity":
        from .parity import run_parity

        return run_parity(**kw)
    if stage == "freeze":
        from .freeze import run_freeze

        return run_freeze(**kw)
    if stage == "interpret":
        from .interpret import run_interpret

        return run_interpret(**kw)
    if stage == "pit_c":
        from .pit_c import run_pit_c

        return run_pit_c(**kw)
    if stage == "discover":
        from .discover import run_discover

        return run_discover(**kw)
    if stage == "evidence_c":
        from .evidence_c import run_evidence_c

        return run_evidence_c(**kw)
    if stage == "train":
        from .train import run_train

        return run_train(**kw)
    if stage == "backtest":
        from .backtest import run_backtest

        return run_backtest(**kw)
    if stage == "ledger":
        from .ledger import run_ledger

        return run_ledger(**kw)
    if stage == "stats":
        from .stats import run_stats

        return run_stats(**kw)
    if stage == "factory":
        from .factory import run_factory

        return run_factory(**kw)
    if stage == "zoo":
        from .zoo import run_zoo

        return run_zoo(**kw)
    if stage == "report":
        from .report import run_report

        return run_report(**kw)
    raise RuntimeError(f"unwired stage: {stage}")
