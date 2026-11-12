"""Round 22.3 stage 编排。"""

from __future__ import annotations

from typing import Any

from .contract import STAGES

ALL_STAGES = frozenset(STAGES)


def expand_stages(raw: str) -> list[str]:
    if raw in {"all", "sequence"}:
        return list(STAGES)
    if raw in {"stage0", "parity_contract"}:
        return ["parity", "freeze"]
    if raw in {"audit"}:
        return ["audit"]
    if raw in {"heads", "replay"}:
        return ["train", "backtest", "ledger"]
    if raw in {"tracks"}:
        return ["c4", "hn2", "fs"]
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
    if stage == "audit":
        from .audit import run_audit

        return run_audit(**kw)
    if stage == "train":
        from .train import run_train

        return run_train(**kw)
    if stage == "backtest":
        from .backtest import run_backtest

        return run_backtest(**kw)
    if stage == "ledger":
        from .ledger import run_ledger

        return run_ledger(**kw)
    if stage == "c4":
        from .c4 import run_c4

        return run_c4(**kw)
    if stage == "hn2":
        from .hn2 import run_hn2

        return run_hn2(**kw)
    if stage == "fs":
        from .fs import run_fs

        return run_fs(**kw)
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
