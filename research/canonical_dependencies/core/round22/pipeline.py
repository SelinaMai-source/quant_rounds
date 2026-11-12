"""Round 22 stage 编排。"""

from __future__ import annotations

from typing import Any

from .contract import STAGES

ALL_STAGES = frozenset(STAGES)


def expand_stages(raw: str) -> list[str]:
    if raw in {"all", "sequence"}:
        return list(STAGES)
    if raw in {"stage0", "parity_contract"}:
        return ["parity", "freeze"]
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
    if stage == "prior":
        from .prior import run_prior

        return run_prior(**kw)
    if stage == "recover":
        from .recover import run_recover

        return run_recover(**kw)
    if stage == "construct_abde":
        from .construct_abde import run_construct_abde

        return run_construct_abde(**kw)
    if stage == "construct_cfgh":
        from .construct_cfgh import run_construct_cfgh

        return run_construct_cfgh(**kw)
    if stage == "pit":
        from .pit import run_pit

        return run_pit(**kw)
    if stage == "evidence":
        from .evidence import run_evidence

        return run_evidence(**kw)
    if stage == "correlate":
        from .correlate import run_correlate

        return run_correlate(**kw)
    if stage == "train":
        from .train import run_train

        return run_train(**kw)
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
    raise RuntimeError(f"unwired stage: {stage}")
