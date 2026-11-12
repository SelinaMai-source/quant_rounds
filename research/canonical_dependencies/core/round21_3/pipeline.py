"""Round 21.3 stage 编排。"""

from __future__ import annotations

from typing import Any

from .contract import POST_STAGES, STAGES, TRACK_A_STAGES, TRACK_B_STAGES

ALL_STAGES = frozenset(STAGES)


def expand_stages(raw: str) -> list[str]:
    if raw in {"all", "sequence"}:
        return list(STAGES)
    if raw in {"A", "track_a", "trackA"}:
        return list(TRACK_A_STAGES)
    if raw in {"B", "track_b", "trackB"}:
        return list(TRACK_B_STAGES)
    if raw in {"post", "zoo_report"}:
        return list(POST_STAGES)
    if raw in {"stage0", "parity_contract"}:
        return ["parity", "contract"]
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
    if stage == "contract":
        from .freeze import run_freeze

        return run_freeze(**kw)
    if stage == "l1_train":
        from .train import run_l1_train

        return run_l1_train(**kw)
    if stage == "l1_backtest":
        from .backtest import run_l1_backtest

        return run_l1_backtest(**kw)
    if stage == "nav_recon":
        from .nav_recon import run_nav_recon

        return run_nav_recon(**kw)
    if stage == "stats":
        from .stats import run_stats

        return run_stats(**kw)
    if stage == "seed":
        from .seed import run_seed

        return run_seed(**kw)
    if stage == "n1n3":
        from .n1n3 import run_n1n3

        return run_n1n3(**kw)
    if stage == "recover":
        from .recover import run_recover

        return run_recover(**kw)
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
    if stage == "forward":
        from .forward import run_forward

        return run_forward(**kw)
    if stage == "zoo":
        from .zoo import run_zoo

        return run_zoo(**kw)
    if stage == "report":
        from .report import run_report

        return run_report(**kw)
    raise RuntimeError(f"unwired stage: {stage}")
