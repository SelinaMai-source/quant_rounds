"""Round 22.1 stage 编排。"""

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
        return ["ic_audit", "evidence", "ledger", "recover", "dfamily"]
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
    if stage == "ic_audit":
        from .ic_audit import run_ic_audit

        return run_ic_audit(**kw)
    if stage == "evidence":
        from .evidence import run_evidence

        return run_evidence(**kw)
    if stage == "ledger":
        from .ledger import run_ledger

        return run_ledger(**kw)
    if stage == "recover":
        from .recover import run_recover

        return run_recover(**kw)
    if stage == "dfamily":
        from .dfamily import run_dfamily

        return run_dfamily(**kw)
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
