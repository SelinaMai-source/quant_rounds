"""Round 21.4 stage 编排。"""

from __future__ import annotations

from typing import Any

from .contract import POST_STAGES, STAGES, TRACK_A_STAGES, TRACK_B_STAGES, TRACK_C_STAGES, TRACK_D_STAGES, TRACK_E_STAGES

ALL_STAGES = frozenset(STAGES)


def expand_stages(raw: str) -> list[str]:
    if raw in {"all", "sequence"}:
        return list(STAGES)
    if raw in {"A", "track_a", "trackA"}:
        return list(TRACK_A_STAGES)
    if raw in {"B", "track_b", "trackB"}:
        return list(TRACK_B_STAGES)
    if raw in {"C", "track_c", "trackC"}:
        return list(TRACK_C_STAGES)
    if raw in {"D", "track_d", "trackD"}:
        return list(TRACK_D_STAGES)
    if raw in {"E", "track_e", "trackE"}:
        return list(TRACK_E_STAGES)
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
    if stage == "forward_fix":
        from .forward import run_forward_fix

        return run_forward_fix(**kw)
    if stage == "coverage":
        from .forward import run_coverage

        return run_coverage(**kw)
    if stage == "recover":
        from .recover import run_recover

        return run_recover(**kw)
    if stage == "amatrix_train":
        from .train import run_amatrix_train

        return run_amatrix_train(**kw)
    if stage == "amatrix_backtest":
        from .backtest import run_amatrix_backtest

        return run_amatrix_backtest(**kw)
    if stage == "attribution":
        from .attribution import run_attribution

        return run_attribution(**kw)
    if stage == "recon43":
        from .nav_recon import run_recon43

        return run_recon43(**kw)
    if stage == "combo_build":
        from .combo import run_combo_build

        return run_combo_build(**kw)
    if stage == "combo_run":
        from .combo import run_combo_run

        return run_combo_run(**kw)
    if stage == "failure":
        from .combo import run_failure

        return run_failure(**kw)
    if stage == "wxyz_construct":
        from .construct import run_construct

        return run_construct(**kw)
    if stage == "wxyz_pit":
        from .pit import run_pit

        return run_pit(**kw)
    if stage == "wxyz_evidence":
        from .evidence import run_evidence

        return run_evidence(**kw)
    if stage == "wxyz_fourway":
        from .backtest import run_wxyz_fourway

        return run_wxyz_fourway(**kw)
    if stage == "wxyz_combo":
        from .combo import run_wxyz_combo

        return run_wxyz_combo(**kw)
    if stage == "factory":
        from .docs import run_factory

        return run_factory(**kw)
    if stage == "zoo":
        from .zoo import run_zoo

        return run_zoo(**kw)
    if stage == "report":
        from .report import run_report

        return run_report(**kw)
    raise RuntimeError(f"unwired stage: {stage}")
