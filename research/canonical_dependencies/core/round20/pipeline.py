"""Round20 stage 编排。combo = 冻结 registry 后训练；model = A/B/C/D。"""

from __future__ import annotations

from typing import Any

from .common import StageNotImplemented
from .contract import LATER_STAGES, PHASE_A_STAGES, PHASE_BC_STAGES, STAGES

IMPLEMENTED = frozenset((*PHASE_A_STAGES, *PHASE_BC_STAGES, "combo", "model", "seed", "backtest", "stats", "report"))
ALL_STAGES = frozenset(STAGES)


def expand_stages(raw: str) -> list[str]:
    if raw == "all":
        raise RuntimeError("禁止 --stage all。请用 --stage phase_a / sequence / 逗号列表。")
    if raw in {"phase_a", "stage0", "a"}:
        return list(PHASE_A_STAGES)
    if raw in {"phase_b", "phase_bc", "b", "bc"}:
        return list(PHASE_BC_STAGES)
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
    if stage == "inventory":
        from .inventory import run_inventory
        from .read import run_read

        try:
            run_read(**kw)
        except Exception:
            pass
        return run_inventory(**kw)
    if stage == "freeze":
        from .freeze import freeze_manifest
        from .inventory import run_inventory
        from .parity import run_parity
        from .read import run_read

        try:
            run_read(**kw)
        except Exception:
            pass
        try:
            run_inventory(**kw)
        except Exception:
            pass
        try:
            run_parity(**kw)
        except Exception:
            pass
        return freeze_manifest(overwrite=False)
    if stage == "pit":
        from .pit import run_pit

        require_frozen()
        return run_pit(**kw)
    if stage == "ic":
        from .ic import run_ic

        require_frozen()
        return run_ic(**kw)
    if stage == "shape":
        from .shape import run_shape

        require_frozen()
        return run_shape(**kw)
    if stage == "redundancy":
        from .redundancy import run_redundancy

        require_frozen()
        return run_redundancy(**kw)
    if stage == "conditional":
        from .conditional import run_conditional

        require_frozen()
        return run_conditional(**kw)
    if stage == "combo":
        from .combo import run_combo

        require_frozen()
        return run_combo(**kw)
    if stage == "model":
        from .model import run_model_selection

        require_frozen()
        return run_model_selection(**kw)
    if stage == "seed":
        from .seed import run_seed

        require_frozen()
        return run_seed(**kw)
    if stage in {"backtest", "stats"}:
        from .backtest import run_backtest

        require_frozen()
        return run_backtest(**kw)
    if stage == "report":
        from .report import run_report

        require_frozen()
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
