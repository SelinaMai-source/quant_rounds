"""Round19 stage 编排。共享 runner：Track A/B/C 可独立跑。Stage 0 不跑后续实验。"""

from __future__ import annotations

from typing import Any

from .contract import STAGE0, STAGES, TRACK_A_STAGES, TRACK_B_STAGES, TRACK_C_STAGES, TRACK_STAGES

IMPLEMENTED_STAGE0 = frozenset(STAGE0)
ALL_STAGES = frozenset(STAGES)


def expand_stages(raw: str, *, track: str = "") -> list[str]:
    if raw == "all":
        raise RuntimeError("禁止 --stage all。请用 --stage stage0 / sequence / 逗号列表。")
    if raw in {"stage0", "shared"}:
        return list(STAGE0)
    if raw == "sequence":
        return list(STAGES)
    if raw == "track" and track in TRACK_STAGES:
        return list(TRACK_STAGES[track])
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts and track in TRACK_STAGES:
        return list(TRACK_STAGES[track])
    bad = [p for p in parts if p not in ALL_STAGES]
    if bad:
        raise RuntimeError(f"unknown stage: {bad}")
    return parts


def require_frozen() -> dict[str, Any]:
    from .freeze import require_frozen as _req

    return _req()


def run_stage(
    stage: str,
    *,
    resume: bool = True,
    jobs: int = 8,
    dry_run: bool = False,
    track: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    if stage not in ALL_STAGES:
        raise RuntimeError(f"unknown stage: {stage}")
    kw = {"resume": resume, "jobs": jobs, "dry_run": dry_run, "track": track, **kwargs}
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
    if stage == "freeze":
        from .freeze import freeze_manifest
        from .read import run_read

        try:
            run_read(**kw)
        except Exception:
            pass
        return freeze_manifest(overwrite=False)
    require_frozen()
    if stage in TRACK_A_STAGES:
        from .track_a import run_track_stage

        return run_track_stage(stage, **kw)
    if stage in TRACK_B_STAGES:
        from .track_b import run_track_stage

        return run_track_stage(stage, **kw)
    if stage in TRACK_C_STAGES:
        from .track_c import run_track_stage

        return run_track_stage(stage, **kw)
    raise RuntimeError(f"unwired stage: {stage}")


__all__ = [
    "ALL_STAGES",
    "IMPLEMENTED_STAGE0",
    "expand_stages",
    "require_frozen",
    "run_stage",
]
