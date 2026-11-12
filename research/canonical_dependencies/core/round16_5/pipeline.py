"""Round16.5 stage 编排。parity 失败 STOP。freeze 之后才允许 V03/V04 NAV。"""

from __future__ import annotations

from typing import Any

from .contract import STAGES

IMPLEMENTED = frozenset(STAGES)


def require_parity() -> dict[str, Any]:
    from .parity import load_parity_status

    parity = load_parity_status()
    if not parity.get("passed"):
        raise RuntimeError("REPRODUCIBILITY_FAIL_OR_MISSING STOP")
    return parity


def run_stage(
    stage: str,
    *,
    resume: bool = True,
    jobs: int = 1,
    experiment: str = "",
    dry_run: bool = False,
    seed: int | None = None,
    **_: Any,
) -> dict[str, Any]:
    if stage not in IMPLEMENTED:
        raise RuntimeError(f"unknown stage: {stage}")
    kwargs = {"resume": resume, "jobs": jobs, "experiment": experiment, "dry_run": dry_run, "seed": seed}
    if stage == "read":
        from .read import run_read
        from .contract import copy_plan

        copy_plan()
        return run_read(**kwargs)
    if stage == "parity":
        from .parity import run_parity
        from .contract import copy_plan
        from .read import run_read

        copy_plan()
        if not (kwargs.get("skip_read")):
            try:
                run_read(**kwargs)
            except Exception:
                pass
        return run_parity(**kwargs)
    require_parity()
    if stage == "freeze":
        from .freeze import freeze_manifest

        return freeze_manifest(overwrite=False)
    from .freeze import require_frozen

    require_frozen()
    if stage == "attribution":
        from .attribution import run_attribution

        return run_attribution(**kwargs)
    if stage == "placebo":
        from .placebo import run_placebo

        return run_placebo(**kwargs)
    if stage == "equity":
        from .equity import run_equity

        return run_equity(**kwargs)
    if stage == "orthogonal":
        from .orthogonal import run_orthogonal

        return run_orthogonal(**kwargs)
    if stage == "quote":
        from .quote import run_pit, run_quote

        q = run_quote(**kwargs)
        p = run_pit(**kwargs)
        return {"quote": q, "pit": p}
    if stage == "prediction":
        from .prediction import run_prediction

        return run_prediction(**kwargs)
    if stage == "stress":
        from .stress import run_stress

        return run_stress(**kwargs)
    if stage == "stats":
        from .stats import run_stats

        return run_stats(**kwargs)
    if stage == "forward":
        from .stats import run_forward

        return run_forward(**kwargs)
    if stage == "report":
        from .decision import run_decision
        from .docs import run_report

        run_decision()
        return run_report(**kwargs)
    raise RuntimeError(f"unwired stage: {stage}")


__all__ = ["IMPLEMENTED", "require_parity", "run_stage"]
