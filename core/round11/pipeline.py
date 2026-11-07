"""Round11 stage 编排。禁止 --stage all。parity 必须先过。"""

from __future__ import annotations

from typing import Any

from .parity import load_parity_status, run_parity

IMPLEMENTED = frozenset({
    "parity",
    "pit",
    "engine",
    "cost",
    "time",
    "model",
    "placebo",
    "trials",
    "alphaforge",
    "stats",
    "capacity",
    "dossier",
})


def require_parity() -> dict[str, Any]:
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
    determinism_runs: int = 3,
) -> dict[str, Any]:
    _ = experiment
    if stage == "parity":
        return run_parity(resume=resume, determinism_runs=determinism_runs)
    require_parity()
    if stage == "pit":
        from .pit import run_pit
        return run_pit(resume=resume, jobs=jobs)
    if stage == "engine":
        from .engine import run_engine
        return run_engine(resume=resume, jobs=jobs)
    if stage == "cost":
        from .cost import run_cost
        return run_cost(resume=resume, jobs=jobs)
    if stage == "time":
        from .time_stage import run_time
        return run_time(resume=resume, jobs=jobs)
    if stage == "model":
        from .model import run_model
        return run_model(resume=resume, jobs=jobs)
    if stage == "placebo":
        from .placebo import run_placebo
        return run_placebo(resume=resume, jobs=jobs)
    if stage == "trials":
        from .trials import run_trials
        return run_trials(resume=resume, jobs=jobs)
    if stage == "alphaforge":
        from .alphaforge import run_alphaforge
        return run_alphaforge(resume=resume, jobs=jobs)
    if stage == "stats":
        from .stats import run_stats
        return run_stats(resume=resume, jobs=jobs)
    if stage == "capacity":
        from .capacity import run_capacity
        return run_capacity(resume=resume, jobs=jobs)
    if stage == "dossier":
        from .dossier import run_dossier
        return run_dossier(resume=resume, jobs=jobs)
    raise RuntimeError(f"unknown stage: {stage}")


__all__ = ["IMPLEMENTED", "require_parity", "run_stage"]
