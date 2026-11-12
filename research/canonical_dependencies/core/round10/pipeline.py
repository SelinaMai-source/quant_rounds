"""Round10 stage 编排。parity 已通过则不再当 blocker 重跑。禁止 --stage all。"""

from __future__ import annotations

from typing import Any

from .parity import load_parity_status, run_parity


def require_parity() -> dict[str, Any]:
    parity = load_parity_status()
    if not parity.get("passed"):
        raise RuntimeError("PARITY_FAIL_OR_MISSING STOP")
    return parity


def run_stage(stage: str, *, resume: bool = True, jobs: int = 1, experiment: str = "") -> dict[str, Any]:
    if stage == "parity":
        return run_parity(resume=resume)
    require_parity()
    if stage == "attribution":
        from .attribution import run_attribution
        return run_attribution(resume=resume, jobs=jobs, experiment=experiment)
    if stage == "pair":
        from .pair import run_pair
        return run_pair(resume=resume, jobs=jobs, experiment=experiment)
    if stage == "shrinkage":
        from .shrinkage import run_shrinkage
        return run_shrinkage(resume=resume, jobs=jobs, experiment=experiment)
    if stage in {"execution", "perturbation", "neighborhood", "capacity", "fault", "live"}:
        from .live import run_live
        return run_live(stage=stage, resume=resume, jobs=jobs, experiment=experiment)
    if stage in {"shadow", "drawdown", "tail", "combine"}:
        from .later import run_later
        return run_later(stage=stage, resume=resume, jobs=jobs, experiment=experiment)
    if stage == "robustness":
        from .robustness import run_robustness_stage
        return run_robustness_stage(resume=resume, jobs=jobs)
    if stage == "conclusion":
        from .docs import evaluate_promotion, patch_tracker, print_complete, write_checklist, write_conclusion

        try:
            from .plots import write_figures
            write_figures()
        except Exception:
            pass
        decision = evaluate_promotion()
        write_conclusion(decision)
        write_checklist(decision)
        patch_tracker(decision)
        print_complete(decision)
        return decision
    raise RuntimeError(f"unknown stage: {stage}")


__all__ = ["require_parity", "run_stage"]
