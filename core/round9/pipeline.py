"""Round9 stage 编排。parity 已通过则不再当 blocker 重跑。禁止第一次 --stage all。"""

from __future__ import annotations

import json
from typing import Any

from ..paths import ROUND9_ROOT, assert_project_result_path
from .docs import evaluate_promotion, patch_tracker, print_complete, write_conclusion, write_named_md
from .forensics import run_forensics
from .later import run_combine, run_direction, run_drawdown, run_execution, run_residual
from .meta import run_meta
from .parity import load_parity_status
from .registry import specs_for_stage
from .robustness import run_robustness_stage
from .runner import run_codes


def require_parity() -> dict[str, Any]:
    parity = load_parity_status()
    if not parity.get("passed"):
        raise RuntimeError("PARITY_FAIL_OR_MISSING STOP")
    return parity


def run_unified(*, resume: bool = True, jobs: int = 1) -> dict[str, Any]:
    require_parity()
    codes = [s.code for s in specs_for_stage("unified")]
    out = run_codes(codes, resume=resume, jobs=jobs)
    write_named_md(
        "02_unified_allocator.md",
        "unified",
        codes,
        "B01 预注册 9 格、B02 `C2=0.5A+0.5S`、B03 conservative、B04=E02 control。\n"
        "B01–B03 是 replacement 0/5/10，**不再乘** E02 1.00/0.75/0.50。看能否接近 J06（53.52/1.818）且更简单。\n",
    )
    return out


def run_stage(stage: str, *, resume: bool = True, jobs: int = 1, experiment: str = "") -> dict[str, Any]:
    require_parity()
    codes = [c.strip() for c in experiment.split(",") if c.strip()] if experiment else None
    if stage == "parity":
        from .parity import run_parity

        return run_parity(resume=resume)
    if stage == "forensics":
        return run_forensics(resume=resume, jobs=jobs)
    if stage == "unified":
        if codes:
            return run_codes(codes, resume=resume, jobs=jobs)
        return run_unified(resume=resume, jobs=jobs)
    if stage == "meta":
        return run_meta(resume=resume, jobs=jobs)
    if stage == "direction":
        return run_direction(resume=resume, jobs=jobs)
    if stage == "execution":
        return run_execution(resume=resume, jobs=jobs)
    if stage == "residual":
        return run_residual(resume=resume, jobs=jobs)
    if stage == "drawdown":
        return run_drawdown(resume=resume, jobs=jobs)
    if stage == "combine":
        return run_combine(resume=resume, jobs=jobs)
    if stage == "robustness":
        return run_robustness_stage(resume=resume, jobs=jobs)
    if stage == "conclusion":
        decision = evaluate_promotion()
        write_conclusion(decision)
        patch_tracker()
        print_complete(decision)
        path = ROUND9_ROOT / "metrics" / "conclusion.json"
        assert_project_result_path(path)
        path.write_text(json.dumps(decision, ensure_ascii=False, indent=2, default=str) + "\n")
        return decision
    if codes is None:
        codes = [s.code for s in specs_for_stage(stage)]
    if not codes:
        return {"n": 0, "jobs": {}, "failed": []}
    return run_codes(codes, resume=resume, jobs=jobs)


__all__ = ["require_parity", "run_stage"]
