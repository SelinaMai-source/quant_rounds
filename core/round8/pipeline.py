"""Round8 stage 编排。parity 已通过则不再当 blocker 重跑。"""

from __future__ import annotations

import json
from typing import Any

from ..paths import ROUND8_ROOT, assert_project_result_path
from .contract import PARENT_CODE
from .diagnostics import run_disagreement_diag, run_geometry, run_left_tail, run_top2_marginal
from .docs import (
    build_answers,
    evaluate_promotion,
    materialize_combines,
    patch_tracker,
    pick_winners,
    write_conclusion,
    write_geometry_md,
    write_stage_md,
)
from .execution import train_d_close
from .parity import load_parity_status
from .registry import SPECS, register_winner_stress, specs_for_stage
from .robustness import run_robustness, v2_anchor_metrics
from .runner import run_codes


def require_parity() -> dict[str, Any]:
    parity = load_parity_status()
    if not parity.get("passed"):
        raise RuntimeError("PARITY_FAIL_OR_MISSING STOP")
    return parity


def run_geometry_stage(*, resume: bool = True) -> dict[str, Any]:
    payload = run_geometry(resume=resume)
    write_geometry_md(payload)
    return payload


def _write_stage_with_extra(stage: str) -> None:
    extra = ""
    if stage == "top2":
        mu = run_top2_marginal()
        extra = (
            "## Top2 边际效用（组合增量，不是单票均值）\n\n"
            f"```json\n{json.dumps(mu, ensure_ascii=False, indent=2)}\n```\n"
        )
    elif stage == "disagreement":
        diag = run_disagreement_diag()
        extra = "## 双头状态诊断\n\n融合权重未改（仍 90/10）。\n\n```json\n" + json.dumps(diag, ensure_ascii=False, indent=2) + "\n```\n"
    elif stage == "ecology":
        g_codes = ["R8_G01", "R8_G02", "R8_G03", "R8_G04"]
        v2 = v2_anchor_metrics()
        any_inc = False
        from .runner import load_summary

        for code in g_codes[1:]:
            s = load_summary(code) or {}
            if s.get("sharpe") is not None and s.get("cagr") is not None:
                if float(s["cagr"]) >= v2["cagr"] and float(s["sharpe"]) >= v2["sharpe"] + 0.01:
                    if float(s["max_drawdown"]) >= float(v2["max_drawdown"]) - 0.01:
                        any_inc = True
        extra = (
            "G01 = exact P08 control。\n\n"
            + ("存在稳定增量。" if any_inc else "**ECOLOGY_REDUNDANT_GIVEN_BASELINE_V2**")
        )
    elif stage == "left-tail":
        left = run_left_tail()
        extra = (
            f"H03 审计结论：`{(left.get('h03') or {}).get('decision', 'SKIP_H03')}`。"
            "禁止为 2022 临时造规则。\n\n"
            f"```json\n{json.dumps(left, ensure_ascii=False, indent=2, default=str)[:12000]}\n```\n"
        )
    elif stage == "shape":
        extra = "F 组把 hard half-rule 换成分位形状（IQR / MAD / 按候选数分组 PIT）。禁止再搜 count bin。"
    elif stage == "confidence":
        extra = "B04 = baseline_v2 硬阈值对照。禁止搜 0.52/0.57…。"
    elif stage == "rank-specific":
        extra = "C 组按 `selection_rank` 写不同 `allocation_multiplier`，**不改 score**。关掉 Top2 时保留原 Top2 行（乘子 MIN_KEEP），避免 Top3 补位。"
    write_stage_md(stage, extra=extra)


def run_backtest_stage(stage: str, *, resume: bool = True, jobs: int = 1, codes: list[str] | None = None) -> dict[str, Any]:
    require_parity()
    if codes is None:
        codes = [s.code for s in specs_for_stage(stage)]
    if stage == "execution":
        train_d_close(resume=True)
    if not codes:
        _write_stage_with_extra(stage)
        return {"n": 0, "jobs": {}, "failed": []}
    out = run_codes(codes, resume=resume, jobs=jobs)
    _write_stage_with_extra(stage)
    return out


def run_combine_stage(*, resume: bool = True, jobs: int = 1) -> dict[str, Any]:
    require_parity()
    runnable = materialize_combines()
    if runnable:
        out = run_codes(runnable, resume=resume, jobs=jobs)
    else:
        out = {"n": 0, "jobs": {}, "failed": [], "note": "no_stable_two_module_combo"}
    write_stage_md("combine", extra="最多 6 个组合。单模块槽位若等于该模块本身则 skip。三模块叠加未启用。")
    return out


def _finalists() -> list[str]:
    v2 = v2_anchor_metrics()
    from .runner import load_summary

    out = []
    for code, spec in SPECS.items():
        if spec.diagnostic or spec.post_hoc or spec.skip:
            continue
        if spec.stage in {"robustness", "parity"}:
            continue
        s = load_summary(code)
        if not s or s.get("sharpe") is None:
            continue
        if float(s["sharpe"]) >= v2["sharpe"] - 1e-12 and float(s["cagr"]) >= v2["cagr"] - 1e-12:
            out.append(code)
    return out


def run_robustness_stage(*, resume: bool = True, jobs: int = 1) -> dict[str, Any]:
    require_parity()
    codes = [s.code for s in specs_for_stage("robustness") if not s.post_hoc]
    winners = pick_winners()
    best = None
    best_sh = -9.0
    for key in ("top2", "smooth", "disagree", "rank", "shape", "left", "ecology"):
        item = winners.get(key)
        if not item or not item.get("stable"):
            continue
        sh = float((item.get("summary") or {}).get("sharpe") or -9)
        if sh > best_sh:
            best_sh = sh
            best = item["code"]
    extra_codes = []
    if best:
        extra_codes = register_winner_stress(best)
    run_list = codes + extra_codes
    out = run_codes(run_list, resume=resume, jobs=jobs) if run_list else {"n": 0, "jobs": {}, "failed": []}
    finals = ["R8_00_baseline_v2", *dict.fromkeys(_finalists())]
    boot = run_robustness(finals, resume=resume)
    extra = (
        "邻域只允许 15/20/25；**20% = baseline_v2 本身**（R8_00 / R8_B04），不另钓鱼。\n\n"
        "winner dependence = **DIAGNOSTIC / NOT REALISTIC**。\n\n"
        f"best R8 for POST_HOC stress = `{best}`。\n\n"
        f"```json\n{json.dumps(boot, ensure_ascii=False, indent=2, default=str)[:12000]}\n```\n"
    )
    write_stage_md("robustness", extra=extra)
    path = ROUND8_ROOT / "metrics" / "robustness_meta.json"
    assert_project_result_path(path)
    path.write_text(json.dumps({"best": best, "finalists": finals, "bootstrap": boot}, ensure_ascii=False, indent=2, default=str) + "\n")
    return out


def run_conclusion_stage() -> dict[str, Any]:
    require_parity()
    from .parity import load_parity_status
    from pathlib import Path

    for st in (
        "confidence", "rank-specific", "top2", "disagreement", "shape",
        "ecology", "left-tail", "execution", "combine", "robustness",
    ):
        md = ROUND8_ROOT / {
            "confidence": "03_smooth_confidence.md",
            "rank-specific": "04_rank_specific_confidence.md",
            "top2": "05_conditional_top2.md",
            "disagreement": "06_dual_head_agreement.md",
            "shape": "07_distribution_shape.md",
            "ecology": "08_ecology_marginal.md",
            "left-tail": "09_left_tail_attribution.md",
            "execution": "10_execution_audit.md",
            "combine": "11_final_combinations.md",
            "robustness": "12_robustness.md",
        }[st]
        if not md.is_file():
            _write_stage_with_extra(st)

    geom_path = ROUND8_ROOT / "metrics" / "geometry_diagnostics.json"
    left_path = ROUND8_ROOT / "metrics" / "left_tail_audit.json"
    dis_path = ROUND8_ROOT / "metrics" / "disagreement_diagnostics.json"
    extras = {
        "parity": load_parity_status(),
        "geometry": json.loads(geom_path.read_text()) if geom_path.is_file() else {},
        "left_tail": json.loads(left_path.read_text()) if left_path.is_file() else {},
        "disagreement": json.loads(dis_path.read_text()) if dis_path.is_file() else {},
        "winners": pick_winners(),
    }
    g_path = ROUND8_ROOT / "08_ecology_marginal.md"
    extras["ecology_redundant"] = (
        "ECOLOGY_REDUNDANT_GIVEN_BASELINE_V2"
        if g_path.is_file() and "ECOLOGY_REDUNDANT_GIVEN_BASELINE_V2" in g_path.read_text(encoding="utf-8")
        else ""
    )
    codes = [
        c for c, s in SPECS.items()
        if s.stage not in {"robustness"} and not s.diagnostic
    ]
    decision = evaluate_promotion(codes)
    if not (decision.get("candidates") or []):
        decision["decision"] = "KEEP_BASELINE_V2"
    extras["answers"] = build_answers(decision, extras)
    write_conclusion(decision, extras)
    patch_tracker(list(SPECS))
    return decision


def run_stage(stage: str, *, resume: bool = True, jobs: int = 1, experiment: str = "") -> dict[str, Any]:
    codes = [c.strip() for c in experiment.split(",") if c.strip()] if experiment else None
    if stage == "geometry":
        return {"geometry": run_geometry_stage(resume=resume)}
    if stage == "combine":
        return run_combine_stage(resume=resume, jobs=jobs)
    if stage == "robustness":
        return run_robustness_stage(resume=resume, jobs=jobs)
    if stage == "conclusion":
        return run_conclusion_stage()
    if stage == "parity":
        raise RuntimeError("parity 由 scripts 单独处理")
    return run_backtest_stage(stage, resume=resume, jobs=jobs, codes=codes)


__all__ = ["require_parity", "run_stage"]
