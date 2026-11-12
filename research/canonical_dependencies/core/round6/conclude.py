"""11 结论 + 追踪表 Round 6 主表。默认 NO_BASELINE_V4，不自动改锁。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from ..metrics import METRIC_VERSION
from ..paths import PROJECT_ROOT, ROUND4_ROOT, ROUND6_ROOT, assert_project_result_path
from .forensics import load_forensics_status
from .parity import load_parity_status
from .registry import (
    CHALLENGER_CAGR_SLACK,
    CHALLENGER_EX_SLACK,
    CHALLENGER_MDD_IMPROVE,
    CHALLENGER_SHARPE_ADD,
    HARD_CAGR,
    HARD_MDD,
    HARD_SHARPE,
    LOYO_MEDIAN_ADD,
    LOW_ACTIVITY_TRADES,
    SPECS,
    V3_CAGR,
    V3_EX2024,
    V3_MDD,
    V3_SHARPE,
    V3_TRADES,
    V4_EX2024,
)
from .runner import load_report, tracking_row

TRACKING = PROJECT_ROOT / "result" / "一进二_最优实验组追踪.md"


def _pct(v: Any, d: int = 2) -> str:
    if v is None:
        return "—"
    return f"{float(v) * 100:.{d}f}%"


def _num(v: Any, d: int = 3) -> str:
    if v is None:
        return "—"
    return f"{float(v):.{d}f}"


def _load(rel: str) -> dict[str, Any]:
    path = ROUND6_ROOT / rel
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _row(code: str) -> dict[str, Any]:
    report = load_report(code) or {}
    spec = SPECS.get(code)
    if spec:
        return tracking_row(spec, report)
    m = report.get("metrics") or {}
    return {
        "code": code,
        "cagr": m.get("annualized_return"),
        "sharpe": m.get("sharpe_zero_rate"),
        "max_drawdown": m.get("max_drawdown"),
        "cagr_ex2024": m.get("cagr_ex2024"),
        "loyo_median": m.get("loyo_median"),
        "loyo_worst": m.get("loyo_worst"),
        "completed_trades": m.get("completed_trades"),
        "title_zh": report.get("experiment_code") or code,
        "features": report.get("features") or [],
        "objective": report.get("objective"),
        "target": report.get("target"),
        "forensic_only": report.get("forensic_only"),
        "diagnostic": False,
    }


def v3_row() -> dict[str, Any]:
    path = ROUND4_ROOT / "R4_B05" / "report.json"
    m = json.loads(path.read_text(encoding="utf-8")).get("metrics") or {}
    return {
        "code": "R4_B05",
        "cagr": m.get("annualized_return") or V3_CAGR,
        "sharpe": m.get("sharpe_zero_rate") or V3_SHARPE,
        "max_drawdown": m.get("max_drawdown") or V3_MDD,
        "cagr_ex2024": m.get("cagr_ex2024") or V3_EX2024,
        "loyo_median": m.get("loyo_median"),
        "loyo_worst": m.get("loyo_worst"),
        "completed_trades": m.get("completed_trades") or V3_TRADES,
    }


def promotion_gate(
    row: dict[str, Any],
    v3: dict[str, Any],
    *,
    sel_alpha_ok: bool = False,
    repro_ok: bool = False,
    oof_quality_ok: bool = False,
) -> dict[str, Any]:
    hard = {
        "pit": True,
        "repro": repro_ok,
        "cagr": (row.get("cagr") or -1) >= HARD_CAGR,
        "sharpe": (row.get("sharpe") or -1) >= HARD_SHARPE,
        "mdd": (row.get("max_drawdown") or -9) >= HARD_MDD,
        "trades": (row.get("completed_trades") or 0) >= LOW_ACTIVITY_TRADES,
    }
    hard_ok = all(hard.values())
    ex_ok = row.get("cagr_ex2024") is not None and float(row["cagr_ex2024"]) >= V4_EX2024
    lo_ok = False
    if row.get("loyo_median") is not None and v3.get("loyo_median") is not None:
        lo_ok = float(row["loyo_median"]) >= float(v3["loyo_median"]) + LOYO_MEDIAN_ADD
    worst_ok = False
    if row.get("loyo_worst") is not None and v3.get("loyo_worst") is not None:
        worst_ok = float(row["loyo_worst"]) >= float(v3["loyo_worst"]) + 0.02
    incr = bool(ex_ok or lo_ok or worst_ok)
    challenger = False
    if row.get("cagr") is not None and row.get("sharpe") is not None and row.get("max_drawdown") is not None:
        challenger = (
            float(row["cagr"]) >= float(v3.get("cagr") or V3_CAGR) - CHALLENGER_CAGR_SLACK
            and float(row["sharpe"]) >= float(v3.get("sharpe") or V3_SHARPE) + CHALLENGER_SHARPE_ADD
            and float(row["max_drawdown"]) >= float(v3.get("max_drawdown") or V3_MDD) + CHALLENGER_MDD_IMPROVE
            and (row.get("cagr_ex2024") is not None and float(row["cagr_ex2024"]) >= float(v3.get("cagr_ex2024") or V3_EX2024) - CHALLENGER_EX_SLACK)
        )
    quality = bool(sel_alpha_ok or oof_quality_ok)
    low_act = bool(row.get("low_activity")) or (
        row.get("completed_trades") is not None and int(row["completed_trades"]) < LOW_ACTIVITY_TRADES
    )
    tag = "未超过 v3"
    if low_act and (row.get("sharpe") or 0) >= HARD_SHARPE:
        tag = "LOW_ACTIVITY_RISK_CONTROL"
    if hard_ok and incr and quality:
        tag = "过门槛"
    elif hard_ok and challenger and quality:
        tag = "BASELINE_V4_CHALLENGER"
    elif hard_ok and (incr or challenger) and not quality:
        tag = "PORTFOLIO_OVERLAY"
    return {
        "hard": hard,
        "hard_ok": hard_ok,
        "increment": incr,
        "challenger": challenger,
        "ex_ok": ex_ok,
        "lo_ok": lo_ok,
        "worst_ok": worst_ok,
        "quality": quality,
        "tag": tag,
        "meets": tag == "过门槛",
    }


def collect_rows() -> list[dict[str, Any]]:
    rows = []
    for code, spec in SPECS.items():
        if spec.diagnostic or spec.forensic_only or spec.stage in {"parity", "forensics"}:
            continue
        if not (ROUND6_ROOT / code / "report.json").is_file():
            continue
        row = _row(code)
        row["title_zh"] = spec.title_zh
        rows.append(row)
    for cfg in sorted((ROUND6_ROOT / "configs").glob("R6_*.json")):
        code = cfg.stem
        if any(r["code"] == code for r in rows):
            continue
        if not (ROUND6_ROOT / code / "report.json").is_file():
            continue
        blob = json.loads(cfg.read_text(encoding="utf-8"))
        if blob.get("diagnostic") or blob.get("forensic_only") or blob.get("stage") in {"parity", "forensics", "robust"}:
            continue
        row = _row(code)
        row["title_zh"] = blob.get("title_zh") or code
        row["features"] = blob.get("features") or row.get("features")
        row["objective"] = blob.get("objective") or row.get("objective")
        row["window_kind"] = blob.get("window_kind") or row.get("window_kind")
        row["representation"] = blob.get("representation") or row.get("representation")
        row["target"] = blob.get("target") or row.get("target")
        rows.append(row)
    return rows


def evaluate_candidates() -> dict[str, Any]:
    v3 = v3_row()
    forensic = load_forensics_status()
    repro_ok = forensic.get("status") in {"REPRODUCED", "ROOT_CAUSE_IDENTIFIED"}
    if forensic.get("status") == "REPRODUCIBILITY_UNRESOLVED":
        repro_ok = False
    sel_ok = False
    oof_ok = False
    struct = _load("06_meta/structure.json")
    if (struct.get("oof_trade_quality") or {}).get("improved"):
        oof_ok = True
    rows = collect_rows()
    evals = []
    for row in rows:
        gate = promotion_gate(row, v3, sel_alpha_ok=sel_ok, repro_ok=repro_ok, oof_quality_ok=oof_ok)
        row["gate"] = gate
        evals.append(row)
    passed = [r for r in evals if (r.get("gate") or {}).get("meets")]
    recommended = None
    if passed and repro_ok:
        passed_sorted = sorted(
            passed,
            key=lambda r: (
                float(r.get("cagr_ex2024") or -9),
                float(r.get("loyo_median") or -9),
                float(r.get("sharpe") or -9),
            ),
            reverse=True,
        )
        recommended = passed_sorted[0]["code"]
    challengers = [
        r for r in evals
        if (r.get("gate") or {}).get("tag") == "BASELINE_V4_CHALLENGER" and repro_ok
    ]
    corners = _corners(evals)
    verdict = "NO_BASELINE_V4"
    if recommended:
        verdict = f"RECOMMENDED_BASELINE_V4={recommended}"
    elif challengers:
        # 只有 Balanced 能当 challenger 写入
        bal = (corners.get("Balanced") or {}).get("code")
        hit = next((c for c in challengers if c["code"] == bal), None)
        if hit:
            verdict = f"BASELINE_V4_CHALLENGER={hit['code']}"
    if forensic.get("status") == "REPRODUCIBILITY_UNRESOLVED":
        verdict = "NO_BASELINE_V4"
        recommended = None
    return {
        "v3": v3,
        "rows": evals,
        "passed": passed,
        "recommended": recommended,
        "corners": corners,
        "forensics_status": forensic.get("status"),
        "verdict": verdict,
        "lock_unchanged": True,
    }


def _corners(evals: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    usable = [r for r in evals if r.get("cagr") is not None]
    if not usable:
        return {}
    cons = max(usable, key=lambda r: float(r.get("max_drawdown") or -9))
    agg = max(usable, key=lambda r: float(r.get("cagr") or -9))
    bal_pool = [r for r in usable if r.get("code") not in {cons.get("code"), agg.get("code")}] or usable
    bal = max(
        bal_pool,
        key=lambda r: (
            float(r.get("cagr_ex2024") or -9),
            float(r.get("sharpe") or -9),
            float(r.get("max_drawdown") or -9),
        ),
    )
    return {"Conservative": cons, "Balanced": bal, "Aggressive": agg}


def write_conclusion(evaluation: dict[str, Any] | None = None) -> dict[str, Any]:
    evaluation = evaluation or evaluate_candidates()
    parity = load_parity_status()
    forensic = load_forensics_status()
    struct = _load("06_meta/structure.json")
    rob = _load("10_robust/robust.json")
    rec = evaluation.get("recommended")
    v3 = evaluation.get("v3") or {}
    q = {
        "Q1": "通过" if parity.get("passed") else "未通过",
        "Q2": forensic.get("status") or "未写",
        "Q3": _module_best("B"),
        "Q4": _module_best("C"),
        "Q5": _module_best("D"),
        "Q6": _module_best("E"),
        "Q7": (struct.get("F") or {}).get("note") or _module_best("F"),
        "Q8": (struct.get("G") or {}).get("note") or "G 未写",
        "Q9": (struct.get("H") or {}).get("note") or _module_best("H"),
        "Q10": _robust_note(rob),
        "Q11": evaluation.get("verdict") or "NO_BASELINE_V4",
        "Q12": "锁仍是 baseline_v3 = R4_B05。不自动改锁。只有 Balanced 能当 challenger。",
    }
    now = datetime.now().astimezone().isoformat()
    lines = [
        "# 11 Round 6 结论",
        "",
        f"- 生成时间：{now}",
        "- 性质：historical development，**不是 final OOS / production**。",
        f"- `metric_version`：`{METRIC_VERSION}`",
        "- 默认对照：`baseline_v3` = `R4_B05`。**不自动改锁。**",
        f"- 裁决：`{evaluation.get('verdict')}`",
        f"- Forensics：`{forensic.get('status')}`",
        "",
        "## Q1 Parity 过了吗？", "", q["Q1"], "",
        "## Q2 Forensics 状态？", "", str(q["Q2"]), "",
        "## Q3 窗口 / horizon Top2？", "", q["Q3"], "",
        "## Q4 横截面表示 Top2？", "", q["Q4"], "",
        "## Q5 pairwise / D04？", "", q["Q5"], "",
        "## Q6 标签？", "", q["Q6"], "",
        "## Q7 OOF meta？", "", str(q["Q7"]), "",
        "## Q8 Ensemble G？", "", str(q["Q8"]), "",
        "## Q9 Context H？", "", str(q["Q9"]), "",
        "## Q10 稳健性 / bootstrap / winner 依赖？", "", q["Q10"], "",
        "## Q11 有无 RECOMMENDED_BASELINE_V4 / CHALLENGER？", "", q["Q11"], "",
        "## Q12 锁怎么处理？", "", q["Q12"], "",
        "达门槛也不自动改锁定代号。三角里只有 Balanced 能当 challenger。默认 **NO_BASELINE_V4**。",
        "",
        "## 主表",
        "",
        "| 代号 | Window/Repr/Obj | CAGR | Sharpe | MaxDD | Ex2024 | LOYO | 笔数 | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in evaluation.get("rows") or []:
        gate = row.get("gate") or {}
        verdict = gate.get("tag") or "未超过 v3"
        wr = row.get("window_kind") or ""
        rp = row.get("representation") or ""
        obj = row.get("objective") or ""
        lines.append(
            f"| `{row['code']}` | {wr}/{rp}/{obj} | {_pct(row.get('cagr'))} | {_num(row.get('sharpe'))} "
            f"| {_pct(row.get('max_drawdown'))} | {_pct(row.get('cagr_ex2024'))} | {_pct(row.get('loyo_median'))} "
            f"| {row.get('completed_trades')} | {verdict} |"
        )
    lines += [
        "",
        "## 三角（只有 Balanced 能当 v4 候选）",
        "",
        "| 角 | 操作 | 代号 | CAGR | Sharpe | MaxDD | 去2024 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for name, row in (evaluation.get("corners") or {}).items():
        lines.append(
            f"| {name} | {row.get('title_zh') or ''} | `{row.get('code')}` | {_pct(row.get('cagr'))} "
            f"| {_num(row.get('sharpe'))} | {_pct(row.get('max_drawdown'))} | {_pct(row.get('cagr_ex2024'))} |"
        )
    lines += [
        "",
        f"v3 锚：CAGR {_pct(v3.get('cagr'))} / Sharpe {_num(v3.get('sharpe'))} / MaxDD {_pct(v3.get('max_drawdown'))} "
        f"/ Ex2024 {_pct(v3.get('cagr_ex2024'))} / {v3.get('completed_trades')} 笔。",
        "",
    ]
    path = ROUND6_ROOT / "11_round6_conclusion.md"
    assert_project_result_path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"questions": q, "recommended": rec, "verdict": evaluation.get("verdict")}


def _robust_note(rob: dict[str, Any]) -> str:
    if not rob:
        return "稳健性未写。"
    boots = rob.get("bootstrap") or {}
    v3b = boots.get("R6_00_baseline_v3") or boots.get("R4_B05") or {}
    return (
        f"bases={rob.get('bases')}；20日块 bootstrap 2000（STATISTICAL RESAMPLING NOT OOS）。"
        f" v3 CAGR p50={_pct(v3b.get('cagr_p50'))}。"
        "winner 依赖标 DIAGNOSTIC。"
    )


def _module_best(tag: str) -> str:
    rows = []
    for code, spec in SPECS.items():
        if tag not in spec.tags:
            continue
        if spec.diagnostic or spec.forensic_only:
            continue
        row = _row(code)
        if row.get("cagr") is None:
            continue
        rows.append(row)
    if not rows:
        return f"{tag} 组无可用结果。"
    ranked = sorted(rows, key=lambda r: (float(r.get("cagr_ex2024") or -9), float(r.get("loyo_median") or -9)), reverse=True)
    top = ranked[:2]
    parts = [f"`{r['code']}` Ex2024 {_pct(r.get('cagr_ex2024'))} / CAGR {_pct(r.get('cagr'))}" for r in top]
    return "Top2：" + "；".join(parts)


def append_tracking(evaluation: dict[str, Any] | None = None) -> None:
    evaluation = evaluation or evaluate_candidates()
    now = datetime.now().astimezone().isoformat()
    parity = load_parity_status()
    lines = [
        "",
        "## Round 6",
        "",
        "historical development，**不是 final OOS / production**。"
        f"`metric_version` = `{METRIC_VERSION}`。默认对照 **`baseline_v3` = `R4_B05`**。"
        "只改学习问题；风险层锁死 v3。新训 subsample=colsample=1.0。A04 0.8 仅 FORENSIC_ONLY。"
        "**不自动改锁。**",
        "",
        f"Parity：R6_00 与 R4_B05 逐笔{'对齐' if parity.get('passed') else '未对齐'}。",
        f"Forensics：`{evaluation.get('forensics_status')}`。",
        f"裁决：`{evaluation.get('verdict')}`。锁仍是 baseline_v3。",
        "",
        "### 主表",
        "",
        "| 代号 | Window/Repr/Obj | CAGR | Sharpe | MaxDD | Ex2024 | LOYO | 笔数 | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in evaluation.get("rows") or []:
        gate = row.get("gate") or {}
        verdict = gate.get("tag") or "未超过 v3"
        wr = row.get("window_kind") or ""
        rp = row.get("representation") or ""
        obj = row.get("objective") or ""
        lines.append(
            f"| `{row['code']}` | {wr}/{rp}/{obj} | {_pct(row.get('cagr'))} | {_num(row.get('sharpe'))} "
            f"| {_pct(row.get('max_drawdown'))} | {_pct(row.get('cagr_ex2024'))} | {_pct(row.get('loyo_median'))} "
            f"| {row.get('completed_trades')} | {verdict} |"
        )
    lines += [
        "",
        "### 三角（只有 Balanced 能当 v4 候选）",
        "",
        "| 角 | 代号 | CAGR | Sharpe | MaxDD |",
        "|---|---|---:|---:|---:|",
    ]
    for name, row in (evaluation.get("corners") or {}).items():
        lines.append(
            f"| {name} | `{row.get('code')}` | {_pct(row.get('cagr'))} | {_num(row.get('sharpe'))} "
            f"| {_pct(row.get('max_drawdown'))} |"
        )
    lines += ["", f"写入时间：{now}", ""]
    text = TRACKING.read_text(encoding="utf-8") if TRACKING.is_file() else ""
    if "## Round 6" in text:
        head = text.split("## Round 6")[0].rstrip()
        text = head + "\n" + "\n".join(lines) + "\n"
    else:
        text = text.rstrip() + "\n" + "\n".join(lines) + "\n"
    TRACKING.write_text(text, encoding="utf-8")
