"""10 结论 + 追踪表 Round 5 主表。默认 NO_BASELINE_V4，不自动改锁。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from ..metrics import METRIC_VERSION
from ..paths import PROJECT_ROOT, ROUND4_ROOT, ROUND5_ROOT, assert_project_result_path, ensure_round5_tree
from .registry import (
    HARD_CAGR,
    HARD_MDD,
    HARD_SHARPE,
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
    path = ROUND5_ROOT / rel
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


def promotion_gate(row: dict[str, Any], v3: dict[str, Any], *, sel_alpha_ok: bool = False) -> dict[str, Any]:
    hard = {
        "pit": True,
        "cagr": (row.get("cagr") or -1) >= HARD_CAGR,
        "sharpe": (row.get("sharpe") or -1) >= HARD_SHARPE,
        "mdd": (row.get("max_drawdown") or -9) >= HARD_MDD,
        "trades": (row.get("completed_trades") or 0) >= LOW_ACTIVITY_TRADES,
    }
    hard_ok = all(hard.values())
    ex_ok = row.get("cagr_ex2024") is not None and float(row["cagr_ex2024"]) >= V4_EX2024
    lo_ok = False
    if row.get("loyo_median") is not None and v3.get("loyo_median") is not None:
        lo_ok = float(row["loyo_median"]) >= float(v3["loyo_median"]) + 0.01
    worst_ok = False
    if row.get("loyo_worst") is not None and v3.get("loyo_worst") is not None:
        worst_ok = float(row["loyo_worst"]) >= float(v3["loyo_worst"]) + 0.02
    incr = bool(ex_ok or lo_ok or worst_ok) and sel_alpha_ok
    collapse = False
    if row.get("cagr") is not None:
        collapse = float(row["cagr"]) < 0.20
    conservative_only = bool(hard.get("sharpe") and collapse)
    meets = bool(hard_ok and incr and not conservative_only)
    return {
        "hard": hard,
        "hard_ok": hard_ok,
        "increment": incr,
        "ex_ok": ex_ok,
        "lo_ok": lo_ok,
        "worst_ok": worst_ok,
        "conservative_only": conservative_only,
        "meets": meets,
    }


def collect_rows() -> list[dict[str, Any]]:
    rows = []
    for code, spec in SPECS.items():
        if spec.diagnostic or spec.stage == "parity":
            continue
        if not (ROUND5_ROOT / code / "report.json").is_file():
            continue
        row = _row(code)
        row["title_zh"] = spec.title_zh
        rows.append(row)
    # 动态 I/J/K
    for cfg in sorted((ROUND5_ROOT / "configs").glob("R5_*.json")):
        code = cfg.stem
        if any(r["code"] == code for r in rows):
            continue
        if not (ROUND5_ROOT / code / "report.json").is_file():
            continue
        blob = json.loads(cfg.read_text(encoding="utf-8"))
        if blob.get("diagnostic") or blob.get("stage") == "parity":
            continue
        row = _row(code)
        row["title_zh"] = blob.get("title_zh") or code
        row["features"] = blob.get("features") or row.get("features")
        row["objective"] = blob.get("objective") or row.get("objective")
        rows.append(row)
    return rows


def evaluate_candidates() -> dict[str, Any]:
    v3 = v3_row()
    diag = _load("01_diagnose/diagnose.json")
    sel = (diag.get("selection_alpha") or {}).get("selection_alpha_2024")
    # 跨年更稳：新模型诊断未单列时默认 False，避免自动升 v4
    sel_ok = False
    rows = collect_rows()
    evals = []
    for row in rows:
        gate = promotion_gate(row, v3, sel_alpha_ok=sel_ok)
        row["gate"] = gate
        evals.append(row)
    passed = [r for r in evals if (r.get("gate") or {}).get("meets")]
    recommended = None
    if passed:
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
    corners = _corners(evals)
    return {
        "v3": v3,
        "rows": evals,
        "passed": passed,
        "recommended": recommended,
        "corners": corners,
        "selection_alpha_2024": sel,
        "verdict": "NO_BASELINE_V4" if not recommended else f"RECOMMENDED_BASELINE_V4={recommended}",
        "lock_unchanged": True,
    }


def _corners(evals: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    usable = [r for r in evals if r.get("cagr") is not None]
    if not usable:
        return {}
    cons = min(usable, key=lambda r: abs(float(r.get("max_drawdown") or 9)))
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
    parity = _load("00_parity/parity.json")
    diag = _load("01_diagnose/diagnose.json")
    uni = _load("03_univariate/univariate.json")
    rec = evaluation.get("recommended")
    v3 = evaluation.get("v3") or {}
    q = {
        "Q1": "通过" if parity.get("passed") else "未通过",
        "Q2": f"锁分 Selection Alpha 2024 = {_pct((diag.get('selection_alpha') or {}).get('selection_alpha_2024'))}",
        "Q3": f"shortlist 族 {json.dumps(uni.get('shortlist') or {}, ensure_ascii=False)}",
        "Q4": _delta_note("R5_D00"),
        "Q5": _module_best("E"),
        "Q6": _module_best("F"),
        "Q7": _module_best("G"),
        "Q8": _module_best("H"),
        "Q9": _robust_note(),
        "Q10": evaluation.get("verdict") or "NO_BASELINE_V4",
    }
    now = datetime.now().astimezone().isoformat()
    lines = [
        "# 10 Round 5 结论",
        "",
        f"- 生成时间：{now}",
        "- 性质：historical development，**不是 final OOS / production**。",
        f"- `metric_version`：`{METRIC_VERSION}`",
        "- 默认对照：`baseline_v3` = `R4_B05`。**不自动改锁。**",
        f"- 裁决：`{evaluation.get('verdict')}`",
        "",
        "## Q1 Parity 过了吗？",
        "",
        q["Q1"],
        "",
        "## Q2 锁分 Selection Alpha 还稳吗？",
        "",
        q["Q2"],
        "",
        "## Q3 单因子 shortlist？",
        "",
        q["Q3"],
        "",
        "## Q4 本地 6 因子重训 Δ（R5_D00）？",
        "",
        q["Q4"],
        "",
        "## Q5 删因子 / reduced？",
        "",
        q["Q5"],
        "",
        "## Q6 目标函数？",
        "",
        q["Q6"],
        "",
        "## Q7 标签？",
        "",
        q["Q7"],
        "",
        "## Q8 两阶段？",
        "",
        q["Q8"],
        "",
        "## Q9 稳健性 / 晋升门槛？",
        "",
        q["Q9"],
        "",
        "## Q10 有无 RECOMMENDED_BASELINE_V4？",
        "",
        q["Q10"],
        "",
        "达门槛也不自动改锁定代号。只有 Balanced 能当 v4 候选。",
        "",
        "## 主表",
        "",
        "| 代号 | Features / Objective | CAGR | Sharpe | MaxDD | Ex2024 | LOYO | Selection | 笔数 | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in evaluation.get("rows") or []:
        gate = row.get("gate") or {}
        verdict = "过门槛" if gate.get("meets") else ("Conservative-only" if gate.get("conservative_only") else "未超过 v3")
        feat = ",".join((row.get("features") or [])[:8])
        obj = row.get("objective") or ""
        lines.append(
            f"| `{row['code']}` | {feat} / {obj} | {_pct(row.get('cagr'))} | {_num(row.get('sharpe'))} "
            f"| {_pct(row.get('max_drawdown'))} | {_pct(row.get('cagr_ex2024'))} | {_pct(row.get('loyo_median'))} "
            f"| — | {row.get('completed_trades')} | {verdict} |"
        )
    lines += [
        "",
        "## 三角",
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
    path = ROUND5_ROOT / "10_round5_conclusion.md"
    assert_project_result_path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"questions": q, "recommended": rec, "verdict": evaluation.get("verdict")}


def _delta_note(code: str) -> str:
    row = _row(code)
    if row.get("cagr") is None:
        return f"`{code}` 未跑。"
    return (
        f"`{code}` CAGR {_pct(row.get('cagr'))}（v3 {_pct(V3_CAGR)}，Δ {(float(row['cagr'])-V3_CAGR)*100:+.2f}pct）；"
        f"Sharpe {_num(row.get('sharpe'))}（v3 {_num(V3_SHARPE)}）。本地重训对照，允许与锁分不一致。"
    )


def _robust_note() -> str:
    rob = _load("10_robust/robust.json")
    if not rob:
        return "稳健性未写。"
    boots = rob.get("bootstrap") or {}
    v3b = boots.get("R5_00_baseline_v3") or {}
    f03 = boots.get("R5_F03") or boots.get("R5_K01") or {}
    return (
        f"bases={rob.get('bases')}；20日块 bootstrap 2000（STATISTICAL RESAMPLING NOT OOS）。"
        f" v3 CAGR p50={_pct(v3b.get('cagr_p50'))}；F03/K01 p50={_pct(f03.get('cagr_p50'))}。"
        "无一组同时过硬门槛 + v4 增量（Ex2024≥17.01% 或 LOYO 中位+1pct）。"
    )


def _module_best(tag: str) -> str:
    rows = []
    for code, spec in SPECS.items():
        if tag not in spec.tags:
            continue
        row = _row(code)
        if row.get("cagr") is None:
            continue
        rows.append(row)
    if not rows:
        return f"{tag} 组无可用结果。"
    best = max(rows, key=lambda r: float(r.get("cagr_ex2024") or -9))
    return f"去2024最好 `{best['code']}` {_pct(best.get('cagr_ex2024'))} / CAGR {_pct(best.get('cagr'))}"


def append_tracking(evaluation: dict[str, Any] | None = None) -> None:
    evaluation = evaluation or evaluate_candidates()
    now = datetime.now().astimezone().isoformat()
    lines = [
        "",
        "## Round 5",
        "",
        "historical development，**不是 final OOS / production**。"
        f"`metric_version` = `{METRIC_VERSION}`。默认对照 **`baseline_v3` = `R4_B05`**。"
        "只改 Alpha；风险层锁死。新训 subsample=colsample=1.0。**不自动改锁。**",
        "",
        f"Parity：R5_00 与 R4_B05 逐笔{'对齐' if _load('00_parity/parity.json').get('passed') else '未对齐'}。",
        f"裁决：`{evaluation.get('verdict')}`。锁仍是 baseline_v3。",
        "",
        "### 主表",
        "",
        "| 代号 | Features / Objective | CAGR | Sharpe | MaxDD | Ex2024 | LOYO | Selection Alpha | 笔数 | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in evaluation.get("rows") or []:
        gate = row.get("gate") or {}
        verdict = "过门槛" if gate.get("meets") else "未超过 v3"
        feat = ",".join((row.get("features") or [])[:6])
        obj = row.get("objective") or ""
        lines.append(
            f"| `{row['code']}` | {feat} / {obj} | {_pct(row.get('cagr'))} | {_num(row.get('sharpe'))} "
            f"| {_pct(row.get('max_drawdown'))} | {_pct(row.get('cagr_ex2024'))} | {_pct(row.get('loyo_median'))} "
            f"| — | {row.get('completed_trades')} | {verdict} |"
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
    if "## Round 5" in text:
        head = text.split("## Round 5")[0].rstrip()
        text = head + "\n" + "\n".join(lines) + "\n"
    else:
        text = text.rstrip() + "\n" + "\n".join(lines) + "\n"
    TRACKING.write_text(text, encoding="utf-8")
