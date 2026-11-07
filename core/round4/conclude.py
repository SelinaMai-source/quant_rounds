"""11 结论、追踪表追加。不自动改锁；达标只写 RECOMMENDED_BASELINE_V3。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from ..metrics import METRIC_VERSION
from ..paths import PROJECT_ROOT, ROUND4_ROOT, assert_project_result_path, ensure_round4_tree
from .analyze import _row, _v2, promotion_gate, shortlist_codes
from .registry import SPECS

TRACKING = PROJECT_ROOT / "result" / "一进二_最优实验组追踪.md"


def _load(rel: str) -> dict[str, Any]:
    path = ROUND4_ROOT / rel
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(v: Any, d: int = 2) -> str:
    if v is None:
        return "—"
    return f"{float(v) * 100:.{d}f}%"


def _signed(v: Any, d: int = 2) -> str:
    if v is None:
        return "—"
    return f"{float(v) * 100:+.{d}f}%"


def _num(v: Any, d: int = 3) -> str:
    if v is None:
        return "—"
    return f"{float(v):.{d}f}"


def _op(code: str, row: dict[str, Any] | None = None) -> str:
    if row and row.get("title_zh"):
        return str(row["title_zh"])
    spec = SPECS.get(code)
    if spec:
        return spec.title_zh
    cfg = ROUND4_ROOT / "configs" / f"{code}.json"
    if cfg.is_file():
        return json.loads(cfg.read_text(encoding="utf-8")).get("title_zh") or code
    return code


def evaluate_candidates() -> dict[str, Any]:
    v2 = _v2()
    codes = shortlist_codes()
    evals = []
    for code in codes:
        if code == "R4_00_baseline_v2":
            continue
        row = _row(code)
        if row.get("cagr") is None:
            continue
        gate = promotion_gate(row, v2)
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
    corners = _corners(evals, v2)
    return {
        "v2": v2,
        "evals": evals,
        "passed": passed,
        "recommended": recommended,
        "corners": corners,
        "written_at": datetime.now().astimezone().isoformat(),
    }


def _corners(evals: list[dict[str, Any]], v2: dict[str, Any]) -> dict[str, Any]:
    usable = [r for r in evals if r.get("cagr") is not None and not r.get("low_activity")]
    if not usable:
        return {
            "Conservative": {"code": "R4_00_baseline_v2", **v2},
            "Balanced": {"code": "R4_00_baseline_v2", **v2},
            "Aggressive": {"code": "R4_00_baseline_v2", **v2},
        }
    cons = min(usable, key=lambda r: abs(float(r.get("max_drawdown") or 0)))
    bal = max(
        usable,
        key=lambda r: (float(r.get("sharpe") or 0) + float(r.get("cagr") or 0) - abs(float(r.get("max_drawdown") or 0))),
    )
    agg = max(usable, key=lambda r: float(r.get("cagr") or 0))
    return {"Conservative": cons, "Balanced": bal, "Aggressive": agg}


def write_conclusion(evaluation: dict[str, Any]) -> dict[str, Any]:
    ensure_round4_tree()
    parity = _load("00_parity/parity.json")
    attr = _load("01_attr/attribution.json")
    ev = attr.get("event_beta_vs_alpha") or {}
    fg = attr.get("f_gate") or {}
    single = _load("08_single/single.json")
    pair = _load("09_pair/pair.json")
    triple = _load("10_triple/triple.json")
    robust = _load("11_robust/robust.json")
    rec = evaluation.get("recommended")
    v2 = evaluation.get("v2") or {}

    def _mod_help(mod: str) -> str:
        rows = (single.get("groups") or {}).get(mod) or []
        if not rows:
            return "未跑或无过关组"
        best = max(rows, key=lambda r: float(r.get("cagr_ex2024") or r.get("cagr") or -9))
        d_ex = None
        if best.get("cagr_ex2024") is not None and v2.get("cagr_ex2024") is not None:
            d_ex = float(best["cagr_ex2024"]) - float(v2["cagr_ex2024"])
        return f"`{best.get('code')}` 去2024 {_pct(best.get('cagr_ex2024'))}（相对 v2 {d_ex:+.2%}）" if d_ex is not None else f"`{best.get('code')}`"

    q = {
        "Q1": "通过" if parity.get("passed") else "未通过",
        "Q2": (
            f"2024 v2={_signed(ev.get('v2_2024'))}；Event Beta（随机Top1中位）={_signed(ev.get('event_beta'))}；"
            f"Selection Alpha={_signed(ev.get('selection_alpha'))}（占 2024 {_pct(ev.get('alpha_share_of_2024'))}）"
        ),
        "Q3": _mod_help("B"),
        "Q4": _mod_help("C"),
        "Q5": f"二维 {_mod_help('D')}；截面 {_mod_help('E')}",
        "Q6": fg.get("status") or "未诊断",
        "Q7": _mod_help("G"),
        "Q8": f"两两 {len(pair.get('rows') or [])} 组，三元 {len(triple.get('rows') or [])} 组",
        "Q9": "有过门槛组" if evaluation.get("passed") else "无一组同时过晋升门槛",
        "Q10": f"RECOMMENDED_BASELINE_V3=`{rec}`" if rec else "无 RECOMMENDED_BASELINE_V3；锁仍是 baseline_v2",
    }
    lines = [
        "# 11 Round 4 结论",
        "",
        f"- 生成时间：{evaluation.get('written_at')}",
        "- 性质：historical development，**不是 final OOS / production**。",
        f"- `metric_version`：`{METRIC_VERSION}`",
        "- 默认对照：`baseline_v2` = `R3_03`。**不自动改锁。**",
        f"- 冻结：**NOT FROZEN**",
        "",
        "## Q1 Parity 过了吗？",
        "",
        q["Q1"],
        "",
        "## Q2 2024 +71.63% 多少是 Event Beta vs Selection Alpha？",
        "",
        q["Q2"],
        "",
        "## Q3 生态模块对去2024 / LOYO 有帮助吗？",
        "",
        q["Q3"],
        "",
        "## Q4 连续分差校准？",
        "",
        q["Q4"],
        "",
        "## Q5 二维 / 截面？",
        "",
        q["Q5"],
        "",
        "## Q6 F 分散过 Gate 了吗？",
        "",
        q["Q6"],
        "",
        "## Q7 CVaR？",
        "",
        q["Q7"],
        "",
        "## Q8 组合增量？",
        "",
        q["Q8"],
        "",
        "## Q9 稳健性 / 晋升门槛？",
        "",
        q["Q9"],
        "",
        "## Q10 有无 RECOMMENDED_BASELINE_V3？",
        "",
        q["Q10"],
        "",
        "达门槛也不自动改锁定代号。等你确认。",
        "",
        "## 三角",
        "",
        "| 角 | 操作 | 代号 | CAGR | Sharpe | MaxDD | 去2024 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for name, row in (evaluation.get("corners") or {}).items():
        lines.append(
            f"| {name} | {_op(row.get('code') or '', row)} | `{row.get('code')}` | {_pct(row.get('cagr'))} "
            f"| {_num(row.get('sharpe'))} | {_pct(row.get('max_drawdown'))} | {_pct(row.get('cagr_ex2024'))} |"
        )
    lines += [
        "",
        "## 过门槛组（只推荐）",
        "",
        "| 操作 | 代号 | CAGR | Sharpe | MaxDD | 去2024 | LOYO中位 | 笔数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in evaluation.get("passed") or []:
        lines.append(
            f"| {_op(row['code'], row)} | `{row['code']}` | {_pct(row.get('cagr'))} | {_num(row.get('sharpe'))} "
            f"| {_pct(row.get('max_drawdown'))} | {_pct(row.get('cagr_ex2024'))} | {_pct(row.get('loyo_median'))} "
            f"| {row.get('completed_trades')} |"
        )
    if not evaluation.get("passed"):
        lines.append("| （无） | | | | | | | |")
    path = ROUND4_ROOT / "11_round4_conclusion.md"
    assert_project_result_path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_tables(evaluation, single, pair, triple)
    return {"questions": q, "recommended": rec}


def _write_tables(evaluation, single, pair, triple) -> None:
    def dump(rel, title, rows):
        lines = [f"# {title}", "", "| 操作 | 代号 | CAGR | Sharpe | MaxDD | 去2024 | LOYO中位 | 笔数 | 结论 |",
                 "|---|---|---:|---:|---:|---:|---:|---:|---|"]
        v2 = evaluation.get("v2") or {}
        for row in rows:
            gate = promotion_gate(row, v2)
            flag = "LOW_ACTIVITY" if row.get("low_activity") else ("过门槛" if gate["meets"] else "未超过 v2")
            lines.append(
                f"| {_op(row.get('code') or '', row)} | `{row.get('code')}` | {_pct(row.get('cagr'))} "
                f"| {_num(row.get('sharpe'))} | {_pct(row.get('max_drawdown'))} | {_pct(row.get('cagr_ex2024'))} "
                f"| {_pct(row.get('loyo_median'))} | {row.get('completed_trades')} | {flag} |"
            )
        path = ROUND4_ROOT / rel
        assert_project_result_path(path)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for mod, name in (("B", "生态"), ("C", "连续分差"), ("D", "二维"), ("E", "截面"), ("F", "分散"), ("G", "CVaR")):
        dump(f"08_single/{mod}.md", f"单模块 {name}", (single.get("groups") or {}).get(mod) or [])
    dump("09_pair/pair.md", "两两组合", pair.get("rows") or [])
    dump("10_triple/triple.md", "三元组合", triple.get("rows") or [])


def append_tracking(evaluation: dict[str, Any]) -> None:
    v2 = evaluation.get("v2") or {}
    rec = evaluation.get("recommended")
    attr = _load("01_attr/attribution.json")
    ev = attr.get("event_beta_vs_alpha") or {}
    fg = attr.get("f_gate") or {}
    single = _load("08_single/single.json")
    pair = _load("09_pair/pair.json")
    triple = _load("10_triple/triple.json")
    now = datetime.now().astimezone().isoformat()
    lines = [
        "",
        "## Round 4",
        "",
        f"historical development，不是 final OOS。`metric_version` = `{METRIC_VERSION}`。默认对照仍是 **`baseline_v2` = `R3_03`**。**不自动改锁。**",
        "",
        f"Parity：R4_00 与 R3_03 逐笔{'对齐' if _load('00_parity/parity.json').get('passed') else '未对齐'}。",
        f"2024 拆解：Event Beta {_signed(ev.get('event_beta'))}，Selection Alpha {_signed(ev.get('selection_alpha'))}。",
        f"F 分散：{fg.get('status') or '未诊断'}。",
        "",
        "### 单模块（相对 v2；换 B 或换 C 已写清）",
        "",
        "| 操作 | 代号 | 换哪层 | CAGR | Sharpe | MaxDD | 去2024 | 笔数 | 结论 |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for mod in ("B", "C", "D", "E", "F", "G"):
        for row in (single.get("groups") or {}).get(mod) or []:
            spec = SPECS.get(row.get("code") or "")
            layer = (spec.replace_module if spec else mod) or mod
            gate = promotion_gate(row, v2)
            flag = "LOW_ACTIVITY" if row.get("low_activity") else ("过门槛" if gate["meets"] else "未超过 v2")
            lines.append(
                f"| {_op(row['code'], row)} | `{row['code']}` | 换{layer} | {_pct(row.get('cagr'))} "
                f"| {_num(row.get('sharpe'))} | {_pct(row.get('max_drawdown'))} | {_pct(row.get('cagr_ex2024'))} "
                f"| {row.get('completed_trades')} | {flag} |"
            )
    lines += [
        "",
        "### 两两（≤12）",
        "",
        "| 操作 | 代号 | CAGR | Sharpe | MaxDD | 去2024 | 笔数 | 结论 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in pair.get("rows") or []:
        gate = promotion_gate(row, v2)
        flag = "LOW_ACTIVITY" if row.get("low_activity") else ("过门槛" if gate["meets"] else "未超过 v2")
        lines.append(
            f"| {_op(row['code'], row)} | `{row['code']}` | {_pct(row.get('cagr'))} | {_num(row.get('sharpe'))} "
            f"| {_pct(row.get('max_drawdown'))} | {_pct(row.get('cagr_ex2024'))} | {row.get('completed_trades')} | {flag} |"
        )
    lines += [
        "",
        "### 三元（≤6）",
        "",
        "| 操作 | 代号 | CAGR | Sharpe | MaxDD | 去2024 | 笔数 | 结论 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in triple.get("rows") or []:
        gate = promotion_gate(row, v2)
        flag = "LOW_ACTIVITY" if row.get("low_activity") else ("过门槛" if gate["meets"] else "未超过 v2")
        lines.append(
            f"| {_op(row['code'], row)} | `{row['code']}` | {_pct(row.get('cagr'))} | {_num(row.get('sharpe'))} "
            f"| {_pct(row.get('max_drawdown'))} | {_pct(row.get('cagr_ex2024'))} | {row.get('completed_trades')} | {flag} |"
        )
    lines += [
        "",
        "### 三角",
        "",
        "| 角 | 操作 | 代号 | CAGR | Sharpe | MaxDD |",
        "|---|---|---|---:|---:|---:|",
    ]
    for name, row in (evaluation.get("corners") or {}).items():
        lines.append(
            f"| {name} | {_op(row.get('code') or '', row)} | `{row.get('code')}` | {_pct(row.get('cagr'))} "
            f"| {_num(row.get('sharpe'))} | {_pct(row.get('max_drawdown'))} |"
        )
    if rec:
        lines += [
            "",
            f"**RECOMMENDED_BASELINE_V3** = `{rec}`（{_op(rec)}）。未改追踪表锁定行。等你确认。",
            "",
        ]
    else:
        lines += ["", "无 RECOMMENDED_BASELINE_V3。锁定仍是 `baseline_v2`。", ""]
    lines.append(f"写入时间：{now}")
    text = TRACKING.read_text(encoding="utf-8") if TRACKING.is_file() else ""
    if "## Round 4" in text:
        head = text.split("## Round 4")[0].rstrip()
        text = head + "\n" + "\n".join(lines) + "\n"
    else:
        text = text.rstrip() + "\n" + "\n".join(lines) + "\n"
    TRACKING.write_text(text, encoding="utf-8")
