"""总报告、状态块、追踪表与标准附录。数字只来自本轮产物。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND22_ROOT
from .common import dump_json, finite_or_none, load_json, write_csv, write_md
from .contract import (
    AUTO_PROMOTE_CURRENT,
    COMPOSITE_OF_KNOWN_FACTORS,
    E0_CODE,
    FACTOR_SPECS,
    FINAL_REPORT,
    FORWARD_OOS_NOT_AVAILABLE,
    KEEP_BASELINE_V5,
    MD,
    RECOMMEND_V6_RESEARCH_CANDIDATE,
    STANDARD_MD,
    TRACKER_PATH,
    V5_ANCHOR,
)
from .zoo import run_zoo

MARKER = "### Round22 新因子发现"
STANDARD_MARKER = "## Round 22 强制增量"


def _fmt_pct(value: Any, digits: int = 2) -> str:
    v = finite_or_none(value)
    if v is None:
        return ""
    return f"{v * 100:.{digits}f}%"


def _fmt_num(value: Any, digits: int = 3) -> str:
    v = finite_or_none(value)
    if v is None:
        return ""
    return f"{v:.{digits}f}"


def _fmt_nav(value: Any) -> str:
    v = finite_or_none(value)
    if v is None:
        return ""
    return f"{v:,.2f}"


def _chapter08() -> None:
    lines = ["# Round22 · 08 假设", "", "32 个公式在冻结契约里。文献卡只说明机制，不是有效性证明。", ""]
    for spec in FACTOR_SPECS:
        lines.append(f"- `{spec['factor_id']}` 族 {spec['family']} Profile {spec['profile']}：{spec.get('short')}。公式 `{spec.get('formula')}`。")
    lines.append("")
    lines.append("D4 是已知封板持续与触板时间的乘积，标签 COMPOSITE_OF_KNOWN_FACTORS。C2/C3 与旧公式同一表达式。龙虎榜只在 Profile B。")
    lines.append("")
    write_md(MD["08"], "\n".join(lines))


def _eligible_v6(row: dict[str, Any], gap_by_code: dict[str, float | None], labels: dict[str, str]) -> bool:
    if not row.get("economically_material"):
        return False
    role = str(row.get("role") or "")
    head = role.split("_", 1)[0]
    fid = str(row.get("factor_id") or "")
    if labels.get(fid) == COMPOSITE_OF_KNOWN_FACTORS:
        return False
    if head in {"E1", "E2", "E3"}:
        return True
    if head in {"E5", "E7", "E9"}:
        gap = gap_by_code.get(str(row.get("code")))
        return gap is not None and gap > 0
    return False


def _decision(stats: dict[str, Any]) -> dict[str, Any]:
    labels = {s["factor_id"]: str(s.get("label") or "") for s in FACTOR_SPECS}
    gaps = {str(r.get("code")): finite_or_none(r.get("delta_sharpe_replace_minus_remove")) for r in (stats.get("replace_minus_remove") or [])}
    winners = [r for r in (stats.get("material") or []) if _eligible_v6(r, gaps, labels)]
    if winners:
        best = max(winners, key=lambda r: float(r.get("delta_sharpe") or -1e9))
        return {
            "final_decision": RECOMMEND_V6_RESEARCH_CANDIDATE,
            "live_ready": False,
            "reason": "经济门、费用方向、Top1/Top2 净收益与新因子增量同时成立。没有独立 Forward，不能当实盘候选。",
            "best_code": best.get("code"),
        }
    return {
        "final_decision": KEEP_BASELINE_V5,
        "live_ready": False,
        "reason": "没有一项同时满足经济门、费用方向、Top1/Top2 净收益，以及来自新因子而不是单独删除旧列的增量。",
        "best_code": None,
    }


def _counts() -> dict[str, Any]:
    pit = load_json(ROUND22_ROOT / "pit" / "pit.json")
    rows = pit.get("rows") or []
    panel = ROUND22_ROOT / "data" / "factor_panel.parquet"
    computed = 0
    if panel.is_file():
        import pyarrow.parquet as pq

        names = set(pq.read_schema(panel).names)
        computed = int(sum(s["factor_id"] in names for s in FACTOR_SPECS))
    def _n(pred) -> int:
        return int(sum(pred(str(r.get("pit_status") or "")) for r in rows))

    train = load_json(ROUND22_ROOT / "model" / "train.json")
    bt = load_json(ROUND22_ROOT / "backtest" / "sleeve.json")
    return {
        "proposed": len(FACTOR_SPECS),
        "computed": computed,
        "pit_pass": int((pit.get("summary") or {}).get("n_pit_pass") or sum(r.get("pit_status") == "PIT_PASS" for r in rows)),
        "pit_blocked": _n(lambda s: s.startswith("BLOCKED")),
        "n_duplicate": _n(lambda s: s == "DUPLICATE_EXPRESSION"),
        "n_blocked_access": _n(lambda s: s == "BLOCKED_ACCESS"),
        "n_blocked_data": _n(lambda s: s == "BLOCKED_DATA"),
        "n_blocked_pit": _n(lambda s: s == "BLOCKED_PIT"),
        "n_blocked_semantics": _n(lambda s: s == "BLOCKED_SEMANTICS"),
        "n_oof": int(train.get("n_oof_pass") or 0),
        "n_trained_real": int(train.get("n_trained_real") or 0),
        "n_imported": int(train.get("n_imported") or 0),
        "n_formal_backtests": int(sum(r.get("status") == "ran" for r in (bt.get("rows") or []))),
        "pit_rows": rows,
    }


def _best(stats: dict[str, Any]) -> dict[str, Any]:
    mats = [r for r in (stats.get("material") or []) if r.get("code") and r.get("code") != E0_CODE and r.get("delta_sharpe") is not None]
    if not mats:
        return {"code": None, "delta_sharpe": None, "delta_cagr": None, "economically_material": False}
    best = max(mats, key=lambda r: float(r["delta_sharpe"]))
    return best


def _status_block(decision: dict[str, Any], counts: dict[str, Any], stats: dict[str, Any], best: dict[str, Any]) -> dict[str, Any]:
    parity = load_json(ROUND22_ROOT / "parity" / "parity.json")
    recover = load_json(ROUND22_ROOT / "data_recovery" / "recovery.json")
    seed = stats.get("seed") or load_json(ROUND22_ROOT / "seed" / "seed_robustness.json")
    spa = stats.get("spa") or {}
    required = [
        ROUND22_ROOT / "parity" / "parity.json",
        ROUND22_ROOT / "configs" / "frozen_gates.json",
        ROUND22_ROOT / "prior" / "prior.json",
        ROUND22_ROOT / "data_recovery" / "recovery.json",
        ROUND22_ROOT / "construct" / "abde.json",
        ROUND22_ROOT / "construct" / "cfgh.json",
        ROUND22_ROOT / "pit" / "pit.json",
        ROUND22_ROOT / "ic" / "ic.json",
        ROUND22_ROOT / "correlation" / "correlation.json",
        ROUND22_ROOT / "model" / "train.json",
        ROUND22_ROOT / "backtest" / "sleeve.json",
        ROUND22_ROOT / "stats" / "stats.json",
        ROUND22_ROOT / "zoo" / "zoo.json",
    ]
    missing = [str(p.relative_to(ROUND22_ROOT)) for p in required if not p.is_file()]
    missing_md = [key for key, path in MD.items() if key != "parity" and not path.is_file()]
    forward = recover.get("forward_status") or recover.get("forward") or FORWARD_OOS_NOT_AVAILABLE
    remaining = []
    if missing:
        remaining.append("missing=" + ",".join(missing))
    if missing_md:
        remaining.append("chapters=" + ",".join(missing_md))
    if forward == FORWARD_OOS_NOT_AVAILABLE:
        remaining.append(f"forward={FORWARD_OOS_NOT_AVAILABLE}")
    if counts["pit_blocked"]:
        remaining.append(f"pit_blocked={counts['pit_blocked']}")
    if recover.get("status") not in {"DONE", ""}:
        remaining.append(f"recover={recover.get('status')}")
    status = "PARTIAL" if remaining else "COMPLETE"
    if not parity.get("passed"):
        status = "PARTIAL"
        remaining.append("parity!=PASS")
    return {
        "ROUND22_STATUS": status,
        "REMAINING": remaining,
        "BASELINE_PARITY": "PASS" if parity.get("passed") else "FAIL",
        "NEW_FACTORS_PROPOSED": 32,
        "NEW_FACTORS_COMPUTED": counts["computed"],
        "PIT_PASS": counts["pit_pass"],
        "PIT_BLOCKED": counts["pit_blocked"],
        "BEST_CANDIDATE": best.get("code"),
        "BEST_DELTA_SHARPE": best.get("delta_sharpe"),
        "BEST_DELTA_CAGR": best.get("delta_cagr"),
        "ECONOMICALLY_MATERIAL": decision["final_decision"] == RECOMMEND_V6_RESEARCH_CANDIDATE,
        "SEED_STATUS": seed.get("status"),
        "SPA_P": spa.get("p_spa"),
        "SPA_STATUS": spa.get("status"),
        "FORWARD": forward,
        "FINAL_DECISION": decision["final_decision"],
        "CURRENT_BASELINE_CHANGED": "FALSE",
        "AUTO_PROMOTE": False,
        "live_ready": False,
        "n_formal_backtests": counts["n_formal_backtests"],
        "n_oof": counts["n_oof"],
        "n_duplicate": counts["n_duplicate"],
        "n_blocked_access": counts["n_blocked_access"],
        "n_blocked_data": counts["n_blocked_data"],
        "n_blocked_pit": counts["n_blocked_pit"],
        "n_blocked_semantics": counts["n_blocked_semantics"],
        "n_trained_real": counts["n_trained_real"],
        "n_imported": counts["n_imported"],
    }


def _tracker(block: dict[str, Any], bt_rows: list[dict[str, Any]], fee_rows: list[dict[str, Any]]) -> None:
    text = TRACKER_PATH.read_text(encoding="utf-8") if TRACKER_PATH.is_file() else ""
    if MARKER in text:
        return
    lines = [
        "",
        MARKER,
        "",
        "锁定 `baseline_v5=R16_O01`。主口径 ROUND7_FLAT_5BP。经济门沿用 R21_3_ECONOMIC_MATERIALITY_V1，看候选回测前冻结。不自动升 CURRENT。",
        "",
        "### 实验组结果",
        "",
        "主表口径=**ROUND7_FLAT_5BP**。只列正式袖套。没有 OOF 入围的配置不进主表。",
        "",
        "| 操作 | 代号 | Sharpe | 年化 | 累计 | MaxDD | 笔数 | NAV | ΔSharpe | Δ年化 | 状态 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    formal = [r for r in bt_rows if r.get("status") == "ran" and finite_or_none(r.get("sharpe")) is not None]
    for row in formal:
        ds = finite_or_none(row.get("delta_sharpe"))
        if row.get("code") == E0_CODE:
            ds = 0.0
        dc = None
        if finite_or_none(row.get("cagr")) is not None and row.get("code") != E0_CODE:
            dc = float(row["cagr"]) - float(V5_ANCHOR["cagr"])
        if row.get("code") == E0_CODE:
            dc = 0.0
        lines.append(
            "| {op} | `{code}` | {sh} | {cagr} | {tot} | {dd} | {n} | {nav} | {dsh} | {dcagr} | {st} |".format(
                op=row.get("role") or row.get("code"),
                code=row.get("code"),
                sh=_fmt_num(row.get("sharpe")),
                cagr=_fmt_pct(row.get("cagr")),
                tot=_fmt_pct(row.get("total_return")),
                dd=_fmt_pct(row.get("max_drawdown")),
                n="" if row.get("completed_trades") is None else int(row.get("completed_trades")),
                nav=_fmt_nav(row.get("nav_end")),
                dsh="" if ds is None else f"{ds:+.4f}",
                dcagr="" if dc is None else f"{dc * 100:+.4f}pct",
                st="已跑",
            )
        )
    lines += ["", "### 费用校正敏感性（分表）", "", "| 操作 | 代号 | Sharpe | 年化 | 累计 | MaxDD | 笔数 | 状态 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in fee_rows:
        if finite_or_none(row.get("sharpe")) is None:
            continue
        lines.append(
            "| 费用校正 {code} | `{code}` | {sh} | {cagr} | {tot} | {dd} | {n} | {st} |".format(
                code=row.get("code"),
                sh=_fmt_num(row.get("sharpe")),
                cagr=_fmt_pct(row.get("cagr")),
                tot=_fmt_pct(row.get("total_return")),
                dd=_fmt_pct(row.get("max_drawdown")),
                n="" if row.get("completed_trades") in {None, ""} else row.get("completed_trades"),
                st=row.get("status") or "",
            )
        )
    lines += ["", "| 项 | 值 |", "| --- | --- |"]
    for key in (
        "BASELINE_PARITY",
        "NEW_FACTORS_PROPOSED",
        "NEW_FACTORS_COMPUTED",
        "PIT_PASS",
        "PIT_BLOCKED",
        "FINAL_DECISION",
        "AUTO_PROMOTE",
        "CURRENT_BASELINE_CHANGED",
        "ECONOMICALLY_MATERIAL",
        "SEED_STATUS",
        "SPA_P",
        "FORWARD",
        "ROUND22_STATUS",
        "n_formal_backtests",
        "n_oof",
        "n_trained_real",
        "n_imported",
    ):
        lines.append(f"| {key} | {block.get(key)} |")
    lines.append("| CURRENT | baseline_v5=R16_O01 |")
    lines.append("| 决策 | " + str(block.get("FINAL_DECISION")) + " |")
    lines.append("| 主表 | ROUND7_FLAT_5BP |")
    lines.append("| 主报告 | `result/round22/ROUND22_ALPHA_DISCOVERY_REPORT.md` |")
    lines.append(f"| REMAINING | {block.get('REMAINING')} |")
    lines.append("")
    TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACKER_PATH.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _standard() -> None:
    if not STANDARD_MD.is_file():
        return
    text = STANDARD_MD.read_text(encoding="utf-8")
    if STANDARD_MARKER in text:
        return
    extra = """

## Round 22 强制增量（Profile 分离 × Head-specific E0–E9 × 新公式）

在 Round 21.4 之上，本轮额外强制：

| 步 | 含义 |
|---|---|
| `PROFILE_SEPARATION` | Profile A 用 D 收盘可得信息。Profile B 用收盘后到次日开盘前。龙虎榜不得进入 Profile A |
| `HEAD_SPECIFIC_E0_E9` | E0 对照；E1/E2/E3 为 Five/Ten/Both Add；E4–E9 只在该头训练折 \\|ρ\\|≥0.60 时 Remove/Replace。低相关删除是 EXPLORATORY_DIAGNOSTIC |
| `FORMULA_EQUIVALENCE_CHECK` | 与 zoo、G/H/T/Y/L/V、封板持续或常数表达式相同则 `DUPLICATE_EXPRESSION` |
| `ECONOMIC_MATERIALITY_GATE` | 沿用 R21_3_ECONOMIC_MATERIALITY_V1，禁止回调 |
| `FORWARD_AVAILABILITY` | 无独立 Forward 不得 live-ready |

决策只允许 `KEEP_BASELINE_V5` 或 `RECOMMEND_V6_RESEARCH_CANDIDATE`。组合因子可测增量，但不能单独当成新的原始信息去升级。`AUTO_PROMOTE=False`。`CURRENT_BASELINE_CHANGED=FALSE`。
"""
    STANDARD_MD.write_text(text.rstrip() + extra, encoding="utf-8")


def _report_md(block: dict[str, Any], decision: dict[str, Any], counts: dict[str, Any]) -> str:
    parity = load_json(ROUND22_ROOT / "parity" / "parity.json")
    obs = parity.get("observed") or {}
    lines = [
        "# Round 22 Alpha 发现",
        "",
        f"最终决策 `{decision['final_decision']}`。{decision['reason']}",
        "CURRENT_BASELINE_CHANGED=FALSE。AUTO_PROMOTE=False。没有独立 Forward，不是 live-ready。",
        "",
        "## 状态块",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
    ]
    for key, value in block.items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## 基线",
        "",
        f"Stage0 重放 v5，口径 ROUND7_FLAT_5BP。CAGR={obs.get('cagr')} Sharpe={obs.get('sharpe')} MaxDD={obs.get('max_drawdown')} trades={obs.get('completed_trades')} NAV={obs.get('nav_end')}。",
        "Five 14 列，Ten 13 列（去掉 g0_amount_log）。融合 0.90 Ten + 0.10 Five。E0 仅在特征哈希与 v5 一致时 IMPORTED_BY_HASH。",
        "决策时点沿用已审计口径：D 收盘决策，D+1 开盘成交，same_open_execution_optimism=True。本轮不改成交引擎。",
        "",
        "## 漏斗",
        "",
        f"提案 {counts['proposed']}。面板里有列 {counts['computed']}。PIT_PASS {counts['pit_pass']}。阻断 {counts['pit_blocked']}（ACCESS {counts['n_blocked_access']} / DATA {counts['n_blocked_data']} / PIT {counts['n_blocked_pit']} / SEMANTICS {counts['n_blocked_semantics']}）。重复表达式 {counts['n_duplicate']}。",
        f"真训 {counts['n_trained_real']}。导入 {counts['n_imported']}。OOF 入围 {counts['n_oof']}。正式回测行 {counts['n_formal_backtests']}（含 E0）。",
        "低相关只允许 Add。Remove/Replace 必须该头训练折最近邻 |ρ|≥0.60，并且用众数最近邻自己的折内 |ρ| 中位数。同级对照一起进袖套。",
        "Profile B 与 Profile A 分开。龙虎榜源若阻断，Profile B 没有训练模型，也不并进 Profile A 表。",
        "",
        "## 章节",
        "",
        "00 契约与 v5 对照，01 谱系，02 文献机制卡，03 Five/Ten，04 截止时点，05–07 数据与 Forward，08 假设，09–10 构造，11 PIT，12 IC，13 相关，14–16 实验与 OOF，17 袖套，18 分年费用，19 Bootstrap/SPA，20 zoo，21 决策。",
        "",
        f"主报告路径 `{FINAL_REPORT}`。",
        "",
    ]
    return "\n".join(lines)


def run_report(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    del resume
    _chapter08()
    stats = load_json(ROUND22_ROOT / "stats" / "stats.json")
    decision = _decision(stats)
    dump_json(ROUND22_ROOT / "report" / "decision.json", decision)
    run_zoo(resume=False)
    counts = _counts()
    best = _best(stats)
    block = _status_block(decision, counts, stats, best)
    # zoo 章节在 run_zoo 里写。这里补 21，并在状态块算完后确认章节都在。
    write_md(
        MD["21"],
        "\n".join(
            [
                "# Round22 · 21 决策",
                "",
                f"`{decision['final_decision']}`。{decision['reason']}",
                "CURRENT_BASELINE_CHANGED=FALSE。不写 CURRENT，不升 v6 实盘。",
                "没有独立 Forward 时即使研究候选也不是 live-ready。",
                f"状态 `{block['ROUND22_STATUS']}`。未完成项 `{block['REMAINING']}`。",
                "",
            ]
        ),
    )
    block = _status_block(decision, counts, stats, best)
    dump_json(ROUND22_ROOT / "report" / "status_block.json", block)
    bt = load_json(ROUND22_ROOT / "backtest" / "sleeve.json")
    fee_path = ROUND22_ROOT / "backtest" / "fee_both_calibers.csv"
    fee_rows = pd.read_csv(fee_path).to_dict(orient="records") if fee_path.is_file() else []
    trials = []
    for row in counts["pit_rows"]:
        trials.append({"kind": "factor", **{k: row.get(k) for k in ("factor_id", "family", "profile", "pit_status", "model_eligible", "reason")}})
    reg = ROUND22_ROOT / "model" / "experiment_registry.csv"
    if reg.is_file():
        for row in pd.read_csv(reg).to_dict(orient="records"):
            trials.append({"kind": "experiment", "factor_id": row.get("factor_id"), "role": row.get("role"), "code": row.get("code"), "status": row.get("status"), "formal": row.get("formal")})
    for row in bt.get("rows") or []:
        trials.append({"kind": "backtest", "code": row.get("code"), "role": row.get("role"), "factor_id": row.get("factor_id"), "status": row.get("status"), "sharpe": row.get("sharpe"), "cagr": row.get("cagr")})
    write_csv(ROUND22_ROOT / "ledger" / "all_trials.csv", trials)
    summary = [{"key": k, "value": v} for k, v in block.items()]
    write_csv(ROUND22_ROOT / "report" / "summary_table.csv", summary)
    exec_path = ROUND22_ROOT / "diagnostics" / "execution_sensitivity.json"
    if exec_path.is_file():
        write_csv(ROUND22_ROOT / "diagnostics" / "execution_sensitivity.csv", [load_json(exec_path)])
    text = _report_md(block, decision, counts)
    write_md(FINAL_REPORT, text)
    write_md(ROUND22_ROOT / "report" / "ROUND22_ALPHA_DISCOVERY_REPORT.md", text)
    _tracker(block, bt.get("rows") or [], fee_rows)
    _standard()
    return {"status": block["ROUND22_STATUS"], "decision": decision["final_decision"], "report": str(FINAL_REPORT), "block": block}
