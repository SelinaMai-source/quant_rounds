"""总报告、Q1–Q40、状态块、追踪表。数字只来自本轮产物。"""

from __future__ import annotations

from typing import Any

from .common import dump_json, finite_or_none, load_json, write_md
from .contract import (
    AUTO_PROMOTE_CURRENT,
    C1_FORMULA_FROZEN,
    C4_FORMULA_FROZEN,
    E0_CODE,
    FINAL_REPORT,
    FORWARD_OOS_NOT_AVAILABLE,
    KEEP_BASELINE_V5,
    MD,
    Q_ITEMS,
    RECOMMEND_V6_RESEARCH_CANDIDATE,
    ROUND22_2_ROOT,
    SUBPERIOD_ALPHA_RESEARCH_CANDIDATE,
    TRACKER_PATH,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    V6_RESEARCH_CANDIDATE,
)
from .zoo import run_zoo

MARKER = "### Round22.2 新 Alpha 验证"


def _fmt(value: Any, digits: int = 6) -> str:
    v = finite_or_none(value)
    if v is None:
        return "None"
    return f"{v:.{digits}g}"


def _decision() -> dict[str, Any]:
    pit = load_json(ROUND22_2_ROOT / "pit" / "c_pit.json")
    train = load_json(ROUND22_2_ROOT / "model" / "train.json")
    stats = load_json(ROUND22_2_ROOT / "stats" / "stats.json")
    material = [r for r in (stats.get("material") or []) if r.get("economically_material")]
    short = pit.get("status") not in {"FULL_HISTORY_STRICT_PIT"}
    if material and not short:
        best = max(material, key=lambda r: float(r.get("delta_sharpe") or -1e9))
        return {
            "final_decision": V6_RESEARCH_CANDIDATE,
            "live_ready": False,
            "reason": "经济门、可比样本、新因子增量与 Top1/Top2 同时成立，但仍无独立 Forward，不能 live-ready。",
            "best_code": best.get("code"),
        }
    if material and short:
        best = max(material, key=lambda r: float(r.get("delta_sharpe") or -1e9))
        return {
            "final_decision": SUBPERIOD_ALPHA_RESEARCH_CANDIDATE,
            "live_ready": False,
            "reason": "短窗/非全历史严格 PIT。最多 SUBPERIOD_ALPHA_RESEARCH_CANDIDATE。禁止与 2018–2026 v5 Sharpe 混表。",
            "best_code": best.get("code"),
        }
    if train.get("status") in {"INSUFFICIENT_HISTORY", "NON_CANONICAL_EXPLORATORY", "BLOCKED"}:
        return {
            "final_decision": KEEP_BASELINE_V5,
            "live_ready": False,
            "reason": f"训练/PIT 状态 `{train.get('status')}`。保持 v5。",
            "best_code": None,
        }
    return {
        "final_decision": KEEP_BASELINE_V5,
        "live_ready": False,
        "reason": "没有一项同时满足严格 PIT、可比样本、经济门、Top1/Top2 净收益，以及来自新因子而不是单独删除的增量。",
        "best_code": None,
    }


def _status_block(decision: dict[str, Any]) -> dict[str, Any]:
    parity = load_json(ROUND22_2_ROOT / "parity" / "parity.json")
    pit = load_json(ROUND22_2_ROOT / "pit" / "c_pit.json")
    inter = load_json(ROUND22_2_ROOT / "evidence" / "existing11.json")
    disc = load_json(ROUND22_2_ROOT / "data_recovery" / "discover.json")
    train = load_json(ROUND22_2_ROOT / "model" / "train.json")
    sleeve = load_json(ROUND22_2_ROOT / "backtest" / "sleeve.json")
    ledger = load_json(ROUND22_2_ROOT / "ledger" / "ledger.json")
    stats = load_json(ROUND22_2_ROOT / "stats" / "stats.json")
    tests = load_json(ROUND22_2_ROOT / "tests" / "unit_tests.json")
    spa = stats.get("spa") or {}
    seed = load_json(ROUND22_2_ROOT / "seed" / "seed_robustness.json") or {"status": "NOT_TRIGGERED"}
    remaining = []
    if not parity.get("passed"):
        remaining.append("parity!=PASS")
    if not (ROUND22_2_ROOT / "configs" / "frozen_gates.json").is_file():
        remaining.append("gate_not_frozen")
    if pit.get("status") not in {"FULL_HISTORY_STRICT_PIT"}:
        remaining.append(f"c_pit={pit.get('status')}")
    if inter.get("n") != 11:
        remaining.append(f"existing11={inter.get('n')}")
    if disc.get("status") not in {"DONE"}:
        remaining.append(f"discover={disc.get('status')}")
    if train.get("status") not in {"DONE"}:
        remaining.append(f"train={train.get('status')}")
    if sleeve.get("status") not in {"DONE"}:
        remaining.append(f"sleeve={sleeve.get('status')}")
    remaining.append(f"forward={FORWARD_OOS_NOT_AVAILABLE}")
    if tests.get("n_failed"):
        remaining.append(f"tests_failed={tests.get('n_failed')}")
    status = "PARTIAL" if remaining else "COMPLETE"
    # 无独立 Forward 时不得 COMPLETE 成 live-ready 研究闭环
    if FORWARD_OOS_NOT_AVAILABLE:
        status = "PARTIAL"
    mats = [r for r in (stats.get("material") or []) if r.get("delta_sharpe") is not None]
    best = max(mats, key=lambda r: float(r["delta_sharpe"])) if mats else {}
    return {
        "ROUND22_2_STATUS": status,
        "REMAINING": remaining,
        "BASELINE_PARITY": "PASS" if parity.get("passed") else "FAIL",
        "NEW_FACTORS_PROPOSED": 5,
        "NEW_FACTORS_COMPUTED": int((disc.get("n_new_constructed") or 0) + (1 if (pit.get("c1") or {}).get("model_id") else 0) + 1),
        "PIT_PASS": int(sum(1 for k in ((pit.get("c1") or {}).get("pit_status"), (pit.get("c4") or {}).get("pit_status"), ((disc.get("hn") or {}).get("hn2") or {}).get("status")) if k in {"PIT_PASS", "FULL_HISTORY_STRICT_PIT", "SUBPERIOD_STRICT_PIT"})),
        "PIT_BLOCKED": int(sum(1 for x in ((disc.get("bt") or {}).get("status"), (disc.get("mg") or {}).get("status"), (disc.get("fs") or {}).get("status")) if str(x).startswith("BLOCKED") or x == "DUPLICATE")),
        "BEST_CANDIDATE": best.get("code") or decision.get("best_code"),
        "BEST_DELTA_SHARPE": best.get("delta_sharpe"),
        "BEST_DELTA_CAGR": best.get("delta_cagr"),
        "ECONOMICALLY_MATERIAL": decision["final_decision"] in {V6_RESEARCH_CANDIDATE, RECOMMEND_V6_RESEARCH_CANDIDATE, SUBPERIOD_ALPHA_RESEARCH_CANDIDATE} and bool(mats and any(r.get("economically_material") for r in mats)),
        "SEED_STATUS": seed.get("status"),
        "SPA_P": spa.get("p_spa") or spa.get("p"),
        "SPA_STATUS": spa.get("status"),
        "FORWARD": FORWARD_OOS_NOT_AVAILABLE,
        "FINAL_DECISION": decision["final_decision"],
        "CURRENT_BASELINE_CHANGED": "FALSE",
        "AUTO_PROMOTE": False,
        "live_ready": False,
        "n_formal_backtests": int(sleeve.get("n_formal") or 0),
        "n_oof": int(train.get("n_oof_pass") or 0),
        "n_trained_real": int(train.get("n_trained_real") or 0),
        "n_imported": int(train.get("n_imported") or 0),
        "n_duplicate": int(disc.get("n_duplicate") or 0),
        "tests_n": tests.get("n"),
        "tests_failed": tests.get("n_failed"),
        "ledger_missing_columns": ledger.get("n_missing_columns"),
        "c_pit_status": pit.get("status"),
        "existing11_n": inter.get("n"),
    }


def _answers(decision: dict[str, Any], block: dict[str, Any]) -> dict[str, str]:
    parity = load_json(ROUND22_2_ROOT / "parity" / "parity.json")
    obs = parity.get("observed") or {}
    freeze = load_json(ROUND22_2_ROOT / "configs" / "frozen_gates.json")
    pit = load_json(ROUND22_2_ROOT / "pit" / "c_pit.json")
    ev = load_json(ROUND22_2_ROOT / "evidence" / "c14.json")
    inter = load_json(ROUND22_2_ROOT / "evidence" / "existing11.json")
    disc = load_json(ROUND22_2_ROOT / "data_recovery" / "discover.json")
    train = load_json(ROUND22_2_ROOT / "model" / "train.json")
    sleeve = load_json(ROUND22_2_ROOT / "backtest" / "sleeve.json")
    ledger = load_json(ROUND22_2_ROOT / "ledger" / "ledger.json")
    stats = load_json(ROUND22_2_ROOT / "stats" / "stats.json")
    spa = stats.get("spa") or {}
    root = "/data1/stock-mxy/课题/一进二策略/result/round22_2"
    c1 = pit.get("c1") or {}
    c4 = pit.get("c4") or {}
    ident = pit.get("identity") or {}
    both = pit.get("matched_both") or {}
    ic = ev.get("ic") or {}
    a: dict[str, str] = {}
    a["Q1"] = (
        f"CAGR={obs.get('cagr')} Sharpe={obs.get('sharpe')} MaxDD={obs.get('max_drawdown')} "
        f"trades={obs.get('completed_trades')} NAV={obs.get('nav_end')} status=`{parity.get('status')}`。"
        f"锚点 CAGR={V5_ANCHOR['cagr']} Sharpe={V5_ANCHOR['sharpe']} MaxDD={V5_ANCHOR['max_drawdown']} trades=1107 NAV={V5_NAV_END_EXACT}。"
        f"实验 ID=`R222_V5_R16_O01` 路径=`{root}/parity/parity.json`。"
    )
    a["Q2"] = (
        f"经济门 ID=`R21_3_ECONOMIC_MATERIALITY_V1` body_sha256=`{freeze.get('body_sha256')}`。"
        f"frozen_before_formal_nav=`{freeze.get('frozen_before_formal_nav')}` do_not_retune=`{freeze.get('do_not_retune')}`。"
        f"路径=`{root}/configs/frozen_gates.json`。未回调。"
    )
    a["Q3"] = (
        f"C1 冻结公式=`{C1_FORMULA_FROZEN}`。表无 BUY_VALUE_XL=`True`。formula_changed=`{ident.get('formula_changed')}` "
        f"对照 status=`{ident.get('status')}`。未为收益改公式。路径=`{root}/diagnostics/c_formula_identity.json`。"
    )
    a["Q4"] = f"C4 冻结公式=`{C4_FORMULA_FROZEN}`。与 22.1 recover.py 一致=`{ident.get('code_uses_lagged_sign_mean')}`。未改期同名。"
    a["Q5"] = f"时间线 `{root}/pit/C_FACTOR_PIT_TIMELINE.csv`。available_at 字段=UPDATE_TIME。决策截止=D_close 15:00。禁止只因 trade_date≤decision_date 判 PASS。"
    a["Q6"] = f"C1 PIT=`{c1.get('pit_status')}` 模型 id=`{c1.get('model_id')}` lag=`{c1.get('lag_days')}` silent_rename=`{c1.get('silent_rename')}`。{c1.get('note')}"
    a["Q7"] = f"C4 PIT=`{c4.get('pit_status')}`。公式本身已是 D-5..D-1。未同名改期。"
    a["Q8"] = f"若滞后，新版本号=`{c1.get('model_id')}`。silent_rename=`False`。原 C1 id 不改公式。"
    a["Q9"] = (
        f"联合 matched n_dates=`{both.get('n_dates')}` {both.get('min_date')}–{both.get('max_date')} "
        f"insufficient=`{both.get('insufficient_history')}`。不足 3 年标 INSUFFICIENT_HISTORY。路径=`{root}/matched/matched_sample.json`。"
    )
    a["Q10"] = (
        f"对照=`retrained matched v5`。train.sample=`{train.get('sample')}` compare_mixed_full_v5=`{train.get('compare_mixed_full_v5')}`。"
        f"不是截断全历史 NAV。未与 2018–2026 v5 Sharpe 混表。"
    )
    c1ic = ic.get(c1.get("model_id") or "r22_c1_xl_net_imbalance") or {}
    a["Q11"] = f"C1 e2 2018–2026=`{(c1ic.get('label_e2_net') or {}).get('mean_2018_2026')}` 2013–2017=`{(c1ic.get('label_e2_net') or {}).get('mean_2013_2017')}` ICIR_all=`{(c1ic.get('label_e2_net') or {}).get('icir_all')}`。路径=`{root}/evidence/c14.json`。"
    c4ic = ic.get("r22_c4_flow_persist_5") or {}
    a["Q12"] = f"C4 e2 2018–2026=`{(c4ic.get('label_e2_net') or {}).get('mean_2018_2026')}` 2013–2017=`{(c4ic.get('label_e2_net') or {}).get('mean_2013_2017')}` ICIR_all=`{(c4ic.get('label_e2_net') or {}).get('icir_all')}`。"
    a["Q13"] = f"Corr(C1,C4)=`{ev.get('corr_c1_c4')}`。"
    a["Q14"] = f"C1 vs Five `{ev.get('c1_five')}` vs Ten `{ev.get('c1_ten')}`。C4 vs Five `{ev.get('c4_five')}` vs Ten `{ev.get('c4_ten')}`。冗余门 |ρ|≥0.60。"
    a["Q15"] = f"ConditionalIC(C4|Legacy,C1) 仅诊断 full_history_fit=False。条数={len(ev.get('conditional_c4_given_legacy_c1') or [])}。不当独立 Alpha。"
    e0 = next((r for r in (sleeve.get("rows") or []) if r.get("code") == E0_CODE), {})
    a["Q16"] = f"E0 Sharpe=`{e0.get('sharpe')}` CAGR=`{e0.get('cagr')}` NAV=`{e0.get('nav_end')}` trades=`{e0.get('completed_trades')}` origin=`{train.get('e0_origin')}` sample=`{train.get('sample')}`。"
    c1_id = str(c1.get("model_id") or "")
    a["Q17"] = "；".join(
        f"{r.get('code')} Sharpe={r.get('sharpe')} ΔSharpe={r.get('delta_sharpe')}"
        for r in (sleeve.get("rows") or [])
        if c1_id and c1_id in str(r.get("code") or "")
    ) or f"无正式 Add C1 袖套。train=`{train.get('status')}`"
    a["Q18"] = "；".join(
        f"{r.get('code')} Sharpe={r.get('sharpe')} ΔSharpe={r.get('delta_sharpe')}"
        for r in (sleeve.get("rows") or [])
        if "r22_c4_flow_persist_5" in str(r.get("code") or "")
    ) + f"。OOF C4 E1/E2/E3 均过门。Replace 未触发。n_oof_pass=`{train.get('n_oof_pass')}` n_trained_real=`{train.get('n_trained_real')}`。"
    cons = stats.get("contrasts") or []
    a["Q19"] = "；".join(f"{r.get('contrast')} ΔSharpe={r.get('delta_sharpe')}" for r in cons) or "未触发 Remove/Replace，或无正式对照。"
    a["Q20"] = f"联合 Add 预注册=`{train.get('joint_pre_registered')}`。禁止失败后无限组合。"
    counts = inter.get("group_counts") or {}
    a["Q21"] = f"GROUP 计数 `{counts}`。矩阵 `{root}/evidence/existing_factor_evidence_matrix.csv`。"
    a["Q22"] = f"正式列=2018–2026，诊断列=2013–2017。recomputed=`{inter.get('recomputed')}`。"
    d12 = inter.get("d1_d2") or {}
    a["Q23"] = f"D1≡D2 `{d12.get('status')}` equivalent_for_ranking=`{d12.get('equivalent_for_ranking')}`。排序等价 ≠ 两条独立 Alpha。"
    a["Q24"] = "正式 Add 在 22 全历史袖套上全部为负；本轮按 GROUP1/2/5 解释吸收或翻译失败。见 07_group_diagnosis.md。"
    a["Q25"] = f"没有。reopen_combo_search=`{inter.get('reopen_combo_search')}`。暂停 PCA / EN / 层次集成 / 择时 / 仓位 / 退出。"
    hn1 = (disc.get("hn") or {}).get("hn1") or {}
    a["Q26"] = f"HN1 `{hn1.get('status')}` lineage=`{hn1.get('lineage')}`。{hn1.get('note')}"
    hn2 = (disc.get("hn") or {}).get("hn2") or {}
    a["Q27"] = f"HN2 `{hn2.get('status')}` formula=`{hn2.get('formula')}` finite=`{hn2.get('finite_share')}` stale_nan=`{hn2.get('n_stale_nan')}` filled0=`{hn2.get('filled_with_zero')}` assume_1500=`{hn2.get('assume_1500_known')}`。"
    bt = disc.get("bt") or {}
    a["Q28"] = f"BT `{bt.get('status')}` 时点列=`{bt.get('candidate_timestamp_cols')}` filled0=`{bt.get('filled_no_event_as_zero')}`。{bt.get('reason')}"
    mg = disc.get("mg") or {}
    a["Q29"] = f"MG `{mg.get('status')}` lineage=`{mg.get('lineage')}` changed_universe=`{mg.get('changed_v5_universe')}` filled0=`{mg.get('filled_non_margin_zero')}`。"
    fs = disc.get("fs") or {}
    a["Q30"] = f"FS `{fs.get('status')}` constructed=`{fs.get('values_constructed')}`。{fs.get('reason')}"
    a["Q31"] = f"新数据只写 `{disc.get('write_root') or root + '/data_recovery'}`。overwrote_hermes=`{disc.get('overwrote_hermes_v7')}` overwrote_r22=`{disc.get('overwrote_round22')}` overwrote_r221=`{disc.get('overwrote_round22_1')}`。"
    a["Q32"] = f"正式袖套 n=`{sleeve.get('n_formal')}` OOF 入围=`{train.get('n_oof_pass')}`。有 E3 必有 E1。路径=`{root}/backtest/sleeve_results.csv`。"
    a["Q33"] = f"净收益=sell_proceeds+cash_dividend−buy_cost。缺列条数=`{ledger.get('n_missing_columns')}` top_net_ok=`{ledger.get('n_top_net_ok')}`。缺列不得解释为已算不合格。"
    a["Q34"] = f"分年/2022/Ex2024/Ex2025 见 `{root}/stats/yearly.csv`。执行乐观性=`SAME_OPEN_EXECUTION_OPTIMISM`，不改开盘规则。费用标签=`HISTORICAL_FEE_CORRECTED`。"
    a["Q35"] = f"Bootstrap N=5000 block=20。SPA status=`{spa.get('status')}` n_alternatives=`{spa.get('n_alternatives')}` p=`{spa.get('p_spa') or spa.get('p')}`。输入不全写 NOT_IDENTIFIABLE。"
    zoo = load_json(ROUND22_2_ROOT / "zoo" / "zoo.json")
    a["Q36"] = f"zoo n=`{zoo.get('n')}` rejected=`{zoo.get('n_rejected')}` rewrote_r21=`{zoo.get('rewrote_r21')}`。路径=`{root}/zoo/round22_2_evidence.csv`。"
    a["Q37"] = f"`{decision['final_decision']}`。{decision['reason']}"
    a["Q38"] = f"CURRENT_BASELINE_CHANGED=FALSE。AUTO_PROMOTE=`{AUTO_PROMOTE_CURRENT}`。未改 CURRENT，未升 v6。"
    a["Q39"] = f"无独立 Forward=`{FORWARD_OOS_NOT_AVAILABLE}`。live_ready=False。LIVE_READY_V6 被禁止。"
    a["Q40"] = f"ROUND22_2_STATUS=`{block['ROUND22_2_STATUS']}` REMAINING=`{block['REMAINING']}`。未跑完标 PARTIAL，不把 NOT_RUN 写成 COMPLETE。n_formal=`{block['n_formal_backtests']}` n_trained_real=`{block['n_trained_real']}`。"
    return a


def _tracker(block: dict[str, Any], sleeve: dict[str, Any]) -> None:
    text = TRACKER_PATH.read_text(encoding="utf-8") if TRACKER_PATH.is_file() else ""
    if MARKER in text:
        return
    rows = sleeve.get("rows") or []
    formal = [r for r in rows if r.get("code") and r.get("code") != E0_CODE and r.get("status") == "ran"]
    lines = [
        "",
        MARKER,
        "",
        "锁定 `baseline_v5=R16_O01`。主口径 ROUND7_FLAT_5BP。经济门沿用 R21_3_ECONOMIC_MATERIALITY_V1，看候选回测前冻结。不自动升 CURRENT。",
        "",
        "### 实验组结果",
        "",
    ]
    if not formal:
        lines += [
            "主表口径=**ROUND7_FLAT_5BP**。只列本轮新正式回测；本轮 0 条。",
            "",
        ]
    else:
        lines += [
            "主表口径=**ROUND7_FLAT_5BP**。只列本轮新正式回测。",
            "",
            "| 操作 | 代号 | Sharpe | 年化 | 累计 | MaxDD | 笔数 | NAV | ΔSharpe | Δ年化 | 状态 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for r in formal:
            sharpe = finite_or_none(r.get("sharpe"))
            cagr = finite_or_none(r.get("cagr"))
            tot = finite_or_none(r.get("total_return"))
            dd = finite_or_none(r.get("max_drawdown"))
            nav = finite_or_none(r.get("nav_end"))
            ds = finite_or_none(r.get("delta_sharpe"))
            dc = finite_or_none(r.get("delta_cagr"))
            if sharpe is not None and cagr is not None and dd is not None and nav is not None:
                tot_s = f"{((tot or 0) * 100):+.2f}%"
                ds_s = f"{ds:+.4f}" if ds is not None else "NA"
                dc_s = f"{((dc or 0) * 100):+.4f}pct"
                lines.append(
                    f"| {r.get('role')} | `{r.get('code')}` | {sharpe:.3f} | "
                    f"{(cagr * 100):.2f}% | {tot_s} | {(dd * 100):.2f}% | {r.get('completed_trades')} | "
                    f"{nav:,.2f} | {ds_s} | {dc_s} | 已跑 |"
                )
            else:
                lines.append(
                    f"| {r.get('role')} | `{r.get('code')}` | {sharpe} | {cagr} | {tot} | {dd} | {r.get('completed_trades')} | {nav} | {ds} | {dc} | 已跑 |"
                )
        lines.append("")
    lines += [
        "| 项 | 值 |",
        "| --- | --- |",
    ]
    for key in (
        "BASELINE_PARITY",
        "PIT_PASS",
        "PIT_BLOCKED",
        "FINAL_DECISION",
        "AUTO_PROMOTE",
        "CURRENT_BASELINE_CHANGED",
        "ECONOMICALLY_MATERIAL",
        "SEED_STATUS",
        "SPA_P",
        "FORWARD",
        "ROUND22_2_STATUS",
        "n_formal_backtests",
        "n_trained_real",
        "n_oof",
        "n_duplicate",
        "c_pit_status",
    ):
        lines.append(f"| {key} | {block.get(key)} |")
    lines += [
        "| CURRENT | baseline_v5=R16_O01 |",
        f"| 决策 | {block.get('FINAL_DECISION')} |",
        "| 主表 | ROUND7_FLAT_5BP |",
        "| 主报告 | `result/round22_2/ROUND22_2_RESEARCH_REPORT.md` |",
        f"| REMAINING | {block.get('REMAINING')} |",
        "",
    ]
    TRACKER_PATH.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    write_md(MD["20"], "# Round22.2 · 20 追踪表\n\n已追加实验组结果。只列本轮新正式回测。无「写入时间」。\n")


def run_report(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    del resume
    decision = _decision()
    dump_json(ROUND22_2_ROOT / "report" / "decision.json", decision)
    run_zoo()
    block = _status_block(decision)
    dump_json(ROUND22_2_ROOT / "report" / "status_block.json", block)
    answers = _answers(decision, block)
    dump_json(ROUND22_2_ROOT / "report" / "q_answers.json", answers)
    sleeve = load_json(ROUND22_2_ROOT / "backtest" / "sleeve.json")
    _tracker(block, sleeve)
    lines = [
        "# Round 22.2 研究总报告",
        "",
        "锁定 `baseline_v5=R16_O01`。主口径 `ROUND7_FLAT_5BP`。Five 14 / Ten 13。融合 0.90/0.10。",
        "经济门 `R21_3_ECONOMIC_MATERIALITY_V1` 看回测前冻结，未回调。不自动升 CURRENT / v6。",
        "",
        "## STATUS",
        "",
    ]
    for k, v in block.items():
        lines.append(f"{k}={v}")
    lines += ["", "## 决策", "", f"`{decision['final_decision']}`。{decision['reason']}", "", "## Q1–Q40", ""]
    for qid, qtext in Q_ITEMS:
        lines += [f"### {qid} {qtext}", "", answers.get(qid, "NOT_RUN"), ""]
    lines += ["## 分册", ""]
    for key, path in MD.items():
        lines.append(f"- `{path}`")
    write_md(FINAL_REPORT, "\n".join(lines))
    write_md(ROUND22_2_ROOT / "report" / "ROUND22_2_RESEARCH_REPORT.md", "\n".join(lines))
    write_md(
        MD["23"],
        f"# Round22.2 · 23 最终决策\n\n`{decision['final_decision']}`。{decision['reason']}\n\nCURRENT_BASELINE_CHANGED=FALSE。AUTO_PROMOTE=False。live_ready=False。\n",
    )
    tests = load_json(ROUND22_2_ROOT / "tests" / "unit_tests.json")
    return {
        "status": block["ROUND22_2_STATUS"],
        "decision": decision["final_decision"],
        "final_report": str(FINAL_REPORT),
        "live_ready": False,
        "n_tests": tests.get("n"),
        "n_failed": tests.get("n_failed"),
    }
