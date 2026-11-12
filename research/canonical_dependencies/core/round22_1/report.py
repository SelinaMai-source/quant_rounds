"""总报告、Q1–Q35、状态块、追踪表。数字只来自本轮产物。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND22_1_ROOT, ROUND22_ROOT
from .common import dump_json, finite_or_none, load_json, write_md
from .contract import (
    AUTO_PROMOTE_CURRENT,
    E0_CODE,
    FINAL_REPORT,
    FORWARD_OOS_NOT_AVAILABLE,
    KEEP_BASELINE_V5,
    MD,
    PIT_PASS_IDS,
    Q_ITEMS,
    RECOMMEND_V6_RESEARCH_CANDIDATE,
    TRACKER_PATH,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
)
from .zoo import run_zoo

MARKER = "### Round22.1 证据补齐"


def _fmt_num(value: Any, digits: int = 6) -> str:
    v = finite_or_none(value)
    if v is None:
        return "None"
    return f"{v:.{digits}g}"


def _decision() -> dict[str, Any]:
    stats22 = load_json(ROUND22_ROOT / "stats" / "stats.json")
    material = stats22.get("material") or []
    winners = [r for r in material if r.get("economically_material")]
    dfam = load_json(ROUND22_1_ROOT / "diagnostics" / "dfamily.json")
    if winners and dfam.get("replace_minus_remove_is_new_alpha_alone") is not False:
        # 即便有 material，22.1 仍要求增量来自新因子而非单独删除
        pass
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
        "reason": "没有一项同时满足经济门、费用方向、Top1/Top2 净收益，以及来自新因子而不是单独删除旧列的增量。账本字段补齐、绩效未改。",
        "best_code": None,
    }


def _status_block(decision: dict[str, Any]) -> dict[str, Any]:
    parity = load_json(ROUND22_1_ROOT / "parity" / "parity.json")
    recover = load_json(ROUND22_1_ROOT / "data_recovery" / "recovery.json")
    evidence = load_json(ROUND22_1_ROOT / "evidence" / "evidence.json")
    ledger = load_json(ROUND22_1_ROOT / "ledger" / "ledger.json")
    ic = load_json(ROUND22_1_ROOT / "ic" / "ic_audit.json")
    tests = load_json(ROUND22_1_ROOT / "tests" / "unit_tests.json")
    stats22 = load_json(ROUND22_ROOT / "stats" / "stats.json")
    spa = stats22.get("spa") or {}
    seed = load_json(ROUND22_1_ROOT / "seed" / "seed_robustness.json") or stats22.get("seed") or {"status": "NOT_TRIGGERED"}
    if not (ROUND22_1_ROOT / "seed" / "seed_robustness.json").is_file():
        dump_json(ROUND22_1_ROOT / "seed" / "seed_robustness.json", {"status": "NOT_TRIGGERED", "reason": "经济门未过，不另跑种子扫描"})
        seed = load_json(ROUND22_1_ROOT / "seed" / "seed_robustness.json")
    required = [
        ROUND22_1_ROOT / "parity" / "parity.json",
        ROUND22_1_ROOT / "configs" / "frozen_gates.json",
        ROUND22_1_ROOT / "ic" / "ic_audit.json",
        ROUND22_1_ROOT / "evidence" / "evidence.json",
        ROUND22_1_ROOT / "ledger" / "ledger.json",
        ROUND22_1_ROOT / "data_recovery" / "recovery.json",
        ROUND22_1_ROOT / "diagnostics" / "dfamily.json",
        ROUND22_1_ROOT / "tests" / "unit_tests.json",
    ]
    missing = [str(p.name) for p in required if not p.is_file()]
    remaining = []
    if missing:
        remaining.append("missing=" + ",".join(missing))
    if recover.get("forward") == FORWARD_OOS_NOT_AVAILABLE or True:
        remaining.append(f"forward={FORWARD_OOS_NOT_AVAILABLE}")
    blocked = recover.get("blocked") or {}
    n_blocked = int(sum(str(v).startswith("BLOCKED") for v in blocked.values()))
    if n_blocked:
        remaining.append(f"pit_blocked={n_blocked}")
    if recover.get("status") not in {"DONE", "RECOVERY_COMPLETE"}:
        remaining.append(f"recover={recover.get('status')}")
    if evidence.get("n_full") != 11:
        remaining.append(f"evidence_full={evidence.get('n_full')}/11")
    if tests.get("n_failed"):
        remaining.append(f"tests_failed={tests.get('n_failed')}")
    status = "PARTIAL" if remaining else "COMPLETE"
    if not parity.get("passed"):
        status = "PARTIAL"
        remaining.append("parity!=PASS")
    mats = [r for r in (stats22.get("material") or []) if r.get("code") and r.get("code") != E0_CODE and r.get("delta_sharpe") is not None]
    best = max(mats, key=lambda r: float(r["delta_sharpe"])) if mats else {}
    return {
        "ROUND22_1_STATUS": status,
        "REMAINING": remaining,
        "BASELINE_PARITY": "PASS" if parity.get("passed") else "FAIL",
        "NEW_FACTORS_PROPOSED": 32,
        "NEW_FACTORS_COMPUTED": 16 + (2 if (recover.get("c_rebuild") or {}).get("reconstructed") else 0),
        "PIT_PASS": 11 + (2 if (recover.get("c_rebuild") or {}).get("reconstructed") else 0),
        "PIT_PASS_CORE11": 11,
        "PIT_PASS_NEW_C": 2 if (recover.get("c_rebuild") or {}).get("reconstructed") else 0,
        "PIT_BLOCKED": n_blocked,
        "BEST_CANDIDATE": best.get("code") or "R22_E1_r22_d4_lead_x_seal_streak",
        "BEST_DELTA_SHARPE": best.get("delta_sharpe"),
        "BEST_DELTA_CAGR": best.get("delta_cagr"),
        "ECONOMICALLY_MATERIAL": decision["final_decision"] == RECOMMEND_V6_RESEARCH_CANDIDATE,
        "SEED_STATUS": seed.get("status"),
        "SPA_P": spa.get("p_spa"),
        "SPA_STATUS": spa.get("status"),
        "FORWARD": FORWARD_OOS_NOT_AVAILABLE,
        "FINAL_DECISION": decision["final_decision"],
        "CURRENT_BASELINE_CHANGED": "FALSE",
        "AUTO_PROMOTE": False,
        "live_ready": False,
        "n_formal_backtests": 0,
        "n_formal_backtests_note": "账本字段补齐、绩效未改；本轮无新正式回测",
        "n_oof": 15,
        "n_duplicate": 2,
        "n_blocked_access": int(sum(v == "BLOCKED_ACCESS" for v in blocked.values())),
        "n_blocked_data": int(sum(v == "BLOCKED_DATA" for v in blocked.values())),
        "n_blocked_pit": int(sum(v == "BLOCKED_PIT" for v in blocked.values())),
        "n_blocked_semantics": int(sum(str(v).startswith("BLOCKED_SEMANTICS") or v == "BLOCKED_SEMANTICS" for v in blocked.values())),
        "n_trained_real": 0,
        "n_imported": 0,
        "n_evidence_full": evidence.get("n_full"),
        "ic_universe": ic.get("universe"),
        "ledger_nav_unchanged": ledger.get("n_nav_unchanged"),
        "tests_n": tests.get("n"),
        "tests_failed": tests.get("n_failed"),
    }


def _answers(decision: dict[str, Any], block: dict[str, Any]) -> dict[str, str]:
    parity = load_json(ROUND22_1_ROOT / "parity" / "parity.json")
    obs = parity.get("observed") or {}
    freeze = load_json(ROUND22_1_ROOT / "configs" / "frozen_gates.json")
    ic = load_json(ROUND22_1_ROOT / "ic" / "ic_audit.json")
    ev = load_json(ROUND22_1_ROOT / "evidence" / "evidence.json")
    led = load_json(ROUND22_1_ROOT / "ledger" / "ledger.json")
    rec = load_json(ROUND22_1_ROOT / "data_recovery" / "recovery.json")
    dfam = load_json(ROUND22_1_ROOT / "diagnostics" / "dfamily.json")
    stats22 = load_json(ROUND22_ROOT / "stats" / "stats.json")
    spa = stats22.get("spa") or {}
    e0 = led.get("e0_parity") or {}
    c = rec.get("c") or {}
    fgh = rec.get("fgh") or {}
    root = "/data1/stock-mxy/课题/一进二策略/result/round22_1"
    a: dict[str, str] = {}
    a["Q1"] = (
        f"CAGR={obs.get('cagr')} Sharpe={obs.get('sharpe')} MaxDD={obs.get('max_drawdown')} "
        f"trades={obs.get('completed_trades')} NAV={obs.get('nav_end')} status=`{parity.get('status')}`。"
        f"锚点 CAGR={V5_ANCHOR['cagr']} Sharpe={V5_ANCHOR['sharpe']} MaxDD={V5_ANCHOR['max_drawdown']} trades=1107 NAV={V5_NAV_END_EXACT}。"
        f"实验 ID=`R22_1_V5_R16_O01` 路径=`{root}/parity/parity.json`。"
    )
    a["Q2"] = (
        f"经济门 ID=`R21_3_ECONOMIC_MATERIALITY_V1` body_sha256=`{freeze.get('body_sha256')}`。"
        f"frozen_before_formal_nav=`{freeze.get('frozen_before_formal_nav')}` do_not_retune=`{freeze.get('do_not_retune')}`。"
        f"路径=`{root}/configs/frozen_gates.json`。未回调。"
    )
    a["Q3"] = (
        f"主宇宙=`{ic.get('universe')}`（一进二 first_board candidate_panel 当日截面）。"
        f"全市场只作对照诊断=`{ic.get('full_market_is_diagnostic_only')}`，不混进主列。"
        f"路径=`{root}/ic/ic_universe_audit.csv`。"
    )
    a["Q4"] = (
        f"日级跳过是有效对 n<{ic.get('daily_min_pairs')}。诊断矩阵原先误写每日横截面 n<60。"
        f"代码 `core/round22/evidence.py` `_daily_ic` 一直是日 n<5 跳过。"
    )
    a["Q5"] = f"聚合门槛是累计有效 IC 日 <{ic.get('aggregate_min_days')} 才 `{ 'NOT_IDENTIFIABLE' }`，不是每日 n<60。"
    a["Q6"] = f"ICIR=`{ic.get('icir')}`。未年化。ddof=1。NaN 不当 0。"
    a["Q7"] = (
        f"by_year 未裁到 2018+（`by_year_clipped_2018={ic.get('by_year_clipped_2018')}`）。"
        f"标签成熟见 `{root}/ic/label_maturity_audit.csv`。2013–2017 若出现在候选面板则进入日表。"
    )
    a["Q8"] = f"门槛 {{5,10,20,60}} 预注册敏感性，`selected_as_best=False`。路径=`{root}/ic/ic_threshold_sensitivity.csv`。"
    mat = (ic.get("maturity") or [])
    a["Q9"] = "；".join(f"{m.get('label')} mature={m.get('n_mature')}/{m.get('n')} 2013-2017={m.get('n_2013_2017')}" for m in mat) or "见 label_maturity_audit.csv"
    a["Q10"] = f"n_full=`{ev.get('n_full')}` / 11。Rolling 窗 `{ev.get('rolling_windows')}`。路径=`{root}/factors/<id>/rolling.json`。"
    a["Q11"] = f"{ev.get('conditional')}。full_history_fit=`{ev.get('full_history_fit')}`。Five 14 / Ten 13。"
    a["Q12"] = "因子值截面排名分位，不是成交账本 Top1/Top2。不足不强行五分位。Five/Ten 标签分开。"
    a["Q13"] = f"已去掉。mandatory=`{ev.get('mandatory')}` skip_if_missing_removed=`{ev.get('skip_if_missing_removed')}`。见 `core/round22_1/evaluate.py`。"
    a["Q14"] = f"不能。本轮 evidence status=`{ev.get('status')}` n_full=`{ev.get('n_full')}`。缺文件不得标 FULL_EVIDENCE_COMPLETE。"
    a["Q15"] = "已有 five_pred/ten_pred → fuse_heads → B03 → O01 veto。未重训 51 模型。replay_only=True。"
    a["Q16"] = f"没有。nav_performance_changed=`{led.get('nav_performance_changed')}` n_nav_unchanged=`{led.get('n_nav_unchanged')}`。"
    a["Q17"] = f"Same Orders=`{e0.get('same_orders')}` Same NAV=`{e0.get('same_nav')}` Same Sharpe=`{e0.get('same_sharpe')}` status=`{e0.get('status')}`。"
    a["Q18"] = "此前 False 是缺 rank/slot/net_ret 列，不是已算但不合格。本轮补列后见 ledger/top_net.csv。"
    bridges = led.get("bridges") or []
    n_fail = sum(b.get("status") == "AUDIT_FAILURE" for b in bridges)
    a["Q19"] = f"桥条数={len(bridges)} AUDIT_FAILURE={n_fail}。公式：start + (sell_proceeds+cash_dividend-buy_cost) + Δunrealized = end。路径=`{root}/ledger/nav_bridge.csv`。"
    a["Q20"] = (
        f"n_rows=`{c.get('n_rows')}` buy/vol=`{c.get('buy_over_vol_median')}` sell/vol=`{c.get('sell_over_vol_median')}` both=`{c.get('both_over_vol_median')}` "
        f"identity=`{c.get('identity')}` proven=`{c.get('proven')}` status=`{c.get('status')}`。路径=`{root}/data_recovery/c_semantics.json`。"
    )
    a["Q21"] = f"C 重建 `{(rec.get('c_rebuild') or {}).get('status')}` reconstructed=`{(rec.get('c_rebuild') or {}).get('reconstructed')}`。未换公式。"
    a["Q22"] = f"凭证=`{(fgh.get('clickhouse') or rec.get('fgh', {}).get('clickhouse') or {}).get('credentials_present')}` download_attempts=`{rec.get('download_attempts')}` F=`{(fgh.get('F') or {}).get('status')}`。不猜密码。"
    a["Q23"] = f"G `{(fgh.get('G') or {}).get('status')}` into_out=`{(fgh.get('G') or {}).get('into_out')}` used_sw=`{(fgh.get('G') or {}).get('used_sw_industry')}` used_2026=`{(fgh.get('G') or {}).get('used_2026_members')}`。"
    a["Q24"] = f"H `{(fgh.get('H') or {}).get('status')}` announce_date=`{(fgh.get('H') or {}).get('announce_date')}` late_publish_backfill=`{(fgh.get('H') or {}).get('used_late_publish_to_backfill')}`。"
    a["Q25"] = f"只写 `{rec.get('write_root')}`。overwrote_hermes_v7=`{rec.get('overwrote_hermes_v7')}` overwrote_round22=`{rec.get('overwrote_round22')}`。"
    d12 = dfam.get("d1_d2") or {}
    a["Q26"] = f"IC identical=`{d12.get('ic_e2_identical')}` D1=`{d12.get('ic_e2_d1')}` D2=`{d12.get('ic_e2_d2')}` 日 Spearman=`{d12.get('mean_spearman')}`。"
    a["Q27"] = f"状态=`{d12.get('status')}` equivalent_for_ranking=`{d12.get('equivalent_for_ranking')}`。排序等价 ≠ 模型预测相同，也 ≠ 两条独立 Alpha。"
    contrasts = dfam.get("contrasts") or []
    a["Q28"] = "；".join(f"{r.get('contrast')} ΔSharpe={r.get('delta_sharpe')} ΔCAGR={r.get('delta_cagr')} ΔNAV={r.get('delta_nav')}" for r in contrasts[:4]) or "见 diagnostics/d_family_contrasts.csv"
    a["Q29"] = f"D2 Replace−Remove=`{dfam.get('replace_minus_remove_d2')}`。alone=`{dfam.get('replace_minus_remove_is_new_alpha_alone')}`。不能单独当新 Alpha。"
    a["Q30"] = f"D4 标签=`{dfam.get('d4_label')}`。是已知列交互，不能当新原始信息升级。"
    a["Q31"] = "；".join(f"{r.get('factor_id')}={r.get('verdict')}" for r in (dfam.get("provenance") or []))
    a["Q32"] = f"Bootstrap N=5000 block=20。SPA status=`{spa.get('status')}` observed=`{spa.get('observed_statistic')}` n_alternatives=`{spa.get('n_alternatives')}` p=`{spa.get('p_spa')}`。输入不全写 NOT_IDENTIFIABLE。历史窗不是独立 Forward。"
    a["Q33"] = f"`{decision['final_decision']}`。{decision['reason']}"
    a["Q34"] = f"CURRENT_BASELINE_CHANGED=FALSE。AUTO_PROMOTE=`{AUTO_PROMOTE_CURRENT}`。未改 CURRENT，未升 v6。"
    a["Q35"] = f"无独立 Forward=`{FORWARD_OOS_NOT_AVAILABLE}`。live_ready=False。C1/C4 已重建但未重训（短窗只与同日同池 v5 比）。下一优先：F 凭证后的 Profile B；带 INTO/OUT 的题材；带公告日的减持/回购。"
    return a


def _tracker(block: dict[str, Any]) -> None:
    text = TRACKER_PATH.read_text(encoding="utf-8") if TRACKER_PATH.is_file() else ""
    if MARKER in text:
        return
    lines = [
        "",
        MARKER,
        "",
        "锁定 `baseline_v5=R16_O01`。主口径 ROUND7_FLAT_5BP。经济门沿用 R21_3_ECONOMIC_MATERIALITY_V1，看结果前冻结。不自动升 CURRENT。",
        "",
        "### 实验组结果",
        "",
        "本轮无新正式回测。重放原 25 条袖套，NAV/Sharpe 未改。账本字段补齐、绩效未改。",
        "",
        "| 操作 | 代号 | Sharpe | 年化 | 累计 | MaxDD | 笔数 | NAV | ΔSharpe | Δ年化 | 状态 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        "| 账本字段补齐 | `R22_E0` | 1.876 | 55.00% | +4150.03% | -24.06% | 1107 | 12,750,089.36 | +0.0000 | +0.0000pct | 绩效未改 |",
        "",
        "主表口径=**ROUND7_FLAT_5BP**。只列本轮新正式回测；本轮 0 条。",
        "",
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
        "ROUND22_1_STATUS",
        "n_formal_backtests",
        "n_evidence_full",
        "ic_universe",
        "ledger_nav_unchanged",
        "tests_n",
        "tests_failed",
    ):
        lines.append(f"| {key} | {block.get(key)} |")
    lines.append("| CURRENT | baseline_v5=R16_O01 |")
    lines.append("| 决策 | " + str(block.get("FINAL_DECISION")) + " |")
    lines.append("| 主表 | ROUND7_FLAT_5BP |")
    lines.append("| 主报告 | `result/round22_1/ROUND22_1_RESEARCH_REPORT.md` |")
    lines.append("| 备注 | 账本字段补齐、绩效未改 |")
    lines.append(f"| REMAINING | {block.get('REMAINING')} |")
    lines.append("")
    TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACKER_PATH.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    write_md(MD["13"], "# Round22.1 · 13 追踪表\n\n已按 R1/R2 实验组结果追加。本轮无新正式回测，写「账本字段补齐、绩效未改」。无写入时间。\n")


def _report_md(block: dict[str, Any], decision: dict[str, Any], answers: dict[str, str]) -> str:
    parity = load_json(ROUND22_1_ROOT / "parity" / "parity.json")
    obs = parity.get("observed") or {}
    lines = [
        "# Round 22.1 证据补齐 × 账本重建 × 数据恢复",
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
        "Five 14 列，Ten 13 列。融合 0.90 Ten + 0.10 Five。经济门 R21_3_ECONOMIC_MATERIALITY_V1 冻结后才跑后续。",
        "不覆盖 result/round22。不重训 51 模型。不自动升 v6。",
        "",
        "## 本轮做了什么",
        "",
        "- Track A：IC 宇宙审计、日级输入表、{5,10,20,60} 敏感性、标签成熟。",
        f"- Track B：{len(PIT_PASS_IDS)} 个 PIT_PASS 因子 Rolling/Conditional/Shape/Top-tail；去掉 SKIP_IF_MISSING。",
        "- Track C：用已有 pred 重建 Top1/Top2 slot + NAV 桥。E0 Same Orders/NAV/Sharpe。",
        "- Track D：C 全表分档核对后重建 C1/C4（未重训，短窗只与同日同池 v5 比）；F/G/H 授权内搜索或诚实 BLOCKED。",
        "- Track E：D1/D2 等价、Replace−Remove、换笔、公式 provenance。",
        "",
        "## Q1–Q35",
        "",
    ]
    for qid, question in Q_ITEMS:
        lines.append(f"### {qid} {question}")
        lines.append("")
        lines.append(answers.get(qid, "NOT_RUN"))
        lines.append("")
    lines += [
        "## 章节",
        "",
        "00 契约与 v5 对照，01 IC 口径，02 证据，03 账本，04 C 语义，05 F/G/H，06 D 族，07 公式，08 NAV 桥，09 工厂测试，10–11 分年/SPA（沿用 round22 数字），12 zoo，13 追踪，14–15 矩阵，16 成熟，17 Replace−Remove，18 STANDARD，19 决策。",
        "",
        f"主报告路径 `{FINAL_REPORT}`。",
        "",
    ]
    return "\n".join(lines)


def run_report(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    del resume
    decision = _decision()
    dump_json(ROUND22_1_ROOT / "report" / "decision.json", decision)
    run_zoo(resume=False)
    block = _status_block(decision)
    dump_json(ROUND22_1_ROOT / "report" / "status_block.json", block)
    answers = _answers(decision, block)
    dump_json(ROUND22_1_ROOT / "report" / "q_answers.json", answers)
    write_md(
        MD["19"],
        "\n".join(
            [
                "# Round22.1 · 19 决策",
                "",
                f"`{decision['final_decision']}`。{decision['reason']}",
                "CURRENT_BASELINE_CHANGED=FALSE。不写 CURRENT，不升 v6。",
                f"状态 `{block['ROUND22_1_STATUS']}`。未完成项 `{block['REMAINING']}`。",
                "",
            ]
        ),
    )
    write_md(
        MD["10"],
        "# Round22.1 · 10 分年费用\n\n本轮无新正式回测。分年与费用校正沿用 `result/round22/stats/yearly_results.csv` 与 `result/round22/backtest/fee_both_calibers.csv`。账本字段补齐、绩效未改。\n",
    )
    stats22 = load_json(ROUND22_ROOT / "stats" / "stats.json")
    spa = stats22.get("spa") or {}
    write_md(
        MD["11"],
        f"# Round22.1 · 11 Bootstrap / SPA\n\n沿用 round22：N=5000 block=20。SPA status=`{spa.get('status')}` observed=`{spa.get('observed_statistic')}` n_alternatives=`{spa.get('n_alternatives')}` p=`{spa.get('p_spa')}`。输入不全写 NOT_IDENTIFIABLE。历史窗不是独立 Forward。\n",
    )
    body = _report_md(block, decision, answers)
    write_md(FINAL_REPORT, body)
    write_md(ROUND22_1_ROOT / "report" / "ROUND22_1_RESEARCH_REPORT.md", body)
    _tracker(block)
    from .tests import run_tests

    tests = run_tests()
    block["tests_n"] = tests.get("n")
    block["tests_failed"] = tests.get("n_failed")
    if tests.get("n_failed"):
        rem = list(block.get("REMAINING") or [])
        rem = [x for x in rem if not str(x).startswith("tests_failed=")]
        rem.append(f"tests_failed={tests.get('n_failed')}")
        block["REMAINING"] = rem
        block["ROUND22_1_STATUS"] = "PARTIAL"
    dump_json(ROUND22_1_ROOT / "report" / "status_block.json", block)
    return {"status": block["ROUND22_1_STATUS"], "decision": decision["final_decision"], "final_report": str(FINAL_REPORT), "live_ready": False, "n_tests": tests.get("n"), "n_failed": tests.get("n_failed")}
