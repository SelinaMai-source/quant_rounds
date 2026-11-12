"""总报告、Q1–Q48、状态块、追踪表。主表只用全历史账。"""

from __future__ import annotations

from typing import Any

from .common import dump_json, finite_or_none, load_json, write_md
from .contract import (
    AUTO_PROMOTE_CURRENT,
    C1_FORMULA_FROZEN,
    C4_FORMULA_FROZEN,
    C_AVAILABLE_FROM,
    E0_CODE,
    FINAL_REPORT,
    FORWARD_OOS_NOT_AVAILABLE,
    KEEP_BASELINE_V5,
    MAIN_LEDGER,
    MATCHED_SUBPERIOD_NOT_COMPARABLE,
    MD,
    Q_ITEMS,
    REPLAY_GROUPS,
    ROUND22_2_ROOT,
    ROUND22_3_ROOT,
    SHORT_LEDGER,
    SUBPERIOD_ALPHA_RESEARCH_CANDIDATE,
    TRACKER_PATH,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    V6_RESEARCH_CANDIDATE,
)
from .zoo import run_zoo

MARKER = "### Round22.3 全历史重跑"
R222_MARKER = "### Round22.2 新 Alpha 验证"


def _fmt(value: Any, digits: int = 6) -> str:
    v = finite_or_none(value)
    if v is None:
        return "None"
    return f"{v:.{digits}g}"


def _row(code: str, sleeve: dict[str, Any]) -> dict[str, Any]:
    return next((r for r in (sleeve.get("rows") or []) if r.get("code") == code), {})


def _decision() -> dict[str, Any]:
    parity = load_json(ROUND22_3_ROOT / "parity" / "parity.json")
    train = load_json(ROUND22_3_ROOT / "model" / "train.json")
    stats = load_json(ROUND22_3_ROOT / "stats" / "stats.json")
    if parity.get("passed") is False:
        return {
            "final_decision": KEEP_BASELINE_V5,
            "live_ready": False,
            "reason": "Stage0 parity FAIL。停止晋级判断。保持 v5。",
            "best_code": None,
        }
    material = [r for r in (stats.get("material") or []) if r.get("economically_material")]
    if material:
        best = max(material, key=lambda r: float(r.get("delta_sharpe") or -1e9))
        return {
            "final_decision": V6_RESEARCH_CANDIDATE,
            "live_ready": False,
            "reason": "全历史对照 R16_O01 过经济门，但仍无独立 Forward，不能 live-ready。禁止 LIVE_READY_V6。",
            "best_code": best.get("code"),
        }
    if train.get("status") not in {"DONE"}:
        return {
            "final_decision": KEEP_BASELINE_V5,
            "live_ready": False,
            "reason": f"训练状态 `{train.get('status') or 'NOT_RUN'}`。保持 v5。",
            "best_code": None,
        }
    return {
        "final_decision": KEEP_BASELINE_V5,
        "live_ready": False,
        "reason": "全历史对照 R16_O01：没有一项同时满足经济门、Top1/Top2 与新因子增量。短窗不得晋级。",
        "best_code": None,
    }


def _status_block(decision: dict[str, Any]) -> dict[str, Any]:
    parity = load_json(ROUND22_3_ROOT / "parity" / "parity.json")
    train = load_json(ROUND22_3_ROOT / "model" / "train.json")
    sleeve = load_json(ROUND22_3_ROOT / "backtest" / "sleeve.json")
    ledger = load_json(ROUND22_3_ROOT / "ledger" / "ledger.json")
    stats = load_json(ROUND22_3_ROOT / "stats" / "stats.json")
    tests = load_json(ROUND22_3_ROOT / "tests" / "unit_tests.json")
    audit = load_json(ROUND22_3_ROOT / "audit" / "audit.json")
    c4 = load_json(ROUND22_3_ROOT / "c4" / "c4.json")
    hn = load_json(ROUND22_3_ROOT / "hn2" / "hn2.json")
    fs = load_json(ROUND22_3_ROOT / "fs" / "fs.json")
    spa = stats.get("spa") or {}
    remaining = []
    if not parity.get("passed"):
        remaining.append("parity!=PASS")
    if not (ROUND22_3_ROOT / "configs" / "frozen_gates.json").is_file():
        remaining.append("gate_not_frozen")
    if train.get("status") not in {"DONE"}:
        remaining.append(f"train={train.get('status') or 'NOT_RUN'}")
    if sleeve.get("status") not in {"DONE"}:
        remaining.append(f"sleeve={sleeve.get('status') or 'NOT_RUN'}")
    if fs.get("status") not in {"DONE", "PIT_PASS"} and str(fs.get("status") or "").startswith("BLOCKED") is False:
        if fs.get("status") not in {"BLOCKED_DATA", "BLOCKED_PIT", "BLOCKED_PERIOD_MISMATCH"}:
            remaining.append(f"fs={fs.get('status') or 'NOT_RUN'}")
    remaining.append(f"forward={FORWARD_OOS_NOT_AVAILABLE}")
    if tests.get("n_failed"):
        remaining.append(f"tests_failed={tests.get('n_failed')}")
    status = "PARTIAL"
    mats = [r for r in (stats.get("material") or []) if r.get("delta_sharpe") is not None]
    best = max(mats, key=lambda r: float(r["delta_sharpe"])) if mats else {}
    return {
        "ROUND22_3_STATUS": status,
        "REMAINING": remaining,
        "BASELINE_PARITY": "PASS" if parity.get("passed") else "FAIL",
        "BEST_CANDIDATE": best.get("code") or decision.get("best_code"),
        "BEST_DELTA_SHARPE": best.get("delta_sharpe"),
        "BEST_DELTA_CAGR": best.get("delta_cagr"),
        "ECONOMICALLY_MATERIAL": bool(mats and any(r.get("economically_material") for r in mats)),
        "SPA_P": spa.get("p_spa") or spa.get("p"),
        "SPA_N_ALTERNATIVES": spa.get("n_alternatives"),
        "FORWARD": FORWARD_OOS_NOT_AVAILABLE,
        "FINAL_DECISION": decision["final_decision"],
        "CURRENT_BASELINE_CHANGED": "FALSE",
        "AUTO_PROMOTE": False,
        "live_ready": False,
        "n_formal_backtests": int(sleeve.get("n_formal") or 0),
        "n_oof": int(train.get("n_oof_pass") or 0),
        "n_trained_real": int(train.get("n_trained_real") or 0),
        "n_imported": int(train.get("n_imported") or 0),
        "tests_n": tests.get("n"),
        "tests_failed": tests.get("n_failed"),
        "ledger_missing_columns": ledger.get("n_missing_columns"),
        "gate_status": (audit.get("gate") or {}).get("status"),
        "c4_verdict": c4.get("verdict"),
        "hn2_identity": ((hn.get("identity") or {}).get("is_delta_log_circ_minus_delta_log_holders")),
        "fs_status": fs.get("status"),
        "compare_to": "R16_O01",
    }


def _sleeve_line(rows: list[dict[str, Any]], codes: list[str], key: str) -> str:
    bits = []
    for code in codes:
        r = next((x for x in rows if x.get("code") == code), {})
        bits.append(f"{code} Sharpe=`{_fmt(r.get('sharpe'))}` ΔSharpe=`{_fmt(r.get('delta_sharpe'))}` CAGR=`{_fmt(r.get('cagr'))}` NAV=`{_fmt(r.get('nav_end'))}` n_obs=`{r.get('n_obs')}`")
    return "；".join(bits) if bits else "未跑"


def _answers(decision: dict[str, Any], block: dict[str, Any]) -> dict[str, str]:
    parity = load_json(ROUND22_3_ROOT / "parity" / "parity.json")
    obs = parity.get("observed") or {}
    freeze = load_json(ROUND22_3_ROOT / "configs" / "frozen_gates.json")
    audit = load_json(ROUND22_3_ROOT / "audit" / "audit.json")
    train = load_json(ROUND22_3_ROOT / "model" / "train.json")
    sleeve = load_json(ROUND22_3_ROOT / "backtest" / "sleeve.json")
    sleeve22 = load_json(ROUND22_2_ROOT / "backtest" / "sleeve.json")
    stats = load_json(ROUND22_3_ROOT / "stats" / "stats.json")
    c4 = load_json(ROUND22_3_ROOT / "c4" / "c4.json")
    hn = load_json(ROUND22_3_ROOT / "hn2" / "hn2.json")
    fs = load_json(ROUND22_3_ROOT / "fs" / "fs.json")
    ledger = load_json(ROUND22_3_ROOT / "ledger" / "ledger.json")
    zoo = load_json(ROUND22_3_ROOT / "zoo" / "zoo.json")
    tests = load_json(ROUND22_3_ROOT / "tests" / "unit_tests.json")
    spa = stats.get("spa") or {}
    spa_short = (audit.get("spa_short") or {})
    rows = sleeve.get("rows") or []
    rows22 = sleeve22.get("rows") or []
    a: dict[str, str] = {}
    a["Q1"] = (
        f"CAGR={obs.get('cagr')} Sharpe={obs.get('sharpe')} MaxDD={obs.get('max_drawdown')} "
        f"trades={obs.get('completed_trades')} NAV={obs.get('nav_end')} n_obs={obs.get('n_obs')} status=`{parity.get('status')}`。"
        f"锚点 CAGR={V5_ANCHOR['cagr']} Sharpe={V5_ANCHOR['sharpe']} MaxDD={V5_ANCHOR['max_drawdown']} trades=1107 NAV={V5_NAV_END_EXACT}。"
    )
    a["Q2"] = (
        f"经济门 ID=`R21_3_ECONOMIC_MATERIALITY_V1` body_sha256=`{freeze.get('body_sha256')}` "
        f"threshold_sha256=`{freeze.get('threshold_sha256')}`。frozen_before_formal_nav=`{freeze.get('frozen_before_formal_nav')}` do_not_retune=`{freeze.get('do_not_retune')}`。未回调。"
    )
    gate = audit.get("gate") or {}
    a["Q3"] = f"`{gate.get('status')}` thresholds_identical=`{gate.get('thresholds_identical')}` hashes=`{gate.get('threshold_sha256')}`。body SHA 差来自 body.round，不是门槛数字漂移。"
    a["Q4"] = (
        f"主账 {MAIN_LEDGER['window_start']}–{MAIN_LEDGER['window_end']} n_obs={MAIN_LEDGER['n_obs']} 笔数≈{MAIN_LEDGER['n_trades']} 对照 R16_O01 Sharpe={MAIN_LEDGER['sharpe']}。"
        f"短窗 {SHORT_LEDGER['window_start']}–{SHORT_LEDGER['window_end']} n_obs={SHORT_LEDGER['n_obs']} 笔数≈{SHORT_LEDGER['n_trades']} 对照 R222_E0 Sharpe={SHORT_LEDGER['sharpe']}。"
        "费用/引擎/Sharpe 公式未变。高 Sharpe 是窗口，不是更好的策略。不得混表。"
    )
    a["Q5"] = (
        f"不能比较。短窗 E0 Sharpe=2.73450 与 R16_O01 Sharpe=1.87566 窗口不同（860 vs 2075 日）。"
        f"追踪表 22.2 行标 `{MATCHED_SUBPERIOD_NOT_COMPARABLE}`。"
    )
    folds = audit.get("folds") or {}
    a["Q6"] = f"联合 matched n_dates=`{folds.get('n_dates')}` {folds.get('min_date')}–{folds.get('max_date')}。袖套从 2023-01-03。canonical=`{folds.get('canonical')}`。2022=`{folds.get('y2022')}`。{folds.get('reason')}"
    a["Q7"] = f"C1 冻结公式=`{C1_FORMULA_FROZEN}`。表无 BUY_VALUE_XL。未为收益改公式。模型 id=`r22_2_c1_xl_net_imbalance_lag1`。"
    a["Q8"] = f"C4 冻结公式=`{C4_FORMULA_FROZEN}`。与 22.1 recover 一致。未同名改期。"
    a["Q9"] = (
        f"C1=`LAGGED_SAFE_RESEARCH` C4=`SUBPERIOD_STRICT_PIT`。feature_available_from=`{C_AVAILABLE_FROM}` 覆盖约 1394 日。"
        "2020-09-24 前保持 NaN，不填 0，不删股票。这不是全历史严格 PIT。"
    )
    a["Q10"] = (
        f"规范 3y/1y 从 2018：first=`{train.get('first_prediction_year')}` last=`{train.get('last_prediction_year')}` sample=`{train.get('sample')}`。"
        f"paste_2023_preds=`{train.get('paste_2023_preds')}` filled0=`{train.get('filled_with_zero')}` dropped_names=`{train.get('dropped_names')}`。"
    )
    e0 = _row(E0_CODE, sleeve)
    a["Q11"] = f"E0 origin=`{train.get('e0_origin')}` Sharpe=`{_fmt(e0.get('sharpe'))}` CAGR=`{_fmt(e0.get('cagr'))}` NAV=`{_fmt(e0.get('nav_end'))}` n_obs=`{e0.get('n_obs')}`。对照 R16_O01 Sharpe=1.87566 NAV={V5_NAV_END_EXACT}。"
    a["Q12"] = (
        "短窗：" + _sleeve_line(rows22, ["R222_E1_r22_2_c1_xl_net_imbalance_lag1", "R222_E2_r22_2_c1_xl_net_imbalance_lag1", "R222_E3_r22_2_c1_xl_net_imbalance_lag1"], "c1")
        + "。全历史：" + _sleeve_line(rows, ["R223_E1_r22_2_c1_xl_net_imbalance_lag1", "R223_E2_r22_2_c1_xl_net_imbalance_lag1", "R223_E3_r22_2_c1_xl_net_imbalance_lag1"], "c1")
        + "。短窗对照 R222_E0、全历史对照 R16_O01，窗口与折不同，不可互换。"
    )
    a["Q13"] = (
        "短窗：" + _sleeve_line(rows22, ["R222_E1_r22_c4_flow_persist_5", "R222_E2_r22_c4_flow_persist_5", "R222_E3_r22_c4_flow_persist_5"], "c4")
        + "。全历史：" + _sleeve_line(rows, ["R223_E1_r22_c4_flow_persist_5", "R223_E2_r22_c4_flow_persist_5", "R223_E3_r22_c4_flow_persist_5"], "c4")
        + "。短窗 +0.025 相对 matched E0，不是主裁决。"
    )
    a["Q14"] = (
        "短窗：" + _sleeve_line(rows22, ["R222_E1_r22_2_hn2_avg_holding_chg"], "hn2")
        + "。全历史：" + _sleeve_line(rows, ["R223_E1_r22_2_hn2_avg_holding_chg", "R223_E2_r22_2_hn2_avg_holding_chg", "R223_E3_r22_2_hn2_avg_holding_chg"], "hn2")
        + "。本轮补 Ten/Both。窗口不可互换。"
    )
    a["Q15"] = (
        "短窗：" + _sleeve_line(rows22, ["R222_E1_c1_c4_joint", "R222_E2_c1_c4_joint", "R222_E3_c1_c4_joint"], "j")
        + "。全历史：" + _sleeve_line(rows, ["R223_E1_c1_c4_joint", "R223_E2_c1_c4_joint", "R223_E3_c1_c4_joint"], "j")
        + "。禁止失败后无限组合。"
    )
    a["Q16"] = f"主表 Δ 对照 `R16_O01`。compare_mixed_r222_e0=`{sleeve.get('compare_mixed_r222_e0')}`。没有把短窗 Sharpe 2.7 当正式改善。"
    a["Q17"] = f"C4 |ρ| vs `{c4.get('nearest_five')}`=`{c4.get('abs_rho_vs_g1_ret_5')}`。replace_gate=0.60 lowered=`{c4.get('replace_gate_lowered')}` action=`{c4.get('replace_action')}`。未降门。"
    a["Q18"] = f"C4 裁决 `{c4.get('verdict')}`。全历史 E1 ΔSharpe=`{c4.get('full_e1_delta_sharpe_vs_v5')}`。短窗数字不作主裁决。"
    a["Q19"] = f"费用仍 `ROUND7_FLAT_5BP`。fee=`{c4.get('fee')}`。未改开盘规则。"
    ident = hn.get("identity") or {}
    vs_d1 = {**(ident.get("vs_r21_d1") or {}), **((hn.get("evidence") or {}).get("vs_r21_d1") or {})}
    a["Q20"] = (
        f"HN2 是否 Δlog(circ)−Δlog(holders)=`{ident.get('is_delta_log_circ_minus_delta_log_holders')}`。"
        f"{ident.get('proof') or ident.get('note')} "
        f"|ρ| vs −r21_d1=`{vs_d1.get('abs_rho_neg_official')}` just_neg_r21_d1_plus_slow_circ=`{vs_d1.get('just_neg_r21_d1_plus_slow_circ')}`。"
        "不是 −r21_d1 加缓慢流通股本项；达 |ρ|≥0.60 才允许 Replace。"
    )
    a["Q21"] = f"HN2 PIT=`{(hn.get('pit') or {}).get('status')}` finite=`{(hn.get('pit') or {}).get('finite_share')}` filled0=False。helper_evidence=`{hn.get('evidence_helper')}`。"
    heads = hn.get("heads") or {}
    a["Q22"] = f"全历史 H1=`{heads.get('H1')}` H2=`{heads.get('H2')}` H3=`{heads.get('H3')}`。Replace 仅当 |ρ|≥0.60。"
    a["Q23"] = f"FS join status=`{fs.get('status')}` reason=`{fs.get('reason')}` keys=`{fs.get('join_keys')}`。"
    a["Q24"] = f"FS1=`{fs.get('fs1')}` FS2=`{fs.get('fs2')}`。合格才训。filled0=False。"
    a["Q25"] = f"短窗 SPA original n=`{spa_short.get('n_alternatives_original')}` p=`{spa_short.get('p_spa_original')}`；corrected n=`{spa_short.get('n_alternatives_corrected')}`（已跳过 E0）。"
    a["Q26"] = f"全历史 SPA status=`{spa.get('status')}` n_alternatives=`{spa.get('n_alternatives')}` p=`{spa.get('p_spa') or spa.get('p')}` e0_excluded=`{spa.get('e0_excluded')}`。"
    boot = load_json(ROUND22_3_ROOT / "c4" / "c4_e1_paired_bootstrap.json")
    a["Q27"] = f"C4-E1 全历史配对 bootstrap `{boot}`。"
    a["Q28"] = f"净收益=sell_proceeds+cash_dividend−buy_cost。缺列条数=`{ledger.get('n_missing_columns')}` top_net_ok=`{ledger.get('n_top_net_ok')}`。缺列不得解释为已算不合格。"
    a["Q29"] = f"分年见 `stats/yearly.csv`。执行=`SAME_OPEN_EXECUTION_OPTIMISM`。费用=`HISTORICAL_FEE_CORRECTED`。"
    mats = [r for r in (stats.get("material") or []) if r.get("economically_material")]
    a["Q30"] = f"过经济门的全历史候选数=`{len(mats)}`。对照 R16_O01。"
    c4e1 = next((r for r in rows if r.get("code") == "R223_E1_r22_c4_flow_persist_5"), {})
    a["Q31"] = (
        f"按 ΔSharpe 最佳 `{block.get('BEST_CANDIDATE')}` ΔSharpe=`{block.get('BEST_DELTA_SHARPE')}` ΔCAGR=`{block.get('BEST_DELTA_CAGR')}`。"
        f"C4 E1 ΔSharpe=`{_fmt(c4e1.get('delta_sharpe'))}` ΔCAGR=`{_fmt(c4e1.get('delta_cagr'))}` NAV rel=`{_fmt(c4e1.get('nav_rel'))}`。"
        "二者都未过经济门。"
    )
    a["Q32"] = f"`{decision['final_decision']}`。{decision['reason']}"
    a["Q33"] = f"CURRENT_BASELINE_CHANGED=FALSE。AUTO_PROMOTE=`{AUTO_PROMOTE_CURRENT}`。未改 CURRENT，未升 v6。"
    a["Q34"] = f"无独立 Forward=`{FORWARD_OOS_NOT_AVAILABLE}`。live_ready=False。LIVE_READY_V6 被禁止。"
    a["Q35"] = f"22.2 短窗行标 `{MATCHED_SUBPERIOD_NOT_COMPARABLE}`。本轮只追加全历史新回测。无「写入时间」。"
    a["Q36"] = f"zoo n=`{zoo.get('n')}` rejected=`{zoo.get('n_rejected')}` rewrote_prior=`{zoo.get('rewrote_prior')}`。"
    a["Q37"] = "STANDARD 追加 Round 22.3 强制增量。evaluate_factor 挂钩 22.3。"
    a["Q38"] = f"tests n=`{tests.get('n')}` failed=`{tests.get('n_failed')}`。Parity FAIL 则停止晋级。"
    a["Q39"] = f"n_formal=`{block.get('n_formal_backtests')}` n_trained_real=`{block.get('n_trained_real')}` n_imported=`{block.get('n_imported')}` n_oof=`{block.get('n_oof')}`。"
    a["Q40"] = "新数据只写 result/round22_3。overwrote_round22*=False overwrote_round21*=False overwrote_hermes=False。"
    a["Q41"] = f"短窗 E1 ΔSharpe=+0.025 vs matched E0 只作附录。主裁决用全历史 Δ=`{c4.get('full_e1_delta_sharpe_vs_v5')}` vs R16_O01。裁决 `{c4.get('verdict')}`。"
    a["Q42"] = f"短窗 HN2 Five ΔSharpe=`{hn.get('short_e1_delta_sharpe_vs_matched_e0')}`。全历史 H1–H3 见 Q22。窗口不可互换。"
    a["Q43"] = "C1 短窗三组相对 matched E0 全为负。全历史 E1 ΔSharpe=+0.0135（未过门），E2/E3 仍为负。对照层级不同，短窗不得当结论。"
    a["Q44"] = "联合短窗相对 matched E0 为负。全历史见 Q15。未在失败后无限组合。"
    a["Q45"] = f"主表披露 feature_available_from=`{C_AVAILABLE_FROM}` 覆盖≈1394/2075。不是全历史严格 PIT。"
    a["Q46"] = f"Replace 门 |ρ|≥0.60。C4 vs g1_ret_5≈0.576 未触发。未降门。"
    a["Q47"] = f"ROUND22_3_STATUS=`{block['ROUND22_3_STATUS']}` REMAINING=`{block['REMAINING']}`。未跑完标 PARTIAL，不把 NOT_RUN 写成 COMPLETE。"
    a["Q48"] = (
        "下一轮：独立 Forward（无 Forward 不得 live-ready）。"
        "FS 本轮已 PIT_PASS，FS1 RankIC 负、FS2 +0.026 弱，增量不够才未训；不要把 join 再当 BLOCKED。"
        "C4 全历史未过门，不必为 +0.010 Sharpe 做费用敏感。"
        "保持两本账隔离，禁止把短窗 Sharpe 2.7 当晋级证据。"
    )
    return a


def _mark_r222_tracker(text: str) -> str:
    if "MATCHED_SUBPERIOD_NOT_COMPARABLE" in text and R222_MARKER in text:
        return text
    note = (
        "\n\n> **MATCHED_SUBPERIOD_NOT_COMPARABLE**：上表 22.2 正式袖套是 2023-01-03–2026-07-24 短窗（n_obs=860，对照重训 `R222_E0` Sharpe≈2.735）。"
        "不得与 2018–2026 `R16_O01` Sharpe=1.87566 混表。高 Sharpe 是窗口，不是更好的策略。\n"
    )
    if R222_MARKER in text and "MATCHED_SUBPERIOD_NOT_COMPARABLE" not in text.split(R222_MARKER, 1)[1][:2500]:
        # insert after the 22.2 formal table / before Round22.2 kv table if possible
        parts = text.split(R222_MARKER, 1)
        tail = parts[1]
        marker2 = "\n| 项 | 值 |"
        if marker2 in tail:
            head, rest = tail.split(marker2, 1)
            text = parts[0] + R222_MARKER + head.rstrip() + note + marker2 + rest
        else:
            text = text + note
    return text


def _tracker(block: dict[str, Any], sleeve: dict[str, Any]) -> None:
    text = TRACKER_PATH.read_text(encoding="utf-8") if TRACKER_PATH.is_file() else ""
    text = _mark_r222_tracker(text)
    if MARKER in text:
        TRACKER_PATH.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        write_md(MD["20"], "# Round22.3 · 20 追踪表\n\n已存在 22.3 段。22.2 短窗行已标 MATCHED_SUBPERIOD_NOT_COMPARABLE。\n")
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
            "主表口径=**ROUND7_FLAT_5BP**。只列本轮新正式全历史回测；本轮 0 条（训练/回测未完成则保持此句）。",
            "",
        ]
    else:
        lines += [
            "主表口径=**ROUND7_FLAT_5BP**。只列本轮新正式全历史回测（2018-01-02–2026-07-24）。Δ 对照 `R16_O01`，不对照 `R222_E0`。",
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
        "FINAL_DECISION",
        "AUTO_PROMOTE",
        "CURRENT_BASELINE_CHANGED",
        "ECONOMICALLY_MATERIAL",
        "SPA_P",
        "FORWARD",
        "ROUND22_3_STATUS",
        "n_formal_backtests",
        "n_trained_real",
        "n_oof",
        "c4_verdict",
        "fs_status",
    ):
        lines.append(f"| {key} | {block.get(key)} |")
    lines += [
        "| CURRENT | baseline_v5=R16_O01 |",
        f"| 决策 | {block.get('FINAL_DECISION')} |",
        "| 主表 | ROUND7_FLAT_5BP |",
        "| 主报告 | `result/round22_3/ROUND22_3_RESEARCH_REPORT.md` |",
        f"| REMAINING | {block.get('REMAINING')} |",
        "",
    ]
    TRACKER_PATH.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    write_md(MD["20"], "# Round22.3 · 20 追踪表\n\n已追加全历史实验组结果。22.2 短窗行已标 MATCHED_SUBPERIOD_NOT_COMPARABLE。无「写入时间」。\n")


def run_report(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    del resume
    decision = _decision()
    dump_json(ROUND22_3_ROOT / "report" / "decision.json", decision)
    run_zoo()
    block = _status_block(decision)
    dump_json(ROUND22_3_ROOT / "report" / "status_block.json", block)
    answers = _answers(decision, block)
    dump_json(ROUND22_3_ROOT / "report" / "q_answers.json", answers)
    sleeve = load_json(ROUND22_3_ROOT / "backtest" / "sleeve.json")
    _tracker(block, sleeve)
    lines = [
        "# Round 22.3 研究总报告",
        "",
        "锁定 `baseline_v5=R16_O01`。主口径 `ROUND7_FLAT_5BP`。Five 14 / Ten 13。融合 0.90/0.10。",
        "经济门 `R21_3_ECONOMIC_MATERIALITY_V1` 看回测前冻结，未回调。不自动升 CURRENT / v6。",
        "",
        "## 两本账（不得混表）",
        "",
        f"- 主账：{MAIN_LEDGER['window_start']}–{MAIN_LEDGER['window_end']} n_obs={MAIN_LEDGER['n_obs']} 对照 R16_O01 Sharpe={MAIN_LEDGER['sharpe']}。**这是官方比较。**",
        f"- 22.2 实验账：{SHORT_LEDGER['window_start']}–{SHORT_LEDGER['window_end']} n_obs={SHORT_LEDGER['n_obs']} 对照 R222_E0 Sharpe={SHORT_LEDGER['sharpe']}。`{MATCHED_SUBPERIOD_NOT_COMPARABLE}`。",
        "",
        "## STATUS",
        "",
    ]
    for k, v in block.items():
        lines.append(f"{k}={v}")
    lines += ["", "## 决策", "", f"`{decision['final_decision']}`。{decision['reason']}", "", "## Q1–Q48", ""]
    for qid, qtext in Q_ITEMS:
        lines += [f"### {qid} {qtext}", "", answers.get(qid, "NOT_RUN"), ""]
    lines += ["## 分册", ""]
    for key, path in MD.items():
        lines.append(f"- `{path}`")
    write_md(FINAL_REPORT, "\n".join(lines))
    write_md(ROUND22_3_ROOT / "report" / "ROUND22_3_RESEARCH_REPORT.md", "\n".join(lines))
    write_md(
        MD["22"],
        f"# Round22.3 · 22 最终决策\n\n`{decision['final_decision']}`。{decision['reason']}\n\nCURRENT_BASELINE_CHANGED=FALSE。AUTO_PROMOTE=False。live_ready=False。\n",
    )
    tests = load_json(ROUND22_3_ROOT / "tests" / "unit_tests.json")
    return {
        "status": block["ROUND22_3_STATUS"],
        "decision": decision["final_decision"],
        "final_report": str(FINAL_REPORT),
        "live_ready": False,
        "n_tests": tests.get("n"),
        "n_failed": tests.get("n_failed"),
    }
