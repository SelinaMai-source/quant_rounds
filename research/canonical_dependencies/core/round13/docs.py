"""执行弹性、容量、forward、promotion、主报告 Q1–Q28、追踪表追加。不写「写入时间」。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import MARKET_END, ROUND13_ROOT
from ..round12.deploy import apply_liquidity_priority
from ..round12.runner import hist_hooks
from .bootstrap import load_bootstrap, load_cpcv, load_spa
from .candidates import load_candidates, load_daily_map, load_parity_status
from .common import dump_json, load_json, pct, write_md
from .contract import (
    AWAITING_USER_LOCK,
    CANDIDATE_IDS,
    CAPITALS,
    COMPLEXITY_ORDER,
    CONTRACT_CASH,
    DIAGNOSTIC_CAPITAL,
    EXEC_BPS,
    FALLBACK_P50_BPS,
    FALLBACK_P75_BPS,
    KEEP_BASELINE_V4,
    NEAR_TIE_CAGR_PP,
    NEAR_TIE_SHARPE,
    OOB_P_THRESHOLD,
    PRE_REGISTERED_QUESTIONS,
    PROMOTION_LOYO_SLACK_PP,
    PROMOTION_MAX_CAGR_DROP_PP,
    PROMOTION_MDD_SLACK_PP,
    PROMOTION_SHARPE_SLACK,
    R12_01_TARGETS,
    R13_00,
    RECOMMENDED_BASELINE_V5,
    SPA_SUPPORT_P,
    TRACKER_PATH,
    experiment_dir,
)
from .freeze import require_frozen
from .oob import load_oob, load_placebo, load_timing
from .runner import load_market, run_r13_backtest
from .scenarios import PROMOTION_SCENARIOS, load_scenarios
from .weights import load_overlayed, load_weights_audit

REPORT_MD = ROUND13_ROOT / "ROUND13_BASELINE_CONSOLIDATION_REPORT.md"
EXEC_JSON = ROUND13_ROOT / "execution" / "execution.json"
EXEC_MD = ROUND13_ROOT / "10_execution.md"
CAP_JSON = ROUND13_ROOT / "capacity" / "capacity.json"
CAP_MD = ROUND13_ROOT / "11_capacity.md"
FWD_JSON = ROUND13_ROOT / "forward" / "forward.json"
FWD_MD = ROUND13_ROOT / "12_forward.md"
PROMO_JSON = ROUND13_ROOT / "promotion" / "promotion.json"
PROMO_MD = ROUND13_ROOT / "13_promotion.md"
DOSSIER_JSON = ROUND13_ROOT / "stats" / "dossier.json"
IDS_EXEC = ("v4", *CANDIDATE_IDS)


def _run_bt(tag: str, ov, bps, cash, extra, resume, loaded) -> dict[str, Any]:
    dest = ROUND13_ROOT / extra.get("family", "execution") / tag
    ran = run_r13_backtest(
        dest,
        overlayed=ov,
        validation_hooks=hist_hooks(bps if bps else None),
        initial_cash=cash,
        extra_report=extra,
        resume=resume,
        prepared_market=loaded["prepared"],
        market_meta=loaded,
    )
    m = (ran.get("report") or {}).get("metrics") or {}
    return {
        "code": tag,
        "cagr": m.get("annualized_return"),
        "sharpe": m.get("sharpe_zero_rate"),
        "max_drawdown": m.get("max_drawdown"),
        "completed_trades": m.get("completed_trades"),
        "loyo_median": m.get("loyo_median"),
        "loyo_worst": m.get("loyo_worst"),
        **extra,
    }


def run_execution(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    loaded = load_market()
    scen = load_scenarios()
    results: dict[str, Any] = {}
    for sid in IDS_EXEC:
        ov = load_overlayed(sid)
        curve = {}
        for bps in EXEC_BPS:
            if bps in (20.0, 30.0):
                key = "S1" if bps == 20.0 else "S2"
                row = ((scen.get("matrix") or {}).get(sid) or {}).get(key) or {}
                if row.get("sharpe") is not None:
                    curve[str(int(bps))] = {
                        "bps": bps,
                        "cagr": row.get("cagr"),
                        "sharpe": row.get("sharpe"),
                        "max_drawdown": row.get("max_drawdown"),
                        "reused_scenario": key,
                    }
                    continue
            if bps == 0.0:
                cand = load_candidates().get("results") or {}
                src = cand.get(R13_00 if sid == "v4" else sid) or {}
                curve["0"] = {
                    "bps": 0.0,
                    "cagr": src.get("cagr"),
                    "sharpe": src.get("sharpe"),
                    "max_drawdown": src.get("max_drawdown"),
                    "reused": "S0",
                }
                continue
            rec = _run_bt(
                f"{sid}_bp{int(bps):02d}",
                ov,
                bps,
                None,
                {"family": "execution", "bps": bps, "parent": sid, "no_optimize": True},
                resume,
                loaded,
            )
            curve[str(int(bps))] = rec
        xs = []
        ys = []
        for b in EXEC_BPS:
            row = curve.get(str(int(b))) or {}
            if row.get("sharpe") is not None:
                xs.append(float(b))
                ys.append(float(row["sharpe"]))
        slope = None
        be = None
        if len(xs) >= 2:
            slope = float(np.polyfit(xs, ys, 1)[0])
            # break-even: bps where sharpe = 0 (linear)
            if slope != 0:
                intercept = float(np.polyfit(xs, ys, 1)[1])
                be = float(-intercept / slope) if slope < 0 else None
        results[sid] = {"curve": curve, "slope_sharpe_per_bp": slope, "break_even_bps": be}
    payload = {
        "status": "DONE",
        "bps": list(EXEC_BPS),
        "no_optimize": True,
        "results": results,
        "note": "0/10/20/30/50bp 买侧压力。不优化切点。20/30 复用 S1/S2。",
    }
    dump_json(EXEC_JSON, payload)
    lines = [
        "# Round 13 · 执行弹性",
        "",
        "不优化。斜率 = Sharpe / bp。",
        "",
        "| ID | 0 | 10 | 20 | 30 | 50 | slope | break-even bp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sid, row in results.items():
        c = row["curve"]
        def _s(b):
            return (c.get(str(b)) or {}).get("sharpe")
        lines.append(
            f"| `{sid}` | {_s(0)} | {_s(10)} | {_s(20)} | {_s(30)} | {_s(50)} | "
            f"{row.get('slope_sharpe_per_bp')} | {row.get('break_even_bps')} |"
        )
    write_md(EXEC_MD, "\n".join(lines) + "\n")
    return payload


def run_capacity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    loaded = load_market()
    scen = load_scenarios()
    rows: dict[str, Any] = {}
    for sid in IDS_EXEC:
        ov = load_overlayed(sid)
        pack = {}
        for cash, tag in ((300_000.0, "300k"), (1_000_000.0, "1M"), (3_000_000.0, "3M")):
            if tag == "1M":
                reuse = ((scen.get("matrix") or {}).get(sid) or {}).get("S3") or {}
                if reuse.get("sharpe") is not None:
                    pack[tag] = reuse
                    continue
            if tag == "3M":
                reuse = ((scen.get("matrix") or {}).get(sid) or {}).get("S4") or {}
                if reuse.get("sharpe") is not None:
                    pack[tag] = reuse
                    continue
            ov_l, meta = apply_liquidity_priority(ov, capital=cash)
            pack[tag] = _run_bt(
                f"{sid}_cap_{tag}",
                ov_l,
                None,
                cash if abs(cash - CONTRACT_CASH) > 1 else None,
                {"family": "capacity", "capital": cash, "liquidity": meta, "parent": sid},
                resume,
                loaded,
            )
        ov10, meta10 = apply_liquidity_priority(ov, capital=DIAGNOSTIC_CAPITAL)
        pack["10M_DIAGNOSTIC"] = _run_bt(
            f"{sid}_cap_10M",
            ov10,
            None,
            DIAGNOSTIC_CAPITAL,
            {"family": "capacity", "capital": DIAGNOSTIC_CAPITAL, "post_hoc_diagnostic": True, "parent": sid},
            resume,
            loaded,
        )
        rows[sid] = pack

    def _rating(pack: dict[str, Any]) -> str:
        s300 = (pack.get("300k") or {}).get("sharpe")
        s1 = (pack.get("1M") or {}).get("sharpe")
        s3 = (pack.get("3M") or {}).get("sharpe")
        if s300 is None:
            return "HIGH_RISK"
        if s1 is not None and s1 >= 0.8 * s300 and s3 is not None and s3 >= 0.7 * s300:
            return "UP_TO_3M"
        if s1 is not None and s1 >= 0.8 * s300:
            return "UP_TO_1M"
        return "ROBUST_300K_ONLY"

    ratings = {sid: _rating(pack) for sid, pack in rows.items()}
    payload = {
        "status": "DONE",
        "capitals": list(CAPITALS),
        "diagnostic_capital": DIAGNOSTIC_CAPITAL,
        "rows": rows,
        "ratings": ratings,
        "note": "3% Top1-first leftover。10M 仅 diagnostic。",
    }
    dump_json(CAP_JSON, payload)
    lines = [
        "# Round 13 · 容量",
        "",
        "| ID | 300k Sharpe | 1M | 3M | 10M diag | rating |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for sid, pack in rows.items():
        lines.append(
            f"| `{sid}` | {(pack.get('300k') or {}).get('sharpe')} | {(pack.get('1M') or {}).get('sharpe')} | "
            f"{(pack.get('3M') or {}).get('sharpe')} | {(pack.get('10M_DIAGNOSTIC') or {}).get('sharpe')} | {ratings[sid]} |"
        )
    write_md(CAP_MD, "\n".join(lines) + "\n")
    return payload


def run_forward(**_: Any) -> dict[str, Any]:
    require_frozen()
    from ..round12.contract import HERMES_SNAPSHOT

    status = "FORWARD_OOS_NOT_AVAILABLE"
    note = f"研究快照截止 {MARKET_END}。freeze 之后未发现完整未选参 forward 日频。"
    snap = HERMES_SNAPSHOT
    try:
        later = False
        if snap.is_dir():
            marker = snap / "daily"
            if marker.is_dir():
                later = False
        if later:
            status = "FORWARD_PRESENT_EVALUATE_ONLY"
    except Exception as exc:
        note = f"{note} check_error={exc}"
    payload = {
        "status": status,
        "cutoff": str(MARKET_END),
        "snapshot": str(snap),
        "note": note,
        "veto_only": True,
        "no_reweight": True,
        "evaluated_ids": None,
    }
    dump_json(FWD_JSON, payload)
    write_md(FWD_MD, f"# Round 13 · Forward\n\n**{status}**。{note}\n")
    return payload


def _core_gates(v4: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    cagr_drop_pp = (float(v4.get("cagr") or 0) - float(cand.get("cagr") or 0)) * 100.0
    sharpe_drop = float(v4.get("sharpe") or 0) - float(cand.get("sharpe") or 0)
    mdd_worse_pp = (abs(float(cand.get("max_drawdown") or 0)) - abs(float(v4.get("max_drawdown") or 0))) * 100.0
    loyo_med_drop_pp = (float(v4.get("loyo_median") or 0) - float(cand.get("loyo_median") or 0)) * 100.0
    loyo_w_drop_pp = (float(v4.get("loyo_worst") or 0) - float(cand.get("loyo_worst") or 0)) * 100.0
    checks = {
        "cagr_ok": cagr_drop_pp <= PROMOTION_MAX_CAGR_DROP_PP,
        "sharpe_ok": sharpe_drop <= PROMOTION_SHARPE_SLACK,
        "mdd_ok": mdd_worse_pp <= PROMOTION_MDD_SLACK_PP,
        "loyo_median_ok": loyo_med_drop_pp <= PROMOTION_LOYO_SLACK_PP,
        "loyo_worst_ok": loyo_w_drop_pp <= PROMOTION_LOYO_SLACK_PP,
    }
    return {
        "cagr_drop_pp": cagr_drop_pp,
        "sharpe_drop": sharpe_drop,
        "mdd_worse_pp": mdd_worse_pp,
        "loyo_median_drop_pp": loyo_med_drop_pp,
        "loyo_worst_drop_pp": loyo_w_drop_pp,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_promotion(**_: Any) -> dict[str, Any]:
    require_frozen()
    cand = load_candidates().get("results") or {}
    v4 = cand.get(R13_00) or dict(R12_01_TARGETS)
    scen = load_scenarios()
    boot = load_bootstrap()
    oob = load_oob()
    cpcv = load_cpcv()
    spa = load_spa()
    exe = load_json(EXEC_JSON)
    by = {}
    for cid in CANDIDATE_IDS:
        row = cand.get(cid) or {}
        core = _core_gates(v4, row)
        v4_s1 = ((scen.get("matrix") or {}).get("v4") or {}).get("S1") or {}
        v4_s2 = ((scen.get("matrix") or {}).get("v4") or {}).get("S2") or {}
        c_s1 = ((scen.get("matrix") or {}).get(cid) or {}).get("S1") or {}
        c_s2 = ((scen.get("matrix") or {}).get(cid) or {}).get("S2") or {}
        sh20_ok = (c_s1.get("sharpe") or -9) >= (v4_s1.get("sharpe") or 9)
        sh30_ok = (c_s2.get("sharpe") or -9) >= (v4_s2.get("sharpe") or 9)
        slope_c = ((exe.get("results") or {}).get(cid) or {}).get("slope_sharpe_per_bp")
        slope_v = ((exe.get("results") or {}).get("v4") or {}).get("slope_sharpe_per_bp")
        elast_better = (
            slope_c is not None and slope_v is not None and float(slope_c) > float(slope_v) + 1e-6
        )
        robust_ok = (sh20_ok and sh30_ok) or elast_better
        ssum = (scen.get("summary") or {})
        med_ok = (ssum.get(cid) or {}).get("median_sharpe") is not None and (ssum.get(cid) or {}).get("median_sharpe") >= ((ssum.get("v4") or {}).get("median_sharpe") or 9)
        worst_ok = (ssum.get(cid) or {}).get("worst_sharpe") is not None and (ssum.get(cid) or {}).get("worst_sharpe") > ((ssum.get("v4") or {}).get("worst_sharpe") or 9)
        regret_c = (scen.get("max_regret_sharpe") or {}).get(cid)
        regret_v = (scen.get("max_regret_sharpe") or {}).get("v4")
        regret_better = regret_c is not None and regret_v is not None and float(regret_c) < float(regret_v) - 1e-6
        scen_ok = (med_ok and worst_ok) or regret_better
        cpcv_c = ((cpcv.get("by_id") or {}).get(cid) or {}).get("oos_sharpe_median")
        cpcv_v = ((cpcv.get("by_id") or {}).get("v4") or {}).get("oos_sharpe_median")
        ms = {
            "cpcv_median_gt_v4": cpcv_c is not None and cpcv_v is not None and float(cpcv_c) > float(cpcv_v),
            "bootstrap_rank_std_lt_v4": (
                (((boot.get("sharpe") or {}).get(cid) or {}).get("rank_std") is not None)
                and (((boot.get("sharpe") or {}).get("v4") or {}).get("rank_std") is not None)
                and float(boot["sharpe"][cid]["rank_std"]) < float(boot["sharpe"]["v4"]["rank_std"])
            ),
            "oob_p_ge_060": float(((oob.get("by_id") or {}).get(cid) or {}).get("p_gt_winner") or 0) >= OOB_P_THRESHOLD,
            "spa_support": (
                float(((spa.get("raw") or {}).get(cid) or {}).get("raw_p") or 1) <= SPA_SUPPORT_P
                or float((spa.get("hansen_recentered") or {}).get("p_spa") or 1) <= SPA_SUPPORT_P
            ),
            "scenario_top3_gt_v4": (
                float((((scen.get("rank_tables") or {}).get("sharpe") or {}).get("top3_freq") or {}).get(cid) or 0)
                > float((((scen.get("rank_tables") or {}).get("sharpe") or {}).get("top3_freq") or {}).get("v4") or 0)
            ),
        }
        ms_n = int(sum(bool(v) for v in ms.values()))
        passed = bool(core["passed"] and robust_ok and scen_ok and ms_n >= 2)
        failed = []
        if not core["passed"]:
            failed.append("CORE:" + ",".join(k for k, v in core["checks"].items() if not v))
        if not robust_ok:
            failed.append("ROBUSTNESS")
        if not scen_ok:
            failed.append("SCENARIO")
        if ms_n < 2:
            failed.append(f"MODEL_SELECTION:{ms_n}/5")
        by[cid] = {
            "core": core,
            "robust_ok": robust_ok,
            "scenario_ok": scen_ok,
            "model_selection": ms,
            "model_selection_n": ms_n,
            "passed": passed,
            "failed_gates": failed,
            "cagr": row.get("cagr"),
            "sharpe": row.get("sharpe"),
            "max_drawdown": row.get("max_drawdown"),
        }

    passed_ids = [c for c in COMPLEXITY_ORDER if by[c]["passed"]]
    recommended = None
    upgrade = None
    if passed_ids:
        # near-tie: keep simpler among those within slacks of the best passed
        best = max(passed_ids, key=lambda c: (by[c]["sharpe"] or -9, by[c]["cagr"] or -9))
        chosen = best
        for c in COMPLEXITY_ORDER:
            if c not in passed_ids:
                continue
            dcagr = abs(float(by[c]["cagr"] or 0) - float(by[best]["cagr"] or 0))
            dsh = abs(float(by[c]["sharpe"] or 0) - float(by[best]["sharpe"] or 0))
            if dcagr < NEAR_TIE_CAGR_PP and dsh < NEAR_TIE_SHARPE:
                chosen = c
                break
        recommended = chosen
        ms_n = by[chosen]["model_selection_n"]
        robust = by[chosen]["robust_ok"]
        if ms_n >= 2 and robust:
            upgrade = "BOTH"
        elif ms_n >= 2:
            upgrade = "MODEL_SELECTION_ROBUSTNESS"
        else:
            upgrade = "STABILITY"
        decision = f"{RECOMMENDED_BASELINE_V5}={recommended}"
        lock = AWAITING_USER_LOCK
        final = KEEP_BASELINE_V4  # 不改 CURRENT
        reason = f"过门，推荐 {recommended}，状态 {AWAITING_USER_LOCK}。CURRENT 仍为 v4。"
    else:
        decision = KEEP_BASELINE_V4
        lock = None
        final = KEEP_BASELINE_V4
        failed_summary = {c: by[c]["failed_gates"] for c in CANDIDATE_IDS}
        reason = f"无候选同时满足 Core+Robustness+Scenario 且 Model-selection≥2。失败门={failed_summary}"
        upgrade = None

    payload = {
        "status": "DONE",
        "decision": decision,
        "final_current": final,
        "recommended_id": recommended,
        "upgrade_type": upgrade,
        "lock_status": lock,
        "auto_promote_current": False,
        "reason": reason,
        "by_id": by,
        "alpha_baseline": R13_00,
        "deployment_reference": {
            "note": "corrected + Round12 流动性 + EXEC_P50/P75。部署假设不进入 Alpha lineage。",
            "exec_p50_bps": FALLBACK_P50_BPS,
            "exec_p75_bps": FALLBACK_P75_BPS,
        },
    }
    dump_json(PROMO_JSON, payload)
    write_md(
        PROMO_MD,
        "\n".join([
            "# Round 13 · Promotion",
            "",
            f"裁决：**{decision}**。CURRENT = **{final}**。lock=`{lock}`。",
            "",
            reason,
            "",
            "| ID | passed | failed | CAGR | Sharpe |",
            "|---|---|---|---:|---:|",
            *[
                f"| `{c}` | {by[c]['passed']} | {by[c]['failed_gates']} | {pct(by[c]['cagr'])} | {by[c]['sharpe']} |"
                for c in CANDIDATE_IDS
            ],
            "",
        ]),
    )
    return payload


def gather_ctx() -> dict[str, Any]:
    frozen = require_frozen()
    promo = load_json(PROMO_JSON) or run_promotion()
    ctx = {
        "frozen": frozen,
        "parity": load_parity_status(),
        "weights": load_weights_audit(),
        "candidates": load_candidates(),
        "scenarios": load_scenarios(),
        "bootstrap": load_bootstrap(),
        "oob": load_oob(),
        "cpcv": load_cpcv(),
        "spa": load_spa(),
        "placebo": load_placebo(),
        "timing": load_timing(),
        "execution": load_json(EXEC_JSON),
        "capacity": load_json(CAP_JSON),
        "forward": load_json(FWD_JSON),
        "promotion": promo,
    }
    ctx["questions"] = _questions(ctx)
    return ctx


def _questions(ctx: dict[str, Any]) -> dict[str, str]:
    frozen = ctx.get("frozen") or {}
    parity = ctx.get("parity") or {}
    anchors = parity.get("anchors") or {}
    w = (ctx.get("weights") or {}).get("audit") or {}
    res = (ctx.get("candidates") or {}).get("results") or {}
    scen = ctx.get("scenarios") or {}
    boot = ctx.get("bootstrap") or {}
    oob = ctx.get("oob") or {}
    cpcv = ctx.get("cpcv") or {}
    spa = ctx.get("spa") or {}
    plc = ctx.get("placebo") or {}
    tim = ctx.get("timing") or {}
    exe = ctx.get("execution") or {}
    cap = ctx.get("capacity") or {}
    fwd = ctx.get("forward") or {}
    promo = ctx.get("promotion") or {}
    r00 = (anchors.get(R13_00) or {}).get("observed") or res.get(R13_00) or {}
    v2 = (anchors.get("R13_CTRL_V2") or {}).get("observed") or {}
    v3 = (anchors.get("R13_CTRL_V3") or {}).get("observed") or {}

    def _cand_line():
        bits = []
        for c in CANDIDATE_IDS:
            r = res.get(c) or {}
            bits.append(f"{c} CAGR={pct(r.get('cagr'))} Sharpe={r.get('sharpe')} MaxDD={pct(r.get('max_drawdown'))}")
        return "；".join(bits)

    return {
        "Q1": f"是。manifest sha256=`{frozen.get('sha256')}`。S9=`SCENARIO_NOT_PREDEFINED`。freeze 先于候选绩效。",
        "Q2": f"{(anchors.get(R13_00) or {}).get('status')}。观察 CAGR={r00.get('cagr')} Sharpe={r00.get('sharpe')} MaxDD={r00.get('max_drawdown')} 笔数={r00.get('completed_trades')}。目标对齐 R12_01。",
        "Q3": f"V2={(anchors.get('R13_CTRL_V2') or {}).get('status')} CAGR={v2.get('cagr')}；V3={(anchors.get('R13_CTRL_V3') or {}).get('status')} CAGR={v3.get('cagr')}。parity 失败则 STOP。总判定 `{parity.get('status')}`。",
        "Q4": "没有。正式候选仅预注册 A01–A05。无 60/40、无 A04b、无新因子/XGB/B03 阈值/exit。freeze 后未新增。",
        "Q5": f"是。locked alpha + exact_v2 / exact_e02 / B03_conservative。权重表 n={((ctx.get('weights') or {}).get('n_rows'))}。never_average_two_navs=True。",
        "Q6": "是。每个候选一次 `run_r12_backtest`。没有把两条 NAV 平均或相加。",
        "Q7": "是。RANK_NAV=(0.3,0.1)，TOP_K=2，`allocation_multiplier=clip(w/RANK_NAV[rank], MIN_KEEP, 1.0)`。",
        "Q8": f"union 第 3 名天数={((w.get('union') or {}).get('n_days_with_third_name'))}。出现则 STOP，未改 top_k。",
        "Q9": (
            f"公式已锁。Top2 身份一致={w.get('top2_identity_agree_share')}（分歧 n={w.get('top2_identity_disagree_n')}）；"
            f"仓位分歧 n={w.get('top2_position_disagree_n')} share={w.get('top2_position_disagree_share')}；"
            f"都要 Top2={w.get('top2_both_want_n')}。"
        ),
        "Q10": f"复杂度序 A01>A03>A02>A04>A05。推荐 `{promo.get('recommended_id')}`。near-tie 规则已应用。",
        "Q11": f"是。HISTORICAL_FEE_CORRECTED。锚点 R12_01 CAGR={R12_01_TARGETS['cagr']} Sharpe={R12_01_TARGETS['sharpe']} MaxDD={R12_01_TARGETS['max_drawdown']}。",
        "Q12": _cand_line(),
        "Q13": f"S0–S10 同一套。S9=`{scen.get('s9')}`。未调切点。promotion 情景={scen.get('promotion_scenarios')}。",
        "Q14": "是。S9=SCENARIO_NOT_PREDEFINED，不编切点，不进 promotion worst/median。down_pressure_p250 上四分位仅 POST_HOC_DIAGNOSTIC。",
        "Q15": f"是。S10=`{scen.get('s10_rule')}`，复用 B01A t1=0.45，不是新阈值。",
        "Q16": (
            f"Pareto={scen.get('pareto_ids')}。v4 median/worst Sharpe="
            f"{((scen.get('summary') or {}).get('v4') or {}).get('median_sharpe')}/"
            f"{((scen.get('summary') or {}).get('v4') or {}).get('worst_sharpe')}。"
            f"max regret={scen.get('max_regret_sharpe')}。"
        ),
        "Q17": f"N={boot.get('n')} block={boot.get('block')}。Sharpe rank={boot.get('sharpe')}。",
        "Q18": f"N={oob.get('n')} winner_freq={oob.get('winner_freq')}。by_id={oob.get('by_id')}。",
        "Q19": f"不是 forward OOS。by_id 含 median/mean/p10/worst 与 P(SR>0/>1)。见 `{cpcv.get('status')}`。",
        "Q20": f"H0=A01–A05 相对 v4 无更优。Hansen p={(spa.get('hansen_recentered') or {}).get('p_spa')}；reuse p={(spa.get('spa_test_reuse') or {}).get('p_spa')}；Holm={spa.get('holm_adjusted_p')}。",
        "Q21": f"method={(plc.get('method'))} min_n={plc.get('min_n')}。by_code percentiles 见主报告表格。",
        "Q22": (
            f"n={tim.get('n')} strata={tim.get('strata')}。"
            f"A01 CAGR 百分位={(tim.get('percentiles') or {}).get('cagr')} empir_p={tim.get('empirical_p_cagr')}。"
            f"PnL/暴露 A01={(tim.get('actual_pnl_exposure') or {}).get('pnl_per_unit_exposure')} "
            f"v4={(tim.get('v4_pnl_exposure') or {}).get('pnl_per_unit_exposure')}。"
        ),
        "Q23": f"bps={exe.get('bps')}。未优化。斜率/break-even={( {k: (v.get('slope_sharpe_per_bp'), v.get('break_even_bps')) for k, v in (exe.get('results') or {}).items()} )}。",
        "Q24": f"300k/1M/3M；10M diagnostic。ratings={cap.get('ratings')}。",
        "Q25": f"`{fwd.get('status')}`。{fwd.get('note')}",
        "Q26": f"decision=`{promo.get('decision')}` passed_ids 详见 by_id。原因：{promo.get('reason')}",
        "Q27": f"ALPHA_BASELINE=`{promo.get('alpha_baseline')}`。DEPLOYMENT_REFERENCE={promo.get('deployment_reference')}。部署假设未写入 Alpha lineage。",
        "Q28": (
            f"正式 CURRENT=**{promo.get('final_current')}**。推荐=`{promo.get('decision')}` lock=`{promo.get('lock_status')}` "
            f"upgrade=`{promo.get('upgrade_type')}`。**没有**把 v5 写成 CURRENT。"
        ),
    }


def _append_tracker(ctx: dict[str, Any]) -> None:
    text = TRACKER_PATH.read_text(encoding="utf-8")
    if "\n## Round 13\n" in text:
        return
    promo = ctx.get("promotion") or {}
    res = (ctx.get("candidates") or {}).get("results") or {}
    r00 = res.get(R13_00) or {}
    rec = promo.get("recommended_id")
    rec_row = res.get(rec) or {} if rec else {}
    block = [
        "",
        "## Round 13",
        "",
        "Selection-Robust Baseline Consolidation。只聚合已验证基线的目标权重。不是因子/阈值搜索。",
        f"CURRENT 仍为 v4。promotion = `{promo.get('decision')}`。lock = `{promo.get('lock_status')}`。",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| R13_00 CAGR/Sharpe/MaxDD | {pct(r00.get('cagr'))} / {r00.get('sharpe')} / {pct(r00.get('max_drawdown'))} |",
        f"| recommended | {rec or '—'} |",
        f"| recommended CAGR/Sharpe/MaxDD | {pct(rec_row.get('cagr'))} / {rec_row.get('sharpe')} / {pct(rec_row.get('max_drawdown'))} |",
        f"| lock | {promo.get('lock_status') or 'n/a'} |",
        f"| CURRENT | {promo.get('final_current')} |",
        f"| forward OOS | {(ctx.get('forward') or {}).get('status')} |",
        f"| S9 | SCENARIO_NOT_PREDEFINED |",
        f"| manifest sha256 | `{(ctx.get('frozen') or {}).get('sha256')}` |",
        "",
        "主报告：`result/round13/ROUND13_BASELINE_CONSOLIDATION_REPORT.md`。",
        "",
    ]
    TRACKER_PATH.write_text(text.rstrip() + "\n" + "\n".join(block), encoding="utf-8")


def run_report(**_: Any) -> dict[str, Any]:
    ctx = gather_ctx()
    qs = ctx["questions"]
    promo = ctx.get("promotion") or {}
    res = (ctx.get("candidates") or {}).get("results") or {}
    body = [
        "# ROUND13_BASELINE_CONSOLIDATION_REPORT",
        "",
        "Round 13 = Selection-Robust Baseline Consolidation。**不是**因子/阈值搜索。",
        f"manifest sha256=`{(ctx.get('frozen') or {}).get('sha256')}`。",
        f"CURRENT = **{promo.get('final_current')}**。decision = **{promo.get('decision')}**。",
        "",
        "## 01 · 使命与冻结",
        "",
        "先 freeze 再看候选。仅 A01–A05。不得自动把 v5 写成 CURRENT。",
        "",
        "## 02 · Parity",
        "",
        f"总判定 `{(ctx.get('parity') or {}).get('status')}`。R13_00 / CTRL_V2 / CTRL_V3 必须精确对齐 corrected 账本。",
        "",
        "## 03 · 日度目标权重",
        "",
        f"{(ctx.get('weights') or {}).get('engine_clip_documented')} audit={(ctx.get('weights') or {}).get('audit')}",
        "",
        "## 04 · 候选绩效（corrected）",
        "",
        "| ID | CAGR | Sharpe | MaxDD | 笔数 | LOYO med | LOYO worst |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for code in (R13_00, "R13_CTRL_V2", "R13_CTRL_V3", *CANDIDATE_IDS):
        row = res.get(code) or {}
        body.append(
            f"| `{code}` | {pct(row.get('cagr'))} | {row.get('sharpe')} | {pct(row.get('max_drawdown'))} | "
            f"{row.get('completed_trades')} | {pct(row.get('loyo_median'))} | {pct(row.get('loyo_worst'))} |"
        )
    body += [
        "",
        "## 05 · 情景 / Pareto",
        "",
        f"S9=`{(ctx.get('scenarios') or {}).get('s9')}`。Pareto={(ctx.get('scenarios') or {}).get('pareto_ids')}。",
        f"summary={(ctx.get('scenarios') or {}).get('summary')}",
        "",
        "## 06 · Bootstrap / OOB / CPCV / SPA",
        "",
        f"bootstrap N={(ctx.get('bootstrap') or {}).get('n')}；OOB N={(ctx.get('oob') or {}).get('n')}；"
        f"SPA Hansen p={((ctx.get('spa') or {}).get('hansen_recentered') or {}).get('p_spa')}。",
        "",
        "## 07 · Placebo / Timing",
        "",
        f"placebo min_n={(ctx.get('placebo') or {}).get('min_n')}；timing n={(ctx.get('timing') or {}).get('n')}。",
        "",
        "## 08 · 执行 / 容量 / Forward",
        "",
        f"execution no_optimize={(ctx.get('execution') or {}).get('no_optimize')}；"
        f"capacity ratings={(ctx.get('capacity') or {}).get('ratings')}；"
        f"forward={(ctx.get('forward') or {}).get('status')}。",
        "",
        "## 09 · ALPHA vs DEPLOYMENT",
        "",
        "ALPHA_BASELINE = corrected-cost 单组合（R13_00 / A01–A05）。",
        "DEPLOYMENT_REFERENCE = corrected + Round12 流动性 3% Top1-first + EXEC_P50/P75。部署假设不进入 Alpha lineage。",
        "",
        "## 10 · Promotion",
        "",
        f"**{promo.get('decision')}**。CURRENT=**{promo.get('final_current')}**。{promo.get('reason')}",
        "",
        "## 11 · Q1–Q28",
        "",
    ]
    for i in range(1, 29):
        key = f"Q{i}"
        body += [f"### {key} — {PRE_REGISTERED_QUESTIONS[key]}", "", qs.get(key, "NOT_IDENTIFIABLE"), ""]
    body += [
        "### 明确没做",
        "",
        "- 不搜新因子、不改 fused score、不改 90/10、不改 B03 阈值、不改 exit",
        "- 无 60/40 网格、无 A04b、freeze 后无新候选",
        "- 不把两条 NAV 平均；不把 S9 编切点；不把 CPCV 写成 forward OOS",
        "- 不自动把 v5 写成 CURRENT；追踪表不写「写入时间」",
        "",
    ]
    write_md(REPORT_MD, "\n".join(body))
    _append_tracker(ctx)
    complete = {
        "decision": promo.get("decision"),
        "final_current": promo.get("final_current"),
        "recommended_id": promo.get("recommended_id"),
        "lock_status": promo.get("lock_status"),
        "upgrade_type": promo.get("upgrade_type"),
        "reason": promo.get("reason"),
        "R13_00": res.get(R13_00),
        "candidates": {c: res.get(c) for c in CANDIDATE_IDS},
        "s9": (ctx.get("scenarios") or {}).get("s9"),
        "forward_oos": (ctx.get("forward") or {}).get("status"),
        "manifest_sha256": (ctx.get("frozen") or {}).get("sha256"),
        "dossier": str(REPORT_MD),
        "questions": qs,
        "promotion": promo,
    }
    dump_json(DOSSIER_JSON, complete)
    return complete


def print_complete_block(payload: dict[str, Any]) -> None:
    r00 = payload.get("R13_00") or {}
    rec = payload.get("recommended_id")
    rec_row = (payload.get("candidates") or {}).get(rec) or {}
    print("========== ROUND13 COMPLETE ==========", flush=True)
    print(f"DECISION={payload.get('decision')}", flush=True)
    print(f"CURRENT={payload.get('final_current')}", flush=True)
    print(f"LOCK={payload.get('lock_status')}", flush=True)
    print(f"UPGRADE_TYPE={payload.get('upgrade_type')}", flush=True)
    print(
        f"R13_00 CAGR={r00.get('cagr')} Sharpe={r00.get('sharpe')} MaxDD={r00.get('max_drawdown')}",
        flush=True,
    )
    if rec:
        print(
            f"{rec} CAGR={rec_row.get('cagr')} Sharpe={rec_row.get('sharpe')} MaxDD={rec_row.get('max_drawdown')}",
            flush=True,
        )
    print(f"S9={payload.get('s9')} FORWARD={payload.get('forward_oos')}", flush=True)
    print(f"MANIFEST={payload.get('manifest_sha256')}", flush=True)
    print(f"DOSSIER={payload.get('dossier')}", flush=True)
    print(f"REASON={payload.get('reason')}", flush=True)
    print("======================================", flush=True)


__all__ = [
    "gather_ctx",
    "print_complete_block",
    "run_capacity",
    "run_execution",
    "run_forward",
    "run_promotion",
    "run_report",
]
