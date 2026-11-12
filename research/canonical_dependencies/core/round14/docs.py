"""执行、容量、尾部、forward、promotion、Q1–Q28、追踪表。不写「写入时间」。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import MARKET_END, ROUND14_ROOT
from ..round12.deploy import apply_liquidity_priority
from ..round12.runner import hist_hooks
from .candidates import load_candidates, load_controls, load_daily_map
from .common import cvar95, dump_json, load_json, max_drawdown_of, pct, write_md
from .contract import (
    AWAITING_USER_LOCK,
    CANDIDATE_IDS,
    CAPITALS,
    COMPLEXITY_ORDER,
    CONTRACT_CASH,
    DE_RISKING_ONLY,
    DIAGNOSTIC_CAPITAL,
    EXEC_BPS,
    FALLBACK_P50_BPS,
    FALLBACK_P75_BPS,
    FINAL_REPORT,
    IMPROVE_N_REQUIRED,
    KEEP_BASELINE_V4,
    NEAR_TIE_CAGR_PP,
    NEAR_TIE_SHARPE,
    PRE_REGISTERED_QUESTIONS,
    PROMOTION_LOYO_SLACK_PP,
    PROMOTION_MAX_CAGR_DROP_PP,
    PROMOTION_MDD_SLACK_PP,
    PROMOTION_SHARPE_SLACK,
    R12_01_TARGETS,
    R14_00,
    RECOMMENDED_BASELINE_V5,
    SPA_SUPPORT_P,
    TRACKER_PATH,
)
from .freeze import require_frozen
from .parity import load_parity_status
from .runner import load_market, run_r14_backtest
from .scenarios import PROMOTION_SCENARIOS, load_scenarios
from .stats import load_bootstrap, load_cpcv, load_matched, load_spa
from .weights import load_overlayed, load_weights_audit

EXEC_JSON = ROUND14_ROOT / "execution" / "execution.json"
EXEC_MD = ROUND14_ROOT / "10_execution.md"
CAP_JSON = ROUND14_ROOT / "capacity" / "capacity.json"
CAP_MD = ROUND14_ROOT / "11_capacity.md"
TAIL_JSON = ROUND14_ROOT / "tail" / "tail.json"
TAIL_MD = ROUND14_ROOT / "12_tail.md"
FWD_JSON = ROUND14_ROOT / "forward" / "forward.json"
FWD_MD = ROUND14_ROOT / "13_forward.md"
PROMO_JSON = ROUND14_ROOT / "promotion" / "promotion.json"
PROMO_MD = ROUND14_ROOT / "14_promotion.md"
DOSSIER_JSON = ROUND14_ROOT / "stats" / "dossier.json"
IDS_EXEC = ("v4", *CANDIDATE_IDS)


def _run_bt(tag: str, ov, bps, cash, extra, resume, loaded) -> dict[str, Any]:
    dest = ROUND14_ROOT / extra.get("family", "execution") / tag
    ran = run_r14_backtest(
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
                src = cand.get(R14_00 if sid == "v4" else sid) or {}
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
        xs, ys = [], []
        for b in EXEC_BPS:
            row = curve.get(str(int(b))) or {}
            if row.get("sharpe") is not None:
                xs.append(float(b))
                ys.append(float(row["sharpe"]))
        slope = be = None
        if len(xs) >= 2:
            slope = float(np.polyfit(xs, ys, 1)[0])
            if slope != 0:
                intercept = float(np.polyfit(xs, ys, 1)[1])
                be = float(-intercept / slope) if slope < 0 else None
        results[sid] = {"curve": curve, "slope_sharpe_per_bp": slope, "break_even_bps": be}
    payload = {
        "status": "DONE",
        "bps": list(EXEC_BPS),
        "no_optimize": True,
        "results": results,
    }
    dump_json(EXEC_JSON, payload)
    lines = [
        "# Round 14 · 执行弹性",
        "",
        "| ID | 0 | 10 | 20 | 30 | 50 | slope | break-even bp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sid, row in results.items():
        c = row["curve"]

        def _s(b, _c=c):
            return (_c.get(str(b)) or {}).get("sharpe")

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
        "# Round 14 · 容量",
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


def _drawdown_episodes(daily: pd.Series, n: int = 5) -> list[dict[str, Any]]:
    x = daily.dropna().astype(float)
    if x.empty:
        return []
    wealth = (1.0 + x).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    episodes = []
    in_dd = False
    start = None
    trough = None
    trough_val = 0.0
    for t, v in dd.items():
        if v < 0 and not in_dd:
            in_dd = True
            start = t
            trough = t
            trough_val = float(v)
        elif in_dd:
            if float(v) < trough_val:
                trough = t
                trough_val = float(v)
            if v >= -1e-12:
                episodes.append({
                    "start": str(pd.Timestamp(start).date()),
                    "trough": str(pd.Timestamp(trough).date()),
                    "end": str(pd.Timestamp(t).date()),
                    "depth": trough_val,
                })
                in_dd = False
    if in_dd and start is not None:
        episodes.append({
            "start": str(pd.Timestamp(start).date()),
            "trough": str(pd.Timestamp(trough).date()),
            "end": str(pd.Timestamp(dd.index[-1]).date()),
            "depth": trough_val,
        })
    episodes.sort(key=lambda r: r["depth"])
    return episodes[:n]


def run_tail(**_: Any) -> dict[str, Any]:
    require_frozen()
    from .risk import load_risk_panel

    dailies = load_daily_map()
    v4 = dailies["v4"]
    episodes = _drawdown_episodes(v4, 5)
    panel = load_risk_panel()
    if not panel.empty:
        panel = panel.copy()
        panel["signal_date"] = pd.to_datetime(panel["signal_date"]).dt.normalize()
        panel = panel.set_index("signal_date")
    by_id: dict[str, Any] = {}
    for sid in ("v4", *CANDIDATE_IDS):
        daily = dailies.get(sid)
        if daily is None:
            continue
        rows = []
        for ep in episodes:
            sl = daily.loc[ep["start"]: ep["end"]]
            rows.append({
                **ep,
                "cagr_window": float((1 + sl).prod() - 1) if len(sl) else None,
                "maxdd_window": max_drawdown_of(sl),
            })
        left = daily.loc[daily <= daily.quantile(0.05)]
        right = daily.loc[daily >= daily.quantile(0.95)]
        v4_left = v4.reindex(left.index)
        v4_right = v4.reindex(right.index)
        by_id[sid] = {
            "episodes": rows,
            "left_tail_mean": float(left.mean()) if len(left) else None,
            "right_tail_mean": float(right.mean()) if len(right) else None,
            "v4_left_mean": float(v4_left.mean()) if len(v4_left) else None,
            "v4_right_mean": float(v4_right.mean()) if len(v4_right) else None,
            "cvar95": cvar95(daily),
        }
    high_risk_cut = {}
    if not panel.empty and episodes:
        for ep in episodes:
            start = pd.Timestamp(ep["start"])
            window = panel.loc[(panel.index >= start - pd.Timedelta(days=5)) & (panel.index <= start)]
            high_risk_cut[ep["start"]] = {
                "exante_sigma2": float(window["sigma2_20"].median()) if len(window) and "sigma2_20" in window else None,
                "exante_mrc2": float(window["MRC2"].median()) if len(window) and "MRC2" in window else None,
                "exante_sigmap": float(window["sigma_portfolio"].median()) if len(window) and "sigma_portfolio" in window else None,
            }
    payload = {
        "status": "DONE",
        "v4_top5_drawdowns": episodes,
        "by_id": by_id,
        "exante_at_dd_start": high_risk_cut,
        "note": "问风险层是否在 ex-ante 风险高时减亏。右尾保留 vs 左尾削减。",
    }
    dump_json(TAIL_JSON, payload)
    lines = [
        "# Round 14 · 尾部 / 回撤归因",
        "",
        f"v4 最大 5 段回撤深度 = {[e['depth'] for e in episodes]}。",
        "",
        "| ID | CVaR95 | 左尾均值 | 右尾均值 | v4左尾 | v4右尾 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sid in ("v4", *CANDIDATE_IDS):
        row = by_id.get(sid) or {}
        lines.append(
            f"| `{sid}` | {row.get('cvar95')} | {row.get('left_tail_mean')} | {row.get('right_tail_mean')} | "
            f"{row.get('v4_left_mean')} | {row.get('v4_right_mean')} |"
        )
    write_md(TAIL_MD, "\n".join(lines) + "\n")
    return payload


def run_forward(**_: Any) -> dict[str, Any]:
    require_frozen()
    from ..round12.contract import HERMES_SNAPSHOT

    status = "FORWARD_OOS_NOT_AVAILABLE"
    note = f"研究快照截止 {MARKET_END}。freeze 之后未发现完整未选参 forward 日频。"
    payload = {
        "status": status,
        "cutoff": str(MARKET_END),
        "snapshot": str(HERMES_SNAPSHOT),
        "note": note,
        "veto_only": True,
        "no_reweight": True,
    }
    dump_json(FWD_JSON, payload)
    write_md(FWD_MD, f"# Round 14 · Forward\n\n**{status}**。{note}\n")
    return payload


def _stability_core(v4: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
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
    ret_gate = {
        "cagr_gt_v4": float(cand.get("cagr") or 0) > float(v4.get("cagr") or 0),
        "sharpe_ge_v4": float(cand.get("sharpe") or 0) >= float(v4.get("sharpe") or 0) - 1e-12,
        "mdd_not_clearly_worse": mdd_worse_pp <= PROMOTION_MDD_SLACK_PP,
    }
    return {
        "cagr_drop_pp": cagr_drop_pp,
        "sharpe_drop": sharpe_drop,
        "mdd_worse_pp": mdd_worse_pp,
        "loyo_median_drop_pp": loyo_med_drop_pp,
        "loyo_worst_drop_pp": loyo_w_drop_pp,
        "checks": checks,
        "stability_passed": all(checks.values()),
        "return_gate": ret_gate,
        "return_passed": all(ret_gate.values()),
    }


def run_promotion(**_: Any) -> dict[str, Any]:
    require_frozen()
    cand = load_candidates().get("results") or {}
    v4 = cand.get(R14_00) or dict(R12_01_TARGETS)
    scen = load_scenarios()
    boot = load_bootstrap()
    cpcv = load_cpcv()
    spa = load_spa()
    matched = load_matched()
    ctrl = load_controls()
    tail = load_json(TAIL_JSON)
    dailies = load_daily_map()
    by = {}
    for cid in CANDIDATE_IDS:
        row = cand.get(cid) or {}
        core = _stability_core(v4, row)
        cmp = ((ctrl.get("comparisons") or {}).get(cid) or {})
        label = cmp.get("label")
        derisk_ban = label == DE_RISKING_ONLY
        v4_s2 = ((scen.get("matrix") or {}).get("v4") or {}).get("S2") or {}
        c_s2 = ((scen.get("matrix") or {}).get(cid) or {}).get("S2") or {}
        v4_s4 = ((scen.get("matrix") or {}).get("v4") or {}).get("S4") or {}
        c_s4 = ((scen.get("matrix") or {}).get(cid) or {}).get("S4") or {}
        ssum = scen.get("summary") or {}
        items = {
            "1_maxdd_ge_1pp": core["mdd_worse_pp"] <= -1.0,
            "2_worst12m_ge_1_5pp": False,
            "3_cvar95": False,
            "4_worst_scen_sharpe_ge_0_10": (
                ((ssum.get(cid) or {}).get("worst_sharpe") or -9)
                >= ((ssum.get("v4") or {}).get("worst_sharpe") or 9) + 0.10
            ),
            "5_plus30_sharpe_ge_0_03": (
                (c_s2.get("sharpe") or -9) >= (v4_s2.get("sharpe") or 9) + 0.03
            ),
            "6_3m_sharpe_ge_0_03": (
                (c_s4.get("sharpe") or -9) >= (v4_s4.get("sharpe") or 9) + 0.03
            ),
            "7_cpcv_p10_ge_0_05": False,
            "8_boot_rank_std_down_15": False,
            "9_p_matched_ge_0_65": float(((matched.get("by_id") or {}).get(cid) or {}).get("p_cand_gt_ctrl") or 0) >= 0.65,
            "10_unit_pnl_ge_v4": bool(cmp.get("unit_pnl_up_vs_v4")),
        }
        if cid in dailies and "v4" in dailies:
            from .common import worst_window_return

            w_c = worst_window_return(dailies[cid], 252).get("return")
            w_v = worst_window_return(dailies["v4"], 252).get("return")
            if w_c is not None and w_v is not None:
                items["2_worst12m_ge_1_5pp"] = (float(w_c) - float(w_v)) * 100.0 >= 1.5
            c_cvar = ((tail.get("by_id") or {}).get(cid) or {}).get("cvar95")
            v_cvar = ((tail.get("by_id") or {}).get("v4") or {}).get("cvar95")
            if c_cvar is not None and v_cvar is not None:
                items["3_cvar95"] = float(c_cvar) > float(v_cvar)
        c_p10 = ((cpcv.get("by_id") or {}).get(cid) or {}).get("oos_sharpe_p10")
        v_p10 = ((cpcv.get("by_id") or {}).get("v4") or {}).get("oos_sharpe_p10")
        if c_p10 is not None and v_p10 is not None:
            items["7_cpcv_p10_ge_0_05"] = float(c_p10) >= float(v_p10) + 0.05
        b_c = ((boot.get("sharpe") or {}).get(cid) or {}).get("rank_std")
        b_v = ((boot.get("sharpe") or {}).get("v4") or {}).get("rank_std")
        if b_c is not None and b_v is not None and float(b_v) > 0:
            items["8_boot_rank_std_down_15"] = float(b_c) <= float(b_v) * 0.85
        n_improve = int(sum(bool(v) for v in items.values()))
        has_789 = bool(items["7_cpcv_p10_ge_0_05"] or items["8_boot_rank_std_down_15"] or items["9_p_matched_ge_0_65"])
        cpcv_c = ((cpcv.get("by_id") or {}).get(cid) or {}).get("oos_sharpe_median")
        cpcv_v = ((cpcv.get("by_id") or {}).get("v4") or {}).get("oos_sharpe_median")
        ms = {
            "cpcv_median_or_p10": (
                (cpcv_c is not None and cpcv_v is not None and float(cpcv_c) > float(cpcv_v))
                or items["7_cpcv_p10_ge_0_05"]
            ),
            "bootstrap_rank": (
                b_c is not None and b_v is not None and float(b_c) < float(b_v)
            ),
            "spa": (
                float(((spa.get("raw") or {}).get(cid) or {}).get("raw_p") or 1) <= SPA_SUPPORT_P
                or float((spa.get("hansen_recentered") or {}).get("p_spa") or 1) <= SPA_SUPPORT_P
            ),
            "matched_bootstrap": float(((matched.get("by_id") or {}).get(cid) or {}).get("p_cand_gt_ctrl") or 0) >= 0.60,
        }
        ms_n = int(sum(bool(v) for v in ms.values()))
        stab_ok = bool(core["stability_passed"] and n_improve >= IMPROVE_N_REQUIRED and has_789)
        ret_ok = bool(core["return_passed"] and stab_ok)
        passed = bool((stab_ok or ret_ok) and ms_n >= 2 and not derisk_ban and row.get("eligible", True))
        failed = []
        if derisk_ban:
            failed.append("DE_RISKING_ONLY")
        if not core["stability_passed"]:
            failed.append("STABILITY:" + ",".join(k for k, v in core["checks"].items() if not v))
        if n_improve < IMPROVE_N_REQUIRED:
            failed.append(f"IMPROVE:{n_improve}/10")
        if not has_789:
            failed.append("NEED_ITEM_7_8_9")
        if ms_n < 2:
            failed.append(f"MODEL_SELECTION:{ms_n}/4")
        if not row.get("eligible", True):
            failed.append("FAIL_ALPHA_MUTATION")
        by[cid] = {
            "core": core,
            "label": label,
            "improve": items,
            "improve_n": n_improve,
            "has_789": has_789,
            "model_selection": ms,
            "model_selection_n": ms_n,
            "stability_ok": stab_ok,
            "return_ok": ret_ok,
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
        if by[chosen]["return_ok"]:
            upgrade = "RETURN"
        else:
            upgrade = "STABILITY/RISK_BUDGET"
        decision = f"{RECOMMENDED_BASELINE_V5}={recommended}"
        lock = AWAITING_USER_LOCK
        final = KEEP_BASELINE_V4
        reason = f"过门，推荐 {recommended}（{upgrade}），状态 {AWAITING_USER_LOCK}。CURRENT 仍为 v4。"
    else:
        decision = KEEP_BASELINE_V4
        lock = None
        final = KEEP_BASELINE_V4
        failed_summary = {c: by[c]["failed_gates"] for c in CANDIDATE_IDS}
        reason = f"无候选同时满足稳定性/收益门 + 3/10 改善（含 7/8/9）+ 模型选择≥2 且非 DE_RISKING_ONLY。失败门={failed_summary}"
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
        "alpha_baseline": R14_00,
        "deployment_reference": {
            "note": "DEPLOYMENT_P50/P75 不进入 Alpha lineage。",
            "exec_p50_bps": FALLBACK_P50_BPS,
            "exec_p75_bps": FALLBACK_P75_BPS,
        },
    }
    dump_json(PROMO_JSON, payload)
    write_md(
        PROMO_MD,
        "\n".join([
            "# Round 14 · Promotion",
            "",
            f"裁决：**{decision}**。CURRENT = **{final}**。lock=`{lock}`。",
            "",
            reason,
            "",
            "| ID | label | passed | failed | CAGR | Sharpe |",
            "|---|---|---|---|---:|---:|",
            *[
                f"| `{c}` | {by[c]['label']} | {by[c]['passed']} | {by[c]['failed_gates']} | {pct(by[c]['cagr'])} | {by[c]['sharpe']} |"
                for c in CANDIDATE_IDS
            ],
            "",
        ]),
    )
    return payload


def gather_ctx() -> dict[str, Any]:
    frozen = require_frozen()
    promo = load_json(PROMO_JSON) or run_promotion()
    return {
        "frozen": frozen,
        "parity": load_parity_status(),
        "weights": load_weights_audit(),
        "candidates": load_candidates(),
        "controls": load_controls(),
        "scenarios": load_scenarios(),
        "bootstrap": load_bootstrap(),
        "matched": load_matched(),
        "cpcv": load_cpcv(),
        "spa": load_spa(),
        "execution": load_json(EXEC_JSON),
        "capacity": load_json(CAP_JSON),
        "tail": load_json(TAIL_JSON),
        "forward": load_json(FWD_JSON),
        "promotion": promo,
        "risk": load_json(ROUND14_ROOT / "risk" / "risk_panel_meta.json"),
        "tests": load_json(ROUND14_ROOT / "tests" / "unit_tests.json"),
    }


def _questions(ctx: dict[str, Any]) -> dict[str, str]:
    frozen = ctx.get("frozen") or {}
    parity = ctx.get("parity") or {}
    anchors = parity.get("anchors") or {}
    res = (ctx.get("candidates") or {}).get("results") or {}
    ctrl = ctx.get("controls") or {}
    scen = ctx.get("scenarios") or {}
    boot = ctx.get("bootstrap") or {}
    matched = ctx.get("matched") or {}
    cpcv = ctx.get("cpcv") or {}
    spa = ctx.get("spa") or {}
    exe = ctx.get("execution") or {}
    cap = ctx.get("capacity") or {}
    tail = ctx.get("tail") or {}
    fwd = ctx.get("forward") or {}
    promo = ctx.get("promotion") or {}
    risk = ctx.get("risk") or {}
    tests = ctx.get("tests") or {}
    w = ctx.get("weights") or {}
    r00 = (anchors.get(R14_00) or {}).get("observed") or res.get(R14_00) or {}
    key = ctrl.get("key") or (ctrl.get("comparisons") or {}).get("B02") or {}

    def _cand_line():
        bits = []
        for c in CANDIDATE_IDS:
            r = res.get(c) or {}
            bits.append(f"{c} CAGR={pct(r.get('cagr'))} Sharpe={r.get('sharpe')} MaxDD={pct(r.get('max_drawdown'))}")
        return "；".join(bits)

    return {
        "Q1": f"是。诊断先于 freeze，freeze 先于候选绩效。manifest sha256=`{frozen.get('sha256')}`。",
        "Q2": (
            f"{(anchors.get(R14_00) or {}).get('status')}。观察 CAGR={r00.get('cagr')} Sharpe={r00.get('sharpe')} "
            f"MaxDD={r00.get('max_drawdown')} 笔数={r00.get('completed_trades')}。对齐 R12_01/R13_00。失败则 STOP。"
        ),
        "Q3": (
            f"PIT 含 D=`{risk.get('include_d')}`。sigma 窗={risk.get('sigma_window')} 最少={risk.get('sigma_min_obs')}。"
            f"missing_fill_zero={risk.get('missing_fill_zero')} fallback_to_v4={risk.get('fallback_to_v4')}。"
        ),
        "Q4": f"rho 窗={risk.get('rho_window')} 最少={risk.get('rho_min_obs')}。不足显式标记。rho 不可用比例={(risk.get('missing') or {}).get('rho')}。",
        "Q5": f"按冻结定义。MRC2≤0 → 因子=1。mrc fallback 比例={(risk.get('missing') or {}).get('fallback_mrc')}。",
        "Q6": f"参考 {risk.get('ref_signal_days')} 个 SIGNAL DAY，最少 {risk.get('ref_min_obs')}。不足回退 v4。",
        "Q7": f"诊断已写 `01_risk_feature_diagnostics.md`。diagnostics_changed_formulas={risk.get('diagnostics_changed_formulas')}。",
        "Q8": "是。仅 A01/B01/B02/B03/C01。无笛卡尔积、无新搜阈值。",
        "Q9": f"身份锁={tests.get('all_identity_pass')}。失败名单={w.get('failed_identity')}。失败=FAIL_ALPHA_MUTATION。",
        "Q10": f"零 Top2 用 MIN_KEEP。单测={(tests.get('zero_top2') or {}).get('status')}。",
        "Q11": "是。A01 单调分段线性，Top1=v4，无 sigmoid/幂。",
        "Q12": "是。B01 只减不增；缺失回退 v4。",
        "Q13": (
            f"是。B02 vs CTRL_EXP_B02：label=`{key.get('label')}` "
            f"Sharpe 候选/对照={key.get('cand_sharpe')}/{key.get('ctrl_sharpe')}。"
        ),
        "Q14": "是。同一 portfolio_risk_factor 乘 w1 与 w2。无 10/12/15% 网格。",
        "Q15": "是。C01 = A01 w2_smooth × B02 风格 MRC（v4 Top1 + smooth Top2 估 Σ）。不再叠层。",
        "Q16": f"是。scale=E_cand/E_v4。无风险信息。关键对照={ctrl.get('key_test')}。",
        "Q17": f"对照判定={( {c: ((ctrl.get('comparisons') or {}).get(c) or {}).get('label') for c in CANDIDATE_IDS} )}。",
        "Q18": (
            f"是。HISTORICAL_FEE_CORRECTED。锚点 R12_01 CAGR={R12_01_TARGETS['cagr']} "
            f"Sharpe={R12_01_TARGETS['sharpe']} MaxDD={R12_01_TARGETS['max_drawdown']}。"
        ),
        "Q19": f"S0–S9 同一套。S8=`{scen.get('s8_rule')}`。S9=`{scen.get('s9')}`。未调切点。",
        "Q20": (
            f"bps={exe.get('bps')}。容量 ratings={cap.get('ratings')}。10M diagnostic。"
        ),
        "Q21": f"N={boot.get('n')} block={boot.get('block')}。Sharpe rank={boot.get('sharpe')}。",
        "Q22": f"N={matched.get('n')}。by_id={matched.get('by_id')}。",
        "Q23": f"不是 forward OOS。status=`{cpcv.get('status')}`。by_id 含 median/p10/worst。",
        "Q24": (
            f"H0=五候选相对 v4 无更优。Hansen p={(spa.get('hansen_recentered') or {}).get('p_spa')}；"
            f"Holm={spa.get('holm_adjusted_p')}。"
        ),
        "Q25": (
            f"v4 最大 5 段回撤={tail.get('v4_top5_drawdowns')}。"
            f"ex-ante={tail.get('exante_at_dd_start')}。"
        ),
        "Q26": f"decision=`{promo.get('decision')}`。原因：{promo.get('reason')}",
        "Q27": (
            f"DE_RISKING_ONLY 未升 v5。ALPHA_BASELINE=`{promo.get('alpha_baseline')}`。"
            f"DEPLOYMENT={promo.get('deployment_reference')}。"
        ),
        "Q28": (
            f"正式 CURRENT=**{promo.get('final_current')}**。推荐=`{promo.get('decision')}` "
            f"lock=`{promo.get('lock_status')}`。**没有**把 v5 写成 CURRENT。追踪表不写「写入时间」。"
        ),
    }


def _append_tracker(ctx: dict[str, Any]) -> None:
    text = TRACKER_PATH.read_text(encoding="utf-8")
    if "\n## Round 14\n" in text:
        return
    promo = ctx.get("promotion") or {}
    res = (ctx.get("candidates") or {}).get("results") or {}
    r00 = res.get(R14_00) or {}
    rec = promo.get("recommended_id")
    rec_row = res.get(rec) or {} if rec else {}
    key = ((ctx.get("controls") or {}).get("key") or {})
    block = [
        "",
        "## Round 14",
        "",
        "Alpha-Preserving Risk Budgeting。只改 Top1/Top2 目标资金，不改选股。",
        f"CURRENT 仍为 v4。promotion = `{promo.get('decision')}`。lock = `{promo.get('lock_status')}`。",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| R14_00 CAGR/Sharpe/MaxDD | {pct(r00.get('cagr'))} / {r00.get('sharpe')} / {pct(r00.get('max_drawdown'))} |",
        f"| B02 vs CTRL_EXP_B02 | {key.get('label')} |",
        f"| recommended | {rec or '—'} |",
        f"| recommended CAGR/Sharpe/MaxDD | {pct(rec_row.get('cagr'))} / {rec_row.get('sharpe')} / {pct(rec_row.get('max_drawdown'))} |",
        f"| lock | {promo.get('lock_status') or 'n/a'} |",
        f"| CURRENT | {promo.get('final_current')} |",
        f"| forward OOS | {(ctx.get('forward') or {}).get('status')} |",
        f"| manifest sha256 | `{(ctx.get('frozen') or {}).get('sha256')}` |",
        "",
        "主报告：`result/round14/ROUND14_RISK_BUDGET_REPORT.md`。",
        "",
    ]
    TRACKER_PATH.write_text(text.rstrip() + "\n" + "\n".join(block), encoding="utf-8")


def run_report(**_: Any) -> dict[str, Any]:
    ctx = gather_ctx()
    ctx["questions"] = _questions(ctx)
    qs = ctx["questions"]
    promo = ctx.get("promotion") or {}
    res = (ctx.get("candidates") or {}).get("results") or {}
    body = [
        "# ROUND14_RISK_BUDGET_REPORT",
        "",
        "Round 14 = Alpha-Preserving Risk Budgeting。**不研究买谁**，只改 Top1/Top2 目标资金。",
        f"manifest sha256=`{(ctx.get('frozen') or {}).get('sha256')}`。",
        f"CURRENT = **{promo.get('final_current')}**。decision = **{promo.get('decision')}**。",
        "",
        "## 01 · 使命与冻结",
        "",
        "parity → 风险诊断 → freeze → 候选。DE_RISKING_ONLY 不能升 v5。不得自动把 v5 写成 CURRENT。",
        "",
        "## 02 · Parity",
        "",
        f"总判定 `{(ctx.get('parity') or {}).get('status')}`。R14_00 必须精确对齐 R12_01 / R13_00。",
        "",
        "## 03 · 风险面板",
        "",
        f"{ctx.get('risk')}",
        "",
        "## 04 · 候选绩效（corrected）",
        "",
        "| ID | CAGR | Sharpe | MaxDD | 笔数 | LOYO med | LOYO worst |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for code in (R14_00, *CANDIDATE_IDS):
        row = res.get(code) or {}
        body.append(
            f"| `{code}` | {pct(row.get('cagr'))} | {row.get('sharpe')} | {pct(row.get('max_drawdown'))} | "
            f"{row.get('completed_trades')} | {pct(row.get('loyo_median'))} | {pct(row.get('loyo_worst'))} |"
        )
    body += [
        "",
        "## 05 · Exposure-matched 对照",
        "",
        f"关键 B02 vs CTRL_EXP_B02 = {(ctx.get('controls') or {}).get('key')}",
        f"comparisons={(ctx.get('controls') or {}).get('comparisons')}",
        "",
        "## 06 · 情景 / 执行 / 容量 / 尾部",
        "",
        f"S8=`{(ctx.get('scenarios') or {}).get('s8_rule')}` S9=`{(ctx.get('scenarios') or {}).get('s9')}`。",
        f"summary={(ctx.get('scenarios') or {}).get('summary')}",
        f"execution no_optimize={(ctx.get('execution') or {}).get('no_optimize')}；"
        f"capacity ratings={(ctx.get('capacity') or {}).get('ratings')}。",
        f"v4 top5 DD={(ctx.get('tail') or {}).get('v4_top5_drawdowns')}",
        "",
        "## 07 · Bootstrap / matched / CPCV / SPA",
        "",
        f"bootstrap N={(ctx.get('bootstrap') or {}).get('n')}；matched N={(ctx.get('matched') or {}).get('n')}；"
        f"SPA Hansen p={((ctx.get('spa') or {}).get('hansen_recentered') or {}).get('p_spa')}。",
        "",
        "## 08 · Forward",
        "",
        f"{(ctx.get('forward') or {}).get('status')}。{(ctx.get('forward') or {}).get('note')}",
        "",
        "## 09 · ALPHA vs DEPLOYMENT",
        "",
        "ALPHA_BASELINE = corrected-cost 单组合（R14_00 / 五候选）。",
        "DEPLOYMENT_P50/P75 不进入 Alpha lineage。",
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
        "- 不改宇宙、双头、种子、特征、标签、90/10、Top1/Top2 身份、d1_open_gap、B03 来源、进出场、市场风险层、Top1 confidence",
        "- 不搜阈值/窗口/组合；无笛卡尔积；无 10/12/15% 波动网格",
        "- 诊断不改公式；freeze 后新东西 = POST_HOC_DIAGNOSTIC",
        "- DE_RISKING_ONLY 不能升 v5；不自动把 v5 写成 CURRENT；追踪表不写「写入时间」",
        "",
    ]
    write_md(FINAL_REPORT, "\n".join(body))
    _append_tracker(ctx)
    complete = {
        "decision": promo.get("decision"),
        "final_current": promo.get("final_current"),
        "recommended_id": promo.get("recommended_id"),
        "lock_status": promo.get("lock_status"),
        "upgrade_type": promo.get("upgrade_type"),
        "reason": promo.get("reason"),
        "R14_00": res.get(R14_00),
        "candidates": {c: res.get(c) for c in CANDIDATE_IDS},
        "b02_vs_ctrl": (ctx.get("controls") or {}).get("key"),
        "forward_oos": (ctx.get("forward") or {}).get("status"),
        "manifest_sha256": (ctx.get("frozen") or {}).get("sha256"),
        "dossier": str(FINAL_REPORT),
        "questions": qs,
        "promotion": promo,
    }
    dump_json(DOSSIER_JSON, complete)
    return complete


def print_complete_block(payload: dict[str, Any]) -> None:
    r00 = payload.get("R14_00") or {}
    rec = payload.get("recommended_id")
    rec_row = (payload.get("candidates") or {}).get(rec) or {}
    print("========== ROUND14 COMPLETE ==========", flush=True)
    print(f"DECISION={payload.get('decision')}", flush=True)
    print(f"CURRENT={payload.get('final_current')}", flush=True)
    print(f"LOCK={payload.get('lock_status')}", flush=True)
    print(f"UPGRADE_TYPE={payload.get('upgrade_type')}", flush=True)
    print(
        f"R14_00 CAGR={r00.get('cagr')} Sharpe={r00.get('sharpe')} MaxDD={r00.get('max_drawdown')}",
        flush=True,
    )
    if rec:
        print(
            f"{rec} CAGR={rec_row.get('cagr')} Sharpe={rec_row.get('sharpe')} MaxDD={rec_row.get('max_drawdown')}",
            flush=True,
        )
    key = payload.get("b02_vs_ctrl") or {}
    print(f"B02_VS_CTRL={key.get('label')} P_OR_SHARPE={key.get('cand_sharpe')}/{key.get('ctrl_sharpe')}", flush=True)
    print(f"FORWARD={payload.get('forward_oos')}", flush=True)
    print(f"MANIFEST={payload.get('manifest_sha256')}", flush=True)
    print(f"DOSSIER={payload.get('dossier')}", flush=True)
    print(f"REASON={payload.get('reason')}", flush=True)
    print("Final Report: /home/mxy/Stock/课题/一进二策略/result/round14/ROUND14_RISK_BUDGET_REPORT.md", flush=True)
    print("======================================", flush=True)


__all__ = [
    "gather_ctx",
    "print_complete_block",
    "run_capacity",
    "run_execution",
    "run_forward",
    "run_promotion",
    "run_report",
    "run_tail",
]
