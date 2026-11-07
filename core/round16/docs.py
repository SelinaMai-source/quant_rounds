"""Forward、promotion、主报告 1–20、Round17 handoff、追踪表追加、PART 32。不写「写入时间」。"""

from __future__ import annotations

from typing import Any

from ..paths import MARKET_END, ROUND16_ROOT
from .audit import load_audit
from .backtest import load_backtest
from .common import dump_json, load_json, pct, write_md
from .contract import (
    AWAITING_USER_LOCK,
    FINAL_REPORT,
    FORWARD_OOS_NOT_AVAILABLE,
    HANDOFF,
    KEEP_BASELINE_V4,
    O01,
    O02,
    O03,
    O04,
    OPTION_IDS,
    PARENT_V4,
    RECOMMENDED_BASELINE_V5,
    TRACKER_PATH,
    V4_ANCHOR,
)
from .controls import load_controls
from .coverage import load_coverage
from .factors import load_factors
from .freeze import require_frozen
from .incremental import load_incremental
from .parity import load_parity_status
from .redundancy import load_redundancy
from .stats import load_cpcv, load_spa, load_stats
from .stress import load_stress

FWD_JSON = ROUND16_ROOT / "forward" / "forward.json"
FWD_MD = ROUND16_ROOT / "12_forward_oos.md"
PROMO_JSON = ROUND16_ROOT / "promotion" / "promotion.json"
PROMO_MD = ROUND16_ROOT / "13_baseline_decision.md"
DOSSIER_JSON = ROUND16_ROOT / "stats" / "dossier.json"


def run_forward(**_: Any) -> dict[str, Any]:
    require_frozen()
    payload = {
        "status": FORWARD_OOS_NOT_AVAILABLE,
        "cutoff": str(MARKET_END),
        "inbox_incomplete": True,
        "forward_validation_supported": False,
        "live_validated": False,
        "note": f"研究快照截止 {MARKET_END}。2026-07-25 之后无完整未参与研究新数据。不得把 CPCV 当 forward OOS。",
    }
    dump_json(FWD_JSON, payload)
    write_md(FWD_MD, f"# Round 16 · 12 Forward OOS\n\n**{FORWARD_OOS_NOT_AVAILABLE}**。{payload['note']}\n")
    return payload


def _gates(row: dict[str, Any], v4: dict[str, Any], ctrl: dict[str, Any], cpcv: dict[str, Any], spa: dict[str, Any]) -> dict[str, Any]:
    failed = []
    if row.get("skip") or row.get("cagr") is None:
        return {"passed": False, "failed": [str(row.get("status") or "SKIP")], "skip": True}
    cagr = float(row.get("cagr") or 0)
    sharpe = float(row.get("sharpe") or 0)
    mdd = float(row.get("max_drawdown") or 0)
    v_cagr = float(v4.get("cagr") or 0)
    v_sh = float(v4.get("sharpe") or 0)
    v_mdd = float(v4.get("max_drawdown") or 0)
    if cagr <= v_cagr:
        failed.append("cagr_gt_v4")
    if sharpe < v_sh:
        failed.append("sharpe_ge_v4")
    if (abs(mdd) - abs(v_mdd)) * 100 > 1.0:
        failed.append("mdd_worse")
    p = (ctrl or {}).get("p_cand_gt_ctrl")
    if p is not None and p < 0.60:
        failed.append("matched_control")
    return {"passed": not failed, "failed": failed, "skip": False, "cagr": cagr, "sharpe": sharpe, "max_drawdown": mdd}


def run_promotion() -> dict[str, Any]:
    require_frozen()
    bt = load_backtest()
    rows = bt.get("results") or {}
    v4 = dict(V4_ANCHOR)
    par = load_parity_status()
    obs = ((par.get("anchors") or {}).get("R16_00_CORRECTED_V4") or {}).get("observed") or {}
    if obs.get("cagr") is not None:
        v4 = {**v4, **obs}
    ctrl = (load_controls().get("results") or {})
    cpcv = load_cpcv()
    spa = load_spa()
    fac = load_factors()
    pit_ok = ((fac.get("pit") or {}).get("status") == "PASS")
    by = {}
    winners = []
    for cid in OPTION_IDS:
        g = _gates(rows.get(cid) or {}, v4, ctrl.get(cid) or {}, cpcv, spa)
        g["pit"] = pit_ok
        if not pit_ok:
            g["passed"] = False
            g["failed"] = list(g.get("failed") or []) + ["PIT"]
        by[cid] = g
        if g.get("passed"):
            winners.append(cid)
    if winners:
        rec = winners[0]
        decision = f"{RECOMMENDED_BASELINE_V5}={rec}"
        lock = AWAITING_USER_LOCK
        current = KEEP_BASELINE_V4
        reason = "过门候选仅推荐，CURRENT 仍 v4，等用户锁。"
    else:
        rec = None
        decision = KEEP_BASELINE_V4
        lock = None
        current = KEEP_BASELINE_V4
        reason = "无 Alpha 候选过 RETURN/对照/PIT 门。"
    payload = {
        "decision": decision,
        "final_current": current,
        "recommended_id": rec,
        "lock_status": lock,
        "by_id": by,
        "reason": reason,
        "auto_promote_v5": False,
    }
    dump_json(PROMO_JSON, payload)
    write_md(PROMO_MD, f"# Round16 · 13 Baseline\n\n**{decision}** CURRENT=**{current}** lock=`{lock}`\n\n{reason}\n")
    return payload


def _append_tracker(ctx: dict[str, Any]) -> None:
    promo = ctx.get("promotion") or {}
    audit = ctx.get("audit") or {}
    cov = ctx.get("coverage") or {}
    bt = (ctx.get("backtest") or {}).get("results") or {}
    rec = promo.get("recommended_id")
    rec_row = bt.get(rec) or {} if rec else {}
    block = [
        "",
        "## Round 16 — Option Data Recovery & Option-Implied Alpha",
        "",
        "期权数据恢复与预注册期权隐含 Alpha。不重训主 Alpha。不升 CURRENT v5。",
        f"promotion = `{promo.get('decision')}`。lock = `{promo.get('lock_status')}`。",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| Round15 data problem | n_kline_days=63 range=2026-04-23..2026-07-24 |",
        f"| Round16 root cause | {audit.get('verdict')} |",
        f"| historical access | {audit.get('history_status')} |",
        f"| option days before/after | 63 / {cov.get('n_valid_days')} |",
        f"| option contracts before/after | 172 / {cov.get('n_contracts')} |",
        f"| O01 | {(bt.get(O01) or {}).get('status')} |",
        f"| O02 | {(bt.get(O02) or {}).get('status')} |",
        f"| O03 | {(bt.get(O03) or {}).get('status')} |",
        f"| O04 | {(bt.get(O04) or {}).get('status')} |",
        f"| recommended | {rec or '—'} |",
        f"| recommended CAGR/Sharpe/MaxDD | {pct(rec_row.get('cagr'))} / {rec_row.get('sharpe')} / {pct(rec_row.get('max_drawdown'))} |",
        f"| lock | {promo.get('lock_status') or 'n/a'} |",
        f"| CURRENT | {promo.get('final_current')} |",
        f"| forward OOS | {(ctx.get('forward') or {}).get('status')} |",
        f"| matched controls | {((ctx.get('controls') or {}).get('perm_n'))} |",
        f"| manifest sha256 | `{(ctx.get('frozen') or {}).get('sha256')}` |",
        "",
        "主报告：`result/round16/ROUND16_OPTION_RESEARCH_REPORT.md`。",
        "Handoff：`result/round16/ROUND17_RESEARCH_HANDOFF.md`。",
        "",
    ]
    text = TRACKER_PATH.read_text(encoding="utf-8")
    if "## Round 16 — Option Data Recovery" in text:
        return
    TRACKER_PATH.write_text(text.rstrip() + "\n" + "\n".join(block), encoding="utf-8")


def _handoff(ctx: dict[str, Any]) -> None:
    inc = ctx.get("incremental") or {}
    red = ctx.get("redundancy") or {}
    fac = ctx.get("factors") or {}
    body = [
        "# ROUND17_RESEARCH_HANDOFF",
        "",
        "只交接，不搜索，不跑 Round17。",
        "",
        "## 1. 已验证期权因子是否有增量",
        "",
        f"incremental=`{inc.get('incremental')}` mean_R2=`{inc.get('mean_incremental_r2')}`。",
        "",
        "## 2. 与股票因子冗余",
        "",
        f"redundancy=`{red.get('verdict')}`。",
        "",
        "## 3. 是否纳入 Round17 股票 Alpha",
        "",
        "仅当 incremental 为 YES 且 redundancy 非 REDUNDANT 时值得考虑作为状态特征，而非本轮重训。",
        "",
        "## 4. 双头五 Seed 集成",
        "",
        "five-head fs00263 + ten-head 风险头；90/10 fusion；B03 allocator。Round16 未改。",
        "",
        "## 5. Seed 间预测相关",
        "",
        "沿用 Round7/9 已锁定 ensemble。本轮未重算 Seed 相关。",
        "",
        "## 6. Ensemble / 融合公式",
        "",
        "ensemble_rank_pct_5seed；target_name_budget = entry_NAV × RANK_NAV[rank] × allocation_multiplier；RANK_NAV=0.3/0.1。",
        "",
        "## 7. 已有 factor search 试验数",
        "",
        "first_board_factor_search_5000_v1 / fs00263。不要重搜 feature subset。",
        "",
        "## 8. Round17 应避免的失败实验",
        "",
        "- 不要再只用 get_option_code_list 当前合约当历史宇宙",
        "- 不要 kline_code_cap=400 截断",
        "- 不要把 hist_code_list 拿了却不下载 K 线",
        "- 不要搜 Top2 0/5/10、加 Seed、改 90/10",
        "- 不要把 CPCV 当 forward OOS",
        f"- 因子覆盖门失败则 SKIP：`{(fac.get('gates') or {})}`",
        "",
        "## 9. 值得再研究的股票 Alpha 特征族",
        "",
        "封板质量、尾盘量能、行业广度（已在 fs00263）。期权状态若 incremental=YES 可作为市场状态，不替代个股特征。",
        "",
        "## 10. 可复用数据与计算",
        "",
        f"- Round16 raw：`{ROUND16_ROOT}/data/raw`",
        f"- option_chain / IV / factors：`{ROUND16_ROOT}/data/processed`",
        "- 回测引擎：`core/round15/runner.py` run_r15_backtest",
        "",
        "禁止根据 Round16 结果提前做 Round17 搜索。",
        "",
    ]
    write_md(HANDOFF, "\n".join(body))


def gather_ctx() -> dict[str, Any]:
    from .coverage import load_coverage
    from .quality import load_quality
    from .iv import load_iv_status
    from .chain import load_chain_status
    from ..stock_readonly import sha256_file
    from .contract import MANIFEST_NAME

    frozen = {}
    mp = ROUND16_ROOT / MANIFEST_NAME
    if mp.is_file():
        frozen = {"sha256": sha256_file(mp), "path": str(mp)}
    return {
        "parity": load_parity_status(),
        "audit": load_audit(),
        "coverage": load_coverage(),
        "chain": load_chain_status(),
        "quality": load_quality(),
        "iv": load_iv_status(),
        "factors": load_factors(),
        "incremental": load_incremental(),
        "redundancy": load_redundancy(),
        "backtest": load_backtest(),
        "controls": load_controls(),
        "stress": load_stress(),
        "stats": load_stats(),
        "spa": load_spa(),
        "cpcv": load_cpcv(),
        "forward": load_json(FWD_JSON),
        "promotion": load_json(PROMO_JSON) or run_promotion(),
        "frozen": frozen,
    }


def run_report(**_: Any) -> dict[str, Any]:
    require_frozen()
    if not (ROUND16_ROOT / "forward" / "forward.json").is_file():
        run_forward()
    if not PROMO_JSON.is_file():
        run_promotion()
    ctx = gather_ctx()
    promo = run_promotion()
    ctx["promotion"] = promo
    audit = ctx.get("audit") or {}
    cov = ctx.get("coverage") or {}
    fac = ctx.get("factors") or {}
    bt = (ctx.get("backtest") or {}).get("results") or {}
    inc = ctx.get("incremental") or {}
    red = ctx.get("redundancy") or {}
    par = ctx.get("parity") or {}
    obs = ((par.get("anchors") or {}).get("R16_00_CORRECTED_V4") or {}).get("observed") or {}
    before = cov.get("before") or {"n_valid_days": 63, "n_contracts": 172, "n_quote_rows": 4929}
    quality = ctx.get("quality") or {}
    pit = (fac.get("pit") or {}).get("status")
    lines = [
        "# ROUND16_OPTION_RESEARCH_REPORT",
        "",
        "Round 16 = 期权数据恢复 × 预注册期权隐含 Alpha。不重训主 Alpha。CURRENT 不得自动升 v5。",
        f"manifest sha256=`{(ctx.get('frozen') or {}).get('sha256')}`。",
        f"parity=`{par.get('status')}`。裁决=**{promo.get('decision')}** CURRENT=**{promo.get('final_current')}**。",
        "",
        "## 1. Executive Verdict",
        "",
        f"根因 `{audit.get('verdict')}`。历史访问 `{audit.get('history_status')}`。",
        f"有效日 {before.get('n_valid_days')} → {cov.get('n_valid_days')}；合约 {before.get('n_contracts')} → {cov.get('n_contracts')}。",
        f"增量 `{inc.get('incremental')}`；冗余 `{red.get('verdict')}`。",
        f"O01=`{(bt.get(O01) or {}).get('status')}` O02=`{(bt.get(O02) or {}).get('status')}` "
        f"O03=`{(bt.get(O03) or {}).get('status')}` O04=`{(bt.get(O04) or {}).get('status')}`。",
        f"**{promo.get('decision')}**。{promo.get('reason')}",
        "",
        "## 2. Why Round15 Had Only 63 Option Days",
        "",
        audit.get("why_63_days") or "",
        "",
        f"代码位置：`{((audit.get('round15_facts') or {}).get('code_locations'))}`。",
        "不是 query_kline 把区间截成 63 天（请求 20150101–20260724）。",
        "",
        "## 3. How the Data Problem Was Fixed",
        "",
        "Round16 用 `get_hist_code_list(EXTRA_ETF_OP)` 多年窗口枚举历史合约，独立 `ad_local`，"
        "无 400 代码帽，标的含 588080.SH。过期合约探针见 00。若历史不可恢复则 blocked，不编造正收益。",
        "",
        "## 4. Remaining Vendor / Permission Limitations",
        "",
        f"permission_hit=`{audit.get('permission_hit')}` history_status=`{audit.get('history_status')}`。",
        "日 K 无 bid/ask；IV 标记 last_close_only。本报告不含 password/token。",
        "",
        "## 5. Option Historical Coverage",
        "",
        f"after days=`{cov.get('n_valid_days')}` contracts=`{cov.get('n_contracts')}` rows=`{cov.get('n_quote_rows')}` "
        f"range=`{cov.get('kline_min')}`→`{cov.get('kline_max')}` years=`{cov.get('year_span')}`。",
        "按 underlying×year 见 `01_option_data_coverage.md` 与 figures。",
        "",
        "## 6. Historical Contract Reconstruction",
        "",
        f"chain=`{(ctx.get('chain') or {})}`",
        "",
        "## 7. Option Quote Quality",
        "",
        f"quality=`{quality}` missing→0=False。无双边不把 last 当可靠 IV。",
        "",
        "## 8. PIT / Leakage Validation",
        "",
        f"PIT=`{pit}` tests=`{fac.get('pit')}`。Asia/Shanghai。任一 FAIL 不得进回测。",
        "",
        "## 9. Implied Volatility Construction",
        "",
        f"iv=`{ctx.get('iv')}` 失败≠0 vol。",
        "",
        "## 10. Option Factors",
        "",
        f"F1 ATM_IV30 / F2 PUT_SKEW25 / F3 PUT_WING_CONVEXITY / F4 IV_RV_SPREAD。gates=`{fac.get('gates')}`。",
        "滚动分位过去 252 有效日最少 126；不足 fallback v4。高阶矩 SKIP_ADVANCED_FACTORS。",
        "",
        "## 11. Equity-Factor Redundancy",
        "",
        f"`{red.get('verdict')}` mean_abs_oos_corr=`{red.get('mean_abs_oos_corr')}` full_sample=`{red.get('full_sample_residualization')}`。",
        "",
        "## 12. Incremental Predictive Evidence",
        "",
        f"`{inc.get('incremental')}` mean_incremental_R2=`{inc.get('mean_incremental_r2')}` in_sample_t_only=`{inc.get('in_sample_t_only')}`。",
        "",
        "## 13. O01–O04 Results",
        "",
        "| ID | status | n_events | CAGR | Sharpe | MaxDD |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for code in OPTION_IDS:
        row = bt.get(code) or {}
        lines.append(
            f"| `{code}` | {row.get('status')} | {row.get('n_events')} | {pct(row.get('cagr'))} | {row.get('sharpe')} | {pct(row.get('max_drawdown'))} |"
        )
    lines += [
        "",
        "## 14. Matched Controls",
        "",
        f"perm_n=`{(ctx.get('controls') or {}).get('perm_n')}` not_daily_exposure_match=True。",
        f"`{(ctx.get('controls') or {}).get('results')}`",
        "",
        "## 15. Execution and Cost Stress",
        "",
        f"`{(ctx.get('stress') or {}).get('status')}` +20/+30bp vs 同窗 v4。",
        "",
        "## 16. CPCV / Bootstrap / SPA",
        "",
        f"stats=`{(ctx.get('stats') or {}).get('status')}` SPA=`{((ctx.get('spa') or {}).get('status'))}` CPCV not_forward=True。",
        "",
        "## 17. Forward OOS",
        "",
        f"**{(ctx.get('forward') or {}).get('status')}**。cutoff=`{MARKET_END}`。不得标记 LIVE_VALIDATED。",
        "",
        "## 18. Baseline Promotion Decision",
        "",
        f"**{promo.get('decision')}** CURRENT=**{promo.get('final_current')}** lock=`{promo.get('lock_status')}`。",
        "过门只写 RECOMMENDED_BASELINE_V5 + AWAITING_USER_LOCK。禁止为出 v5 降数据质量门。",
        "",
        "## 19. Remaining Risks",
        "",
        "- last_close_only IV，无历史买卖报价",
        "- 覆盖不足则 SKIP_FACTOR_FOR_TRADING",
        "- survivorship 未在本轮量化",
        "- 2018–2026 是 HISTORICAL DEVELOPMENT DATA",
        "",
        "## 20. Implications for Round17",
        "",
        "见 ROUND17_RESEARCH_HANDOFF.md。不在本轮搜索。",
        "",
        "### 明确没做",
        "",
        "- 不重训 five/ten、不搜 feature、不改 90/10、不加 Seed、不搜 Top2 0/5/10",
        "- 无 O05+；不看完收益再改阈值",
        "- 不自动把 v5 写成 CURRENT；追踪表不写「写入时间」",
        "- 不发明期权数据；SKIP 不是 PASS；不跑 Round17",
        "",
    ]
    write_md(FINAL_REPORT, "\n".join(lines))
    _append_tracker(ctx)
    _handoff(ctx)
    complete = {
        "decision": promo.get("decision"),
        "final_current": promo.get("final_current"),
        "recommended_id": promo.get("recommended_id"),
        "lock_status": promo.get("lock_status"),
        "reason": promo.get("reason"),
        "parity": par.get("status"),
        "verdict": audit.get("verdict"),
        "history_status": audit.get("history_status"),
        "n_days_before": 63,
        "n_days_after": cov.get("n_valid_days"),
        "n_contracts_before": 172,
        "n_contracts_after": cov.get("n_contracts"),
        "pit": pit,
        "redundancy": red.get("verdict"),
        "incremental": inc.get("incremental"),
        "options": {c: bt.get(c) for c in OPTION_IDS},
        "forward_oos": (ctx.get("forward") or {}).get("status"),
        "manifest_sha256": (ctx.get("frozen") or {}).get("sha256"),
        "dossier": str(FINAL_REPORT),
        "handoff": str(HANDOFF),
        "R16_00": obs,
        "gates": fac.get("gates"),
        "quality": quality,
        "promotion": promo,
        "parent": PARENT_V4,
    }
    dump_json(DOSSIER_JSON, complete)
    return complete


def print_complete_block(payload: dict[str, Any]) -> None:
    bt = payload.get("options") or {}
    gates = payload.get("gates") or {}
    def _m(code):
        row = bt.get(code) or {}
        if row.get("skip") or row.get("cagr") is None:
            return str(row.get("status") or "SKIP")
        return f"CAGR={row.get('cagr')} Sharpe={row.get('sharpe')} MaxDD={row.get('max_drawdown')}"

    print("ROUND16 COMPLETE", flush=True)
    print("", flush=True)
    print("Current baseline:", flush=True)
    print("baseline_v4 = R9_B03", flush=True)
    print("", flush=True)
    print("Baseline parity:", flush=True)
    print(payload.get("parity") or "FAIL", flush=True)
    print("", flush=True)
    print("Round15 option problem:", flush=True)
    print(payload.get("verdict") or "H.ROOT_CAUSE_NOT_IDENTIFIED", flush=True)
    print("", flush=True)
    print("AmazingData historical access:", flush=True)
    print(payload.get("history_status") or "UNAVAILABLE", flush=True)
    print("", flush=True)
    print("Option valid days before:", flush=True)
    print("63", flush=True)
    print("", flush=True)
    print("Option valid days after:", flush=True)
    print(payload.get("n_days_after"), flush=True)
    print("", flush=True)
    print("Option contracts before:", flush=True)
    print("172", flush=True)
    print("", flush=True)
    print("Option contracts after:", flush=True)
    print(payload.get("n_contracts_after"), flush=True)
    print("", flush=True)
    print("Valid ATM IV30:", flush=True)
    print((gates.get("F1") or {}).get("n_valid_days"), flush=True)
    print("", flush=True)
    print("Valid 25D skew:", flush=True)
    print((gates.get("F2") or {}).get("n_valid_days"), flush=True)
    print("", flush=True)
    print("Valid convexity:", flush=True)
    print((gates.get("F3") or {}).get("n_valid_days"), flush=True)
    print("", flush=True)
    q = payload.get("quality") or {}
    qstat = "PASS" if float(q.get("valid_quote_rate") or 0) >= 0.8 else ("PARTIAL" if q else "FAIL")
    print("Option data quality:", flush=True)
    print(qstat, flush=True)
    print("", flush=True)
    print("PIT:", flush=True)
    print(payload.get("pit") or "FAIL", flush=True)
    print("", flush=True)
    print("Equity-option redundancy:", flush=True)
    print(payload.get("redundancy") or "INCONCLUSIVE", flush=True)
    print("", flush=True)
    print("Incremental option information:", flush=True)
    print(payload.get("incremental") or "INCONCLUSIVE", flush=True)
    print("", flush=True)
    print("R16_O01:", flush=True)
    print(_m(O01), flush=True)
    print("", flush=True)
    print("R16_O02:", flush=True)
    print(_m(O02), flush=True)
    print("", flush=True)
    print("R16_O03:", flush=True)
    print(_m(O03), flush=True)
    print("", flush=True)
    print("R16_O04:", flush=True)
    print(_m(O04), flush=True)
    print("", flush=True)
    print("Best candidate:", flush=True)
    print(payload.get("recommended_id") or "NONE", flush=True)
    print("", flush=True)
    print("Matched-control evidence:", flush=True)
    print((payload.get("promotion") or {}).get("reason") or "n/a", flush=True)
    print("", flush=True)
    print("Family selection correction:", flush=True)
    spa = load_spa()
    print(spa.get("status") or "SKIP_NO_ALTS", flush=True)
    print("", flush=True)
    print("Forward OOS:", flush=True)
    print("NOT_AVAILABLE" if "NOT_AVAILABLE" in str(payload.get("forward_oos") or "") else payload.get("forward_oos"), flush=True)
    print("", flush=True)
    print("Baseline decision:", flush=True)
    dec = payload.get("decision") or KEEP_BASELINE_V4
    print("RECOMMENDED_BASELINE_V5" if str(dec).startswith("RECOMMENDED") else "KEEP_BASELINE_V4", flush=True)
    print("", flush=True)
    print("Final report:", flush=True)
    print(str(FINAL_REPORT), flush=True)
    print("", flush=True)
    print("Round17 handoff:", flush=True)
    print(str(HANDOFF), flush=True)
    print("", flush=True)
    print(f"Final Report: {FINAL_REPORT}", flush=True)
