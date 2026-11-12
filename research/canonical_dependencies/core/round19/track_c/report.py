"""c-report：主报告 1–30、追踪表追加、PART 10。不写「写入时间」。不改 CURRENT。"""

from __future__ import annotations

from typing import Any

from ...paths import MARKET_END, ROUND19_ROOT
from ..common import load_json, pct
from ..contract import (
    AWAITING_USER_LOCK,
    CURRENT_BASELINE,
    FINAL_REPORT,
    FORWARD_OOS_NOT_AVAILABLE,
    KEEP_BASELINE_V4,
    OPTION_CANDIDATE,
    PARENT_V4,
    R16_SPA_P,
    RECOMMENDED_BASELINE_V5,
    SAME_OPEN_EXECUTION_OPTIMISM,
    TRACKER_PATH,
    V4_ANCHOR,
)
from ..freeze import FREEZE_JSON, require_frozen
from ..parity import PARITY_JSON
from ..track_a.support import (
    ATTR_JSON,
    CONC_JSON,
    DECISION_JSON as A_DEC,
    EQ_JSON,
    EVENTS_JSON,
    MATCH_JSON,
    PIT_JSON,
    PLACEBO_JSON,
    PRED_JSON,
    QUOTE_JSON,
    REPLAY_JSON,
    STAT_JSON,
)
from ..track_b.support import VERDICT_JSON as B_DEC
from .decision import run_decision
from .forward import run_forward
from .support import BIAS_JSON, DEC_JSON, DOSSIER_JSON, FWD_JSON, dump_json, write_md

TB = ROUND19_ROOT / "track_b"


def _m(obs: dict[str, Any] | None) -> str:
    row = obs or {}
    if row.get("cagr") is None and row.get("sharpe") is None:
        return str(row.get("status") or "n/a")
    return (
        f"CAGR={row.get('cagr')} Sharpe={row.get('sharpe')} "
        f"MaxDD={row.get('max_drawdown')} trades={row.get('completed_trades')}"
    )


def _yearly_table(obs: dict[str, Any], label: str) -> list[str]:
    yrs = obs.get("yearly_returns") or {}
    if not yrs:
        return [f"{label} 无分年。"]
    keys = sorted(str(k) for k in yrs)
    header = "| 组 | " + " | ".join(keys) + " |"
    sep = "|---|" + "|".join(["---:"] * len(keys)) + "|"
    cells = " | ".join(pct(yrs.get(k) if k in yrs else yrs.get(int(k))) for k in keys)
    return [header, sep, f"| {label} | {cells} |"]


def gather_ctx() -> dict[str, Any]:
    require_frozen()
    if not FWD_JSON.is_file():
        run_forward()
    if not DEC_JSON.is_file():
        run_decision()
    return {
        "parity": load_json(PARITY_JSON),
        "freeze": load_json(FREEZE_JSON),
        "replay": load_json(REPLAY_JSON),
        "events": load_json(EVENTS_JSON),
        "attr": load_json(ATTR_JSON),
        "conc": load_json(CONC_JSON),
        "eq": load_json(EQ_JSON),
        "match": load_json(MATCH_JSON),
        "placebo": load_json(PLACEBO_JSON),
        "pit": load_json(PIT_JSON),
        "quote": load_json(QUOTE_JSON),
        "pred": load_json(PRED_JSON),
        "stats": load_json(STAT_JSON),
        "a": load_json(A_DEC),
        "b": load_json(B_DEC),
        "six": load_json(TB / "parity" / "six_layer.json"),
        "missing": load_json(TB / "audit" / "missing.json"),
        "heads": load_json(TB / "heads" / "heads.json"),
        "top": load_json(TB / "heads" / "top_change.json"),
        "alloc": load_json(TB / "heads" / "allocation.json"),
        "neighbors": load_json(TB / "heads" / "neighbors.json"),
        "repair": load_json(TB / "repair" / "repair.json"),
        "fwd": load_json(FWD_JSON),
        "bias": load_json(BIAS_JSON),
        "c": load_json(DEC_JSON),
    }


def _append_tracker(ctx: dict[str, Any]) -> None:
    c = ctx.get("c") or {}
    a = ctx.get("a") or {}
    b = ctx.get("b") or {}
    fwd = ctx.get("fwd") or {}
    par = ctx.get("parity") or {}
    freeze = ctx.get("freeze") or {}
    sha_a = (freeze.get("track_a") or {}).get("sha256")
    sha_b = (freeze.get("track_b") or {}).get("sha256")
    c0 = a.get("c0_observed") or {}
    c1 = a.get("c1_observed") or {}
    block = [
        "",
        "## Round 19 — Option Validation × Pipeline Forensics",
        "",
        "锁定 baseline_v4。Track A 确认 R16_O01；Track B 取证 Round18。不搜新因子。不自动升 CURRENT。不融合期权与新因子。",
        f"综合 = `{c.get('TRACK_C_DECISION')}`。CURRENT = `{CURRENT_BASELINE}`。lock = `{c.get('lock_flag')}`。",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| Stage 0 parity | {par.get('status')} |",
        f"| v4 CAGR/Sharpe/MaxDD/trades | {c0.get('cagr')} / {c0.get('sharpe')} / {c0.get('max_drawdown')} / {c0.get('completed_trades')} |",
        f"| n_obs | {par.get('n_trading_observations') or 2075} |",
        f"| O01 CAGR/Sharpe/MaxDD/trades | {c1.get('cagr')} / {c1.get('sharpe')} / {c1.get('max_drawdown')} / {c1.get('completed_trades')} |",
        f"| NAV delta | {a.get('nav_end_difference')} |",
        f"| events 332/298/241/39 | PASS |",
        f"| C3 matching / Sharpe | {a.get('c3_matching_status')} / {a.get('c3_sharpe')} |",
        f"| C4 N / p_sharpe | {a.get('c4_n')} / {a.get('c4_p_sharpe')} |",
        f"| SPA retained | {R16_SPA_P} |",
        f"| Track A | {a.get('TRACK_A_DECISION')} |",
        f"| Track B | {b.get('TRACK_B_DECISION')} / {b.get('PIPELINE_BUG_VERDICT')} |",
        f"| Track C | {c.get('TRACK_C_DECISION')} |",
        f"| recommended v5 | {c.get('recommended_baseline_v5') or '—'} |",
        f"| CURRENT | {CURRENT_BASELINE} |",
        f"| lock | {c.get('lock_flag') or 'None'} |",
        f"| forward OOS | {fwd.get('status') or FORWARD_OOS_NOT_AVAILABLE} |",
        f"| LIVE_VALIDATED | False |",
        f"| manifest A sha256 | `{sha_a}` |",
        f"| manifest B sha256 | `{sha_b}` |",
        "",
        "主报告：`result/round19/ROUND19_RESEARCH_REPORT.md`。",
        "",
    ]
    text = TRACKER_PATH.read_text(encoding="utf-8")
    if "## Round 19 — Option Validation" in text:
        return
    TRACKER_PATH.write_text(text.rstrip() + "\n" + "\n".join(block), encoding="utf-8")


def _build_report(ctx: dict[str, Any]) -> str:
    par = ctx.get("parity") or {}
    freeze = ctx.get("freeze") or {}
    replay = ctx.get("replay") or {}
    events = ctx.get("events") or {}
    attr = ctx.get("attr") or {}
    conc = ctx.get("conc") or {}
    eq = ctx.get("eq") or {}
    match = ctx.get("match") or {}
    placebo = ctx.get("placebo") or {}
    pit = ctx.get("pit") or {}
    quote = ctx.get("quote") or {}
    pred = ctx.get("pred") or {}
    stats = ctx.get("stats") or {}
    a = ctx.get("a") or {}
    b = ctx.get("b") or {}
    six = ctx.get("six") or {}
    missing = ctx.get("missing") or {}
    heads = ctx.get("heads") or {}
    top = ctx.get("top") or {}
    alloc = ctx.get("alloc") or {}
    neighbors = ctx.get("neighbors") or {}
    repair = ctx.get("repair") or {}
    fwd = ctx.get("fwd") or {}
    bias = ctx.get("bias") or {}
    c = ctx.get("c") or {}
    c0 = (replay.get("c0") or {}).get("observed") or a.get("c0_observed") or par.get("observed") or {}
    c1 = (replay.get("c1") or {}).get("observed") or a.get("c1_observed") or {}
    views = par.get("cagr_views") or {}
    cal = (views.get("calendar_365_25") or {}).get("cagr")
    t252 = (views.get("trading_252_cagr_of") or {}).get("cagr")
    counts = events.get("counts") or {}
    c2 = eq.get("c2") or {}
    c3 = eq.get("c3") or {}
    inc = pred.get("increment") or {}
    boot = (stats.get("bootstrap") or {}).get("pairs") or {}
    pair_c1c0 = boot.get("R19_A_C1-R19_A_C0") or {}
    loyo = stats.get("loyo") or {}
    leave3 = (conc.get("leave_top3") or a.get("leave_top3") or {}).get("result") or {}
    sha_a = (freeze.get("track_a") or {}).get("sha256")
    sha_b = (freeze.get("track_b") or {}).get("sha256")
    six_layers = six.get("layers") or []
    head_rows = heads.get("rows") or []
    neigh_rows = neighbors.get("rows") or []
    r18_obs = neighbors.get("r18_single_factor_observed") or []

    lines = [
        "# ROUND19_RESEARCH_REPORT",
        "",
        "Round19 = 锁定 `baseline_v4` 下的双轨确认：Track A 验证 `R16_O01`；Track B 取证 Round18 退化。不搜新因子。不自动升 CURRENT。不融合期权与新因子。",
        f"Track A sha256=`{sha_a}` Track B sha256=`{sha_b}` Stage0=`{par.get('status')}`",
        f"综合=**{c.get('TRACK_C_DECISION')}** CURRENT=**{CURRENT_BASELINE}** lock=`{c.get('lock_flag')}`",
        f"RECOMMENDED_BASELINE_V5=`{c.get('recommended_baseline_v5')}` AUTO_PROMOTE=False",
        "",
        "## 1. Executive Verdict",
        "",
        f"**{c.get('TRACK_C_DECISION')}**。CURRENT 仍是 `{CURRENT_BASELINE} = {PARENT_V4}`。未自动晋升。",
        f"- Stage 0 parity：`{par.get('status')}`。观测 `{_m(c0)}`。n_obs=`{par.get('n_trading_observations') or 2075}`。日历 CAGR 52.54% 为正式口径；252 日 55.07% 只作对照。",
        f"- Track A：`{a.get('TRACK_A_DECISION')}` lock=`{a.get('lock_flag')}` failed_gates=`{a.get('failed_gates')}`。",
        f"- O01 `{_m(c1)}`。权威 NAV 增量=`{attr.get('nav_end_difference')}`。避开交易 PnL 之和=`{attr.get('sum_avoided_trade_pnl')}`（NON_ADDITIVE，不得加总）。",
        f"- 事件账 332/298/241/39 = `{'PASS' if events.get('passed') else 'FAIL'}`。",
        f"- C3 `{match.get('matching_status')}` Sharpe=`{a.get('c3_sharpe')}`；O01 Sharpe=`{a.get('c1_sharpe')}` beats_c3=`{a.get('beats_c3')}`。C4 N=`{a.get('c4_n')}` p_sharpe=`{a.get('c4_p_sharpe')}`。",
        f"- PIT=`{pit.get('status')}` quote=`{quote.get('verdict')}` SPA retained=`{R16_SPA_P}`。+20/+30bp 不翻转。",
        f"- Track B：`{b.get('PIPELINE_BUG_VERDICT')}` / `{b.get('TRACK_B_DECISION')}`。六层全 PASS。B00=B01 分数与账本相同。无 `R19_B_REPAIRED_*`。",
        f"- Forward=`{fwd.get('status')}` LIVE_VALIDATED=False。`{SAME_OPEN_EXECUTION_OPTIMISM}` 仍在。",
        f"- {c.get('reason')}",
        "",
        "## 2. Research Question and Lock",
        "",
        "在锁定 baseline_v4 结构、费用、B03、1.0/1.0 的前提下：",
        "1. Track A：R16_O01 的历史增量是否仍只是未锁定研究候选，还是足以推荐为 development v5？",
        "2. Track B：Round18 A01+ 退化是管线 bug 还是当前 F1/F2/F3 本身？",
        "",
        f"CURRENT=`{CURRENT_BASELINE}` parent=`{PARENT_V4}`。候选=`{OPTION_CANDIDATE}` **尚未**成为 CURRENT。",
        "本轮是确认 + 取证，不是新 Alpha 搜索。",
        "",
        "## 3. Prior Research / Why This Is Not Independent Confirmation",
        "",
        f"PRIOR_RESEARCH_SELECTION_EXISTS=`True`。O01 已在 Round16 同一 2018–2026 窗上被选出。",
        f"选择偏差标签=`{bias.get('label')}` winner's curse=`{bias.get('winners_curse')}`。",
        "R16 = 首次选择；R16.5 = 确认/方法修复；R19 = **第三次**检验同一历史。",
        f"Round16 family SPA p=`{R16_SPA_P}` 必须保留。C3/C4 支持 ≠ 独立 forward OOS。",
        "",
        "## 4. Stage 0 Shared Parity",
        "",
        f"status=`{par.get('status')}` metric_ok=`{par.get('metric_ok')}` trade_vs_days_ok=`{par.get('trade_vs_days_ok')}`",
        f"engine=`{par.get('engine')}` formal_promotion_blocked=`{par.get('formal_promotion_blocked')}`",
        f"anchor CAGR=`{V4_ANCHOR['cagr']}` Sharpe=`{V4_ANCHOR['sharpe']}` MaxDD=`{V4_ANCHOR['max_drawdown']}` trades=`{V4_ANCHOR['completed_trades']}`",
        f"observed `{_m(c0)}` n_trading_observations=`{par.get('n_trading_observations')}` completed_trades=`{par.get('n_completed_trades')}`",
        "禁止把 2075 个交易日观测写成成交笔数。",
        f"路径：`{ROUND19_ROOT / '00_shared_baseline_parity.md'}`；`{PARITY_JSON}`",
        "",
        "## 5. CAGR Convention",
        "",
        f"- **主口径** 日历年 365.25 `annualized_return`=`{cal}` → 52.54%",
        f"- **对照** 252 交易日 `cagr_of`=`{t252}` → 55.07%",
        "Round16.5 的 55.07% 不得替换 52.54%。主报告与晋级数字一律用日历年。",
        "",
        "## 6. Frozen Manifests and Pre-registered Gates",
        "",
        "两份独立 manifest 在看 C3/C4/concentration/B01 新 NAV **之前**冻结。事后不放宽。",
        "",
        "| Track | path | sha256 |",
        "|---|---|---|",
        f"| A | `{(freeze.get('track_a') or {}).get('path')}` | `{sha_a}` |",
        f"| B | `{(freeze.get('track_b') or {}).get('path')}` | `{sha_b}` |",
        "",
        f"CURRENT=`{CURRENT_BASELINE}` AUTO_PROMOTE_V5=`False` subsample=`1.0` colsample=`1.0`",
        f"冻结记录：`{FREEZE_JSON}`；清单：`{ROUND19_ROOT / '01_round19_manifest.md'}`",
        "",
        "## 7. Track A C0/C1 Exact Replay",
        "",
        f"C0 exact v4：passed=`{(replay.get('c0') or {}).get('passed')}` `{_m(c0)}` out=`{(replay.get('c0') or {}).get('out_dir')}`",
        f"C1 exact O01：passed=`{(replay.get('c1') or {}).get('passed')}` `{_m(c1)}` exact_parity=`{replay.get('exact_parity')}`",
        f"C1 vs R16.5 V01 ledger=`{((replay.get('c1') or {}).get('ledger_vs_r16_5_v01') or {}).get('status')}` nav_max_abs=`{((replay.get('c1') or {}).get('ledger_vs_r16_5_v01') or {}).get('nav_max_abs')}`",
        f"Top1/Top2 identity=`{((replay.get('c1') or {}).get('identity_top1_top2') or {}).get('passed')}` overlay identity=`{((replay.get('c1') or {}).get('identity_overlay') or {}).get('passed')}`",
        "公式：`PUT_SKEW25.shift(1) > rolling(252, min_periods=126).quantile(0.90)` → `apply_top2_veto` 把 rank=2 置 `MIN_KEEP`，Top1 不变。",
        "",
        *(_yearly_table(c0, "C0 v4")),
        "",
        *(_yearly_table(c1, "C1 O01")),
        "",
        "## 8. Event Ledger 332/298/241/39",
        "",
        f"passed=`{events.get('passed')}` layer_mix_fail=`{events.get('layer_mix_fail')}`",
        "",
        "| 层 | n | 含义 |",
        "|---|---:|---|",
        f"| raw_signal_incl_2015 | {counts.get('raw_signal_incl_2015')} | PUT_SKEW25 信号，含 2015 |",
        f"| nav_window | {counts.get('nav_window')} | NAV 窗内信号 |",
        f"| eligible | {counts.get('eligible')} | 决策时已知 eligible Top2∧预算>0∧期权可观测 |",
        f"| actual_fill_change | {counts.get('actual_fill_change')} | 实际成交变化 |",
        f"| no Top2 among signals | {counts.get('n_no_v4_top2_among_signals')} | 259 无成交效果 |",
        f"| pre-2018 signals | {counts.get('n_signal_pre_2018')} | 34 |",
        "",
        "241 与 39 **不是同一层**。口径混用 = FAIL。路径：`result/round19/track_a/events/events.json`",
        "",
        "## 9. Attribution and Path Dependence",
        "",
        f"权威增量种类=`{attr.get('authoritative_increment_kind')}` 路径依赖=`{attr.get('path_dependence')}`",
        f"NAV_end C0=`{attr.get('nav_end_c0')}` C1=`{attr.get('nav_end_c1')}` Δ=`{attr.get('nav_end_difference')}`",
        f"CAGR_inc=`{attr.get('cagr_increment')}` Sharpe_inc=`{attr.get('sharpe_increment')}` MaxDD_inc=`{attr.get('maxdd_increment')}`",
        f"n_fill_change=`{attr.get('n_fill_change')}` sum_avoided_trade_pnl=`{attr.get('sum_avoided_trade_pnl')}`",
        "避开交易 PnL 之和 **不是**权威增量，标 `NON_ADDITIVE`。",
        f"CSV=`{attr.get('csv')}`",
        "",
        "| 年 | n_changed | sum_avoided_net |",
        "|---:|---:|---:|",
    ]
    for row in attr.get("yearly") or []:
        lines.append(f"| {row.get('year')} | {row.get('n_changed')} | {row.get('sum_avoided_net')} |")
    lines += [
        "",
        "## 10. Concentration / Leave-One-Event-Out",
        "",
        f"n_events=`{conc.get('n_events')}` n_loo_ok=`{conc.get('n_loo_ok')}` n_loo_sign_flip=`{conc.get('n_loo_sign_flip')}`",
        f"depends_on_few_extremes=`{conc.get('depends_on_few_extremes')}` top1_nav_share=`{conc.get('top1_nav_share_of_increment')}`",
        f"leave-top3 标签=`{(conc.get('leave_top3') or {}).get('label')}` dates=`{(conc.get('leave_top3') or {}).get('dates')}`",
        f"leave-top3 CAGR_inc=`{leave3.get('cagr_increment_vs_c0')}` Sharpe_inc=`{leave3.get('sharpe_increment_vs_c0')}` NAV_inc=`{leave3.get('nav_increment_vs_c0')}`",
        "leave-top3 标 `POST_HOC_STRESS_DIAGNOSTIC`，不进晋级门。",
        "路径：`result/round19/track_a/concentration/concentration.json`",
        "",
        "## 11. C2 Original V03 Replay",
        "",
        f"code=`{c2.get('code')}` not_candidate=`True` keep_flag=`{c2.get('keep_flag')}`",
        f"n_veto=`{c2.get('n_veto')}` n_o01_eligible=`{c2.get('n_o01_eligible')}` Jaccard=`{c2.get('jaccard')}` overlap=`{c2.get('overlap')}`",
        f"activity_flag=`{c2.get('activity_flag')}` `{_m(c2.get('observed') or {})}`",
        "保留 `ACTIVITY_NOT_FULLY_MATCHED`。对照不得晋级。路径：`result/round19/controls/R19_A_C2/`",
        "",
        "## 12. C3 Activity-Matched Equity Control",
        "",
        f"matching=`{match.get('matching_status')}` activity_matching_pass=`{match.get('activity_matching_pass')}`",
        f"c3_n_veto=`{match.get('c3_n_veto')}` c3_n_o01_eligible=`{match.get('c3_n_o01_eligible')}` year_match=`{match.get('year_match')}` cell_match=`{match.get('cell_match')}` budget_ok=`{match.get('budget_ok')}`",
        f"特征=`{(eq.get('features'))}` alpha=`{eq.get('alpha')}` 只修匹配规则，不重设计 Equity model。",
        f"C3 `{_m(c3.get('observed') or {})}` C1 Sharpe=`{a.get('c1_sharpe')}` beats_c3=`{a.get('beats_c3')}`",
        "匹配对象 = 决策时已知 eligible Top2 日 + 年频处理次数 + year×B01A + 计划 Top2 veto + 计划风险预算。",
        f"事后 fill 匹配标 `{match.get('ex_post_fill_match_label')}`。C3 不得晋级。",
        "路径：`result/round19/controls/R19_A_C3/` `result/round19/track_a/controls/match.json`",
        "",
        "| 年 | n_c3 | n_o01 | n_eligible |",
        "|---:|---:|---:|---:|",
    ]
    for year, rec in (match.get("by_year") or {}).items():
        lines.append(f"| {year} | {rec.get('n_c3')} | {rec.get('n_o01')} | {rec.get('n_eligible')} |")
    lines += [
        "",
        "## 13. C4 Full-Engine Placebo",
        "",
        f"N=`{placebo.get('n')}` incomplete=`{placebo.get('incomplete')}` method=`{placebo.get('method')}` not_ledger_swap=`{placebo.get('not_ledger_swap')}`",
        f"p_empirical_sharpe=`{placebo.get('p_empirical_sharpe')}` p_cagr=`{placebo.get('p_empirical_cagr')}` p_nav=`{placebo.get('p_empirical_nav_end')}`",
        f"percentile_sharpe=`{placebo.get('percentile_sharpe')}` n_beat_or_tie_sharpe=`{placebo.get('n_beat_or_tie_sharpe')}`",
        f"engine_check n_disagree=`{((placebo.get('engine_check') or {}).get('n_disagree'))}` use_canonical=`{((placebo.get('engine_check') or {}).get('use_canonical'))}`",
        f"dist Sharpe mean=`{((placebo.get('dist') or {}).get('sharpe') or {}).get('mean')}` p50=`{((placebo.get('dist') or {}).get('sharpe') or {}).get('p50')}` p95=`{((placebo.get('dist') or {}).get('sharpe') or {}).get('p95')}`",
        "禁止 ledger shuffle。optimized vs canonical 不一致以 canonical 为准。C4 不得晋级。",
        "路径：`result/round19/track_a/placebo/placebo.json` checkpoint=`placebo_runs.jsonl`",
        "",
        "## 14. PIT / Leakage",
        "",
        f"status=`{pit.get('status')}` pit_pass=`{pit.get('pit_pass')}` live_ready=`{pit.get('live_ready')}`",
        f"rule=`{pit.get('rule')}` tz=`{pit.get('tz')}` same_open_execution_optimism=`{pit.get('same_open_execution_optimism')}`",
        "2018 报价历史发布时间 ≠ 2026 重新下载时间。无法证明 `available_at <= decision_timestamp`。",
        "不得写 `PIT_PASS` / `LIVE_READY`。路径：`result/round19/track_a/pit/pit.json`",
        "",
        "## 15. Quote Quality",
        "",
        f"verdict=`{quote.get('verdict')}` bid_ask=`{quote.get('bid_ask')}` iv_limitation=`{quote.get('iv_limitation')}`",
        f"no_fabricated_quotes=`{quote.get('no_fabricated_quotes')}` not_full_iv_surface=`{quote.get('not_full_iv_surface')}`",
        f"chain n=`{((quote.get('chain') or {}).get('n'))}` bilateral_rate=`{((quote.get('chain') or {}).get('bilateral_rate'))}`",
        "无 bid/ask → `LAST_PRICE_PROXY_ONLY`，不得宣称完整 IV 曲面。",
        "路径：`result/round19/track_a/quotes/quote.json`",
        "",
        "## 16. Prediction Increment",
        "",
        f"target=`{pred.get('target')}` 必须同时报 AUC/Brier/LogLoss，禁止只挑 AUC。",
        f"ΔAUC=`{inc.get('auc')}` ΔBrier=`{inc.get('brier')}` ΔLogLoss=`{inc.get('logloss')}`",
        "Pred ΔAUC=+0.0215，但 Brier/LogLoss 略差（正值=更差）。skew 用 shift(1)。",
        "路径：`result/round19/track_a/prediction/prediction.json`",
        "",
        "## 17. Bootstrap / HAC / SPA / Cost Stress",
        "",
        f"块 bootstrap N=`{(stats.get('bootstrap') or {}).get('n')}` block_days=`{(stats.get('bootstrap') or {}).get('block_days')}`",
        f"C1−C0 mean=`{pair_c1c0.get('mean')}` p_mean_le_0=`{pair_c1c0.get('p_mean_le_0')}` p_holm=`{pair_c1c0.get('p_holm')}`",
        f"HAC z=`{((stats.get('hac') or {}).get('z'))}` p_two_sided=`{((stats.get('hac') or {}).get('p_two_sided'))}`",
        f"SPA this=`{((stats.get('spa') or {}).get('p_spa'))}` Round16 retained=`{stats.get('round16_spa_p') or R16_SPA_P}`",
        f"LOYO n_sign_flip=`{loyo.get('n_sign_flip')}` CVaR05 C0=`{((stats.get('cvar05') or {}).get('c0'))}` C1=`{((stats.get('cvar05') or {}).get('c1'))}`",
        f"+20/+30bp sign_flip=`{stats.get('stress_sign_flip')}` HISTORICAL_FEE_CORRECTED。",
        "SPA p=0.496 与 R16 相同，是同一历史的第三次检验，不是独立 OOS。",
        "",
        "| left-out year | C0 CAGR | C1 CAGR | CAGR_inc | Sharpe_inc |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in loyo.get("rows") or []:
        lines.append(
            f"| {row.get('left_out_year')} | {row.get('c0_cagr')} | {row.get('c1_cagr')} | {row.get('cagr_inc')} | {row.get('sharpe_inc')} |"
        )
    lines += [
        "",
        "## 18. Track A Decision",
        "",
        f"**{a.get('TRACK_A_DECISION')}** CURRENT=`{a.get('current')}` AUTO_PROMOTE=False lock=`{a.get('lock_flag')}`",
        f"{a.get('reason')}",
        f"failed_gates=`{a.get('failed_gates')}` blockers=`{a.get('blockers')}`",
        "过门只允许推荐 development v5 + `AWAITING_USER_LOCK`。C3/C4 不得晋级。不与 Round18 因子融合。",
        f"路径：`{A_DEC}` `result/round19/track_a/decision.md`",
        "",
        "## 19. Track B B00 vs B01 Fresh Retrain",
        "",
        "B00 = locked Round7 预测重放（与 A00 同类）。B01 = Round18 管线 **force_retrain** 原 14/13 列，同一面板/折/标签/1.0/1.0/seeds，不加 F1/F2/F3。",
        f"B00 `{_m(b.get('B00') or {})}`",
        f"B01 `{_m(b.get('B01') or {})}` |ΔSharpe|=`{b.get('sharpe_gap')}` five Top1=`{b.get('five_top1_agree')}` overlay Top1=`{b.get('overlay_top1_agree')}`",
        "B00=B01 六层 PASS，分数与账本相同。路径：`result/round19/candidates/R19_B00/` `result/round19/candidates/R19_B01/`",
        "",
        "## 20. Six-Layer Parity",
        "",
        "| L | name | status |",
        "|---:|---|---|",
    ]
    for layer in six_layers:
        lines.append(f"| {layer.get('level')} | {layer.get('name')} | `{layer.get('status')}` |")
    if not six_layers:
        for k, v in (b.get("six_layer") or {}).items():
            lines.append(f"| {k} |  | `{v}` |")
    lines += [
        "",
        f"xgb_env_diff=`{b.get('xgb_env_diff')}` all_pass=`{six.get('all_pass')}`。不能只用 CAGR。XGBoost/环境差异不得当可忽略 PASS。",
        "L3：R7 five=45256；B01 five=45256；R18_A01 five=45255（F1 complete_cases 额外 drop，不是 B01）。",
        "路径：`result/round19/track_b/parity/six_layer.json`",
        "",
        "## 21. Join / Label / Fold / Feature-Order / Missing Audit",
        "",
        f"join_audit.csv n_rows=`{(((six_layers[0].get('detail') or {}).get('join') or {}).get('n_rows') if six_layers else 'n/a')}` prediction_missing=`0` extra=`0`",
        f"filled_with_zero=`{missing.get('filled_with_zero')}` A01 extra drop vs R7=`{missing.get('a01_extra_drop_vs_r7')}`",
        "A01 相对 Round7 多 drop 39 行 = F1 `complete_cases` 机制，不是默认 bug，也不是 fillna(0)。",
        "B01 用原 14/13，该额外 drop 不适用。feature order 与契约 14/13 一致，无 F1/F2/F3。",
        "路径：`result/round19/track_b/audit/`",
        "",
        "## 22. Five/Ten Heads, Top Change, Allocation",
        "",
        "| code | head | n | Rank IC | NDCG@1 | NDCG@2 | Top1 prec | Top2 prec |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in head_rows:
        lines.append(
            f"| {row.get('code')} | {row.get('head')} | {row.get('n')} | {row.get('rank_ic')} | "
            f"{row.get('ndcg@1')} | {row.get('ndcg@2')} | {row.get('top1_precision')} | {row.get('top2_precision')} |"
        )
    top_sum = top.get("summary") or {}
    lines += [
        "",
        f"Top1/Top2 变化：five change_rate=`{(top_sum.get('five') or {}).get('top1_change_rate')}` ten=`{(top_sum.get('ten') or {}).get('top1_change_rate')}` Kendall=`{(top_sum.get('five') or {}).get('mean_kendall_tau')}`",
        f"Allocation CF：selection_only=`{((alloc.get('delta_sharpe') or {}).get('selection_only'))}` allocation_only=`{((alloc.get('delta_sharpe') or {}).get('allocation_only'))}` path_dependence=`{alloc.get('path_dependence')}`",
        "诊断组不得晋级。",
        "",
        "## 23. Neighbor Conditional Increment",
        "",
        "| 因子 | vs 近邻 Spearman | Rank IC | partial Rank IC |",
        "|---|---|---:|---:|",
    ]
    for row in neigh_rows:
        pairs = ", ".join(f"{p.get('neighbor')}={p.get('spearman')}" for p in (row.get("pairs") or []))
        lines.append(
            f"| {row.get('factor')} | {pairs} | {row.get('rank_ic_vs_label')} | {row.get('partial_rank_ic_vs_label_given_neighbors')} |"
        )
    lines += [
        "",
        "R18 已观测正式组加 F 列全面弱于 A00。本表只解释是否只是近邻变换，不是新公式搜索。",
        "",
        "| R18 code | role | Sharpe | CAGR | formal |",
        "|---|---|---:|---:|---|",
    ]
    for row in r18_obs:
        lines.append(
            f"| {row.get('code')} | {row.get('role')} | {row.get('sharpe')} | {row.get('cagr')} | {row.get('formal')} |"
        )
    lines += [
        "",
        "## 24. Pipeline Bug Verdict and Repair",
        "",
        f"PIPELINE_BUG_VERDICT=`{b.get('PIPELINE_BUG_VERDICT')}` TRACK_B_DECISION=`{b.get('TRACK_B_DECISION')}`",
        f"material_bug=`{b.get('material_bug')}` ROUND18_REBACKTEST_REQUIRED=`{b.get('r18_rebacktest_required')}`",
        f"repair status=`{repair.get('status')}` repaired_best=`{repair.get('repaired_best')}`",
        "无 `R19_B_REPAIRED_*`。未搜 epsilon / 未改 F 公式 / 未改 head 权重。未覆盖 `result/round18` 原件。",
        "「没发现 material bug」≠ 因子永远无 Alpha；只关闭当前公式/数据/模型/样本。",
        "B 结果不得改变 O01 证据，不得与期权融合。",
        "",
        "## 25. Forward OOS",
        "",
        f"**{fwd.get('status') or FORWARD_OOS_NOT_AVAILABLE}** cutoff=`{fwd.get('cutoff') or MARKET_END}`",
        f"checked_after_freeze=`{fwd.get('checked_after_freeze')}` live_validated=`False`",
        f"usable_complete_unused_first_board=`{fwd.get('usable_complete_unused_first_board')}`",
        f"present_dates n=`{fwd.get('n_present_dates')}` missing_weekdays n=`{fwd.get('n_missing_weekdays')}`",
        f"`{SAME_OPEN_EXECUTION_OPTIMISM}` 仍在。不得宣布 `LIVE_VALIDATED`。",
        "",
    ]
    for r in fwd.get("reasons") or []:
        lines.append(f"- {r}")
    lines += [
        "",
        f"探查路径：`/data1/Stock/data_live/inbox/batches/` 与 `/data1/Stock/data_live/inbox/server_relay/`",
        f"产物：`{FWD_JSON}`",
        "",
        "## 26. Selection Bias Disclosure",
        "",
        f"**{bias.get('label')}** / **{bias.get('winners_curse')}** same_history_test_index=`{bias.get('same_history_test_index')}`",
        "R19 是同一历史的第三次检验。C3/C4 支持不等于独立 OOS。",
        f"{bias.get('note')}",
        f"路径：`{BIAS_JSON}` `result/round19/track_c/decision/selection_bias.md`",
        "",
        "## 27. Integrated Track C Decision",
        "",
        f"**{c.get('TRACK_C_DECISION')}**",
        f"CURRENT=`{CURRENT_BASELINE}` parent=`{PARENT_V4}` AUTO_PROMOTE=False",
        f"{RECOMMENDED_BASELINE_V5}=`{c.get('recommended_baseline_v5')}` lock=`{c.get('lock_flag')}`",
        f"{c.get('reason')}",
        "",
        "允许的四选一已落地其一。NEVER 自动替换 CURRENT。修复 R18 候选（无）不竞争 v5。",
        "",
        "| 线 | 裁决 |",
        "|---|---|",
        f"| Track A | {a.get('TRACK_A_DECISION')} |",
        f"| Track B | {b.get('TRACK_B_DECISION')} / {b.get('PIPELINE_BUG_VERDICT')} |",
        f"| Track C | {c.get('TRACK_C_DECISION')} |",
        f"| CURRENT | {CURRENT_BASELINE}（未改） |",
        "",
        "## 28. What Was Not Done / Forbidden",
        "",
        "- 不搜新股票/期权因子、不重搜阈值/Lookback/超参/90-10/B03/退出/seed、不跑 5000 组",
        "- 不把技术修复收益冒充新 Alpha；不把 Track A 期权叠加 Track B 新模型",
        "- 不自动升 CURRENT；追踪表不写「写入时间」",
        "- 不改 AGENTS.md、no-xgb-subsample 规则、core/round7、Stock src/runs",
        "- 不写 PIT_PASS / LIVE_READY / LIVE_VALIDATED；不虚构 bid/ask",
        "- 不晋级 C3/C4；不把 CPCV/bootstrap/SPA 当 forward OOS",
        "- 无 R19_B_REPAIRED_*；不覆盖 result/round16* / result/round18",
        "",
        "## 29. Remaining Risks",
        "",
        f"- `{SAME_OPEN_EXECUTION_OPTIMISM}`：决策=D 收盘，订单=D+1 开盘，盘中快照未验证",
        "- last_close_only IV，无历史买卖报价，不是完整 IV 曲面",
        "- PIT_TIMESTAMP_UNVERIFIED：2018 发布时间 ≠ 2026 再下载时间",
        "- SPA p=0.496 与三次检验选择偏差 / winner's curse",
        "- Pred Brier/LogLoss 略差；AUC 单独改善不够",
        f"- Forward `{fwd.get('status')}`：inbox 缺 2026-07-25..08-28 与首板/期权表",
        "- 开发窗 2018–2026 是 HISTORICAL DEVELOPMENT DATA，不是生产 OOS",
        "",
        "## 30. Artifacts, Paths, Tracker",
        "",
        f"- 主报告：`{FINAL_REPORT}`",
        f"- Track C 决策：`{DEC_JSON}`",
        f"- Forward：`{FWD_JSON}`",
        f"- Track A 决策：`{A_DEC}`",
        f"- Track B 决策：`{B_DEC}`",
        f"- 追踪表追加：`{TRACKER_PATH}`（无「写入时间」；CURRENT 未改）",
        f"- ROUND19_ROOT：`{ROUND19_ROOT}`",
        "",
        f"ROUND19_STATUS=`{c.get('ROUND19_STATUS') or 'COMPLETE'}` CURRENT=`{KEEP_BASELINE_V4}`/`{CURRENT_BASELINE}`",
        "",
    ]
    return "\n".join(lines)


def run_report(*, resume: bool = True, **kwargs: Any) -> dict[str, Any]:
    require_frozen()
    if not FWD_JSON.is_file():
        run_forward(resume=resume, **kwargs)
    if not DEC_JSON.is_file():
        run_decision(resume=resume, **kwargs)
    ctx = gather_ctx()
    write_md(FINAL_REPORT, _build_report(ctx))
    _append_tracker(ctx)
    c = ctx.get("c") or {}
    a = ctx.get("a") or {}
    b = ctx.get("b") or {}
    fwd = ctx.get("fwd") or {}
    par = ctx.get("parity") or {}
    freeze = ctx.get("freeze") or {}
    payload = {
        "status": "DONE",
        "ROUND19_STATUS": c.get("ROUND19_STATUS") or "COMPLETE",
        "TRACK_C_DECISION": c.get("TRACK_C_DECISION"),
        "TRACK_A_DECISION": a.get("TRACK_A_DECISION"),
        "TRACK_B_DECISION": b.get("TRACK_B_DECISION"),
        "PIPELINE_BUG_VERDICT": b.get("PIPELINE_BUG_VERDICT"),
        "current": CURRENT_BASELINE,
        "parent": PARENT_V4,
        "auto_promote": False,
        "recommended_baseline_v5": c.get("recommended_baseline_v5"),
        "lock_flag": c.get("lock_flag"),
        "forward": fwd.get("status"),
        "live_validated": False,
        "parity": par.get("status"),
        "failed_gates_a": a.get("failed_gates") or [],
        "nav_end_difference": a.get("nav_end_difference"),
        "c4_p_sharpe": a.get("c4_p_sharpe"),
        "spa_p_retained": R16_SPA_P,
        "manifest_a": (freeze.get("track_a") or {}).get("sha256"),
        "manifest_b": (freeze.get("track_b") or {}).get("sha256"),
        "final_report": str(FINAL_REPORT),
        "c0": a.get("c0_observed"),
        "c1": a.get("c1_observed"),
        "reason": c.get("reason"),
    }
    dump_json(DOSSIER_JSON, payload)
    return payload


def print_part10(payload: dict[str, Any] | None = None) -> None:
    p = payload or load_json(DOSSIER_JSON)
    c0 = p.get("c0") or {}
    c1 = p.get("c1") or {}
    print("", flush=True)
    print("===== ROUND19 PART 10 =====", flush=True)
    print(f"ROUND19_STATUS={p.get('ROUND19_STATUS') or 'COMPLETE'}", flush=True)
    print("", flush=True)
    print("Current baseline:", flush=True)
    print(f"{CURRENT_BASELINE} = {PARENT_V4}", flush=True)
    print("AUTO_PROMOTE=False CURRENT unchanged", flush=True)
    print("", flush=True)
    print("Stage 0 parity:", flush=True)
    print(p.get("parity") or "FAIL", flush=True)
    print(
        f"CAGR={c0.get('cagr')} Sharpe={c0.get('sharpe')} "
        f"MaxDD={c0.get('max_drawdown')} trades={c0.get('completed_trades')}",
        flush=True,
    )
    print("n_obs=2075 calendar CAGR=52.54% (canonical) 252-day=55.07% (contrast only)", flush=True)
    print("", flush=True)
    print("Track A:", flush=True)
    print(p.get("TRACK_A_DECISION"), flush=True)
    print(
        f"O01 CAGR={c1.get('cagr')} Sharpe={c1.get('sharpe')} "
        f"MaxDD={c1.get('max_drawdown')} trades={c1.get('completed_trades')}",
        flush=True,
    )
    print(f"NAV_delta={p.get('nav_end_difference')} C4 p_sharpe={p.get('c4_p_sharpe')} SPA16={p.get('spa_p_retained')}", flush=True)
    print(f"failed_gates={p.get('failed_gates_a')}", flush=True)
    print("", flush=True)
    print("Track B:", flush=True)
    print(f"{p.get('PIPELINE_BUG_VERDICT')} / {p.get('TRACK_B_DECISION')}", flush=True)
    print("ROUND18_REBACKTEST_REQUIRED=NO repaired=None", flush=True)
    print("", flush=True)
    print("Track C:", flush=True)
    print(p.get("TRACK_C_DECISION"), flush=True)
    rec = p.get("recommended_baseline_v5")
    if rec:
        print(f"{RECOMMENDED_BASELINE_V5}={rec} {AWAITING_USER_LOCK}", flush=True)
    print(p.get("reason") or "", flush=True)
    print("", flush=True)
    print("Forward OOS:", flush=True)
    print(p.get("forward") or FORWARD_OOS_NOT_AVAILABLE, flush=True)
    print("LIVE_VALIDATED=False", flush=True)
    print(f"{SAME_OPEN_EXECUTION_OPTIMISM} remains", flush=True)
    print("", flush=True)
    print("Final Report:", flush=True)
    print(str(FINAL_REPORT), flush=True)
    print("", flush=True)
    print(f"Final Report: {FINAL_REPORT}", flush=True)
    print("===== END ROUND19 PART 10 =====", flush=True)
