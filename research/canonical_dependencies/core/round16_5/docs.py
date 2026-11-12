"""主报告 1–23 + Q1–Q34、handoff、追踪表追加、PART 36。不写「写入时间」。"""

from __future__ import annotations

from typing import Any

from ..paths import MARKET_END, ROUND16_5_ROOT
from ..stock_readonly import sha256_file
from .attribution import load_attribution
from .common import dump_json, load_json, pct, write_md
from .contract import (
    FINAL_REPORT,
    HANDOFF,
    KEEP_BASELINE_V4,
    MANIFEST_NAME,
    O01_ROUND16,
    PARENT_V4,
    PRIOR_SELECTION_EXISTS,
    R16_SPA_P,
    TRACKER_PATH,
    V00,
    V01,
    V03,
    V04,
)
from .decision import load_decision, run_decision
from .equity import load_equity
from .figures import make_figures
from .freeze import require_frozen
from .orthogonal import load_orthogonal
from .parity import load_parity_status
from .placebo import load_placebo
from .prediction import load_prediction
from .quote import load_pit, load_quote
from .read import load_read
from .stats import load_bootstrap, load_cpcv, load_forward, load_spa, load_stats
from .stress import load_stress

DOSSIER_JSON = ROUND16_5_ROOT / "stats" / "dossier.json"


def gather_ctx() -> dict[str, Any]:
    frozen = {}
    mp = ROUND16_5_ROOT / MANIFEST_NAME
    if mp.is_file():
        frozen = {"sha256": sha256_file(mp), "path": str(mp)}
    promo = load_decision() or run_decision()
    return {
        "read": load_read(),
        "parity": load_parity_status(),
        "attr": load_attribution(),
        "placebo": load_placebo(),
        "equity": load_equity(),
        "orthogonal": load_orthogonal(),
        "quote": load_quote(),
        "pit": load_pit(),
        "prediction": load_prediction(),
        "stress": load_stress(),
        "stats": load_stats(),
        "bootstrap": load_bootstrap(),
        "cpcv": load_cpcv(),
        "spa": load_spa(),
        "forward": load_forward(),
        "promotion": promo,
        "frozen": frozen,
    }


def _append_tracker(ctx: dict[str, Any]) -> None:
    promo = ctx.get("promotion") or {}
    attr = ctx.get("attr") or {}
    placebo = ctx.get("placebo") or {}
    eq = ctx.get("equity") or {}
    ort = ctx.get("orthogonal") or {}
    par = ctx.get("parity") or {}
    recon = attr.get("reconciliation") or {}
    v00 = attr.get("v00") or {}
    v01 = attr.get("v01") or {}
    v03 = eq.get("observed") or {}
    v04 = ort.get("observed") or {}
    block = [
        "",
        "## Round 16.5 — Option Tail Confirmatory Validation",
        "",
        "确认性验证，不搜阈值。CURRENT 不自动升 v5。PRIOR_SELECTION_EXISTS=TRUE。",
        f"四选一 = `{promo.get('four_way')}`。CURRENT = `{promo.get('final_current')}`。CANDIDATE = `{promo.get('candidate')}`。",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| parity | {par.get('status')} |",
        f"| V00 CAGR/Sharpe/MaxDD | {pct(v00.get('cagr'))} / {v00.get('sharpe')} / {pct(v00.get('max_drawdown'))} |",
        f"| V01 CAGR/Sharpe/MaxDD | {pct(v01.get('cagr'))} / {v01.get('sharpe')} / {pct(v01.get('max_drawdown'))} |",
        f"| n_signal_332 | {recon.get('n_signal_all_including_2015')} |",
        f"| n_json_in_NAV (298) | {recon.get('n_json_in_nav_window')} |",
        f"| n_changed_fills | {recon.get('n_changed_fills')} |",
        f"| n_avoided | {recon.get('n_avoided_trades')} |",
        f"| FULL_ENGINE_NAV_DIFFERENCE | {attr.get('nav_end_difference')} |",
        f"| V02 N / p_sharpe / percentile | {placebo.get('n')} / {placebo.get('p_empirical_sharpe')} / {placebo.get('percentile_sharpe')} |",
        f"| V03 Sharpe / Jaccard | {(v03.get('sharpe'))} / {eq.get('jaccard')} |",
        f"| V04 CAGR/Sharpe/MaxDD | {pct(v04.get('cagr'))} / {v04.get('sharpe')} / {pct(v04.get('max_drawdown'))} |",
        f"| quote | {(ctx.get('quote') or {}).get('verdict')} |",
        f"| SPA this round / Round16 SPA | {((ctx.get('spa') or {}).get('spa') or {}).get('p_spa')} / {R16_SPA_P} |",
        f"| four_way | {promo.get('four_way')} |",
        f"| CURRENT | {promo.get('final_current')} |",
        f"| CANDIDATE | {promo.get('candidate')} |",
        f"| LIVE | {promo.get('live')} |",
        f"| forward OOS | {(ctx.get('forward') or {}).get('status')} |",
        f"| lock | {promo.get('lock_status')} |",
        f"| manifest sha256 | `{(ctx.get('frozen') or {}).get('sha256')}` |",
        "",
        "主报告：`result/round16_5/ROUND16_5_CONFIRMATORY_REPORT.md`。",
        "Handoff：`result/round16_5/ROUND17_RESEARCH_HANDOFF.md`。",
        "",
    ]
    text = TRACKER_PATH.read_text(encoding="utf-8")
    if "## Round 16.5 — Option Tail Confirmatory" in text:
        return
    TRACKER_PATH.write_text(text.rstrip() + "\n" + "\n".join(block), encoding="utf-8")


def _handoff(ctx: dict[str, Any]) -> None:
    promo = ctx.get("promotion") or {}
    attr = ctx.get("attr") or {}
    placebo = ctx.get("placebo") or {}
    eq = ctx.get("equity") or {}
    ort = ctx.get("orthogonal") or {}
    quote = ctx.get("quote") or {}
    body = [
        "# ROUND17_RESEARCH_HANDOFF",
        "",
        "只交接，不搜索，不跑 Round17。",
        "",
        "## 1. 已验证期权因子是否有增量",
        "",
        f"四选一=`{promo.get('four_way')}` FULL_ENGINE_NAV_DIFFERENCE=`{attr.get('nav_end_difference')}` "
        f"V02 N=`{placebo.get('n')}` p_sharpe=`{placebo.get('p_empirical_sharpe')}`。",
        f"PRIOR_SELECTION_EXISTS=`{PRIOR_SELECTION_EXISTS}`，不是独立确认。",
        "",
        "## 2. 与股票因子冗余",
        "",
        f"V03 Jaccard=`{eq.get('jaccard')}` activity_matched=`{eq.get('activity_matched')}` de_risking_only=`{promo.get('de_risking_only')}`。",
        "Round16 redundancy=`REDUNDANT` 仍成立，除非本轮 V01 明确优于活动匹配 V03。",
        "",
        "## 3. 是否纳入 Round17 股票 Alpha",
        "",
        "仅当 confirmatory 为 STRENGTHEN 且非 DE_RISKING_ONLY 时考虑作为状态特征，而非本轮重训。默认不纳入。",
        "",
        "## 4. 双头五 Seed 集成",
        "",
        "five-head + ten-head；90/10 fusion；B03 allocator。Round16.5 未改。",
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
        "- 不要再用账本日收益置换当 placebo",
        "- 不要报 p=0",
        "- 不要把避开交易 PnL 之和当权威增量",
        "- 不要搜 Top2 0/5/10、加 Seed、改 90/10、重训 XGB",
        "- 不要把 CPCV 当 forward OOS",
        "- 不要把 2018–2026 重复检验写成独立确认（PRIOR_SELECTION_EXISTS）",
        "- 不要编造 bid/ask",
        f"- V04 若更差不要换第二套残差（本轮 worse_than_v01=`{ort.get('worse_than_v01')}`）",
        "",
        "## 9. 值得再研究的股票 Alpha 特征族",
        "",
        "封板质量、尾盘量能、行业广度。期权状态仅在 confirmatory 加强后作为市场状态，不替代个股特征。",
        "",
        "## 10. 可复用数据与计算",
        "",
        f"- Round16 factors：`result/round16/data/processed/option_factors_daily.parquet`",
        f"- Round16.5 V00/V01：`result/round16_5/parity/`",
        f"- V02 checkpoint：`result/round16_5/placebo/placebo_runs.jsonl` N=`{placebo.get('n')}`",
        "- 回测引擎：`core/round15/runner.py` run_r15_backtest / apply_top2_veto",
        f"- quote=`{quote.get('verdict')}`",
        "",
        "禁止根据本轮结果提前做 Round17 搜索。CURRENT 仍 baseline_v4。",
        "",
    ]
    write_md(HANDOFF, "\n".join(body))


def _q(ctx: dict[str, Any]) -> list[str]:
    attr = ctx.get("attr") or {}
    recon = attr.get("reconciliation") or {}
    par = ctx.get("parity") or {}
    placebo = ctx.get("placebo") or {}
    eq = ctx.get("equity") or {}
    ort = ctx.get("orthogonal") or {}
    quote = ctx.get("quote") or {}
    pit = ctx.get("pit") or {}
    pred = ctx.get("prediction") or {}
    stress = ctx.get("stress") or {}
    boot = ctx.get("bootstrap") or {}
    spa = ctx.get("spa") or {}
    fwd = ctx.get("forward") or {}
    promo = ctx.get("promotion") or {}
    v00 = attr.get("v00") or {}
    v01 = attr.get("v01") or {}
    v03 = eq.get("observed") or {}
    v04 = ort.get("observed") or {}
    pairs = boot.get("pairs") or {}
    loss = ((pred.get("results") or {}).get("top2_loss_net_lt_0") or {})
    qs = [
        ("Q1", "V00 parity 是否 PASS？", str(par.get("status"))),
        ("Q2", "V01 parity 是否 PASS？", str(((par.get("anchors") or {}).get(V01) or {}).get("status"))),
        ("Q3", "332 是什么？", f"信号日含 2015+ = {recon.get('n_signal_all_including_2015')}（treat_dates.json）"),
        ("Q4", "298 是什么？", f"JSON ∩ 2018+ NAV = {recon.get('n_json_in_nav_window')}（Round16 n_treat_days）"),
        ("Q5", "回测窗内信号日？", str(recon.get("n_signal_in_backtest_window"))),
        ("Q6", "合格 Top2 且预算>0 且期权可观测？", str(recon.get("n_eligible_top2_nonzero_budget_observable"))),
        ("Q7", "实际改变成交日？", str(recon.get("n_changed_fills"))),
        ("Q8", "避开交易数？", str(recon.get("n_avoided_trades"))),
        ("Q9", "无效果日？", str(recon.get("n_no_effect_days"))),
        ("Q10", "权威 NAV 增量？", f"{attr.get('nav_end_difference')} ({attr.get('authoritative_increment_kind')})"),
        ("Q11", "避开交易 PnL 之和？", f"{attr.get('sum_avoided_trade_pnl')}（非权威）"),
        ("Q12", "是否 CONCENTRATED_INCREMENT_RISK？", str(attr.get("concentrated_flag"))),
        ("Q13", "V02 实际完成 N？", str(placebo.get("n"))),
        ("Q14", "V02 经验 p（Sharpe）？", str(placebo.get("p_empirical_sharpe"))),
        ("Q15", "V02 Sharpe 百分位？", str(placebo.get("percentile_sharpe"))),
        ("Q16", "placebo 是否全引擎？", str(placebo.get("method"))),
        ("Q17", "V03 是否活动匹配？", f"{eq.get('activity_matched')} flag={eq.get('activity_flag')}"),
        ("Q18", "V03 Jaccard vs O01？", str(eq.get("jaccard"))),
        ("Q19", "V03 Sharpe？", str(v03.get("sharpe"))),
        ("Q20", "V01 是否优于 V03？", f"Sharpe V01={v01.get('sharpe')} vs V03={v03.get('sharpe')}；活动匹配门 beats_v03=`{promo.get('beats_v03')}`（V03 ACTIVITY_NOT_FULLY_MATCHED）"),
        ("Q21", "是否 DE_RISKING_ONLY？", str(promo.get("de_risking_only"))),
        ("Q22", "V04 构造？", "Ridge α=1.0 window 504/252 then residual 90th 252/126 shift(1)"),
        ("Q23", "V04 CAGR/Sharpe/MaxDD？", f"{v04.get('cagr')} / {v04.get('sharpe')} / {v04.get('max_drawdown')}"),
        ("Q24", "V04 是否差于 V01？", str(ort.get("worse_than_v01"))),
        ("Q25", "报价裁决？", str(quote.get("verdict"))),
        ("Q26", "bid/ask？", str(quote.get("bid_ask"))),
        ("Q27", "PIT？", str(pit.get("status"))),
        ("Q28", "Top2 亏损 OOS AUC equity vs +skew？", f"{(loss.get('equity_only') or {}).get('auc')} vs {(loss.get('equity_plus_skew') or {}).get('auc')}"),
        ("Q29", "+20/+30bp 是否翻符号？", str(stress.get("sign_flip"))),
        ("Q30", "Bootstrap V01-V00 p？", str((pairs.get(f"{V01}-{V00}") or {}).get("p_mean_le_0"))),
        ("Q31", "CPCV 是否 forward？", "否"),
        ("Q32", "本轮 SPA p / Round16 SPA p？", f"{(spa.get('spa') or {}).get('p_spa')} / {R16_SPA_P}"),
        ("Q33", "Forward OOS / LIVE？", f"{fwd.get('status')} / {promo.get('live')}"),
        ("Q34", "四选一 / CURRENT / CANDIDATE？", f"{promo.get('four_way')} / {promo.get('final_current')} / {promo.get('candidate')}"),
    ]
    lines = ["## 22. Q1–Q34", ""]
    for code, q, a in qs:
        lines += [f"**{code}. {q}**", "", str(a), ""]
    return lines


def run_report(**_: Any) -> dict[str, Any]:
    require_frozen()
    ctx = gather_ctx()
    figs = make_figures()
    ctx["figures"] = figs
    promo = ctx.get("promotion") or run_decision()
    ctx["promotion"] = promo
    attr = ctx.get("attr") or {}
    recon = attr.get("reconciliation") or {}
    par = ctx.get("parity") or {}
    placebo = ctx.get("placebo") or {}
    eq = ctx.get("equity") or {}
    ort = ctx.get("orthogonal") or {}
    quote = ctx.get("quote") or {}
    pit = ctx.get("pit") or {}
    pred = ctx.get("prediction") or {}
    stress = ctx.get("stress") or {}
    boot = ctx.get("bootstrap") or {}
    spa = ctx.get("spa") or {}
    fwd = ctx.get("forward") or {}
    v00 = attr.get("v00") or {}
    v01 = attr.get("v01") or {}
    v03 = eq.get("observed") or {}
    v04 = ort.get("observed") or {}
    lines = [
        "# ROUND16_5_CONFIRMATORY_REPORT",
        "",
        "Round 16.5 = 期权尾部确认性验证。不搜阈值。CURRENT 不得自动升 v5。",
        f"manifest sha256=`{(ctx.get('frozen') or {}).get('sha256')}`。parity=`{par.get('status')}`。",
        f"四选一=**{promo.get('four_way')}** CURRENT=**{promo.get('final_current')}** CANDIDATE=`{promo.get('candidate')}` LIVE=`{promo.get('live')}`。",
        f"PRIOR_SELECTION_EXISTS=`{PRIOR_SELECTION_EXISTS}`。Round16 SPA p=`{R16_SPA_P}` 保留。",
        "",
        "## 1. Executive Verdict",
        "",
        f"{promo.get('reason')}",
        f"V00 Sharpe=`{v00.get('sharpe')}` V01 Sharpe=`{v01.get('sharpe')}` NAV 增量=`{attr.get('nav_end_difference')}`。",
        f"V02 N=`{placebo.get('n')}` p=`{placebo.get('p_empirical_sharpe')}` percentile=`{placebo.get('percentile_sharpe')}`。",
        f"V03 Sharpe=`{v03.get('sharpe')}` Jaccard=`{eq.get('jaccard')}`。V04 Sharpe=`{v04.get('sharpe')}`。",
        f"quote=`{quote.get('verdict')}` PIT=`{pit.get('status')}` forward=`{fwd.get('status')}`。",
        "",
        "## 2. Research Question and Closed Family",
        "",
        "ETF 期权隐含下行 skew 是否对识别不利 Top2 交易提供真实、增量、经济上有用的信息？",
        "",
        f"- V00 `{V00}` 策略 = baseline_v4",
        f"- V01 `{V01}` 策略 = 精确 O01",
        "- V02 全引擎随机 Top2 veto 对照族（不是候选）",
        f"- V03 `{V03}` 股票风险 veto（不是候选）",
        f"- V04 `{V04}` 唯一新 Alpha = 时序残差 Put Skew",
        "",
        "## 3. Prior Selection and Why This Is Not Independent Confirmation",
        "",
        f"PRIOR_SELECTION_EXISTS=`{PRIOR_SELECTION_EXISTS}`。O01 已在 Round16 的同一 2018–2026 窗上被选出。",
        "本轮重复检验是确认性/方法学修复，**不是**独立前向确认。",
        "",
        "## 4. Round16 Bugs Being Repaired",
        "",
        "- Placebo 曾是账本日收益置换，不是全引擎。",
        "- n_events=332 含 2015；n_treat_days=298 是 NAV 对齐。必须拆层。",
        "- 经验 p 禁止报 0。",
        "- 权威增量=FULL_ENGINE_NAV_DIFFERENCE。",
        f"- Round16 SPA p=`{R16_SPA_P}` 必须保留在讨论中。",
        "",
        "## 5. V00/V01 Parity",
        "",
        f"status=`{par.get('status')}`。失败则本报告不应产生后续统计；pipeline 会 STOP。",
        f"V00 观察 CAGR=`{v00.get('cagr')}` Sharpe=`{v00.get('sharpe')}` MaxDD=`{v00.get('max_drawdown')}` 笔数=`{v00.get('n_days')}`。",
        f"V01 观察 CAGR=`{v01.get('cagr')}` Sharpe=`{v01.get('sharpe')}` MaxDD=`{v01.get('max_drawdown')}`。",
        "详见 `00_parity.md`。",
        "",
        "## 6. Freeze and Pre-registered Gates",
        "",
        f"manifest=`{(ctx.get('frozen') or {}).get('path')}` sha256=`{(ctx.get('frozen') or {}).get('sha256')}`。",
        "promotion 数值门在看 V03/V04 NAV 之前锁定。之后不得放宽。",
        "",
        "## 7. Event Count Reconciliation (332 vs 298)",
        "",
        f"| 层 | N |",
        f"|---|---:|",
        f"| 信号日含 2015 | {recon.get('n_signal_all_including_2015')} |",
        f"| JSON treat_dates | {recon.get('n_json_treat_dates')} |",
        f"| JSON ∩ NAV 窗 | {recon.get('n_json_in_nav_window')} |",
        f"| 回测窗内信号 | {recon.get('n_signal_in_backtest_window')} |",
        f"| 2015–2017 | {recon.get('n_signal_pre_2018')} |",
        f"| 合格日 | {recon.get('n_eligible_top2_nonzero_budget_observable')} |",
        f"| 改变成交 | {recon.get('n_changed_fills')} |",
        f"| 避开 | {recon.get('n_avoided_trades')} |",
        f"| 无效果 | {recon.get('n_no_effect_days')} |",
        "",
        "详见 `02_event_count_reconciliation.md`。",
        "",
        "## 8. Trade-level Attribution",
        "",
        f"CSV=`{attr.get('csv')}`。分年与 Top1/3/5/10 日贡献见 attribution.json。集中度=`{attr.get('concentrated_flag')}`。",
        "",
        "## 9. Authoritative FULL_ENGINE_NAV_DIFFERENCE",
        "",
        f"期末 NAV 差=`{attr.get('nav_end_difference')}` CAGR 增量=`{attr.get('cagr_increment')}` Sharpe 增量=`{attr.get('sharpe_increment')}`。",
        f"避开交易 PnL 之和=`{attr.get('sum_avoided_trade_pnl')}`（明确 **不是** 权威增量）。",
        "",
        "## 10. Placebo Engine Check",
        "",
        f"`{(placebo.get('engine_check') or {})}`",
        "",
        "## 11. V02 Full-Engine Placebo",
        "",
        f"N=`{placebo.get('n')}` / 目标 1000。method=`{placebo.get('method')}` not_ledger_swap=`{placebo.get('not_ledger_swap')}`。",
        f"p_sharpe=`{placebo.get('p_empirical_sharpe')}` p_cagr=`{placebo.get('p_empirical_cagr')}` percentile=`{placebo.get('percentile_sharpe')}`。",
        f"公式 `{placebo.get('formula')}`。分层=`{(placebo.get('matching') or {}).get('layer')}`。",
        "",
        "## 12. V03 Equity-only Risk Veto",
        "",
        f"**不是候选**。Sharpe=`{v03.get('sharpe')}` CAGR=`{v03.get('cagr')}` Jaccard=`{eq.get('jaccard')}` matched=`{eq.get('activity_matched')}`。",
        "",
        "## 13. V04 Orthogonal Residual Put Skew",
        "",
        f"唯一新 Alpha。n_events=`{ort.get('n_events')}` CAGR=`{v04.get('cagr')}` Sharpe=`{v04.get('sharpe')}` MaxDD=`{v04.get('max_drawdown')}` worse_than_v01=`{ort.get('worse_than_v01')}`。",
        "不换第二套残差模型。",
        "",
        "## 14. Quote Quality and PIT",
        "",
        f"quote=`{quote.get('verdict')}` bid/ask=`{quote.get('bid_ask')}` IV=`last_close_only`。PIT=`{pit.get('status')}`。",
        "无双边不得写成完整 IV 面。相对优势与 DEPLOYMENT_FEASIBILITY / SAME_OPEN_EXECUTION_OPTIMISM 分开。",
        "",
        "## 15. Top2 Loss Prediction",
        "",
        f"`{pred.get('results')}`",
        "",
        "## 16. Same-window Performance",
        "",
        f"`{((ctx.get('stats') or {}).get('windows') or {}).get('windows')}`",
        "",
        "## 17. Cost Stress",
        "",
        f"sign_flip=`{stress.get('sign_flip')}` HISTORICAL_FEE_CORRECTED 为主评价。+20/+30bp 见 `12_cost_stress.md`。",
        "",
        "## 18. Bootstrap / CPCV / SPA",
        "",
        f"bootstrap N=`{boot.get('n')}` pairs=`{boot.get('pairs')}`。",
        f"CPCV not_forward=`True`。本轮 SPA p=`{(spa.get('spa') or {}).get('p_spa')}`。Round16 SPA p=`{R16_SPA_P}` 保留。",
        "方法学差异：Round16 family=O01+O02 vs v4；本轮 family=V01+V04 vs V00，且 placebo 改为全引擎。同一历史窗，不是独立确认。",
        "",
        "## 19. Forward OOS",
        "",
        f"**{fwd.get('status')}** cutoff=`{MARKET_END}` LIVE=`{promo.get('live')}`。",
        "",
        "## 20. Promotion Decision",
        "",
        f"四选一=**{promo.get('four_way')}** CURRENT=**{KEEP_BASELINE_V4}** CANDIDATE=`{promo.get('candidate')}` lock=`{promo.get('lock_status')}`。",
        f"{promo.get('reason')}",
        "禁止为出 v5 降门。",
        "",
        "## 21. Remaining Risks",
        "",
        "- last_close_only IV；无历史买卖报价",
        "- PRIOR_SELECTION_EXISTS；选择偏差与 Round16 SPA",
        "- 双袖套路径依赖使避开交易 PnL ≠ 组合差额",
        "- V02 若 N<1000 则功效不足，以实际 N 为准",
        "- SAME_OPEN_EXECUTION_OPTIMISM",
        "- 2018–2026 是 HISTORICAL DEVELOPMENT DATA",
        "",
    ]
    lines += _q(ctx)
    lines += [
        "## 23. Implications for Round17",
        "",
        "见 ROUND17_RESEARCH_HANDOFF.md。不在本轮搜索、不跑 Round17。",
        "",
        "### 明确没做",
        "",
        "- 不搜 80/85/90/95、lookback、Top2 0/5/10",
        "- 不重训 XGB、不改 90/10、不加期权因子",
        "- 不自动把 v5 写成 CURRENT；追踪表不写「写入时间」",
        "- 不编造 bid/ask；不把 CPCV 当 forward",
        "- 不声称独立确认",
        "",
        "### Figures",
        "",
        f"`{(figs or {}).get('paths')}`",
        "",
    ]
    write_md(FINAL_REPORT, "\n".join(lines))
    _append_tracker(ctx)
    _handoff(ctx)
    complete = {
        "four_way": promo.get("four_way"),
        "final_current": promo.get("final_current"),
        "candidate": promo.get("candidate"),
        "live": promo.get("live"),
        "reason": promo.get("reason"),
        "parity": par.get("status"),
        "recon": recon,
        "nav_increment": attr.get("nav_end_difference"),
        "n_changed": recon.get("n_changed_fills"),
        "n_avoided": recon.get("n_avoided_trades"),
        "placebo_n": placebo.get("n"),
        "placebo_p": placebo.get("p_empirical_sharpe"),
        "placebo_percentile": placebo.get("percentile_sharpe"),
        "v00": v00,
        "v01": v01,
        "v03": v03,
        "v04": v04,
        "quote": quote.get("verdict"),
        "pit": pit.get("status"),
        "spa": (spa.get("spa") or {}).get("p_spa"),
        "round16_spa_p": R16_SPA_P,
        "forward": fwd.get("status"),
        "manifest_sha256": (ctx.get("frozen") or {}).get("sha256"),
        "dossier": str(FINAL_REPORT),
        "handoff": str(HANDOFF),
        "parent": PARENT_V4,
        "o01": O01_ROUND16,
        "figures": figs,
        "promotion": promo,
        "blockers": promo.get("failed_gates") or [],
    }
    dump_json(DOSSIER_JSON, complete)
    return complete


def print_complete_block(payload: dict[str, Any]) -> None:
    print("ROUND16.5 COMPLETE", flush=True)
    print("", flush=True)
    print("Current baseline:", flush=True)
    print("baseline_v4 = R9_B03", flush=True)
    print("", flush=True)
    print("Parity:", flush=True)
    print(payload.get("parity") or "FAIL", flush=True)
    print("", flush=True)
    print("332 vs 298:", flush=True)
    recon = payload.get("recon") or {}
    print(
        f"signal={recon.get('n_signal_all_including_2015')} json_in_nav={recon.get('n_json_in_nav_window')} "
        f"changed={recon.get('n_changed_fills')} avoided={recon.get('n_avoided_trades')} no_effect={recon.get('n_no_effect_days')}",
        flush=True,
    )
    print("", flush=True)
    print("Authoritative NAV increment:", flush=True)
    print(payload.get("nav_increment"), flush=True)
    print("", flush=True)
    print("V02 N / percentile / p:", flush=True)
    print(f"{payload.get('placebo_n')} / {payload.get('placebo_percentile')} / {payload.get('placebo_p')}", flush=True)
    print("", flush=True)
    print("V03:", flush=True)
    v03 = payload.get("v03") or {}
    print(f"CAGR={v03.get('cagr')} Sharpe={v03.get('sharpe')} MaxDD={v03.get('max_drawdown')}", flush=True)
    print("", flush=True)
    print("V04:", flush=True)
    v04 = payload.get("v04") or {}
    print(f"CAGR={v04.get('cagr')} Sharpe={v04.get('sharpe')} MaxDD={v04.get('max_drawdown')}", flush=True)
    print("", flush=True)
    print("Quote quality:", flush=True)
    print(payload.get("quote"), flush=True)
    print("", flush=True)
    print("SPA this round / Round16 SPA:", flush=True)
    print(f"{payload.get('spa')} / {payload.get('round16_spa_p')}", flush=True)
    print("", flush=True)
    print("Forward:", flush=True)
    print(payload.get("forward"), flush=True)
    print("", flush=True)
    print("Four-way:", flush=True)
    print(payload.get("four_way"), flush=True)
    print("", flush=True)
    print("CURRENT / CANDIDATE / LIVE:", flush=True)
    print(f"{payload.get('final_current')} / {payload.get('candidate')} / {payload.get('live')}", flush=True)
    print("", flush=True)
    print("Final report:", flush=True)
    print(str(FINAL_REPORT), flush=True)
    print("", flush=True)
    print("Round17 handoff:", flush=True)
    print(str(HANDOFF), flush=True)
    print("", flush=True)
    print("Blockers:", flush=True)
    print(payload.get("blockers"), flush=True)
    print("", flush=True)
    print(f"Final Report: {FINAL_REPORT}", flush=True)
