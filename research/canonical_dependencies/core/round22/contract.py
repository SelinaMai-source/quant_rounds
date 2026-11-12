"""Round 22 冻结契约。看候选回测前写入 manifest，禁止回调。不改交易引擎，不升 v6。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT, RESULT_ROOT, ROUND21_4_ROOT, ROUND22_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, SEEDS, TEN_FEATURES, TEN_HEAD
from ..round21.contract import (
    AUTO_PROMOTE_CURRENT,
    COLSAMPLE,
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    FEE_SCHEDULE,
    HERMES_REBUILD_RAW,
    HERMES_V7,
    NAV_END_ABS_TOL,
    O01_FORMULA,
    PARITY_METRIC_ABS,
    SUBSAMPLE,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    XGB_N_JOBS,
)
from ..round21_4.contract import ECONOMIC_MATERIALITY as R21_3_GATE

KEEP_BASELINE_V5 = "KEEP_BASELINE_V5"
RECOMMEND_V6_RESEARCH_CANDIDATE = "RECOMMEND_V6_RESEARCH_CANDIDATE"
AUTO_PROMOTE_CURRENT = AUTO_PROMOTE_CURRENT
CURRENT_BASELINE = CURRENT_BASELINE
CURRENT_ALIAS = CURRENT_ALIAS
NOT_TRIGGERED = "NOT_TRIGGERED"
BLOCKED_PIT = "BLOCKED_PIT"
BLOCKED_DATA = "BLOCKED_DATA"
BLOCKED_ACCESS = "BLOCKED_ACCESS"
BLOCKED_SEMANTICS = "BLOCKED_SEMANTICS"
BLOCKED = "BLOCKED"
FAILED = "FAILED"
PARTIAL = "PARTIAL"
COMPLETE = "COMPLETE"
PIT_PASS = "PIT_PASS"
DUPLICATE_EXPRESSION = "DUPLICATE_EXPRESSION"
CROSS_SECTIONALLY_DEGENERATE = "CROSS_SECTIONALLY_DEGENERATE"
COMPOSITE_OF_KNOWN_FACTORS = "COMPOSITE_OF_KNOWN_FACTORS"
EXPLORATORY_DIAGNOSTIC = "EXPLORATORY_DIAGNOSTIC"
NOT_APPLICABLE = "NOT_APPLICABLE"
NEVER_SUSPENDED = "NEVER_SUSPENDED"
NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"
STRICT_PIT_SUBPERIOD_RESEARCH = "STRICT_PIT_SUBPERIOD_RESEARCH"
INSUFFICIENT_TRAIN_HISTORY = "INSUFFICIENT_TRAIN_HISTORY"
FORWARD_OOS_NOT_AVAILABLE = "FORWARD_OOS_NOT_AVAILABLE"
HISTORICAL_FEE_CORRECTED = "HISTORICAL_FEE_CORRECTED"
IMPORTED_BY_HASH = "IMPORTED_BY_HASH"
TRAINED_REAL = "TRAINED_REAL"
REPEATED_HISTORICAL_TESTING = "REPEATED_HISTORICAL_TESTING"
PROFILE_A = "BASELINE_SIGNAL_CUTOFF"
PROFILE_B = "POST_CLOSE_TO_NEXT_PREOPEN"
SAME_OPEN_EXECUTION_OPTIMISM = "SAME_OPEN_EXECUTION_OPTIMISM"

FIVE_FEATURES = tuple(FIVE_FEATURES)
TEN_FEATURES = tuple(TEN_FEATURES)
FIVE_ONLY_FEATURE = "g0_amount_log"
FIVE_HEAD = FIVE_HEAD
TEN_HEAD = TEN_HEAD
SEEDS = tuple(SEEDS)
SUBSAMPLE = SUBSAMPLE
COLSAMPLE = COLSAMPLE
XGB_N_JOBS = XGB_N_JOBS
FEE_SCHEDULE = FEE_SCHEDULE
V5_ANCHOR = dict(V5_ANCHOR)
V5_NAV_END_EXACT = V5_NAV_END_EXACT
NAV_END_ABS_TOL = NAV_END_ABS_TOL
PARITY_METRIC_ABS = PARITY_METRIC_ABS
O01_FORMULA = O01_FORMULA
FUSION_WEIGHTS = {"ten": 0.90, "five": 0.10}
HERMES_V7 = HERMES_V7
HERMES_REBUILD_RAW = HERMES_REBUILD_RAW
DATA_LIVE_RELEASES = Path("/data1/Stock/data_live/releases")
CLICKHOUSE_HOST = "127.0.0.1"
CLICKHOUSE_PORT = 18123
MARKET_CUTOFF = "2026-07-24"
DECISION_AUDIT = "decision=D_close; order=D+1_open; same_open_execution_optimism=True"

PIT_MIN_FINITE = 0.80
REDUNDANCY_RHO = 0.60
DUPLICATE_RHO = 0.999
CONSTANT_DAY_SHARE = 0.80
BLOCK_DAYS = 20
BOOT_N = 5000
SPA_BOOT = 5000
KEYS = ["trade_date", "instrument_id"]
EPS = 1e-12
LABELS = ("label_e2_net", "label_exit_rule_net")
E0_CODE = "R22_E0"
PROFILE_B_E0 = "R22B_E0"

# 沿用 R21_3_ECONOMIC_MATERIALITY_V1，禁止回调。
ECONOMIC_MATERIALITY = {
    **{k: v for k, v in R21_3_GATE.items() if k != "note"},
    "id": "R21_3_ECONOMIC_MATERIALITY_V1",
    "carried_from": "round21_3",
    "frozen_before_formal_nav": True,
    "do_not_retune": True,
    "round": "22",
    "note": "ΔCAGR≥+1.0pct、ΔSharpe≥+0.05、|ΔNAV|/NAV≥1%、Top1/Top2 变化>0、MaxDD 恶化≤2pct、费用方向不翻、不只靠 2024 或 2025。看候选回测前冻结。",
}

OOF_INCREMENT_GATE = {
    "id": "R22_OOF_INCREMENT_V1",
    "compare_to": E0_CODE,
    "metric": "oof_rankic_mean",
    "not_worse_both_heads": True,
    "strict_better_any_head": True,
    "min_oos_days": 200,
    "min_delta": 0.0,
    "eps": 1e-12,
    "frozen_before_formal_nav": True,
}

TRAIN_GATE = {
    "id": "R22_TRAIN_GATE_V1",
    "require_pit_pass": True,
    "forbid_duplicate": True,
    "forbid_degenerate": True,
    "finite_share_min": PIT_MIN_FINITE,
    "require_nonzero_cs_var": True,
    "full_history_only": True,
    "profile_a_main_model": True,
    "profile_b_separate": True,
    "dragon_tiger_profile": PROFILE_B,
    "low_rho_action": "ADD_ONLY",
    "high_rho_action": "REMOVE_REPLACE_ALLOWED",
    "rho_threshold": REDUNDANCY_RHO,
    "exploratory_low_rho_delete": EXPLORATORY_DIAGNOSTIC,
    "matched_tiers": [["E1", "E2", "E3"], ["E4", "E5"], ["E6", "E7"], ["E8", "E9"]],
    "forbid_e3_without_e1": True,
    "forbid_replace_without_remove": True,
    "short_pit_compare": "same_window_v5_only",
    "frozen_before_formal_nav": True,
}

DECISION_RULE = {
    "allowed": [KEEP_BASELINE_V5, RECOMMEND_V6_RESEARCH_CANDIDATE],
    "v6_requires": [
        "economic_gate",
        "fee_direction_holds",
        "top1_top2_net_not_worse",
        "increment_from_new_factor_not_remove_alone",
    ],
    "no_independent_forward_not_live_ready": True,
    "current_baseline_changed": False,
    "auto_promote": False,
}

FACTOR_SPECS: tuple[dict[str, Any], ...] = (
    {"factor_id": "r22_a1_own_close_limit_excess_20", "family": "A", "profile": "A", "short": "个股收盘涨停占比减市场收盘涨停广度", "window": "D-20..D-1", "formula": "mean(I_close_limit)_{D-20..D-1} - mean(mkt_close_breadth)_{D-20..D-1}", "inputs": ["daily_raw.close", "daily_raw.limit_up", "daily_raw.pre_close"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_a2_own_touch_limit_excess_20", "family": "A", "profile": "A", "short": "个股触板占比减市场触板广度", "window": "D-20..D-1", "formula": "mean(I_high_touch_limit)_{D-20..D-1} - mean(mkt_touch_breadth)_{D-20..D-1}", "inputs": ["daily_raw.high", "daily_raw.limit_up"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_a3_ret20_mkt_beta_resid", "family": "A", "profile": "A", "short": "20日收益对市场收益的滚动 beta 残差", "window": "D-60..D-1", "formula": "ret_20 - beta_60(ret, mkt_ret) * mkt_ret_20；beta 仅用 D-60..D-1", "inputs": ["daily_raw.close", "daily_raw.pre_close"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_a4_touch5_over_mkt_breadth", "family": "A", "profile": "A", "short": "近5日触板占比除以市场收盘广度", "window": "D-20..D-1", "formula": "mean(I_touch)_{D-5..D-1} / (eps + mean(mkt_close_breadth)_{D-20..D-1})", "inputs": ["daily_raw.high", "daily_raw.limit_up"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_b1_dyn_peer_ret_d1", "family": "B", "profile": "A", "short": "动态正相关 peer 的 D-1 收益均值", "window": "corr D-60..D-1", "formula": "mean(ret_{p,D-1}) over top_k=5 positive corr, overlap>=40, exclude self", "inputs": ["daily_raw.close", "daily_raw.pre_close"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_b2_dyn_peer_ret_disp", "family": "B", "profile": "A", "short": "动态 peer 的 D-1 收益离散度", "window": "corr D-60..D-1", "formula": "std(ret_{p,D-1}) of the same peer set; <2 peers → NA", "inputs": ["daily_raw.close"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_b3_dyn_peer_to_innov", "family": "B", "profile": "A", "short": "动态 peer 的换手创新", "window": "D-21..D-1", "formula": "mean(amount_p,D-1 / mean(amount_p,D-21..D-2) - 1)", "inputs": ["daily_raw.amount"], "duplicate_of": "check_vs_r21_y1", "label": None},
    {"factor_id": "r22_b4_dyn_peer_ret_over_amp", "family": "B", "profile": "A", "short": "动态 peer 收益除以合法涨停幅度", "window": "D-1", "formula": "mean(ret_p,D-1 / legal_amp_p)；禁止 10% 与 20% 原始收益直接比", "inputs": ["daily_raw.close", "daily_raw.limit_up", "daily_raw.pre_close"], "duplicate_of": "check_vs_r21_n1", "label": None},
    {"factor_id": "r22_c1_xl_net_imbalance", "family": "C", "profile": "A", "short": "超大单净额占成交额", "window": "D close", "formula": "(BUY_VALUE_XL - SELL_VALUE_XL) / amount", "inputs": ["mkt_equ_mf_new", "daily_raw.amount"], "duplicate_of": "check_vs_h4_and_T", "label": None},
    {"factor_id": "r22_c2_ticket_size_innov", "family": "C", "profile": "A", "short": "笔均金额相对 20 日中位的变化", "window": "D-20..D", "formula": "(amount/deal_count) / median_20(amount/deal_count) - 1", "inputs": ["mkt_equd.DEAL_AMOUNT", "daily_raw.amount"], "duplicate_of": "check_vs_r21_t1", "label": None},
    {"factor_id": "r22_c3_buy_sell_count_gap", "family": "C", "profile": "A", "short": "买卖笔数差占合计", "window": "D close", "formula": "(buy_orders - sell_orders) / (buy_orders + sell_orders)", "inputs": ["mkt_equ_mf_new"], "duplicate_of": "check_vs_r21_g3", "label": None},
    {"factor_id": "r22_c4_flow_persist_5", "family": "C", "profile": "A", "short": "净流入方向 5 日持续", "window": "D-5..D-1", "formula": "mean(sign(net_inflow)_{D-5..D-1})", "inputs": ["mkt_equ_mf_new"], "duplicate_of": "check_vs_T", "label": None},
    {"factor_id": "r22_d1_minute_high_touch_rank", "family": "D", "profile": "A", "short": "分钟 high 首次触及的截面排名", "window": "D minute high", "formula": "rank_pct(min bar_sec where high>=limit_px) among same-day names; n<2 → NA", "inputs": ["min1.high", "limit_px"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_d2_minute_high_lead_min", "family": "D", "profile": "A", "short": "开盘至分钟 high 首触的分钟数", "window": "D minute high", "formula": "(first_high_touch_sec - 09:30) / 60", "inputs": ["min1.high", "limit_px"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_d3_lead_resid_past_only", "family": "D", "profile": "A", "short": "领先分钟对金额与幅度的过去残差", "window": "past 120d only", "formula": "D2 - E_past[D2 | log1p(amount), legal_amp]；系数只用 D 之前", "inputs": ["r22_d2", "daily_raw.amount", "legal_amp"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_d4_lead_x_seal_streak", "family": "D", "profile": "A", "short": "领先分钟与封板稳定度的预注册交互", "window": "D close", "formula": "r22_d2 * fb_close_seal_streak_norm", "inputs": ["r22_d2", "fb_close_seal_streak_norm"], "duplicate_of": None, "label": COMPOSITE_OF_KNOWN_FACTORS},
    {"factor_id": "r22_e1_days_since_halt", "family": "E", "profile": "A", "short": "距最近停牌的交易日", "window": "history through D-1", "formula": "trading days since last paused day; never → NA + NEVER_SUSPENDED", "inputs": ["daily_raw.paused", "md_sec_halt"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_e2_halt_count_60", "family": "E", "profile": "A", "short": "过去 60 日停牌次数", "window": "D-60..D-1", "formula": "count(paused)_{D-60..D-1}; never in history → NA + NEVER_SUSPENDED，禁止填 0", "inputs": ["daily_raw.paused"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_e3_resumed_within_5d", "family": "E", "profile": "A", "short": "是否在 5 个交易日内复牌", "window": "D-5..D-1", "formula": "1 if last halt ended within 5 sessions, 0 if halted earlier, NA if never", "inputs": ["daily_raw.paused"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_e4_last_halt_duration", "family": "E", "profile": "A", "short": "最近一段停牌的持续交易日", "window": "history through D-1", "formula": "length of the latest consecutive paused spell; never → NA", "inputs": ["daily_raw.paused"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_f1_lhb_net_buy", "family": "F", "profile": "B", "short": "龙虎榜净买额", "window": "post close", "formula": "on-list net buy known after D close; 未上榜 ≠ 0", "inputs": ["dragon_tiger"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_f2_lhb_inst_share", "family": "F", "profile": "B", "short": "龙虎榜机构席位占比", "window": "post close", "formula": "institution seat buy / on-list buy; 未上榜 → NA", "inputs": ["dragon_tiger"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_f3_lhb_appear_5d", "family": "F", "profile": "B", "short": "近 5 日上榜次数", "window": "post close, D-5..D", "formula": "count(on list)_{disclosed days}; 未上榜日不计为 0 金额", "inputs": ["dragon_tiger"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_f4_lhb_same_day_amt_ratio", "family": "F", "profile": "B", "short": "上榜日净买除以上榜日成交额", "window": "post close", "formula": "same-day on-list net / same-day amount；禁止多日累计除以单日成交额", "inputs": ["dragon_tiger", "daily_raw.amount"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_g1_theme_limit_breadth", "family": "G", "profile": "A", "short": "题材内收盘涨停广度", "window": "D-1 membership", "formula": "theme close-limit share using membership known at D-1; 无 INTO/OUT → BLOCKED_PIT", "inputs": ["theme.INTO_DATE", "theme.OUT_DATE"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_g2_theme_count", "family": "G", "profile": "A", "short": "个股所属题材数", "window": "D-1", "formula": "count(themes containing name at D-1); 日截面常数 → CROSS_SECTIONALLY_DEGENERATE", "inputs": ["theme.INTO_DATE"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_g3_theme_ret_d1", "family": "G", "profile": "A", "short": "题材 D-1 等权收益", "window": "D-1", "formula": "mean member ret on D-1; membership PIT-safe", "inputs": ["theme", "daily_raw.close"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_g4_theme_peer_overlap", "family": "G", "profile": "A", "short": "与同日首板共享题材数", "window": "D-1 membership", "formula": "count of shared themes with same-day candidates using D-1 membership", "inputs": ["theme.INTO_DATE"], "duplicate_of": None, "label": None},
    {"factor_id": "r22_h1_days_since_reduction_ann", "family": "H", "profile": "A", "short": "距减持公告日", "window": "announce date", "formula": "trading days since reduction announcement; 只有生效日不得 PIT_PASS", "inputs": ["announcement.announce_date"], "duplicate_of": "check_vs_L", "label": None},
    {"factor_id": "r22_h2_reduction_shares_to_float", "family": "H", "profile": "A", "short": "已公告减持股数除以流通股本", "window": "announce date", "formula": "announced shares / float; 计划上限 ≠ 已成交", "inputs": ["announcement", "share_capital"], "duplicate_of": "check_vs_r21_l1", "label": None},
    {"factor_id": "r22_h3_buyback_announced_not_done", "family": "H", "profile": "A", "short": "已公告未完成回购", "window": "announce date", "formula": "1 if buyback announced and not completed by D close, using announce date", "inputs": ["buyback.announce_date"], "duplicate_of": "check_vs_V", "label": None},
    {"factor_id": "r22_h4_supply_event_20d", "family": "H", "profile": "A", "short": "20 日内供给公告", "window": "announce date D-20..D", "formula": "1 if reduction or buyback announcement in D-20..D; 无公告日 → BLOCKED_PIT", "inputs": ["announcement.announce_date"], "duplicate_of": "check_vs_L_V", "label": None},
)

assert len(FACTOR_SPECS) == 32
FACTOR_IDS = tuple(s["factor_id"] for s in FACTOR_SPECS)
PROFILE_A_IDS = tuple(s["factor_id"] for s in FACTOR_SPECS if s["profile"] == "A")
PROFILE_B_IDS = tuple(s["factor_id"] for s in FACTOR_SPECS if s["profile"] == "B")

LEGACY_NEAR = (
    "fb_close_seal_streak_norm",
    "g0_amount_log",
    "h4_large_net_to_turnover",
    "h4_retail_net_to_turnover",
    "g3_days_since_limit",
    "ind_seal_rate_excess",
    "r21_n1_lagged_peer_limit_up",
    "r21_y1_turnover_innov_minus_peer",
    "r21_y2_turnover_accel_diff",
    "r21_y3_peer_quantile_exposure",
    "r21_y4_impact_diff_vs_peer",
    "r21_l1_known_float_chg",
    "r21_t1_ticket_vs_med20",
    "r21_t4_dlog_count_minus_dlog_amt",
    "r21_g1_avg_ticket_value",
    "r21_g3_order_count_imbalance",
)

FACTORY_STEPS_EXTRA = (
    "DATA_RECOVERY_BEFORE_COVERAGE_REJECT",
    "FORMULA_EQUIVALENCE_CHECK",
    "TRUE_DELETE_ASSERTION",
    "FEE_LEDGER_COMPLETENESS",
    "BOOTSTRAP_SPA_NUMERICAL",
    "SEED_STATUS_ENUM",
    "ECONOMIC_MATERIALITY_GATE",
    "FORWARD_AVAILABILITY",
    "PROFILE_SEPARATION",
    "HEAD_SPECIFIC_E0_E9",
)

TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
FACTOR_ZOO_ROOT = PROJECT_ROOT / "research" / "factor_zoo"
STANDARD_MD = PROJECT_ROOT / "docs" / "ALPHA_FACTOR_RESEARCH_STANDARD.md"
FINAL_REPORT = ROUND22_ROOT / "ROUND22_ALPHA_DISCOVERY_REPORT.md"
MANIFEST_PATH = ROUND22_ROOT / "round22_manifest.yaml"
E0_IMPORT_CANDIDATES = (
    ROUND21_4_ROOT / "candidates" / "R21_4_A0",
    ROUND21_4_ROOT / "candidates" / "R21_4_A7",
)

MD = {
    "00": ROUND22_ROOT / "00_research_contract.md",
    "parity": ROUND22_ROOT / "00_v5_parity.md",
    "01": ROUND22_ROOT / "01_prior_lineage.md",
    "02": ROUND22_ROOT / "02_literature_cards.md",
    "03": ROUND22_ROOT / "03_five_ten_contract.md",
    "04": ROUND22_ROOT / "04_profile_cutoff.md",
    "05": ROUND22_ROOT / "05_data_audit.md",
    "06": ROUND22_ROOT / "06_data_recovery.md",
    "07": ROUND22_ROOT / "07_forward_coverage.md",
    "08": ROUND22_ROOT / "08_hypotheses.md",
    "09": ROUND22_ROOT / "09_construct_abde.md",
    "10": ROUND22_ROOT / "10_construct_cfgh.md",
    "11": ROUND22_ROOT / "11_pit_coverage.md",
    "12": ROUND22_ROOT / "12_evidence_ic.md",
    "13": ROUND22_ROOT / "13_correlation.md",
    "14": ROUND22_ROOT / "14_experiment_registry.md",
    "15": ROUND22_ROOT / "15_oof_add.md",
    "16": ROUND22_ROOT / "16_remove_replace.md",
    "17": ROUND22_ROOT / "17_full_sleeve.md",
    "18": ROUND22_ROOT / "18_yearly_fee.md",
    "19": ROUND22_ROOT / "19_bootstrap_spa.md",
    "20": ROUND22_ROOT / "20_zoo.md",
    "21": ROUND22_ROOT / "21_final_decision.md",
}

STAGES = (
    "parity",
    "freeze",
    "prior",
    "recover",
    "construct_abde",
    "construct_cfgh",
    "pit",
    "evidence",
    "correlate",
    "train",
    "backtest",
    "stats",
    "zoo",
    "report",
)


def exp_dir(code: str) -> Path:
    return ROUND22_ROOT / "candidates" / str(code)


def overlayed_path(code: str) -> Path:
    return ROUND22_ROOT / "weights" / f"overlayed_{code}.parquet"


def model_parameters(seed: int, *, n_jobs: int = XGB_N_JOBS) -> dict[str, Any]:
    return {
        "objective": "rank:ndcg",
        "eval_metric": None,
        "ndcg_exp_gain": True,
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_child_weight": 1.0,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "tree_method": "hist",
        "random_state": int(seed),
        "n_jobs": int(n_jobs),
    }


def spec_by_id(factor_id: str) -> dict[str, Any]:
    for spec in FACTOR_SPECS:
        if spec["factor_id"] == factor_id:
            return dict(spec)
    raise KeyError(factor_id)
