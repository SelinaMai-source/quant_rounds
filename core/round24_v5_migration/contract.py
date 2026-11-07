"""Round24 V5 migration 冻结契约。"""

from __future__ import annotations

from core.paths import RESULT_ROOT

ROOT = RESULT_ROOT / "round24_v5_migration"
R24_2 = RESULT_ROOT / "round24_2"
R24 = RESULT_ROOT / "round24"
BASELINE_AUDIT = ROOT / "baseline_audit"

PERF_START = "2018-01-02"
PERF_END = "2026-07-24"
CAND_START = "2015-01-01"
WARMUP_TRAIN_START = "2015-01-01"

# CORE3：二板版本的 v5 日线三因子
CORE3 = (
    "g0_amount_log",
    "g0_amt_vs_5",
    "g1_ret_5",
)

# TARGET7：沿用 round24.2 定义（tail seal = 收盘前最终连续封板）
TARGET7 = (
    "second_board_amount_log",
    "second_board_turnover_surprise",
    "second_board_return_5",
    "first_to_second_amount_ratio",
    "second_board_market_breadth",
    "second_board_first_touch_time",
    "second_board_tail_seal_fraction",
)

# V5_MIGRATED：D-close 合法迁移因子（排除 d1_open_gap）
# 分钟/封单类若数据不足则保留列但允许 NaN；不得为提分删列
V5_MIGRATED_PLANNED = (
    "g_prim_px_vol_corr",
    "fb_close_seal_streak_norm",
    "fb_last30_amount_ratio",
    "h5_last_first_limit_value_change_sym",
    "h1_consistent_buy_ratio",
    "h1_last15m_volume_ratio",
    "h4_large_net_to_turnover",
    "h4_retail_net_to_turnover",
    "g0_amount_log",
    "g0_amt_vs_5",
    "g1_ret_5",
    "g3_days_since_prev_limit_before_streak",  # 新 Factor ID
    "ind_seal_rate_excess",
)

EXECUTION_ONLY = ("d1_open_gap",)

HYPOTHESES = (
    {"id": "H24M_FACTOR", "claim": "逐因子按 baseline_v5 源码迁移，公式未核实则 UNVERIFIED/BLOCKED", "pass_if": "mapping 表完整"},
    {"id": "H24M_LABEL", "claim": "标签唯一来自 simulate_single_opportunity，无 next2_date 成熟捷径", "pass_if": "parity>=100"},
    {"id": "H24M_DUAL", "claim": "Five/Ten 双头可在剔除 d1_open_gap 后重训", "pass_if": "V5DUAL ran 或 BLOCKED_FEATURE"},
)
