"""R24 Migration V2 契约。严格时点失败则不得运行名为 FULL 的模型。"""

from __future__ import annotations

from core.paths import RESULT_ROOT

ROOT = RESULT_ROOT / "round24_v5_migration_v2"
ANCHOR_BASELINE_V5 = "feda6870a26f6efe5d722fd9c3b11130ca266d85"
ANCHOR_MIGRATION = "68efe0a29c2b89c044ae916af3981e49ab5be345"
DECISION_HOUR = 22
EVAL_START = "2018-01-02"
EVAL_END = "2026-07-24"
INITIAL_CASH = 300_000.0

HERMES_V7 = "/data1/stock-mxy/data/datasets/hermes-cn-a-2010-20260724-v7-all-features"
HERMES_RAW = "/data1/Stock/data/acquisitions/hermes-cn-a-2010-20260724-v7-rebuild/raw"

FIVE_THEORY = (
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
    "r24v2_days_since_limit_before_current_streak",
    "ind_seal_rate_excess",
)
TEN_THEORY = tuple(c for c in FIVE_THEORY if c != "g0_amount_log")
