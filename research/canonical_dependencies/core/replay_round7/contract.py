"""Round7 锁定契约。预测只读 Stock result/round7，正式账本写课题 replay_round7。"""

from __future__ import annotations

from pathlib import Path

from ..paths import STOCK_ROUND7_ROOT

CANDIDATE_CODE = "R7_FS01122_DUAL90_10_CLEAN"
SEEDS = (17, 42, 73, 101, 202)
TEN_WEIGHT = 0.90
GAP_FLOOR = -0.01
TOP_K = 2
RANK_NAV = (0.3, 0.1)
DEV_START = "2018-01-01"
DEV_END = "2021-12-31"

TARGET_TOTAL_RETURN = 25.979409141148523
TARGET_N_TRADES = 2158
TARGET_MAXDD = -0.25415996098636673
TARGET_SHARPE = 1.6684780129023091
TARGET_CAGR = 0.46980280179436673
PARITY_ABS_NAV = 1e-4

STOCK_R7 = STOCK_ROUND7_ROOT
BLEND_PRED = STOCK_R7 / "predictions" / "dual90_10_gap_floor" / "predictions.parquet"
RISK_PRED = STOCK_R7 / "predictions" / "clean_risk" / "predictions.parquet"
DAILY_STATE = STOCK_R7 / "predictions" / "clean_risk" / "daily_state.parquet"
CANDIDATE_LEDGER = STOCK_R7 / CANDIDATE_CODE

FIVE_FEATURES = (
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
    "g3_days_since_limit",
    "ind_seal_rate_excess",
    "d1_open_gap",
)
TEN_FEATURES = tuple(name for name in FIVE_FEATURES if name != "g0_amount_log")

TEN_PANEL = Path(
    "/home/mxy/Stock/data/first_board_hermes/"
    "aligned_original_candidate_fs01122_chain_label_v1"
)
MARKET_CONTEXT = Path(
    "/home/mxy/Stock/data/factor_cache/first_board_single_model_combined_2013_20260724_v1/"
    "market_context.parquet"
)

EXIT_OVERLAY_KINDS = frozenset({"exit_regime", "exit_low_conf", "exit_ma5"})
LOCKED_RISK_KINDS = frozenset({None, "", "identity"}) | EXIT_OVERLAY_KINDS

F_ABORT_PREFIXES = (
    "R4_F_",
    "R4_TOP2_diag",
    "R4_P_11",
    "R4_P_12",
    "R4_T06",
)
