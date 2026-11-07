"""baseline_v5 冻结契约。权威数字来自 ROUND7_FLAT_5BP 全历史回放。"""

from __future__ import annotations

import hashlib
from typing import Any

IDENTITY = "baseline_v5"
ALIAS = "R16_O01"
PARENT = "R9_B03"
CURRENT_BASELINE_CHANGED = False
PRODUCTION_CONFIG_CHANGED = False
AUTO_PROMOTE_CURRENT = False

MAIN_WINDOW_START = "2018-01-02"
MAIN_WINDOW_END = "2026-07-24"
INITIAL_CAPITAL = 300000.0
FEE_SCHEDULE = "ROUND7_FLAT_5BP"
INFORMATION_LANE = "LEGACY_OPEN_INFORMATION_LANE"
DECISION_TIME = "D_close"
EXECUTION_TIME = "D+1_open"
USES_D1_OPEN_GAP = True

O01_FORMULA = 'PUT_SKEW25.shift(1) > rolling(252, min_periods=126).quantile(0.90) → apply_top2_veto rank=2 MIN_KEEP'
FUSION_WEIGHTS = {"ten": 0.90, "five": 0.10}
GAP_FLOOR = -0.01
SEEDS = (17, 42, 73, 101, 202)
MIN_KEEP = 1e-12

FIVE_FEATURES = ('g_prim_px_vol_corr', 'fb_close_seal_streak_norm', 'fb_last30_amount_ratio', 'h5_last_first_limit_value_change_sym', 'h1_consistent_buy_ratio', 'h1_last15m_volume_ratio', 'h4_large_net_to_turnover', 'h4_retail_net_to_turnover', 'g0_amount_log', 'g0_amt_vs_5', 'g1_ret_5', 'g3_days_since_limit', 'ind_seal_rate_excess', 'd1_open_gap')
TEN_FEATURES = ('g_prim_px_vol_corr', 'fb_close_seal_streak_norm', 'fb_last30_amount_ratio', 'h5_last_first_limit_value_change_sym', 'h1_consistent_buy_ratio', 'h1_last15m_volume_ratio', 'h4_large_net_to_turnover', 'h4_retail_net_to_turnover', 'g0_amt_vs_5', 'g1_ret_5', 'g3_days_since_limit', 'ind_seal_rate_excess', 'd1_open_gap')

FIVE_LABEL = "label_e2_net"
FIVE_LABEL_END = "next2_date"
FIVE_RANK_BINS = 5
FIVE_EVAL_METRIC = "ndcg@1"

TEN_LABEL = "label_exit_rule_net"
TEN_LABEL_END = "actual_exit_date"
TEN_RANK_BINS = 10
TEN_EVAL_METRIC = "ndcg@2"

XGB_PARAMS = {
    "objective": "rank:ndcg",
    "eval_metric": None,
    "ndcg_exp_gain": True,
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 2,
    "min_child_weight": 1.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "tree_method": "hist",
    "n_jobs": 2,
}

FEE_COMPONENTS = {
    "slippage_bps": 12.3,
    "commission_bps": 1.0,
    "transfer_fee_bps": 0.1,
    "sell_tax_bps": 5.0
}

PORTFOLIO = {
    "sleeve_count": 2,
    "sleeve_cash": 150000.0,
    "top_k": 2,
    "rank_name_budget_nav_fractions": (0.3, 0.1),
    "name_budget": 10000.0,
    "lot_size": 100,
    "exit_mode": "close_from_d2",
    "limit_up_tolerance": 0.005,
    "clean_risk": True,
    "allocator": "B03",
    "o01_overlay": True,
}

B03_CELL = {
    ("A", "LOW"): 0.00, ("A", "MID"): 0.05, ("A", "HIGH"): 0.10,
    ("B", "LOW"): 0.00, ("B", "MID"): 0.00, ("B", "HIGH"): 0.05,
    ("C", "LOW"): 0.00, ("C", "MID"): 0.00, ("C", "HIGH"): 0.00,
}

V5_ANCHOR = {
    "cagr": 0.5499812832273525,
    "sharpe": 1.8756627913361097,
    "max_drawdown": -0.24056327715394388,
    "completed_trades": 1107,
    "n_obs": 2075,
    "nav_end": 12750089.355724689,
    "initial_capital": 300000.0,
    "window_start": MAIN_WINDOW_START,
    "window_end": MAIN_WINDOW_END,
    "fee_schedule": FEE_SCHEDULE,
}

TRAINING_FOLDS = {
    "scheme": "walk_forward_Y_uses_Y-3..Y-1_mature_labels",
    "first_prediction_year": 2018,
    "last_prediction_year": 2026,
    "note": "预测年 Y 的训练窗为 Y-3..Y-1，且 label_end_column 必须在训练 cutoff 前已成熟；group 按 signal_date。",
}

AUTHORITATIVE_ARTIFACTS = {
    "replay_r7_caliber": "result/replay_r7_caliber/v5_R16_O01",
    "parity_round22_3": "result/round22_3/parity/R223_V5_R16_O01",
    "overlay": "result/replay_r7_caliber/overlayed_v5_R16_O01.parquet",
}


def feature_order_hash(features: tuple[str, ...] | list[str]) -> str:
    payload = "\n".join(features).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def frozen_feature_hashes() -> dict[str, str]:
    return {
        "five14": feature_order_hash(FIVE_FEATURES),
        "ten13": feature_order_hash(TEN_FEATURES),
    }
