"""Round 24.1 契约。二进三独立于一进二，不微调一进二模型。"""

from __future__ import annotations

from ..paths import RESULT_ROOT

ROOT = RESULT_ROOT / "round24_1"
ROUND24 = RESULT_ROOT / "round24"
FULL_FEATURES = (
    "r241_second_board_amount_log",
    "r241_second_board_turnover_surprise",
    "r241_second_board_return_5",
    "r241_first_to_second_amount_ratio",
    "r241_second_board_first_touch_time",
    "r241_second_board_seal_duration",
    "r241_second_board_market_breadth",
    "r241_second_board_peer_strength",
)
REDUCED_FEATURES = (
    "r241_second_board_amount_log",
    "r241_second_board_turnover_surprise",
    "r241_second_board_return_5",
    "r241_first_to_second_amount_ratio",
    "r241_second_board_market_breadth",
)
ABSTAIN = (0.50, 0.55, 0.60)
XGB_TREE = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 2,
    "min_child_weight": 1,
    "reg_lambda": 1,
    "subsample": 1,
    "colsample_bytree": 1,
    "seed": 17,
}
MIN_NAMES_FOR_RANKIC = 5
MIN_INNER_DAYS = 120
FEATURE_COVERAGE_MIN = 0.50
GATE = {
    "cagr_min": 0.0,
    "sharpe_min": 0.5,
    "maxdd_floor": -0.40,
    "catastrophic_maxdd": -0.60,
    "catastrophic_cagr": -0.20,
}
HYPOTHESES = (
    {"id": "H241_BOARD", "claim": "二板状态含 2026-07-06 主板 ST 规则，且没有用统一的 1.099 阈值"},
    {"id": "H241_NO_DOUBLE_FEE", "claim": "训练标签里的成本不会在净值里再扣一次"},
    {"id": "H241_FEATURES", "claim": "只用规定的 8 个特征；缺失则降为 F1–F4+F7，不填 0"},
    {"id": "H241_STAGE", "claim": "A1–A4 在二进三门上同时过 T0 与 T2 的配对增量、费用和保守成交"},
)
