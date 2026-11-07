"""Frozen contract for the clean dual-head candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ..paths import stock_path

CANDIDATE_CODE = "R7_FS01122_DUAL90_10_CLEAN"
SEEDS = (17, 42, 73, 101, 202)
DEV_START = "2018-01-01"
DEV_END = "2021-12-31"
GAP_FLOOR = -0.01
TEN_WEIGHT = 0.90

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

FIVE_PANEL = stock_path(
    "data/first_board_hermes/"
    "aligned_original_candidate_full44_plus_d1_open_gap_v1"
)
TEN_PANEL = stock_path(
    "data/first_board_hermes/"
    "aligned_original_candidate_fs01122_chain_label_v1"
)
MARKET_CONTEXT = stock_path(
    "data/factor_cache/first_board_single_model_combined_2013_20260724_v1/"
    "market_context.parquet"
)
LIMIT_DOWN_CONTEXT = stock_path(
    "data/factor_cache/first_board_market_limit_down_2013_20260724_v1/"
    "market_limit_down_context.parquet"
)


@dataclass(frozen=True)
class HeadContract:
    name: str
    panel_dir: Path
    features: tuple[str, ...]
    target: str
    label_end_column: str
    rank_n_bins: int
    eval_metric: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["panel_dir"] = str(self.panel_dir)
        return payload


FIVE_HEAD = HeadContract(
    name="five",
    panel_dir=FIVE_PANEL,
    features=FIVE_FEATURES,
    target="label_e2_net",
    label_end_column="next2_date",
    rank_n_bins=5,
    eval_metric="ndcg@1",
)
TEN_HEAD = HeadContract(
    name="ten",
    panel_dir=TEN_PANEL,
    features=TEN_FEATURES,
    target="label_exit_rule_net",
    label_end_column="actual_exit_date",
    rank_n_bins=10,
    eval_metric="ndcg@2",
)


def model_parameters(seed: int) -> dict:
    return {
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
        "random_state": int(seed),
        "n_jobs": 2,
    }
