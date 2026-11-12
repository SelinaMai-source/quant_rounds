"""Pure-Hermes first-board research dataset construction."""

from .dataset import BASE_FEATURES, HERMES_FEATURES, build_dataset
from .baseline import (
    FEATURES, FEATURE_SETS, RANKER_PRESETS,
    RankerConfig,
    daily_five_bin_relevance,
    daily_ordinal_relevance,
    daily_relevance,
    in_memory_complete_cases,
    rolling_train_predict,
)
from .labels import (
    ExitLabelCosts,
    attach_board_chain_labels,
    attach_exit_aligned_labels,
)
from .fixed_sleeve_backtest import (
    FixedSleeveConfig,
    PreparedFixedSleeveMarket,
    prepare_fixed_sleeve_market,
    run_fixed_dual_sleeve_top3_backtest,
)

__all__ = [
    "BASE_FEATURES", "HERMES_FEATURES", "build_dataset", "FEATURES",
    "RankerConfig", "daily_five_bin_relevance", "daily_ordinal_relevance",
    "daily_relevance", "in_memory_complete_cases",
    "rolling_train_predict", "FEATURE_SETS", "RANKER_PRESETS",
    "ExitLabelCosts", "attach_board_chain_labels", "attach_exit_aligned_labels",
    "FixedSleeveConfig", "PreparedFixedSleeveMarket", "prepare_fixed_sleeve_market",
    "run_fixed_dual_sleeve_top3_backtest",
]
