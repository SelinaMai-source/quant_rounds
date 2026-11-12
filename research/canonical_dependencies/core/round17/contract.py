"""Round17 冻结契约。孤立 0.8/0.8 仅本命名空间。默认 KEEP_BASELINE_V4。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT, RESULT_ROOT, ROUND16_5_ROOT, ROUND17_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, SEEDS, TEN_FEATURES, TEN_HEAD
from ..round11.contract import (
    PARITY_ABS_NAV,
    PARITY_METRIC_ABS,
    SNAPSHOT_FINGERPRINT,
    SNAPSHOT_ID,
    STRATEGY_SOURCE_FILES,
)
from ..round12.contract import RANK_NAV, TOP_K
from ..round16_5.contract import V4_ANCHOR as R16_5_V4_ANCHOR

KEEP_BASELINE_V4 = "KEEP_BASELINE_V4"
RECOMMENDED_BASELINE_V5 = "RECOMMENDED_BASELINE_V5"
AWAITING_USER_LOCK = "AWAITING_USER_LOCK"
AUTO_PROMOTE_V5 = False
PARENT_V4 = "R9_B03"
CURRENT_BASELINE = "baseline_v4"
OPTION_CANDIDATE = "R16_O01"
FORWARD_OOS_NOT_AVAILABLE = "FORWARD_OOS_NOT_AVAILABLE"
HISTORICAL_RETROSPECTIVE_AUDIT = "HISTORICAL_RETROSPECTIVE_AUDIT"
SEED_DIVERSITY_DEGENERATE = "SEED_DIVERSITY_DEGENERATE"
SEED_DIVERSITY_PASS = "SEED_DIVERSITY_PASS"
CONTRACT_BAGGING = "R17_STOCHASTIC_BAGGING_08_08"
CONTRACT_DETERMINISTIC = "R17_DETERMINISTIC_10_10"
LIVE_NOT_VALIDATED = "NOT_VALIDATED"

MANIFEST_NAME = "round17_search_manifest.yaml"
CURSOR_PLAN = Path("/home/mxy/.cursor/plans/round17_factor_search_fceb87cf.plan.md")
PLAN_COPY = ROUND17_ROOT / "ROUND17_PLAN.md"
TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
FINAL_REPORT = ROUND17_ROOT / "ROUND17_LARGE_SCALE_FACTOR_SEARCH_REPORT.md"

A00 = "R17_A00"
A01 = "R17_A01"

V4_ANCHOR = dict(R16_5_V4_ANCHOR)

DEV_START = date(2018, 1, 1)
DEV_END = date(2021, 12, 31)
VAL_START = date(2022, 1, 1)
VAL_END = date(2023, 12, 31)
AUDIT_START = date(2018, 1, 1)
AUDIT_END = date(2026, 7, 24)
FORWARD_AFTER = date(2026, 7, 24)

SEEDS = tuple(SEEDS)
FIVE_FEATURES = tuple(FIVE_FEATURES)
TEN_FEATURES = tuple(TEN_FEATURES)
FIVE_HEAD = FIVE_HEAD
TEN_HEAD = TEN_HEAD

N_TRIALS = 5000
BUDGET_TEN_ONLY = 3500
BUDGET_FIVE_ONLY = 1000
BUDGET_BOTH = 500
MANIFEST_SEED = 20261017
EB_PRIOR_N = 20.0
DISCOVERY_XGB_N_JOBS = 1
MAXDD_WORSEN_CAP = 0.02
SELECTION_WEIGHTS = {
    "dev_sharpe": 0.40,
    "dev_median_year": 0.25,
    "dev_worst_year": 0.20,
    "dev_OOS_NDCG": 0.15,
}
SIZE_MIX = {1: 0.15, 2: 0.25, 3: 0.35, 4: 0.25}
BLOCK_DAYS = 20
BOOT_N = 5000
SPA_BOOT = 2000
EMBARGO = 12
MIN_BLOCK = 60
COST_BPS = (20.0, 30.0)
CAPACITY_NAV = (300_000.0, 1_000_000.0, 3_000_000.0)
CAPACITY_DIAG_NAV = 10_000_000.0
PARITY_ABS_NAV = PARITY_ABS_NAV
PARITY_METRIC_ABS = PARITY_METRIC_ABS
SNAPSHOT_ID = SNAPSHOT_ID
SNAPSHOT_FINGERPRINT = SNAPSHOT_FINGERPRINT
RANK_NAV = RANK_NAV
TOP_K = TOP_K

PROMOTION_GATES = {
    "same_window_sharpe_competitive": True,
    "maxdd_worsen_cap": MAXDD_WORSEN_CAP,
    "not_only_2024_2025": True,
    "seed_diversity_pass": True,
    "plus20_plus30_no_sign_flip": True,
    "disclose_selection_bias": True,
    "a01_luck_not_promotion": True,
    "r6_a04_bagging_only_not_alpha": True,
    "do_not_merge_r16_o01": True,
    "do_not_auto_promote_current": True,
    "current_default": CURRENT_BASELINE,
    "keep_default": KEEP_BASELINE_V4,
}

STAGES = (
    "read",
    "parity",
    "preflight",
    "inventory",
    "factors",
    "pit",
    "freeze",
    "budget",
    "search",
    "select",
    "validate",
    "fourway",
    "stats",
    "forward",
    "report",
)

PROTOCOL_SOURCE_FILES = (
    "core/round17/__init__.py",
    "core/round17/contract.py",
    "core/round17/common.py",
    "core/round17/read.py",
    "core/round17/parity.py",
    "core/round17/diversity.py",
    "core/round17/inventory.py",
    "core/round17/factors.py",
    "core/round17/pit.py",
    "core/round17/freeze.py",
    "core/round17/train.py",
    "core/round17/ensemble.py",
    "core/round17/backtest.py",
    "core/round17/search.py",
    "core/round17/select.py",
    "core/round17/fourway.py",
    "core/round17/stats.py",
    "core/round17/decision.py",
    "core/round17/figures.py",
    "core/round17/docs.py",
    "core/round17/pipeline.py",
    "tools/run_round17.py",
)

FORBIDDEN = (
    "edit_AGENTS_md",
    "edit_no_xgb_subsample_rule",
    "edit_core_round7",
    "edit_stock_src_or_runs",
    "fill_missing_features_with_zero",
    "search_threshold_lookback_top2_hyperparams",
    "replay_historical_44_subset_search",
    "merge_r16_o01_option_overlay",
    "modify_option_candidate",
    "assert_seed_rank_invariance_on_08_08",
    "add_noise_to_fake_diversity",
    "change_hyperparams_to_fake_bagging",
    "promote_on_a01_luck",
    "silently_reduce_5000_to_500",
    "fill_failed_metrics_with_zero",
    "auto_promote_CURRENT_v5",
    "write_写入时间_to_tracker",
    "call_2018_2026_untouched_oos",
    "call_2022_2023_independent_oos",
    "call_cpcv_forward_oos",
)

XGB_BASE = {
    "objective": "rank:ndcg",
    "ndcg_exp_gain": True,
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 2,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "tree_method": "hist",
}


def model_parameters(seed: int, *, subsample: float, colsample_bytree: float, n_jobs: int) -> dict[str, Any]:
    return {
        **XGB_BASE,
        "eval_metric": None,
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "random_state": int(seed),
        "n_jobs": int(n_jobs),
    }


def copy_plan() -> Path:
    from ..paths import assert_project_result_path, ensure_round17_tree

    ensure_round17_tree()
    assert_project_result_path(PLAN_COPY)
    text = CURSOR_PLAN.read_text(encoding="utf-8")
    header = "<!-- 本文件是开工计划的课题落盘副本。禁止回写 Cursor plan 文件。 -->\n\n"
    if not text.startswith("<!-- 本文件是开工计划"):
        PLAN_COPY.write_text(header + text, encoding="utf-8")
    else:
        PLAN_COPY.write_text(text, encoding="utf-8")
    return PLAN_COPY


def trial_dir(trial_id: str) -> Path:
    return ROUND17_ROOT / "search" / "trials" / str(trial_id)


def fourway_dir(code: str) -> Path:
    return ROUND17_ROOT / "fourway" / str(code)


def parity_dir(code: str = A00) -> Path:
    return ROUND17_ROOT / "parity" / str(code)
