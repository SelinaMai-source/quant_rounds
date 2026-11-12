"""Round 21.1 冻结契约。不改 14/13 常量。不自动升 v6。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT, RESULT_ROOT, ROUND21_1_ROOT, ROUND21_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, SEEDS, TEN_FEATURES, TEN_HEAD
from ..round21.contract import (
    AUTO_PROMOTE_CURRENT,
    COLSAMPLE,
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    FEE_SCHEDULE,
    KEEP_BASELINE_V5,
    NAV_END_ABS_TOL,
    NOT_TRIGGERED,
    O01_FORMULA,
    PARITY_METRIC_ABS,
    RECOMMEND_NEW_ALPHA_CANDIDATE,
    SUBSAMPLE,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    XGB_N_JOBS,
)

KEEP_BASELINE_V5 = KEEP_BASELINE_V5
RECOMMEND_NEW_ALPHA_CANDIDATE = RECOMMEND_NEW_ALPHA_CANDIDATE
AWAITING_USER_REVIEW = "AWAITING_USER_REVIEW"
AUTO_PROMOTE_CURRENT = AUTO_PROMOTE_CURRENT
CURRENT_BASELINE = CURRENT_BASELINE
CURRENT_ALIAS = CURRENT_ALIAS
NOT_TRIGGERED = NOT_TRIGGERED
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
BLOCKED_PIT = "BLOCKED_PIT"
PIT_ELIGIBLE_MODEL = "PIT_ELIGIBLE_MODEL"
NEAREST_LEGACY_NOT_IDENTIFIABLE = "NEAREST_LEGACY_NOT_IDENTIFIABLE"
IDENTICAL_BY_DESIGN = "IDENTICAL_BY_DESIGN"
REPLACE_NOT_APPLIED = "REPLACE_NOT_APPLIED"
REPEATED_HISTORICAL_TESTING = "REPEATED_HISTORICAL_TESTING"
FORWARD_OOS_NOT_AVAILABLE = "FORWARD_OOS_NOT_AVAILABLE"
HISTORICAL_FEE_CORRECTED = "HISTORICAL_FEE_CORRECTED"
EXPLORATORY_ONLY = "EXPLORATORY_ONLY"

FIVE_FEATURES = tuple(FIVE_FEATURES)
TEN_FEATURES = tuple(TEN_FEATURES)
SHARED_13 = tuple(TEN_FEATURES)
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

PIT_MIN_FINITE = 0.80
CLUSTER_SPEARMAN_ABS = 0.85
RIDGE_ALPHA = 1.0
BLOCK_DAYS = 20
BOOT_N = 5000
SPA_BOOT = 2000
IC_YEARS = tuple(range(2018, 2027))
PREDICT_YEARS = tuple(range(2018, 2027))
LABELS = ("label_e2_net", "label_exit_rule_net")
KEYS = ["trade_date", "instrument_id"]

E0_CODE = "R21_1_E0"
A00 = "R21_1_A00"

COVERAGE_CLASSES = (
    "STRUCTURAL_NOT_APPLICABLE",
    "EVENT_CONDITIONAL",
    "TRUE_DATA_MISSING",
    "SOURCE_ALIGNMENT_FAILURE",
    "FORMULA_UNDEFINED",
    "TIMESTAMP_UNVERIFIED",
)

PIT_PASS_R21 = (
    "r21_a3_retouch_count",
    "r21_a4_tail_seal_share",
    "r21_b1_fb_amount_share",
    "r21_b2_seal_rel_strength",
    "r21_b3_amt_surprise_vs_peer",
    "r21_c1_min1_amihud",
    "r21_d1_log_holder_ratio",
    "r21_d2_holder_daily_chg",
    "r21_d3_holder_per_circ",
    "r21_d4_holder_x_float",
)
PIT_FAIL_COVERAGE_R21 = (
    "r21_a1_vw_limit_gap",
    "r21_a2_minutes_to_reseal",
    "r21_b4_activity_x_resilience",
    "r21_c2_post_touch_amt_vs_impact",
    "r21_c3_post_touch_path_eff",
    "r21_c4_downside_quad_var",
    "r21_f1_fin_bal_chg",
    "r21_f2_fin_repay_pressure",
    "r21_f3_fin_buy_intensity",
    "r21_f4_fin_bal_accel",
)
BLOCKED_PIT_R21 = (
    "r21_e1_block_discount",
    "r21_e2_block_vol20_to_circ",
    "r21_e3_discount_wtd_pressure",
    "r21_e4_block_event_intensity_60",
)

FACTORY_STEPS_EXTRA = (
    "NEW_VS_BASELINE_CORRELATION",
    "NEAREST_LEGACY_FACTOR_IDENTIFICATION",
    "FOUR_WAY_REPLACEMENT_TEST",
    "CLUSTER_REPLACEMENT_IF_APPLICABLE",
    "OOF_INCREMENTALITY",
    "FULL_PORTFOLIO_INCREMENTALITY",
)

TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
FACTOR_ZOO_ROOT = PROJECT_ROOT / "research" / "factor_zoo"
STANDARD_MD = PROJECT_ROOT / "docs" / "ALPHA_FACTOR_RESEARCH_STANDARD.md"
CURSOR_PLAN = Path("/home/mxy/.cursor/plans/round21.1_replace_forensics_cb88b7cc.plan.md")
PLAN_COPY = ROUND21_1_ROOT / "ROUND21_1_PLAN.md"
FINAL_REPORT = ROUND21_1_ROOT / "ROUND21_1_RESEARCH_REPORT.md"

PARITY_MD = ROUND21_1_ROOT / "00_v5_parity.md"
CONTRACT_MD = ROUND21_1_ROOT / "00_research_contract.md"
COVERAGE_MD = ROUND21_1_ROOT / "01_coverage_forensics.md"
AUDIT_MD = ROUND21_1_ROOT / "02_round21_experiment_parity.md"
CORR_MD = ROUND21_1_ROOT / "03_correlation.md"
NEAREST_MD = ROUND21_1_ROOT / "04_fold_nearest.md"
TRAIN_MD = ROUND21_1_ROOT / "05_four_way_train.md"
COND_MD = ROUND21_1_ROOT / "06_conditional_ic.md"
ATTR_MD = ROUND21_1_ROOT / "07_attribution.md"
WF_MD = ROUND21_1_ROOT / "08_walk_forward.md"
CLUSTER_MD = ROUND21_1_ROOT / "09_cluster.md"
SEED_MD = ROUND21_1_ROOT / "10_seed_robustness.md"
BACKTEST_MD = ROUND21_1_ROOT / "11_backtest_decision.md"
STATS_MD = ROUND21_1_ROOT / "12_stats.md"
ZOO_MD = ROUND21_1_ROOT / "13_factor_zoo.md"

STAGES = (
    "parity",
    "coverage",
    "audit",
    "correlate",
    "train",
    "conditional",
    "backtest",
    "attribution",
    "stats",
    "seed",
    "zoo",
    "report",
)

INCREMENT_GATE = {
    "id": "R21_1_OOF_INCREMENT_V1",
    "compare_to": E0_CODE,
    "metric": "oof_rankic_mean",
    "not_worse_both_heads": True,
    "strict_better_any_head": True,
    "min_oos_days": 200,
    "min_delta": 0.0,
    "eps": 1e-12,
    "seed_only_if_gate_hit": True,
}


def exp_dir(code: str) -> Path:
    return ROUND21_1_ROOT / "candidates" / str(code)


def overlayed_path(code: str) -> Path:
    return ROUND21_1_ROOT / "weights" / f"overlayed_{code}.parquet"


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


def copy_plan() -> Path | None:
    from ..paths import assert_project_result_path, ensure_round21_1_tree
    from .common import write_md

    ensure_round21_1_tree()
    assert_project_result_path(PLAN_COPY)
    if not CURSOR_PLAN.is_file():
        return None
    text = CURSOR_PLAN.read_text(encoding="utf-8")
    header = "<!-- 本文件是开工计划的课题落盘副本。禁止回写 Cursor plan 文件。 -->\n\n"
    if not text.startswith("<!-- 本文件是开工计划"):
        write_md(PLAN_COPY, header + text)
    else:
        write_md(PLAN_COPY, text)
    return PLAN_COPY
