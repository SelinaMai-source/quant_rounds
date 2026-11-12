"""Round20 冻结契约。IC/筛选/预算/seed/晋级门在看新候选 NAV 之前锁定。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT, RESULT_ROOT, ROUND17_ROOT, ROUND18_ROOT, ROUND20_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, SEEDS, TEN_FEATURES, TEN_HEAD
from ..round8.overlays import MIN_KEEP
from ..round11.contract import PARITY_ABS_NAV, PARITY_METRIC_ABS, SNAPSHOT_FINGERPRINT, SNAPSHOT_ID
from ..round12.contract import RANK_NAV, TOP_K
from ..round16_5.contract import O01_ANCHOR as R16_5_O01_ANCHOR
from ..round18.contract import (
    F1_FORMULA,
    F1_NAME,
    F2_FORMULA,
    F2_NAME,
    F3_FORMULA,
    F3_NAME,
)
from ..round18.contract import PIT_TIMESTAMP_UNVERIFIED as R18_PIT_TIMESTAMP_UNVERIFIED
from ..round19.contract import O01_FORMULA

KEEP_BASELINE_V5 = "KEEP_BASELINE_V5"
RECOMMEND_NEW_ALPHA_CANDIDATE = "RECOMMEND_NEW_ALPHA_CANDIDATE"
AWAITING_USER_REVIEW = "AWAITING_USER_REVIEW"
AUTO_PROMOTE_CURRENT = False
PARENT_V4 = "R9_B03"
CURRENT_BASELINE = "baseline_v5"
CURRENT_ALIAS = "R16_O01"
FORWARD_OOS_NOT_AVAILABLE = "FORWARD_OOS_NOT_AVAILABLE"
PRIOR_RESEARCH_SELECTION_EXISTS = True
PIT_TIMESTAMP_UNVERIFIED = R18_PIT_TIMESTAMP_UNVERIFIED
FORMAL_PROMOTION_BLOCKED = "FORMAL_PROMOTION_BLOCKED"
SKIP_DEPENDENCY = "SKIP_DEPENDENCY"
NOT_TRIGGERED = "NOT_TRIGGERED"
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"
HISTORICAL_FEE_CORRECTED_SENSITIVITY = "HISTORICAL_FEE_CORRECTED_SENSITIVITY"
NATIVE_SEED_DIVERSITY_AUDIT = "NATIVE_SEED_DIVERSITY_AUDIT"

MANIFEST_NAME = "round20_manifest.yaml"
CURSOR_PLAN = Path("/home/mxy/.cursor/plans/round20_factor_diagnostics_31c04dd8.plan.md")
PLAN_COPY = ROUND20_ROOT / "ROUND20_PLAN.md"
TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
FINAL_REPORT = ROUND20_ROOT / "ROUND20_FACTOR_RESEARCH_REPORT.md"
PARITY_MD = ROUND20_ROOT / "00_v5_parity.md"
CONTRACT_MD = ROUND20_ROOT / "00_research_contract.md"
INVENTORY_MD = ROUND20_ROOT / "01_factor_inventory.md"
R17_REGISTRY = ROUND17_ROOT / "inventory" / "factor_registry.csv"
R18_PIT_JSON = ROUND18_ROOT / "pit" / "pit.json"

A00 = "R20_A00"
A01 = "R20_A01"
B01 = "R20_B01"
B02 = "R20_B02"
B03 = "R20_B03"
B04 = "R20_B04"
C01 = "R20_C01"
C02 = "R20_C02"
C03 = "R20_C03"
C04 = "R20_C04"
D01 = "R20_D01"
D02 = "R20_D02"
D03 = "R20_D03"
M0 = "R20_M0"
M1 = "R20_M1"
M2 = "R20_M2"

# Round7 5bp caliber v5 target from replay_r7_caliber/v5_R16_O01.
V5_ANCHOR = {
    "cagr": 0.5499812832273525,
    "sharpe": 1.8756627913361097,
    "max_drawdown": -0.24056327715394388,
    "completed_trades": 1107,
    "n_obs": 2075,
    "nav_end": 12750089.36,
}
V5_NAV_END_EXACT = 12750089.355724689
N_TRADING_OBSERVATIONS_ANCHOR = 2075
FEE_SCHEDULE = "ROUND7_FLAT_5BP"
NAV_END_ABS_TOL = 0.01

SEEDS = tuple(SEEDS)
FIVE_FEATURES = tuple(FIVE_FEATURES)
TEN_FEATURES = tuple(TEN_FEATURES)
FIVE_HEAD = FIVE_HEAD
TEN_HEAD = TEN_HEAD
SUBSAMPLE = 1.0
COLSAMPLE = 1.0
XGB_N_JOBS = 2
PARITY_ABS_NAV = PARITY_ABS_NAV
PARITY_METRIC_ABS = PARITY_METRIC_ABS
RANK_NAV = RANK_NAV
TOP_K = TOP_K
MIN_KEEP = MIN_KEEP
SNAPSHOT_ID = SNAPSHOT_ID
SNAPSHOT_FINGERPRINT = SNAPSHOT_FINGERPRINT
O01_FORMULA = O01_FORMULA
F1_NAME = F1_NAME
F2_NAME = F2_NAME
F3_NAME = F3_NAME
F1_FORMULA = F1_FORMULA
F2_FORMULA = F2_FORMULA
F3_FORMULA = F3_FORMULA
R16_5_O01_ANCHOR = dict(R16_5_O01_ANCHOR)

# Round5 univariate skipped a day when finite names < 5 and did not record IC=0.
IC_MIN_VALID_NAMES_PER_DAY = 5
IC_MIN_SOURCE = "core/round5/univariate.py::_rank_ic_day mask.sum()<5 → skip day, do not record IC=0"
CLUSTER_SPEARMAN_ABS = 0.85
MAX_NEW_FACTORS_IN_MAIN = 12
COMBO_BUDGET = {
    "single": 12,
    "pair": 36,
    "triple": 20,
    "cross_family": 20,
    "pruned_ortho_pca": 12,
    "max_total": 100,
    "lofo_extra": True,
}
PCA_EXPLAINED_VARIANCE = 0.90
BLOCK_DAYS = 20
BOOT_N = 5000
SPA_BOOT = 2000
PAIRED_POS_RATIO_MIN = 0.80
SEED_LADDER = {
    "S1_native": 5,
    "S2_group_date": 30,
    "S2_max_advance": 6,
    "S3_group_date": 100,
    "S3_max_advance": 2,
    "S4_group_date": 300,
    "S4_trigger_pos_ratio_lo": 0.75,
    "S4_trigger_pos_ratio_hi": 0.85,
    "S4_trigger_abs_dmean_sharpe": 0.02,
    "untriggered_label": NOT_TRIGGERED,
    "forbid_fake_complete_with_10_seeds": True,
}
GROUP_DATE_SPEC = {
    "frac": 0.80,
    "replace": False,
    "stratify_by_year": True,
    "whole_day_query_in_or_out": True,
    "subsample": SUBSAMPLE,
    "colsample": COLSAMPLE,
}
MODELS = {
    "M0": "canonical XGBRanker",
    "M1": "Ridge/ElasticNet rank proxy; disclose loss≠NDCG",
    "M2": "LightGBM LambdaRank or SKIP_DEPENDENCY",
}
LABELS = ("label_e2_net", "label_exit_rule_net")

PROMOTION_GATES = {
    "pit_pass": True,
    "v5_parity_pass": True,
    "increment_explainable": True,
    "oof_not_worse": True,
    "same_caliber_net_better_than_v5": True,
    "maxdd_acceptable": True,
    "not_only_2024_2025": True,
    "fee_corrected_no_severe_reversal": True,
    "selection_bias_disclosed": True,
    "pass_allows_only": RECOMMEND_NEW_ALPHA_CANDIDATE,
    "pass_requires_lock_flag": AWAITING_USER_REVIEW,
    "default": KEEP_BASELINE_V5,
    "do_not_lock_v6": True,
    "do_not_auto_promote_current": True,
    "f2_not_promotion_eligible": True,
    "f1_f3_not_rebranded_as_new": True,
    "no_5000_random_search": True,
}

IC_RULES = {
    "min_valid_names_per_day": IC_MIN_VALID_NAMES_PER_DAY,
    "min_source": IC_MIN_SOURCE,
    "skip_day_if_below_min": True,
    "do_not_record_ic_as_zero_when_skipped": True,
    "labels": list(LABELS),
    "metrics": ("IC", "RankIC", "HAC", "block_bootstrap"),
    "cluster_spearman_abs_init": CLUSTER_SPEARMAN_ABS,
    "cluster_not_sole_drop_reason": True,
}

SELECTION_RULES = {
    "max_new_factors_in_main_combo_research": MAX_NEW_FACTORS_IN_MAIN,
    "do_not_pad_to_quota": True,
    "combo_registry_before_formal_combo_nav": True,
    "forbid_post_hoc_best_x_best": True,
    "pca_within_family": True,
    "pca_explained_variance_ratio": PCA_EXPLAINED_VARIANCE,
    "pca_no_threshold_search": True,
    "ortho_fit_inside_fold": True,
    "forbid_full_history_pca_or_ortho": True,
    "lofo_diagnostic_not_delete_from_v5": True,
}

PHASE_A_STAGES = ("read", "parity", "inventory", "freeze")
PHASE_BC_STAGES = ("pit", "ic", "shape", "redundancy", "conditional")
LATER_STAGES = (
    "pit",
    "ic",
    "shape",
    "redundancy",
    "conditional",
    "combo",
    "model",
    "seed",
    "backtest",
    "stats",
    "report",
)
STAGES = PHASE_A_STAGES + LATER_STAGES
PIT_MD = ROUND20_ROOT / "02_pit.md"
IC_MD = ROUND20_ROOT / "03_ic_rankic.md"
SHAPE_MD = ROUND20_ROOT / "04_shape_toptail.md"
REDUNDANCY_MD = ROUND20_ROOT / "05_redundancy.md"
CONDITIONAL_MD = ROUND20_ROOT / "06_conditional_ortho_pca_lofo.md"
COMBO_REG_MD = ROUND20_ROOT / "07_combo_registry.md"
COMBO_TRAIN_MD = ROUND20_ROOT / "08_combo_train.md"
MODEL_MD = ROUND20_ROOT / "09_model_selection.md"
RIDGE_ALPHA = 1.0
PIT_MIN_FINITE = 0.80
PIT_YEAR_GAP = 0.50
IC_YEARS = tuple(range(2018, 2027))
COMPLETE_IC_YEARS = tuple(range(2018, 2026))

PRIOR_CONSTRAINTS = {
    "factor_library_r17_rows": 75,
    "r17_breakdown": "baseline 14/13 + unused 31 + r17_new 30",
    "r18_add": (F1_NAME, F2_NAME, F3_NAME),
    "do_not_invent_factors": True,
    "f2_pit": PIT_TIMESTAMP_UNVERIFIED,
    "f2_not_promotion_eligible": True,
    "f1_f3_formulas_frozen": True,
    "f1_f3_not_rebranded": True,
    "r17_5000_already_run": True,
    "forbid_another_5000_search": True,
    "r19_track_b_no_material_train_bug": True,
    "a00_locked_vs_open_retrain_original_features_agree": True,
    "native_5_seed_rank_degenerate_at_1_0": True,
    "native_100_seeds_same_model_if_degenerate": True,
    "years_2018_2026_already_researched": True,
    "discovery_validation_audit_not_independent_oos": True,
    "walk_forward": "predict year Y using train Y-3..Y-1",
}

FORBIDDEN = (
    "edit_AGENTS_md",
    "edit_no_xgb_subsample_rule",
    "edit_core_round7",
    "edit_stock_src_or_runs",
    "auto_promote_CURRENT",
    "change_o01_formula",
    "change_b03_nine_grid",
    "change_90_10",
    "change_clean_risk_exit_size_gap",
    "search_5000_trials",
    "bagging_08_08",
    "fillna_missing_with_zero",
    "subsample_or_colsample_lt_1",
    "invent_factors_outside_r17_registry_plus_r18_f123",
    "rebrand_f1_f3_as_new_hypotheses",
    "promote_f2",
    "post_hoc_best_x_best",
    "full_history_pca_or_ortho",
    "fake_s4_with_10_seeds",
    "write_80_of_100_as_oos_p",
    "lock_v6",
    "mix_historical_fee_corrected_into_legacy_main_table",
    "write_写入时间_to_tracker",
)

PROTOCOL_SOURCE_FILES = (
    "core/round20/__init__.py",
    "core/round20/contract.py",
    "core/round20/common.py",
    "core/round20/read.py",
    "core/round20/inventory.py",
    "core/round20/parity.py",
    "core/round20/freeze.py",
    "core/round20/pipeline.py",
    "core/round20/docs.py",
    "core/round20/engine.py",
    "core/round20/train.py",
    "core/round20/runner.py",
    "core/round20/model.py",
    "core/round20/seed.py",
    "core/round20/backtest.py",
    "core/round20/report.py",
    "tools/run_round20.py",
    "core/paths.py",
)


def copy_plan() -> Path | None:
    from ..paths import assert_project_result_path, ensure_round20_tree

    ensure_round20_tree()
    assert_project_result_path(PLAN_COPY)
    if not CURSOR_PLAN.is_file():
        return None
    text = CURSOR_PLAN.read_text(encoding="utf-8")
    header = "<!-- 本文件是开工计划的课题落盘副本。禁止回写 Cursor plan 文件。 -->\n\n"
    if not text.startswith("<!-- 本文件是开工计划"):
        PLAN_COPY.write_text(header + text, encoding="utf-8")
    else:
        PLAN_COPY.write_text(text, encoding="utf-8")
    return PLAN_COPY


def exp_dir(code: str) -> Path:
    return ROUND20_ROOT / "candidates" / str(code)


def parity_dir(code: str = "R20_V5_R16_O01") -> Path:
    return ROUND20_ROOT / "parity" / str(code)


def overlayed_path(code: str) -> Path:
    return ROUND20_ROOT / "weights" / f"overlayed_{code}.parquet"


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
