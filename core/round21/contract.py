"""Round21 冻结契约。假设/滚动窗/预算/晋级门在看任何新因子 NAV 之前锁定。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT, RESULT_ROOT, ROUND20_ROOT, ROUND21_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, SEEDS, TEN_FEATURES, TEN_HEAD
from ..round8.overlays import MIN_KEEP
from ..round11.contract import PARITY_ABS_NAV, PARITY_METRIC_ABS, SNAPSHOT_FINGERPRINT, SNAPSHOT_ID
from ..round12.contract import RANK_NAV, TOP_K
from ..round18.contract import F1_FORMULA, F1_NAME, F2_FORMULA, F2_NAME, F3_FORMULA, F3_NAME
from ..round18.contract import HERMES_V7, MIN1_2013, MIN1_2022
from ..round18.contract import PIT_TIMESTAMP_UNVERIFIED as R18_PIT_TIMESTAMP_UNVERIFIED
from ..round19.contract import O01_FORMULA
from ..round20.contract import V5_ANCHOR as R20_V5_ANCHOR
from ..round20.contract import V5_NAV_END_EXACT as R20_V5_NAV_END_EXACT

KEEP_BASELINE_V5 = "KEEP_BASELINE_V5"
RECOMMEND_NEW_ALPHA_CANDIDATE = "RECOMMEND_NEW_ALPHA_CANDIDATE"
AWAITING_USER_REVIEW = "AWAITING_USER_REVIEW"
AUTO_PROMOTE_CURRENT = False
PARENT_V4 = "R9_B03"
CURRENT_BASELINE = "baseline_v5"
CURRENT_ALIAS = "R16_O01"
FORMAL_PROMOTION_BLOCKED = "FORMAL_PROMOTION_BLOCKED"
SKIP_DEPENDENCY = "SKIP_DEPENDENCY"
NOT_TRIGGERED = "NOT_TRIGGERED"
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
BLOCKED_DATA = "BLOCKED_DATA"
BLOCKED_PIT = "BLOCKED_PIT"
PIT_ELIGIBLE_MODEL = "PIT_ELIGIBLE_MODEL"
PIT_TIMESTAMP_UNVERIFIED = R18_PIT_TIMESTAMP_UNVERIFIED
GITHUB_METHOD_STATUS_UNVERIFIED = "REFERENCE_UNVERIFIED"
GITHUB_METHOD_STATUS_CLONED = "SOURCE_CLONED_AND_READ"
EXACT_GITHUB_REPRODUCTION_PASS = "EXACT_GITHUB_REPRODUCTION_PASS"
ROUND21_INTERNAL = "ROUND21_INTERNAL"
GITHUB_EXACT = "GITHUB_EXACT"
SAME_OPEN_EXECUTION_OPTIMISM = "SAME_OPEN_EXECUTION_OPTIMISM"

MANIFEST_NAME = "round21_manifest.yaml"
CURSOR_PLAN = Path("/home/mxy/.cursor/plans/round21_alpha_factory_0bd6caf9.plan.md")
PLAN_COPY = ROUND21_ROOT / "ROUND21_PLAN.md"
TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
FINAL_REPORT = ROUND21_ROOT / "ROUND21_ALPHA_DISCOVERY_REPORT.md"
FACTOR_ZOO_ROOT = PROJECT_ROOT / "research" / "factor_zoo"
PARITY_MD = ROUND21_ROOT / "00_v5_parity.md"
CONTRACT_MD = ROUND21_ROOT / "00_research_contract.md"
GITHUB_MD = ROUND21_ROOT / "00_github_method.md"
AUDIT_MD = ROUND21_ROOT / "01_source_audit.md"
HYP_MD = ROUND21_ROOT / "01_hypotheses.md"

V5_ANCHOR = {
    "cagr": 0.5499812832273525,
    "sharpe": 1.8756627913361097,
    "max_drawdown": -0.24056327715394388,
    "completed_trades": 1107,
    "n_obs": 2075,
    "nav_end": 12750089.355724689,
}
V5_NAV_END_EXACT = 12750089.355724689
N_TRADING_OBSERVATIONS_ANCHOR = 2075
FEE_SCHEDULE = "ROUND7_FLAT_5BP"
NAV_END_ABS_TOL = 0.01
# 计划展示位可四舍五入，正式对照用 V5_NAV_END_EXACT。
V5_ANCHOR_DISPLAY = dict(R20_V5_ANCHOR)
R20_V5_NAV_END_EXACT = R20_V5_NAV_END_EXACT

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

HERMES_V7 = HERMES_V7
MIN1_2013 = MIN1_2013
MIN1_2022 = MIN1_2022
HERMES_SCHEMA = Path(
    "/data1/Stock/data/acquisitions/hermes-cn-a-2010-20260724-v7-rebuild/source_schema.json"
)
HERMES_REBUILD_RAW = Path("/data1/Stock/data/acquisitions/hermes-cn-a-2010-20260724-v7-rebuild/raw")
HOLDER_NUM_PARQUET = Path("/home/mxy/AmazingData/data/amazingdata_raw/holder_num/all.parquet")
BLOCK_TRADING_PARQUET = Path("/home/mxy/AmazingData/data/amazingdata_raw/block_trading/all.parquet")
ZYQUANT_350_ROOT = Path("/data1/Stock/.zyquant/cache/factors-pit-ex-st-v2")
GITHUB_REPO = "https://github.com/xiejinglover/alpha-forge"
GITHUB_SPARSE_DIR = "rolling-factor-evidence"
GITHUB_CLONE_DEST = Path("/tmp/round21_alpha_forge_reference")

IC_MIN_VALID_NAMES_PER_DAY = 5
IC_MIN_SOURCE = "core/round5/univariate.py::_rank_ic_day mask.sum()<5 → skip day, do not record IC=0"
PIT_MIN_FINITE = 0.80
PIT_YEAR_GAP = 0.50
COMPLETE_IC_YEARS = tuple(range(2018, 2026))
IC_START = "2018-01-01"
TICK_SIZE = 0.01
TAIL_START_SEC = 14 * 3600 + 30 * 60
CONSTRUCT_STAGES = ("construct", "pit", "evidence")
GITHUB_EXACT_SUBSET = {
    "start": "2018-01-02",
    "end": "2018-06-29",
    "window_id": "GITHUB_EXACT_WINDOW_SUBSET_2018H1",
    "deployment_id": "GITHUB_EXACT_DEPLOY_2018H1",
    "study_id": "r21-github-exact-subset-2018h1",
}
CLUSTER_SPEARMAN_ABS = 0.85
MAX_NEW_FACTORS_IN_MAIN = 12
COMBO_BUDGET = {
    "single": 12,
    "pair": 66,
    "triple": 36,
    "ortho_pca_nonlinear": 24,
    "max_confirm_batch": 12 + 66 + 36 + 24,
    "overflow_opens_next_batch": True,
    "forbid_chase_winner_inside_confirm_batch": True,
    "lofo_extra": True,
}
PCA_EXPLAINED_VARIANCE = 0.90
BLOCK_DAYS = 20
BOOT_N = 5000
SPA_BOOT = 2000
PAIRED_POS_RATIO_MIN = 0.80
INTERNAL_ROLLING_WINDOWS = (60, 120, 252)
INTERNAL_WINDOW_NAMES = (
    "ROUND21_INTERNAL_W60",
    "ROUND21_INTERNAL_W120",
    "ROUND21_INTERNAL_W252",
)
LABELS = ("label_e2_net", "label_exit_rule_net")
MATURITY_AWARE_LABEL = "label_exit_rule_net"
MATURITY_BY = "actual_exit_date"

SEED_LADDER = {
    "S1_native": 5,
    "native_degenerate_then": "group-date",
    "do_not_pick_best_from_100_or_300": True,
    "run_only_after_preregistered_increment_gate": True,
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
    "M0": "canonical XGBRanker train_head_open 3y→1y",
    "FIXED_FACTOR_SET_REPLAY": "FIXED_FACTOR_SET_REPLAY",
    "NESTED_WALK_FORWARD_FEATURE_SELECTION": "NESTED_WALK_FORWARD_FEATURE_SELECTION",
    "both_are_not_independent_oos": True,
}

PROMOTION_GATES = {
    "pit_pass": True,
    "quality_pass": True,
    "ic_rolling_conditional": True,
    "top_tail": True,
    "replace_or_complement": True,
    "oof_not_worse": True,
    "same_caliber_net_better_than_v5": True,
    "robustness": True,
    "pass_allows_only": RECOMMEND_NEW_ALPHA_CANDIDATE,
    "pass_requires_lock_flag": AWAITING_USER_REVIEW,
    "default": KEEP_BASELINE_V5,
    "do_not_lock_v6": True,
    "do_not_auto_promote_current": True,
    "no_5000_random_search": True,
    "internal_evidence_is_vector_hard_filter_pareto": True,
    "do_not_optimize_hand_weighted_score": True,
    "do_not_invent_github_score": True,
}

IC_RULES = {
    "min_valid_names_per_day": IC_MIN_VALID_NAMES_PER_DAY,
    "min_source": IC_MIN_SOURCE,
    "skip_day_if_below_min": True,
    "do_not_record_ic_as_zero_when_skipped": True,
    "labels": list(LABELS),
    "maturity_aware_label": MATURITY_AWARE_LABEL,
    "maturity_by": MATURITY_BY,
    "metrics": ("IC", "RankIC", "HAC", "block_bootstrap"),
    "cluster_spearman_abs_init": CLUSTER_SPEARMAN_ABS,
    "cluster_not_sole_drop_reason": True,
    "cluster_must_run": ("Original", "New", "Add", "Replace"),
    "internal_windows": list(INTERNAL_WINDOW_NAMES),
    "do_not_pick_window_after_seeing_nav": True,
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
    "add_replace_ablation_all_pit_valid_new": True,
    "single_factor_evidence_all_computable_new": True,
}

R20_PRIOR = {
    "inventory_n": 78,
    "pit_n": 63,
    "keep_n": 45,
    "combo_n": 100,
    "best_code": "R20_P32",
    "best_sharpe": 1.6738176828632965,
    "v5_sharpe": 1.8756627913361097,
    "best_worse_than_v5": True,
    "decision": KEEP_BASELINE_V5,
}

PHASE_A_STAGES = ("read", "parity", "github", "audit", "hypotheses", "freeze")
LATER_STAGES = (
    "construct",
    "pit",
    "evidence",
    "redundancy",
    "conditional",
    "combo",
    "model",
    "seed",
    "backtest",
    "stats",
    "zoo",
    "report",
)
STAGES = PHASE_A_STAGES + LATER_STAGES
RECOMBINE_STAGES = ("redundancy", "conditional", "combo", "model", "seed", "backtest", "stats")
REPORT_STAGES = ("zoo", "report")

A00 = "R21_A00"
A01 = "R21_A01"
FIXED_CODE = "R21_FIXED"
NESTED_CODE = "R21_NESTED"
RIDGE_ALPHA = 1.0
IC_YEARS = tuple(range(2018, 2027))
HISTORICAL_FEE_CORRECTED_SENSITIVITY = "HISTORICAL_FEE_CORRECTED_SENSITIVITY"
FORWARD_OOS_NOT_AVAILABLE = "FORWARD_OOS_NOT_AVAILABLE"
NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"
NATIVE_SEED_DIVERSITY_AUDIT = "NATIVE_SEED_DIVERSITY_AUDIT"
REDUNDANCY_MD = ROUND21_ROOT / "05_redundancy.md"
CONDITIONAL_MD = ROUND21_ROOT / "06_conditional.md"
COMBO_REG_MD = ROUND21_ROOT / "07_combo_registry.md"
MODEL_MD = ROUND21_ROOT / "08_walk_forward.md"
SEED_MD = ROUND21_ROOT / "10_seed_robustness.md"
BACKTEST_MD = ROUND21_ROOT / "11_backtest_decision.md"
STATS_MD = ROUND21_ROOT / "12_stats.md"

# 预注册 OOF 增量门：写在看组合 NAV 之前。禁止事后改阈值。
INCREMENT_GATE = {
    "id": "R21_OOF_INCREMENT_V1",
    "compare_to": A01,
    "metric": "oof_rankic_mean",
    "not_worse_both_heads": True,
    "strict_better_any_head": True,
    "min_oos_days": 200,
    "min_delta": 0.0,
    "eps": 1e-12,
    "frozen_before_combo_nav": True,
    "seed_only_if_gate_hit": True,
}

FACTORY_STEPS = (
    "FactorDefinition",
    "DataAudit",
    "PIT",
    "IC",
    "Rolling",
    "Shape",
    "TopTail",
    "Redundancy",
    "ConditionalIC",
    "Combination",
    "ModelIncrementality",
    "PortfolioEvidence",
    "PromotionGate",
)

PRIOR_CONSTRAINTS = {
    "r20_inventory": 78,
    "r20_pit": 63,
    "r20_keep": 45,
    "r20_combo": 100,
    "r20_best": "R20_P32 Sharpe 1.674 < 1.876",
    "do_not_rebrand_r17_r18_r20_failures": True,
    "f2_not_promotion_eligible": True,
    "f1_is_v_seal_over_a_below": True,
    "a1_is_not_r18_f1": True,
    "zyquant_350_novelty_library_only": True,
    "min1_is_close_limit_subset": True,
    "no_full_market_minute_mean": True,
    "do_not_fillna_missing_with_zero": True,
    "non_margin_financing_null_not_zero": True,
    "buy_ord_not_in_v7_landed_tables": True,
    "round19_1_does_not_exist": True,
    "walk_forward": "predict year Y using train Y-3..Y-1",
}

NEAR_NEIGHBORS = {
    "r21_a4_tail_seal_share": ("fb_close_seal_streak_norm",),
    "r21_b1_fb_amount_share": ("r17_d_amt_xs_rank",),
    "r21_c3_post_touch_path_eff": ("h_min1_path_eff_full",),
    "r21_a1_vw_limit_gap": ("fb_post_limit_supply_absorption",),
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
    "rebrand_r17_r18_r20_failures",
    "claim_exact_github_pass_without_source",
    "invent_missing_tables",
    "full_market_minute_mean",
    "promote_block_trading_without_announcement_pit",
    "invent_github_score",
    "optimize_hand_weighted_evidence_score",
    "lock_v6",
    "write_写入时间_to_tracker",
)

PROTOCOL_SOURCE_FILES = (
    "core/round21/__init__.py",
    "core/round21/contract.py",
    "core/round21/common.py",
    "core/round21/hypotheses.py",
    "core/round21/families.yaml",
    "core/round21/read.py",
    "core/round21/parity.py",
    "core/round21/github.py",
    "core/round21/audit.py",
    "core/round21/freeze.py",
    "core/round21/pipeline.py",
    "core/round21/docs.py",
    "core/round21/evaluate.py",
    "core/round21/zoo.py",
    "core/round21/report.py",
    "core/round21/panel.py",
    "core/round21/min1.py",
    "core/round21/construct.py",
    "core/round21/pit.py",
    "core/round21/evidence.py",
    "core/round21/github_exact.py",
    "tools/run_round21.py",
    "tools/evaluate_factor.py",
    "core/paths.py",
)


def copy_plan() -> Path | None:
    from ..paths import assert_project_result_path, ensure_round21_tree

    ensure_round21_tree()
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
    return ROUND21_ROOT / "candidates" / str(code)


def parity_dir(code: str = "R21_V5_R16_O01") -> Path:
    return ROUND21_ROOT / "parity" / str(code)


def overlayed_path(code: str) -> Path:
    return ROUND21_ROOT / "weights" / f"overlayed_{code}.parquet"


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
