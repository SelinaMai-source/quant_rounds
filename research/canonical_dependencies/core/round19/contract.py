"""Round19 冻结契约。Stage 0 看新候选 NAV 之前锁定 Track A/B 门。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT, RESULT_ROOT, ROUND16_5_ROOT, ROUND18_ROOT, ROUND19_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, SEEDS, TEN_FEATURES, TEN_HEAD
from ..round8.overlays import MIN_KEEP
from ..round11.contract import PARITY_ABS_NAV, PARITY_METRIC_ABS, SNAPSHOT_FINGERPRINT, SNAPSHOT_ID
from ..round12.contract import RANK_NAV, TOP_K
from ..round16_5.contract import (
    ACTIVITY_NOT_FULLY_MATCHED,
    O01_ANCHOR,
    O01_ROUND16,
    PCT_MIN_OBS,
    PCT_WINDOW,
    PERM_N,
    PERM_SEED,
    R16_SPA_P,
    SKEW_Q,
    V00 as R16_5_V00,
    V01 as R16_5_V01,
    V03 as R16_5_V03,
    V03_RIDGE_ALPHA,
    V4_ANCHOR as R16_5_V4_ANCHOR,
)
from ..round18.contract import A00 as R18_A00

KEEP_BASELINE_V4 = "KEEP_BASELINE_V4"
RECOMMENDED_BASELINE_V5 = "RECOMMENDED_BASELINE_V5"
AWAITING_USER_LOCK = "AWAITING_USER_LOCK"
AUTO_PROMOTE_V5 = False
PARENT_V4 = "R9_B03"
CURRENT_BASELINE = "baseline_v4"
OPTION_CANDIDATE = "R16_O01"
CANDIDATE_BASELINE_V5 = "R16_O01"
FORWARD_OOS_NOT_AVAILABLE = "FORWARD_OOS_NOT_AVAILABLE"
SAME_OPEN_EXECUTION_OPTIMISM = "SAME_OPEN_EXECUTION_OPTIMISM"
REPEATED_HISTORICAL_TESTING = "REPEATED_HISTORICAL_TESTING"
WINNERS_CURSE = "WINNERS_CURSE"
LIVE_VALIDATED = "LIVE_VALIDATED"
HISTORICAL_FEE_CORRECTED = "HISTORICAL_FEE_CORRECTED"
PRIOR_RESEARCH_SELECTION_EXISTS = True
PIT_TIMESTAMP_UNVERIFIED = "PIT_TIMESTAMP_UNVERIFIED"
LAST_PRICE_PROXY_ONLY = "LAST_PRICE_PROXY_ONLY"
EX_POST_DIAGNOSTIC_ONLY = "EX_POST_DIAGNOSTIC_ONLY"
POST_HOC_STRESS_DIAGNOSTIC = "POST_HOC_STRESS_DIAGNOSTIC"
ACTIVITY_MATCHING_PASS = "ACTIVITY_MATCHING_PASS"
FORMAL_PROMOTION_BLOCKED = "FORMAL_PROMOTION_BLOCKED"
LIVE_NOT_VALIDATED = "NOT_VALIDATED"
PROMISING_BUT_UNCONFIRMED = "PROMISING_BUT_UNCONFIRMED"
NON_ADDITIVE_PATH_DEPENDENCE = "NON_ADDITIVE_PATH_DEPENDENCE"

A_RECOMMEND_LOCK_O01_AS_DEVELOPMENT_V5 = "A_RECOMMEND_LOCK_O01_AS_DEVELOPMENT_V5"
A_KEEP_O01_AS_UNCONFIRMED_CANDIDATE = "A_KEEP_O01_AS_UNCONFIRMED_CANDIDATE"
A_BLOCKED_BY_PIT_OR_DATA = "A_BLOCKED_BY_PIT_OR_DATA"
CLOSE_ROUND18_FACTOR_HYPOTHESES = "CLOSE_ROUND18_FACTOR_HYPOTHESES"
INCONCLUSIVE_DUE_TO_DATA_PIT = "INCONCLUSIVE_DUE_TO_DATA_PIT"
REPAIRED_ROUND18_CANDIDATE = "REPAIRED_ROUND18_CANDIDATE"
C_KEEP_BASELINE_V4_KEEP_O01_CANDIDATE = "C_KEEP_BASELINE_V4_KEEP_O01_CANDIDATE"
C_KEEP_BASELINE_V4_WITHDRAW_O01 = "C_KEEP_BASELINE_V4_WITHDRAW_O01"
C_RECOMMEND_O01_AS_DEVELOPMENT_V5 = "C_RECOMMEND_O01_AS_DEVELOPMENT_V5"
C_BLOCKED_INSUFFICIENT_EVIDENCE = "C_BLOCKED_INSUFFICIENT_EVIDENCE"

MANIFEST_A_NAME = "round19_track_a_manifest.yaml"
MANIFEST_B_NAME = "round19_track_b_manifest.yaml"
CURSOR_PLAN = Path("/home/mxy/.cursor/plans/round19_dual_validation_dc02c5f2.plan.md")
PLAN_COPY = ROUND19_ROOT / "ROUND19_PLAN.md"
TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
FINAL_REPORT = ROUND19_ROOT / "ROUND19_RESEARCH_REPORT.md"
PARITY_MD = ROUND19_ROOT / "00_shared_baseline_parity.md"
MANIFEST_MD = ROUND19_ROOT / "01_round19_manifest.md"

A_C0 = "R19_A_C0"
A_C1 = "R19_A_C1"
A_C2 = "R19_A_C2"
A_C3 = "R19_A_C3"
A_C4 = "R19_A_C4"
B00 = "R19_B00"
B01 = "R19_B01"

V4_ANCHOR = dict(R16_5_V4_ANCHOR)
O01_ANCHOR = dict(O01_ANCHOR)
N_TRADING_OBSERVATIONS_ANCHOR = 2075
CALENDAR_CAGR_PCT_DISPLAY = "52.54%"
CAGR_OF_252_PCT_DISPLAY = "55.07%"
EVENT_LAYERS = {"raw_signal_incl_2015": 332, "nav_window": 298, "eligible": 241, "actual_fill_change": 39}

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
PERM_N = PERM_N
PERM_SEED = PERM_SEED
PCT_WINDOW = PCT_WINDOW
PCT_MIN_OBS = PCT_MIN_OBS
SKEW_Q = SKEW_Q
R16_SPA_P = R16_SPA_P
V03_RIDGE_ALPHA = V03_RIDGE_ALPHA
V03_FEATURES = ("market_up_ratio", "B01A", "candidate_count", "index_RV20")
BOOT_N = 5000
SPA_BOOT = 2000
BLOCK_DAYS = 20
COST_BPS = (20.0, 30.0)
R18_A00 = R18_A00
R16_5_V00 = R16_5_V00
R16_5_V01 = R16_5_V01
R16_5_V03 = R16_5_V03
R16_5_V00_DIR = ROUND16_5_ROOT / "parity" / R16_5_V00
R16_5_V00_OVERLAY = ROUND16_5_ROOT / "weights" / f"overlayed_{R16_5_V00}.parquet"
R18_A00_DIR = ROUND18_ROOT / "parity" / R18_A00
O01_FORMULA = "PUT_SKEW25.shift(1) > rolling(252, min_periods=126).quantile(0.90) → apply_top2_veto rank=2 MIN_KEEP"

TRACK_A_GATES = {
    "c0_c1_exact_parity_required": True,
    "event_layers_required": dict(EVENT_LAYERS),
    "layer_mix_is_fail": True,
    "c3_activity_matching_pass_before_sharpe": True,
    "c3_match_object": (
        "decision_time_known_eligible_top2_days",
        "yearly_treatment_count",
        "year_x_B01A_distribution",
        "planned_top2_veto_count",
        "planned_risk_budget",
    ),
    "c3_forbidden_future_pnl_or_fill_selection": True,
    "c3_ex_post_fill_match_label": EX_POST_DIAGNOSTIC_ONLY,
    "c3_model": "Ridge",
    "c3_ridge_alpha": V03_RIDGE_ALPHA,
    "c3_features": list(V03_FEATURES),
    "c3_no_redesign_equity_model": True,
    "c3_is_control_not_candidate": True,
    "c4_n": PERM_N,
    "c4_full_engine": True,
    "c4_forbid_ledger_shuffle": True,
    "c4_canonical_wins_if_optimized_disagrees": True,
    "c4_is_control_not_candidate": True,
    "o01_must_beat_matched_c3": True,
    "o01_c4_empirical_p_supportive": True,
    "o01_leave_one_event_out_not_few_extremes": True,
    "plus20_plus30_no_sign_flip": True,
    "disclose_spa": R16_SPA_P,
    "disclose_three_test_selection_bias": True,
    "p_031_alone_not_stable_alpha": True,
    "pit_unproven_forbids_PIT_PASS_and_LIVE_READY": True,
    "no_bid_ask": LAST_PRICE_PROXY_ONLY,
    "pass_allows_only": A_RECOMMEND_LOCK_O01_AS_DEVELOPMENT_V5,
    "pass_requires_lock_flag": AWAITING_USER_LOCK,
    "default_if_insufficient": A_KEEP_O01_AS_UNCONFIRMED_CANDIDATE,
    "blocked_if_pit_or_data": A_BLOCKED_BY_PIT_OR_DATA,
    "do_not_auto_promote_current": True,
    "current_default": CURRENT_BASELINE,
}

TRACK_B_GATES = {
    "b00_role": "locked_round7_prediction_replay_like_A00",
    "b01_role": "round18_pipeline_force_retrain_original_14_13",
    "b01_no_f1_f2_f3": True,
    "same_panel_fold_label_seeds": True,
    "subsample": SUBSAMPLE,
    "colsample": COLSAMPLE,
    "six_layer_not_cagr_only": True,
    "xgb_env_diff_not_ignorable_pass": True,
    "material_bug_isolate_repair_then_rerun_r18_original_formulas": True,
    "repaired_prefix": "R19_B_REPAIRED_",
    "no_bug_closes_or_inconclusive": True,
    "close_label": CLOSE_ROUND18_FACTOR_HYPOTHESES,
    "inconclusive_label": INCONCLUSIVE_DUE_TO_DATA_PIT,
    "forbid_search_epsilon": True,
    "forbid_change_f_formulas": True,
    "forbid_change_head_weights": True,
    "diagnostic_not_eligible": True,
    "do_not_merge_r16_o01": True,
    "do_not_auto_promote_current": True,
    "current_default": CURRENT_BASELINE,
}

SIX_LAYER_PARITY = (
    {"level": 1, "name": "join_keys", "what": "fold/head/date 键；baseline vs retrain 行数、missing/extra/dup"},
    {"level": 2, "name": "labels_groups", "what": "label / group / next2_date 是否一致"},
    {"level": 3, "name": "prediction_rows", "what": "Round7 five ensemble 45256 vs A01 45255 必须对齐"},
    {"level": 4, "name": "feature_order_and_missing", "what": "原 14/13 列顺序与 complete_cases；无 fillna(0)"},
    {"level": 5, "name": "oof_scores", "what": "Five/Ten OOF NDCG / Rank IC / Top1/2 precision"},
    {"level": 6, "name": "canonical_ledger", "what": "NAV/成交/Top1/Top2/allocation_multiplier；禁止只比 CAGR"},
)

STAGE0 = ("read", "parity", "freeze")
TRACK_A_STAGES = (
    "a-replay",
    "a-events",
    "a-attribution",
    "a-concentration",
    "a-equity-control",
    "a-match",
    "a-placebo",
    "a-pit",
    "a-quotes",
    "a-prediction",
    "a-stats",
    "a-decision",
)
TRACK_B_STAGES = (
    "b-fresh-retrain",
    "b-parity",
    "b-join-audit",
    "b-labels",
    "b-folds",
    "b-feature-order",
    "b-missing",
    "b-heads",
    "b-top-change",
    "b-allocation",
    "b-neighbors",
    "b-bug-verdict",
    "b-repair",
)
TRACK_C_STAGES = ("c-forward", "c-decision", "c-report")
STAGES = STAGE0 + TRACK_A_STAGES + TRACK_B_STAGES + TRACK_C_STAGES
TRACK_STAGES = {"a": TRACK_A_STAGES, "b": TRACK_B_STAGES, "c": TRACK_C_STAGES, "shared": STAGE0}

TRACK_A_EXPERIMENTS: tuple[dict[str, Any], ...] = (
    {"code": A_C0, "role": "exact_v4_replay", "is_candidate": False, "formal": True},
    {"code": A_C1, "role": "exact_o01", "is_candidate": True, "formal": True, "formula": O01_FORMULA},
    {"code": A_C2, "role": "original_v03_replay", "is_candidate": False, "keep_flag": ACTIVITY_NOT_FULLY_MATCHED},
    {"code": A_C3, "role": "activity_matched_equity_only_control", "is_candidate": False, "formal": False},
    {"code": A_C4, "role": "full_engine_matched_random_control", "is_candidate": False, "formal": False},
)
TRACK_B_EXPERIMENTS: tuple[dict[str, Any], ...] = (
    {"code": B00, "role": "locked_round7_prediction_replay", "retrain": False, "extras": (), "formal": True},
    {"code": B01, "role": "force_retrain_original_14_13", "retrain": True, "force_retrain": True, "extras": (), "formal": True},
)

FORBIDDEN = (
    "edit_AGENTS_md",
    "edit_no_xgb_subsample_rule",
    "edit_core_round7",
    "edit_stock_src_or_runs",
    "auto_promote_CURRENT_v5",
    "fuse_r16_o01_with_round18_factors",
    "search_new_stock_or_option_factors",
    "search_thresholds_lookback_hyperparams_90_10_b03_exits_seeds",
    "search_5000_trials",
    "search_epsilon_after_nav",
    "bagging_08_08",
    "fillna_missing_with_zero",
    "claim_technical_fix_as_new_alpha",
    "write_写入时间_to_tracker",
    "ledger_daily_return_swap_placebo",
    "invent_bid_ask",
    "write_PIT_PASS_without_available_at_proof",
    "write_LIVE_READY_or_LIVE_VALIDATED",
    "promote_c3_or_c4_controls",
    "relax_gates_after_seeing_new_nav",
)

PROTOCOL_SOURCE_FILES = (
    "core/round19/__init__.py",
    "core/round19/contract.py",
    "core/round19/common.py",
    "core/round19/read.py",
    "core/round19/parity.py",
    "core/round19/freeze.py",
    "core/round19/pipeline.py",
    "core/round19/docs.py",
    "core/round19/track_a/__init__.py",
    "core/round19/track_b/__init__.py",
    "core/round19/track_c/__init__.py",
    "tools/run_round19.py",
    "core/paths.py",
)


def copy_plan() -> Path | None:
    from ..paths import assert_project_result_path, ensure_round19_tree

    ensure_round19_tree()
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
    return ROUND19_ROOT / "candidates" / str(code)


def parity_dir(code: str = A_C0) -> Path:
    return ROUND19_ROOT / "parity" / str(code)


def overlayed_path(code: str) -> Path:
    return ROUND19_ROOT / "weights" / f"overlayed_{code}.parquet"


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
