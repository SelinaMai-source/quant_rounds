"""Round 16.5 冻结契约。确认性验证，封闭五 ID。默认 KEEP_BASELINE_V4。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import RESULT_ROOT, ROUND15_ROOT, ROUND16_5_ROOT, ROUND16_ROOT
from ..round11.contract import (
    BASELINE_V4,
    PARITY_ABS_NAV,
    PARITY_METRIC_ABS,
    SNAPSHOT_FINGERPRINT,
    SNAPSHOT_ID,
    STRATEGY_SOURCE_FILES,
)
from ..round12.contract import RANK_NAV, TOP_K
from ..round15.contract import R12_01, R12_01_DIR, R13_00_DIR, R14_00_DIR
from ..round16.contract import R16_00, V4_ANCHOR as R16_V4_ANCHOR

KEEP_BASELINE_V4 = "KEEP_BASELINE_V4"
RECOMMENDED_BASELINE_V5 = "RECOMMENDED_BASELINE_V5"
AWAITING_USER_LOCK = "AWAITING_USER_LOCK"
AUTO_PROMOTE_V5 = False
PARENT_V4 = "R9_B03"
PRIOR_SELECTION_EXISTS = True
FORWARD_OOS_NOT_AVAILABLE = "FORWARD_OOS_NOT_AVAILABLE"
LIVE_NOT_VALIDATED = "NOT_VALIDATED"
PROMISING_BUT_UNCONFIRMED = "PROMISING_BUT_UNCONFIRMED"
HISTORICAL_DEVELOPMENT_ONLY = "HISTORICAL_DEVELOPMENT_ONLY"
BID_ASK_NOT_AVAILABLE = "BID_ASK_NOT_AVAILABLE"
PIT_TIMESTAMP_UNVERIFIED = "PIT_TIMESTAMP_UNVERIFIED"
ACTIVITY_NOT_FULLY_MATCHED = "ACTIVITY_NOT_FULLY_MATCHED"
MATCHING_CONSTRAINT_FAILURE = "MATCHING_CONSTRAINT_FAILURE"
CONCENTRATED_INCREMENT_RISK = "CONCENTRATED_INCREMENT_RISK"
DE_RISKING_ONLY = "DE_RISKING_ONLY"
POST_HOC_DIAGNOSTIC = "POST_HOC_DIAGNOSTIC"
FULL_ENGINE_NAV_DIFFERENCE = "FULL_ENGINE_NAV_DIFFERENCE"
INCONCLUSIVE_LOW_N = "INCONCLUSIVE_LOW_N"

MANIFEST_NAME = "round16_5_manifest_v1.yaml"
CURSOR_PLAN = Path("/home/mxy/.cursor/plans/round16.5_confirmatory_925c6c3f.plan.md")
PLAN_COPY = ROUND16_5_ROOT / "ROUND16_5_PLAN.md"
TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
FINAL_REPORT = ROUND16_5_ROOT / "ROUND16_5_CONFIRMATORY_REPORT.md"
HANDOFF = ROUND16_5_ROOT / "ROUND17_RESEARCH_HANDOFF.md"

V00 = "R16_5_V00_BASELINE_V4"
V01 = "R16_5_V01_O01_EXACT"
V02 = "R16_5_V02_RANDOM_TOP2_VETO"
V03 = "R16_5_V03_EQUITY_RISK_VETO"
V04 = "R16_5_V04_RESIDUAL_PUT_SKEW"
STRATEGY_IDS = (V00, V01)
CONTROL_IDS = (V02, V03)
CANDIDATE_IDS = (V01, V04)
PRE_REGISTERED_IDS = (V00, V01, V02, V03, V04)
O01_ROUND16 = "R16_O01"

R15_00_DIR = ROUND15_ROOT / "parity" / "R15_00_CORRECTED_V4"
R16_00_DIR = ROUND16_ROOT / "parity" / R16_00
R16_O01_DIR = ROUND16_ROOT / "candidates" / O01_ROUND16
R16_FACTORS = ROUND16_ROOT / "data" / "processed" / "option_factors_daily.parquet"
R16_TREAT = ROUND16_ROOT / "options" / "treat_dates.json"
R16_SPA_P = 0.496

V4_ANCHOR = {
    "cagr": 0.5253672918108749,
    "sharpe": 1.8120914675763191,
    "max_drawdown": -0.25455988069271596,
    "completed_trades": 1149,
}
O01_ANCHOR = {
    "cagr": 0.5361620306263046,
    "sharpe": 1.8404039786206279,
    "max_drawdown": -0.24792541309267202,
    "completed_trades": 1107,
}

PCT_WINDOW = 252
PCT_MIN_OBS = 126
SKEW_Q = 0.90
RIDGE_ALPHA = 1.0
V04_FIT_WINDOW = 504
V04_FIT_MIN = 252
V03_RIDGE_ALPHA = 1.0
PERM_N = 1000
PERM_SEED = 20261016
MASK_CHECK_N = 30
BLOCK_DAYS = 20
BOOT_N = 5000
BOOT_N_MIN = 2000
SPA_BOOT = 2000
EMBARGO = 12
MIN_BLOCK = 60
RF_FALLBACK = 0.02
RF_GRID = (0.01, 0.02, 0.03)
COST_BPS = (20.0, 30.0)
LOSS_NET_LT_0 = 0.0
SEVERE_LOSS_RET = -0.05
ENGINE_CHECK_SEEDS = tuple(range(1, MASK_CHECK_N + 1))

# 预注册 promotion 数值门（看 V03/V04 NAV 之前锁定，事后不得放宽）
PROMOTION_GATES = {
    "placebo_n_target": PERM_N,
    "empirical_p_formula": "(1 + n_beat_or_tie) / (1 + N)",
    "never_report_p_zero": True,
    "v01_increment_not_only_lower_exposure": True,
    "v01_beats_activity_matched_v03": True,
    "v03_not_candidate": True,
    "quote_not_fail": True,
    "no_major_right_tail_damage": True,
    "plus20_plus30_no_sign_flip": True,
    "disclose_prior_selection": True,
    "disclose_round16_spa": True,
    "round16_spa_p": R16_SPA_P,
    "strengthen_requires_placebo_p_lt": 0.10,
    "keep_default_if_inconclusive": True,
    "current_default": "baseline_v4",
    "candidate_default": O01_ROUND16,
    "live_default": LIVE_NOT_VALIDATED,
    "status_default": PROMISING_BUT_UNCONFIRMED,
}

FOUR_WAY = (
    "KEEP_CANDIDATE_RECOMMENDATION",
    "STRENGTHEN_CANDIDATE_RECOMMENDATION",
    "WITHDRAW_CANDIDATE_RECOMMENDATION",
    "RECOMMEND_V04_AS_ALTERNATIVE",
)

QUOTE_VERDICTS = (
    "QUOTE_QUALITY_SUPPORTED",
    "QUOTE_QUALITY_PARTIALLY_SUPPORTED",
    "LAST_PRICE_PROXY_ONLY",
    "QUOTE_QUALITY_FAIL",
)

STAGES = (
    "read",
    "parity",
    "freeze",
    "attribution",
    "placebo",
    "equity",
    "orthogonal",
    "quote",
    "prediction",
    "stress",
    "stats",
    "forward",
    "report",
)

PROTOCOL_SOURCE_FILES = (
    "core/round16_5/__init__.py",
    "core/round16_5/contract.py",
    "core/round16_5/common.py",
    "core/round16_5/engine.py",
    "core/round16_5/read.py",
    "core/round16_5/parity.py",
    "core/round16_5/freeze.py",
    "core/round16_5/attribution.py",
    "core/round16_5/placebo.py",
    "core/round16_5/equity.py",
    "core/round16_5/orthogonal.py",
    "core/round16_5/quote.py",
    "core/round16_5/prediction.py",
    "core/round16_5/stress.py",
    "core/round16_5/stats.py",
    "core/round16_5/decision.py",
    "core/round16_5/figures.py",
    "core/round16_5/docs.py",
    "core/round16_5/pipeline.py",
    "tools/run_round16_5.py",
)

CANDIDATE_FORMULAS = {
    V00: "exact replay of baseline_v4 = R9_B03 via run_r15_backtest + HISTORICAL_FEE_CORRECTED + B03.",
    V01: "exact Round16 O01. PUT_SKEW25 > past-only 90th of 252 valid option days (min 126), shift(1) → Top2 MIN_KEEP. No Top3 fill.",
    V02: "full-engine year-stratified random Top2 veto of same count as V01 on eligible days. NOT a candidate.",
    V03: "equity-only risk veto. Ridge α=1.0, chronological, mature labels, activity-matched on past train window. NOT a candidate.",
    V04: "ONLY new alpha. PUT_SKEW25 residual vs X_equity Ridge α=1.0 window 504/min 252 then residual 90th of past 252/min 126, shift(1) → Top2 MIN_KEEP.",
}

FORBIDDEN = (
    "search_80_85_90_95",
    "search_lookback",
    "top2_0_5_10_grid",
    "retrain_xgb",
    "change_90_10",
    "add_option_factors",
    "add_seed",
    "auto_promote_CURRENT_v5",
    "write_写入时间_to_tracker",
    "ledger_daily_return_swap_placebo",
    "report_p_equals_zero",
    "use_sum_avoided_pnl_as_authoritative_increment",
    "relax_gates_after_v03_v04_nav",
    "invent_bid_ask",
    "write_secrets",
    "run_round17",
    "claim_independent_confirmation",
)


def experiment_dir(code: str) -> Path:
    return ROUND16_5_ROOT / "candidates" / str(code)


def control_dir(code: str) -> Path:
    return ROUND16_5_ROOT / "controls" / str(code)


def parity_dir(code: str) -> Path:
    return ROUND16_5_ROOT / "parity" / str(code)


def overlayed_path(code: str) -> Path:
    return ROUND16_5_ROOT / "weights" / f"overlayed_{code}.parquet"


def copy_plan() -> Path:
    from ..paths import assert_project_result_path, ensure_round16_5_tree

    ensure_round16_5_tree()
    assert_project_result_path(PLAN_COPY)
    text = CURSOR_PLAN.read_text(encoding="utf-8")
    header = "<!-- 本文件是开工计划的课题落盘副本。禁止回写 Cursor plan 文件。 -->\n\n"
    if not text.startswith("<!-- 本文件是开工计划"):
        PLAN_COPY.write_text(header + text, encoding="utf-8")
    else:
        PLAN_COPY.write_text(text, encoding="utf-8")
    return PLAN_COPY


def pre_registered_spec(code: str) -> dict[str, Any]:
    table = {
        V00: {"role": "strategy_parity_anchor", "parent": PARENT_V4, "is_candidate": False, "is_v5": False},
        V01: {"role": "strategy_exact_o01", "parent": V00, "is_candidate": True, "is_v5": False, "formula": CANDIDATE_FORMULAS[V01]},
        V02: {"role": "full_engine_random_top2_veto_family", "parent": V00, "is_candidate": False, "formula": CANDIDATE_FORMULAS[V02]},
        V03: {"role": "equity_only_risk_veto_control", "parent": V00, "is_candidate": False, "formula": CANDIDATE_FORMULAS[V03]},
        V04: {"role": "orthogonal_residual_put_skew", "parent": V00, "is_candidate": True, "is_v5": False, "formula": CANDIDATE_FORMULAS[V04]},
    }
    if code not in table:
        raise KeyError(code)
    spec = dict(table[code])
    spec["code"] = code
    spec["fee"] = "HISTORICAL_FEE_CORRECTED"
    spec["search"] = False
    return spec


__all__ = [
    "AWAITING_USER_LOCK",
    "ACTIVITY_NOT_FULLY_MATCHED",
    "AUTO_PROMOTE_V5",
    "BASELINE_V4",
    "BID_ASK_NOT_AVAILABLE",
    "BLOCK_DAYS",
    "BOOT_N",
    "BOOT_N_MIN",
    "CANDIDATE_FORMULAS",
    "CANDIDATE_IDS",
    "CONCENTRATED_INCREMENT_RISK",
    "CONTROL_IDS",
    "COST_BPS",
    "CURSOR_PLAN",
    "DE_RISKING_ONLY",
    "EMBARGO",
    "ENGINE_CHECK_SEEDS",
    "FINAL_REPORT",
    "FORBIDDEN",
    "FORWARD_OOS_NOT_AVAILABLE",
    "FOUR_WAY",
    "FULL_ENGINE_NAV_DIFFERENCE",
    "HANDOFF",
    "HISTORICAL_DEVELOPMENT_ONLY",
    "INCONCLUSIVE_LOW_N",
    "KEEP_BASELINE_V4",
    "LIVE_NOT_VALIDATED",
    "LOSS_NET_LT_0",
    "MANIFEST_NAME",
    "MASK_CHECK_N",
    "MATCHING_CONSTRAINT_FAILURE",
    "MIN_BLOCK",
    "O01_ANCHOR",
    "O01_ROUND16",
    "PARITY_ABS_NAV",
    "PARITY_METRIC_ABS",
    "PARENT_V4",
    "PCT_MIN_OBS",
    "PCT_WINDOW",
    "PERM_N",
    "PERM_SEED",
    "PIT_TIMESTAMP_UNVERIFIED",
    "PLAN_COPY",
    "POST_HOC_DIAGNOSTIC",
    "PRE_REGISTERED_IDS",
    "PRIOR_SELECTION_EXISTS",
    "PROMOTION_GATES",
    "PROMISING_BUT_UNCONFIRMED",
    "PROTOCOL_SOURCE_FILES",
    "QUOTE_VERDICTS",
    "R12_01",
    "R12_01_DIR",
    "R13_00_DIR",
    "R14_00_DIR",
    "R15_00_DIR",
    "R16_00_DIR",
    "R16_FACTORS",
    "R16_O01_DIR",
    "R16_SPA_P",
    "R16_TREAT",
    "R16_V4_ANCHOR",
    "RANK_NAV",
    "RECOMMENDED_BASELINE_V5",
    "RF_FALLBACK",
    "RF_GRID",
    "RIDGE_ALPHA",
    "SEVERE_LOSS_RET",
    "SKEW_Q",
    "SNAPSHOT_FINGERPRINT",
    "SNAPSHOT_ID",
    "SPA_BOOT",
    "STAGES",
    "STRATEGY_IDS",
    "STRATEGY_SOURCE_FILES",
    "TOP_K",
    "TRACKER_PATH",
    "V00",
    "V01",
    "V02",
    "V03",
    "V03_RIDGE_ALPHA",
    "V04",
    "V04_FIT_MIN",
    "V04_FIT_WINDOW",
    "V4_ANCHOR",
    "copy_plan",
    "control_dir",
    "experiment_dir",
    "overlayed_path",
    "parity_dir",
    "pre_registered_spec",
]
