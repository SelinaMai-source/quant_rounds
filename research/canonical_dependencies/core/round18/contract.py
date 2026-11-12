"""Round18 冻结契约。公式与晋级门在看到候选 NAV 之前锁定。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT, RESULT_ROOT, ROUND18_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, SEEDS, TEN_FEATURES, TEN_HEAD
from ..round11.contract import PARITY_ABS_NAV, PARITY_METRIC_ABS
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
HISTORICAL_FEE_CORRECTED = "HISTORICAL_FEE_CORRECTED"
PRIOR_RESEARCH_SELECTION_EXISTS = True
PIT_TIMESTAMP_UNVERIFIED = "PIT_TIMESTAMP_UNVERIFIED"
LAGGED_DIAGNOSTIC = "LAGGED_DIAGNOSTIC"
SKIP_FACTOR_FOR_PROMOTION = "SKIP_FACTOR_FOR_PROMOTION"
PARTIAL_FACTOR_COVERAGE = "PARTIAL_FACTOR_COVERAGE"
TOUCH_TIME_APPROXIMATED = "TOUCH_TIME_APPROXIMATED"
PERFORMANCE_CONCENTRATION_RISK = "PERFORMANCE_CONCENTRATION_RISK"
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"

MANIFEST_NAME = "round18_microstructure_manifest.yaml"
CURSOR_PLAN = Path("/home/mxy/.cursor/plans/round18_microstructure_alpha_a5e39fb1.plan.md")
PLAN_COPY = ROUND18_ROOT / "ROUND18_PLAN.md"
TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
FINAL_REPORT = ROUND18_ROOT / "ROUND18_MICROSTRUCTURE_ALPHA_REPORT.md"

A00 = "R18_A00"
A01 = "R18_A01"
A02 = "R18_A02"
A03 = "R18_A03"
B01 = "R18_B01"
B02 = "R18_B02"
B03 = "R18_B03"
C01 = "R18_C01"
D01 = "R18_D01"
D02 = "R18_D02"
D03 = "R18_D03"
D04 = "R18_D04"
D05 = "R18_D05"
D06 = "R18_D06"

V4_ANCHOR = dict(R16_5_V4_ANCHOR)

DEV_START = date(2018, 1, 1)
DEV_END = date(2021, 12, 31)
AUDIT_START = date(2018, 1, 1)
AUDIT_END = date(2026, 7, 24)
FORWARD_AFTER = date(2026, 7, 24)

SEEDS = tuple(SEEDS)
FIVE_FEATURES = tuple(FIVE_FEATURES)
TEN_FEATURES = tuple(TEN_FEATURES)
FIVE_HEAD = FIVE_HEAD
TEN_HEAD = TEN_HEAD
SUBSAMPLE = 1.0
COLSAMPLE = 1.0
XGB_N_JOBS = 2
MAXDD_WORSEN_CAP = 0.02
COVERAGE_MIN = 0.95
BLOCK_DAYS = 20
BOOT_N = 5000
SPA_BOOT = 2000
COST_BPS = (20.0, 30.0)
CAPACITY_NAV = (300_000.0, 1_000_000.0, 3_000_000.0)
PARITY_ABS_NAV = PARITY_ABS_NAV
PARITY_METRIC_ABS = PARITY_METRIC_ABS
RANK_NAV = RANK_NAV
TOP_K = TOP_K

F1_NAME = "fb_post_limit_supply_absorption"
F2_NAME = "fb_order_size_asymmetry"
F3_NAME = "fb_seal_to_free_float_pressure"
F1_EPS_A_DAY = 0.01
F1_FORMULA = "log1p(V_seal / (A_below + 0.01*A_day))"
F2_FORMULA = "log(AvgBuySize_LXL / AvgSellSize_LXL)"
F3_FORMULA = "V_seal / FFMV"
F1_DENOM_NOTE = "A_below = 首次有效触板后分钟 close < 涨停价的成交额之和；不是 ActualSellVolume"
F2_NOTE = "L+XL 档；两边订单数>0；禁止分母补1；不能解读为机构/游资"
F3_NOTE = "FFMV = as-of-D free_float_shares * D 未复权收盘；禁止总市值顶替"

FACTOR_NAMES = (F1_NAME, F2_NAME, F3_NAME)

# 主实验：因子同时进 five 与 ten。D 为单头消融，DIAGNOSTIC_ONLY。
EXPERIMENTS: tuple[dict[str, Any], ...] = (
    {"code": A00, "role": "exact_v4_replay", "five": (), "ten": (), "retrain": False, "formal": True},
    {"code": A01, "role": "plus_F1", "five": (F1_NAME,), "ten": (F1_NAME,), "retrain": True, "formal": True},
    {"code": A02, "role": "plus_F2", "five": (F2_NAME,), "ten": (F2_NAME,), "retrain": True, "formal": True},
    {"code": A03, "role": "plus_F3", "five": (F3_NAME,), "ten": (F3_NAME,), "retrain": True, "formal": True},
    {"code": B01, "role": "F1_F2", "five": (F1_NAME, F2_NAME), "ten": (F1_NAME, F2_NAME), "retrain": True, "formal": True},
    {"code": B02, "role": "F1_F3", "five": (F1_NAME, F3_NAME), "ten": (F1_NAME, F3_NAME), "retrain": True, "formal": True},
    {"code": B03, "role": "F2_F3", "five": (F2_NAME, F3_NAME), "ten": (F2_NAME, F3_NAME), "retrain": True, "formal": True},
    {"code": C01, "role": "F1_F2_F3", "five": (F1_NAME, F2_NAME, F3_NAME), "ten": (F1_NAME, F2_NAME, F3_NAME), "retrain": True, "formal": True},
    {"code": D01, "role": "F1_five_only", "five": (F1_NAME,), "ten": (), "retrain": True, "formal": False},
    {"code": D02, "role": "F1_ten_only", "five": (), "ten": (F1_NAME,), "retrain": True, "formal": False},
    {"code": D03, "role": "F2_five_only", "five": (F2_NAME,), "ten": (), "retrain": True, "formal": False},
    {"code": D04, "role": "F2_ten_only", "five": (), "ten": (F2_NAME,), "retrain": True, "formal": False},
    {"code": D05, "role": "F3_five_only", "five": (F3_NAME,), "ten": (), "retrain": True, "formal": False},
    {"code": D06, "role": "F3_ten_only", "five": (), "ten": (F3_NAME,), "retrain": True, "formal": False},
)

PROMOTION_GATES = {
    "same_window_sharpe_competitive": True,
    "maxdd_worsen_cap": MAXDD_WORSEN_CAP,
    "not_only_2024_2025": True,
    "plus20_plus30_no_sign_flip": True,
    "capacity_300k_1m_3m_no_flip": True,
    "pit_and_coverage_pass": True,
    "coverage_min": COVERAGE_MIN,
    "prior_research_selection_exists": PRIOR_RESEARCH_SELECTION_EXISTS,
    "family_spa_holm_maxstat": True,
    "d_series_not_eligible": True,
    "do_not_merge_r16_o01": True,
    "do_not_auto_promote_current": True,
    "current_default": CURRENT_BASELINE,
    "keep_default": KEEP_BASELINE_V4,
}

STAGES = (
    "read",
    "inventory",
    "parity",
    "recover",
    "build",
    "pit",
    "freeze",
    "train",
    "backtest",
    "stats",
    "forward",
    "report",
)

PROTOCOL_SOURCE_FILES = (
    "core/round18/__init__.py",
    "core/round18/contract.py",
    "core/round18/common.py",
    "core/round18/read.py",
    "core/round18/inventory.py",
    "core/round18/overlap.py",
    "core/round18/parity.py",
    "core/round18/download.py",
    "core/round18/f1.py",
    "core/round18/f2.py",
    "core/round18/f3.py",
    "core/round18/factors.py",
    "core/round18/pit.py",
    "core/round18/freeze.py",
    "core/round18/train.py",
    "core/round18/backtest.py",
    "core/round18/runner.py",
    "core/round18/stats.py",
    "core/round18/decision.py",
    "core/round18/figures.py",
    "core/round18/docs.py",
    "core/round18/pipeline.py",
    "tools/run_round18.py",
    "tools/download_round18_data.py",
)

FORBIDDEN = (
    "edit_AGENTS_md",
    "edit_no_xgb_subsample_rule",
    "edit_core_round7",
    "edit_stock_src_or_runs",
    "fill_missing_features_with_zero",
    "search_epsilon_after_nav",
    "search_timing_b03_exits_hyperparams_seeds",
    "bagging_08_08",
    "merge_r16_o01_option_overlay",
    "auto_promote_CURRENT_v5",
    "write_写入时间_to_tracker",
    "promote_from_D_series",
    "substitute_total_mv_for_ffmv",
    "silent_swap_f2_pit",
)

MIN1_2013 = Path(
    "/data1/stock-mxy/data/source_snapshots/amazingdata-cn-a-close-limit-min1-20130104-20211231-v1"
)
MIN1_2022 = Path(
    "/data1/stock-mxy/data/source_snapshots/amazingdata-cn-a-close-limit-min1-20220104-20260724-v1"
)
HERMES_RAW = Path("/data1/Stock/data/acquisitions/hermes-cn-a-2010-20260724-v7-rebuild/raw")
HERMES_SCHEMA = Path(
    "/data1/Stock/data/acquisitions/hermes-cn-a-2010-20260724-v7-rebuild/source_schema.json"
)
HERMES_V7 = Path("/data1/stock-mxy/data/datasets/hermes-cn-a-2010-20260724-v7-all-features")
MF_RAW = HERMES_RAW / "mkt_equ_mf_new"

R17_T4033_EXTRAS = (
    "r17_a_eb_size_success",
    "r17_a_stock_prior_success",
    "r17_e_prior_fb_count_log",
)

NEIGHBORS = {
    F1_NAME: (
        "fb_post_touch_drawdown",
        "fb_close_seal_streak_norm",
        "h5_close_limit_value_to_amount",
    ),
    F2_NAME: ("h4_large_net_to_turnover", "r17_c_large_minus_retail"),
    F3_NAME: (
        "h5_close_limit_value_to_amount",
        "g0_float_mv_log",
        "r17_d_mv_xs_rank",
    ),
}


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


def copy_plan() -> Path:
    from ..paths import assert_project_result_path, ensure_round18_tree

    ensure_round18_tree()
    assert_project_result_path(PLAN_COPY)
    text = CURSOR_PLAN.read_text(encoding="utf-8")
    header = "<!-- 本文件是开工计划的课题落盘副本。禁止回写 Cursor plan 文件。 -->\n\n"
    if not text.startswith("<!-- 本文件是开工计划"):
        PLAN_COPY.write_text(header + text, encoding="utf-8")
    else:
        PLAN_COPY.write_text(text, encoding="utf-8")
    return PLAN_COPY


def exp_dir(code: str) -> Path:
    return ROUND18_ROOT / "candidates" / str(code)


def parity_dir(code: str = A00) -> Path:
    return ROUND18_ROOT / "parity" / str(code)
