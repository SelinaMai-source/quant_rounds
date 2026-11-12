"""Round 16 冻结契约。期权数据恢复 × 预注册期权 Alpha。默认 KEEP_BASELINE_V4。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import RESULT_ROOT, ROUND15_ROOT, ROUND16_ROOT
from ..round11.contract import (
    BASELINE_V4,
    BLEND_PRED,
    PARITY_ABS_NAV,
    PARITY_METRIC_ABS,
    RISK_PRED,
    SNAPSHOT_FINGERPRINT,
    SNAPSHOT_ID,
    STRATEGY_SOURCE_FILES,
)
from ..round12.contract import RANK_NAV, TOP_K
from ..round15.contract import R12_01, R12_01_DIR, R12_01_TARGETS, R13_00_DIR, R14_00_DIR

KEEP_BASELINE_V4 = "KEEP_BASELINE_V4"
RECOMMENDED_BASELINE_V5 = "RECOMMENDED_BASELINE_V5"
AWAITING_USER_LOCK = "AWAITING_USER_LOCK"
AUTO_PROMOTE_V5 = False
PARENT_V4 = "R9_B03"
FORWARD_OOS_NOT_AVAILABLE = "FORWARD_OOS_NOT_AVAILABLE"
FULL_PERIOD_NOT_AVAILABLE = "FULL_PERIOD_NOT_AVAILABLE"
FULL_HISTORY_NOT_AVAILABLE = "FULL_HISTORY_NOT_AVAILABLE"
INCONCLUSIVE_LOW_EVENT_COUNT = "INCONCLUSIVE_LOW_EVENT_COUNT"
SKIP_FACTOR_FOR_TRADING = "SKIP_FACTOR_FOR_TRADING"
SKIP_ADVANCED_FACTORS = "SKIP_ADVANCED_FACTORS"
HISTORICAL_DEVELOPMENT_ONLY = "HISTORICAL_DEVELOPMENT_ONLY"

MANIFEST_NAME = "round16_manifest_v1.yaml"
CURSOR_PLAN = Path("/home/mxy/.cursor/plans/round16_option_recovery_bd9794d5.plan.md")
PLAN_COPY = ROUND16_ROOT / "ROUND16_PLAN.md"
TRACKER_PATH = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
FINAL_REPORT = ROUND16_ROOT / "ROUND16_OPTION_RESEARCH_REPORT.md"
HANDOFF = ROUND16_ROOT / "ROUND17_RESEARCH_HANDOFF.md"

R16_00 = "R16_00_CORRECTED_V4"
O01 = "R16_O01"
O02 = "R16_O02"
O03 = "R16_O03"
O04 = "R16_O04"
OPTION_IDS = (O01, O02, O03, O04)
PRE_REGISTERED_IDS = (R16_00, *OPTION_IDS)

R15_00_DIR = ROUND15_ROOT / "parity" / "R15_00_CORRECTED_V4"
R15_DATA = ROUND15_ROOT / "data"

# Round15 漏了 588080；SDK 实际 ETF 期权还包括 159901/159919/159922。
ETF_UNDERLYINGS = (
    "510050.SH",
    "510300.SH",
    "510500.SH",
    "159915.SZ",
    "588000.SH",
    "588080.SH",
    "159901.SZ",
    "159919.SZ",
    "159922.SZ",
)
OPTION_SECURITY_TYPE = "EXTRA_ETF_OP"
KLINE_BEGIN = 20150101
KLINE_END = 20260724

PCT_WINDOW = 252
PCT_MIN_OBS = 126
SKEW_Q = 0.90
IV_DELTA_Q = 0.90
O02_HALVE = 0.5
O04_BETA_WINDOW = 60
PERM_N = 1000
BLOCK_DAYS = 20
BOOT_N = 5000
BOOT_N_MIN = 2000
SPA_BOOT = 2000
RF_FALLBACK = 0.02

COV_MIN_DAYS = 400
COV_MIN_YEARS = 3
COV_MIN_SIGNAL = 250
IV_MIN_CHAIN = 4

V4_ANCHOR = {
    "cagr": 0.5253672918108749,
    "sharpe": 1.8120914675763191,
    "max_drawdown": -0.25455988069271596,
    "completed_trades": 1149,
}

VERDICTS = (
    "A.CURRENT_CONTRACT_ONLY_BUG",
    "B.DATE_RANGE_BUG",
    "C.CACHE_OR_PAGINATION_BUG",
    "D.HISTORICAL_CONTRACT_MAPPING_BUG",
    "E.SOURCE_PERMISSION_LIMIT",
    "F.DATA_PROVIDER_HISTORY_LIMIT",
    "G.MULTIPLE_CAUSES",
    "H.ROOT_CAUSE_NOT_IDENTIFIED",
)

STAGES = (
    "audit",
    "download",
    "coverage",
    "chain",
    "quality",
    "iv",
    "factors",
    "incremental",
    "redundancy",
    "backtest",
    "controls",
    "stress",
    "stats",
    "forward",
    "report",
)

PROTOCOL_SOURCE_FILES = (
    "core/round16/__init__.py",
    "core/round16/contract.py",
    "core/round16/common.py",
    "core/round16/sdk.py",
    "core/round16/parity.py",
    "core/round16/audit.py",
    "core/round16/download.py",
    "core/round16/coverage.py",
    "core/round16/chain.py",
    "core/round16/quality.py",
    "core/round16/iv.py",
    "core/round16/factors.py",
    "core/round16/freeze.py",
    "core/round16/incremental.py",
    "core/round16/redundancy.py",
    "core/round16/backtest.py",
    "core/round16/controls.py",
    "core/round16/stress.py",
    "core/round16/stats.py",
    "core/round16/docs.py",
    "core/round16/pipeline.py",
    "tools/run_round16.py",
    "tools/download_round16_options.py",
)

CANDIDATE_FORMULAS = {
    O01: "parent=v4. PUT_SKEW25 > past-only 90th of 252 valid option days (min 126) → Top2 MIN_KEEP, Top1 unchanged. No Top3 fill.",
    O02: "parent=v4. IV_SHOCK=ATM_IV30(D)-ATM_IV30(D-5) > past-only 90th → new entries target budget ×0.5; do not cut existing holdings.",
    O03: "parent=v4. PUT_SKEW25 AND PUT_WING_CONVEXITY both > own past-only 90th → Top2 MIN_KEEP. F3 missing → SKIP_O03.",
    O04: "stock vs matching-ETF PIT beta × that ETF option tail; past-only ridge residual correction + OOF + purge/embargo; five/ten frozen. No cross-section → SKIP_O04.",
}

FORBIDDEN = (
    "retrain_five_ten",
    "feature_search",
    "change_90_10",
    "add_seed",
    "top2_0_5_10_grid",
    "complex_timing_exit",
    "future_option_quotes",
    "choose_factors_from_nav",
    "change_thresholds_after_nav",
    "full_sample_percentile",
    "auto_promote_CURRENT_v5",
    "write_写入时间_to_tracker",
    "invent_option_data",
    "run_round17",
)


def experiment_dir(code: str) -> Path:
    return ROUND16_ROOT / "candidates" / str(code)


def control_dir(code: str) -> Path:
    return ROUND16_ROOT / "controls" / str(code)


def parity_dir(code: str) -> Path:
    return ROUND16_ROOT / "parity" / str(code)


def overlayed_path(code: str) -> Path:
    return ROUND16_ROOT / "weights" / f"overlayed_{code}.parquet"


def copy_plan() -> Path:
    from ..paths import assert_project_result_path, ensure_round16_tree

    ensure_round16_tree()
    assert_project_result_path(PLAN_COPY)
    text = CURSOR_PLAN.read_text(encoding="utf-8")
    header = "<!-- 本文件是开工计划的课题落盘副本。禁止回写 Cursor plan 文件。 -->\n\n"
    if not text.startswith("<!-- 本文件是开工计划"):
        PLAN_COPY.write_text(header + text, encoding="utf-8")
    else:
        PLAN_COPY.write_text(text, encoding="utf-8")
    return PLAN_COPY


def pre_registered_spec(code: str) -> dict[str, Any]:
    table: dict[str, dict[str, Any]] = {
        R16_00: {
            "role": "corrected_v4_parity_anchor",
            "fee": "HISTORICAL_FEE_CORRECTED",
            "parent": R12_01,
            "is_v5": False,
        },
    }
    for cid in OPTION_IDS:
        table[cid] = {
            "role": "option_alpha_candidate",
            "fee": "HISTORICAL_FEE_CORRECTED",
            "formula": CANDIDATE_FORMULAS[cid],
            "parent": "v4",
            "family": "option",
            "is_v5": False,
            "search": False,
        }
    if code not in table:
        raise KeyError(code)
    return {"code": code, **table[code]}


__all__ = [
    "AWAITING_USER_LOCK",
    "AUTO_PROMOTE_V5",
    "BASELINE_V4",
    "BLEND_PRED",
    "BLOCK_DAYS",
    "BOOT_N",
    "BOOT_N_MIN",
    "CANDIDATE_FORMULAS",
    "COV_MIN_DAYS",
    "COV_MIN_SIGNAL",
    "COV_MIN_YEARS",
    "CURSOR_PLAN",
    "ETF_UNDERLYINGS",
    "FINAL_REPORT",
    "FORBIDDEN",
    "FORWARD_OOS_NOT_AVAILABLE",
    "FULL_HISTORY_NOT_AVAILABLE",
    "FULL_PERIOD_NOT_AVAILABLE",
    "HANDOFF",
    "HISTORICAL_DEVELOPMENT_ONLY",
    "INCONCLUSIVE_LOW_EVENT_COUNT",
    "IV_DELTA_Q",
    "IV_MIN_CHAIN",
    "KEEP_BASELINE_V4",
    "KLINE_BEGIN",
    "KLINE_END",
    "MANIFEST_NAME",
    "O01",
    "O02",
    "O02_HALVE",
    "O03",
    "O04",
    "O04_BETA_WINDOW",
    "OPTION_IDS",
    "OPTION_SECURITY_TYPE",
    "PARITY_ABS_NAV",
    "PARITY_METRIC_ABS",
    "PARENT_V4",
    "PCT_MIN_OBS",
    "PCT_WINDOW",
    "PERM_N",
    "PLAN_COPY",
    "PRE_REGISTERED_IDS",
    "PROTOCOL_SOURCE_FILES",
    "R12_01",
    "R12_01_DIR",
    "R12_01_TARGETS",
    "R13_00_DIR",
    "R14_00_DIR",
    "R15_00_DIR",
    "R15_DATA",
    "R16_00",
    "RANK_NAV",
    "RECOMMENDED_BASELINE_V5",
    "RF_FALLBACK",
    "RISK_PRED",
    "SKIP_ADVANCED_FACTORS",
    "SKIP_FACTOR_FOR_TRADING",
    "SKEW_Q",
    "SNAPSHOT_FINGERPRINT",
    "SNAPSHOT_ID",
    "SPA_BOOT",
    "STAGES",
    "STRATEGY_SOURCE_FILES",
    "TOP_K",
    "TRACKER_PATH",
    "V4_ANCHOR",
    "VERDICTS",
    "copy_plan",
    "control_dir",
    "experiment_dir",
    "overlayed_path",
    "parity_dir",
    "pre_registered_spec",
]
