"""Round 22.4 契约。经济门沿用 R21_3，不改。"""

from __future__ import annotations

from core.paths import RESULT_ROOT
from core.round22_3.contract import (
    C1_LAG1_ID,
    C4_ID,
    C_AVAILABLE_FROM,
    COLSAMPLE,
    ECONOMIC_MATERIALITY,
    FEE_SCHEDULE,
    FIVE_FEATURES,
    FIVE_HEAD,
    MAIN_WINDOW_END,
    MAIN_WINDOW_START,
    SEEDS,
    SUBSAMPLE,
    TEN_FEATURES,
    TEN_HEAD,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
)

ROUND = "22.4"
ROOT = RESULT_ROOT / "round22_4"
KEEP = "KEEP_BASELINE_V5"
RECOMMEND = "RECOMMEND_BASELINE_V6_RESEARCH"
SCOPES = (
    "FULL_HISTORY_OFFICIAL",
    "MATCHED_SUBPERIOD_DIAGNOSTIC",
    "FORWARD_PROSPECTIVE",
    "EXPLORATORY",
    "NOT_COMPARABLE",
)
FS_CHANGE_ID = "r22_4_fs2_cash_conversion_change"
FS1_ID = "r22_3_fs1_accrual_minus_ocf"
FS2_ID = "r22_3_fs2_ocf_over_assets"
ALLNAN_COL = "r22_4_c4_allnan"
SHUFFLE_COL = "r22_4_c4_shuffle_source"
RHO_GATE = 0.60

# 看现金转化 IC 之前冻结。阈值来自既有 DROP_WEAK 的反面，不是按 0.026 现调。
FS_EVIDENCE = {
    "id": "R224_FS_EVIDENCE_V1",
    "frozen_before_new_factor_ic": True,
    "source": "core/round21/redundancy.py DROP_WEAK 的反面，再要求 RankIC 为正",
    "min_event_n": 200,
    "min_abs_rankic": 0.02,
    "min_abs_icir": 0.20,
    "require_positive_rankic": True,
    "require_pit": True,
    "never_fill_zero": True,
    "event_level_not_repeated_trade_days": True,
    "pass_rule": "PIT 且事件样本够，且 RankIC>0，且（|RankIC|>=0.02 或 |ICIR|>=0.20）",
}

NULL_CONTROLS = (
    {"code": "R224_CTRL_V5", "kind": "baseline", "column": None},
    {"code": "R224_CTRL_ALLNAN", "kind": "all_nan", "column": ALLNAN_COL},
    {"code": "R224_CTRL_SHUFFLE", "kind": "train_fold_shuffle", "column": SHUFFLE_COL},
)

STAGES = (
    "freeze",
    "parity",
    "lineage",
    "tracker",
    "coverage",
    "period",
    "null",
    "redundancy",
    "fs",
    "decision",
    "report",
)


def evidence_pass(rankic: float | None, icir: float | None, n_events: int, pit_ok: bool) -> bool:
    if not pit_ok or n_events < int(FS_EVIDENCE["min_event_n"]):
        return False
    if rankic is None or not (rankic > 0):
        return False
    ic_ok = abs(rankic) >= float(FS_EVIDENCE["min_abs_rankic"])
    ir_ok = icir is not None and abs(icir) >= float(FS_EVIDENCE["min_abs_icir"])
    return bool(ic_ok or ir_ok)
