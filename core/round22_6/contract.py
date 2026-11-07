"""Round 22.6 冻结契约。看结果前写入，之后不改假设。"""

from __future__ import annotations

from core.paths import RESULT_ROOT
from core.round21.contract import V5_ANCHOR, V5_NAV_END_EXACT
from core.round22_3.contract import (
    ECONOMIC_MATERIALITY,
    FEE_SCHEDULE,
    FIVE_FEATURES,
    MAIN_WINDOW_END,
    MAIN_WINDOW_START,
    TEN_FEATURES,
)

ROOT = RESULT_ROOT / "round22_6"
R22_3 = RESULT_ROOT / "round22_3"
R22_5 = RESULT_ROOT / "round22_5"
E0 = R22_3 / "candidates" / "R223_E0"
PANEL = R22_3 / "data" / "five_candidate.parquet"
TRAIN = R22_3 / "data" / "five_training.parquet"
TEN_PANEL = R22_3 / "data" / "ten_candidate.parquet"
TEN_TRAIN = R22_3 / "data" / "ten_training.parquet"
FS2_PANEL = R22_3 / "data_recovery" / "fs2_panel.parquet"

C1 = "r22_2_c1_xl_net_imbalance_lag1"
C4 = "r22_c4_flow_persist_5"
FS2 = "r22_3_fs2_ocf_over_assets"
FS2_FORMULA = "OperatingCashFlow/AverageAssets"
C4_FORMULA = "mean(sign(NET_FLOW).shift(1).rolling(5, min_periods=3))"
PERM_SEEDS = (17, 42, 73)
CHALLENGER_BUFFER = 0.002
CHALLENGER_MIN_TRAIN_PAIR_DAYS = 250
RIDGE_ALPHA = 10.0

HYPOTHESES = (
    {
        "id": "H226_FS2_PARTIAL_INFO",
        "claim": "FS2 相对 Five14/Ten13 的线性残差信息可识别（允许负向）",
        "pass_if": "LINEAR_PARTIAL_INFORMATION_DIAGNOSTIC 对至少一头有可识别日相关，且不把 raw IC 误称为 conditional IC",
    },
    {
        "id": "H226_FS2_DIRECT_ADD",
        "claim": "Five+FS2 或 Ten+FS2 相对 v5 有增量并过经济门",
        "pass_if": "F1 或 F2 增量门与 CORRECTED 经济门同时成立；否则 FS2_DIRECT_ADD_NOT_SUPPORTED",
    },
    {
        "id": "H226_C4_FEATURE_PLACEBO",
        "claim": "真实 C4 增量明显区别于仅打乱 C4 列的 feature placebo",
        "pass_if": "真实 C4 与三组 placebo 对照完成；不得报告伪精确 p 值",
    },
    {
        "id": "H226_CHALLENGER_FS2",
        "claim": "在冻结 v5 上仅交换 rank2/3 时，CH2 相对 CH0 与 CH1 均改善",
        "pass_if": "若成立则 LOCAL_FS2_INCREMENT_CANDIDATE；仅 CH1 改善则为 LOCAL_REORDER_EFFECT_ONLY",
    },
)


def anchor_copy() -> dict:
    return {
        "code": "R16_O01",
        "window": [MAIN_WINDOW_START, MAIN_WINDOW_END],
        "sharpe": V5_ANCHOR["sharpe"],
        "cagr": V5_ANCHOR["cagr"],
        "max_drawdown": V5_ANCHOR["max_drawdown"],
        "nav_end": V5_NAV_END_EXACT,
        "trades": 1107,
        "capital": 300000,
        "fee_schedule": FEE_SCHEDULE,
        "information_lane": "LEGACY_OPEN_INFORMATION_LANE",
    }
