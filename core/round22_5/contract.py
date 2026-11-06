"""Round 22.5 冻结契约。看结果前写入，之后不改假设。"""

from __future__ import annotations

from pathlib import Path

from ..paths import RESULT_ROOT
from ..round21.contract import V5_ANCHOR, V5_NAV_END_EXACT
from ..round22_3.contract import (
    C_AVAILABLE_FROM,
    ECONOMIC_MATERIALITY,
    FEE_SCHEDULE,
    FIVE_FEATURES,
    MAIN_WINDOW_END,
    MAIN_WINDOW_START,
    TEN_FEATURES,
)

ROOT = RESULT_ROOT / "round22_5"
R22_3 = RESULT_ROOT / "round22_3"
R22_4 = RESULT_ROOT / "round22_4"
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
SHUFFLE_SHARPE = 1.8125087429942996
PERM_SEEDS = (17, 42, 73)
FULL_TRAIN_START = "2024-01-01"

HYPOTHESES = (
    {
        "id": "H225_C1_FULL_TRAIN_INCREMENT",
        "claim": "C1 在 2024-01-01 起的 FULL_TRAIN 上相对 v5 有稳定增量",
        "pass_if": "FULL_TRAIN wealth ratio>1 且切片超额为负的结论不成立，并且全历史经济门通过",
    },
    {
        "id": "H225_C4_ECONOMIC",
        "claim": "C4 的全历史增量过 R21_3 经济门",
        "pass_if": "经济门全部为真",
    },
    {
        "id": "H225_ALLNAN_INVARIANT",
        "claim": "全 NaN 对照的 NAV 与 v5 相同",
        "pass_if": "NAV 与锁定锚点一致",
    },
    {
        "id": "H225_SHUFFLE_RECHECK",
        "claim": "既有 SHUFFLE 袖套 Sharpe 等于冻结数字",
        "pass_if": "与 1.8125087429942996 的差小于 1e-9",
    },
    {
        "id": "H225_PERM_NOT_ALPHA",
        "claim": "日内截面置换不是可晋级 alpha",
        "pass_if": "三颗种子都不计 alpha 且不自动晋级",
    },
    {
        "id": "H225_FS2_OOF_SCREEN",
        "claim": "FS2=OCF/平均资产 通过 PIT 与 OOF 筛选并可进入袖套",
        "pass_if": "PIT 通过且至少 3 个完整外折年且 OOF Top2 成熟标签为正且至少 60% 外折同号",
    },
    {
        "id": "H225_NAV_RECONCILED",
        "claim": "v5 净值桥包含现金、费用、未实现和复利",
        "pass_if": "现金+市值等于总资产且闭合残差相对净值不超过 1%",
    },
    {
        "id": "H225_C4_ROLLING_ROWWISE",
        "claim": "C4 滚动按行，不把日历空档插进窗口",
        "pass_if": "合成缺口上按行窗口与按日历窗口不一致，且原公式未改",
    },
)

SCOPES = (
    "FULL_HISTORY_FIRST_TO_SECOND",
    "STRICT_PIT_SUBPERIOD_FIRST_TO_SECOND",
    "SECOND_TO_THIRD_INDEPENDENT",
    "EXECUTION_SCENARIO",
    "DIAGNOSTIC_ONLY",
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
    }
