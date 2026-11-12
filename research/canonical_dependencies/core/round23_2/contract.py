"""Round 23.2 冻结契约。"""

from __future__ import annotations

from core.paths import RESULT_ROOT
from core.round22_6.contract import anchor_copy as _anchor

ROOT = RESULT_ROOT / "round23_2"
R22_3 = RESULT_ROOT / "round22_3"
R23_1 = RESULT_ROOT / "round23_1"
E0 = R22_3 / "candidates" / "R223_E0"
PANEL = R22_3 / "data" / "five_candidate.parquet"
TRAIN = R22_3 / "data" / "five_training.parquet"
TEN_PANEL = R22_3 / "data" / "ten_candidate.parquet"
TEN_TRAIN = R22_3 / "data" / "ten_training.parquet"

LGBM_WEIGHTS = (0.00, 0.10, 0.25)
TEN_WEIGHTS = (0.75, 0.90, 1.00)
DEFAULT_W_LGBM = 0.00
DEFAULT_W_TEN = 0.90
GAP7 = 0.07
INNER_BOOT_N = 1000
INNER_BLOCK = 20
MIN_INNER_DAYS = 120
MIN_INNER_TRADES = 40

HYPOTHESES = (
    {"id": "H232_ENDPOINT_IDENTITY", "claim": "w_ten=0.90 且 w_lgbm=0 与冻结 v5 预测/排序/订单/NAV 一致", "pass_if": "endpoint_parity PASS"},
    {"id": "H232_LGBM_TOPK", "claim": "truncation_level Five=4/Ten=5 相对旧 LGBM 有条件增量", "pass_if": "正式融合证据来自内层重放，而非多样性"},
    {"id": "H232_M2_ZERO_OK", "claim": "若无增量则各年可选 w_lgbm=0", "pass_if": "允许全年份默认 0"},
    {"id": "H232_RISK_VS_ALPHA", "claim": "R1 GAP7 半仓相对 R2 matched-exposure 有否增量", "pass_if": "分账报告，不自动替换 alpha baseline"},
)


def anchor_copy() -> dict:
    return _anchor()
