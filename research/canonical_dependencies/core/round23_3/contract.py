"""Round 23.3 冻结契约。"""

from __future__ import annotations

from core.paths import RESULT_ROOT
from core.round22_6.contract import anchor_copy as _anchor

ROOT = RESULT_ROOT / "round23_3"
R22_3 = RESULT_ROOT / "round22_3"
E0 = R22_3 / "candidates" / "R223_E0"
PANEL = R22_3 / "data" / "five_candidate.parquet"
TRAIN = R22_3 / "data" / "five_training.parquet"
TEN_PANEL = R22_3 / "data" / "ten_candidate.parquet"
TEN_TRAIN = R22_3 / "data" / "ten_training.parquet"
AUDIT = ROOT / "_audit"

TEN_WEIGHTS = (0.75, 0.90, 1.00)
LGBM_WEIGHTS = (0.00, 0.10, 0.25, 0.50)
DEFAULT_W_TEN = 0.90
DEFAULT_W_LGBM = 0.00

INNER_BOOT_N = 5000
INNER_BLOCK = 20
MIN_INNER_DAYS = 120
MIN_INNER_TRADES = 40
MAX_INNER_NAV_DAYS = 200  # 半年块若出现几千 NAV 日立即 FAIL
MAXDD_WORSEN = 0.02
BOOT_LB_Q = 0.10

LGBM_PARAMS = {
    "objective": "lambdarank",
    "learning_rate": 0.03,
    "n_estimators": 200,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 50,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "seed": 17,
}

E4_TRIGGER_DELTA = 0.002
E4_MIN_PAIRS = 250


def anchor_copy() -> dict:
    return _anchor()
