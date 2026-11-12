"""Round 23.1 冻结契约。不按日 Spearman 选权重，不搜索新阈值，不做三种子均值，不用 CatBoost。"""

from __future__ import annotations

from ..paths import RESULT_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, TEN_FEATURES, TEN_HEAD

ROOT = RESULT_ROOT / "round23_1"
FORMAL_TEN_WEIGHTS = (0.50, 0.75, 0.90, 1.00)
DIAGNOSTIC_TEN_WEIGHT = 0.0
LGBM_BLEND = (0.25, 0.50, 0.75)
LGBM_PARAMS = {
    "objective": "lambdarank",
    "learning_rate": 0.03,
    "n_estimators": 200,
    "num_leaves": 7,
    "max_depth": 3,
    "min_data_in_leaf": 50,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "verbosity": -1,
    "seed": 17,
}
COMMON_LABEL = "label_exit_rule_net"
TIE_BREAK = "closest_to_0.90_then_higher_ten_weight"
MIN_INNER_DAYS = 120
DIVERSITY_SPEARMAN_MAX = 0.90
DIVERSITY_TOP1_DISAGREE_MIN = 0.10

HYPOTHESES = (
    {"id": "H231_O1_NOT_PROMOTED", "claim": "旧 Spearman 目标只作为失败对照，不重新晋级"},
    {"id": "H231_O4_BLEND", "claim": "内折 O4 选出的 Five/Ten 权重相对 v5 过经济门、统计门、Top 变化和执行门"},
    {"id": "H231_LGBM_ENSEMBLE", "claim": "多样性门通过后的 LGBM 融合过同一组晋级门"},
    {"id": "H231_RULES_EXPLAINED", "claim": "R1/R2/R3 的净收益下降能用过滤后的替补或执行假设解释，而不是未记录的引擎错单"},
)
