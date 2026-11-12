"""Round 5 预注册：默认对照 baseline_v3 = R4_B05。不自动改锁。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..paths import FS00263_FEATURES, ROUND4_ROOT, round5_dir
from ..round2.data import SEEDS
from ..round4.registry import B01A, C03_HALF, RANDOM_SEEDS, stack

PARENT_DEFAULT = "baseline_v3"
BASELINE_V3 = "R4_B05"
BASELINE_V3_ALIAS = "R5_00_baseline_v3"

V3_CAGR = 0.2386
V3_SHARPE = 1.454
V3_MDD = -0.1428
V3_EX2024 = 0.1601
V3_TRADES = 877
LOW_ACTIVITY_TRADES = 526  # 877 * 0.60

V3_MARKET: dict[str, Any] = {
    "kind": "combine",
    "how": "min",
    "parts": [
        dict(B01A),
        {"kind": "eco_relimit_p250", "low": 0.20, "low_mult": 0.5},
    ],
}
V3_CONFIDENCE: dict[str, Any] = dict(C03_HALF)
V3_OVERLAY: dict[str, Any] = stack(V3_MARKET, V3_CONFIDENCE)

TRAIN_SEEDS: tuple[int, ...] = SEEDS  # 17/42/73/101/202
LOCKED_FEATURES: tuple[str, ...] = tuple(FS00263_FEATURES)

FAMILY_LIMIT = "L"
FAMILY_MONEY = "M"
FAMILY_INTRA = "I"
FAMILY_REL = "R"
FAMILY_LIQ = "V"
FAMILIES: tuple[str, ...] = (FAMILY_LIMIT, FAMILY_MONEY, FAMILY_INTRA, FAMILY_REL, FAMILY_LIQ)
MAX_FORMAL_FACTORS = 15
MAX_FAMILY_PICK = 3
MAX_FAMILY_COMBOS = 6
MAX_REDUCED_NEW = 4
MAX_FINALISTS = 5

HARD_CAGR = 0.22
HARD_SHARPE = 1.35
HARD_MDD = -0.18
V4_EX2024 = 0.1701  # v3 16.01 + 1pct


@dataclass
class R5Spec:
    code: str
    title_zh: str
    description_zh: str
    overlay: dict[str, Any] = field(default_factory=dict)
    parent: str = PARENT_DEFAULT
    exit_mode: str = "close_from_d2"
    name_budget_nav_fraction: float = 0.3
    market_up_ratio_min: float = 0.45
    top_k: int = 1
    rank_name_budget_nav_fractions: tuple[float, ...] | None = None
    random_seed: int | None = None
    skip_years: tuple[int, ...] = ()
    seeds: tuple[int, ...] | None = None
    seed_leave_out: int | None = None
    cost_overrides: dict[str, Any] = field(default_factory=dict)
    stage: str = "single"
    tags: tuple[str, ...] = ()
    pit_ok: bool = True
    diagnostic: bool = False
    alpha_source: str = "locked_fs00263"
    prediction_key: str | None = None
    features: tuple[str, ...] = ()
    objective: str = "rank:ndcg"
    eval_metric: str = "ndcg@3"
    target: str = "label_e2_net"
    n_estimators: int = 50
    max_depth: int = 2
    learning_rate: float = 0.05
    family: str | None = None
    stage1_kind: str | None = None

    def to_config(self) -> dict[str, Any]:
        return asdict(self)


SPECS: dict[str, R5Spec] = {}


def _register(spec: R5Spec) -> R5Spec:
    SPECS[spec.code] = spec
    return spec


def get_spec(code: str) -> R5Spec:
    key = str(code or "").strip()
    if key not in SPECS:
        raise KeyError(f"未知 R5 代号 {key!r}")
    return SPECS[key]


def list_codes() -> list[str]:
    return list(SPECS)


def clone_spec(spec: R5Spec, **kwargs) -> R5Spec:
    data = asdict(spec)
    data.update(kwargs)
    data["skip_years"] = tuple(data.get("skip_years") or ())
    data["tags"] = tuple(data.get("tags") or ())
    data["features"] = tuple(data.get("features") or ())
    if data.get("seeds") is not None:
        data["seeds"] = tuple(data["seeds"])
    if data.get("rank_name_budget_nav_fractions") is not None:
        data["rank_name_budget_nav_fractions"] = tuple(data["rank_name_budget_nav_fractions"])
    return R5Spec(**data)


def spec_from_overlay(
    code: str,
    title_zh: str,
    overlay: dict[str, Any] | None = None,
    **kwargs,
) -> R5Spec:
    return R5Spec(
        code=code,
        title_zh=title_zh,
        description_zh=kwargs.pop("description_zh", "") or title_zh,
        overlay=overlay or dict(V3_OVERLAY),
        **kwargs,
    )


def result_dir_for(code: str) -> Path:
    return round5_dir(code)


def r4_b05_dir() -> Path:
    return ROUND4_ROOT / BASELINE_V3


def v3_risk_overlay() -> dict[str, Any]:
    return dict(V3_OVERLAY)


# --- 门禁 ---
_register(R5Spec(
    code=BASELINE_V3_ALIAS,
    title_zh="复现 baseline_v3（R4_B05）",
    description_zh="锁分 fs00263 五种子集成 + R4_B05 overlay。必须与 R4_B05 逐笔一致。",
    overlay=dict(V3_OVERLAY),
    stage="parity",
    tags=("parity", "baseline_v3", "locked"),
    alpha_source="locked_fs00263",
    features=LOCKED_FEATURES,
))

# --- D 对照（特征在 univariate shortlist 后填）---
_register(R5Spec(
    code="R5_D00",
    title_zh="本地重训 6 因子对照（n50_d2, subsample=1.0）",
    description_zh="课题内重训锁 6 因子；允许与锁分不一致，必须单独报 Δ。",
    overlay=dict(V3_OVERLAY),
    stage="train_d",
    tags=("D", "control", "local_retrain"),
    alpha_source="local_retrain",
    prediction_key="D00_6factor_n50d2",
    features=LOCKED_FEATURES,
    target="label_e2_net",
))

for i, (fam, title) in enumerate(
    (
        (FAMILY_LIMIT, "Limit"),
        (FAMILY_MONEY, "MoneyFlow"),
        (FAMILY_INTRA, "Intraday"),
        (FAMILY_REL, "Relative"),
        (FAMILY_LIQ, "Liquidity"),
    ),
    start=1,
):
    _register(R5Spec(
        code=f"R5_D{i:02d}",
        title_zh=f"6 因子 + {title} 族",
        description_zh=f"锁 6 因子加上 Family {fam} shortlist（每族最多 3）。套 v3 风险层。",
        overlay=dict(V3_OVERLAY),
        stage="train_d",
        tags=("D", "family", fam),
        alpha_source="local_retrain",
        prediction_key=f"D{i:02d}_{fam}",
        features=LOCKED_FEATURES,
        family=fam,
        target="label_e2_net",
    ))

# --- E 逐个删除 ---
for i, feat in enumerate(LOCKED_FEATURES, start=1):
    kept = tuple(f for f in LOCKED_FEATURES if f != feat)
    _register(R5Spec(
        code=f"R5_E{i:02d}",
        title_zh=f"删 {feat}",
        description_zh="六因子逐个删除。全样本不差且 Ex2024/LOYO 更好才允许 reduced，不自动晋升。",
        overlay=dict(V3_OVERLAY),
        stage="train_e",
        tags=("E", "ablation"),
        alpha_source="local_retrain",
        prediction_key=f"E{i:02d}_drop_{feat}",
        features=kept,
        target="label_e2_net",
    ))

# --- F 目标函数 ---
_register(R5Spec(
    code="R5_F00",
    title_zh="复现 Stock objective（ndcg@3 + 5 档 + label_exit_rule_net）",
    description_zh="n50_d2 + subsample 1.0。本地重训对照，报 Δ。",
    overlay=dict(V3_OVERLAY),
    stage="train_f",
    tags=("F", "control", "local_retrain"),
    alpha_source="local_retrain",
    prediction_key="F00_stock_objective",
    features=LOCKED_FEATURES,
    objective="rank:ndcg",
    eval_metric="ndcg@3",
    target="label_exit_rule_net",
))
_register(R5Spec(
    code="R5_F01",
    title_zh="objective ndcg@1",
    description_zh="同 F00 特征/标签，eval ndcg@1。",
    overlay=dict(V3_OVERLAY),
    stage="train_f",
    tags=("F",),
    alpha_source="local_retrain",
    prediction_key="F01_ndcg1",
    features=LOCKED_FEATURES,
    objective="rank:ndcg",
    eval_metric="ndcg@1",
    target="label_exit_rule_net",
))
_register(R5Spec(
    code="R5_F02",
    title_zh="objective ndcg@3",
    description_zh="显式 ndcg@3（与 F00 同目标，独立跑一遍便于对照）。",
    overlay=dict(V3_OVERLAY),
    stage="train_f",
    tags=("F",),
    alpha_source="local_retrain",
    prediction_key="F02_ndcg3",
    features=LOCKED_FEATURES,
    objective="rank:ndcg",
    eval_metric="ndcg@3",
    target="label_exit_rule_net",
))
_register(R5Spec(
    code="R5_F03",
    title_zh="objective pairwise",
    description_zh="rank:pairwise。评价看 NDCG@1 / Top1 Lift / Ex2024 / LOYO / v3 组合。",
    overlay=dict(V3_OVERLAY),
    stage="train_f",
    tags=("F",),
    alpha_source="local_retrain",
    prediction_key="F03_pairwise",
    features=LOCKED_FEATURES,
    objective="rank:pairwise",
    eval_metric="ndcg@3",
    target="label_exit_rule_net",
))
_register(R5Spec(
    code="R5_F04",
    title_zh="回归 label_exit_rule_net",
    description_zh="XGBRegressor。Huber 若不支持则另标 SKIPPED_UNSUPPORTED。",
    overlay=dict(V3_OVERLAY),
    stage="train_f",
    tags=("F", "regressor"),
    alpha_source="local_retrain",
    prediction_key="F04_reg_exit_net",
    features=LOCKED_FEATURES,
    objective="reg:squarederror",
    eval_metric="rmse",
    target="label_exit_rule_net",
))
_register(R5Spec(
    code="R5_F05_huber",
    title_zh="Huber 回归（仅原生支持）",
    description_zh="reg:pseudohubererror；不支持则 SKIPPED_UNSUPPORTED。",
    overlay=dict(V3_OVERLAY),
    stage="train_f",
    tags=("F", "regressor", "optional"),
    alpha_source="local_retrain",
    prediction_key="F05_huber",
    features=LOCKED_FEATURES,
    objective="reg:pseudohubererror",
    eval_metric="rmse",
    target="label_exit_rule_net",
))

# --- G 标签 ---
_register(R5Spec(
    code="R5_G01",
    title_zh="标签 = 当前净收益 label_exit_rule_net",
    description_zh="与 F00 同目标，独立代号便于标签组对照。",
    overlay=dict(V3_OVERLAY),
    stage="train_g",
    tags=("G",),
    alpha_source="local_retrain",
    prediction_key="G01_exit_net",
    features=LOCKED_FEATURES,
    target="label_exit_rule_net",
))
_register(R5Spec(
    code="R5_G02",
    title_zh="标签 = D+1 连板二分类",
    description_zh="label_d1_touch；标签成熟时点须 < prediction_date。",
    overlay=dict(V3_OVERLAY),
    stage="train_g",
    tags=("G", "binary"),
    alpha_source="local_retrain",
    prediction_key="G02_d1_touch",
    features=LOCKED_FEATURES,
    objective="binary:logistic",
    eval_metric="logloss",
    target="label_d1_touch",
))
_register(R5Spec(
    code="R5_G03",
    title_zh="标签 = 大亏≤−5% 二分类",
    description_zh="label_exit_rule_net ≤ −5%。分数用 1-P(大亏) 以便 Top1。",
    overlay=dict(V3_OVERLAY),
    stage="train_g",
    tags=("G", "binary"),
    alpha_source="local_retrain",
    prediction_key="G03_bigloss",
    features=LOCKED_FEATURES,
    objective="binary:logistic",
    eval_metric="logloss",
    target="label_bigloss_5",
))

# --- H 两阶段 ---
for code, kind, title in (
    ("R5_H01", "touch", "两阶段：连板概率 → skip/half/full"),
    ("R5_H02", "ret", "两阶段：期望收益 → skip/half/full"),
    ("R5_H03", "downside", "两阶段：下行概率 → skip/half/full"),
):
    _register(R5Spec(
        code=code,
        title_zh=title,
        description_zh="Stage1 只出 skip/half/full，再进 v3 风险层，不突破 NAV30%。禁止混分。",
        overlay=dict(V3_OVERLAY),
        stage="train_h",
        tags=("H", "twostage"),
        alpha_source="locked_fs00263",
        features=LOCKED_FEATURES,
        stage1_kind=kind,
    ))


def dump_registry() -> str:
    return json.dumps({k: v.to_config() for k, v in SPECS.items()}, ensure_ascii=False, indent=2, default=str)
