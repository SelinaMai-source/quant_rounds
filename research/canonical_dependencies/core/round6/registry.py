"""Round 6 预注册：默认对照 baseline_v3 = R4_B05。不自动改锁。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..paths import FS00263_FEATURES, ROUND4_ROOT, ROUND5_ROOT, round6_dir
from ..round2.data import SEEDS
from ..round4.registry import B01A, C03_HALF, RANDOM_SEEDS, stack

PARENT_DEFAULT = "baseline_v3"
BASELINE_V3 = "R4_B05"
BASELINE_V3_ALIAS = "R6_00_baseline_v3"
CANONICAL_CODE = "R6_CANONICAL_RETRAIN"

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
DEFAULT_TARGET = "label_exit_rule_net"

HARD_CAGR = 0.23
HARD_SHARPE = 1.40
HARD_MDD = -0.16
V4_EX2024 = 0.1676  # v3 16.01 + 0.75pct
CHALLENGER_CAGR_SLACK = 0.005
CHALLENGER_SHARPE_ADD = 0.10
CHALLENGER_MDD_IMPROVE = 0.02
CHALLENGER_EX_SLACK = 0.005
LOYO_MEDIAN_ADD = 0.0075

MAX_FINALISTS = 5
MAX_K = 6
GROUP_ORDER: tuple[str, ...] = ("B", "C", "D", "E", "F", "G", "H", "K", "S")

HYBRID_PCT_FEATURES: tuple[str, ...] = (
    "ind_stock_seal_quality_rank",
    "ind_first_board_count_log",
    "fb_limit_close_minute_ratio",
    "h5_close_limit_retention",
)


@dataclass
class R6Spec:
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
    forensic_only: bool = False
    alpha_source: str = "locked_fs00263"
    prediction_key: str | None = None
    reuse_round5_key: str | None = None
    features: tuple[str, ...] = ()
    objective: str = "rank:ndcg"
    eval_metric: str = "ndcg@3"
    target: str = DEFAULT_TARGET
    n_estimators: int = 50
    max_depth: int = 2
    learning_rate: float = 0.05
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    window_kind: str = "3y"
    fold_rule: str = "stock"
    representation: str = "raw"
    half_life_months: int | None = None
    label_transform: str = "none"
    use_stock_rolling: bool | None = None
    meta_kind: str | None = None
    context_features: tuple[str, ...] = ()
    primary_code: str | None = None

    def to_config(self) -> dict[str, Any]:
        return asdict(self)


SPECS: dict[str, R6Spec] = {}


def _register(spec: R6Spec) -> R6Spec:
    SPECS[spec.code] = spec
    return spec


def get_spec(code: str) -> R6Spec:
    key = str(code or "").strip()
    if key not in SPECS:
        raise KeyError(f"未知 R6 代号 {key!r}")
    return SPECS[key]


def list_codes() -> list[str]:
    return list(SPECS)


def clone_spec(spec: R6Spec, **kwargs) -> R6Spec:
    data = asdict(spec)
    data.update(kwargs)
    data["skip_years"] = tuple(data.get("skip_years") or ())
    data["tags"] = tuple(data.get("tags") or ())
    data["features"] = tuple(data.get("features") or ())
    data["context_features"] = tuple(data.get("context_features") or ())
    if data.get("seeds") is not None:
        data["seeds"] = tuple(data["seeds"])
    if data.get("rank_name_budget_nav_fractions") is not None:
        data["rank_name_budget_nav_fractions"] = tuple(data["rank_name_budget_nav_fractions"])
    return R6Spec(**data)


def spec_from_overlay(
    code: str,
    title_zh: str,
    overlay: dict[str, Any] | None = None,
    **kwargs,
) -> R6Spec:
    return R6Spec(
        code=code,
        title_zh=title_zh,
        description_zh=kwargs.pop("description_zh", "") or title_zh,
        overlay=overlay or dict(V3_OVERLAY),
        **kwargs,
    )


def result_dir_for(code: str) -> Path:
    return round6_dir(code)


def r4_b05_dir() -> Path:
    return ROUND4_ROOT / BASELINE_V3


def r5_pred_dir(key: str) -> Path:
    return ROUND5_ROOT / "predictions" / key


def v3_risk_overlay() -> dict[str, Any]:
    return dict(V3_OVERLAY)


def canonical_kwargs() -> dict[str, Any]:
    return dict(
        overlay=dict(V3_OVERLAY),
        alpha_source="local_retrain",
        features=LOCKED_FEATURES,
        objective="rank:ndcg",
        eval_metric="ndcg@3",
        target=DEFAULT_TARGET,
        n_estimators=50,
        max_depth=2,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        window_kind="3y",
        fold_rule="stock",
        representation="raw",
        label_transform="none",
        use_stock_rolling=True,
    )


# --- 门禁 ---
_register(R6Spec(
    code=BASELINE_V3_ALIAS,
    title_zh="复现 baseline_v3（R4_B05）",
    description_zh="锁分 fs00263 五种子集成 + R4_B05 overlay。必须与 R4_B05 逐笔一致。",
    overlay=dict(V3_OVERLAY),
    stage="parity",
    tags=("parity", "baseline_v3", "locked"),
    alpha_source="locked_fs00263",
    features=LOCKED_FEATURES,
))

# --- Stage A forensics ---
_register(R6Spec(
    code="R6_A00",
    title_zh="锁分 + 当前引擎",
    description_zh="锁分 fs00263 + 当前 local_executor / v3 overlay。应等于 v3。",
    overlay=dict(V3_OVERLAY),
    stage="forensics",
    tags=("A", "A00"),
    alpha_source="locked_fs00263",
    features=LOCKED_FEATURES,
))
_register(R6Spec(
    code="R6_A01",
    title_zh="精确复现 R5_D00",
    description_zh="label_e2_net + subsample 1.0。复用 R5_D00 预测再回测。",
    overlay=dict(V3_OVERLAY),
    stage="forensics",
    tags=("A", "A01"),
    alpha_source="local_retrain",
    prediction_key="A01_r5_d00",
    reuse_round5_key="D00_6factor_n50d2",
    features=LOCKED_FEATURES,
    target="label_e2_net",
    subsample=1.0,
    colsample_bytree=1.0,
))
_register(R6Spec(
    code="R6_A02",
    title_zh="冻结候选宇宙 + 当前重训",
    description_zh="只在冻结 fs00263 评分宇宙上，按 canonical 配方重训。",
    overlay=dict(V3_OVERLAY),
    stage="forensics",
    tags=("A", "A02"),
    alpha_source="local_retrain",
    prediction_key="A02_frozen_universe",
    features=LOCKED_FEATURES,
    **{k: v for k, v in canonical_kwargs().items() if k not in {"overlay", "alpha_source", "features"}},
))
_register(R6Spec(
    code="R6_A03",
    title_zh="冻结 feature 值 + 当前训练",
    description_zh="训练面板 = Stock training_panel.parquet；SHA 对照后按当前配方训。",
    overlay=dict(V3_OVERLAY),
    stage="forensics",
    tags=("A", "A03"),
    alpha_source="local_retrain",
    prediction_key="A03_frozen_panel",
    features=LOCKED_FEATURES,
    **{k: v for k, v in canonical_kwargs().items() if k not in {"overlay", "alpha_source", "features"}},
))
_register(R6Spec(
    code="R6_A04",
    title_zh="历史 config subsample=0.8（FORENSIC_ONLY）",
    description_zh="当前 panel + 历史 0.8/0.8 + 历史窗/种子 + label_exit_rule_net。不得当晋升候选。",
    overlay=dict(V3_OVERLAY),
    stage="forensics",
    tags=("A", "A04", "FORENSIC_ONLY"),
    forensic_only=True,
    diagnostic=True,
    alpha_source="local_retrain",
    prediction_key="A04_hist_0p8",
    features=LOCKED_FEATURES,
    target=DEFAULT_TARGET,
    subsample=0.8,
    colsample_bytree=0.8,
    window_kind="3y",
    fold_rule="stock",
    representation="raw",
    use_stock_rolling=True,
))
_register(R6Spec(
    code="R6_A05",
    title_zh="冻结 booster replay",
    description_zh="无冻结 .json/.ubj，标 SKIPPED_NO_ARTIFACT。",
    overlay=dict(V3_OVERLAY),
    stage="forensics",
    tags=("A", "A05"),
    diagnostic=True,
    alpha_source="skipped",
    prediction_key="A05_booster",
    features=LOCKED_FEATURES,
))
_register(R6Spec(
    code="R6_A06",
    title_zh="单 seed vs 5-seed",
    description_zh="聚合必须读 Stock ensemble_report.json（日内 pct 等权）。",
    overlay=dict(V3_OVERLAY),
    stage="forensics",
    tags=("A", "A06"),
    diagnostic=True,
    alpha_source="locked_fs00263",
    features=LOCKED_FEATURES,
))
_register(R6Spec(
    code="R6_A07",
    title_zh="fold 规则对照",
    description_zh="Stock 年切 vs R5 label_end < predict_year-01-01。比样本数/组数/标签分布。",
    overlay=dict(V3_OVERLAY),
    stage="forensics",
    tags=("A", "A07"),
    diagnostic=True,
    alpha_source="local_retrain",
    prediction_key="A07_r5_fold",
    features=LOCKED_FEATURES,
    target=DEFAULT_TARGET,
    window_kind="3y",
    fold_rule="r5_local",
    use_stock_rolling=False,
    subsample=1.0,
    colsample_bytree=1.0,
))

_register(R6Spec(
    code=CANONICAL_CODE,
    title_zh="canonical 重训（冻结配方）",
    description_zh="label_exit_rule_net + actual_exit_date 年切 + Stock rolling + n50_d2 + 五种子 rank-pct。采样 1.0。",
    stage="canonical",
    tags=("canonical",),
    prediction_key="CANONICAL",
    reuse_round5_key="F00_stock_objective",
    **canonical_kwargs(),
))

# --- B horizon（canonical 上只改窗口）---
_register(R6Spec(
    code="R6_B01",
    title_zh="窗口 2y",
    description_zh="canonical 上只改训练窗为近 2 个完整年。",
    stage="learn_b",
    tags=("B", "horizon"),
    prediction_key="B01_2y",
    **{**canonical_kwargs(), "window_kind": "2y", "use_stock_rolling": False},
))
_register(R6Spec(
    code="R6_B02",
    title_zh="窗口 3y（canonical）",
    description_zh="与 canonical 同窗，独立代号便于对照。",
    stage="learn_b",
    tags=("B", "horizon"),
    prediction_key="CANONICAL",
    reuse_round5_key="F00_stock_objective",
    **canonical_kwargs(),
))
_register(R6Spec(
    code="R6_B03",
    title_zh="窗口 5y",
    description_zh="canonical 上只改训练窗为近 5 个完整年。",
    stage="learn_b",
    tags=("B", "horizon"),
    prediction_key="B03_5y",
    **{**canonical_kwargs(), "window_kind": "5y", "use_stock_rolling": False},
))
_register(R6Spec(
    code="R6_B04",
    title_zh="窗口 expanding",
    description_zh="从首个完整年扩到 train_end。",
    stage="learn_b",
    tags=("B", "horizon"),
    prediction_key="B04_expanding",
    **{**canonical_kwargs(), "window_kind": "expanding", "use_stock_rolling": False},
))
for code, months in (("R6_B05", 6), ("R6_B06", 12), ("R6_B07", 24)):
    _register(R6Spec(
        code=code,
        title_zh=f"近因半寿期 {months}M",
        description_zh=f"3y 窗 + trade_date 行权重 half-life={months}M。XGB ranking 用 instance sample_weight。",
        stage="learn_b",
        tags=("B", "horizon", "halflife"),
        prediction_key=f"B{code[-2:]}_hl{months}",
        **{**canonical_kwargs(), "window_kind": "3y", "use_stock_rolling": False, "half_life_months": months},
    ))

# --- C representation ---
_register(R6Spec(
    code="R6_C01",
    title_zh="表示 raw（canonical）",
    description_zh="与 canonical 同表示。",
    stage="learn_c",
    tags=("C", "repr"),
    prediction_key="CANONICAL",
    reuse_round5_key="F00_stock_objective",
    **canonical_kwargs(),
))
_register(R6Spec(
    code="R6_C02",
    title_zh="表示 same-day percentile",
    description_zh="特征日内 rank-pct；<3 候选日 fallback=0.5，不删日。",
    stage="learn_c",
    tags=("C", "repr"),
    prediction_key="C02_pct",
    **{**canonical_kwargs(), "representation": "pct", "use_stock_rolling": False},
))
_register(R6Spec(
    code="R6_C03",
    title_zh="表示 robust-z",
    description_zh="日内 (x-median)/IQR；<3 候选日 fallback=0，不删日。",
    stage="learn_c",
    tags=("C", "repr"),
    prediction_key="C03_robustz",
    **{**canonical_kwargs(), "representation": "robust_z", "use_stock_rolling": False},
))
_register(R6Spec(
    code="R6_C04",
    title_zh="表示预注册 hybrid",
    description_zh="rank/封板类用 pct，量额类用 robust-z。",
    stage="learn_c",
    tags=("C", "repr"),
    prediction_key="C04_hybrid",
    **{**canonical_kwargs(), "representation": "hybrid", "use_stock_rolling": False},
))

# --- D pairwise ---
_register(R6Spec(
    code="R6_D01",
    title_zh="pairwise × canonical",
    description_zh="复现 R5_F03 方向：rank:pairwise，其余锁 canonical。",
    stage="learn_d",
    tags=("D", "pairwise"),
    prediction_key="D01_pair_canonical",
    reuse_round5_key="F03_pairwise",
    **{**canonical_kwargs(), "objective": "rank:pairwise", "use_stock_rolling": False},
))
_register(R6Spec(
    code="R6_D02",
    title_zh="pairwise × 预留 best window",
    description_zh="shortlist 后由 runner 改 window_kind。",
    stage="learn_d",
    tags=("D", "pairwise"),
    prediction_key="D02_pair_window",
    **{**canonical_kwargs(), "objective": "rank:pairwise", "use_stock_rolling": False},
))
_register(R6Spec(
    code="R6_D03",
    title_zh="pairwise × 预留 best repr",
    description_zh="shortlist 后由 runner 改 representation。",
    stage="learn_d",
    tags=("D", "pairwise"),
    prediction_key="D03_pair_repr",
    **{**canonical_kwargs(), "objective": "rank:pairwise", "use_stock_rolling": False},
))
_register(R6Spec(
    code="R6_D04",
    title_zh="pairwise + best window + best repr",
    description_zh="第一 challenger。shortlist 后填窗/表示。",
    stage="learn_d",
    tags=("D", "pairwise", "challenger"),
    prediction_key="D04_pair_best",
    **{**canonical_kwargs(), "objective": "rank:pairwise", "use_stock_rolling": False},
))

# --- E labels ---
_register(R6Spec(
    code="R6_E01",
    title_zh="标签 current（canonical）",
    description_zh="label_exit_rule_net。",
    stage="learn_e",
    tags=("E", "label"),
    prediction_key="CANONICAL",
    reuse_round5_key="F00_stock_objective",
    **canonical_kwargs(),
))
_register(R6Spec(
    code="R6_E02",
    title_zh="标签日内 ordinal 0–4",
    description_zh="每日五档等频。标签硬规则 actual_exit_date < prediction_date。",
    stage="learn_e",
    tags=("E", "label"),
    prediction_key="E02_ordinal04",
    **{**canonical_kwargs(), "label_transform": "ordinal04", "use_stock_rolling": False},
))
_register(R6Spec(
    code="R6_E03",
    title_zh="标签 clip(−10%,+20%) 再 rank",
    description_zh="先 clip 再日内排序。",
    stage="learn_e",
    tags=("E", "label"),
    prediction_key="E03_clip_rank",
    **{**canonical_kwargs(), "label_transform": "clip_rank", "use_stock_rolling": False},
))
_register(R6Spec(
    code="R6_E04",
    title_zh="标签 Top1 binary（诊断）",
    description_zh="当日收益最高=1。DIAGNOSTIC，不得晋升。",
    stage="learn_e",
    tags=("E", "label"),
    diagnostic=True,
    prediction_key="E04_top1_binary",
    **{
        **canonical_kwargs(),
        "label_transform": "top1_binary",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_stock_rolling": False,
    },
))


def dump_registry() -> str:
    return json.dumps({k: v.to_config() for k, v in SPECS.items()}, ensure_ascii=False, indent=2, default=str)
