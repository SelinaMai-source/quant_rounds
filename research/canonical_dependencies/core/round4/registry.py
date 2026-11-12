"""Round 4 预注册：默认对照 baseline_v2 = R3_03。不自动改锁。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..paths import ROUND3_ROOT, round4_dir

BASELINE_V2 = "R3_03"
BASELINE_V2_ALIAS = "R4_00_baseline_v2"
PARENT_DEFAULT = "baseline_v2"

V2_OVERLAY: dict[str, Any] = {
    "kind": "stack",
    "market": {"kind": "breadth_tier", "t1": 0.45, "t2": 0.50, "t3": 0.60},
    "confidence": {"kind": "margin", "low_mult": 0.5, "cutoff": 0.30},
}

B01A: dict[str, Any] = {"kind": "breadth_tier", "t1": 0.45, "t2": 0.50, "t3": 0.60}
C03_HALF: dict[str, Any] = {"kind": "margin", "low_mult": 0.5, "cutoff": 0.30}

RANDOM_SEEDS: tuple[int, ...] = (
    17, 42, 73, 101, 202, 7, 13, 23, 29, 31,
    37, 53, 67, 89, 97, 109, 127, 149, 163, 181,
)

LOW_ACTIVITY_TRADES = 526  # 876 * 0.60
V2_TRADES = 876
V2_CAGR = 0.2344
V2_SHARPE = 1.394
V2_MDD = -0.1523
V2_WORST12M = -0.0759


@dataclass
class R4Spec:
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
    replace_module: str | None = None
    pit_ok: bool = True
    diagnostic: bool = False

    def to_config(self) -> dict[str, Any]:
        return asdict(self)


SPECS: dict[str, R4Spec] = {}


def _register(spec: R4Spec) -> R4Spec:
    SPECS[spec.code] = spec
    return spec


def get_spec(code: str) -> R4Spec:
    key = str(code or "").strip()
    if key not in SPECS:
        raise KeyError(f"未知 R4 代号 {key!r}")
    return SPECS[key]


def list_codes() -> list[str]:
    return list(SPECS)


def clone_spec(spec: R4Spec, **kwargs) -> R4Spec:
    data = asdict(spec)
    data.update(kwargs)
    data["skip_years"] = tuple(data.get("skip_years") or ())
    data["tags"] = tuple(data.get("tags") or ())
    if data.get("seeds") is not None:
        data["seeds"] = tuple(data["seeds"])
    if data.get("rank_name_budget_nav_fractions") is not None:
        data["rank_name_budget_nav_fractions"] = tuple(data["rank_name_budget_nav_fractions"])
    return R4Spec(**data)


def spec_from_overlay(
    code: str,
    title_zh: str,
    overlay: dict[str, Any],
    *,
    description_zh: str = "",
    replace_module: str | None = None,
    top_k: int = 1,
    rank_name_budget_nav_fractions: tuple[float, ...] | None = None,
    random_seed: int | None = None,
    skip_years: tuple[int, ...] = (),
    seeds: tuple[int, ...] | None = None,
    seed_leave_out: int | None = None,
    cost_overrides: dict[str, Any] | None = None,
    name_budget_nav_fraction: float = 0.3,
    market_up_ratio_min: float = 0.45,
    stage: str = "single",
    tags: tuple[str, ...] = (),
    diagnostic: bool = False,
    pit_ok: bool = True,
    parent: str = PARENT_DEFAULT,
) -> R4Spec:
    return R4Spec(
        code=code,
        title_zh=title_zh,
        description_zh=description_zh or title_zh,
        overlay=overlay,
        parent=parent,
        name_budget_nav_fraction=name_budget_nav_fraction,
        market_up_ratio_min=market_up_ratio_min,
        top_k=top_k,
        rank_name_budget_nav_fractions=rank_name_budget_nav_fractions,
        random_seed=random_seed,
        skip_years=skip_years,
        seeds=seeds,
        seed_leave_out=seed_leave_out,
        cost_overrides=cost_overrides or {},
        stage=stage,
        tags=tags,
        replace_module=replace_module,
        pit_ok=pit_ok,
        diagnostic=diagnostic,
    )


def stack(market: dict[str, Any] | None, confidence: dict[str, Any] | None,
          cvar: dict[str, Any] | None = None, cvar_mode: str = "product") -> dict[str, Any]:
    return {
        "kind": "stack",
        "market": market or {"kind": "identity"},
        "confidence": confidence or {"kind": "identity"},
        "cvar": cvar,
        "cvar_mode": cvar_mode,
    }


def result_dir_for(code: str) -> Path:
    return round4_dir(code)


def r3_03_dir() -> Path:
    return ROUND3_ROOT / BASELINE_V2


# --- 门禁 / 归因 ---
_register(R4Spec(
    code=BASELINE_V2_ALIAS,
    title_zh="复现 baseline_v2（R3_03）",
    description_zh="上涨家数 45/50/60 × 分差偏低半仓。必须与 R3_03 逐笔一致。",
    overlay=dict(V2_OVERLAY),
    stage="parity",
    tags=("parity", "baseline_v2"),
    replace_module=None,
))

for seed in RANDOM_SEEDS:
    _register(R4Spec(
        code=f"R4_RAND_{seed}",
        title_zh=f"随机 Top1 seed={seed}",
        description_zh="同一套入场/退出/费用/缺口/双袖套与 v2 乘子，不用模型 score 选股。",
        overlay=dict(V2_OVERLAY),
        random_seed=seed,
        stage="attr",
        tags=("attr", "random_top1"),
        diagnostic=True,
    ))

_register(R4Spec(
    code="R4_TOP2_diag",
    title_zh="诊断 Top2 等分 15/15",
    description_zh="同契约 v2 乘子；总预算仍 30%。只作诊断 / F Gate。",
    overlay=dict(V2_OVERLAY),
    top_k=2,
    rank_name_budget_nav_fractions=(0.15, 0.15),
    stage="attr",
    tags=("attr", "topk", "f_diag"),
    diagnostic=True,
))
_register(R4Spec(
    code="R4_TOP3_diag",
    title_zh="诊断 Top3 等分 10/10/10",
    description_zh="同契约 v2 乘子；总预算仍 30%。只作诊断。",
    overlay=dict(V2_OVERLAY),
    top_k=3,
    rank_name_budget_nav_fractions=(0.10, 0.10, 0.10),
    stage="attr",
    tags=("attr", "topk"),
    diagnostic=True,
))

# --- B 生态：换 B，C03 保留 ---
_register(R4Spec(
    code="R4_B01",
    title_zh="换 B：昨日首板今日连板率 250 日分位底 20%×0.5",
    description_zh="换掉 B01a，C03 半仓保留。滚动分位，禁止全样本。",
    overlay=stack({"kind": "eco_relimit_p250", "low": 0.20, "low_mult": 0.5}, C03_HALF),
    stage="ecology",
    tags=("single", "B"),
    replace_module="B",
))
_register(R4Spec(
    code="R4_B02",
    title_zh="换 B：20 日 EWMA(开收中位数) 低分位降仓/不买",
    description_zh="p250≤20% 不买，≤40% 半仓。换 B01a，C03 保留。",
    overlay=stack(
        {"kind": "eco_oc_ewma", "span": 20, "skip": 0.20, "half": 0.40},
        C03_HALF,
    ),
    stage="ecology",
    tags=("single", "B"),
    replace_module="B",
))
_register(R4Spec(
    code="R4_B03",
    title_zh="换 B：大亏率高分位降仓/不买",
    description_zh="大亏=当日收益≤−5%。p250≥80% 不买，≥70% 半仓。换 B01a，C03 保留。",
    overlay=stack(
        {"kind": "eco_bigloss", "skip": 0.80, "half": 0.70},
        C03_HALF,
    ),
    stage="ecology",
    tags=("single", "B"),
    replace_module="B",
))
_register(R4Spec(
    code="R4_B04a",
    title_zh="换 B：等权综合五档硬切",
    description_zh="连板/开收/反大亏/供给中性/反 HHI。Very Strong 不加到 >1。",
    overlay=stack({"kind": "eco_five", "hard": True}, C03_HALF),
    stage="ecology",
    tags=("single", "B"),
    replace_module="B",
))
_register(R4Spec(
    code="R4_B04b",
    title_zh="换 B：等权综合五档软切",
    description_zh="同五因子，档间线性。Very Strong 不加到 >1。",
    overlay=stack({"kind": "eco_five", "hard": False}, C03_HALF),
    stage="ecology",
    tags=("single", "B"),
    replace_module="B",
))
_register(R4Spec(
    code="R4_B05",
    title_zh="B 层 min(B01a, 连板率半仓)，再 × C03",
    description_zh="与 B01a 只做 min，禁止再乘一层生态。C03 保留。",
    overlay=stack(
        {"kind": "combine", "how": "min", "parts": [B01A, {"kind": "eco_relimit_p250", "low": 0.20, "low_mult": 0.5}]},
        C03_HALF,
    ),
    stage="ecology",
    tags=("single", "B"),
    replace_module="B",
))
_register(R4Spec(
    code="R4_B06",
    title_zh="B 层 mean(B01a, 连板率半仓)，再 × C03",
    description_zh="与 B01a 只做 mean，禁止再乘一层生态。C03 保留。",
    overlay=stack(
        {"kind": "combine", "how": "mean", "parts": [B01A, {"kind": "eco_relimit_p250", "low": 0.20, "low_mult": 0.5}]},
        C03_HALF,
    ),
    stage="ecology",
    tags=("single", "B"),
    replace_module="B",
))

# --- C 连续分差：保留 B01a，只改 C03 ---
_register(R4Spec(
    code="R4_C01",
    title_zh="换 C：分差五档单调 0.40–1.00",
    description_zh="保留 B01a。分差 250 日分位五档 0.40/0.55/0.70/0.85/1.00。",
    overlay=stack(B01A, {"kind": "margin_tiers", "mults": (0.40, 0.55, 0.70, 0.85, 1.00)}),
    stage="margin",
    tags=("single", "C"),
    replace_module="C",
))
_register(R4Spec(
    code="R4_C02",
    title_zh="换 C：线性 0.4+0.6m",
    description_zh="保留 B01a。乘子=0.4+0.6×分差250日分位，clip≤1。",
    overlay=stack(B01A, {"kind": "margin_linear", "lo": 0.4, "hi": 1.0}),
    stage="margin",
    tags=("single", "C"),
    replace_module="C",
))
_register(R4Spec(
    code="R4_C03",
    title_zh="换 C：线性 0.5+0.5m",
    description_zh="保留 B01a。乘子=0.5+0.5×分差250日分位，clip≤1。",
    overlay=stack(B01A, {"kind": "margin_linear", "lo": 0.5, "hi": 1.0}),
    stage="margin",
    tags=("single", "C"),
    replace_module="C",
))
_register(R4Spec(
    code="R4_C04",
    title_zh="换 C：年度 walk-forward 保序回归 P(赢|分差)",
    description_zh="标签须 actual_exit_date < 预测日。保留 B01a。",
    overlay=stack(B01A, {"kind": "margin_isotonic", "target": "win"}),
    stage="margin",
    tags=("single", "C"),
    replace_module="C",
))
_register(R4Spec(
    code="R4_C05",
    title_zh="换 C：收益校准；不稳就弃",
    description_zh="保序 E[ret|分差]。不单调或样本不足则 ABANDONED。保留 B01a。",
    overlay=stack(B01A, {"kind": "margin_isotonic", "target": "ret"}),
    stage="margin",
    tags=("single", "C"),
    replace_module="C",
))

# --- D 二维 ---
_register(R4Spec(
    code="R4_D01",
    title_zh="换 C：二维预注册分档（低+低最低 0.25）",
    description_zh="人数≤3/4/5 × 分差低中高。低+低=0.25，不 skip。保留 B01a。",
    overlay=stack(B01A, {"kind": "twodim_prereg"}),
    stage="twodim",
    tags=("single", "D"),
    replace_module="C",
))
_register(R4Spec(
    code="R4_D02",
    title_zh="换 C：二维格子历史校准（n≥50，只用已实现）",
    description_zh="exit<signal。满 50 笔才调仓，最低 0.25，不 skip。保留 B01a。",
    overlay=stack(B01A, {"kind": "twodim_hist"}),
    stage="twodim",
    tags=("single", "D"),
    replace_module="C",
))

# --- G CVaR（在 v2 底上乘，不 skip）---
_register(R4Spec(
    code="R4_G01",
    title_zh="v2 底 × 格子历史 CVaR10 相对三档",
    description_zh="n≥50、exit<signal。相对分位高则降仓，不 skip。",
    overlay=stack(B01A, C03_HALF, {"kind": "cvar_rel"}, cvar_mode="product"),
    stage="cvar",
    tags=("single", "G"),
    replace_module="G",
))
_register(R4Spec(
    code="R4_G02",
    title_zh="v2 底 × 格子历史 CVaR10 绝对 −3/−5/−8",
    description_zh="CVaR≤−8%×0.5；≤−5%×0.67；≤−3%×0.85。不 skip。",
    overlay=stack(B01A, C03_HALF, {"kind": "cvar_abs"}, cvar_mode="product"),
    stage="cvar",
    tags=("single", "G"),
    replace_module="G",
))

# F 实盘组不预注册；过 Gate 才动态加入。
# E 过单调诊断才动态加入。


def dump_registry() -> str:
    return json.dumps({k: v.to_config() for k, v in SPECS.items()}, ensure_ascii=False, indent=2, default=str)
