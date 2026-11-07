"""Round8 预注册实验。freeze 后新增必须标 POST_HOC_EXPLORATORY。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .contract import P08_OVERLAY, PARENT_CODE, V2_OVERLAY


@dataclass
class R8Spec:
    code: str
    title_zh: str
    stage: str
    overlay_spec: dict[str, Any] = field(default_factory=dict)
    parent: str = PARENT_CODE
    post_hoc: bool = False
    diagnostic: bool = False
    skip: bool = False
    skip_reason: str = ""
    cost_overrides: dict[str, Any] = field(default_factory=dict)
    skip_years: tuple[int, ...] = ()
    predictions: str = "locked"  # locked | d_close
    bottom: float = 0.20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SPECS: dict[str, R8Spec] = {}


def _add(spec: R8Spec) -> R8Spec:
    SPECS[spec.code] = spec
    return spec


def get_spec(code: str) -> R8Spec:
    if code not in SPECS:
        raise KeyError(code)
    return SPECS[code]


def specs_for_stage(stage: str) -> list[R8Spec]:
    return [s for s in SPECS.values() if s.stage == stage]


_add(R8Spec(
    code="R8_00_baseline_v2",
    title_zh="复现 baseline_v2（R4_E_top1_minus_median）",
    stage="parity",
    overlay_spec={"kind": "exact_v2"},
))

# B smooth confidence
_add(R8Spec("R8_B01", "三档 20/50：0.50/0.75/1.00", "confidence", {"kind": "custom", "conf_kind": "B01"}))
_add(R8Spec("R8_B02", "四档 20/40/70：0.50/0.70/0.85/1.00", "confidence", {"kind": "custom", "conf_kind": "B02"}))
_add(R8Spec("R8_B03", "线性 0.5+0.5p，clip[0.5,1]", "confidence", {"kind": "custom", "conf_kind": "B03"}))
_add(R8Spec("R8_B04", "对照：baseline_v2 硬阈值", "confidence", {"kind": "exact_v2"}))

# C rank-specific
_add(R8Spec("R8_C01", "仅 Top1 吃 top1-median 置信度", "rank-specific", {"kind": "custom", "rank_mode": "top1_only"}))
_add(R8Spec("R8_C02", "仅 Top2 吃 top1-median 置信度", "rank-specific", {"kind": "custom", "rank_mode": "top2_only"}))
_add(R8Spec("R8_C03", "Top1/Top2 各自用自身−median", "rank-specific", {"kind": "custom", "rank_mode": "separate"}))
_add(R8Spec("R8_C04", "Top1 强且 Top2 弱 → 关掉 Top2", "rank-specific", {"kind": "custom", "c04": True}))

# D conditional Top2
_add(R8Spec("R8_D01", "Top2-median 底20% → Top2=0", "top2", {"kind": "custom", "d_rule": "D01"}))
_add(R8Spec("R8_D02", "Top2-median 底20% → Top2=5%", "top2", {"kind": "custom", "d_rule": "D02"}))
_add(R8Spec("R8_D03", "Top2 0/5/10 三档", "top2", {"kind": "custom", "d_rule": "D03"}))
_add(R8Spec("R8_D04", "Top2-median>p50 → Top2=15%，不挤占 Top1", "top2", {"kind": "custom", "d_rule": "D04"}))
_add(R8Spec(
    "R8_D_TOP1_ONLY",
    "诊断：只做 Top1 30%（边际效用）",
    "top2",
    {"kind": "custom", "top_k": 1},
    diagnostic=True,
))

# E disagreement
_add(R8Spec("R8_E01", "双头分歧全日 ×1.00/0.75/0.50", "disagreement", {"kind": "custom", "disagree": "E01"}))
_add(R8Spec("R8_E02", "双头分歧只乘 Top2", "disagreement", {"kind": "custom", "e02": True}))
_add(R8Spec("R8_E03", "仅低分离且高分歧再 ×0.5", "disagreement", {"kind": "custom", "disagree": "E03"}))

# F shape
_add(R8Spec("R8_F01", "(Top1−Median)/IQR 底20%×0.5", "shape", {"kind": "custom", "conf_kind": "F01"}))
_add(R8Spec("R8_F02", "(Top1−Median)/MAD 底20%×0.5", "shape", {"kind": "custom", "conf_kind": "F02"}))
_add(R8Spec("R8_F03", "按候选数分组的 PIT 底20%×0.5", "shape", {"kind": "custom", "conf_kind": "F03"}))

# G ecology
_add(R8Spec("R8_G01", "对照：精确 P08（生态+截面）", "ecology", {"kind": "exact_p08", "overlay": dict(P08_OVERLAY)}))
_add(R8Spec("R8_G02", "仅 Top1-median 底20% 日生效生态", "ecology", {"kind": "custom", "ecology": "low_sep"}))
_add(R8Spec("R8_G03", "生态只作用于 Top2", "ecology", {"kind": "custom", "ecology": "top2_only"}))
_add(R8Spec("R8_G04", "Top1 与 Top2 都底20% 才生效生态", "ecology", {"kind": "custom", "ecology": "both_low"}))

# H left-tail
_add(R8Spec("R8_H01", "低分离 + 高分歧 → 再 ×0.5", "left-tail", {"kind": "custom", "h01": True}))
_add(R8Spec("R8_H02", "Top2 低分离 → 关掉 Top2", "left-tail", {"kind": "custom", "h02": True}))
_add(R8Spec(
    "R8_H03",
    "定向市场规则（默认 SKIP）",
    "left-tail",
    {"kind": "exact_v2"},
    skip=True,
    skip_reason="SKIP_H03_pending_audit",
))

# I execution
_add(R8Spec(
    "R8_I01_D_CLOSE_ONLY",
    "去掉 d1_open_gap 后重训再套 top1-minus-median",
    "execution",
    {"kind": "exact_v2"},
    predictions="d_close",
    diagnostic=True,
))

# threshold neighborhood (robustness, pre-registered)
for cut in (0.15, 0.25):
    tag = f"{int(cut * 100):02d}"
    _add(R8Spec(
        f"R8_NB_cut{tag}",
        f"邻域阈值 {tag}%（稳健性，不钓鱼）",
        "robustness",
        {"kind": "custom", "conf_kind": "v2", "bottom": cut},
        bottom=cut,
    ))

# cost / adverse on v2 (robustness)
_add(R8Spec("R8_RB_v2_fee15", "baseline_v2 成本×1.5", "robustness", {"kind": "exact_v2"}, cost_overrides={"fee_mult": 1.5}))
_add(R8Spec("R8_RB_v2_fee20", "baseline_v2 成本×2", "robustness", {"kind": "exact_v2"}, cost_overrides={"fee_mult": 2.0}))
for bp in (10, 20, 30, 50):
    _add(R8Spec(
        f"R8_ADV_v2_{bp}bp",
        f"baseline_v2 买入不利 +{bp}bp",
        "execution",
        {"kind": "exact_v2"},
        cost_overrides={"buy_adverse_bps": float(bp)},
        diagnostic=True,
    ))
    _add(R8Spec(
        f"R8_ADV_v1_{bp}bp",
        f"baseline_v1 买入不利 +{bp}bp",
        "execution",
        {"kind": "v1_clean"},
        cost_overrides={"buy_adverse_bps": float(bp)},
        diagnostic=True,
    ))

# J combination slots — formulas frozen; winners filled at combine stage
for i, title in enumerate(
    (
        "v2 + 最佳平滑置信度",
        "v2 + 最佳条件 Top2",
        "v2 + 最佳双头分歧",
        "最佳平滑 + 最佳 Top2",
        "最佳平滑 + 最佳分歧",
        "最佳 Top2 + 最佳分歧",
    ),
    start=1,
):
    _add(R8Spec(
        f"R8_J{i:02d}",
        title,
        "combine",
        {"kind": "combine_slot", "slot": i},
    ))


def add_post_hoc(spec: R8Spec) -> R8Spec:
    spec.post_hoc = True
    return _add(spec)


def register_winner_stress(winner_code: str) -> list[str]:
    """freeze 之后对 best R8 的费用/不利成交。POST_HOC，不进 v3。"""
    from copy import deepcopy

    if winner_code not in SPECS:
        return []
    base = SPECS[winner_code]
    codes = []
    for bp in (10, 20, 30, 50):
        code = f"R8_ADV_best_{bp}bp"
        if code in SPECS:
            codes.append(code)
            continue
        add_post_hoc(R8Spec(
            code,
            f"best R8({winner_code}) 买入不利 +{bp}bp",
            "execution",
            deepcopy(base.overlay_spec),
            cost_overrides={"buy_adverse_bps": float(bp)},
            diagnostic=True,
            predictions=base.predictions,
            bottom=base.bottom,
        ))
        codes.append(code)
    for tag, mult in (("fee15", 1.5), ("fee20", 2.0)):
        code = f"R8_RB_best_{tag}"
        if code in SPECS:
            codes.append(code)
            continue
        add_post_hoc(R8Spec(
            code,
            f"best R8({winner_code}) 成本×{mult}",
            "robustness",
            deepcopy(base.overlay_spec),
            cost_overrides={"fee_mult": float(mult)},
            diagnostic=True,
            predictions=base.predictions,
            bottom=base.bottom,
        ))
        codes.append(code)
    return codes


FORMAL_BACKTEST_STAGES = (
    "parity",
    "confidence",
    "rank-specific",
    "top2",
    "disagreement",
    "shape",
    "ecology",
    "left-tail",
    "execution",
    "combine",
    "robustness",
)

__all__ = [
    "FORMAL_BACKTEST_STAGES",
    "R8Spec",
    "SPECS",
    "add_post_hoc",
    "get_spec",
    "register_winner_stress",
    "specs_for_stage",
]
