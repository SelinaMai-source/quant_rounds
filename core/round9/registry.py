"""Round9 预注册实验。freeze 后新增必须标 POST_HOC_EXPLORATORY。本阶段只跑 R9_00。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .contract import E02_OVERLAY, PARENT_CODE


@dataclass
class R9Spec:
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
    predictions: str = "locked"
    bottom: float = 0.20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SPECS: dict[str, R9Spec] = {}


def _add(spec: R9Spec) -> R9Spec:
    SPECS[spec.code] = spec
    return spec


def get_spec(code: str) -> R9Spec:
    if code not in SPECS:
        raise KeyError(code)
    return SPECS[code]


def specs_for_stage(stage: str) -> list[R9Spec]:
    return [s for s in SPECS.values() if s.stage == stage]


_add(R9Spec(
    code="R9_00_baseline_v3",
    title_zh="复现 baseline_v3（R8_E02）",
    stage="parity",
    overlay_spec={"kind": "exact_e02"},
))

_add(R9Spec(
    "R9_A01",
    "诊断：Top2=0（边际效用）",
    "forensics",
    {"kind": "preregistered", "allocator": "top2_zero"},
    diagnostic=True,
))

_add(R9Spec("R9_B01", "预注册 9 格 replacement allocator（0/5/10）", "unified", {"kind": "preregistered", "allocator": "B01_9cell"}))
_add(R9Spec("R9_B02", "C2=0.5A+0.5S replacement", "unified", {"kind": "preregistered", "allocator": "B02_blend"}))
_add(R9Spec("R9_B03", "conservative replacement", "unified", {"kind": "preregistered", "allocator": "B03_conservative"}))
_add(R9Spec("R9_B04", "E02 control（仍是 v3，不是乘子叠加）", "unified", {"kind": "exact_e02"}))

_add(R9Spec("R9_C01", "严格 3Y walk-forward：回归 Top2 净收益", "meta", {"kind": "preregistered", "allocator": "C01_net", "alloc_code": "R9_C01"}))
_add(R9Spec("R9_C02", "严格 3Y walk-forward：胜率", "meta", {"kind": "preregistered", "allocator": "C02_win", "alloc_code": "R9_C02"}))
_add(R9Spec("R9_C03", "严格 3Y walk-forward：三分位", "meta", {"kind": "preregistered", "allocator": "C03_tertile", "alloc_code": "R9_C03"}))
_add(R9Spec("R9_C04", "仅 C01 单调才 isotonic", "meta", {"kind": "preregistered", "allocator": "C04_isotonic", "alloc_code": "R9_C04"}))

_add(R9Spec("R9_D00", "方向诊断 CASE_A–D", "direction", {"kind": "preregistered", "allocator": "D00_diag"}, diagnostic=True, skip=True, skip_reason="DIAGNOSTIC_NO_BT"))
for i in (1, 2, 3):
    _add(R9Spec(
        f"R9_D{i:02d}",
        f"方向规则 D{i:02d}（待 D00，无稳定 OOF 则 SKIP）",
        "direction",
        {"kind": "preregistered", "allocator": f"D{i:02d}"},
        skip=True,
        skip_reason="SKIP_PENDING_D00",
    ))

_add(R9Spec("R9_E01", "已净费用之上 +10bp buffer", "execution", {"kind": "preregistered", "allocator": "E01_buf10", "alloc_code": "R9_E01"}))
_add(R9Spec("R9_E02", "已净费用之上 +20bp buffer", "execution", {"kind": "preregistered", "allocator": "E02_buf20", "alloc_code": "R9_E02"}))
_add(R9Spec("R9_E03", "训练折 gap top20%→20bp 否则 10bp", "execution", {"kind": "preregistered", "allocator": "E03_gap", "alloc_code": "R9_E03"}))

for i, title in enumerate(
    ("封板 residual family", "资金流 residual family", "日内 residual family", "best two family 组合"),
    start=1,
):
    _add(R9Spec(f"R9_F{i:02d}", title, "residual", {"kind": "preregistered", "allocator": f"F{i:02d}", "alloc_code": f"R9_F{i:02d}"}))

_add(R9Spec(
    "R9_DD01",
    "v3 Top5 回撤：只调 Top2 allocator",
    "drawdown",
    {"kind": "preregistered", "allocator": "drawdown_top2", "alloc_code": "R9_DD01"},
))

for i in range(1, 6):
    _add(R9Spec(
        f"R9_K{i:02d}",
        f"最终组合槽 K{i:02d}（最多 5；两模块改预算必须 primary vs veto）",
        "combine",
        {"kind": "preregistered", "allocator": "daily_file", "alloc_code": f"R9_K{i:02d}", "slot": i},
    ))

_add(R9Spec("R9_RB_v3_fee15", "baseline_v3 成本×1.5", "robustness", {"kind": "exact_e02"}, cost_overrides={"fee_mult": 1.5}))
_add(R9Spec("R9_RB_v3_fee20", "baseline_v3 成本×2", "robustness", {"kind": "exact_e02"}, cost_overrides={"fee_mult": 2.0}))
for bp in (10, 20, 30, 50):
    _add(R9Spec(
        f"R9_ADV_v3_{bp}bp",
        f"baseline_v3 买入不利 +{bp}bp",
        "robustness",
        {"kind": "exact_e02"},
        cost_overrides={"buy_adverse_bps": float(bp)},
        diagnostic=True,
    ))


def add_post_hoc(spec: R9Spec) -> R9Spec:
    spec.post_hoc = True
    return _add(spec)


def register_winner_stress(winner_code: str) -> list[str]:
    """freeze 之后对 candidate 的费用/不利成交。POST_HOC，不进 v4。"""
    from copy import deepcopy

    if winner_code not in SPECS:
        return []
    base = SPECS[winner_code]
    codes = []
    for bp in (10, 20, 30, 50):
        code = f"R9_ADV_best_{bp}bp"
        if code in SPECS:
            codes.append(code)
            continue
        add_post_hoc(R9Spec(
            code,
            f"candidate({winner_code}) 买入不利 +{bp}bp",
            "robustness",
            deepcopy(base.overlay_spec),
            cost_overrides={"buy_adverse_bps": float(bp)},
            diagnostic=True,
            predictions=base.predictions,
            bottom=base.bottom,
        ))
        codes.append(code)
    for tag, mult in (("fee15", 1.5), ("fee20", 2.0)):
        code = f"R9_RB_best_{tag}"
        if code in SPECS:
            codes.append(code)
            continue
        add_post_hoc(R9Spec(
            code,
            f"candidate({winner_code}) 成本×{mult}",
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
    "forensics",
    "unified",
    "meta",
    "direction",
    "execution",
    "residual",
    "drawdown",
    "combine",
    "robustness",
)

__all__ = [
    "FORMAL_BACKTEST_STAGES",
    "R9Spec",
    "SPECS",
    "add_post_hoc",
    "get_spec",
    "register_winner_stress",
    "specs_for_stage",
]
