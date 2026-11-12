"""Round10 预注册。A–K 在 Phase 1 计划中已写死，不是看到结果后才加的 POST_HOC。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .contract import (
    A01_AGREE_NAV,
    A02_SEP_NAV,
    A03_CELL,
    A04_CELL,
    ADVERSE_BPS,
    B03_OVERLAY,
    PARENT_CODE,
    PERTURB_SEEDS,
)


@dataclass
class R10Spec:
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
    lock_top1: bool = True
    lock_top2: bool = True
    lock_score: bool = True
    initial_cash: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SPECS: dict[str, R10Spec] = {}


def _add(spec: R10Spec) -> R10Spec:
    SPECS[spec.code] = spec
    return spec


def get_spec(code: str) -> R10Spec:
    if code not in SPECS:
        raise KeyError(code)
    return SPECS[code]


def specs_for_stage(stage: str) -> list[R10Spec]:
    return [s for s in SPECS.values() if s.stage == stage]


def add_post_hoc(spec: R10Spec) -> R10Spec:
    spec.post_hoc = True
    return _add(spec)


_add(R10Spec(
    code="R10_00_baseline_v4",
    title_zh="复现 baseline_v4（R9_B03）",
    stage="parity",
    overlay_spec=dict(B03_OVERLAY),
    parent=PARENT_CODE,
))

_add(R10Spec("R10_A01", "归因：只用 agreement", "attribution", {
    "kind": "agree_only", "agree_nav": dict(A01_AGREE_NAV),
}))
_add(R10Spec("R10_A02", "归因：只用 separation", "attribution", {
    "kind": "sep_only", "sep_nav": dict(A02_SEP_NAV),
}))
_add(R10Spec("R10_A03", "归因：0% 角改 5%", "attribution", {
    "kind": "cell_table", "cell": {f"{a}_{s}": v for (a, s), v in A03_CELL.items()},
}))
_add(R10Spec("R10_A04", "归因：10% 高置信改 5%", "attribution", {
    "kind": "cell_table", "cell": {f"{a}_{s}": v for (a, s), v in A04_CELL.items()},
}))

for code, title, reason in (
    ("R10_B01", "行业关系（待 PIT 诊断）", "SKIP_PENDING_PAIR"),
    ("R10_B02", "配对相关（待诊断）", "SKIP_PENDING_PAIR"),
    ("R10_B03", "相对强弱一条规则（待诊断）", "SKIP_PENDING_PAIR"),
):
    _add(R10Spec(code, title, "pair", {"kind": "daily_file", "alloc_code": code}, skip=True, skip_reason=reason))

_add(R10Spec("R10_C01", "3Y WF shrinkage 均值 k=50", "shrinkage", {"kind": "daily_file", "alloc_code": "R10_C01"}))
_add(R10Spec("R10_C02", "3Y WF conservative posterior", "shrinkage", {"kind": "daily_file", "alloc_code": "R10_C02"}))
_add(R10Spec("R10_C03", "3Y WF 扣 20bp 再 shrinkage", "shrinkage", {"kind": "daily_file", "alloc_code": "R10_C03"}))

for bp in ADVERSE_BPS:
    _add(R10Spec(
        f"R10_D_{bp:03d}bp",
        f"v4 买侧不利 +{bp}bp",
        "execution",
        dict(B03_OVERLAY),
        cost_overrides={"buy_adverse_bps": float(bp)},
        diagnostic=True,
    ))

for scale, tag in ((0.005, "E01"), (0.01, "E02")):
    for seed in PERTURB_SEEDS:
        _add(R10Spec(
            f"R10_{tag}_s{seed}",
            f"日内分位扰动 scale={scale} seed={seed}",
            "perturbation",
            {"kind": "perturb", "scale": scale, "seed": seed},
            diagnostic=True,
            lock_top1=False,
            lock_top2=False,
            lock_score=False,
        ))

_add(R10Spec("R10_F01_sep_low", "分离阈值邻域 15/45", "neighborhood", {
    "kind": "sep_neighbor", "low": 0.15, "mid": 0.45,
}))
_add(R10Spec("R10_F02_sep_high", "分离阈值邻域 25/55", "neighborhood", {
    "kind": "sep_neighbor", "low": 0.25, "mid": 0.55,
}))
_add(R10Spec("R10_F03_alloc_048", "分配邻域 0/4/8", "neighborhood", {
    "kind": "alloc_neighbor", "top2_nav": 0.08, "ladder": (0.00, 0.04, 0.08),
}))
_add(R10Spec("R10_F04_alloc_0612", "分配邻域 0/6/12", "neighborhood", {
    "kind": "alloc_neighbor", "top2_nav": 0.12, "ladder": (0.00, 0.06, 0.12),
}))

_add(R10Spec(
    "R10_G01_1m_p3", "资本 1M + 日成交额 3% 上限", "capacity",
    {"kind": "capacity", "capital": 1_000_000.0, "participation": 0.03},
    diagnostic=True, initial_cash=1_000_000.0,
))
_add(R10Spec(
    "R10_G02_3m_p3", "资本 3M + 日成交额 3% 上限", "capacity",
    {"kind": "capacity", "capital": 3_000_000.0, "participation": 0.03},
    diagnostic=True, initial_cash=3_000_000.0,
))
_add(R10Spec(
    "R10_G03_10m_p3", "资本 10M + 日成交额 3% 上限", "capacity",
    {"kind": "capacity", "capital": 10_000_000.0, "participation": 0.03},
    diagnostic=True, initial_cash=10_000_000.0,
))
_add(R10Spec(
    "R10_G04_1m_p1", "资本 1M + 日成交额 1% 上限", "capacity",
    {"kind": "capacity", "capital": 1_000_000.0, "participation": 0.01},
    diagnostic=True, initial_cash=1_000_000.0,
))
_add(R10Spec(
    "R10_G05_1m_p5", "资本 1M + 日成交额 5% 上限", "capacity",
    {"kind": "capacity", "capital": 1_000_000.0, "participation": 0.05},
    diagnostic=True, initial_cash=1_000_000.0,
))

for i in range(1, 6):
    _add(R10Spec(
        f"R10_K{i:02d}",
        f"最终组合槽 K{i:02d}（pair 只能 veto）",
        "combine",
        {"kind": "daily_file", "alloc_code": f"R10_K{i:02d}"},
        skip=True,
        skip_reason="SKIP_PENDING_COMBINE",
    ))


def register_finalist_stress(codes: list[str]) -> list[str]:
    from copy import deepcopy

    out = []
    for base_code in codes:
        if base_code not in SPECS:
            continue
        base = SPECS[base_code]
        for bp in (20, 30):
            code = f"R10_ADV_{base_code}_{bp}bp"
            if code in SPECS:
                out.append(code)
                continue
            add_post_hoc(R10Spec(
                code,
                f"{base_code} 买侧不利 +{bp}bp",
                "robustness",
                deepcopy(base.overlay_spec),
                cost_overrides={"buy_adverse_bps": float(bp)},
                diagnostic=True,
                predictions=base.predictions,
                lock_top1=base.lock_top1,
                lock_top2=base.lock_top2,
                lock_score=base.lock_score,
                initial_cash=base.initial_cash,
            ))
            out.append(code)
    return out


__all__ = [
    "R10Spec",
    "SPECS",
    "add_post_hoc",
    "get_spec",
    "register_finalist_stress",
    "specs_for_stage",
]
