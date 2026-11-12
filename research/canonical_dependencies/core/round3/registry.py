"""Round 3 预注册：shortlist 从 Round 2 config 原样引用，不手抄数字。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..paths import ROUND2_ROOT, round3_dir


@dataclass
class R3Spec:
    code: str
    title_zh: str
    description_zh: str
    source_round2: str | None = None
    overlay: dict[str, Any] = field(default_factory=dict)
    exit_mode: str = "close_from_d2"
    name_budget_nav_fraction: float = 0.3
    market_up_ratio_min: float = 0.45
    pit_correct_g02: bool = False
    skip_years: tuple[int, ...] = ()
    seed_leave_out: int | None = None
    seeds: tuple[int, ...] | None = None
    cost_overrides: dict[str, Any] = field(default_factory=dict)
    stage: str = "repro"
    tags: tuple[str, ...] = ()
    uses_g02: bool = False
    ref_r2_cagr: float | None = None
    ref_r2_sharpe: float | None = None
    ref_r2_mdd: float | None = None

    def to_config(self) -> dict[str, Any]:
        return asdict(self)


SHORTLIST: dict[str, R3Spec] = {}


def _register(spec: R3Spec) -> R3Spec:
    SHORTLIST[spec.code] = spec
    return spec


def _r2_cfg(code: str) -> dict[str, Any]:
    path = ROUND2_ROOT / "configs" / f"{code}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"code": code}


def source_config_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


_register(R3Spec(
    code="R3_00",
    title_zh="锁定 Alpha，无叠加",
    description_zh="fs00263 / n50_d2 / 五种子集成 / Top1 / 双门控 / NAV30% / close_from_d2。",
    source_round2="R1_00_baseline",
    overlay={"kind": "identity"},
    stage="repro",
    tags=("shortlist", "baseline"),
    ref_r2_cagr=0.2293,
    ref_r2_sharpe=1.174,
    ref_r2_mdd=-0.2062,
))
_register(R3Spec(
    code="R3_01",
    title_zh="5/5 满仓，4/5→25%，3/5→15%，≤2 不买",
    description_zh="从 R2_C02_graded 原样拷贝。",
    source_round2="R2_C02_graded",
    overlay={"kind": "confidence_agree", "mode": "graded"},
    stage="repro",
    tags=("shortlist", "c02"),
    ref_r2_cagr=0.2293,
    ref_r2_sharpe=1.411,
    ref_r2_mdd=-0.1581,
))
_register(R3Spec(
    code="R3_02",
    title_zh="Top1−Top2 分差 250 日分位偏低则半仓",
    description_zh="从 R2_C03_half 原样拷贝。",
    source_round2="R2_C03_half",
    overlay={"kind": "margin", "low_mult": 0.5, "cutoff": 0.30},
    stage="repro",
    tags=("shortlist", "c03"),
    ref_r2_cagr=0.2428,
    ref_r2_sharpe=1.329,
    ref_r2_mdd=-0.1659,
))
_register(R3Spec(
    code="R3_03",
    title_zh="上涨家数分档 × 分差半仓",
    description_zh="从 R2_P04 原样拷贝。",
    source_round2="R2_P04_R2_B01a_R2_C03_half",
    overlay={"kind": "product", "components": ["R2_B01a", "R2_C03_half"]},
    stage="repro",
    tags=("shortlist", "p04"),
    ref_r2_cagr=0.2344,
    ref_r2_sharpe=1.394,
    ref_r2_mdd=-0.1523,
))
_register(R3Spec(
    code="R3_04",
    title_zh="上涨家数分档 × seed 分档 × 行业拥挤",
    description_zh="从 R2_T06 原样拷贝。G02 全样本分位则 INVALID。",
    source_round2="R2_T06_R2_B01a_R2_C02_graded_R2_G02",
    overlay={"kind": "product", "components": ["R2_B01a", "R2_C02_graded", "R2_G02"]},
    stage="repro",
    tags=("shortlist", "t06"),
    uses_g02=True,
    ref_r2_cagr=0.2174,
    ref_r2_sharpe=1.552,
    ref_r2_mdd=-0.1362,
))
_register(R3Spec(
    code="R3_05",
    title_zh="seed 分档 × 行业拥挤（T06 消融）",
    description_zh="从 R2_P13 原样拷贝。G02 全样本分位则 INVALID。",
    source_round2="R2_P13_R2_C02_graded_R2_G02",
    overlay={"kind": "product", "components": ["R2_C02_graded", "R2_G02"]},
    stage="repro",
    tags=("shortlist", "p13"),
    uses_g02=True,
    ref_r2_cagr=0.2258,
    ref_r2_sharpe=1.477,
    ref_r2_mdd=-0.1604,
))
_register(R3Spec(
    code="R3_06",
    title_zh="上涨家数分档 × seed 分档（T06 消融）",
    description_zh="从 R2_P03 原样拷贝。",
    source_round2="R2_P03_R2_B01a_R2_C02_graded",
    overlay={"kind": "product", "components": ["R2_B01a", "R2_C02_graded"]},
    stage="repro",
    tags=("shortlist", "p03"),
    ref_r2_cagr=0.2204,
    ref_r2_sharpe=1.474,
    ref_r2_mdd=-0.1525,
))
_register(R3Spec(
    code="R3_07",
    title_zh="分差半仓 × 行业拥挤",
    description_zh="从 R2_P14 原样拷贝。G02 全样本分位则 INVALID。",
    source_round2="R2_P14_R2_C03_half_R2_G02",
    overlay={"kind": "product", "components": ["R2_C03_half", "R2_G02"]},
    stage="repro",
    tags=("shortlist", "p14"),
    uses_g02=True,
    ref_r2_cagr=0.2316,
    ref_r2_sharpe=1.362,
    ref_r2_mdd=-0.1630,
))


R2_CHILD_OVERLAYS: dict[str, dict[str, Any]] = {
    "R2_B00": {"kind": "identity"},
    "R2_B01a": {"kind": "breadth_tier", "t1": 0.45, "t2": 0.50, "t3": 0.60},
    "R2_B01b": {"kind": "breadth_tier", "t1": 0.40, "t2": 0.50, "t3": 0.60},
    "R2_B01c": {"kind": "breadth_tier", "t1": 0.45, "t2": 0.55, "t3": 0.65},
    "R2_C02_graded": {"kind": "confidence_agree", "mode": "graded"},
    "R2_C02_ge4": {"kind": "confidence_agree", "mode": "ge4"},
    "R2_C02_unanimous": {"kind": "confidence_agree", "mode": "unanimous"},
    "R2_C03_half": {"kind": "margin", "low_mult": 0.5, "cutoff": 0.30},
    "R2_C03_skip": {"kind": "margin", "low_mult": 0.0, "cutoff": 0.30},
    "R2_G02": {"kind": "industry_crowd", "half": 0.80, "skip": 0.90},
}


def get_spec(code: str) -> R3Spec:
    key = str(code or "").strip()
    if key not in SHORTLIST:
        raise KeyError(f"未知 shortlist 代号 {key!r}")
    return SHORTLIST[key]


def list_shortlist_codes() -> list[str]:
    return list(SHORTLIST)


def r2_source_payload(spec: R3Spec) -> dict[str, Any]:
    if spec.source_round2 and spec.source_round2.startswith("R2_"):
        return _r2_cfg(spec.source_round2)
    return {
        "code": spec.source_round2 or spec.code,
        "overlay": spec.overlay,
        "exit_mode": spec.exit_mode,
        "name_budget_nav_fraction": spec.name_budget_nav_fraction,
        "market_up_ratio_min": spec.market_up_ratio_min,
    }


def clone_spec(spec: R3Spec, **kwargs) -> R3Spec:
    data = asdict(spec)
    data.update(kwargs)
    data["skip_years"] = tuple(data.get("skip_years") or ())
    data["tags"] = tuple(data.get("tags") or ())
    if data.get("seeds") is not None:
        data["seeds"] = tuple(data["seeds"])
    return R3Spec(**data)


def spec_from_overlay(
    code: str,
    title_zh: str,
    overlay: dict[str, Any],
    *,
    description_zh: str = "",
    source_round2: str | None = None,
    name_budget_nav_fraction: float = 0.3,
    market_up_ratio_min: float = 0.45,
    pit_correct_g02: bool = False,
    skip_years: tuple[int, ...] = (),
    seed_leave_out: int | None = None,
    seeds: tuple[int, ...] | None = None,
    cost_overrides: dict[str, Any] | None = None,
    stage: str = "rest",
    tags: tuple[str, ...] = (),
    uses_g02: bool | None = None,
) -> R3Spec:
    blob = json.dumps(overlay, sort_keys=True, default=str)
    g02 = uses_g02 if uses_g02 is not None else ("G02" in blob or overlay.get("kind") == "industry_crowd")
    return R3Spec(
        code=code,
        title_zh=title_zh,
        description_zh=description_zh or title_zh,
        source_round2=source_round2,
        overlay=overlay,
        name_budget_nav_fraction=name_budget_nav_fraction,
        market_up_ratio_min=market_up_ratio_min,
        pit_correct_g02=pit_correct_g02,
        skip_years=skip_years,
        seed_leave_out=seed_leave_out,
        seeds=seeds,
        cost_overrides=cost_overrides or {},
        stage=stage,
        tags=tags,
        uses_g02=g02,
    )


def result_dir_for(code: str) -> Path:
    return round3_dir(code)
