"""Round 2 预注册粗档。组是代号不是新 notebook。禁止全笛卡尔积。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..paths import round2_dir


@dataclass
class R2Spec:
    code: str
    module: str
    title_zh: str
    description_zh: str
    overlay: dict[str, Any] = field(default_factory=dict)
    exit_mode: str = "close_from_d2"
    name_budget_nav_fraction: float = 0.3
    market_up_ratio_min: float = 0.45
    read_existing_ledger: bool = False
    diagnostic_only: bool = False
    skip: bool = False
    skip_reason: str = ""
    needs_overlay_loop: bool = False
    neighbor_codes: tuple[str, ...] = ()


def _spec(**kwargs) -> R2Spec:
    return R2Spec(**kwargs)


EXPERIMENTS: dict[str, R2Spec] = {}


def _register(spec: R2Spec) -> R2Spec:
    EXPERIMENTS[spec.code] = spec
    return spec


# --- B Market Regime ---
_register(_spec(
    code="R2_B00",
    module="B",
    title_zh="锁定 baseline 别名（不重跑 Stock）",
    description_zh="fs00263 / n50_d2 / 五种子集成 / Top1 / 双门控 / NAV30% / close_from_d2。读锁定账本。",
    read_existing_ledger=True,
))
_register(_spec(
    code="R2_B01a",
    module="B",
    title_zh="分档仓位 45/50/60",
    description_zh="上涨家数比 <45% 不交易；45–50%×0.50；50–60%×0.75；≥60% 满仓。禁止每 1% 搜。",
    overlay={"kind": "breadth_tier", "t1": 0.45, "t2": 0.50, "t3": 0.60},
    neighbor_codes=("R2_B01b", "R2_B01c"),
))
_register(_spec(
    code="R2_B01b",
    module="B",
    title_zh="分档仓位 40/50/60",
    description_zh="门槛下放到 40%。<40% 不交易；40–50%×0.50；50–60%×0.75；≥60% 满仓。",
    overlay={"kind": "breadth_tier", "t1": 0.40, "t2": 0.50, "t3": 0.60},
    market_up_ratio_min=0.40,
    neighbor_codes=("R2_B01a", "R2_B01c"),
))
_register(_spec(
    code="R2_B01c",
    module="B",
    title_zh="分档仓位 45/55/65",
    description_zh="<45% 不交易；45–55%×0.50；55–65%×0.75；≥65% 满仓。",
    overlay={"kind": "breadth_tier", "t1": 0.45, "t2": 0.55, "t3": 0.65},
    neighbor_codes=("R2_B01a", "R2_B01b"),
))
_register(_spec(
    code="R2_B02",
    module="B",
    title_zh="UP_NUM 250d 分位 低/中/高",
    description_zh="涨停家数 250 日分位：低 1/3×0.5，中满仓，高 1/3×0.5（拥挤降仓）。",
    overlay={"kind": "up_num_tertile", "low": 0.5, "mid": 1.0, "high": 0.5},
))
_register(_spec(
    code="R2_B03_p70",
    module="B",
    title_zh="FAIL_RATIO≥p70 → 0",
    description_zh="炸板率 250 日分位 ≥70% 当日不交易。",
    overlay={"kind": "fail_high", "threshold": 0.70, "high_mult": 0.0},
    neighbor_codes=("R2_B03_p80", "R2_B03_p90"),
))
_register(_spec(
    code="R2_B03_p80",
    module="B",
    title_zh="FAIL_RATIO≥p80 → 0",
    description_zh="炸板率 250 日分位 ≥80% 当日不交易。",
    overlay={"kind": "fail_high", "threshold": 0.80, "high_mult": 0.0},
    neighbor_codes=("R2_B03_p70", "R2_B03_p90"),
))
_register(_spec(
    code="R2_B03_p90",
    module="B",
    title_zh="FAIL_RATIO≥p90 → 0",
    description_zh="炸板率 250 日分位 ≥90% 当日不交易。",
    overlay={"kind": "fail_high", "threshold": 0.90, "high_mult": 0.0},
    neighbor_codes=("R2_B03_p70", "R2_B03_p80"),
))
_register(_spec(
    code="R2_B04",
    module="B",
    title_zh="SUC/CON 弱环境降仓",
    description_zh="封成率与晋级率 250 日分位均 ≤30% →×0.5；均 ≤10% → 不交易。",
    overlay={"kind": "suc_con_weak", "weak": 0.30, "very_weak": 0.10},
))
_register(_spec(
    code="R2_B05a",
    module="B",
    title_zh="跌停家数高 → 0",
    description_zh="收盘跌停家数 250 日分位 ≥80% 不交易。",
    overlay={"kind": "down_high", "threshold": 0.80, "high_mult": 0.0},
    neighbor_codes=("R2_B05b",),
))
_register(_spec(
    code="R2_B05b",
    module="B",
    title_zh="跌停家数高 → 0.5",
    description_zh="收盘跌停家数 250 日分位 ≥80% → 半仓。",
    overlay={"kind": "down_high", "threshold": 0.80, "high_mult": 0.5},
    neighbor_codes=("R2_B05a",),
))
_register(_spec(
    code="R2_B06_hard",
    module="B",
    title_zh="五档 Regime Score（硬切）",
    description_zh="广度/反炸板/封成晋级/反跌停压/涨停家数适中，五档硬切。Very Good 不加到 >30%。",
    overlay={"kind": "regime_score", "hard": True},
    neighbor_codes=("R2_B06_soft",),
))
_register(_spec(
    code="R2_B06_soft",
    module="B",
    title_zh="五档 Regime Score（软插值）",
    description_zh="同一套五因子得分，档间线性插值。Very Good 乘数封顶 1.0。",
    overlay={"kind": "regime_score", "hard": False},
    neighbor_codes=("R2_B06_hard",),
))

# --- C Confidence ---
_register(_spec(
    code="R2_C01",
    module="C",
    title_zh="五种子 rank_pct 分歧诊断（不锁参数）",
    description_zh="std(5 seed rank_pct) vs 前瞻收益/大亏/连板。只写诊断，不单独回测。",
    diagnostic_only=True,
))
_register(_spec(
    code="R2_C02_graded",
    module="C",
    title_zh="种子一致 5/5→30%、4/5→25%、3/5→15%、≤2→0",
    description_zh="Ensemble Top1 被几只种子同时选为 Top1。5→满仓，4→25/30，3→15/30，≤2 不交易。",
    overlay={"kind": "confidence_agree", "mode": "graded"},
    neighbor_codes=("R2_C02_ge4", "R2_C02_unanimous"),
))
_register(_spec(
    code="R2_C02_ge4",
    module="C",
    title_zh="只做 ≥4/5 一致",
    description_zh="至少 4 只种子与 ensemble Top1 一致才交易，否则跳过。",
    overlay={"kind": "confidence_agree", "mode": "ge4"},
    neighbor_codes=("R2_C02_graded", "R2_C02_unanimous"),
))
_register(_spec(
    code="R2_C02_unanimous",
    module="C",
    title_zh="只做 5/5 一致",
    description_zh="五种子 Top1 完全一致才交易。",
    overlay={"kind": "confidence_agree", "mode": "unanimous"},
    neighbor_codes=("R2_C02_graded", "R2_C02_ge4"),
))
_register(_spec(
    code="R2_C03_half",
    module="C",
    title_zh="Top1−Top2 低 margin 半仓",
    description_zh="截面分差 250 日分位 ≤30% → 半仓。",
    overlay={"kind": "margin", "low_mult": 0.5, "cutoff": 0.30},
    neighbor_codes=("R2_C03_skip",),
))
_register(_spec(
    code="R2_C03_skip",
    module="C",
    title_zh="Top1−Top2 低 margin 不交易",
    description_zh="截面分差 250 日分位 ≤30% → 不交易。",
    overlay={"kind": "margin", "low_mult": 0.0, "cutoff": 0.30},
    neighbor_codes=("R2_C03_half",),
))

# --- D Sizing ---
for code, frac, title in (
    ("R2_D01", 0.10, "NAV 10%"),
    ("R2_D02", 0.15, "NAV 15%"),
    ("R2_D03", 0.20, "NAV 20%"),
    ("R2_D04", 0.25, "NAV 25%"),
    ("R2_D05", 0.30, "NAV 30%（=baseline 别名）"),
):
    _register(_spec(
        code=code,
        module="D",
        title_zh=title,
        description_zh=f"只改 name_budget_nav_fraction={frac:.0%}，入场/退出与锁定相同。看 MDD 是否近似线性。",
        name_budget_nav_fraction=frac,
        read_existing_ledger=(code == "R2_D05"),
        neighbor_codes=tuple(
            c for c, _, _ in (
                ("R2_D01", 0.10, ""),
                ("R2_D02", 0.15, ""),
                ("R2_D03", 0.20, ""),
                ("R2_D04", 0.25, ""),
                ("R2_D05", 0.30, ""),
            ) if c != code
        )[:2],
    ))
_register(_spec(
    code="R2_D06a",
    module="D",
    title_zh="已实现回撤降仓 10%/15%",
    description_zh="开仓时用当时组合 NAV：回撤≥10%→半仓，≥15%→不买。只用当时 NAV，反弹自动回到上一档。",
    overlay={"kind": "nav_dd", "soft": 0.10, "hard": 0.15, "soft_mult": 0.5},
    needs_overlay_loop=True,
    neighbor_codes=("R2_D06b",),
))
_register(_spec(
    code="R2_D06b",
    module="D",
    title_zh="已实现回撤降仓 8%/12%",
    description_zh="开仓时用当时组合 NAV：回撤≥8%→半仓，≥12%→不买。",
    overlay={"kind": "nav_dd", "soft": 0.08, "hard": 0.12, "soft_mult": 0.5},
    needs_overlay_loop=True,
    neighbor_codes=("R2_D06a",),
))

# --- E Exit ---
_register(_spec(
    code="R2_E01",
    module="E",
    title_zh="D+1 收盘未涨停 → D+2 开盘卖",
    description_zh="Stock 已有 next_open_after_non_limit_close，课题内接线，不改 Stock。",
    exit_mode="next_open_after_non_limit_close",
))
_register(_spec(
    code="R2_E02",
    module="E",
    title_zh="持仓中 regime 恶化提前出",
    description_zh="D+2 起若当日 Regime 落入 Very Bad / 炸板高 / 跌停高压，可卖则收盘出；否则仍 close_from_d2。",
    overlay={"kind": "exit_regime"},
    needs_overlay_loop=True,
    neighbor_codes=("R2_E01", "R2_E03"),
))
_register(_spec(
    code="R2_E03",
    module="E",
    title_zh="低 confidence 且 D+1 未连板加速出",
    description_zh="种子一致 ≤2/5 且 D+1 收盘未涨停 → 次日开盘卖；高 consistency 保持 close_from_d2。",
    overlay={"kind": "exit_low_conf"},
    needs_overlay_loop=True,
    neighbor_codes=("R2_E01", "R2_E02"),
))
_register(_spec(
    code="R2_E04_ma5",
    module="E",
    title_zh="close<MA5 提前出（可回退 close_from_d2）",
    description_zh="D+2 起 ma_price<ma5×1.00 则收盘出，否则仍 close_from_d2。R1_03 的 ×1.05 只作对照。",
    overlay={"kind": "exit_ma5", "buffer": 1.00},
    needs_overlay_loop=True,
    neighbor_codes=("R2_E04_ma5x098", "R2_E04_ma5x102"),
))
_register(_spec(
    code="R2_E04_ma5x098",
    module="E",
    title_zh="close<MA5×0.98 提前出",
    description_zh="更宽松止损：ma_price<ma5×0.98 才提前出，否则 close_from_d2。",
    overlay={"kind": "exit_ma5", "buffer": 0.98},
    needs_overlay_loop=True,
    neighbor_codes=("R2_E04_ma5", "R2_E04_ma5x102"),
))
_register(_spec(
    code="R2_E04_ma5x102",
    module="E",
    title_zh="close<MA5×1.02 提前出",
    description_zh="更紧止损：ma_price<ma5×1.02 提前出，否则 close_from_d2。",
    overlay={"kind": "exit_ma5", "buffer": 1.02},
    needs_overlay_loop=True,
    neighbor_codes=("R2_E04_ma5", "R2_E04_ma5x098"),
))

# --- F Risk model ---
_register(_spec(
    code="R2_F01_logit",
    module="F",
    title_zh="独立大亏 Logistic（标签 −5%）",
    description_zh="特征仅 D 收盘可见。逐年扩展窗训练。P(loss) 高 → 不买/半仓。不改 fs00263 score。",
    overlay={"kind": "f01_logit", "label": -0.05},
))
_register(_spec(
    code="R2_F01_xgb",
    module="F",
    title_zh="独立大亏浅 XGB（标签 −5%）",
    description_zh="同一套 PIT 特征与标签，浅 XGB。若本机无 xgboost 则记 skip。",
    overlay={"kind": "f01_xgb", "label": -0.05},
))
_register(_spec(
    code="R2_F02",
    module="F",
    title_zh="F2 跳过（label maturity 无法 PIT）",
    description_zh="候选面板 label_d1_touch_available_at 晚于决策时点，F2 不做。",
    skip=True,
    skip_reason="label_d1_touch_available_at 不是 D 收盘 / D+1 开盘前可见，禁止用未来成熟标签做当日门控。",
))
_register(_spec(
    code="R2_F03",
    module="F",
    title_zh="独立连板确认模型",
    description_zh="预测 D+1 是否连板，不替代 fs00263。低 P(continue) 降仓。",
    overlay={"kind": "f03_continue"},
))

# --- G research overlays ---
_register(_spec(
    code="R2_G01",
    module="G",
    title_zh="主力资金净流入弱 → 降仓",
    description_zh="daily_money_flow 市场合计主力净流入/成交额 250d 分位 ≤30% →×0.5；≤10%→0。",
    overlay={"kind": "money_flow_weak", "weak": 0.30, "very_weak": 0.10},
))
_register(_spec(
    code="R2_G02",
    module="G",
    title_zh="行业首板拥挤 → 降仓",
    description_zh="Top1 的 ind_first_board_count_log 当日截面分位 ≥80% →×0.5；≥90%→0。",
    overlay={"kind": "industry_crowd", "half": 0.80, "skip": 0.90},
))
_register(_spec(
    code="R2_G03",
    module="G",
    title_zh="跌停压力高 → 降仓",
    description_zh="market_down_pressure 250d 分位 ≥80% →×0.5；≥90%→0。",
    overlay={"kind": "down_pressure", "half": 0.80, "skip": 0.90},
))


def list_experiment_codes() -> list[str]:
    return list(EXPERIMENTS)


def get_experiment(code: str) -> R2Spec:
    key = str(code or "").strip()
    if key not in EXPERIMENTS:
        available = ", ".join(list_experiment_codes())
        raise KeyError(f"未知实验代号 {key!r}；可用代号：{available}")
    return EXPERIMENTS[key]


def result_dir_for(code: str) -> Any:
    return round2_dir(get_experiment(code).code)


def runnable_codes(*, module: str | None = None) -> list[str]:
    out = []
    for spec in EXPERIMENTS.values():
        if spec.diagnostic_only or spec.skip:
            continue
        if module and spec.module != module:
            continue
        out.append(spec.code)
    return out


def register_combo(
    code: str,
    *,
    module: str,
    title_zh: str,
    description_zh: str,
    overlay: dict[str, Any],
    exit_mode: str = "close_from_d2",
    name_budget_nav_fraction: float = 0.3,
    market_up_ratio_min: float = 0.45,
    needs_overlay_loop: bool = False,
) -> R2Spec:
    spec = R2Spec(
        code=code,
        module=module,
        title_zh=title_zh,
        description_zh=description_zh,
        overlay=overlay,
        exit_mode=exit_mode,
        name_budget_nav_fraction=name_budget_nav_fraction,
        market_up_ratio_min=market_up_ratio_min,
        needs_overlay_loop=needs_overlay_loop,
    )
    return _register(spec)


def registry_table() -> list[dict[str, Any]]:
    rows = []
    for spec in EXPERIMENTS.values():
        rows.append({
            "code": spec.code,
            "module": spec.module,
            "title_zh": spec.title_zh,
            "exit_mode": spec.exit_mode,
            "nav_frac": spec.name_budget_nav_fraction,
            "read_existing_ledger": spec.read_existing_ledger,
            "diagnostic_only": spec.diagnostic_only,
            "skip": spec.skip,
            "result_dir": str(round2_dir(spec.code)),
            "description_zh": spec.description_zh,
        })
    return rows
