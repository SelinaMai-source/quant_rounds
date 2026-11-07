"""NAV 后处理：年化 / 波动 / sharpe_zero_rate，与因子搜索脚本同公式。

年化：`(最终资产 / 初始资金) ** (1 / years) - 1`，
`years = (最后交易日 - 首个交易日).days / 365.25`。

Sharpe：日收益均值 / 日收益标准差(ddof=1) × √252，无风险利率按 0。
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd

from .paths import PREDICTION_START

# 与锁定账本同一评价窗：丢掉 2017 MA5 预热日。years / 日收益都从该日起算。
METRIC_VERSION = "r3_locked_window_v1"
EVAL_WINDOW_START = PREDICTION_START
SHARPE_SQRT_DAYS = 252
RISK_FREE_RATE = 0.0
CAGR_ABS_TOL = 1e-4
SHARPE_ABS_TOL = 1e-3

# 追踪表 / notebook 锁定展示（文档 §12.3 / Round1 计划，不重跑）。
LOCKED_BASELINE: dict[str, Any] = {
    "code": "R1_00_baseline",
    "trial_id": "fs00263",
    "capacity": "n50_d2",
    "ensemble": "ensemble_rank_pct_5seed",
    "variant": "top1_both_gates_nav30pct_v1",
    "exit_mode": "close_from_d2",
    "sharpe": 1.174,
    "annualized_return": 0.2293,
    "total_return": 4.8477,
    "max_drawdown": -0.2062,
    "completed_trades": 874,
    "trade_win_rate": 0.4783,
    "average_holding_sessions": 2.10,
    "yearly_returns_pct": {
        "2018": -2.89,
        "2019": 28.21,
        "2020": 14.20,
        "2021": 7.85,
        "2022": 19.22,
        "2023": 29.92,
        "2024": 58.77,
        "2025": 27.92,
        "2026": 21.20,
    },
}

def clip_nav_for_metrics(
    nav: pd.DataFrame,
    *,
    start: date | None = None,
) -> pd.DataFrame:
    """Drop warmup rows before the locked evaluation window (default 2018-01-01)."""
    if nav is None or nav.empty:
        return nav
    ordered = nav.sort_values("trade_date").copy()
    ordered["trade_date"] = pd.to_datetime(ordered["trade_date"])
    start_ts = pd.Timestamp(start or EVAL_WINDOW_START)
    clipped = ordered.loc[ordered["trade_date"] >= start_ts]
    return clipped if not clipped.empty else ordered


REPLACE_RULES: dict[str, Any] = {
    "delta_sharpe_min": 0.02,
    "max_drawdown_floor": -0.30,
    "max_drawdown_vs_baseline_slack": 0.05,
}


def attach_nav_metrics(
    nav: pd.DataFrame,
    metrics: dict[str, Any],
    *,
    initial_cash: float = 300_000.0,
) -> dict[str, Any]:
    """Mutate and return ``metrics`` with annualized_return / vol / sharpe_zero_rate."""
    ordered = clip_nav_for_metrics(nav)
    if ordered is None or ordered.empty:
        metrics.update({
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe_zero_rate": None,
            "metric_version": METRIC_VERSION,
        })
        return metrics
    values = ordered["total_asset"].astype(float)
    daily_return = values.pct_change().dropna()
    first = pd.Timestamp(ordered["trade_date"].iloc[0])
    last = pd.Timestamp(ordered["trade_date"].iloc[-1])
    years = max((last - first).days / 365.25, 1 / 365.25)
    annualized_return = (float(values.iloc[-1]) / float(initial_cash)) ** (1 / years) - 1
    if len(daily_return) > 1:
        vol = float(daily_return.std(ddof=1))
        annualized_volatility = vol * math.sqrt(SHARPE_SQRT_DAYS) if vol == vol else None
        sharpe = (
            float(daily_return.mean() / vol * math.sqrt(SHARPE_SQRT_DAYS))
            if vol > 0 else None
        )
    else:
        annualized_volatility = None
        sharpe = None
    metrics.update({
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_zero_rate": sharpe,
        "metric_version": METRIC_VERSION,
        "metric_nav_start": str(first.date()),
        "metric_nav_end": str(last.date()),
        "metric_years": years,
        "metric_n_daily_returns": int(len(daily_return)),
        "risk_free_rate": RISK_FREE_RATE,
    })
    return metrics


def delta_vs_locked_baseline(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """ΔSharpe / Δret versus the locked display baseline (not a re-run)."""
    if not metrics:
        return {
            "delta_sharpe": None,
            "delta_ret": None,
            "delta_ret_pct_points": None,
        }
    sharpe = metrics.get("sharpe_zero_rate")
    total_return = metrics.get("total_return")
    delta_sharpe = (
        None if sharpe is None else float(sharpe) - float(LOCKED_BASELINE["sharpe"])
    )
    delta_ret = (
        None if total_return is None
        else float(total_return) - float(LOCKED_BASELINE["total_return"])
    )
    return {
        "delta_sharpe": delta_sharpe,
        "delta_ret": delta_ret,
        "delta_ret_pct_points": None if delta_ret is None else delta_ret * 100.0,
    }


def replacement_gate(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """是否达到替换门槛。达门槛也不自动改锁定代号。"""
    if not metrics:
        return {
            "meets_gate": False,
            "reason": "未跑",
            "delta_sharpe_ok": False,
            "maxdd_floor_ok": False,
            "maxdd_vs_baseline_ok": False,
        }
    sharpe = metrics.get("sharpe_zero_rate")
    max_dd = metrics.get("max_drawdown")
    delta = None if sharpe is None else float(sharpe) - float(LOCKED_BASELINE["sharpe"])
    slack = float(REPLACE_RULES["max_drawdown_vs_baseline_slack"])
    baseline_dd = float(LOCKED_BASELINE["max_drawdown"])
    delta_ok = delta is not None and delta > float(REPLACE_RULES["delta_sharpe_min"])
    floor_ok = max_dd is not None and float(max_dd) >= float(REPLACE_RULES["max_drawdown_floor"])
    vs_ok = max_dd is not None and float(max_dd) >= baseline_dd - slack
    meets = bool(delta_ok and floor_ok and vs_ok)
    reasons: list[str] = []
    if not delta_ok:
        reasons.append(
            f"ΔSharpe 须 > +{REPLACE_RULES['delta_sharpe_min']:.2f}（相对锁定 {LOCKED_BASELINE['sharpe']:.3f}）"
        )
    if not floor_ok:
        reasons.append("MaxDD 须 ≥ −30%")
    if not vs_ok:
        reasons.append(
            f"MaxDD 不得比 baseline {baseline_dd:.2%} 差超过 {slack:.0%}（约 {baseline_dd - slack:.2%}）"
        )
    return {
        "meets_gate": meets,
        "reason": "达门槛（不自动改锁定代号）" if meets else "；".join(reasons) or "未跑",
        "delta_sharpe_ok": delta_ok,
        "maxdd_floor_ok": floor_ok,
        "maxdd_vs_baseline_ok": vs_ok,
    }
