"""Round 2 报表：Sortino / Calmar / worst year / rolling / 右尾 / Pareto。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..metrics import LOCKED_BASELINE, attach_nav_metrics, clip_nav_for_metrics

P0_MAXDD_FLOOR = -0.30
PREFER_WORST_YEAR = -0.10
PREFER_WORST_12M = -0.15
NEAR_ZERO_TRADE_EPS = 1e-6
WINDOWS = ("2018-2020", "2021-2023", "2024-2026")


def _daily_returns(nav: pd.DataFrame) -> pd.Series:
    ordered = nav.sort_values("trade_date").copy()
    values = ordered["total_asset"].astype(float)
    ret = values.pct_change()
    ret.index = pd.to_datetime(ordered["trade_date"])
    return ret.dropna()


def sortino_zero_rate(daily: pd.Series) -> float | None:
    if len(daily) < 2:
        return None
    downside = daily.clip(upper=0.0)
    denom = float(np.sqrt((downside ** 2).mean()))
    if denom <= 0 or not np.isfinite(denom):
        return None
    return float(daily.mean() / denom * math.sqrt(252))


def worst_calendar_year(nav: pd.DataFrame, *, initial_cash: float = 300_000.0) -> dict[str, Any]:
    ordered = nav.sort_values("trade_date").copy()
    if ordered.empty:
        return {"worst_year": None, "worst_year_return": None, "yearly_returns": {}}
    ordered["year"] = pd.to_datetime(ordered["trade_date"]).dt.year
    year_end = ordered.groupby("year", sort=True).tail(1)
    yearly: dict[str, float] = {}
    prior = float(initial_cash)
    for row in year_end.itertuples(index=False):
        yearly[str(int(row.year))] = float(row.total_asset) / prior - 1.0
        prior = float(row.total_asset)
    if not yearly:
        return {"worst_year": None, "worst_year_return": None, "yearly_returns": {}}
    worst_key = min(yearly, key=yearly.get)
    return {
        "worst_year": worst_key,
        "worst_year_return": yearly[worst_key],
        "yearly_returns": yearly,
    }


def worst_rolling(daily: pd.Series, windows: tuple[int, ...] = (63, 126, 252)) -> dict[str, Any]:
    out: dict[str, Any] = {}
    wealth = (1.0 + daily).cumprod()
    for days in windows:
        if len(wealth) < days:
            out[f"worst_rolling_{days}d"] = None
            continue
        rolled = wealth / wealth.shift(days) - 1.0
        out[f"worst_rolling_{days}d"] = float(rolled.min())
    out["worst_rolling_3m"] = out.get("worst_rolling_63d")
    out["worst_rolling_6m"] = out.get("worst_rolling_126d")
    out["worst_rolling_12m"] = out.get("worst_rolling_252d")
    return out


def right_tail_contribution(trades: pd.DataFrame) -> dict[str, Any]:
    if trades is None or trades.empty or "ret" not in trades:
        return {
            "n_trades": 0,
            "sum_ret": None,
            "top1pct_share": None,
            "top5pct_share": None,
            "top10pct_share": None,
            "without_top1pct_sum": None,
            "without_top5pct_sum": None,
            "without_top10pct_sum": None,
        }
    ret = pd.to_numeric(trades["ret"], errors="coerce").dropna()
    n = int(len(ret))
    total = float(ret.sum())
    ordered = ret.sort_values(ascending=False).reset_index(drop=True)

    def _share(frac: float) -> tuple[float | None, float | None]:
        k = max(1, int(math.ceil(n * frac)))
        head = float(ordered.iloc[:k].sum())
        if total == 0:
            return None, float(total - head)
        return head / total, float(total - head)

    s1, w1 = _share(0.01)
    s5, w5 = _share(0.05)
    s10, w10 = _share(0.10)
    return {
        "n_trades": n,
        "sum_ret": total,
        "top1pct_share": s1,
        "top5pct_share": s5,
        "top10pct_share": s10,
        "without_top1pct_sum": w1,
        "without_top5pct_sum": w5,
        "without_top10pct_sum": w10,
    }


def window_slice_metrics(nav: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    ordered = nav.sort_values("trade_date").copy()
    ordered["trade_date"] = pd.to_datetime(ordered["trade_date"])
    sl = ordered.loc[
        ordered["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))
    ]
    if sl.empty:
        return {"start": start, "end": end, "empty": True}
    first = float(sl["total_asset"].iloc[0])
    # use previous day's asset as start if available
    prior = ordered.loc[ordered["trade_date"] < pd.Timestamp(start)]
    start_asset = float(prior["total_asset"].iloc[-1]) if not prior.empty else first
    last = float(sl["total_asset"].iloc[-1])
    values = sl["total_asset"].astype(float)
    dd = float((values / values.cummax() - 1.0).min()) if len(values) else None
    years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 1 / 365.25)
    cagr = (last / start_asset) ** (1 / years) - 1 if start_asset > 0 else None
    return {
        "start": start,
        "end": end,
        "empty": False,
        "total_return": last / start_asset - 1.0 if start_asset else None,
        "cagr": cagr,
        "max_drawdown": dd,
    }


def attach_round2_metrics(
    nav: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: dict[str, Any],
    *,
    initial_cash: float = 300_000.0,
) -> dict[str, Any]:
    nav = clip_nav_for_metrics(nav) if nav is not None and not nav.empty else nav
    attach_nav_metrics(nav, metrics, initial_cash=initial_cash)
    daily = _daily_returns(nav) if nav is not None and not nav.empty else pd.Series(dtype=float)
    yearly = worst_calendar_year(nav, initial_cash=initial_cash)
    rolling = worst_rolling(daily) if len(daily) else {}
    cagr = metrics.get("annualized_return")
    max_dd = metrics.get("max_drawdown")
    calmar = None
    if cagr is not None and max_dd not in (None, 0) and float(max_dd) < 0:
        calmar = float(cagr) / abs(float(max_dd))
    metrics.update({
        "sortino_zero_rate": sortino_zero_rate(daily) if len(daily) else None,
        "calmar": calmar,
        "worst_year": yearly["worst_year"],
        "worst_year_return": yearly["worst_year_return"],
        "yearly_returns": yearly["yearly_returns"],
        **rolling,
        "right_tail": right_tail_contribution(trades),
        "windows": {
            "2018-2020": window_slice_metrics(nav, "2018-01-01", "2020-12-31"),
            "2021-2023": window_slice_metrics(nav, "2021-01-01", "2023-12-31"),
            "2024-2026": window_slice_metrics(nav, "2024-01-01", "2026-07-24"),
        },
        "p0_maxdd_ok": max_dd is not None and float(max_dd) >= P0_MAXDD_FLOOR,
        "prefer_worst_year_ok": (
            yearly["worst_year_return"] is not None
            and float(yearly["worst_year_return"]) >= PREFER_WORST_YEAR
        ),
        "prefer_worst_12m_ok": (
            rolling.get("worst_rolling_12m") is not None
            and float(rolling["worst_rolling_12m"]) >= PREFER_WORST_12M
        ),
        "near_zero_trade_curve": bool(
            trades is None or trades.empty or len(trades) < 20
        ),
        "locked_baseline_ref": {
            "code": LOCKED_BASELINE["code"],
            "sharpe": LOCKED_BASELINE["sharpe"],
            "cagr": LOCKED_BASELINE["annualized_return"],
            "max_drawdown": LOCKED_BASELINE["max_drawdown"],
        },
    })
    return metrics


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Maximize CAGR / Sharpe / Calmar / MaxDD / Worst Year. No sort-by-Sharpe."""
    usable = [r for r in rows if r.get("cagr") is not None and r.get("sharpe") is not None]
    keys = ("cagr", "sharpe", "calmar", "max_drawdown", "worst_year_return")

    def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
        better_or_equal = True
        strictly_better = False
        for key in keys:
            av, bv = a.get(key), b.get(key)
            if av is None or bv is None:
                continue
            if float(av) < float(bv):
                better_or_equal = False
                break
            if float(av) > float(bv):
                strictly_better = True
        return better_or_equal and strictly_better

    front = []
    for row in usable:
        if any(dominates(other, row) for other in usable if other is not row):
            continue
        front.append(row)
    return front


def neighbor_fragility(center: dict[str, Any], neighbors: list[dict[str, Any]]) -> str:
    """If all neighbor files collapse vs center, mark fragile."""
    if not neighbors:
        return "unknown"
    collapsed = 0
    for nb in neighbors:
        cagr_ok = nb.get("cagr") is not None and float(nb["cagr"]) > 0
        dd_ok = nb.get("max_drawdown") is not None and float(nb["max_drawdown"]) >= P0_MAXDD_FLOOR
        if not (cagr_ok and dd_ok):
            collapsed += 1
    if collapsed == len(neighbors):
        return "fragile"
    if collapsed:
        return "mixed"
    return "stable"
