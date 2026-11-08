"""路径风险指标。不把交易盈亏加总当成复利净值差。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def nav_frame(nav: pd.DataFrame) -> pd.DataFrame:
    frame = nav.copy()
    if "trade_date" not in frame.columns:
        raise KeyError("nav missing trade_date")
    value_col = "total_asset" if "total_asset" in frame.columns else "nav"
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame.dropna(subset=["trade_date", value_col]).sort_values("trade_date")
    frame = frame.drop_duplicates("trade_date", keep="last")
    frame["nav"] = frame[value_col].astype(float)
    return frame.reset_index(drop=True)


def daily_returns(nav: pd.DataFrame) -> pd.Series:
    frame = nav_frame(nav)
    ret = frame.set_index("trade_date")["nav"].pct_change()
    ret.name = "ret"
    return ret.dropna()


def drawdown_path(nav: pd.DataFrame) -> pd.DataFrame:
    frame = nav_frame(nav)
    peak = frame["nav"].cummax()
    frame["peak_nav"] = peak
    frame["drawdown"] = 1.0 - frame["nav"] / peak
    return frame


def ulcer_index(drawdown: pd.Series) -> float:
    values = pd.to_numeric(drawdown, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values))))


def max_drawdown(drawdown: pd.Series) -> float:
    values = pd.to_numeric(drawdown, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(-values.max())


def rolling_window_maxdd(returns: pd.Series, window: int) -> pd.Series:
    out = {}
    values = returns.dropna()
    if len(values) < window:
        return pd.Series(dtype=float)
    arr = values.to_numpy(dtype=float)
    idx = values.index
    for i in range(window - 1, len(arr)):
        path = np.cumprod(1.0 + arr[i - window + 1 : i + 1])
        peak = np.maximum.accumulate(path)
        out[idx[i]] = float((path / peak - 1.0).min())
    return pd.Series(out, name=f"maxdd_{window}")


def worst_fraction_mean(series: pd.Series, fraction: float = 0.10) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().sort_values()
    if values.empty:
        return float("nan")
    k = max(1, int(np.ceil(len(values) * fraction)))
    return float(values.iloc[:k].mean())


def worst_compound(returns: pd.Series, window: int) -> float:
    values = returns.dropna()
    if len(values) < window:
        return float("nan")
    wealth = (1.0 + values).rolling(window).apply(np.prod, raw=True) - 1.0
    return float(wealth.min())


def expected_shortfall_95(returns: pd.Series) -> float:
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) < 20:
        return float("nan")
    cutoff = np.quantile(values, 0.05)
    tail = values[values <= cutoff]
    if len(tail) == 0:
        return float("nan")
    return float(-tail.mean())


def cagr_365(nav: pd.DataFrame, initial_cash: float = 300_000.0) -> float:
    frame = nav_frame(nav)
    if len(frame) < 2:
        return float("nan")
    years = max((frame["trade_date"].iloc[-1] - frame["trade_date"].iloc[0]).days / 365.25, 1 / 365.25)
    return float((frame["nav"].iloc[-1] / float(initial_cash)) ** (1.0 / years) - 1.0)


def calmar(cagr: float, maxdd: float) -> float:
    if maxdd is None or not np.isfinite(maxdd) or abs(float(maxdd)) == 0:
        return float("nan")
    return float(cagr / abs(float(maxdd)))


def annualized_log_growth(returns: pd.Series) -> float:
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) == 0 or np.any(values <= -1.0):
        return float("nan")
    return float(252.0 * np.mean(np.log1p(values)))


def drawdown_episodes(path: pd.DataFrame, *, min_depth: float = 1e-6) -> list[dict[str, Any]]:
    """峰值到恢复。样本结束仍未回到峰值则右删失。"""
    episodes: list[dict[str, Any]] = []
    in_dd = False
    peak_i = 0
    trough_i = 0
    dates = path["trade_date"]
    dd = path["drawdown"].to_numpy(dtype=float)
    nav = path["nav"].to_numpy(dtype=float)
    for i in range(len(path)):
        if not in_dd:
            if dd[i] > min_depth:
                in_dd = True
                peak_i = i - 1 if i else 0
                while peak_i > 0 and nav[peak_i] < nav[peak_i - 1]:
                    peak_i -= 1
                trough_i = i
        else:
            if dd[i] >= dd[trough_i]:
                trough_i = i
            if dd[i] <= min_depth:
                episodes.append(_episode(dates, dd, nav, peak_i, trough_i, i, recovered=True))
                in_dd = False
    if in_dd:
        episodes.append(_episode(dates, dd, nav, peak_i, trough_i, len(path) - 1, recovered=False))
    episodes.sort(key=lambda row: row["depth"], reverse=True)
    return episodes


def _episode(dates, dd, nav, peak_i, trough_i, end_i, *, recovered: bool) -> dict[str, Any]:
    return {
        "peak_date": str(pd.Timestamp(dates.iloc[peak_i]).date()),
        "trough_date": str(pd.Timestamp(dates.iloc[trough_i]).date()),
        "recovery_date": str(pd.Timestamp(dates.iloc[end_i]).date()) if recovered else None,
        "right_censored": not recovered,
        "depth": float(dd[trough_i]),
        "peak_nav": float(nav[peak_i]),
        "trough_nav": float(nav[trough_i]),
        "duration_days": int(end_i - peak_i),
    }


def top_nonoverlapping(episodes: list[dict[str, Any]], k: int = 3) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for episode in episodes:
        start = pd.Timestamp(episode["peak_date"])
        end = pd.Timestamp(episode["recovery_date"] or episode["trough_date"])
        overlap = False
        for prev in chosen:
            ps = pd.Timestamp(prev["peak_date"])
            pe = pd.Timestamp(prev["recovery_date"] or prev["trough_date"])
            if start <= pe and end >= ps:
                overlap = True
                break
        if not overlap:
            chosen.append(episode)
        if len(chosen) >= k:
            break
    return chosen


def summarize_nav(nav: pd.DataFrame, *, initial_cash: float = 300_000.0) -> dict[str, Any]:
    path = drawdown_path(nav)
    returns = path["nav"].pct_change().dropna()
    window63 = rolling_window_maxdd(returns, 63)
    cagr = cagr_365(nav, initial_cash=initial_cash)
    mdd = max_drawdown(path["drawdown"])
    return {
        "n_nav_days": int(len(path)),
        "n_return_days": int(returns.shape[0]),
        "cagr_365_25": cagr,
        "ulcer_index": ulcer_index(path["drawdown"]),
        "max_drawdown": mdd,
        "calmar": calmar(cagr, mdd),
        "es95": expected_shortfall_95(returns),
        "worst_21d_compound": worst_compound(returns, 21),
        "worst_63d_compound": worst_compound(returns, 63),
        "window63_count": int(window63.shape[0]),
        "window63_worst10pct_mean": worst_fraction_mean(window63, 0.10),
        "annualized_log_growth": annualized_log_growth(returns),
        "cagr_annualization": "terminal/initial ** (365.25/calendar_days) - 1",
    }
