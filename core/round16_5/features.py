"""已有 PIT 股票状态。不发明新股票因子。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..round16.download import load_etf
from .engine import cached_market, cached_overlay_v4, with_rank
from .contract import V00, parity_dir


def _up_ratio() -> pd.Series:
    prepared = cached_market()["prepared"]
    up = getattr(prepared, "market_up_ratio", None)
    if up is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(pd.Series(up), errors="coerce")
    s.index = pd.to_datetime(s.index).normalize()
    s.name = "market_up_ratio"
    return s.sort_index()


def _b01a(idx: pd.DatetimeIndex) -> pd.Series:
    from ..round8.overlays import market_b01a
    from ..round15.runner import load_alpha

    pred = load_alpha()
    series = market_b01a(pred)
    series.index = pd.to_datetime(series.index).normalize()
    return pd.to_numeric(series.reindex(idx), errors="coerce").rename("B01A")


def _candidate_count(idx: pd.DatetimeIndex) -> pd.Series:
    ov = cached_overlay_v4()
    ov = ov.copy()
    ov["signal_date"] = pd.to_datetime(ov["signal_date"]).dt.normalize()
    n = ov.groupby("signal_date")["instrument_id"].nunique()
    n.index = pd.to_datetime(n.index).normalize()
    return pd.to_numeric(n.reindex(idx), errors="coerce").rename("candidate_count")


def _index_rv(idx: pd.DatetimeIndex) -> pd.Series:
    etf = load_etf()
    if etf.empty:
        return pd.Series(np.nan, index=idx, name="index_RV20")
    e = etf.copy()
    code_col = "code" if "code" in e.columns else ("_key" if "_key" in e.columns else None)
    time_col = "kline_time" if "kline_time" in e.columns else ("trade_date" if "trade_date" in e.columns else None)
    if code_col is None or time_col is None or "close" not in e.columns:
        return pd.Series(np.nan, index=idx, name="index_RV20")
    e["underlying_code"] = e[code_col].astype(str)
    e["trade_date"] = pd.to_datetime(e[time_col], errors="coerce").dt.normalize()
    e["close"] = pd.to_numeric(e["close"], errors="coerce")
    e = e.dropna(subset=["trade_date", "close"]).sort_values(["underlying_code", "trade_date"])
    prefer = e["underlying_code"].astype(str).str.contains("510300")
    if prefer.any():
        e = e.loc[prefer | (e["underlying_code"].nunique() == 1)]
        if e["underlying_code"].nunique() > 1:
            e = e.loc[e["underlying_code"].astype(str).str.contains("510300")]
    e["logret"] = e.groupby("underlying_code")["close"].transform(lambda s: np.log(s).diff())
    e["RV20"] = e.groupby("underlying_code")["logret"].transform(lambda s: s.rolling(20, min_periods=15).std() * np.sqrt(252))
    rv = e.groupby("trade_date")["RV20"].median()
    rv.index = pd.to_datetime(rv.index).normalize()
    return pd.to_numeric(rv.reindex(idx), errors="coerce").rename("index_RV20")


def equity_panel() -> pd.DataFrame:
    up = _up_ratio()
    if up.empty:
        ov = cached_overlay_v4()
        idx = pd.DatetimeIndex(pd.to_datetime(ov["signal_date"]).dt.normalize().unique()).sort_values()
    else:
        idx = pd.DatetimeIndex(up.index).sort_values()
    out = pd.DataFrame(index=idx)
    out["market_up_ratio"] = up.reindex(idx)
    out["B01A"] = _b01a(idx)
    out["candidate_count"] = _candidate_count(idx)
    out["index_RV20"] = _index_rv(idx)
    return out


def v4_top2_labels() -> pd.DataFrame:
    path = parity_dir(V00) / "trades.parquet"
    t = with_rank(pd.read_parquet(path))
    t2 = t.loc[t["_rank"].eq(2)].copy()
    t2["signal_date"] = pd.to_datetime(t2["signal_date"]).dt.normalize()
    t2["actual_exit_date"] = pd.to_datetime(t2["actual_exit_date"]).dt.normalize()
    t2["entry_date"] = pd.to_datetime(t2["entry_date"]).dt.normalize()
    t2["buy_cost"] = pd.to_numeric(t2["buy_cost"], errors="coerce")
    t2["ret_net"] = t2["net"] / t2["buy_cost"].replace(0, np.nan)
    g = t2.groupby("signal_date").agg(
        top2_net=("net", "sum"),
        top2_ret=("ret_net", "mean"),
        actual_exit_date=("actual_exit_date", "max"),
        n_top2=("instrument_id", "size"),
        budget=("target_name_budget", "sum"),
    )
    g["loss"] = (g["top2_net"] < 0).astype(float)
    g["severe_loss"] = (g["top2_ret"] < -0.05).astype(float)
    return g.reset_index()


def ridge_fit(X: np.ndarray, y: np.ndarray, *, alpha: float, intercept: bool = True) -> np.ndarray:
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if intercept:
        Xd = np.column_stack([np.ones(len(X)), X])
        penalty = np.eye(Xd.shape[1]) * float(alpha)
        penalty[0, 0] = 0.0
    else:
        Xd = X
        penalty = np.eye(Xd.shape[1]) * float(alpha)
    xtx = Xd.T @ Xd + penalty
    xty = Xd.T @ y
    try:
        beta = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(xtx, xty, rcond=None)
    return beta


def ridge_predict(X: np.ndarray, beta: np.ndarray, *, intercept: bool = True) -> np.ndarray:
    X = np.asarray(X, float)
    if intercept:
        Xd = np.column_stack([np.ones(len(X)), X])
    else:
        Xd = X
    return Xd @ beta


def standardize_train(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    tr = (train - mu) / sd
    te = (test - mu) / sd
    tr = np.where(np.isfinite(tr), tr, 0.0)
    te = np.where(np.isfinite(te), te, 0.0)
    return tr, te, mu, sd
