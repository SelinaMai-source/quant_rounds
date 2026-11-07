"""全引擎回放。缓存 load_market() 一次。veto 走 apply_top2_veto + run_r15_backtest。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_5_ROOT, assert_project_result_path
from ..round12.runner import hist_hooks
from ..round15.runner import apply_top2_veto, build_version_overlayed, load_market, run_r15_backtest
from ..round16.backtest import _signals
from ..round16.factors import load_factor_daily
from .common import dump_json, observed_from_report
from .contract import PCT_MIN_OBS, PCT_WINDOW, SKEW_Q, overlayed_path

_CACHE: dict[str, Any] = {}


def cached_market() -> dict[str, Any]:
    if "market" not in _CACHE:
        _CACHE["market"] = load_market()
    return _CACHE["market"]


def cached_overlay_v4() -> pd.DataFrame:
    if "ov4" not in _CACHE:
        ov4, _, _ = build_version_overlayed("v4")
        _CACHE["ov4"] = ov4.copy()
    return _CACHE["ov4"].copy()


def o01_signal_dates() -> pd.DatetimeIndex:
    if "o01_dates" not in _CACHE:
        _CACHE["o01_dates"] = _signals("PUT_SKEW25", q=SKEW_Q)
    return pd.DatetimeIndex(_CACHE["o01_dates"])


def put_skew_series() -> pd.Series:
    if "put_skew" not in _CACHE:
        daily = load_factor_daily()
        if daily.empty or "PUT_SKEW25" not in daily.columns:
            _CACHE["put_skew"] = pd.Series(dtype=float)
        else:
            s = pd.to_numeric(
                daily.set_index(pd.to_datetime(daily["trade_date"]).dt.normalize())["PUT_SKEW25"],
                errors="coerce",
            ).sort_index()
            _CACHE["put_skew"] = s
    return _CACHE["put_skew"]


def option_observable_dates() -> pd.DatetimeIndex:
    s = put_skew_series()
    past = s.shift(1)
    past_q = past.rolling(PCT_WINDOW, min_periods=PCT_MIN_OBS).quantile(SKEW_Q)
    ok = past_q.notna()
    return pd.DatetimeIndex(ok.index[ok])


def run_engine(
    dest,
    overlayed: pd.DataFrame,
    *,
    extra: dict[str, Any] | None = None,
    resume: bool = True,
    write: bool = True,
    extra_buy_bps: float | None = None,
) -> dict[str, Any]:
    loaded = cached_market()
    dest = Path(dest)
    if write:
        dest.mkdir(parents=True, exist_ok=True)
        assert_project_result_path(dest / "report.json")
    ran = run_r15_backtest(
        dest,
        overlayed=overlayed,
        validation_hooks=hist_hooks(extra_buy_bps=extra_buy_bps),
        extra_report=extra or {},
        resume=resume,
        write=write,
        prepared_market=loaded["prepared"],
        market_meta=loaded,
    )
    report = ran.get("report") or {}
    obs = observed_from_report(report)
    obs.update({"code": ran.get("code") or dest.name, "status": ran.get("status"), "out_dir": str(dest)})
    if write and (dest / "nav.parquet").is_file():
        obs["nav_path"] = str(dest / "nav.parquet")
        obs["trades_path"] = str(dest / "trades.parquet")
    ran["observed"] = obs
    if write and not (dest / "summary.json").is_file() and obs.get("cagr") is not None:
        dump_json(dest / "summary.json", obs)
    return ran


def run_veto(
    dest,
    veto_dates,
    *,
    extra: dict[str, Any] | None = None,
    resume: bool = True,
    write: bool = True,
    extra_buy_bps: float | None = None,
    overlayed: pd.DataFrame | None = None,
) -> dict[str, Any]:
    ov = overlayed if overlayed is not None else cached_overlay_v4()
    ov = apply_top2_veto(ov, veto_dates)
    return run_engine(
        dest,
        ov,
        extra=extra,
        resume=resume,
        write=write,
        extra_buy_bps=extra_buy_bps,
    )


def save_overlay(code: str, overlayed: pd.DataFrame) -> Path:
    path = overlayed_path(code)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(path)
    overlayed.to_parquet(path, index=False)
    return path


def pack_from_ran(ran: dict[str, Any]) -> dict[str, Any]:
    obs = ran.get("observed") or observed_from_report(ran.get("report") or {})
    nav = ran.get("nav")
    trades = ran.get("trades")
    if nav is None and ran.get("out_dir"):
        p = Path(ran["out_dir"]) / "nav.parquet"
        if p.is_file():
            nav = pd.read_parquet(p)
    if trades is None and ran.get("out_dir"):
        p = Path(ran["out_dir"]) / "trades.parquet"
        if p.is_file():
            trades = pd.read_parquet(p)
    return {"observed": obs, "nav": nav, "trades": trades, "report": ran.get("report"), "status": ran.get("status")}


def metrics_close(obs: dict[str, Any], target: dict[str, Any], *, abs_tol: float) -> bool:
    try:
        ok = (
            abs(float(obs.get("cagr") or 0) - float(target["cagr"])) < abs_tol
            and abs(float(obs.get("sharpe") or 0) - float(target["sharpe"])) < abs_tol
            and abs(float(obs.get("max_drawdown") or 0) - float(target["max_drawdown"])) < abs_tol
        )
        if "completed_trades" in target and target["completed_trades"] is not None:
            ok = ok and int(obs.get("completed_trades") or -1) == int(target["completed_trades"])
        return bool(ok)
    except (TypeError, ValueError, KeyError):
        return False


def with_rank(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    t["signal_date"] = pd.to_datetime(t["signal_date"]).dt.normalize()
    t["instrument_id"] = t["instrument_id"].astype(str)
    t["_rank"] = t.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    t["net"] = pd.to_numeric(t.get("sell_proceeds"), errors="coerce") - pd.to_numeric(t.get("buy_cost"), errors="coerce")
    return t


__all__ = [
    "cached_market",
    "cached_overlay_v4",
    "metrics_close",
    "o01_signal_dates",
    "option_observable_dates",
    "pack_from_ran",
    "put_skew_series",
    "run_engine",
    "run_veto",
    "save_overlay",
    "with_rank",
]
