"""Round11 共用：NAV 分析、落盘、legacy 日收益。不改策略。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..metrics import clip_nav_for_metrics
from ..paths import ROUND11_ROOT, assert_project_result_path, ensure_round11_tree
from .contract import BASELINES, BaselineSpec

COMPLETE_YEARS = tuple(range(2018, 2026))


def dump_json(path: Path, payload: Any) -> Path:
    ensure_round11_tree()
    assert_project_result_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def write_md(path: Path, body: str) -> Path:
    ensure_round11_tree()
    assert_project_result_path(path)
    text = body if body.endswith("\n") else body + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def pct(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def daily_from_nav(nav: pd.DataFrame) -> pd.Series:
    ordered = clip_nav_for_metrics(nav).sort_values("trade_date").copy()
    ordered["trade_date"] = pd.to_datetime(ordered["trade_date"]).dt.normalize()
    s = ordered.set_index("trade_date")["total_asset"].astype(float).pct_change().dropna()
    s.name = "ret"
    return s


def metrics_from_nav(nav: pd.DataFrame, *, initial_cash: float = 300_000.0) -> dict[str, Any]:
    from ..metrics import attach_nav_metrics
    from ..round4.runner import attach_round4_fields

    metrics: dict[str, Any] = {}
    attach_nav_metrics(nav, metrics, initial_cash=initial_cash)
    attach_round4_fields(nav, pd.DataFrame(), metrics)
    return metrics


def observed_from_report(report: dict[str, Any]) -> dict[str, Any]:
    m = report.get("metrics") or {}
    return {
        "cagr": m.get("annualized_return"),
        "sharpe": m.get("sharpe_zero_rate"),
        "max_drawdown": m.get("max_drawdown"),
        "completed_trades": m.get("completed_trades"),
        "total_return": m.get("total_return"),
        "loyo_median": m.get("loyo_median"),
        "loyo_worst": m.get("loyo_worst"),
        "yearly_returns": m.get("yearly_returns") or {},
    }


def load_anchor_nav(spec: BaselineSpec) -> pd.DataFrame:
    return pd.read_parquet(spec.anchor_dir / "nav.parquet")


def load_anchor_trades(spec: BaselineSpec) -> pd.DataFrame:
    return pd.read_parquet(spec.anchor_dir / "trades.parquet")


def load_legacy_dailies() -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for key, spec in BASELINES.items():
        out[key] = daily_from_nav(load_anchor_nav(spec))
    return out


def rolling_cagr(daily: pd.Series, window: int) -> pd.Series:
    if len(daily) < window:
        return pd.Series(dtype=float)
    wealth = daily.rolling(window).apply(lambda x: float(np.prod(1.0 + x)), raw=True)
    years = window / 252.0
    return wealth ** (1.0 / years) - 1.0


def worst_window_return(daily: pd.Series, days: int) -> dict[str, Any]:
    if len(daily) < days:
        return {"days": days, "return": None, "end": None}
    rolled = daily.rolling(days).apply(lambda x: float(np.prod(1.0 + x)) - 1.0, raw=True)
    idx = rolled.idxmin()
    return {"days": days, "return": float(rolled.min()), "end": str(pd.Timestamp(idx).date())}


def exclude_year_cagr(daily: pd.Series, year: int) -> float | None:
    adj = daily.copy()
    adj.loc[adj.index.year == int(year)] = 0.0
    if adj.empty:
        return None
    wealth = float((1.0 + adj).prod())
    years = max(len(daily) / 252.0, 1e-9)
    return wealth ** (1.0 / years) - 1.0


def sharpe_of(daily: pd.Series) -> float | None:
    if len(daily) < 3:
        return None
    vol = float(daily.std(ddof=1))
    if vol <= 0 or not np.isfinite(vol):
        return None
    return float(daily.mean() / vol * math.sqrt(252))


def max_drawdown_of(daily: pd.Series) -> float | None:
    if daily.empty:
        return None
    nav = (1.0 + daily).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def moments(daily: pd.Series) -> dict[str, float]:
    x = daily.dropna().astype(float)
    if len(x) < 4:
        return {"n": float(len(x)), "mean": float("nan"), "std": float("nan"), "skew": float("nan"), "kurt": float("nan")}
    mu = float(x.mean())
    sigma = float(x.std(ddof=1))
    z = (x - mu) / sigma if sigma > 0 else x * 0.0
    return {
        "n": float(len(x)),
        "mean": mu,
        "std": sigma,
        "skew": float((z ** 3).mean()),
        "kurt": float((z ** 4).mean()),
    }


def stage_dir(name: str) -> Path:
    path = ROUND11_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "COMPLETE_YEARS",
    "daily_from_nav",
    "dump_json",
    "exclude_year_cagr",
    "load_anchor_nav",
    "load_anchor_trades",
    "load_legacy_dailies",
    "max_drawdown_of",
    "metrics_from_nav",
    "moments",
    "observed_from_report",
    "pct",
    "rolling_cagr",
    "sharpe_of",
    "stage_dir",
    "worst_window_return",
    "write_md",
]
