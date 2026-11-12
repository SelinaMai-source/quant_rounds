"""时间 / 制度 / 尾部。赢家移除标 NOT REALISTIC。只分析，不改策略。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND11_ROOT
from .common import (
    COMPLETE_YEARS,
    daily_from_nav,
    dump_json,
    exclude_year_cagr,
    load_anchor_nav,
    load_anchor_trades,
    load_legacy_dailies,
    max_drawdown_of,
    pct,
    rolling_cagr,
    sharpe_of,
    stage_dir,
    worst_window_return,
    write_md,
)
from .contract import BASELINES
from .cost import load_cost_status

TIME_MD = ROUND11_ROOT / "05_time_regime_tail.md"


def _yearly(daily: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for year, block in daily.groupby(daily.index.year):
        if block.empty:
            continue
        out[str(int(year))] = float((1.0 + block).prod() - 1.0)
    return out


def _rolling_summary(daily: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, days in (("6M", 126), ("12M", 252), ("24M", 504)):
        series = rolling_cagr(daily, days)
        out[label] = {
            "median": float(series.median()) if len(series) else None,
            "worst": float(series.min()) if len(series) else None,
            "best": float(series.max()) if len(series) else None,
            "n": int(series.notna().sum()),
        }
    return out


def _regime_quartiles(daily: pd.Series) -> dict[str, Any]:
    """按日历年收益四分位分组（只分析，不调参）。"""
    yearly = _yearly(daily)
    complete = {k: v for k, v in yearly.items() if k.isdigit() and int(k) in COMPLETE_YEARS}
    if len(complete) < 4:
        return {"status": "TOO_FEW_YEARS", "yearly": yearly}
    vals = pd.Series(complete)
    qs = vals.quantile([0.25, 0.5, 0.75]).to_dict()
    buckets = {"Q1_low": [], "Q2": [], "Q3": [], "Q4_high": []}
    for year, ret in complete.items():
        if ret <= qs[0.25]:
            buckets["Q1_low"].append({year: ret})
        elif ret <= qs[0.5]:
            buckets["Q2"].append({year: ret})
        elif ret <= qs[0.75]:
            buckets["Q3"].append({year: ret})
        else:
            buckets["Q4_high"].append({year: ret})
    return {"status": "ANALYSIS_ONLY", "quantiles": {str(k): float(v) for k, v in qs.items()}, "buckets": buckets}


def _winner_removal(trades: pd.DataFrame) -> dict[str, Any]:
    t = trades.copy()
    if t.empty or "ret" not in t.columns:
        return {"status": "NO_TRADES"}
    t["dollar"] = pd.to_numeric(t.get("sell_proceeds"), errors="coerce") - pd.to_numeric(
        t.get("buy_cost"), errors="coerce"
    )
    t = t.dropna(subset=["dollar"])
    if t.empty:
        return {"status": "NO_PNL"}
    total = float(t["dollar"].sum())
    top = t.nlargest(max(1, int(np.ceil(len(t) * 0.01))), "dollar")
    top5 = t.nlargest(max(1, int(np.ceil(len(t) * 0.05))), "dollar")
    return {
        "status": "NOT REALISTIC",
        "note": "移除赢家不是可交易规则，只作尾部依赖诊断。",
        "n": int(len(t)),
        "total_pnl": total,
        "top1pct_share": float(top["dollar"].sum() / total) if total else None,
        "top5pct_share": float(top5["dollar"].sum() / total) if total else None,
        "best_trade_pnl": float(t["dollar"].max()),
    }


def _profile(name: str, daily: pd.Series, trades: pd.DataFrame | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "n": int(len(daily)),
        "sharpe": sharpe_of(daily),
        "max_drawdown": max_drawdown_of(daily),
        "yearly": _yearly(daily),
        "rolling": _rolling_summary(daily),
        "loyo": {str(y): exclude_year_cagr(daily, y) for y in COMPLETE_YEARS},
        "ex2024_cagr": exclude_year_cagr(daily, 2024),
        "ex2025_cagr": exclude_year_cagr(daily, 2025),
        "worst_3m": worst_window_return(daily, 63),
        "worst_6m": worst_window_return(daily, 126),
        "worst_12m": worst_window_return(daily, 252),
        "regime_quartiles": _regime_quartiles(daily),
        "winner_removal": _winner_removal(trades) if trades is not None else {"status": "SKIP"},
    }


def run_time(**_: Any) -> dict[str, Any]:
    dest = stage_dir("time")
    legacy = load_legacy_dailies()
    profiles: dict[str, Any] = {}
    for key, spec in BASELINES.items():
        profiles[f"legacy_{key}"] = _profile(f"legacy_{key}", legacy[key], load_anchor_trades(spec))
    cost = load_cost_status()
    for key, row in (cost.get("corrected") or {}).items():
        nav_path = row.get("nav_path")
        if not nav_path:
            continue
        nav = pd.read_parquet(nav_path)
        trades_path = str(nav_path).replace("nav.parquet", "trades.parquet")
        trades = pd.read_parquet(trades_path) if trades_path.endswith("trades.parquet") else None
        try:
            trades = pd.read_parquet(row["out_dir"] + "/trades.parquet")
        except Exception:
            trades = None
        profiles[f"corrected_{key}"] = _profile(f"corrected_{key}", daily_from_nav(nav), trades)
    dump_json(dest / "time.json", {"profiles": profiles, "post_hoc_diagnostic": True})
    v4l = profiles.get("legacy_v4") or {}
    v4c = profiles.get("corrected_v4") or {}
    lines = [
        "# Round 11 · 时间 / 制度 / 尾部",
        "",
        "只分析。赢家移除 = **NOT REALISTIC**。不为过关改策略。",
        "",
        "## v4 legacy vs corrected",
        "",
        "| 套 | Sharpe | MaxDD | Ex2024 CAGR | Ex2025 CAGR | 最差 3M | 最差 6M | 最差 12M |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, prof in (("legacy", v4l), ("corrected", v4c)):
        lines.append(
            "| {t} | {sh} | {dd} | {e4} | {e5} | {w3} | {w6} | {w12} |".format(
                t=tag,
                sh=prof.get("sharpe"),
                dd=pct(prof.get("max_drawdown")),
                e4=pct(prof.get("ex2024_cagr")),
                e5=pct(prof.get("ex2025_cagr")),
                w3=pct((prof.get("worst_3m") or {}).get("return")),
                w6=pct((prof.get("worst_6m") or {}).get("return")),
                w12=pct((prof.get("worst_12m") or {}).get("return")),
            )
        )
    lines += ["", "## 分年", "", "| 套 | " + " | ".join(str(y) for y in range(2018, 2027)) + " |", "|---|" + "---:|" * 9]
    for tag, prof in (("legacy_v4", v4l), ("corrected_v4", v4c)):
        yearly = prof.get("yearly") or {}
        cells = [pct(yearly.get(str(y))) for y in range(2018, 2027)]
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")
    wr = (v4l.get("winner_removal") or {})
    lines += [
        "",
        "## 赢家移除（NOT REALISTIC）",
        "",
        f"- top1% 贡献 = {wr.get('top1pct_share')}",
        f"- top5% 贡献 = {wr.get('top5pct_share')}",
        f"- {wr.get('note')}",
        "",
        "## 滚动 CAGR",
        "",
    ]
    for label in ("6M", "12M", "24M"):
        r = (v4l.get("rolling") or {}).get(label) or {}
        lines.append(f"- legacy v4 {label} median={pct(r.get('median'))} worst={pct(r.get('worst'))}")
    write_md(TIME_MD, "\n".join(lines))
    return {"status": "DONE", "profiles": {k: {kk: vv for kk, vv in v.items() if kk != "yearly"} for k, v in profiles.items()}}


__all__ = ["run_time"]
