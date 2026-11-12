"""Round14 共用：落盘、日收益。不改策略。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND14_ROOT, assert_project_result_path, ensure_round14_tree
from ..round11.common import (
    COMPLETE_YEARS,
    daily_from_nav,
    exclude_year_cagr,
    max_drawdown_of,
    metrics_from_nav,
    moments,
    observed_from_report,
    pct,
    rolling_cagr,
    sharpe_of,
    worst_window_return,
)


def dump_json(path: Path, payload: Any) -> Path:
    ensure_round14_tree()
    assert_project_result_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def write_md(path: Path, body: str) -> Path:
    ensure_round14_tree()
    assert_project_result_path(path)
    text = body if body.endswith("\n") else body + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def stage_dir(name: str) -> Path:
    path = ROUND14_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cagr_of(daily: pd.Series) -> float | None:
    x = daily.dropna().astype(float)
    if x.empty:
        return None
    wealth = float((1.0 + x).prod())
    years = max(len(x) / 252.0, 1e-9)
    return wealth ** (1.0 / years) - 1.0


def pack_daily(daily: pd.Series) -> dict[str, Any]:
    w12 = worst_window_return(daily, 252)
    return {
        "cagr": cagr_of(daily),
        "sharpe": sharpe_of(daily),
        "max_drawdown": max_drawdown_of(daily),
        "worst12m": w12.get("return"),
        "worst12m_end": w12.get("end"),
        "n_days": int(daily.dropna().shape[0]),
    }


def cvar95(daily: pd.Series) -> float | None:
    x = daily.dropna().astype(float)
    if x.empty:
        return None
    q = float(np.nanpercentile(x.to_numpy(), 5))
    tail = x.loc[x <= q]
    if tail.empty:
        return q
    return float(tail.mean())


__all__ = [
    "COMPLETE_YEARS",
    "cagr_of",
    "cvar95",
    "daily_from_nav",
    "dump_json",
    "exclude_year_cagr",
    "load_json",
    "max_drawdown_of",
    "metrics_from_nav",
    "moments",
    "observed_from_report",
    "pack_daily",
    "pct",
    "rolling_cagr",
    "sharpe_of",
    "stage_dir",
    "worst_window_return",
    "write_md",
]
