"""Round16 共用：落盘、打码、哈希。不改策略。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_ROOT, assert_project_result_path, ensure_round16_tree
from ..round11.common import (
    COMPLETE_YEARS,
    daily_from_nav,
    max_drawdown_of,
    metrics_from_nav,
    observed_from_report,
    pct,
    rolling_cagr,
    sharpe_of,
    worst_window_return,
)


def dump_json(path: Path, payload: Any) -> Path:
    ensure_round16_tree()
    assert_project_result_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def write_md(path: Path, body: str) -> Path:
    ensure_round16_tree()
    assert_project_result_path(path)
    text = body if body.endswith("\n") else body + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def redact(obj: Any) -> Any:
    secret = {"password", "passwd", "token", "secret", "api_key", "apikey", "host", "username", "user", "account", "ip"}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in secret or "password" in kl or "token" in kl or kl == "host":
                out[k] = "***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact(x) for x in obj]
    return obj


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


def pick_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    for c in df.columns:
        cl = str(c).lower()
        if any(n.lower() in cl for n in names):
            return c
    return None


def to_ymd(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, (int, np.integer)):
        v = int(value)
        return v if v >= 19000101 else None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        s = str(value).strip()
        digits = "".join(ch for ch in s if ch.isdigit())
        if len(digits) >= 8:
            return int(digits[:8])
        return None
    return int(ts.strftime("%Y%m%d"))


__all__ = [
    "COMPLETE_YEARS",
    "cagr_of",
    "daily_from_nav",
    "dump_json",
    "file_sha256",
    "load_json",
    "max_drawdown_of",
    "metrics_from_nav",
    "observed_from_report",
    "pack_daily",
    "pct",
    "pick_col",
    "redact",
    "rolling_cagr",
    "sharpe_of",
    "to_ymd",
    "worst_window_return",
    "write_md",
]
