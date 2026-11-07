"""Track A 路径与共用。产物只写 result/round19。不覆盖 round16*/round18。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...paths import ROUND16_5_ROOT, ROUND16_ROOT, ROUND19_ROOT, assert_project_result_path
from ...round16.common import pack_daily
from ...round16_5.common import empirical_p, nav_end
from ...round16_5.engine import with_rank
from ..common import dump_json, load_json, slim_obs, write_md
from ..contract import A_C0, A_C1, A_C2, A_C3, A_C4, exp_dir, overlayed_path, parity_dir

TA = ROUND19_ROOT / "track_a"
REPLAY_JSON = TA / "replay" / "replay.json"
EVENTS_JSON = TA / "events" / "events.json"
ATTR_JSON = TA / "attribution" / "attribution.json"
ATTR_CSV = TA / "attribution" / "O01_treated_trades.csv"
CONC_JSON = TA / "concentration" / "concentration.json"
CONC_CKPT = TA / "concentration" / "loo_runs.jsonl"
EQ_JSON = TA / "controls" / "equity.json"
MATCH_JSON = TA / "controls" / "match.json"
PLACEBO_JSON = TA / "placebo" / "placebo.json"
PLACEBO_CKPT = TA / "placebo" / "placebo_runs.jsonl"
CHECK_JSON = TA / "placebo" / "engine_check.json"
PIT_JSON = TA / "pit" / "pit.json"
QUOTE_JSON = TA / "quotes" / "quote.json"
PRED_JSON = TA / "prediction" / "prediction.json"
STAT_JSON = TA / "stats" / "stats.json"
BOOT_JSON = ROUND19_ROOT / "bootstrap" / "bootstrap.json"
SPA_JSON = ROUND19_ROOT / "spa" / "spa.json"
STRESS_JSON = ROUND19_ROOT / "stress" / "stress.json"
DECISION_JSON = TA / "decision.json"
R16_5_V01_DIR = ROUND16_5_ROOT / "parity" / "R16_5_V01_O01_EXACT"
R16_O01_DIR = ROUND16_ROOT / "candidates" / "R16_O01"


def control_dir(code: str) -> Path:
    return ROUND19_ROOT / "controls" / str(code)


def ta_md(name: str) -> Path:
    return TA / name


def skip_result(stage: str, reason: str, **extra: Any) -> dict[str, Any]:
    payload = {"status": "SKIP", "stage": stage, "reason": reason, **extra}
    return payload


def blocked_result(stage: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": "BLOCKED", "stage": stage, "reason": reason, **extra}


def resume_ok(path: Path, *, status: str = "DONE") -> dict[str, Any] | None:
    if not path.is_file():
        return None
    existing = load_json(path)
    if existing.get("status") == status:
        return existing
    return None


def idx_dates(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(values))).normalize().unique().sort_values()


def ymd_list(idx) -> list[str]:
    if idx is None:
        return []
    out = pd.DatetimeIndex(pd.to_datetime(list(idx))).normalize().unique().sort_values()
    return [str(x.date()) for x in out]


def load_nav_trades(dest: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Return (trades, nav)."""
    nav_p = dest / "nav.parquet"
    tr_p = dest / "trades.parquet"
    nav = pd.read_parquet(nav_p) if nav_p.is_file() else None
    tr = pd.read_parquet(tr_p) if tr_p.is_file() else None
    return tr, nav


def top2(trades: pd.DataFrame) -> pd.DataFrame:
    t = with_rank(trades)
    return t.loc[t["_rank"].eq(2)].copy()


def planned_budget_by_date(top2_trades: pd.DataFrame) -> pd.Series:
    t = top2_trades.copy()
    t["signal_date"] = pd.to_datetime(t["signal_date"]).dt.normalize()
    t["target_name_budget"] = pd.to_numeric(t["target_name_budget"], errors="coerce")
    return t.groupby("signal_date")["target_name_budget"].sum().sort_index()


def pack_obs_from_dir(dest: Path) -> dict[str, Any]:
    summary = load_json(dest / "summary.json")
    if summary.get("cagr") is not None:
        return slim_obs(summary)
    _, nav = load_nav_trades(dest)
    if nav is None:
        return {}
    from ..common import daily_from_nav, observed_from_report

    report = load_json(dest / "report.json")
    if report:
        return slim_obs(observed_from_report(report))
    return slim_obs(pack_daily(daily_from_nav(nav)))


def require_c0() -> Path:
    dest = parity_dir(A_C0)
    if not (dest / "trades.parquet").is_file() or not (dest / "nav.parquet").is_file():
        raise FileNotFoundError("C0_MISSING")
    return dest


def require_c1() -> Path:
    dest = parity_dir(A_C1)
    if not (dest / "trades.parquet").is_file() or not (dest / "nav.parquet").is_file():
        raise FileNotFoundError("C1_MISSING")
    return dest


__all__ = [
    "A_C0",
    "A_C1",
    "A_C2",
    "A_C3",
    "A_C4",
    "ATTR_CSV",
    "ATTR_JSON",
    "BOOT_JSON",
    "CHECK_JSON",
    "CONC_CKPT",
    "CONC_JSON",
    "DECISION_JSON",
    "EQ_JSON",
    "EVENTS_JSON",
    "MATCH_JSON",
    "PIT_JSON",
    "PLACEBO_CKPT",
    "PLACEBO_JSON",
    "PRED_JSON",
    "QUOTE_JSON",
    "R16_5_V01_DIR",
    "R16_O01_DIR",
    "REPLAY_JSON",
    "SPA_JSON",
    "STAT_JSON",
    "STRESS_JSON",
    "TA",
    "assert_project_result_path",
    "blocked_result",
    "control_dir",
    "dump_json",
    "empirical_p",
    "exp_dir",
    "idx_dates",
    "load_json",
    "load_nav_trades",
    "nav_end",
    "overlayed_path",
    "pack_daily",
    "pack_obs_from_dir",
    "parity_dir",
    "planned_budget_by_date",
    "require_c0",
    "require_c1",
    "resume_ok",
    "skip_result",
    "slim_obs",
    "ta_md",
    "top2",
    "write_md",
    "ymd_list",
]
