"""正式袖套。复用 local_executor，不另起执行器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .research_io import dump_json, finite_or_none, load_json
from .round22_3.contract import FEE_SCHEDULE, MAIN_WINDOW_END, MAIN_WINDOW_START, material_vs_v5


def _scores(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if "signal_date" not in work.columns and "trade_date" in work.columns:
        work = work.rename(columns={"trade_date": "signal_date"})
    work["signal_date"] = pd.to_datetime(work["signal_date"])
    work["instrument_id"] = work["instrument_id"].astype(str)
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work = work.dropna(subset=["signal_date", "instrument_id", "score"])
    return work[["signal_date", "instrument_id", "score"]]


def metrics_from_nav_file(nav_path: Path) -> dict[str, Any]:
    from .round11.common import daily_from_nav, metrics_from_nav

    nav = pd.read_parquet(nav_path)
    if "trade_date" in nav.columns:
        nav["trade_date"] = pd.to_datetime(nav["trade_date"])
    daily = daily_from_nav(nav)
    vol = float(daily.std(ddof=1)) if len(daily) > 2 else None
    sharpe = float(daily.mean() / vol * (252 ** 0.5)) if vol and vol > 0 else None
    wealth = (1.0 + daily.astype(float)).cumprod()
    peak = wealth.cummax()
    dd = float((wealth / peak - 1.0).min()) if len(wealth) else None
    packed = metrics_from_nav(nav)
    nav_end = float(nav["total_asset"].iloc[-1])
    nav0 = float(nav["total_asset"].iloc[0])
    return {
        "cagr": finite_or_none(packed.get("annualized_return")),
        "sharpe": sharpe,
        "max_drawdown": dd,
        "nav_end": nav_end,
        "nav_start": nav0,
        "total_return": (nav_end / nav0 - 1.0) if nav0 else None,
        "n_obs": int(len(daily)),
        "window_start": str(pd.Timestamp(nav["trade_date"].iloc[0]).date()) if "trade_date" in nav.columns and len(nav) else None,
        "window_end": str(pd.Timestamp(nav["trade_date"].iloc[-1]).date()) if "trade_date" in nav.columns and len(nav) else None,
    }


def _yearly_not_only(nav_path: Path, base_nav: Path | None) -> bool:
    if not nav_path.is_file() or base_nav is None or not base_nav.is_file():
        return True
    from .round11.common import daily_from_nav

    nav = pd.read_parquet(nav_path)
    base = pd.read_parquet(base_nav)
    nav["trade_date"] = pd.to_datetime(nav["trade_date"])
    base["trade_date"] = pd.to_datetime(base["trade_date"])
    a = daily_from_nav(nav)
    b = daily_from_nav(base)
    if not isinstance(a.index, pd.DatetimeIndex):
        return True
    excess = (a - b).dropna()
    if excess.empty:
        return True
    by_year = excess.groupby(excess.index.year).sum()
    positive = [int(y) for y, v in by_year.items() if v > 0]
    if not positive:
        return True
    return set(positive) - {2024, 2025} != set()


def run_heads_sleeve(
    dest: Path,
    five: pd.DataFrame,
    ten: pd.DataFrame,
    extra: dict[str, Any],
    *,
    resume: bool = True,
    base_nav: Path | None = None,
) -> dict[str, Any]:
    from .paths import assert_project_result_path
    from .round15.runner import apply_top2_veto
    from .round16_5.engine import o01_signal_dates
    from .round17.backtest import apply_b03, fuse_heads
    from .round20.parity import _pack, _run_r7

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest / "report.json")
    code = str(extra.get("code") or dest.name)
    if resume and (dest / "summary.json").is_file() and (dest / "nav.parquet").is_file():
        cached = load_json(dest / "summary.json")
        if cached.get("code") == code and cached.get("status") == "ran":
            return cached
    fused = fuse_heads(_scores(five), _scores(ten))
    fused.to_parquet(dest / "predictions.parquet", index=False)
    overlayed, meta = apply_b03(fused)
    dump_json(dest / "overlay_meta.json", meta)
    final = apply_top2_veto(overlayed, o01_signal_dates())
    final.to_parquet(dest / "overlayed.parquet", index=False)
    payload = {
        "fee_schedule": FEE_SCHEDULE,
        "validation_hooks": None,
        "historical_stamp_tax": False,
        "engine": "fuse_heads 90/10 → apply_b03 → o01 → local_executor",
        "restarted_capital": False,
        "window": f"{MAIN_WINDOW_START}–{MAIN_WINDOW_END}",
        "compare_to": "R16_O01",
    }
    payload.update(extra)
    payload["code"] = code
    ran = _run_r7(dest, final, payload)
    packed = _pack(ran, dest)
    obs = metrics_from_nav_file(dest / "nav.parquet")
    trades = packed.get("completed_trades")
    obs.update(
        {
            "code": code,
            "completed_trades": trades,
            "status": "ran",
            "fee_schedule": FEE_SCHEDULE,
            "compare_to": "R16_O01",
            "window_start": obs.get("window_start") or MAIN_WINDOW_START,
            "window_end": obs.get("window_end") or MAIN_WINDOW_END,
            "restarted_capital": False,
            "scenario": bool(extra.get("scenario")),
            "round": extra.get("round"),
        }
    )
    gate = material_vs_v5(
        obs,
        top_changed=bool(extra.get("top_changed", True)),
        fee_same_direction=True,
        not_only_year=_yearly_not_only(dest / "nav.parquet", base_nav),
    )
    obs.update(gate)
    dump_json(dest / "summary.json", obs)
    return obs


def run_scored_sleeve(
    dest: Path,
    scored: pd.DataFrame,
    extra: dict[str, Any],
    *,
    resume: bool = True,
    base_nav: Path | None = None,
) -> dict[str, Any]:
    """已经融合好的分数，再走 B03、O01 和正式执行器。不再套一层 90/10。"""
    from .paths import assert_project_result_path
    from .round15.runner import apply_top2_veto
    from .round16_5.engine import o01_signal_dates
    from .round17.backtest import apply_b03
    from .round20.parity import _pack, _run_r7

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest / "report.json")
    code = str(extra.get("code") or dest.name)
    if resume and (dest / "summary.json").is_file() and (dest / "nav.parquet").is_file():
        cached = load_json(dest / "summary.json")
        if cached.get("code") == code and cached.get("status") == "ran":
            return cached
    frame = _scores(scored)
    frame.to_parquet(dest / "predictions.parquet", index=False)
    overlayed, meta = apply_b03(frame)
    dump_json(dest / "overlay_meta.json", meta)
    final = apply_top2_veto(overlayed, o01_signal_dates())
    final.to_parquet(dest / "overlayed.parquet", index=False)
    payload = {
        "fee_schedule": FEE_SCHEDULE,
        "validation_hooks": None,
        "historical_stamp_tax": False,
        "engine": "prefused score → apply_b03 → o01 → local_executor",
        "restarted_capital": False,
        "window": f"{MAIN_WINDOW_START}–{MAIN_WINDOW_END}",
        "compare_to": "R16_O01",
    }
    payload.update(extra)
    payload["code"] = code
    ran = _run_r7(dest, final, payload)
    packed = _pack(ran, dest)
    obs = metrics_from_nav_file(dest / "nav.parquet")
    obs.update(
        {
            "code": code,
            "completed_trades": packed.get("completed_trades"),
            "status": "ran",
            "fee_schedule": FEE_SCHEDULE,
            "compare_to": "R16_O01",
            "restarted_capital": False,
            "scenario": bool(extra.get("scenario")),
            "round": extra.get("round"),
        }
    )
    gate = material_vs_v5(
        obs,
        top_changed=bool(extra.get("top_changed", True)),
        fee_same_direction=True,
        not_only_year=_yearly_not_only(dest / "nav.parquet", base_nav),
    )
    obs.update(gate)
    dump_json(dest / "summary.json", obs)
    return obs


def run_overlay_sleeve(
    dest: Path,
    overlayed: pd.DataFrame,
    extra: dict[str, Any],
    *,
    resume: bool = True,
    base_nav: Path | None = None,
) -> dict[str, Any]:
    from .paths import assert_project_result_path
    from .round20.parity import _pack, _run_r7

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest / "report.json")
    code = str(extra.get("code") or dest.name)
    if resume and (dest / "summary.json").is_file() and (dest / "nav.parquet").is_file():
        cached = load_json(dest / "summary.json")
        if cached.get("code") == code and cached.get("status") == "ran":
            return cached
    frame = overlayed.copy()
    if "signal_date" not in frame.columns and "trade_date" in frame.columns:
        frame = frame.rename(columns={"trade_date": "signal_date"})
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.dropna(subset=["signal_date", "instrument_id", "score"])
    frame.to_parquet(dest / "overlayed.parquet", index=False)
    payload = {
        "fee_schedule": FEE_SCHEDULE,
        "validation_hooks": None,
        "historical_stamp_tax": False,
        "engine": "filtered v5 overlay → local_executor",
        "restarted_capital": False,
        "window": f"{MAIN_WINDOW_START}–{MAIN_WINDOW_END}",
        "compare_to": "R16_O01",
    }
    payload.update(extra)
    payload["code"] = code
    ran = _run_r7(dest, frame, payload)
    packed = _pack(ran, dest)
    obs = metrics_from_nav_file(dest / "nav.parquet")
    obs.update(
        {
            "code": code,
            "completed_trades": packed.get("completed_trades"),
            "status": "ran",
            "fee_schedule": FEE_SCHEDULE,
            "compare_to": "R16_O01",
            "restarted_capital": False,
            "scenario": bool(extra.get("scenario")),
            "round": extra.get("round"),
        }
    )
    gate = material_vs_v5(
        obs,
        top_changed=bool(extra.get("top_changed", True)),
        fee_same_direction=True,
        not_only_year=_yearly_not_only(dest / "nav.parquet", base_nav),
    )
    obs.update(gate)
    dump_json(dest / "summary.json", obs)
    return obs
