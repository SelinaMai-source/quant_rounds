"""O01–O04 正式回测。复用 run_r15_backtest / apply_top2_veto / apply_budget_scale。禁止玩具 NAV。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_ROOT
from ..round12.runner import hist_hooks
from ..round15.runner import apply_budget_scale, apply_top2_veto, build_version_overlayed, load_market, run_r15_backtest
from .common import dump_json, load_json, observed_from_report, pct, write_md
from .contract import (
    INCONCLUSIVE_LOW_EVENT_COUNT,
    O01,
    O02,
    O02_HALVE,
    O03,
    O04,
    OPTION_IDS,
    PCT_MIN_OBS,
    PCT_WINDOW,
    SKIP_FACTOR_FOR_TRADING,
    experiment_dir,
    overlayed_path,
)
from .factors import load_factor_daily, load_factors
from .freeze import load_skips, require_frozen

BT_JSON = ROUND16_ROOT / "options" / "options.json"
BT_MD = ROUND16_ROOT / "08_experiment_results.md"
MIN_EVENTS = 20


def _signals(col: str, q: float = 0.90) -> pd.DatetimeIndex:
    daily = load_factor_daily()
    if daily.empty or col not in daily.columns:
        return pd.DatetimeIndex([])
    s = pd.to_numeric(daily.set_index(pd.to_datetime(daily["trade_date"]).dt.normalize())[col], errors="coerce").sort_index()
    past = s.shift(1)
    past_q = past.rolling(PCT_WINDOW, min_periods=PCT_MIN_OBS).quantile(q)
    hit = past > past_q
    return pd.DatetimeIndex(hit.index[hit.fillna(False)])


def _iv_shock_dates() -> pd.DatetimeIndex:
    daily = load_factor_daily()
    if daily.empty or "ATM_IV30" not in daily.columns:
        return pd.DatetimeIndex([])
    iv = pd.to_numeric(daily.set_index(pd.to_datetime(daily["trade_date"]).dt.normalize())["ATM_IV30"], errors="coerce").sort_index()
    shock = iv.diff(5)
    past = shock.shift(1)
    past_q = past.rolling(PCT_WINDOW, min_periods=PCT_MIN_OBS).quantile(0.90)
    hit = past > past_q
    return pd.DatetimeIndex(hit.index[hit.fillna(False)])


def _joint_dates() -> pd.DatetimeIndex:
    a = set(_signals("PUT_SKEW25"))
    b = set(_signals("PUT_WING_CONVEXITY"))
    return pd.DatetimeIndex(sorted(a & b))


def _summarize(ran: dict[str, Any]) -> dict[str, Any]:
    report = ran.get("report") or {}
    obs = observed_from_report(report)
    obs.update({"code": ran.get("code"), "status": ran.get("status"), "out_dir": ran.get("out_dir")})
    return obs


def _run(code: str, ov: pd.DataFrame, dest, *, resume: bool, loaded) -> dict[str, Any]:
    path = overlayed_path(code)
    path.parent.mkdir(parents=True, exist_ok=True)
    ov.to_parquet(path, index=False)
    ran = run_r15_backtest(
        dest,
        overlayed=ov,
        validation_hooks=hist_hooks(),
        extra_report={"fee_schedule": "HISTORICAL_FEE_CORRECTED", "candidate": code, "family": "option", "round": 16},
        resume=resume,
        prepared_market=loaded["prepared"],
        market_meta=loaded,
    )
    out = _summarize(ran)
    out["nav_path"] = str(dest / "nav.parquet")
    return out


def _skip_row(code: str, reason: str) -> dict[str, Any]:
    return {"code": code, "status": reason, "skip": True, "cagr": None, "sharpe": None, "max_drawdown": None}


def run_backtest(*, resume: bool = True, experiment: str = "", **_: Any) -> dict[str, Any]:
    require_frozen()
    skips = load_skips()
    fac = load_factors()
    pit_fail = ((fac.get("pit") or {}).get("status") == "FAIL")
    ov4, _, _ = build_version_overlayed("v4")
    loaded = load_market()
    want = [experiment] if experiment and experiment in OPTION_IDS else list(OPTION_IDS)
    results: dict[str, Any] = {}
    treat: dict[str, Any] = {}

    def maybe_skip(code, flag, label):
        if pit_fail:
            results[code] = _skip_row(code, "STOP_OPTION_BACKTEST_PIT_FAIL")
            return True
        if skips.get(flag, True) or fac.get("skip"):
            results[code] = _skip_row(code, label)
            return True
        return False

    if O01 in want:
        if not maybe_skip(O01, "SKIP_O01", "SKIP_O01"):
            dates = _signals("PUT_SKEW25")
            treat[O01] = [str(x.date()) for x in dates]
            if len(dates) < MIN_EVENTS:
                row = _skip_row(O01, INCONCLUSIVE_LOW_EVENT_COUNT)
                row["n_events"] = int(len(dates))
                results[O01] = row
            else:
                ov = apply_top2_veto(ov4, dates)
                row = _run(O01, ov, experiment_dir(O01), resume=resume, loaded=loaded)
                row["n_events"] = int(len(dates))
                row["skip"] = False
                results[O01] = row

    if O02 in want:
        if not maybe_skip(O02, "SKIP_O02", "SKIP_O02"):
            dates = _iv_shock_dates()
            treat[O02] = [str(x.date()) for x in dates]
            if len(dates) < MIN_EVENTS:
                row = _skip_row(O02, INCONCLUSIVE_LOW_EVENT_COUNT)
                row["n_events"] = int(len(dates))
                results[O02] = row
            else:
                ov = apply_budget_scale(ov4, dates, O02_HALVE)
                row = _run(O02, ov, experiment_dir(O02), resume=resume, loaded=loaded)
                row["n_events"] = int(len(dates))
                row["skip"] = False
                results[O02] = row

    if O03 in want:
        if not maybe_skip(O03, "SKIP_O03", "SKIP_O03"):
            dates = _joint_dates()
            treat[O03] = [str(x.date()) for x in dates]
            if len(dates) < MIN_EVENTS:
                row = _skip_row(O03, INCONCLUSIVE_LOW_EVENT_COUNT)
                row["n_events"] = int(len(dates))
                results[O03] = row
            else:
                ov = apply_top2_veto(ov4, dates)
                row = _run(O03, ov, experiment_dir(O03), resume=resume, loaded=loaded)
                row["n_events"] = int(len(dates))
                row["skip"] = False
                results[O03] = row

    if O04 in want:
        # 无横截面差异化 → SKIP_O04。不重训 XGB。
        results[O04] = _skip_row(O04, "SKIP_O04")
        results[O04]["reason"] = "no_cross_sectional_option_exposure_differentiation_on_same_day_universe"

    dump_json(ROUND16_ROOT / "options" / "treat_dates.json", treat)
    payload = {
        "status": "DONE",
        "results": results,
        "treat_dates": {k: len(v) for k, v in treat.items()},
        "engine": "run_r15_backtest+HISTORICAL_FEE_CORRECTED+B03",
        "toy_nav": False,
        "no_O05": True,
    }
    dump_json(BT_JSON, payload)
    lines = [
        "# Round 16 · 08 实验组结果",
        "",
        "Parent=baseline_v4。禁止 O05+。正式引擎 `run_r15_backtest`。",
        "",
        "| ID | status | n_events | CAGR | Sharpe | MaxDD | 笔数 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for code in OPTION_IDS:
        row = results.get(code) or {}
        lines.append(
            f"| `{code}` | {row.get('status')} | {row.get('n_events')} | {pct(row.get('cagr'))} | "
            f"{row.get('sharpe')} | {pct(row.get('max_drawdown'))} | {row.get('completed_trades')} |"
        )
    lines += ["", "触发过少 = INCONCLUSIVE_LOW_EVENT_COUNT。PIT FAIL 不得进回测。", ""]
    write_md(BT_MD, "\n".join(lines))
    return payload


def load_backtest() -> dict[str, Any]:
    return load_json(BT_JSON)


def load_treat_dates() -> dict[str, list[str]]:
    return load_json(ROUND16_ROOT / "options" / "treat_dates.json")
