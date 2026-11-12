"""a-events：从信号 + C0/C1 账本重建 332/298/241/39。口径混用 = FAIL。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...round16.backtest import load_treat_dates
from ...round16.contract import O01 as R16_O01
from ..common import daily_from_nav, dump_json, write_md
from ..contract import EVENT_LAYERS
from .engine import o01_signal_dates, option_observable_dates
from .support import (
    EVENTS_JSON,
    idx_dates,
    load_nav_trades,
    require_c0,
    require_c1,
    resume_ok,
    skip_result,
    ta_md,
    top2,
    ymd_list,
)


def rebuild_layers() -> dict[str, Any]:
    c0 = require_c0()
    t0, n0 = load_nav_trades(c0)
    if t0 is None or n0 is None:
        raise FileNotFoundError("C0_LEDGER_MISSING")
    t1 = n1 = None
    try:
        t1, n1 = load_nav_trades(require_c1())
    except FileNotFoundError:
        t1 = n1 = None
    d0 = daily_from_nav(n0)
    nav_window = pd.DatetimeIndex(d0.index).normalize().unique().sort_values()
    all_signals = idx_dates(o01_signal_dates())
    treat_json = load_treat_dates()
    json_dates = idx_dates(treat_json.get(R16_O01) or [])
    observable = idx_dates(option_observable_dates())
    t2 = top2(t0)
    t2["signal_date"] = pd.to_datetime(t2["signal_date"]).dt.normalize()
    t2["target_name_budget"] = pd.to_numeric(t2["target_name_budget"], errors="coerce")
    t2["initial_quantity"] = pd.to_numeric(t2["initial_quantity"], errors="coerce")
    top2_days = idx_dates(t2["signal_date"])
    nonzero_days = idx_dates(t2.loc[t2["target_name_budget"] > 0, "signal_date"])
    fill_days = idx_dates(t2.loc[t2["initial_quantity"] > 0, "signal_date"])
    in_window = all_signals.intersection(nav_window)
    eligible = nonzero_days.intersection(observable).intersection(nav_window)
    planned_treat = eligible.intersection(in_window)

    changed = []
    if t1 is not None:
        o2 = top2(t1)
        o2["signal_date"] = pd.to_datetime(o2["signal_date"]).dt.normalize()
        o2["initial_quantity"] = pd.to_numeric(o2["initial_quantity"], errors="coerce")
        o2["target_name_budget"] = pd.to_numeric(o2["target_name_budget"], errors="coerce")
        for dt in in_window:
            v4_day = t2.loc[t2["signal_date"].eq(dt)]
            o_day = o2.loc[o2["signal_date"].eq(dt)]
            if v4_day.empty:
                continue
            v4_ids = set(v4_day["instrument_id"].astype(str))
            o_ids = set(o_day["instrument_id"].astype(str)) if len(o_day) else set()
            qty_v4 = float(v4_day["initial_quantity"].sum())
            qty_o = float(o_day["initial_quantity"].sum()) if len(o_day) else 0.0
            bud_v4 = float(v4_day["target_name_budget"].sum())
            bud_o = float(o_day["target_name_budget"].sum()) if len(o_day) else 0.0
            if (v4_ids != o_ids) or (abs(qty_v4 - qty_o) > 0.5) or (abs(bud_v4 - bud_o) > 1.0):
                changed.append(dt)
    changed_idx = idx_dates(changed)
    counts = {
        "raw_signal_incl_2015": int(len(all_signals)),
        "nav_window": int(len(in_window)),
        "eligible": int(len(eligible)),
        "actual_fill_change": int(len(changed_idx)) if t1 is not None else None,
        "planned_treat_eligible_intersect_signal": int(len(planned_treat)),
        "n_json_treat": int(len(json_dates)),
        "n_json_in_nav": int(json_dates.isin(nav_window).sum()) if len(json_dates) else 0,
        "n_v4_top2_days": int(len(top2_days)),
        "n_v4_top2_nonzero_budget": int(len(nonzero_days)),
        "n_v4_top2_actual_fills": int(len(fill_days)),
        "n_no_v4_top2_among_signals": int(len(in_window) - len(in_window.intersection(top2_days))),
        "n_signal_pre_2018": int((all_signals < nav_window.min()).sum()) if len(nav_window) else None,
    }
    expected = dict(EVENT_LAYERS)
    fill_ok = counts["actual_fill_change"] == expected["actual_fill_change"]
    layer_ok = (
        counts["raw_signal_incl_2015"] == expected["raw_signal_incl_2015"]
        and counts["nav_window"] == expected["nav_window"]
        and counts["eligible"] == expected["eligible"]
        and bool(fill_ok)
    )
    mix_fail = (
        counts["eligible"] == counts["actual_fill_change"]
        or counts["nav_window"] == counts["actual_fill_change"]
        or counts["raw_signal_incl_2015"] == counts["nav_window"]
    )
    # 241 vs 39 必须分层；332 vs 298 也必须分层
    return {
        "counts": counts,
        "expected": expected,
        "layer_ok": bool(layer_ok),
        "layer_mix_fail": bool(mix_fail and not layer_ok),
        "nav_window": [str(nav_window.min().date()), str(nav_window.max().date())] if len(nav_window) else [],
        "eligible_dates": ymd_list(eligible),
        "planned_treat_dates": ymd_list(planned_treat),
        "fill_change_dates": ymd_list(changed_idx),
        "treat_in_window": ymd_list(in_window),
        "all_signals": ymd_list(all_signals),
        "note": (
            "332=PUT_SKEW25 信号（含2015）；298=NAV窗内信号；241=决策时已知 eligible Top2∧预算>0∧期权可观测；"
            "39=实际成交变化。241 与 39 不是同一层。planned_treat=eligible∩信号，禁止用未来成交筛选。"
        ),
    }


def run_events(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    existing = resume_ok(EVENTS_JSON) if resume else None
    if existing is not None:
        return existing
    try:
        layers = rebuild_layers()
    except FileNotFoundError as exc:
        payload = skip_result("a-events", str(exc), status="BLOCKED")
        dump_json(EVENTS_JSON, payload)
        return payload
    except Exception as exc:
        payload = skip_result("a-events", f"{type(exc).__name__}:{exc}")
        dump_json(EVENTS_JSON, payload)
        return payload
    counts = layers["counts"]
    payload = {
        "status": "DONE" if layers["layer_ok"] else "FAIL",
        "passed": bool(layers["layer_ok"]),
        **layers,
        "frozen_target": dict(EVENT_LAYERS),
    }
    dump_json(EVENTS_JSON, payload)
    write_md(
        ta_md("events.md"),
        "\n".join(
            [
                "# Round19 Track A · a-events",
                "",
                f"结果：**{'PASS' if layers['layer_ok'] else 'FAIL'}**。",
                "",
                "| 层 | 重建 | 冻结 |",
                "|---|---:|---:|",
                f"| raw 含2015 | {counts['raw_signal_incl_2015']} | {EVENT_LAYERS['raw_signal_incl_2015']} |",
                f"| NAV 窗 | {counts['nav_window']} | {EVENT_LAYERS['nav_window']} |",
                f"| eligible | {counts['eligible']} | {EVENT_LAYERS['eligible']} |",
                f"| 实际成交变化 | {counts['actual_fill_change']} | {EVENT_LAYERS['actual_fill_change']} |",
                "",
                layers["note"],
                "",
            ]
        ),
    )
    return payload
