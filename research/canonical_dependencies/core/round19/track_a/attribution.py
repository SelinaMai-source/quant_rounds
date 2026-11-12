"""a-attribution：39 日经济归因。权威增量 = 全引擎期末 NAV 差。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...round16_5.common import nav_end
from ...round16_5.contract import FULL_ENGINE_NAV_DIFFERENCE
from ..common import daily_from_nav, dump_json, write_md
from ..contract import NON_ADDITIVE_PATH_DEPENDENCE
from .events import rebuild_layers
from .support import (
    ATTR_CSV,
    ATTR_JSON,
    EVENTS_JSON,
    load_json,
    load_nav_trades,
    pack_daily,
    pack_obs_from_dir,
    require_c0,
    require_c1,
    resume_ok,
    skip_result,
    ta_md,
    top2,
)


def run_attribution(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    existing = resume_ok(ATTR_JSON) if resume else None
    if existing is not None:
        return existing
    try:
        c0 = require_c0()
        c1 = require_c1()
        t0, n0 = load_nav_trades(c0)
        t1, n1 = load_nav_trades(c1)
        layers = load_json(EVENTS_JSON) or rebuild_layers()
    except FileNotFoundError as exc:
        payload = skip_result("a-attribution", str(exc), status="BLOCKED")
        dump_json(ATTR_JSON, payload)
        return payload
    if t0 is None or n0 is None or t1 is None or n1 is None:
        payload = skip_result("a-attribution", "LEDGER_MISSING", status="BLOCKED")
        dump_json(ATTR_JSON, payload)
        return payload

    d0 = daily_from_nav(n0)
    d1 = daily_from_nav(n1)
    aligned = pd.concat({"c0": d0, "c1": d1}, axis=1).dropna()
    v4 = top2(t0)
    o01 = top2(t1)
    v4["signal_date"] = pd.to_datetime(v4["signal_date"]).dt.normalize()
    o01["signal_date"] = pd.to_datetime(o01["signal_date"]).dt.normalize()
    fill_days = pd.DatetimeIndex(pd.to_datetime(layers.get("fill_change_dates") or [])).normalize()
    rows = []
    for dt in fill_days:
        v4_day = v4.loc[v4["signal_date"].eq(dt)]
        o_day = o01.loc[o01["signal_date"].eq(dt)]
        net_v4 = float(pd.to_numeric(v4_day.get("net"), errors="coerce").sum()) if len(v4_day) else 0.0
        rows.append(
            {
                "signal_date": str(dt.date()),
                "year": int(dt.year),
                "v4_instrument": ",".join(sorted(v4_day["instrument_id"].astype(str))) if len(v4_day) else "",
                "o01_instrument": ",".join(sorted(o_day["instrument_id"].astype(str))) if len(o_day) else "",
                "v4_qty": float(pd.to_numeric(v4_day["initial_quantity"], errors="coerce").sum()) if len(v4_day) else 0.0,
                "o01_qty": float(pd.to_numeric(o_day["initial_quantity"], errors="coerce").sum()) if len(o_day) else 0.0,
                "v4_budget": float(pd.to_numeric(v4_day["target_name_budget"], errors="coerce").sum()) if len(v4_day) else 0.0,
                "o01_budget": float(pd.to_numeric(o_day["target_name_budget"], errors="coerce").sum()) if len(o_day) else 0.0,
                "v4_net": net_v4,
                "avoided": True,
            }
        )
    attr_df = pd.DataFrame(rows)
    ATTR_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not attr_df.empty:
        attr_df.to_csv(ATTR_CSV, index=False)
    pack0 = pack_obs_from_dir(c0)
    pack1 = pack_obs_from_dir(c1)
    end0 = nav_end(n0)
    end1 = nav_end(n1)
    nav_diff = None if (end0 is None or end1 is None) else float(end1 - end0)
    sum_avoided = float(attr_df["v4_net"].sum()) if not attr_df.empty else 0.0
    yearly = []
    for y in range(2018, 2027):
        sub = attr_df.loc[attr_df["year"].eq(y)] if not attr_df.empty else attr_df
        dsub = aligned.loc[aligned.index.year == y]
        yearly.append(
            {
                "year": y,
                "n_changed": int(len(sub)),
                "sum_avoided_net": float(sub["v4_net"].sum()) if not sub.empty else 0.0,
                "c0": pack_daily(dsub["c0"]) if len(dsub) else None,
                "c1": pack_daily(dsub["c1"]) if len(dsub) else None,
            }
        )
    payload = {
        "status": "DONE",
        "authoritative_increment_kind": FULL_ENGINE_NAV_DIFFERENCE,
        "path_dependence": NON_ADDITIVE_PATH_DEPENDENCE,
        "c0": pack0,
        "c1": pack1,
        "nav_end_c0": end0,
        "nav_end_c1": end1,
        "nav_end_difference": nav_diff,
        "cagr_increment": (pack1.get("cagr") or 0) - (pack0.get("cagr") or 0),
        "sharpe_increment": (pack1.get("sharpe") or 0) - (pack0.get("sharpe") or 0),
        "maxdd_increment": (pack1.get("max_drawdown") or 0) - (pack0.get("max_drawdown") or 0),
        "n_fill_change": int(len(attr_df)),
        "sum_avoided_trade_pnl": sum_avoided,
        "sum_avoided_is_not_authoritative": True,
        "yearly": yearly,
        "fill_change_dates": layers.get("fill_change_dates") or ymd_from_df(attr_df),
        "csv": str(ATTR_CSV),
        "note": "权威增量是 NAV 路径差，不是避开交易 PnL 之和。",
    }
    dump_json(ATTR_JSON, payload)
    write_md(
        ta_md("attribution.md"),
        "\n".join(
            [
                "# Round19 Track A · a-attribution",
                "",
                f"权威增量 `{FULL_ENGINE_NAV_DIFFERENCE}` 期末 NAV 差=`{nav_diff}`。",
                f"CAGR 增量=`{payload['cagr_increment']}` Sharpe 增量=`{payload['sharpe_increment']}`。",
                f"避开交易 PnL 之和=`{sum_avoided}`（**不是**权威增量）。`{NON_ADDITIVE_PATH_DEPENDENCE}`。",
                f"成交变化日 N=`{len(attr_df)}` CSV=`{ATTR_CSV}`。",
                "",
            ]
        ),
    )
    return payload


def ymd_from_df(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "signal_date" not in df.columns:
        return []
    return [str(x) for x in df["signal_date"].tolist()]
