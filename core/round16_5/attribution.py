"""332 vs 298 对齐、交易级 O01 归因、权威 FULL_ENGINE_NAV_DIFFERENCE。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_5_ROOT
from ..round16.backtest import load_treat_dates
from .common import daily_from_nav, dump_json, load_json, nav_end, pack_daily, pct, write_md
from .contract import CONCENTRATED_INCREMENT_RISK, FULL_ENGINE_NAV_DIFFERENCE, O01_ROUND16, V00, V01, parity_dir
from .engine import o01_signal_dates, option_observable_dates, with_rank
from .freeze import require_frozen

ATTR_JSON = ROUND16_5_ROOT / "attribution" / "attribution.json"
ATTR_MD = ROUND16_5_ROOT / "01_O01_trade_attribution.md"
REC_MD = ROUND16_5_ROOT / "02_event_count_reconciliation.md"
CSV_PATH = ROUND16_5_ROOT / "attribution" / "O01_treated_trades.csv"


def _load_pair() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t0 = pd.read_parquet(parity_dir(V00) / "trades.parquet")
    n0 = pd.read_parquet(parity_dir(V00) / "nav.parquet")
    t1 = pd.read_parquet(parity_dir(V01) / "trades.parquet")
    n1 = pd.read_parquet(parity_dir(V01) / "nav.parquet")
    return t0, n0, t1, n1


def _top2(trades: pd.DataFrame) -> pd.DataFrame:
    t = with_rank(trades)
    return t.loc[t["_rank"].eq(2)].copy()


def _daily_nav_diff(n0: pd.DataFrame, n1: pd.DataFrame) -> pd.Series:
    d0 = daily_from_nav(n0)
    d1 = daily_from_nav(n1)
    return (d1 - d0).dropna()


def run_attribution(**_: Any) -> dict[str, Any]:
    require_frozen()
    t0, n0, t1, n1 = _load_pair()
    d0 = daily_from_nav(n0)
    d1 = daily_from_nav(n1)
    aligned = pd.concat({"v00": d0, "v01": d1}, axis=1).dropna()
    nav_window = pd.DatetimeIndex(aligned.index)
    nav_start = nav_window.min()
    nav_end_dt = nav_window.max()

    all_signals = pd.DatetimeIndex(pd.to_datetime(o01_signal_dates()).normalize()).unique().sort_values()
    treat_json = load_treat_dates()
    json_dates = pd.DatetimeIndex(pd.to_datetime(treat_json.get(O01_ROUND16) or [])).normalize().unique().sort_values()
    observable = option_observable_dates()

    top2_v4 = _top2(t0)
    top2_v4["signal_date"] = pd.to_datetime(top2_v4["signal_date"]).dt.normalize()
    top2_days = pd.DatetimeIndex(top2_v4["signal_date"].unique()).sort_values()
    nonzero_budget = top2_v4.loc[pd.to_numeric(top2_v4["target_name_budget"], errors="coerce") > 0]
    nonzero_days = pd.DatetimeIndex(nonzero_budget["signal_date"].unique()).sort_values()
    filled = top2_v4.loc[pd.to_numeric(top2_v4["initial_quantity"], errors="coerce") > 0]
    fill_days = pd.DatetimeIndex(filled["signal_date"].unique()).sort_values()

    in_window = all_signals.intersection(nav_window)
    eligible = nonzero_days.intersection(observable).intersection(nav_window)

    top2_o01 = _top2(t1)
    top2_o01["signal_date"] = pd.to_datetime(top2_o01["signal_date"]).dt.normalize()

    v4_key = top2_v4.set_index(["signal_date", "instrument_id"])
    o01_key = top2_o01.set_index(["signal_date", "instrument_id"]) if len(top2_o01) else pd.DataFrame()

    changed_rows = []
    avoided = []
    no_effect_days = []
    treat_in_window = in_window
    for dt in treat_in_window:
        v4_day = top2_v4.loc[top2_v4["signal_date"].eq(dt)]
        o01_day = top2_o01.loc[top2_o01["signal_date"].eq(dt)] if len(top2_o01) else v4_day.iloc[0:0]
        if v4_day.empty:
            no_effect_days.append({"signal_date": str(dt.date()), "reason": "no_v4_top2_fill"})
            continue
        v4_ids = set(v4_day["instrument_id"].astype(str))
        o01_ids = set(o01_day["instrument_id"].astype(str)) if len(o01_day) else set()
        qty_v4 = float(pd.to_numeric(v4_day["initial_quantity"], errors="coerce").sum())
        qty_o01 = float(pd.to_numeric(o01_day["initial_quantity"], errors="coerce").sum()) if len(o01_day) else 0.0
        bud_v4 = float(pd.to_numeric(v4_day["target_name_budget"], errors="coerce").sum())
        bud_o01 = float(pd.to_numeric(o01_day["target_name_budget"], errors="coerce").sum()) if len(o01_day) else 0.0
        changed = (v4_ids != o01_ids) or (abs(qty_v4 - qty_o01) > 0.5) or (abs(bud_v4 - bud_o01) > 1.0)
        net_v4 = float(pd.to_numeric(v4_day["net"], errors="coerce").sum())
        row = {
            "signal_date": str(dt.date()),
            "year": int(dt.year),
            "v4_instrument": ",".join(sorted(v4_ids)),
            "o01_instrument": ",".join(sorted(o01_ids)) if o01_ids else "",
            "v4_qty": qty_v4,
            "o01_qty": qty_o01,
            "v4_budget": bud_v4,
            "o01_budget": bud_o01,
            "v4_net": net_v4,
            "avoided": bool(changed and qty_o01 < qty_v4),
            "changed_fill": bool(changed),
            "no_effect": not changed,
        }
        changed_rows.append(row)
        if not changed:
            no_effect_days.append({"signal_date": str(dt.date()), "reason": "fill_unchanged"})
        elif row["avoided"]:
            avoided.append(row)

    attr_df = pd.DataFrame(changed_rows)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not attr_df.empty:
        attr_df.to_csv(CSV_PATH, index=False)

    n_changed = int(attr_df["changed_fill"].sum()) if not attr_df.empty else 0
    n_avoided = int(attr_df["avoided"].sum()) if not attr_df.empty else 0
    n_unchanged = int(attr_df["no_effect"].sum()) if not attr_df.empty else 0
    n_no_top2 = int(len(no_effect_days) - n_unchanged) if no_effect_days else (int(len(treat_in_window) - len(attr_df)) if not attr_df.empty else int(len(treat_in_window)))
    n_no_effect = int(n_unchanged + n_no_top2)
    sum_avoided = float(attr_df.loc[attr_df["avoided"], "v4_net"].sum()) if n_avoided else 0.0

    end0 = nav_end(n0)
    end1 = nav_end(n1)
    nav_diff = None if (end0 is None or end1 is None) else float(end1 - end0)
    pack0 = pack_daily(d0)
    pack1 = pack_daily(d1)
    s0 = load_json(parity_dir(V00) / "summary.json")
    s1 = load_json(parity_dir(V01) / "summary.json")
    if s0.get("cagr") is not None:
        pack0 = {**pack0, **{k: s0.get(k) for k in ("cagr", "sharpe", "max_drawdown", "completed_trades") if s0.get(k) is not None}}
    if s1.get("cagr") is not None:
        pack1 = {**pack1, **{k: s1.get(k) for k in ("cagr", "sharpe", "max_drawdown", "completed_trades") if s1.get(k) is not None}}

    diff = aligned["v01"] - aligned["v00"]
    contrib = diff.sort_values()
    top_pos = contrib.sort_values(ascending=False)
    conc = {}
    for k in (1, 3, 5, 10):
        topk = top_pos.head(k)
        conc[str(k)] = {
            "dates": [str(x.date()) for x in topk.index],
            "sum_daily_diff": float(topk.sum()),
            "share_of_total_diff": float(topk.sum() / diff.sum()) if abs(float(diff.sum())) > 1e-12 else None,
        }

    def _zero_days(days) -> dict[str, Any]:
        adj = aligned.copy()
        idx = pd.DatetimeIndex(pd.to_datetime(list(days))).normalize()
        mask = adj.index.isin(idx)
        adj.loc[mask, "v01"] = adj.loc[mask, "v00"]
        from .common import cagr_of, sharpe_of, max_drawdown_of

        return {
            "cagr_v01_adj": cagr_of(adj["v01"]),
            "sharpe_v01_adj": sharpe_of(adj["v01"]),
            "maxdd_v01_adj": max_drawdown_of(adj["v01"]),
            "cagr_inc_adj": (cagr_of(adj["v01"]) or 0) - (cagr_of(adj["v00"]) or 0),
            "n_zeroed": int(mask.sum()),
        }

    leave_one = {}
    for dt in top_pos.head(10).index:
        leave_one[str(dt.date())] = _zero_days([dt])

    ex = {
        "ex2024": _zero_days(aligned.index[aligned.index.year == 2024]),
        "ex2025": _zero_days(aligned.index[aligned.index.year == 2025]),
        "ex2024_2025": _zero_days(aligned.index[aligned.index.year.isin({2024, 2025})]),
    }
    vanished = False
    inc0 = (pack1.get("cagr") or 0) - (pack0.get("cagr") or 0)
    for name, rec in [("top1", conc["1"]), ("ex2024", ex["ex2024"]), ("ex2025", ex["ex2025"])]:
        adj_inc = rec.get("cagr_inc_adj") if "cagr_inc_adj" in rec else None
        if adj_inc is not None and inc0 > 0 and adj_inc <= 0:
            vanished = True
        if name == "top1" and rec.get("share_of_total_diff") is not None and abs(rec["share_of_total_diff"]) > 0.5:
            vanished = True
    if leave_one:
        first = next(iter(leave_one.values()))
        if inc0 > 0 and (first.get("cagr_inc_adj") or 0) <= 0:
            vanished = True

    yearly = []
    for y in range(2018, 2027):
        sub = attr_df.loc[attr_df["year"].eq(y)] if not attr_df.empty else attr_df
        dsub = aligned.loc[aligned.index.year == y]
        yearly.append({
            "year": y,
            "n_signal_in_window": int((in_window.year == y).sum()),
            "n_changed": int(sub["changed_fill"].sum()) if not sub.empty else 0,
            "n_avoided": int(sub["avoided"].sum()) if not sub.empty else 0,
            "sum_avoided_net": float(sub.loc[sub["avoided"], "v4_net"].sum()) if not sub.empty else 0.0,
            "cagr_v00": pack_daily(dsub["v00"]).get("cagr") if len(dsub) else None,
            "cagr_v01": pack_daily(dsub["v01"]).get("cagr") if len(dsub) else None,
            "sharpe_v00": pack_daily(dsub["v00"]).get("sharpe") if len(dsub) else None,
            "sharpe_v01": pack_daily(dsub["v01"]).get("sharpe") if len(dsub) else None,
        })

    # eligible vs json 298
    n_json = int(len(json_dates))
    n_json_in_nav = int(json_dates.isin(nav_window).sum())
    recon = {
        "n_signal_all_including_2015": int(len(all_signals)),
        "n_json_treat_dates": n_json,
        "n_json_in_nav_window": n_json_in_nav,
        "n_signal_in_backtest_window": int(len(in_window)),
        "n_option_observable_in_window": int(observable.isin(nav_window).sum()) if len(observable) else 0,
        "n_v4_top2_days": int(len(top2_days)),
        "n_v4_top2_nonzero_budget": int(len(nonzero_days)),
        "n_v4_top2_actual_fills": int(len(fill_days)),
        "n_eligible_top2_nonzero_budget_observable": int(len(eligible)),
        "n_treat_in_window": int(len(treat_in_window)),
        "n_signal_in_window_with_v4_top2": int(len(attr_df)) if not attr_df.empty else 0,
        "n_changed_fills": n_changed,
        "n_avoided_trades": n_avoided,
        "n_unchanged_fills": n_unchanged,
        "n_no_v4_top2_among_signals": n_no_top2,
        "n_no_effect_days": n_no_effect,
        "n_signal_pre_2018": int((all_signals < nav_start).sum()),
        "nav_window": [str(nav_start.date()), str(nav_end_dt.date())],
        "note_332_vs_298": (
            "332 = PUT_SKEW25 信号日（含 2015 起，options/treat_dates.json）。"
            "298 = 与 2018+ NAV 对齐后的处理日计数（Round16 controls n_treat_days）。"
        ),
    }

    payload = {
        "status": "DONE",
        "reconciliation": recon,
        "authoritative_increment_kind": FULL_ENGINE_NAV_DIFFERENCE,
        "v00": pack0,
        "v01": pack1,
        "nav_end_v00": end0,
        "nav_end_v01": end1,
        "nav_end_difference": nav_diff,
        "cagr_increment": (pack1.get("cagr") or 0) - (pack0.get("cagr") or 0),
        "sharpe_increment": (pack1.get("sharpe") or 0) - (pack0.get("sharpe") or 0),
        "maxdd_increment": (pack1.get("max_drawdown") or 0) - (pack0.get("max_drawdown") or 0),
        "sum_avoided_trade_pnl": sum_avoided,
        "sum_avoided_is_not_authoritative": True,
        "concentration": conc,
        "leave_one_event_top10_daily_diff": leave_one,
        "ex_years": ex,
        "yearly": yearly,
        "concentrated_flag": CONCENTRATED_INCREMENT_RISK if vanished else None,
        "csv": str(CSV_PATH),
        "eligible_dates": [str(x.date()) for x in eligible],
        "treat_in_window": [str(x.date()) for x in treat_in_window],
        "n_v4_top2_trades": int(len(top2_v4)),
        "n_v01_top2_trades": int(len(top2_o01)),
        "n_v00_trades": int(len(t0)),
        "n_v01_trades": int(len(t1)),
    }
    dump_json(ATTR_JSON, payload)
    write_md(
        ATTR_MD,
        "\n".join([
            "# Round 16.5 · 01 O01 交易归因",
            "",
            f"权威增量 = **{FULL_ENGINE_NAV_DIFFERENCE}** = `{nav_diff}`（期末 NAV）。",
            f"CAGR 增量=`{payload['cagr_increment']}` Sharpe 增量=`{payload['sharpe_increment']}` MaxDD 增量=`{payload['maxdd_increment']}`。",
            f"避开交易 PnL 之和=`{sum_avoided}`（**不是**权威增量）。",
            f"改变成交日=`{n_changed}` 避开=`{n_avoided}` 无效果=`{n_no_effect}`。",
            f"集中度标记=`{payload['concentrated_flag']}`。",
            f"CSV=`{CSV_PATH}`。",
            "",
            "分年见 attribution.json yearly。Leave-one-event 用日收益差置零（路径诊断）；权威数字仍来自全引擎 NAV。",
            "",
        ]),
    )
    rec_lines = [
        "# Round 16.5 · 02 事件计数对齐",
        "",
        "| 层 | N |",
        "|---|---:|",
        f"| 信号日（含 2015，treat_dates.json / `_signals`） | {recon['n_signal_all_including_2015']} |",
        f"| JSON treat_dates | {recon['n_json_treat_dates']} |",
        f"| 回测窗内信号日（2018+ NAV） | {recon['n_signal_in_backtest_window']} |",
        f"| JSON ∩ NAV 窗（应对齐 Round16 n_treat_days=298） | {recon['n_json_in_nav_window']} |",
        f"| 2015–2017 信号日 | {recon['n_signal_pre_2018']} |",
        f"| v4 有 Top2 日 | {recon['n_v4_top2_days']} |",
        f"| v4 Top2 且预算>0 | {recon['n_v4_top2_nonzero_budget']} |",
        f"| v4 Top2 实际成交 | {recon['n_v4_top2_actual_fills']} |",
        f"| 合格日（Top2∧预算>0∧期权可观测） | {recon['n_eligible_top2_nonzero_budget_observable']} |",
        f"| 窗内信号 ∩ v4 Top2 | {recon.get('n_signal_in_window_with_v4_top2')} |",
        f"| 实际改变成交 | {recon['n_changed_fills']} |",
        f"| 避开交易 | {recon['n_avoided_trades']} |",
        f"| 窗内信号但无 v4 Top2 | {recon.get('n_no_v4_top2_among_signals')} |",
        f"| 成交未变 | {recon.get('n_unchanged_fills')} |",
        f"| 无效果日（无Top2+未变） | {recon['n_no_effect_days']} |",
        "",
        recon["note_332_vs_298"],
        "",
        f"NAV 窗 `{recon['nav_window']}`。",
        "",
    ]
    write_md(REC_MD, "\n".join(rec_lines))
    return payload


def load_attribution() -> dict[str, Any]:
    return load_json(ATTR_JSON)
