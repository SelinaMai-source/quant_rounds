"""v5 与 E2 重放的回撤归因。交易盈亏加总不等于复利净值差。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from core.research_integrity_v3.risk_metrics import (
    drawdown_episodes,
    drawdown_path,
    summarize_nav,
    top_nonoverlapping,
)


def _trade_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    frame = trades.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["sleeve_id"] = frame.get("sleeve_id", 0)
    proceeds = pd.to_numeric(frame.get("sell_proceeds"), errors="coerce").fillna(0.0)
    dividend = pd.to_numeric(frame.get("cash_dividend"), errors="coerce").fillna(0.0)
    cost = pd.to_numeric(frame.get("buy_cost"), errors="coerce").fillna(0.0)
    frame["cash_pnl"] = proceeds + dividend - cost
    frame["qty"] = pd.to_numeric(frame.get("initial_quantity"), errors="coerce")
    # 字符串键。用 tuple 做 Index 时 .loc[tuple] 会被当成多层索引。
    frame["key"] = (
        frame["signal_date"].dt.strftime("%Y-%m-%d")
        + "\t"
        + frame["instrument_id"].astype(str)
        + "\t"
        + frame["sleeve_id"].astype(str)
    )
    return frame


def classify_trades(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    left = _trade_pnl(baseline)
    right = _trade_pnl(candidate)
    lmap = left.set_index("key")
    rmap = right.set_index("key")
    keys = sorted(set(lmap.index) | set(rmap.index))
    rows = []
    for key in keys:
        in_l = key in lmap.index
        in_r = key in rmap.index
        if in_l and in_r:
            lrow = lmap.loc[key]
            rrow = rmap.loc[key]
            if isinstance(lrow, pd.DataFrame):
                lrow = lrow.iloc[0]
            if isinstance(rrow, pd.DataFrame):
                rrow = rrow.iloc[0]
            same_qty = np.isclose(float(lrow["qty"]), float(rrow["qty"]), rtol=0, atol=1e-6) if pd.notna(lrow["qty"]) and pd.notna(rrow["qty"]) else False
            kind = "common" if same_qty else "same_stock_different_qty"
            rows.append(
                {
                    "key": key,
                    "signal_date": key.split("\t")[0],
                    "instrument_id": key.split("\t")[1],
                    "sleeve_id": key.split("\t")[2],
                    "class": kind,
                    "baseline_cash_pnl": float(lrow["cash_pnl"]),
                    "candidate_cash_pnl": float(rrow["cash_pnl"]),
                    "baseline_qty": lrow["qty"],
                    "candidate_qty": rrow["qty"],
                }
            )
        elif in_l:
            lrow = lmap.loc[key]
            if isinstance(lrow, pd.DataFrame):
                lrow = lrow.iloc[0]
            rows.append(
                {
                    "key": key,
                    "signal_date": key.split("\t")[0],
                    "instrument_id": key.split("\t")[1],
                    "sleeve_id": key.split("\t")[2],
                    "class": "baseline_only",
                    "baseline_cash_pnl": float(lrow["cash_pnl"]),
                    "candidate_cash_pnl": 0.0,
                    "baseline_qty": lrow["qty"],
                    "candidate_qty": np.nan,
                }
            )
        else:
            rrow = rmap.loc[key]
            if isinstance(rrow, pd.DataFrame):
                rrow = rrow.iloc[0]
            rows.append(
                {
                    "key": key,
                    "signal_date": key.split("\t")[0],
                    "instrument_id": key.split("\t")[1],
                    "sleeve_id": key.split("\t")[2],
                    "class": "candidate_only",
                    "baseline_cash_pnl": 0.0,
                    "candidate_cash_pnl": float(rrow["cash_pnl"]),
                    "baseline_qty": np.nan,
                    "candidate_qty": rrow["qty"],
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # 同股不同槽位：同一信号日同一股票出现在不同 sleeve
    dup = out.groupby(["signal_date", "instrument_id"])["sleeve_id"].nunique()
    moved = {idx for idx, n in dup.items() if n > 1}
    if moved:
        mask = out.apply(lambda r: (r["signal_date"], r["instrument_id"]) in moved and r["class"] == "common", axis=1)
        out.loc[mask, "class"] = "same_stock_different_slot"
    return out


def _episode_trade_pnl(trades: pd.DataFrame, start: str, end: str) -> float:
    if trades is None or trades.empty:
        return 0.0
    frame = _trade_pnl(trades)
    exit_col = "actual_exit_date" if "actual_exit_date" in frame.columns else "signal_date"
    exits = pd.to_datetime(frame[exit_col])
    mask = exits.between(pd.Timestamp(start), pd.Timestamp(end))
    return float(frame.loc[mask, "cash_pnl"].sum())


def build_audit(
    baseline_nav: pd.DataFrame,
    candidate_nav: pd.DataFrame,
    baseline_trades: pd.DataFrame,
    candidate_trades: pd.DataFrame,
    *,
    baseline_costs: pd.DataFrame | None = None,
    candidate_costs: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    bpath = drawdown_path(baseline_nav)
    cpath = drawdown_path(candidate_nav)
    beps = top_nonoverlapping(drawdown_episodes(bpath), 3)
    ceps = top_nonoverlapping(drawdown_episodes(cpath), 3)
    rows = []
    for side, episodes, trades, other in (
        ("baseline_worst", beps, baseline_trades, candidate_trades),
        ("candidate_worst", ceps, candidate_trades, baseline_trades),
    ):
        for i, ep in enumerate(episodes, start=1):
            end = ep["recovery_date"] or ep["trough_date"]
            own = _episode_trade_pnl(trades, ep["peak_date"], end)
            other_pnl = _episode_trade_pnl(other, ep["peak_date"], end)
            bnav = bpath.loc[bpath["trade_date"].between(ep["peak_date"], end), "nav"]
            cnav = cpath.loc[cpath["trade_date"].between(ep["peak_date"], end), "nav"]
            nav_delta = float(cnav.iloc[-1] - cnav.iloc[0]) - float(bnav.iloc[-1] - bnav.iloc[0]) if len(bnav) and len(cnav) else None
            rows.append(
                {
                    "episode_set": side,
                    "rank": i,
                    **ep,
                    "baseline_trade_cash_pnl": own if side.startswith("baseline") else other_pnl,
                    "candidate_trade_cash_pnl": other_pnl if side.startswith("baseline") else own,
                    "trade_cash_pnl_diff": (other_pnl - own) if side.startswith("baseline") else (own - other_pnl),
                    "nav_path_delta_diff": nav_delta,
                    "trade_pnl_is_not_compounded_nav_delta": True,
                }
            )
    audit = pd.DataFrame(rows)
    classes = classify_trades(baseline_trades, candidate_trades)
    bsum = summarize_nav(baseline_nav)
    csum = summarize_nav(candidate_nav)
    fee_b = _fee_sum(baseline_costs)
    fee_c = _fee_sum(candidate_costs)
    nav_b = float(bpath["nav"].iloc[-1])
    nav_c = float(cpath["nav"].iloc[-1])
    bridge = pd.DataFrame(
        [
            {"item": "terminal_nav_delta", "value": nav_c - nav_b, "note": "复利净值差"},
            {"item": "sum_trade_cash_pnl_diff", "value": float(classes["candidate_cash_pnl"].sum() - classes["baseline_cash_pnl"].sum()) if len(classes) else 0.0, "note": "交易现金盈亏加总，不等于复利净值差"},
            {"item": "fee_total_baseline", "value": fee_b, "note": "成本表合计"},
            {"item": "fee_total_candidate", "value": fee_c, "note": "成本表合计"},
            {"item": "cagr_baseline", "value": bsum["cagr_365_25"], "note": "365.25"},
            {"item": "cagr_candidate", "value": csum["cagr_365_25"], "note": "365.25"},
            {"item": "ulcer_baseline", "value": bsum["ulcer_index"], "note": ""},
            {"item": "ulcer_candidate", "value": csum["ulcer_index"], "note": ""},
            {"item": "maxdd_baseline", "value": bsum["max_drawdown"], "note": ""},
            {"item": "maxdd_candidate", "value": csum["max_drawdown"], "note": ""},
            {"item": "es95_baseline", "value": bsum["es95"], "note": "最差5%日收益的负均值"},
            {"item": "es95_candidate", "value": csum["es95"], "note": ""},
            {"item": "window63_worst10_baseline", "value": bsum["window63_worst10pct_mean"], "note": ""},
            {"item": "window63_worst10_candidate", "value": csum["window63_worst10pct_mean"], "note": ""},
            {"item": "class_counts", "value": json.dumps(classes["class"].value_counts().to_dict() if len(classes) else {}, ensure_ascii=False), "note": ""},
        ]
    )
    meta = {"baseline": bsum, "candidate": csum, "class_counts": classes["class"].value_counts().to_dict() if len(classes) else {}}
    return audit, bridge, {"classes": classes, "meta": meta, "baseline_path": bpath, "candidate_path": cpath}


def _fee_sum(costs: pd.DataFrame | None) -> float | None:
    if costs is None or costs.empty:
        return None
    cols = [c for c in ("commission", "transfer_fee", "stamp_tax", "buy_commission", "sell_commission") if c in costs.columns]
    if not cols:
        return None
    return float(costs[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum().sum())


def yearly_table(nav: pd.DataFrame) -> pd.DataFrame:
    path = drawdown_path(nav)
    path["year"] = path["trade_date"].dt.year
    rows = []
    for year, block in path.groupby("year"):
        ret = block["nav"].pct_change().dropna()
        rows.append(
            {
                "year": int(year),
                "n_days": int(len(block)),
                "compound": float(block["nav"].iloc[-1] / block["nav"].iloc[0] - 1.0) if len(block) else None,
                "ulcer_index": float(((block["drawdown"] ** 2).mean()) ** 0.5),
                "max_drawdown": float(-block["drawdown"].max()),
            }
        )
    return pd.DataFrame(rows)
