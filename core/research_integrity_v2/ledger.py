"""账本谱系与 NAV 对账。容差 max(0.01 RMB, 1e-8*NAV)。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def nav_tolerance(nav: float) -> float:
    return max(0.01, 1e-8 * abs(float(nav)))


def reconcile_nav(nav: pd.DataFrame, *, cash_col: str = "cash", mv_col: str = "market_value", total_col: str = "total_asset") -> dict[str, Any]:
    work = nav.copy()
    gap = (work[cash_col] + work[mv_col] - work[total_col]).abs()
    end_nav = float(work[total_col].iloc[-1])
    tol = nav_tolerance(end_nav)
    max_gap = float(gap.max()) if len(gap) else None
    ok = max_gap is not None and max_gap <= tol
    return {
        "cash_plus_mv_equals_total": bool(ok),
        "max_identity_gap": max_gap,
        "tolerance": tol,
        "tolerance_rule": "max(0.01 RMB, 1e-8 * NAV)",
        "end_nav": end_nav,
        "n_days": int(len(work)),
        "status": "PASS" if ok else "FAIL",
    }


def lineage_template() -> list[str]:
    return ["signal_id", "order_id", "fill_id", "position_id", "exit", "net_pnl"]


def build_lineage_rows(
    *,
    scores: pd.DataFrame,
    trades: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """最小谱系：rank → intended slot；成交后挂 fill/position/exit。"""
    ranked = scores.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True], kind="mergesort").copy()
    ranked["raw_rank"] = ranked.groupby("signal_date").cumcount() + 1
    ranked["eligible_rank"] = ranked["raw_rank"]
    ranked["intended_slot"] = np.where(ranked["raw_rank"] <= 2, ranked["raw_rank"], np.nan)
    ranked["signal_id"] = ranked["signal_date"].dt.strftime("%Y%m%d") + "|" + ranked["instrument_id"].astype(str)
    ranked["order_id"] = ranked["signal_id"].where(ranked["intended_slot"].notna())
    if trades is None or trades.empty:
        ranked["fill_id"] = pd.NA
        ranked["position_id"] = pd.NA
        ranked["filled_quantity"] = np.nan
        ranked["unfilled_quantity"] = np.nan
        ranked["actual_slot"] = np.nan
        return ranked
    t = trades.copy()
    if "signal_date" not in t.columns and "entry_signal_date" in t.columns:
        t["signal_date"] = pd.to_datetime(t["entry_signal_date"])
    t["signal_date"] = pd.to_datetime(t["signal_date"])
    t["instrument_id"] = t["instrument_id"].astype(str)
    t["fill_id"] = t["signal_date"].dt.strftime("%Y%m%d") + "|F|" + t["instrument_id"]
    t["position_id"] = t["signal_date"].dt.strftime("%Y%m%d") + "|P|" + t["instrument_id"]
    keep = ["signal_date", "instrument_id", "fill_id", "position_id"]
    for col in ("quantity", "filled_quantity", "buy_cost", "sell_proceeds", "actual_exit_date"):
        if col in t.columns:
            keep.append(col)
    merged = ranked.merge(t[keep].drop_duplicates(["signal_date", "instrument_id"]), on=["signal_date", "instrument_id"], how="left", validate="one_to_one")
    if "quantity" in merged.columns and "filled_quantity" not in merged.columns:
        merged["filled_quantity"] = merged["quantity"]
    if "filled_quantity" in merged.columns:
        merged["unfilled_quantity"] = np.where(merged["intended_slot"].notna() & merged["filled_quantity"].isna(), 1.0, 0.0)
    merged["actual_slot"] = merged["intended_slot"].where(merged["fill_id"].notna())
    return merged
