"""用完整成交账本做 Top1/Top2 归因，不用 trades.head。"""

from __future__ import annotations

import pandas as pd


def _ranks(scores: pd.DataFrame) -> pd.DataFrame:
    work = scores.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"])
    work["instrument_id"] = work["instrument_id"].astype(str)
    work["rank"] = work.groupby("signal_date")["score"].rank(ascending=False, method="first")
    return work


def top_changes(base: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    left = _ranks(base)
    right = _ranks(cand)
    rows = []
    for rank in (1, 2):
        a = left.loc[left["rank"].eq(rank), ["signal_date", "instrument_id"]].rename(columns={"instrument_id": "base_id"})
        b = right.loc[right["rank"].eq(rank), ["signal_date", "instrument_id"]].rename(columns={"instrument_id": "cand_id"})
        merged = a.merge(b, on="signal_date", how="outer")
        merged["rank"] = rank
        merged["changed"] = merged["base_id"].fillna("") != merged["cand_id"].fillna("")
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def trade_delta(base_trades: pd.DataFrame, cand_trades: pd.DataFrame) -> pd.DataFrame:
    keys = ["signal_date", "instrument_id"]
    left = base_trades.copy()
    right = cand_trades.copy()
    for frame in (left, right):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    merged = left[keys + ["ret"]].merge(right[keys + ["ret"]], on=keys, how="outer", suffixes=("_base", "_cand"))
    merged["status"] = "both"
    merged.loc[merged["ret_base"].isna(), "status"] = "added"
    merged.loc[merged["ret_cand"].isna(), "status"] = "removed"
    return merged


def attribution_summary(changes: pd.DataFrame, trades: pd.DataFrame) -> dict:
    summary = {
        "top1_changed": int(changes.loc[changes["rank"].eq(1), "changed"].sum()) if len(changes) else 0,
        "top2_changed": int(changes.loc[changes["rank"].eq(2), "changed"].sum()) if len(changes) else 0,
        "trades_added": int((trades["status"] == "added").sum()) if len(trades) else 0,
        "trades_removed": int((trades["status"] == "removed").sum()) if len(trades) else 0,
        "added_ret_sum": float(trades.loc[trades["status"].eq("added"), "ret_cand"].sum()) if len(trades) else 0.0,
        "removed_ret_sum": float(trades.loc[trades["status"].eq("removed"), "ret_base"].sum()) if len(trades) else 0.0,
    }
    return summary
