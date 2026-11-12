"""每日一行：v4 Top1/Top2、agreement、separation、Top2 收益。PIT。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND10_ROOT, assert_project_result_path, ensure_round10_tree, round10_dir
from ..round8.geometry import load_gaps
from ..round9.geometry import build_geometry
from .allocator import sep_bucket
from .contract import DISAGREE_LABEL, R10_00


def _daily_nav(path) -> pd.Series:
    nav = pd.read_parquet(path)
    nav["trade_date"] = pd.to_datetime(nav["trade_date"]).dt.normalize()
    return nav.sort_values("trade_date").drop_duplicates("trade_date").set_index("trade_date")["total_asset"].astype(float)


def _rank_trades(code: str, rank: int) -> pd.DataFrame:
    path = round10_dir(code) / "trades.parquet"
    if not path.is_file():
        return pd.DataFrame()
    t = pd.read_parquet(path)
    t["signal_date"] = pd.to_datetime(t["signal_date"]).dt.normalize()
    t["instrument_id"] = t["instrument_id"].astype(str)
    t["_rank"] = t.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    t["ret"] = pd.to_numeric(t.get("ret"), errors="coerce")
    t["dollar"] = pd.to_numeric(t.get("sell_proceeds"), errors="coerce") - pd.to_numeric(
        t.get("buy_cost"), errors="coerce"
    )
    if "actual_exit_date" in t.columns:
        t["actual_exit_date"] = pd.to_datetime(t["actual_exit_date"]).dt.normalize()
    return t.loc[t["_rank"].eq(rank)].copy()


def build_daily_frame(*, v4_code: str = R10_00) -> pd.DataFrame:
    geo = build_geometry().copy()
    geo = geo.reset_index()
    geo["signal_date"] = pd.to_datetime(geo["signal_date"]).dt.normalize()
    geo = geo.set_index("signal_date").sort_index()
    out = pd.DataFrame(index=geo.index)
    out["top1_id"] = geo["top1_id"].astype(str)
    out["top2_id"] = geo["top2_id"].astype(str)
    out["head_state"] = geo["head_state"].astype(str)
    out["agreement"] = out["head_state"]
    out["agreement_alias"] = out["agreement"].map(DISAGREE_LABEL)
    out["candidate_count"] = pd.to_numeric(geo.get("candidate_count"), errors="coerce")
    for col in (
        "top1_minus_median_p250",
        "top2_minus_median_p250",
        "top1_minus_top2_p250",
        "five_top1",
        "ten_top1",
    ):
        if col in geo.columns:
            out[col] = geo[col]
    out["sep_bucket"] = sep_bucket(out["top2_minus_median_p250"])

    gaps = load_gaps().copy()
    gaps["signal_date"] = pd.to_datetime(gaps["signal_date"]).dt.normalize()
    gaps["instrument_id"] = gaps["instrument_id"].astype(str)
    for rank, col in ((1, "top1_open_gap"), (2, "top2_open_gap")):
        keys = out.reset_index()[["signal_date", f"top{rank}_id"]].rename(columns={f"top{rank}_id": "instrument_id"})
        g = keys.merge(gaps, on=["signal_date", "instrument_id"], how="left")
        out[col] = pd.to_numeric(g["d1_open_gap"], errors="coerce").to_numpy()
    out["gap_difference"] = out["top2_open_gap"] - out["top1_open_gap"]

    t2 = _rank_trades(v4_code, 2)
    if not t2.empty:
        one = t2.drop_duplicates("signal_date").set_index("signal_date")
        out["top2_ret"] = one.reindex(out.index)["ret"]
        out["top2_dollar"] = one.reindex(out.index)["dollar"]
        out["top2_net_return"] = out["top2_ret"]
        out["label_exit_date"] = one.reindex(out.index)["actual_exit_date"] if "actual_exit_date" in one else pd.NaT
        out["top2_budget"] = pd.to_numeric(one.reindex(out.index).get("target_name_budget"), errors="coerce")
        out["top2_score"] = pd.to_numeric(one.reindex(out.index).get("score"), errors="coerce")
    else:
        out["top2_ret"] = np.nan
        out["top2_dollar"] = np.nan
        out["top2_net_return"] = np.nan
        out["label_exit_date"] = pd.NaT
        out["top2_budget"] = np.nan
        out["top2_score"] = np.nan

    t1 = _rank_trades(v4_code, 1)
    if not t1.empty:
        one = t1.drop_duplicates("signal_date").set_index("signal_date")
        out["top1_ret"] = one.reindex(out.index)["ret"]
        out["top1_score"] = pd.to_numeric(one.reindex(out.index).get("score"), errors="coerce")
    else:
        out["top1_ret"] = np.nan
        out["top1_score"] = np.nan
    out["score_gap"] = out["top1_score"] - out["top2_score"]
    out["top2_incremental_pnl"] = out["top2_dollar"]
    return out


def persist_daily(daily: pd.DataFrame, name: str = "v4_daily_features.parquet") -> Any:
    ensure_round10_tree()
    path = ROUND10_ROOT / "diagnostics" / name
    assert_project_result_path(path)
    frame = daily.reset_index()
    if "signal_date" not in frame.columns:
        frame = frame.rename(columns={"index": "signal_date"})
    frame.to_parquet(path, index=False)
    return path


def load_daily(name: str = "v4_daily_features.parquet") -> pd.DataFrame:
    path = ROUND10_ROOT / "diagnostics" / name
    if not path.is_file():
        daily = build_daily_frame()
        persist_daily(daily, name)
        return daily
    frame = pd.read_parquet(path)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.normalize()
    return frame.set_index("signal_date").sort_index()


def train_mask(daily: pd.DataFrame, predict_year: int) -> pd.Series:
    years = daily.index.year
    train_years = {predict_year - 3, predict_year - 2, predict_year - 1}
    in_train = years.isin(train_years)
    exit_y = pd.to_datetime(daily.get("label_exit_date"), errors="coerce")
    mature = exit_y.dt.year.fillna(predict_year - 1).le(predict_year - 1)
    y = pd.to_numeric(daily.get("top2_ret"), errors="coerce")
    return in_train & mature & y.notna()


__all__ = ["build_daily_frame", "load_daily", "persist_daily", "train_mask"]
