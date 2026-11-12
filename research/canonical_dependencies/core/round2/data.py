"""只读加载 Stock 市场情绪 / 多种子预测 / 候选面板。禁止写入 Stock。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import (
    CANDIDATE_PANEL_RELATIVE,
    FS00263_FEATURES,
    PREDICTION_START,
    SEARCH_RUN,
    SIGNAL_END,
    STOCK_ROOT,
    TRIAL_ID,
    stock_path,
)
from ..stock_readonly import load_fs00263_ensemble_predictions

SEEDS = (17, 42, 73, 101, 202)
MARKET_CONTEXT_RELATIVE = Path(
    "data/factor_cache/first_board_single_model_combined_2013_20260724_v1/market_context.parquet"
)
LIMIT_DOWN_RELATIVE = Path(
    "data/factor_cache/first_board_market_limit_down_2013_20260724_v1/"
    "market_limit_down_context.parquet"
)


def load_market_context() -> pd.DataFrame:
    path = stock_path(MARKET_CONTEXT_RELATIVE)
    if not path.is_file():
        raise FileNotFoundError(f"missing market_context (read-only): {path}")
    frame = pd.read_parquet(path)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame.sort_values("trade_date").reset_index(drop=True)


def load_limit_down_context() -> pd.DataFrame:
    path = stock_path(LIMIT_DOWN_RELATIVE)
    if not path.is_file():
        raise FileNotFoundError(f"missing limit_down context (read-only): {path}")
    frame = pd.read_parquet(path)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame.sort_values("trade_date").reset_index(drop=True)


def load_regime_frame() -> pd.DataFrame:
    """Join existing caches. Formulas documented in 02_research_notes.md."""
    ctx = load_market_context()
    down = load_limit_down_context()
    frame = ctx.merge(down, on="trade_date", how="left", suffixes=("", "_dn"))
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    frame["up_num"] = pd.to_numeric(frame["market_limit_count"], errors="coerce")
    frame["fail_ratio"] = pd.to_numeric(frame["market_failed_limit_rate"], errors="coerce")
    frame["suc_ratio"] = pd.to_numeric(frame["market_seal_success_rate"], errors="coerce")
    frame["con_ratio"] = pd.to_numeric(frame["prev_limit_relimit_rate"], errors="coerce")
    frame["down_count"] = pd.to_numeric(frame["market_close_limit_down_count"], errors="coerce")
    frame["down_pressure"] = pd.to_numeric(frame["market_down_pressure"], errors="coerce")
    frame["up_ratio"] = pd.to_numeric(frame["market_up_ratio"], errors="coerce")
    for col in ("up_num", "fail_ratio", "suc_ratio", "con_ratio", "down_count", "down_pressure", "up_ratio"):
        frame[f"{col}_p250"] = frame[col].rolling(250, min_periods=40).rank(pct=True)
    # same-day D close is visible before D+1 open
    frame = frame.set_index("trade_date")
    return frame


def load_seed_predictions(*, seed: int, start=PREDICTION_START, signal_end=SIGNAL_END) -> pd.DataFrame:
    path = stock_path(
        Path(f"runs/{SEARCH_RUN}/predictions/n50_d2/seed{seed}/{TRIAL_ID}/predictions.parquet")
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing seed predictions (read-only): {path}")
    frame = pd.read_parquet(path, columns=["signal_date", "instrument_id", "score"])
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["seed"] = int(seed)
    return frame.loc[frame["signal_date"].dt.date.between(start, signal_end)].copy()


def load_all_seed_predictions() -> pd.DataFrame:
    return pd.concat([load_seed_predictions(seed=seed) for seed in SEEDS], ignore_index=True)


def seed_top1_agreement(ensemble: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    """For each signal_date: ensemble Top1 and how many seeds also pick it as Top1."""
    ens = ensemble.copy()
    ens["signal_date"] = pd.to_datetime(ens["signal_date"])
    top1 = (
        ens.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True])
        .groupby("signal_date", sort=True)
        .head(1)[["signal_date", "instrument_id", "score"]]
        .rename(columns={"instrument_id": "top1_id", "score": "top1_score"})
    )
    second = (
        ens.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True])
        .groupby("signal_date", sort=True)
        .nth(1)[["signal_date", "instrument_id", "score"]]
        .rename(columns={"instrument_id": "top2_id", "score": "top2_score"})
    )
    if "signal_date" not in second.columns:
        second = second.reset_index()
    margin = top1.merge(second, on="signal_date", how="left")
    margin["top1_top2_margin"] = margin["top1_score"] - margin["top2_score"]
    seed_top = (
        seeds.sort_values(["seed", "signal_date", "score", "instrument_id"], ascending=[True, True, False, True])
        .groupby(["seed", "signal_date"], sort=True)
        .head(1)[["seed", "signal_date", "instrument_id"]]
    )
    seed_top["signal_date"] = pd.to_datetime(seed_top["signal_date"])
    joined = seed_top.merge(
        margin[["signal_date", "top1_id"]],
        on="signal_date",
        how="left",
    )
    joined["agree"] = joined["instrument_id"] == joined["top1_id"]
    agree = joined.groupby("signal_date")["agree"].sum().rename("agree_n")
    out = margin.merge(agree, on="signal_date", how="left")
    out = out.sort_values("signal_date")
    out["agree_n"] = out["agree_n"].fillna(0).astype(int)
    out["margin_p250"] = out["top1_top2_margin"].rolling(250, min_periods=40).rank(pct=True)
    return out.set_index("signal_date").sort_index()


def seed_rank_pct_std(seeds: pd.DataFrame) -> pd.DataFrame:
    frame = seeds.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["rank_pct"] = frame.groupby(["seed", "signal_date"])["score"].rank(pct=True)
    std = (
        frame.groupby(["signal_date", "instrument_id"])["rank_pct"]
        .std(ddof=0)
        .rename("seed_rank_pct_std")
        .reset_index()
    )
    return std


def load_candidate_features(columns: list[str] | None = None) -> pd.DataFrame:
    path = stock_path(CANDIDATE_PANEL_RELATIVE)
    cols = columns or (
        ["trade_date", "instrument_id"]
        + list(FS00263_FEATURES)
        + [
            "ind_first_board_count_log",
            "ind_first_board_share",
            "label_e2_net",
            "label_d1_touch",
            "label_d1_touch_available_at",
            "is_first_board_hard",
            "g0_amt_vs_5",
            "h4_main_net_to_turnover",
        ]
    )
    # keep unique while preserving order
    seen: set[str] = set()
    use: list[str] = []
    for col in cols:
        if col not in seen:
            seen.add(col)
            use.append(col)
    frame = pd.read_parquet(path, columns=use)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    return frame


def load_money_flow_daily_agg(
    *,
    start=PREDICTION_START,
    end=SIGNAL_END,
) -> pd.DataFrame:
    """Market-level main/retail net inflow. D close visible. Missing → NaN, never filled."""
    from ..paths import MARKET_END, SNAPSHOT_RELATIVE
    from ..stock_readonly import ensure_stock_src_on_path

    ensure_stock_src_on_path()
    from zyquant.data import DataSnapshot

    snapshot = DataSnapshot(stock_path(SNAPSHOT_RELATIVE), verify_hashes=False)
    table = snapshot.table(
        "daily_money_flow",
        start=start,
        end=end if end else MARKET_END,
        cutoff=snapshot.metadata.as_of_date,
        fields=["trade_date", "instrument_id", "main_net_inflow", "retail_net_inflow", "turnover_value"],
    )
    table["trade_date"] = pd.to_datetime(table["trade_date"])
    grouped = table.groupby("trade_date", sort=True).agg(
        main_net_inflow=("main_net_inflow", "sum"),
        retail_net_inflow=("retail_net_inflow", "sum"),
        turnover_value=("turnover_value", "sum"),
        n_names=("instrument_id", "nunique"),
    )
    grouped["main_net_to_turnover"] = grouped["main_net_inflow"] / grouped["turnover_value"].replace(0, np.nan)
    grouped["main_net_p250"] = grouped["main_net_to_turnover"].rolling(250, min_periods=40).rank(pct=True)
    return grouped


def ensemble_with_top1(predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = predictions if predictions is not None else load_fs00263_ensemble_predictions()
    frame = frame.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["is_top1"] = False
    top_idx = (
        frame.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True])
        .groupby("signal_date", sort=True)
        .head(1)
        .index
    )
    frame.loc[top_idx, "is_top1"] = True
    return frame


def readonly_roots() -> dict[str, str]:
    return {
        "stock_root": str(STOCK_ROOT),
        "stock_root_resolved": str(STOCK_ROOT.resolve()),
        "market_context": str(stock_path(MARKET_CONTEXT_RELATIVE)),
        "limit_down_context": str(stock_path(LIMIT_DOWN_RELATIVE)),
    }
