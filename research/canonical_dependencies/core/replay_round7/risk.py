"""Round7 clean_risk 规则：只读 Stock 上下文，把乘子乘到当前分数上。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contract import DEV_END, DEV_START, MARKET_CONTEXT, TEN_PANEL

KEYS = ["signal_date", "instrument_id"]
VALID_MULTIPLIERS = {0.25, 0.375, 0.5, 0.75, 1.0}


@dataclass(frozen=True)
class ScoreConfig:
    percentile_window: int = 500
    healthy_gap_cap: float = 0.05
    auction_shrinkage: float = 30.0
    style_shrinkage: float = 30.0


def _rolling_percentile(values: pd.Series, window: int) -> pd.Series:
    minimum = max(20, window // 4)

    def last_rank(block: np.ndarray) -> float:
        valid = block[np.isfinite(block)]
        if len(valid) < minimum or not np.isfinite(block[-1]):
            return np.nan
        return float((valid <= block[-1]).mean())

    return values.astype(float).rolling(window, min_periods=minimum).apply(last_rank, raw=True)


def _shrunk_rate(rate: pd.Series, count: pd.Series, strength: float, prior: pd.Series) -> pd.Series:
    n = pd.to_numeric(count, errors="coerce").fillna(0).clip(lower=0)
    r = pd.to_numeric(rate, errors="coerce")
    return (n * r + strength * prior) / (n + strength)


def _board(instrument_id: pd.Series) -> pd.Series:
    symbol = instrument_id.astype(str).str.split(".").str[0]
    return pd.Series(
        np.select(
            [symbol.str.startswith(("300", "301")), symbol.str.startswith("688")],
            ["chinext", "star"],
            default="main",
        ),
        index=instrument_id.index,
    )


def auction_and_style(panel: pd.DataFrame, config: ScoreConfig = ScoreConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = panel.copy()
    work["board"] = _board(work["instrument_id"])
    work["buyable"] = ~work["label_reject_reason"].isin(["buy_open_limit_up", "missing_buy_open"])
    work["red_buyable"] = work["buyable"] & work["d1_open_gap"].gt(0)
    work["healthy_buyable"] = (
        work["buyable"]
        & work["d1_open_gap"].ge(-0.01)
        & work["d1_open_gap"].lt(config.healthy_gap_cap)
    )
    daily = work.groupby("signal_date", as_index=False).agg(
        cohort_count=("instrument_id", "size"),
        buyable_rate=("buyable", "mean"),
        red_buyable_rate=("red_buyable", "mean"),
        healthy_buyable_rate=("healthy_buyable", "mean"),
    ).sort_values("signal_date")
    for name in ("buyable_rate", "red_buyable_rate", "healthy_buyable_rate"):
        prior = daily[name].expanding(min_periods=20).mean().shift(1)
        daily[name] = _shrunk_rate(daily[name], daily["cohort_count"], config.auction_shrinkage, prior)
        daily[f"pct_{name}"] = _rolling_percentile(daily[name], config.percentile_window)
    daily["auction_score"] = daily[
        ["pct_buyable_rate", "pct_red_buyable_rate", "pct_healthy_buyable_rate"]
    ].mean(axis=1, skipna=True)

    grouped = work.groupby(["signal_date", "board"], as_index=False).agg(
        board_count=("instrument_id", "size"),
        board_buyable_rate=("buyable", "mean"),
        board_red_buyable_rate=("red_buyable", "mean"),
        board_healthy_buyable_rate=("healthy_buyable", "mean"),
    ).merge(
        daily[["signal_date", "buyable_rate", "red_buyable_rate", "healthy_buyable_rate"]],
        on="signal_date",
        how="left",
        validate="many_to_one",
    ).sort_values(["board", "signal_date"])
    pct_columns = []
    for board_name, global_name in (
        ("board_buyable_rate", "buyable_rate"),
        ("board_red_buyable_rate", "red_buyable_rate"),
        ("board_healthy_buyable_rate", "healthy_buyable_rate"),
    ):
        grouped[board_name] = (
            grouped["board_count"] * grouped[board_name]
            + config.style_shrinkage * grouped[global_name]
        ) / (grouped["board_count"] + config.style_shrinkage)
        pct_name = f"pct_{board_name}"
        grouped[pct_name] = grouped.groupby("board", group_keys=False)[board_name].transform(
            lambda values: _rolling_percentile(values, config.percentile_window)
        )
        pct_columns.append(pct_name)
    grouped["style_score"] = grouped[pct_columns].mean(axis=1, skipna=True)
    return daily[["signal_date", "auction_score"]], grouped[["signal_date", "board", "style_score"]]


def score_top2_days(predictions: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    auction, style = auction_and_style(panel)
    scored = predictions[[*KEYS, "score"]].merge(
        auction, on="signal_date", how="left", validate="many_to_one"
    )
    scored["board"] = _board(scored["instrument_id"])
    scored = scored.merge(style, on=["signal_date", "board"], how="left", validate="many_to_one")
    ranked = scored.sort_values(
        ["signal_date", "score", "instrument_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).groupby("signal_date", sort=False).head(2).copy()
    ranked["selection_rank"] = ranked.groupby("signal_date", sort=False).cumcount() + 1
    ranked["rank_weight"] = ranked["selection_rank"].map({1: 0.75, 2: 0.25})
    rows = []
    for day, block in ranked.groupby("signal_date", sort=True):
        row = {"signal_date": day}
        for column in ("auction_score", "style_score"):
            valid = block[column].notna()
            row[column] = (
                float(np.average(block.loc[valid, column], weights=block.loc[valid, "rank_weight"]))
                if valid.any() else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_daily_allocations(daily_scores: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    context = market.copy().sort_values("trade_date")
    context["signal_date"] = pd.to_datetime(context["trade_date"])
    context["up_count_ma20"] = pd.to_numeric(context["market_limit_count"], errors="coerce").rolling(
        20, min_periods=20
    ).mean()
    state = daily_scores.merge(
        context[["signal_date", "market_up_ratio", "market_limit_count", "up_count_ma20"]],
        on="signal_date",
        how="left",
        validate="one_to_one",
    )
    state["breadth_pass"] = state["market_up_ratio"].ge(0.45)
    development = state.loc[state["signal_date"].between(DEV_START, DEV_END)]
    style_low = float(development.loc[development["breadth_pass"], "style_score"].quantile(0.20))
    auction_high = float(development.loc[~development["breadth_pass"], "auction_score"].quantile(0.80))
    if not np.isfinite(style_low) or not np.isfinite(auction_high):
        raise RuntimeError("development thresholds are not finite")

    state["allocation_multiplier"] = 0.0
    strong = state["breadth_pass"] & state["style_score"].ge(style_low)
    weak = state["breadth_pass"] & state["style_score"].lt(style_low)
    rescue = ~state["breadth_pass"] & state["auction_score"].ge(auction_high)
    state.loc[strong, "allocation_multiplier"] = 1.0
    state.loc[weak, "allocation_multiplier"] = 0.75
    state.loc[rescue, "allocation_multiplier"] = 0.25
    up_weak = state["breadth_pass"] & state["market_limit_count"].lt(state["up_count_ma20"])
    state.loc[up_weak, "allocation_multiplier"] *= 0.5
    observed = set(state.loc[state["allocation_multiplier"].gt(0), "allocation_multiplier"].unique())
    if not observed <= VALID_MULTIPLIERS:
        raise RuntimeError(f"unexpected allocation multipliers: {observed}")
    return state


_CACHE: dict[str, pd.DataFrame] = {}


def _panel_market() -> tuple[pd.DataFrame, pd.DataFrame]:
    if "panel" not in _CACHE:
        panel = pd.read_parquet(
            TEN_PANEL / "candidate_panel.parquet",
            columns=["trade_date", "instrument_id", "d1_open_gap", "label_reject_reason"],
        ).rename(columns={"trade_date": "signal_date"})
        panel["signal_date"] = pd.to_datetime(panel["signal_date"])
        panel["instrument_id"] = panel["instrument_id"].astype(str)
        _CACHE["panel"] = panel
    if "market" not in _CACHE:
        market = pd.read_parquet(MARKET_CONTEXT)
        market["trade_date"] = pd.to_datetime(market["trade_date"])
        _CACHE["market"] = market
    return _CACHE["panel"], _CACHE["market"]


def apply_round7_risk(predictions: pd.DataFrame) -> pd.DataFrame:
    """按 Round7 clean_risk 结构，用当前分数重算日乘子。不写 Stock。"""
    work = predictions.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"])
    work["instrument_id"] = work["instrument_id"].astype(str)
    panel, market = _panel_market()
    daily_scores = score_top2_days(work, panel)
    daily_state = compute_daily_allocations(daily_scores, market)
    active = daily_state.loc[
        daily_state["allocation_multiplier"].gt(0),
        ["signal_date", "allocation_multiplier"],
    ]
    output = work.merge(active, on="signal_date", how="inner", validate="many_to_one")
    if output.empty or not np.isfinite(output["allocation_multiplier"]).all():
        raise RuntimeError("invalid Round7 risk output")
    return output
