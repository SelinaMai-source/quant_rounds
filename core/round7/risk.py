"""Recompute the historical daily eligibility and sizing rules from raw context."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ..paths import assert_project_result_path, ensure_round7_tree
from ..stock_readonly import sha256_file
from .contract import DEV_END, DEV_START, MARKET_CONTEXT, TEN_HEAD

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


def compute_daily_allocations(daily_scores: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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
    up_weak = (
        state["breadth_pass"]
        & state["market_limit_count"].lt(state["up_count_ma20"])
    )
    state.loc[up_weak, "allocation_multiplier"] *= 0.5
    observed = set(state.loc[state["allocation_multiplier"].gt(0), "allocation_multiplier"].unique())
    if not observed <= VALID_MULTIPLIERS:
        raise RuntimeError(f"unexpected allocation multipliers: {observed}")
    metadata = {
        "development_period": [DEV_START, DEV_END],
        "style_low_quantile": 0.20,
        "style_low_threshold": style_low,
        "auction_high_quantile": 0.80,
        "auction_high_threshold": auction_high,
        "breadth_min": 0.45,
        "weak_style_multiplier": 0.75,
        "auction_rescue_multiplier": 0.25,
        "up_weak_multiplier": 0.5,
        "active_days": int(state.loc[state["allocation_multiplier"].gt(0), "signal_date"].nunique()),
        "all_days": int(state["signal_date"].nunique()),
        "multiplier_counts": {
            str(key): int(value)
            for key, value in state["allocation_multiplier"].value_counts().sort_index().items()
        },
    }
    return state, metadata


def build_risk_predictions(*, resume: bool = True) -> dict:
    source = ensure_round7_tree() / "predictions" / "dual90_10_gap_floor" / "predictions.parquet"
    root = ensure_round7_tree() / "predictions" / "clean_risk"
    root.mkdir(parents=True, exist_ok=True)
    target = root / "predictions.parquet"
    daily_path = root / "daily_state.parquet"
    report_path = root / "report.json"
    assert_project_result_path(target)
    if resume and target.is_file() and report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))

    predictions = pd.read_parquet(source)
    predictions["signal_date"] = pd.to_datetime(predictions["signal_date"])
    panel_path = TEN_HEAD.panel_dir / "candidate_panel.parquet"
    panel = pd.read_parquet(
        panel_path,
        columns=["trade_date", "instrument_id", "d1_open_gap", "label_reject_reason"],
    ).rename(columns={"trade_date": "signal_date"})
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    panel["instrument_id"] = panel["instrument_id"].astype(str)
    market = pd.read_parquet(MARKET_CONTEXT)
    market["trade_date"] = pd.to_datetime(market["trade_date"])

    daily_scores = score_top2_days(predictions, panel)
    daily_state, metadata = compute_daily_allocations(daily_scores, market)
    active = daily_state.loc[daily_state["allocation_multiplier"].gt(0), ["signal_date", "allocation_multiplier"]]
    output = predictions.merge(active, on="signal_date", how="inner", validate="many_to_one")
    if output.empty or not np.isfinite(output["allocation_multiplier"]).all():
        raise RuntimeError("invalid risk output")
    output.to_parquet(target, index=False)
    daily_state.to_parquet(daily_path, index=False)
    payload = {
        "status": "complete",
        "runtime_dependency_on_legacy_anchor": False,
        "evidence_role": "consumed_historical_development_recomputed_cleanly",
        "decision_clock": "D close market context plus D+1 09:25 auction context",
        "same_open_fill_execution_risk": True,
        "input_rows": int(len(predictions)),
        "output_rows": int(len(output)),
        "output_days": int(output["signal_date"].nunique()),
        "rules": metadata,
        "source_sha256": sha256_file(source),
        "candidate_panel_sha256": sha256_file(panel_path),
        "market_context_sha256": sha256_file(MARKET_CONTEXT),
        "output_sha256": sha256_file(target),
        "daily_state_sha256": sha256_file(daily_path),
        "written_at": datetime.now().astimezone().isoformat(),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
