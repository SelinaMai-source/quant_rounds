"""Fixed-count dual-sleeve executor for the first-board strategy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .labels import ExitLabelCosts, _modeled_price, open_buy_block_reason


KEYS = ["trade_date", "instrument_id"]
PREDICTION_KEYS = ["signal_date", "instrument_id"]


@dataclass(frozen=True)
class FixedSleeveConfig:
    initial_cash: float = 300_000.0
    sleeve_count: int = 2
    sleeve_cash: float = 150_000.0
    top_k: int | None = 3
    name_budget: float = 45_000.0
    name_budget_nav_fraction: float | None = None
    rank_name_budget_nav_fractions: tuple[float, ...] | None = None
    cohort_cash_fraction: float = 1.0
    lot_size: int = 100
    costs: ExitLabelCosts = field(default_factory=ExitLabelCosts)
    limit_up_tolerance: float = 0.005
    market_up_ratio_min: float | None = None
    market_up_ratio_full_allocation_min: float | None = None
    reduced_name_budget_nav_fraction: float | None = None
    high_gap_threshold: float | None = None
    high_gap_extra_entry_slippage_bps: float = 0.0
    open_gap_min: float | None = None
    open_gap_max: float | None = None
    minimum_score: float | None = None
    exit_mode: str = "close_from_d2"
    ma_window: int = 5
    max_holding_sessions: int = 10


@dataclass
class _Position:
    cohort_id: str
    instrument_id: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    planned_exit_date: pd.Timestamp
    initial_quantity: int
    quantity: int
    entry_reference_price: float
    entry_price: float
    buy_notional: float
    buy_commission: float
    buy_transfer_fee: float
    buy_cost: float
    score: float
    last_close: float
    entry_portfolio_nav: float
    target_name_budget: float
    allocation_multiplier: float
    cash_dividend: float = 0.0
    exit_armed_on: pd.Timestamp | None = None
    exit_trigger: str | None = None


@dataclass
class _Sleeve:
    sleeve_id: int
    cash: float
    positions: dict[str, _Position] = field(default_factory=dict)
    occupied_sessions: int = 0


@dataclass(frozen=True)
class PreparedFixedSleeveMarket:
    """Read-only market state reusable across many identical backtests."""

    bars: pd.DataFrame
    calendar: pd.DatetimeIndex
    market_lookup: pd.DataFrame
    market_up_ratio: dict[pd.Timestamp, float]
    action_lookup: dict[tuple[pd.Timestamp, str], list[dict[str, Any]]]
    ma_window: int


def prepare_fixed_sleeve_market(
    market: pd.DataFrame,
    *,
    corporate_actions: pd.DataFrame | None = None,
    ma_window: int = 5,
) -> PreparedFixedSleeveMarket:
    """Prepare immutable lookup state once for a batch of strategy runs."""
    bars = _prepare_market(market, ma_window=ma_window)
    actions = _prepare_corporate_actions(corporate_actions)
    return PreparedFixedSleeveMarket(
        bars=bars,
        calendar=pd.DatetimeIndex(bars["trade_date"].drop_duplicates()).sort_values(),
        market_lookup=bars.set_index(KEYS, verify_integrity=True),
        market_up_ratio=_market_up_ratio(bars),
        action_lookup=_action_lookup(actions),
        ma_window=ma_window,
    )


def run_fixed_dual_sleeve_top3_backtest(
    predictions: pd.DataFrame,
    market: pd.DataFrame | PreparedFixedSleeveMarket,
    *,
    corporate_actions: pd.DataFrame | None = None,
    entry_price_overrides: pd.DataFrame | None = None,
    config: FixedSleeveConfig = FixedSleeveConfig(),
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> dict[str, Any]:
    """Run a strict two-sleeve Top3 backtest with limit-up close holds.

    Two fixed cash sleeves alternate by entry session. Limit-up leftovers keep
    consuming their own sleeve capital, but do not lock cash released by other
    names in the same cohort.
    """
    _validate_config(config)
    scores = _prepare_predictions(predictions)
    entry_price_lookup = _prepare_entry_price_overrides(entry_price_overrides)
    if isinstance(market, PreparedFixedSleeveMarket):
        if corporate_actions is not None:
            raise ValueError("corporate_actions must be included when preparing market state")
        if market.ma_window != config.ma_window:
            raise ValueError("prepared market ma_window does not match execution config")
        bars = market.bars
        calendar = market.calendar
        market_lookup = market.market_lookup
        market_up_ratio = market.market_up_ratio
        action_lookup = market.action_lookup
    else:
        prepared = prepare_fixed_sleeve_market(
            market, corporate_actions=corporate_actions, ma_window=config.ma_window
        )
        bars = prepared.bars
        calendar = prepared.calendar
        market_lookup = prepared.market_lookup
        market_up_ratio = prepared.market_up_ratio
        action_lookup = prepared.action_lookup
    if start_date is not None:
        calendar = calendar[calendar >= pd.Timestamp(start_date)]
    if end_date is not None:
        calendar = calendar[calendar <= pd.Timestamp(end_date)]
    if len(calendar) < 3:
        raise ValueError("backtest calendar must contain at least three sessions")
    calendar_position = {day: offset for offset, day in enumerate(calendar)}

    entry_schedule: dict[pd.Timestamp, dict[str, Any]] = {}
    insufficient_horizon: list[dict[str, Any]] = []
    for signal_date, block in scores.groupby("signal_date", sort=True):
        signal_date = pd.Timestamp(signal_date)
        offset = calendar_position.get(signal_date)
        if offset is None or offset + 2 >= len(calendar):
            insufficient_horizon.append({
                "event": "signal_rejected", "reason": "insufficient_horizon",
                "signal_date": signal_date, "execution_date": pd.NaT,
                "sleeve_id": pd.NA, "cohort_id": pd.NA,
                "instrument_id": pd.NA,
            })
            continue
        chosen = block.sort_values(
            ["score", "instrument_id"], ascending=[False, True]
        )
        if config.top_k is not None:
            chosen = chosen.head(config.top_k)
        score_gate_rejections = 0
        if config.minimum_score is not None:
            score_gate_rejections = int(chosen["score"].le(config.minimum_score).sum())
            chosen = chosen.loc[chosen["score"].gt(config.minimum_score)].copy()
        chosen = chosen.copy()
        chosen["selection_rank"] = np.arange(1, len(chosen) + 1, dtype=int)
        up_ratio = market_up_ratio.get(signal_date, np.nan)
        market_gate_pass = (
            config.market_up_ratio_min is None
            or (_finite(up_ratio) and float(up_ratio) >= config.market_up_ratio_min)
        )
        allocation_tier = "full"
        effective_name_budget_nav_fraction = config.name_budget_nav_fraction
        if (
            market_gate_pass
            and config.market_up_ratio_full_allocation_min is not None
            and float(up_ratio) < config.market_up_ratio_full_allocation_min
        ):
            allocation_tier = "reduced"
            effective_name_budget_nav_fraction = (
                config.reduced_name_budget_nav_fraction
            )
        entry_schedule[calendar[offset + 1]] = {
            "signal_date": signal_date,
            "planned_exit_date": calendar[offset + 2],
            "sleeve_id": int((offset + 1) % config.sleeve_count),
            "market_up_ratio": up_ratio,
            "market_gate_pass": market_gate_pass,
            "allocation_tier": allocation_tier,
            "name_budget_nav_fraction": effective_name_budget_nav_fraction,
            "score_gate_rejections": score_gate_rejections,
            "picks": chosen[[
                "instrument_id", "score", "selection_rank", "allocation_multiplier",
            ]].to_dict("records"),
        }

    sleeves = [_Sleeve(sleeve_id=index, cash=config.sleeve_cash)
               for index in range(config.sleeve_count)]
    trades: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = list(insufficient_horizon)
    positions: list[dict[str, Any]] = []
    sleeve_states: list[dict[str, Any]] = []
    nav_rows: list[dict[str, Any]] = []
    cashflows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    skipped_insufficient_sleeve_cash = 0
    market_gate_rejected_signals = 0
    score_gate_rejected_names = 0
    limit_up_delay_attempts = 0
    limit_down_delay_attempts = 0
    unavailable_sell_attempts = 0

    for day in calendar:
        day = pd.Timestamp(day)
        _apply_actions(day, sleeves, action_lookup, cashflows)

        # In the first-board open-exit mode, a non-limit close arms an order for
        # the following session's open. Process it before allocating new cash.
        if config.exit_mode in {"next_open_after_non_limit_close", "ma5_next_open"}:
            for sleeve in sleeves:
                for position_key, position in list(sleeve.positions.items()):
                    if position.exit_armed_on is None or day <= position.exit_armed_on:
                        continue
                    bar = _market_row(market_lookup, day, position.instrument_id)
                    reason = _sell_open_block_reason(bar, config.limit_up_tolerance)
                    if reason is not None:
                        if reason == "open_limit_down":
                            limit_down_delay_attempts += 1
                        else:
                            unavailable_sell_attempts += 1
                        rejections.append({
                            "event": "sell_open_rejected", "reason": reason,
                            "signal_date": position.signal_date, "execution_date": day,
                            "sleeve_id": sleeve.sleeve_id, "cohort_id": position.cohort_id,
                            "instrument_id": position.instrument_id,
                            "exit_armed_on": position.exit_armed_on,
                        })
                        continue
                    trade, sell_costs = _close_position(
                        sleeve, position, day, bar, config, reference_field="open"
                    )
                    trades.append(trade)
                    cost_rows.extend(sell_costs)
                    del sleeve.positions[position_key]

        scheduled = entry_schedule.get(day)
        if scheduled is not None:
            score_gate_rejected_names += int(scheduled["score_gate_rejections"])
            if not scheduled["market_gate_pass"]:
                market_gate_rejected_signals += 1
                rejections.append({
                    "event": "signal_rejected", "reason": "market_up_ratio_below_min",
                    "signal_date": scheduled["signal_date"], "execution_date": day,
                    "sleeve_id": scheduled["sleeve_id"], "cohort_id": pd.NA,
                    "instrument_id": pd.NA,
                    "market_up_ratio": scheduled["market_up_ratio"],
                })
            else:
                sleeve = sleeves[int(scheduled["sleeve_id"])]
                entry_portfolio_nav = _portfolio_nav_at_open(
                    sleeves, market_lookup, day
                )
                opened = _open_cohort(
                    sleeve, scheduled, day, market_lookup, config,
                    rejections, cost_rows, entry_portfolio_nav,
                    entry_price_lookup,
                )
                if not opened and sleeve.cash < config.lot_size:
                    skipped_insufficient_sleeve_cash += 1

        for sleeve in sleeves:
            if not sleeve.positions:
                continue
            sleeve.occupied_sessions += 1
            for position_key, position in list(sleeve.positions.items()):
                instrument_id = position.instrument_id
                bar = _market_row(market_lookup, day, instrument_id)
                if bar is not None and _finite(bar.get("close")):
                    position.last_close = float(bar["close"])
                if config.exit_mode == "next_open_after_non_limit_close":
                    if position.exit_armed_on is not None:
                        continue
                    if bar is None or (
                        bool(bar["paused"]) if pd.notna(bar.get("paused")) else False
                    ) or not _finite(bar.get("volume")) or float(bar["volume"]) <= 0 \
                            or not _finite(bar.get("close")):
                        unavailable_sell_attempts += 1
                        continue
                    if _finite(bar.get("limit_up")) and float(bar["close"]) >= (
                        float(bar["limit_up"]) - config.limit_up_tolerance
                    ):
                        limit_up_delay_attempts += 1
                    else:
                        position.exit_armed_on = day
                        position.exit_trigger = "non_limit_close"
                    continue
                if config.exit_mode == "ma5_next_open":
                    if position.exit_armed_on is not None:
                        continue
                    holding_sessions = (
                        calendar_position[day] - calendar_position[position.entry_date] + 1
                    )
                    if holding_sessions >= config.max_holding_sessions:
                        position.exit_armed_on = day
                        position.exit_trigger = "max_holding_sessions"
                        continue
                    if day < position.planned_exit_date:
                        continue
                    if bar is None or (
                        bool(bar["paused"]) if pd.notna(bar.get("paused")) else False
                    ) or not _finite(bar.get("volume")) or float(bar["volume"]) <= 0 \
                            or not _finite(bar.get("ma_price")) or not _finite(bar.get("ma5")):
                        unavailable_sell_attempts += 1
                        continue
                    if float(bar["ma_price"]) < float(bar["ma5"]):
                        position.exit_armed_on = day
                        position.exit_trigger = "ma5_breach"
                    continue
                if day < position.planned_exit_date:
                    continue
                reason = _sell_block_reason(bar, config.limit_up_tolerance)
                if reason is not None:
                    if reason == "limit_up_hold":
                        limit_up_delay_attempts += 1
                    elif reason == "one_price_limit_down":
                        limit_down_delay_attempts += 1
                    else:
                        unavailable_sell_attempts += 1
                    rejections.append({
                        "event": "sell_rejected", "reason": reason,
                        "signal_date": position.signal_date, "execution_date": day,
                        "sleeve_id": sleeve.sleeve_id, "cohort_id": position.cohort_id,
                        "instrument_id": instrument_id,
                    })
                    continue
                trade, sell_costs = _close_position(sleeve, position, day, bar, config)
                trades.append(trade)
                cost_rows.extend(sell_costs)
                del sleeve.positions[position_key]

        total_market_value = 0.0
        for sleeve in sleeves:
            sleeve_market_value = 0.0
            for position in sleeve.positions.values():
                market_value = float(position.quantity) * float(position.last_close)
                sleeve_market_value += market_value
                positions.append({
                    "trade_date": day, "sleeve_id": sleeve.sleeve_id,
                    "cohort_id": position.cohort_id,
                    "instrument_id": position.instrument_id,
                    "quantity": position.quantity, "last_close": position.last_close,
                    "market_value": market_value,
                    "signal_date": position.signal_date,
                    "planned_exit_date": position.planned_exit_date,
                })
            total_market_value += sleeve_market_value
            sleeve_states.append({
                "trade_date": day, "sleeve_id": sleeve.sleeve_id,
                "cash": sleeve.cash, "market_value": sleeve_market_value,
                "equity": sleeve.cash + sleeve_market_value,
                "occupied": bool(sleeve.positions),
                "live_cohort_count": len({p.cohort_id for p in sleeve.positions.values()}),
                "position_count": len(sleeve.positions),
            })
        total_cash = float(sum(sleeve.cash for sleeve in sleeves))
        nav_rows.append({
            "trade_date": day, "cash": total_cash,
            "market_value": total_market_value,
            "total_asset": total_cash + total_market_value,
            "live_sleeves": sum(bool(sleeve.positions) for sleeve in sleeves),
        })

    frames = {
        "trades": pd.DataFrame(trades),
        "rejections": pd.DataFrame(rejections),
        "positions": pd.DataFrame(positions),
        "sleeve_states": pd.DataFrame(sleeve_states),
        "nav": pd.DataFrame(nav_rows),
        "cashflows": pd.DataFrame(cashflows),
        "costs": pd.DataFrame(cost_rows),
    }
    report = _build_report(frames, sleeves, config, skipped_insufficient_sleeve_cash, {
        "limit_up": limit_up_delay_attempts,
        "limit_down": limit_down_delay_attempts,
        "unavailable": unavailable_sell_attempts,
    }, market_gate_rejected_signals, score_gate_rejected_names)
    return {**frames, "report": report}


def _validate_config(config: FixedSleeveConfig) -> None:
    if config.sleeve_count != 2:
        raise ValueError("fixed dual-sleeve executor requires sleeve_count=2")
    if not np.isclose(config.sleeve_count * config.sleeve_cash, config.initial_cash):
        raise ValueError("sleeve_count * sleeve_cash must equal initial_cash")
    if config.top_k is not None and config.top_k <= 0:
        raise ValueError("top_k must be positive or None")
    if (
        config.name_budget_nav_fraction is None
        and config.top_k is not None
        and config.name_budget * config.top_k > config.sleeve_cash + 1e-9
    ):
        raise ValueError("Top3 name budgets exceed sleeve cash")
    if config.name_budget_nav_fraction is not None and not (
        0 < config.name_budget_nav_fraction <= 1
    ):
        raise ValueError("name_budget_nav_fraction must be in (0, 1]")
    if config.rank_name_budget_nav_fractions is not None:
        fractions = tuple(config.rank_name_budget_nav_fractions)
        if config.top_k is None or len(fractions) != config.top_k:
            raise ValueError("rank budget fractions must contain exactly top_k values")
        if any(not 0 < value <= 1 for value in fractions):
            raise ValueError("rank budget fractions must be in (0, 1]")
        if config.name_budget_nav_fraction is not None:
            raise ValueError("rank budgets and equal NAV fraction are mutually exclusive")
        if sum(fractions) > 1 + 1e-12:
            raise ValueError("rank budget fractions cannot exceed total NAV")
    if config.market_up_ratio_min is not None and not (
        0 <= config.market_up_ratio_min <= 1
    ):
        raise ValueError("market_up_ratio_min must be in [0, 1]")
    soft_gate_values = (
        config.market_up_ratio_full_allocation_min,
        config.reduced_name_budget_nav_fraction,
    )
    if any(value is not None for value in soft_gate_values):
        if any(value is None for value in soft_gate_values):
            raise ValueError(
                "market_up_ratio_full_allocation_min and "
                "reduced_name_budget_nav_fraction must be set together"
            )
        if config.market_up_ratio_min is None:
            raise ValueError("soft market allocation requires market_up_ratio_min")
        if config.name_budget_nav_fraction is None:
            raise ValueError("soft market allocation requires name_budget_nav_fraction")
        if not (
            config.market_up_ratio_min
            <= config.market_up_ratio_full_allocation_min
            <= 1
        ):
            raise ValueError("market allocation thresholds are invalid")
        if not (
            0 < config.reduced_name_budget_nav_fraction
            <= config.name_budget_nav_fraction
        ):
            raise ValueError("reduced_name_budget_nav_fraction is invalid")
    if not 0 < config.cohort_cash_fraction <= 1:
        raise ValueError("cohort_cash_fraction must be in (0, 1]")
    if config.high_gap_extra_entry_slippage_bps < 0:
        raise ValueError("high-gap extra entry slippage must be nonnegative")
    if config.high_gap_extra_entry_slippage_bps > 0 and config.high_gap_threshold is None:
        raise ValueError("high-gap threshold is required for extra entry slippage")
    if config.exit_mode not in {
        "close_from_d2", "next_open_after_non_limit_close", "ma5_next_open",
    }:
        raise ValueError("unsupported exit_mode")
    if config.ma_window < 2:
        raise ValueError("ma_window must be at least 2")
    if config.max_holding_sessions < 2:
        raise ValueError("max_holding_sessions must be at least 2")
    if config.lot_size <= 0 or config.name_budget <= 0:
        raise ValueError("lot_size and name_budget must be positive")
    if (
        config.open_gap_min is not None and config.open_gap_max is not None
        and config.open_gap_min > config.open_gap_max
    ):
        raise ValueError("open_gap_min must not exceed open_gap_max")


def _prepare_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(PREDICTION_KEYS + ["score"]) - set(frame.columns))
    if missing:
        raise ValueError(f"predictions missing columns: {missing}")
    columns = PREDICTION_KEYS + ["score"]
    if "allocation_multiplier" in frame:
        columns.append("allocation_multiplier")
    result = frame[columns].copy()
    result["signal_date"] = pd.to_datetime(result["signal_date"])
    result["instrument_id"] = result["instrument_id"].astype(str)
    result["score"] = pd.to_numeric(result["score"], errors="coerce")
    if "allocation_multiplier" not in result:
        result["allocation_multiplier"] = 1.0
    result["allocation_multiplier"] = pd.to_numeric(
        result["allocation_multiplier"], errors="coerce"
    )
    if result.duplicated(PREDICTION_KEYS).any():
        raise ValueError("duplicate prediction keys")
    valid_multiplier = (
        np.isfinite(result["allocation_multiplier"])
        & result["allocation_multiplier"].gt(0)
        & result["allocation_multiplier"].le(1)
    )
    if not valid_multiplier.all():
        raise ValueError("allocation_multiplier must be finite and in (0, 1]")
    return result.loc[np.isfinite(result["score"])].copy()


def _prepare_entry_price_overrides(
    frame: pd.DataFrame | None,
) -> pd.Series | None:
    if frame is None:
        return None
    required = ["entry_date", "instrument_id", "entry_reference_price"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"entry price overrides missing columns: {missing}")
    result = frame[required].copy()
    result["entry_date"] = pd.to_datetime(result["entry_date"])
    result["instrument_id"] = result["instrument_id"].astype(str)
    result["entry_reference_price"] = pd.to_numeric(
        result["entry_reference_price"], errors="coerce"
    )
    if result.duplicated(["entry_date", "instrument_id"]).any():
        raise ValueError("duplicate entry price override keys")
    if not np.isfinite(result["entry_reference_price"]).all() or (
        result["entry_reference_price"] <= 0
    ).any():
        raise ValueError("entry price overrides must be finite and positive")
    return result.set_index(["entry_date", "instrument_id"])["entry_reference_price"]


def _prepare_market(frame: pd.DataFrame, *, ma_window: int = 5) -> pd.DataFrame:
    required = KEYS + [
        "open", "high", "low", "close", "volume", "paused",
        "limit_up", "limit_down",
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"market missing columns: {missing}")
    result = frame[required].copy()
    if "pre_close" in frame.columns:
        result["pre_close"] = frame["pre_close"].values
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    result["instrument_id"] = result["instrument_id"].astype(str)
    if result.duplicated(KEYS).any():
        raise ValueError("duplicate market keys")
    result = result.sort_values(KEYS).reset_index(drop=True)
    close = pd.to_numeric(result["close"], errors="coerce")
    pre_close = (
        pd.to_numeric(result["pre_close"], errors="coerce")
        if "pre_close" in result
        else pd.Series(np.nan, index=result.index, dtype="float64")
    )
    prior_raw_close = close.groupby(result["instrument_id"], sort=False).shift(1)
    ratio = close / pre_close.where(pre_close.gt(0))
    fallback = close / prior_raw_close.where(prior_raw_close.gt(0))
    ratio = ratio.where(np.isfinite(ratio) & ratio.gt(0), fallback)
    # Rows are date-major (the executor lookup contract), so adjacent rows
    # normally belong to different instruments.  Detect the first observation
    # within each instrument rather than comparing adjacent rows.
    first = result.groupby("instrument_id", sort=False).cumcount().eq(0)
    ratio = ratio.where(~first, 1.0).fillna(1.0)
    log_index = np.log(ratio).groupby(result["instrument_id"], sort=False).cumsum()
    result["ma_price"] = np.exp(log_index)
    result["ma5"] = result.groupby("instrument_id", sort=False)["ma_price"].transform(
        lambda values: values.rolling(ma_window, min_periods=ma_window).mean()
    )
    return result


def _market_up_ratio(frame: pd.DataFrame) -> dict[pd.Timestamp, float]:
    work = frame.copy()
    if "pre_close" not in work.columns:
        work["pre_close"] = work.sort_values(KEYS).groupby("instrument_id")["close"].shift(1)
    # Match the original market_daily formula: every stock row with a valid
    # daily return remains in the denominator, including a suspended 0% row.
    valid = (
        pd.to_numeric(work["close"], errors="coerce").notna()
        & pd.to_numeric(work["pre_close"], errors="coerce").gt(0)
    )
    eligible = work.loc[valid].copy()
    eligible["is_up"] = eligible["close"].astype(float) > eligible["pre_close"].astype(float)
    return eligible.groupby("trade_date")["is_up"].mean().astype(float).to_dict()


def _prepare_corporate_actions(frame: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["event_id", "instrument_id", "ex_date", "cash_per_share", "share_ratio"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"corporate actions missing columns: {missing}")
    result = frame[columns].copy()
    result["ex_date"] = pd.to_datetime(result["ex_date"])
    result["instrument_id"] = result["instrument_id"].astype(str)
    result["cash_per_share"] = pd.to_numeric(result["cash_per_share"], errors="coerce").fillna(0.0)
    result["share_ratio"] = pd.to_numeric(result["share_ratio"], errors="coerce").fillna(0.0)
    return result


def _action_lookup(frame: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], list[dict[str, Any]]]:
    lookup: dict[tuple[pd.Timestamp, str], list[dict[str, Any]]] = {}
    for row in frame.to_dict("records"):
        lookup.setdefault((pd.Timestamp(row["ex_date"]), row["instrument_id"]), []).append(row)
    return lookup


def _market_row(lookup: pd.DataFrame, day: pd.Timestamp, instrument_id: str) -> pd.Series | None:
    try:
        row = lookup.loc[(day, instrument_id)]
    except KeyError:
        return None
    if isinstance(row, pd.DataFrame):
        raise ValueError(f"duplicate market row for {day.date()} {instrument_id}")
    return row


def _open_cohort(
    sleeve: _Sleeve,
    scheduled: dict[str, Any],
    day: pd.Timestamp,
    market_lookup: pd.DataFrame,
    config: FixedSleeveConfig,
    rejections: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    entry_portfolio_nav: float,
    entry_price_lookup: pd.Series | None,
) -> bool:
    picks = scheduled["picks"]
    cohort_id = f"{scheduled['signal_date'].date()}-sleeve-{sleeve.sleeve_id}"
    if config.rank_name_budget_nav_fractions is not None:
        target_name_budgets = [
            entry_portfolio_nav
            * config.rank_name_budget_nav_fractions[int(pick["selection_rank"]) - 1]
            * float(pick["allocation_multiplier"])
            for pick in picks
        ]
        total_target = sum(target_name_budgets)
        scale = min(
            1.0,
            sleeve.cash * config.cohort_cash_fraction / max(total_target, 1e-12),
        )
        per_name_cash_caps = [value * scale for value in target_name_budgets]
    else:
        target_name_budget = (
            entry_portfolio_nav * scheduled["name_budget_nav_fraction"]
            if scheduled["name_budget_nav_fraction"] is not None
            else config.name_budget
        )
        target_name_budgets = [
            target_name_budget * float(pick["allocation_multiplier"])
            for pick in picks
        ]
        total_target = sum(target_name_budgets)
        scale = min(
            1.0,
            sleeve.cash * config.cohort_cash_fraction / max(total_target, 1e-12),
        )
        per_name_cash_caps = [value * scale for value in target_name_budgets]
    opened = False
    for pick, target_name_budget, per_name_cash in zip(
        picks, target_name_budgets, per_name_cash_caps, strict=True
    ):
        instrument_id = str(pick["instrument_id"])
        bar = _market_row(market_lookup, day, instrument_id)
        reason = _buy_block_reason(bar, config.limit_up_tolerance)
        open_gap = np.nan
        if reason is None and (
            config.open_gap_min is not None
            or config.open_gap_max is not None
            or config.high_gap_threshold is not None
        ):
            signal_bar = _market_row(
                market_lookup, pd.Timestamp(scheduled["signal_date"]), instrument_id
            )
            if signal_bar is None or not _finite(signal_bar.get("close")):
                reason = "missing_signal_close"
            else:
                open_gap = float(bar["open"]) / float(signal_bar["close"]) - 1.0
                if config.open_gap_min is not None and open_gap < config.open_gap_min:
                    reason = "open_gap_too_low"
                elif config.open_gap_max is not None and open_gap > config.open_gap_max:
                    reason = "open_gap_too_high"
        if reason is not None:
            rejections.append({
                "event": "buy_rejected", "reason": reason,
                "signal_date": scheduled["signal_date"], "execution_date": day,
                "sleeve_id": sleeve.sleeve_id, "cohort_id": cohort_id,
                "instrument_id": instrument_id,
                "open_gap": open_gap,
            })
            continue
        entry_slippage_bps = config.costs.slippage_bps
        if (
            config.high_gap_threshold is not None
            and _finite(open_gap)
            and open_gap > config.high_gap_threshold
        ):
            entry_slippage_bps += config.high_gap_extra_entry_slippage_bps
        entry_reference_price = float(bar["open"])
        if entry_price_lookup is not None:
            key = (day, instrument_id)
            if key not in entry_price_lookup.index:
                raise RuntimeError(f"missing entry price override for {day.date()} {instrument_id}")
            entry_reference_price = float(entry_price_lookup.loc[key])
        price = _modeled_buy_reference_price(
            bar, entry_reference_price, entry_slippage_bps
        )
        quantity = _affordable_lot_quantity(per_name_cash, price, config)
        if quantity < config.lot_size:
            rejections.append({
                "event": "buy_rejected", "reason": "insufficient_cash_lot",
                "signal_date": scheduled["signal_date"], "execution_date": day,
                "sleeve_id": sleeve.sleeve_id, "cohort_id": cohort_id,
                "instrument_id": instrument_id,
            })
            continue
        notional = quantity * price
        commission = notional * config.costs.commission_bps / 10_000.0
        transfer_fee = notional * config.costs.transfer_fee_bps / 10_000.0
        total = notional + commission + transfer_fee
        if total > sleeve.cash + 1e-8:
            raise RuntimeError("buy would create negative sleeve cash")
        sleeve.cash -= total
        position_key = f"{cohort_id}|{instrument_id}"
        if position_key in sleeve.positions:
            raise RuntimeError(f"duplicate live position key: {position_key}")
        sleeve.positions[position_key] = _Position(
            cohort_id=cohort_id,
            instrument_id=instrument_id,
            signal_date=scheduled["signal_date"], entry_date=day,
            planned_exit_date=scheduled["planned_exit_date"],
            initial_quantity=quantity, quantity=quantity,
            entry_reference_price=entry_reference_price, entry_price=price,
            buy_notional=notional, buy_commission=commission,
            buy_transfer_fee=transfer_fee, buy_cost=total,
            score=float(pick["score"]), last_close=float(bar["close"]),
            entry_portfolio_nav=entry_portfolio_nav,
            target_name_budget=target_name_budget,
            allocation_multiplier=float(pick["allocation_multiplier"]),
        )
        cost_rows.append({
            "trade_date": day, "sleeve_id": sleeve.sleeve_id,
            "cohort_id": cohort_id, "instrument_id": instrument_id,
            "side": "buy", "commission": commission, "transfer_fee": transfer_fee,
            "stamp_tax": 0.0,
            "entry_slippage_bps": entry_slippage_bps,
            "slippage_cost": quantity * (price - entry_reference_price),
        })
        opened = True
    return opened


def _modeled_buy_reference_price(
    bar: pd.Series, reference: float, slippage_bps: float,
) -> float:
    modeled = float(reference) * (1.0 + float(slippage_bps) / 10_000.0)
    lower = max(float(bar["low"]), float(bar["limit_down"]) if _finite(bar.get("limit_down")) else -np.inf)
    upper = min(float(bar["high"]), float(bar["limit_up"]) if _finite(bar.get("limit_up")) else np.inf)
    return min(max(modeled, lower), upper)


def _portfolio_nav_at_open(
    sleeves: list[_Sleeve],
    market_lookup: pd.DataFrame,
    day: pd.Timestamp,
) -> float:
    """Causal portfolio NAV immediately before the day's new order."""
    value = float(sum(sleeve.cash for sleeve in sleeves))
    for sleeve in sleeves:
        for position in sleeve.positions.values():
            bar = _market_row(market_lookup, day, position.instrument_id)
            mark = (
                float(bar["open"])
                if bar is not None and _finite(bar.get("open"))
                else float(position.last_close)
            )
            value += float(position.quantity) * mark
    return value


def _affordable_lot_quantity(cash_cap: float, modeled_price: float, config: FixedSleeveConfig) -> int:
    buy_fee_rate = (config.costs.commission_bps + config.costs.transfer_fee_bps) / 10_000.0
    raw = int(cash_cap / (modeled_price * (1.0 + buy_fee_rate)))
    return raw // config.lot_size * config.lot_size


def _buy_block_reason(bar: pd.Series | None, tolerance: float) -> str | None:
    return open_buy_block_reason(bar, tolerance)


def _sell_block_reason(bar: pd.Series | None, tolerance: float) -> str | None:
    if bar is None:
        return "missing_market_row"
    if bool(bar["paused"]) if pd.notna(bar["paused"]) else False:
        return "paused"
    if not _finite(bar["volume"]) or float(bar["volume"]) <= 0:
        return "zero_volume"
    if not _finite(bar["close"]):
        return "invalid_close"
    if _finite(bar["limit_up"]) and float(bar["close"]) >= float(bar["limit_up"]) - tolerance:
        return "limit_up_hold"
    if _finite(bar["limit_down"]) and _finite(bar["high"]) and float(bar["high"]) <= float(bar["limit_down"]) + 1e-12:
        return "one_price_limit_down"
    return None


def _sell_open_block_reason(bar: pd.Series | None, tolerance: float) -> str | None:
    if bar is None:
        return "missing_market_row"
    if bool(bar["paused"]) if pd.notna(bar["paused"]) else False:
        return "paused"
    if not _finite(bar["volume"]) or float(bar["volume"]) <= 0:
        return "zero_volume"
    if not _finite(bar["open"]):
        return "invalid_open"
    if _finite(bar["limit_down"]) and float(bar["open"]) <= float(bar["limit_down"]) + tolerance:
        return "open_limit_down"
    return None


def _close_position(
    sleeve: _Sleeve,
    position: _Position,
    day: pd.Timestamp,
    bar: pd.Series,
    config: FixedSleeveConfig,
    *,
    reference_field: str = "close",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference = float(bar[reference_field])
    if reference_field == "close":
        price = _modeled_price(bar, "sell", config.costs.slippage_bps)
    elif reference_field == "open":
        price = reference * (1.0 - config.costs.slippage_bps / 10_000.0)
        if _finite(bar.get("limit_down")):
            price = max(price, float(bar["limit_down"]))
        if _finite(bar.get("limit_up")):
            price = min(price, float(bar["limit_up"]))
    else:
        raise ValueError(f"unsupported exit reference field: {reference_field}")
    notional = position.quantity * price
    commission = notional * config.costs.commission_bps / 10_000.0
    transfer_fee = notional * config.costs.transfer_fee_bps / 10_000.0
    stamp_tax = notional * config.costs.sell_tax_bps / 10_000.0
    proceeds = notional - commission - transfer_fee - stamp_tax
    sleeve.cash += proceeds
    ret = proceeds / position.buy_cost - 1.0
    trade = {
        "sleeve_id": sleeve.sleeve_id, "cohort_id": position.cohort_id,
        "signal_date": position.signal_date, "instrument_id": position.instrument_id,
        "entry_date": position.entry_date,
        "planned_exit_date": position.planned_exit_date, "actual_exit_date": day,
        "initial_quantity": position.initial_quantity, "exit_quantity": position.quantity,
        "entry_reference_price": position.entry_reference_price,
        "entry_price": position.entry_price, "exit_reference_price": reference,
        "exit_price": price, "exit_phase": reference_field,
        "exit_armed_on": position.exit_armed_on,
        "exit_trigger": position.exit_trigger,
        "buy_cost": position.buy_cost,
        "sell_proceeds": proceeds, "cash_dividend": position.cash_dividend,
        "buy_commission": position.buy_commission,
        "buy_transfer_fee": position.buy_transfer_fee,
        "sell_commission": commission, "sell_transfer_fee": transfer_fee,
        "stamp_tax": stamp_tax, "ret": ret, "score": position.score,
        "entry_portfolio_nav": position.entry_portfolio_nav,
        "target_name_budget": position.target_name_budget,
        "allocation_multiplier": position.allocation_multiplier,
    }
    cost = {
        "trade_date": day, "sleeve_id": sleeve.sleeve_id,
        "cohort_id": position.cohort_id, "instrument_id": position.instrument_id,
        "side": "sell", "commission": commission, "transfer_fee": transfer_fee,
        "stamp_tax": stamp_tax,
        "slippage_cost": position.quantity * (reference - price),
    }
    return trade, [cost]


def _apply_actions(
    day: pd.Timestamp,
    sleeves: list[_Sleeve],
    lookup: dict[tuple[pd.Timestamp, str], list[dict[str, Any]]],
    cashflows: list[dict[str, Any]],
) -> None:
    for sleeve in sleeves:
        for position in sleeve.positions.values():
            for action in lookup.get((day, position.instrument_id), []):
                cash = position.quantity * float(action["cash_per_share"])
                if cash:
                    sleeve.cash += cash
                    position.cash_dividend += cash
                old_quantity = position.quantity
                position.quantity = int(round(position.quantity * (1.0 + float(action["share_ratio"]))))
                cashflows.append({
                    "trade_date": day, "sleeve_id": sleeve.sleeve_id,
                    "cohort_id": position.cohort_id, "instrument_id": position.instrument_id,
                    "event_id": action["event_id"], "cash_dividend": cash,
                    "quantity_before": old_quantity, "quantity_after": position.quantity,
                })


def _build_report(
    frames: dict[str, pd.DataFrame],
    sleeves: list[_Sleeve],
    config: FixedSleeveConfig,
    skipped_insufficient_sleeve_cash: int,
    delayed: dict[str, int],
    market_gate_rejected_signals: int,
    score_gate_rejected_names: int,
) -> dict[str, Any]:
    nav = frames["nav"]
    values = nav["total_asset"].astype(float)
    drawdown = values / values.cummax() - 1.0
    costs = frames["costs"]
    trades = frames["trades"]
    sleeve_states = frames["sleeve_states"]
    terminal_positions = int(sum(len(sleeve.positions) for sleeve in sleeves))
    asset_error = float((nav["total_asset"] - nav["cash"] - nav["market_value"]).abs().max())
    year_end = nav.assign(year=nav["trade_date"].dt.year).groupby("year", sort=True).tail(1)
    yearly_returns: dict[str, float] = {}
    prior_asset = float(config.initial_cash)
    for row in year_end.itertuples(index=False):
        yearly_returns[str(int(row.year))] = float(row.total_asset / prior_asset - 1.0)
        prior_asset = float(row.total_asset)
    calendar_position = {
        day: offset for offset, day in enumerate(pd.DatetimeIndex(nav["trade_date"]))
    }
    holding_sessions = pd.Series(dtype="float64")
    if not trades.empty:
        holding_sessions = trades.apply(
            lambda row: calendar_position[pd.Timestamp(row["actual_exit_date"])]
            - calendar_position[pd.Timestamp(row["entry_date"])] + 1,
            axis=1,
        )
    cash_ratio = nav["cash"].astype(float) / nav["total_asset"].astype(float)
    report = {
        "status": "complete",
        "contract": (
            "two alternating sleeves initialized with 150k cash each; "
            + ("all supplied predictions" if config.top_k is None else f"Top{config.top_k}")
            + (
                f"; per-name budget={config.name_budget_nav_fraction:.2%} of entry NAV"
                if config.name_budget_nav_fraction is not None
                else f"; fixed per-name budget={config.name_budget:.2f}"
            )
            + (
                "; reduced per-name budget="
                f"{config.reduced_name_budget_nav_fraction:.2%} below market breadth "
                f"{config.market_up_ratio_full_allocation_min:.2%}"
                if config.market_up_ratio_full_allocation_min is not None
                else ""
            )
            + f"; exit_mode={config.exit_mode}; leftovers do not lock released cash"
        ),
        "config": {**asdict(config), "costs": asdict(config.costs)},
        "metrics": {
            "total_return": float(values.iloc[-1] / config.initial_cash - 1.0),
            "max_drawdown": float(drawdown.min()),
            "final_asset": float(values.iloc[-1]),
            "completed_trades": int(len(frames["trades"])),
            "trade_win_rate": float(trades["ret"].gt(0).mean()) if not trades.empty else None,
            "mean_trade_return": float(trades["ret"].mean()) if not trades.empty else None,
            "average_holding_sessions": (
                float(holding_sessions.mean()) if len(holding_sessions) else None
            ),
            "maximum_holding_sessions_realized": (
                int(holding_sessions.max()) if len(holding_sessions) else None
            ),
            "yearly_returns": yearly_returns,
            "max_live_sleeves": int(nav["live_sleeves"].max()),
            "days_over_two_live_sleeves": int(nav["live_sleeves"].gt(2).sum()),
            "skipped_signals_insufficient_sleeve_cash": int(skipped_insufficient_sleeve_cash),
            "market_gate_rejected_signals": int(market_gate_rejected_signals),
            "score_gate_rejected_names": int(score_gate_rejected_names),
            "open_gap_rejections": int(
                frames["rejections"]["reason"].isin(["open_gap_too_low", "open_gap_too_high"]).sum()
            ) if not frames["rejections"].empty else 0,
            "max_live_cohorts": int(sleeve_states["live_cohort_count"].groupby(sleeve_states["trade_date"]).sum().max()),
            "limit_up_delay_attempts": int(delayed["limit_up"]),
            "limit_down_delay_attempts": int(delayed["limit_down"]),
            "unavailable_sell_attempts": int(delayed["unavailable"]),
            "terminal_positions": terminal_positions,
            "minimum_cash": float(nav["cash"].min()),
            "minimum_cash_ratio": float(cash_ratio.min()),
            "mean_cash_ratio": float(cash_ratio.mean()),
            "minimum_sleeve_cash": float(sleeve_states["cash"].min()),
            "max_live_positions": int(
                sleeve_states.groupby("trade_date")["position_count"].sum().max()
            ),
            "cash_insufficient_buy_rejections": int(
                frames["rejections"]["reason"].eq("insufficient_cash_lot").sum()
            ) if not frames["rejections"].empty else 0,
            "ma5_exit_count": int(
                trades["exit_trigger"].eq("ma5_breach").sum()
            ) if not trades.empty and "exit_trigger" in trades else 0,
            "max_holding_exit_count": int(
                trades["exit_trigger"].eq("max_holding_sessions").sum()
            ) if not trades.empty and "exit_trigger" in trades else 0,
            "asset_identity_max_abs": asset_error,
            "commission": float(costs["commission"].sum()) if not costs.empty else 0.0,
            "transfer_fee": float(costs["transfer_fee"].sum()) if not costs.empty else 0.0,
            "stamp_tax": float(costs["stamp_tax"].sum()) if not costs.empty else 0.0,
            "slippage_cost": float(costs["slippage_cost"].sum()) if not costs.empty else 0.0,
            "sleeve_occupied_sessions": {
                str(sleeve.sleeve_id): sleeve.occupied_sessions for sleeve in sleeves
            },
        },
        "frame_rows": {name: int(len(frame)) for name, frame in frames.items()},
    }
    if report["metrics"]["max_live_sleeves"] > config.sleeve_count:
        raise RuntimeError("live sleeve count exceeded fixed sleeve count")
    if report["metrics"]["minimum_cash"] < -1e-8:
        raise RuntimeError("negative portfolio cash")
    if asset_error > 1e-6:
        raise RuntimeError("asset identity failed")
    return report


def frame_sha256(frame: pd.DataFrame) -> str:
    """Stable content hash used by the fixed-sleeve artifact writer."""
    payload = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, default=str)


def _finite(value: Any) -> bool:
    return pd.notna(value) and np.isfinite(float(value))
