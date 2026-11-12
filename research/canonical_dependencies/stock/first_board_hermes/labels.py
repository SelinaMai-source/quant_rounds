"""Execution-aligned labels for the first-board dual-sleeve strategy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .dataset import KEYS


@dataclass(frozen=True)
class ExitLabelCosts:
    slippage_bps: float = 12.3
    commission_bps: float = 1.0
    transfer_fee_bps: float = 0.1
    sell_tax_bps: float = 5.0


def attach_exit_aligned_labels(
    panel: pd.DataFrame,
    market: pd.DataFrame,
    *,
    costs: ExitLabelCosts = ExitLabelCosts(),
    limit_up_tolerance: float = 0.005,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach D+1-open to first executable close from D+2 onward.

    ``cal_idx`` must use the same exchange calendar in both inputs. Entry and
    exit tradability use the same predicates as the backtest executor. A sell
    is delayed for a limit-up close, suspension/zero volume, or one-price
    limit-down.
    """
    required_panel = {*KEYS, "cal_idx"}
    required_market = {
        *KEYS, "cal_idx", "open", "high", "low", "close", "volume",
        "paused", "limit_up", "limit_down",
    }
    missing_panel = sorted(required_panel - set(panel.columns))
    missing_market = sorted(required_market - set(market.columns))
    if missing_panel or missing_market:
        raise ValueError(
            f"missing label inputs: panel={missing_panel}, market={missing_market}"
        )
    if panel.duplicated(KEYS).any() or market.duplicated(KEYS).any():
        raise ValueError("duplicate instrument/date keys in label inputs")

    result = panel.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    prices = market.copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices["instrument_id"] = prices["instrument_id"].astype(str)
    prices = prices.sort_values(["instrument_id", "cal_idx"])
    by_instrument = {
        str(instrument): block.set_index("cal_idx", drop=False)
        for instrument, block in prices.groupby("instrument_id", sort=False)
    }
    max_cal_idx = int(prices["cal_idx"].max())

    values: list[dict[str, Any]] = []
    for row in result.itertuples(index=False):
        instrument = str(row.instrument_id)
        signal_idx = int(row.cal_idx)
        block = by_instrument.get(instrument)
        payload: dict[str, Any] = {
            "actual_entry_date": pd.NaT,
            "actual_exit_date": pd.NaT,
            "exit_signal_offset": np.nan,
            "holding_sessions": np.nan,
            "label_exit_rule_net": np.nan,
            "label_exit_rule_reason": "missing_market_history",
        }
        if block is None or signal_idx + 1 not in block.index:
            values.append(payload)
            continue
        buy = block.loc[signal_idx + 1]
        if isinstance(buy, pd.DataFrame):
            raise ValueError(f"duplicate market cal_idx for {instrument}")
        buy_block_reason = open_buy_block_reason(buy, limit_up_tolerance)
        if buy_block_reason is not None:
            payload["label_exit_rule_reason"] = f"buy_{buy_block_reason}"
            values.append(payload)
            continue
        buy_price = _modeled_price(buy, "buy", costs.slippage_bps)
        payload["actual_entry_date"] = buy["trade_date"]

        sell = None
        last_reason = "no_executable_exit_before_cutoff"
        for exit_idx in range(signal_idx + 2, max_cal_idx + 1):
            if exit_idx not in block.index:
                continue
            candidate = block.loc[exit_idx]
            paused = bool(candidate["paused"]) if pd.notna(candidate["paused"]) else False
            if paused or not np.isfinite(float(candidate["volume"])) or float(candidate["volume"]) <= 0:
                last_reason = "sell_paused_or_zero_volume"
                continue
            sell_limit_up = _finite_or_none(candidate["limit_up"])
            if (
                sell_limit_up is not None
                and float(candidate["close"]) >= sell_limit_up - limit_up_tolerance
            ):
                last_reason = "sell_limit_up_hold"
                continue
            sell_limit_down = _finite_or_none(candidate["limit_down"])
            if sell_limit_down is not None and float(candidate["high"]) <= sell_limit_down + 1e-12:
                last_reason = "sell_one_price_limit_down"
                continue
            sell = candidate
            break
        if sell is None:
            payload["label_exit_rule_reason"] = last_reason
            values.append(payload)
            continue

        sell_price = _modeled_price(sell, "sell", costs.slippage_bps)
        buy_fee = (costs.commission_bps + costs.transfer_fee_bps) / 10_000.0
        sell_fee = (
            costs.commission_bps + costs.transfer_fee_bps + costs.sell_tax_bps
        ) / 10_000.0
        payload.update({
            "actual_exit_date": sell["trade_date"],
            "exit_signal_offset": int(sell["cal_idx"] - signal_idx),
            "holding_sessions": int(sell["cal_idx"] - (signal_idx + 1)),
            "label_exit_rule_net": sell_price * (1.0 - sell_fee) / (
                buy_price * (1.0 + buy_fee)
            ) - 1.0,
            "label_exit_rule_reason": "ok",
        })
        values.append(payload)

    attached = pd.DataFrame(values, index=result.index)
    for column in attached:
        result[column] = attached[column]
    audit = {
        "contract": "D+1 modeled open buy; first executable modeled close from D+2; limit-up close held",
        "costs": asdict(costs),
        "rows": int(len(result)),
        "valid_labels": int(result["label_exit_rule_net"].notna().sum()),
        "reasons": {
            str(k): int(v)
            for k, v in result["label_exit_rule_reason"].value_counts(dropna=False).items()
        },
        "delayed_exit_rows": int(result["exit_signal_offset"].gt(2).sum()),
        "maximum_exit_signal_offset": int(result["exit_signal_offset"].max())
        if result["exit_signal_offset"].notna().any() else None,
    }
    return result, audit


def attach_board_chain_labels(
    panel: pd.DataFrame,
    market: pd.DataFrame,
    *,
    limit_up_tolerance: float = 0.005,
    maximum_relevance: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach an execution-eligible continuation-board relevance label.

    The signal day is already the first close-limit day.  Consecutive close
    limits are counted from D+1.  A one-day continuation is therefore a
    second board (relevance 2), two days is a third board (relevance 3), and
    so on.  Five boards and above are capped at relevance 5.  If D+1 does not
    close at limit, a profitable execution-aligned exit receives relevance 1
    and a non-profitable exit receives relevance 0.

    Rows without a valid execution-aligned label remain unavailable.  The
    label maturity timestamp is the actual exit date, so a continuing board
    chain cannot enter a fit before its outcome is observable.
    """
    if not 2 <= int(maximum_relevance) <= 31:
        raise ValueError("maximum_relevance must be between 2 and 31")
    required_panel = {
        *KEYS, "cal_idx", "label_exit_rule_net", "actual_exit_date",
    }
    required_market = {
        *KEYS, "cal_idx", "close", "volume", "paused", "limit_up",
    }
    missing_panel = sorted(required_panel - set(panel.columns))
    missing_market = sorted(required_market - set(market.columns))
    if missing_panel or missing_market:
        raise ValueError(
            f"missing board-chain inputs: panel={missing_panel}, market={missing_market}"
        )
    if panel.duplicated(KEYS).any() or market.duplicated(KEYS).any():
        raise ValueError("duplicate instrument/date keys in board-chain inputs")

    result = panel.copy()
    prices = market.copy()
    prices["instrument_id"] = prices["instrument_id"].astype(str)
    by_instrument = {
        str(instrument): block.set_index("cal_idx", drop=False)
        for instrument, block in prices.groupby("instrument_id", sort=False)
    }
    chain_lengths: list[float] = []
    relevances: list[float] = []
    reasons: list[str] = []
    for row in result.itertuples(index=False):
        exit_return = _finite_or_none(getattr(row, "label_exit_rule_net"))
        actual_exit = getattr(row, "actual_exit_date")
        if exit_return is None or pd.isna(actual_exit):
            chain_lengths.append(np.nan)
            relevances.append(np.nan)
            reasons.append("execution_label_unavailable")
            continue
        block = by_instrument.get(str(row.instrument_id))
        if block is None:
            chain_lengths.append(np.nan)
            relevances.append(np.nan)
            reasons.append("missing_market_history")
            continue
        chain = 0
        next_idx = int(row.cal_idx) + 1
        while next_idx in block.index:
            bar = block.loc[next_idx]
            if isinstance(bar, pd.DataFrame):
                raise ValueError(
                    f"duplicate market cal_idx for {row.instrument_id} at {next_idx}"
                )
            paused = bool(bar["paused"]) if pd.notna(bar["paused"]) else False
            volume = _finite_or_none(bar["volume"])
            close = _finite_or_none(bar["close"])
            limit_up = _finite_or_none(bar["limit_up"])
            if (
                paused or volume is None or volume <= 0 or close is None
                or limit_up is None or close < limit_up - limit_up_tolerance
            ):
                break
            chain += 1
            next_idx += 1
        relevance = (
            min(int(maximum_relevance), chain + 1)
            if chain > 0
            else int(exit_return > 0.0)
        )
        chain_lengths.append(float(chain))
        relevances.append(float(relevance))
        reasons.append("ok")

    result["future_consecutive_close_limit_days"] = chain_lengths
    result["label_board_chain_relevance"] = pd.Series(
        relevances, index=result.index, dtype="Float64"
    )
    result["board_chain_label_end_date"] = pd.to_datetime(
        result["actual_exit_date"]
    ).where(result["label_board_chain_relevance"].notna())
    result["label_board_chain_reason"] = reasons
    valid = result["label_board_chain_relevance"].dropna().astype(int)
    audit = {
        "contract": (
            "D signal first board; consecutive tradable close-limit sessions from D+1; "
            f"relevance capped at {int(maximum_relevance)}; no continuation maps to "
            "profit sign; maturity is actual_exit_date"
        ),
        "rows": int(len(result)),
        "valid_labels": int(len(valid)),
        "maximum_relevance": int(maximum_relevance),
        "relevance_counts": {
            str(k): int(v) for k, v in valid.value_counts().sort_index().items()
        },
        "maximum_observed_chain": (
            int(result["future_consecutive_close_limit_days"].max())
            if result["future_consecutive_close_limit_days"].notna().any()
            else None
        ),
        "reasons": {
            str(k): int(v)
            for k, v in result["label_board_chain_reason"].value_counts().items()
        },
    }
    return result, audit


def _finite_or_none(value: Any) -> float | None:
    return float(value) if pd.notna(value) and np.isfinite(float(value)) else None


def open_buy_block_reason(
    bar: pd.Series | None,
    limit_up_tolerance: float,
) -> str | None:
    """Return the shared D+1-open buy rejection reason.

    Labels and execution must call this exact predicate. In particular, an
    opening limit-up is not a hypothetical fill merely because the board opens
    later in the session.
    """
    if bar is None:
        return "missing_market_row"
    if bool(bar["paused"]) if pd.notna(bar.get("paused")) else False:
        return "paused"
    if not np.isfinite(float(bar.get("volume", np.nan))) or float(bar["volume"]) <= 0:
        return "zero_volume"
    if not np.isfinite(float(bar.get("open", np.nan))):
        return "invalid_open"
    limit_up = _finite_or_none(bar.get("limit_up"))
    if limit_up is not None and float(bar["open"]) >= limit_up - limit_up_tolerance:
        return "open_limit_up"
    return None


def _modeled_price(row: pd.Series, side: str, slippage_bps: float) -> float:
    reference = float(row["open"] if side == "buy" else row["close"])
    sign = 1.0 if side == "buy" else -1.0
    modeled = reference * (1.0 + sign * slippage_bps / 10_000.0)
    lower = max(float(row["low"]), _finite_or_none(row["limit_down"]) or -np.inf)
    upper = min(float(row["high"]), _finite_or_none(row["limit_up"]) or np.inf)
    return min(max(modeled, lower), upper)
