"""AI_07.21 candidate-universe alignment for the Hermes first-board panel."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .dataset import KEYS

AI071_FILTERS = {
    "min_listing_days": 60,
    "min_amount": 50_000_000.0,
    "min_float_mv": 1_000_000_000.0,
    "exclude_one_word_first_board": True,
    "exclude_bj": True,
    "exclude_resume_anomaly": True,
}


def apply_ai071_candidate_filters(
    panel: pd.DataFrame,
    daily_raw: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    min_listing_days: int = 60,
    min_amount: float = 50_000_000.0,
    min_float_mv: float = 1_000_000_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply the original AI_07.21 research filters in their recorded order.

    ``panel`` is the hard first-board panel before PIT-ST filtering. Required
    source values are joined from the published Hermes snapshot. Missing
    filter inputs fail closed and are retained in the row-level audit.
    """
    required_panel = {
        *KEYS,
        "is_first_board_hard",
        "universe_eligible",
        "g7_listing_days_log",
        "g0_float_mv_log",
    }
    required_raw = {
        *KEYS,
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "paused",
        "limit_up",
    }
    required_calendar = {"trade_date"}
    missing_panel = sorted(required_panel - set(panel.columns))
    missing_raw = sorted(required_raw - set(daily_raw.columns))
    missing_calendar = sorted(required_calendar - set(calendar.columns))
    if missing_panel or missing_raw or missing_calendar:
        raise ValueError(
            "missing AI_07.21 filter inputs: "
            f"panel={missing_panel}, raw={missing_raw}, calendar={missing_calendar}"
        )
    if panel.duplicated(KEYS).any() or daily_raw.duplicated(KEYS).any():
        raise ValueError("duplicate instrument/date keys in candidate filter inputs")

    work = panel.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work["instrument_id"] = work["instrument_id"].astype(str)
    raw = daily_raw.copy()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    raw["instrument_id"] = raw["instrument_id"].astype(str)

    signal = raw[[*KEYS, "open", "high", "low", "close", "volume", "amount", "paused", "limit_up"]]
    signal = signal.rename(columns={name: f"signal_{name}" for name in signal.columns if name not in KEYS})
    work = work.merge(signal, on=KEYS, how="left", validate="one_to_one")

    dates = pd.DatetimeIndex(pd.to_datetime(calendar["trade_date"]).drop_duplicates().sort_values())
    previous = pd.Series(dates[:-1].to_numpy(), index=dates[1:])
    work["previous_market_date"] = work["trade_date"].map(previous)
    previous_state = raw[[*KEYS, "amount", "volume", "paused"]].rename(
        columns={
            "trade_date": "previous_market_date",
            "amount": "previous_amount",
            "volume": "previous_volume",
            "paused": "previous_paused",
        }
    )
    work = work.merge(
        previous_state,
        on=["previous_market_date", "instrument_id"],
        how="left",
        validate="one_to_one",
    )

    work["listing_days"] = np.expm1(
        pd.to_numeric(work["g7_listing_days_log"], errors="coerce")
    )
    work["float_mv"] = np.expm1(
        pd.to_numeric(work["g0_float_mv_log"], errors="coerce")
    )
    work["is_one_word_first_board"] = (
        pd.to_numeric(work["signal_low"], errors="coerce")
        >= pd.to_numeric(work["signal_limit_up"], errors="coerce") - 1e-4
    )
    previous_invalid = (
        work["previous_market_date"].isna()
        | work["previous_amount"].isna()
        | work["previous_volume"].isna()
        | work["previous_paused"].isna()
        | pd.to_numeric(work["previous_amount"], errors="coerce").le(0)
        | pd.to_numeric(work["previous_volume"], errors="coerce").le(0)
        | work["previous_paused"].fillna(True).astype(bool)
    )

    work["candidate_pass"] = True
    work["candidate_reject_reason"] = ""
    layers: list[dict[str, Any]] = []

    def reject(layer: str, fail: pd.Series, reason: str) -> None:
        before = int(work["candidate_pass"].sum())
        mask = work["candidate_pass"] & fail.fillna(True).astype(bool)
        work.loc[mask, "candidate_pass"] = False
        work.loc[mask, "candidate_reject_reason"] = reason
        after = int(work["candidate_pass"].sum())
        layers.append({
            "layer": layer,
            "before": before,
            "rejected": before - after,
            "after": after,
            "reason": reason,
        })

    reject("C0_first_board_hard", ~work["is_first_board_hard"].fillna(False), "not_hard_first_board")
    reject("filter_ST", ~work["universe_eligible"].fillna(False), "ST")
    listing_days = pd.to_numeric(work["listing_days"], errors="coerce")
    signal_amount = pd.to_numeric(work["signal_amount"], errors="coerce")
    float_mv = pd.to_numeric(work["float_mv"], errors="coerce")
    reject(
        "filter_listing_days",
        listing_days.isna() | listing_days.lt(min_listing_days),
        "listing_days",
    )
    reject(
        "filter_amount", signal_amount.isna() | signal_amount.lt(min_amount), "amount"
    )
    reject(
        "filter_float_mv", float_mv.isna() | float_mv.lt(min_float_mv), "float_mv"
    )
    one_word_input_missing = work[["signal_low", "signal_limit_up"]].isna().any(axis=1)
    reject(
        "filter_one_word",
        one_word_input_missing | work["is_one_word_first_board"],
        "one_word_board",
    )
    reject(
        "filter_bj",
        work["instrument_id"].str.endswith((".XBEI", ".XBSE")),
        "bj_board",
    )
    reject("filter_resume_anomaly", previous_invalid, "resume_anomaly")

    passed = work.loc[work["candidate_pass"]].copy()
    audit_columns = [
        *KEYS,
        "candidate_pass",
        "candidate_reject_reason",
        "listing_days",
        "signal_amount",
        "float_mv",
        "is_one_word_first_board",
        "previous_market_date",
        "previous_amount",
        "previous_volume",
        "previous_paused",
    ]
    detail = work[audit_columns].sort_values(KEYS).reset_index(drop=True)
    summary = {
        "contract": "AI_07.21 original candidate research filters after hard first-board definition",
        "parameters": {
            "min_listing_days": min_listing_days,
            "min_amount": min_amount,
            "min_float_mv": min_float_mv,
            "exclude_one_word_first_board": True,
            "exclude_bj": True,
            "exclude_resume_anomaly": True,
            "missing_filter_inputs": "fail_closed",
        },
        "layers": layers,
        "rows_before": len(work),
        "rows_after": len(passed),
        "reject_reasons": {
            str(key): int(value)
            for key, value in detail.loc[~detail["candidate_pass"], "candidate_reject_reason"]
            .value_counts()
            .items()
        },
    }
    drop_columns = [column for column in work.columns if column.startswith("signal_")]
    drop_columns += [
        "previous_market_date",
        "previous_amount",
        "previous_volume",
        "previous_paused",
        "listing_days",
        "float_mv",
        "is_one_word_first_board",
        "candidate_pass",
        "candidate_reject_reason",
    ]
    return passed.drop(columns=drop_columns), detail, summary
