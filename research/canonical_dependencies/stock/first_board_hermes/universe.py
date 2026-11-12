"""Point-in-time universe eligibility for first-board research and execution."""

from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd
from zyquant.core.exceptions import DataContractError

ALL_A = "CN_ALL_A"
EX_ST = "CN_A_EX_ST"


def resolve_daily_members(
    membership: pd.DataFrame,
    special_treatment: pd.DataFrame | None,
    days: Iterable[date],
    *,
    universe_id: str = EX_ST,
) -> dict[date, set[str]]:
    """Resolve the eligible set on every day without using later disclosures.

    ``special_treatment`` is a transition log.  For a given day we use only
    transitions whose effective date and disclosure/known date have both
    arrived, then apply the latest such transition for each instrument.
    """
    requested_days = sorted({_as_date(day) for day in days})
    if not requested_days:
        return {}
    windows = _normalise_membership(membership)
    if universe_id != EX_ST:
        return _membership_sets(windows, universe_id, requested_days)

    base_rows = windows[windows["universe_id"].eq(ALL_A)]
    exact_rows = windows[windows["universe_id"].eq(EX_ST)]
    if base_rows.empty:
        if exact_rows.empty:
            raise DataContractError("snapshot has neither CN_ALL_A nor CN_A_EX_ST")
        # Compatibility for small synthetic snapshots. Production snapshots
        # must contain CN_ALL_A so the dynamic ex-ST result can be derived.
        return _membership_sets(windows, EX_ST, requested_days)

    base = _membership_sets(windows, ALL_A, requested_days)
    transitions = _normalise_special_treatment(special_treatment)
    derived: dict[date, set[str]] = {}
    for day in requested_days:
        day_ts = pd.Timestamp(day)
        active_st: set[str] = set()
        if not transitions.empty:
            visible = transitions[
                transitions["effective_from"].le(day_ts)
                & transitions["known_at"].le(day_ts)
            ]
            if not visible.empty:
                latest = (
                    visible.sort_values(
                        ["instrument_id", "effective_from", "known_at", "_order"]
                    )
                    .drop_duplicates("instrument_id", keep="last")
                )
                flagged = latest["name"].fillna("").astype(str).str.contains(
                    "ST", regex=False
                )
                active_st = set(latest.loc[flagged, "instrument_id"])
        derived[day] = base[day] - active_st

    if not exact_rows.empty:
        exact = _membership_sets(windows, EX_ST, requested_days)
        mismatches = [
            (day, sorted(derived[day] - exact[day]), sorted(exact[day] - derived[day]))
            for day in requested_days
            if derived[day] != exact[day]
        ]
        if mismatches:
            day, derived_only, exact_only = mismatches[0]
            raise DataContractError(
                "CN_A_EX_ST disagrees with PIT special_treatment resolution on "
                f"{day}: derived_only={derived_only[:10]}, exact_only={exact_only[:10]} "
                f"({len(mismatches)} mismatched days)"
            )
    return derived


def eligibility_mask(
    trade_dates: pd.Series,
    instrument_ids: pd.Series,
    eligible: dict[date, set[str]],
) -> pd.Series:
    days = pd.to_datetime(trade_dates).dt.date
    ids = instrument_ids.astype(str)
    return pd.Series(
        [asset_id in eligible.get(day, set()) for day, asset_id in zip(days, ids)],
        index=trade_dates.index,
        dtype=bool,
    )


def _membership_sets(
    windows: pd.DataFrame, universe_id: str, days: list[date]
) -> dict[date, set[str]]:
    selected = windows[windows["universe_id"].eq(universe_id)]
    return {
        day: set(
            selected.loc[
                selected["effective_from"].le(pd.Timestamp(day))
                & (
                    selected["effective_to"].isna()
                    | selected["effective_to"].ge(pd.Timestamp(day))
                ),
                "instrument_id",
            ]
        )
        for day in days
    }


def _normalise_membership(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"universe_id", "instrument_id", "effective_from", "effective_to"}
    missing = required - set(frame.columns)
    if missing:
        raise DataContractError(f"universe_membership missing columns: {sorted(missing)}")
    out = frame[list(required)].copy()
    out["universe_id"] = out["universe_id"].astype(str)
    out["instrument_id"] = out["instrument_id"].astype(str)
    out["effective_from"] = pd.to_datetime(out["effective_from"])
    out["effective_to"] = pd.to_datetime(out["effective_to"], errors="coerce")
    return out


def _normalise_special_treatment(frame: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["instrument_id", "effective_from", "known_at", "name", "_order"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    required = {"instrument_id", "effective_from", "name"}
    missing = required - set(frame.columns)
    if missing:
        raise DataContractError(f"special_treatment missing columns: {sorted(missing)}")
    out = frame.copy().reset_index(drop=True)
    out["_order"] = out.index
    out["instrument_id"] = out["instrument_id"].astype(str)
    out["effective_from"] = pd.to_datetime(out["effective_from"])
    known = out["known_at"] if "known_at" in out else out["effective_from"]
    out["known_at"] = pd.to_datetime(known)
    return out[columns]


def _as_date(value) -> date:
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()
