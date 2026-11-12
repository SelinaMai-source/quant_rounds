"""Auditable v7-native expansion cache for the first-board strategy."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from zyquant.core.hashing import canonical_json, hash_file
from zyquant.data import DataSnapshot

from .universe import EX_ST, eligibility_mask, resolve_daily_members

from .native_factors import (
    KEYS,
    _assert_unique,
    _row_key_hash,
    _safe_ratio,
    _source_reference_audit,
    apply_security_repairs,
)

FORMULA_VERSION = "first_board_hermes_expansion_v1"
GENERIC_FEATURES = [
    "h8_ret_3",
    "h8_volatility_5",
    "h8_downside_volatility_10",
    "h8_max_drawdown_20",
    "h8_amount_zscore",
    "h8_turnover_rate_zscore",
    "h8_volume_price_sync",
    "h8_close_range_pct_20",
    "h8_pullback_from_20d_high",
    "h8_market_heat_score",
    "h8_bp",
]
LIMIT_FEATURES = [
    "h5_first_limit_value_to_amount",
    "h5_close_limit_retention",
    "h5_last_first_limit_value_ratio",
]
INTRADAY_FEATURES = ["h6_weighted_close_ratio"]
EXPANSION_FEATURES = [*GENERIC_FEATURES, *LIMIT_FEATURES, *INTRADAY_FEATURES]
REQUIRED_COLUMNS = [*KEYS, *EXPANSION_FEATURES]


def _rolling(grouped: pd.core.groupby.SeriesGroupBy, window: int, operation: str) -> pd.Series:
    values = getattr(grouped.rolling(window, min_periods=window), operation)()
    return values.reset_index(level=0, drop=True).sort_index()


def compute_expansion_factors(
    raw: pd.DataFrame,
    adjusted: pd.DataFrame,
    valuation: pd.DataFrame,
    limit_events: pd.DataFrame,
    intraday: pd.DataFrame,
    eligible: pd.Series,
) -> pd.DataFrame:
    """Compute the frozen 15-factor block on the complete raw key space."""
    raw = raw.copy()
    adjusted = adjusted.copy()
    valuation = valuation.copy()
    limit_events = limit_events.copy()
    intraday = intraday.copy()
    for frame in (raw, adjusted, valuation, limit_events, intraday):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    raw = raw.sort_values(KEYS).reset_index(drop=True)
    eligible = pd.Series(eligible, index=raw.index, dtype=bool)
    _assert_unique("daily_raw", raw)
    _assert_unique("daily_post_adjusted", adjusted)
    _assert_unique("daily_valuation", valuation)

    base = raw.merge(
        adjusted[[*KEYS, "close_post"]], on=KEYS, how="left", validate="one_to_one"
    ).merge(
        valuation[[*KEYS, "turnover_rate", "pb"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    traded = (
        ~base["paused"].fillna(True).astype(bool)
        & pd.to_numeric(base["volume"], errors="coerce").gt(0)
        & pd.to_numeric(base["close_post"], errors="coerce").notna()
    )
    ts = base.loc[traded, [*KEYS, "close_post", "volume"]].copy()
    ts = ts.sort_values(KEYS)
    close = pd.to_numeric(ts["close_post"], errors="coerce")
    volume = pd.to_numeric(ts["volume"], errors="coerce")
    ids = ts["instrument_id"]
    close_group = close.groupby(ids, sort=False)
    ret1 = close_group.pct_change(1, fill_method=None)
    ts["h8_ret_3"] = close_group.pct_change(3, fill_method=None)
    ts["h8_volatility_5"] = _rolling(ret1.groupby(ids, sort=False), 5, "std")
    downside = ret1.where(ret1.lt(0), 0.0)
    ts["h8_downside_volatility_10"] = _rolling(
        downside.groupby(ids, sort=False), 10, "std"
    )
    rolling_high = _rolling(close_group, 20, "max")
    point_drawdown = close / rolling_high.replace(0, np.nan) - 1.0
    ts["h8_max_drawdown_20"] = _rolling(
        point_drawdown.groupby(ids, sort=False), 20, "min"
    )
    rolling_low = _rolling(close_group, 20, "min")
    ts["h8_close_range_pct_20"] = (close - rolling_low) / (
        rolling_high - rolling_low
    ).replace(0, np.nan)
    ts["h8_pullback_from_20d_high"] = close / rolling_high.replace(0, np.nan) - 1.0
    volume_change = volume.groupby(ids, sort=False).pct_change(1, fill_method=None)
    ts["h8_volume_price_sync"] = (
        (ret1.gt(0) & volume_change.gt(0)) | (ret1.lt(0) & volume_change.lt(0))
    ).astype(float)
    ts_values = ts[[*KEYS, *[
        name for name in GENERIC_FEATURES
        if name not in {"h8_amount_zscore", "h8_turnover_rate_zscore", "h8_market_heat_score", "h8_bp"}
    ]]]
    output = base[[*KEYS]].merge(ts_values, on=KEYS, how="left", validate="one_to_one")

    def zscore(values: pd.Series) -> pd.Series:
        selected = pd.to_numeric(values, errors="coerce").where(eligible)
        grouped = selected.groupby(base["trade_date"], sort=False)
        return (selected - grouped.transform("mean")) / grouped.transform("std").replace(0, np.nan)

    output["h8_amount_zscore"] = zscore(base["amount"])
    output["h8_turnover_rate_zscore"] = zscore(base["turnover_rate"])
    output["h8_bp"] = _safe_ratio(pd.Series(1.0, index=base.index), base["pb"])

    ret1_full = output[[*KEYS]].merge(
        ts[[*KEYS]].assign(ret1=ret1), on=KEYS, how="left", validate="one_to_one"
    )["ret1"]
    close5 = close / _rolling(close_group, 5, "mean").replace(0, np.nan) - 1.0
    close20 = close / _rolling(close_group, 20, "mean").replace(0, np.nan) - 1.0
    trends = ts[[*KEYS]].copy()
    trends["above5"] = close5.gt(0).astype(float)
    trends["above20"] = close20.gt(0).astype(float)
    trends = output[[*KEYS]].merge(trends, on=KEYS, how="left", validate="one_to_one")
    daily = pd.DataFrame({
        "trade_date": base["trade_date"],
        "eligible": eligible,
        "up": ret1_full.gt(0).astype(float),
        "above5": trends["above5"],
        "above20": trends["above20"],
    })
    pop = daily.loc[daily["eligible"]]
    heat = pop.groupby("trade_date")[["up", "above5", "above20"]].mean().mean(axis=1)
    output["h8_market_heat_score"] = base["trade_date"].map(heat)

    up_limits = limit_events.loc[
        limit_events["limit_type"].astype(str).str.lower().eq("up")
    ].copy()
    _assert_unique("daily_limit_events[limit_type=up]", up_limits)
    limit_values = up_limits[[
        *KEYS, "first_limit_value", "last_limit_value", "max_limit_value",
        "close_limit_value",
    ]]
    output = output.merge(limit_values, on=KEYS, how="left", validate="one_to_one")
    output["h5_first_limit_value_to_amount"] = _safe_ratio(
        output.pop("first_limit_value"), base["amount"]
    )
    output["h5_close_limit_retention"] = _safe_ratio(
        output.pop("close_limit_value"), output.pop("max_limit_value")
    )
    output["h5_last_first_limit_value_ratio"] = _safe_ratio(
        output.pop("last_limit_value"),
        up_limits.set_index(KEYS)["first_limit_value"].reindex(
            pd.MultiIndex.from_frame(output[KEYS])
        ).reset_index(drop=True),
    )
    _assert_unique("daily_intraday_factors", intraday)
    output = output.merge(
        intraday[[*KEYS, "weighted_close_ratio"]].rename(
            columns={"weighted_close_ratio": "h6_weighted_close_ratio"}
        ),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    output[EXPANSION_FEATURES] = output[EXPANSION_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    _assert_unique("Hermes expansion cache", output)
    return output[[*KEYS, *EXPANSION_FEATURES]].sort_values(KEYS).reset_index(drop=True)


def build_expansion_factor_cache(
    snapshot_path: Path,
    output_path: Path,
    *,
    repair_manifest: Path | None = None,
    start: date = date(2014, 1, 1),
    end: date = date(2026, 7, 24),
    cutoff: date = date(2026, 7, 24),
) -> dict[str, Any]:
    snapshot = DataSnapshot(snapshot_path, verify_hashes=True)
    raw = snapshot.table(
        "daily_raw", start=start, end=end, cutoff=cutoff,
        fields=[*KEYS, "volume", "amount", "paused"],
    )
    adjusted = snapshot.table(
        "daily_post_adjusted", start=start, end=end, cutoff=cutoff,
        fields=[*KEYS, "close_post"],
    )
    valuation = snapshot.table(
        "daily_valuation", start=start, end=end, cutoff=cutoff,
        fields=[*KEYS, "turnover_rate", "pb"],
    )
    limits = snapshot.table(
        "daily_limit_events", start=start, end=end, cutoff=cutoff,
        fields=[*KEYS, "limit_type", "first_limit_value", "last_limit_value", "max_limit_value", "close_limit_value"],
    )
    intraday = snapshot.table(
        "daily_intraday_factors", start=start, end=end, cutoff=cutoff,
        fields=[*KEYS, "weighted_close_ratio", "source_record_id"],
    )
    for frame in (raw, adjusted, valuation, limits, intraday):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    raw = raw.sort_values(KEYS).reset_index(drop=True)
    repair_audit: list[dict[str, Any]] = []
    repair_metadata = None
    if repair_manifest is not None:
        repair_metadata = json.loads(repair_manifest.read_text(encoding="utf-8"))
        if repair_metadata.get("snapshot_fingerprint") != snapshot.metadata.fingerprint:
            raise RuntimeError("security repair manifest parent snapshot mismatch")
        intraday, repair_audit = apply_security_repairs(
            intraday, repair_metadata, source_table="daily_intraday_factors"
        )
    reference_audit = _source_reference_audit(
        raw,
        {
            "daily_post_adjusted": adjusted,
            "daily_valuation": valuation,
            "daily_limit_events": limits,
            "daily_intraday_factors": intraday,
        },
    )
    if any(reference_audit.values()):
        raise RuntimeError(f"expansion factor sources contain orphan keys: {reference_audit}")
    membership = snapshot.table("universe_membership", cutoff=cutoff)
    special = snapshot.table("special_treatment", cutoff=cutoff)
    calendar = sorted(pd.to_datetime(raw["trade_date"]).dt.date.unique())
    members = resolve_daily_members(membership, special, calendar, universe_id=EX_ST)
    eligible = eligibility_mask(raw["trade_date"], raw["instrument_id"], members)
    factors = compute_expansion_factors(
        raw, adjusted, valuation, limits, intraday, eligible
    )
    if factors.duplicated(KEYS).any() or factors[KEYS].isna().any(axis=None):
        raise RuntimeError("expansion cache has duplicate or null keys")
    inf_count = int(np.isinf(factors[EXPANSION_FEATURES].to_numpy(dtype=float)).sum())
    if inf_count:
        raise RuntimeError(f"expansion cache contains {inf_count} infinite values")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    factors.to_parquet(temporary, index=False)
    os.replace(temporary, output_path)
    metadata = {
        "status": "ready_for_modeling",
        "formula_version": FORMULA_VERSION,
        "snapshot_id": snapshot.metadata.dataset_id,
        "snapshot_fingerprint": snapshot.metadata.fingerprint,
        "start": start.isoformat(), "end": end.isoformat(), "cutoff": cutoff.isoformat(),
        "rows": len(factors), "columns": list(factors.columns), "features": EXPANSION_FEATURES,
        "source_definition": "v7 recomputation of frozen generic ZyQuant/NShape definitions plus Hermes limit/intraday fields",
        "source_orphan_rows": reference_audit,
        "security_repairs": {
            "manifest": str(repair_manifest) if repair_manifest is not None else None,
            "manifest_sha256": hash_file(repair_manifest) if repair_manifest is not None else None,
            "version": repair_metadata.get("version") if repair_metadata is not None else None,
            "audit": repair_audit,
        },
        "duplicate_key_rows": int(factors.duplicated(KEYS).sum()),
        "null_key_rows": int(factors[KEYS].isna().any(axis=1).sum()),
        "row_key_sha256": _row_key_hash(factors),
        "coverage": {name: {"non_null_rows": int(factors[name].notna().sum()), "missing_rows": int(factors[name].isna().sum()), "coverage": float(factors[name].notna().mean())} for name in EXPANSION_FEATURES},
        "yearly_coverage": {str(year): {name: float(block[name].notna().mean()) for name in EXPANSION_FEATURES} for year, block in factors.groupby(factors["trade_date"].dt.year)},
        "range_checks": {"infinite_numeric_values": inf_count},
        "sha256": hash_file(output_path),
    }
    metadata_path = output_path.with_suffix(".meta.json")
    temporary_meta = metadata_path.with_suffix(".json.tmp")
    temporary_meta.write_text(canonical_json(metadata), encoding="utf-8")
    os.replace(temporary_meta, metadata_path)
    return metadata


def verify_expansion_factor_cache(cache_path: Path, *, snapshot_fingerprint: str) -> dict[str, Any]:
    metadata_path = cache_path.with_suffix(".meta.json")
    if not cache_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"expansion cache or metadata is missing: {cache_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "ready_for_modeling":
        raise RuntimeError("expansion cache is not ready_for_modeling")
    if metadata.get("formula_version") != FORMULA_VERSION:
        raise RuntimeError("expansion cache formula version mismatch")
    if metadata.get("snapshot_fingerprint") != snapshot_fingerprint:
        raise RuntimeError("expansion cache parent snapshot mismatch")
    if metadata.get("columns") != REQUIRED_COLUMNS:
        raise RuntimeError("expansion cache column contract mismatch")
    if metadata.get("sha256") != hash_file(cache_path):
        raise RuntimeError("expansion cache content hash mismatch")
    return metadata
