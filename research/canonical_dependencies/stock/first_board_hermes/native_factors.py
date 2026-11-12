"""Auditable cache for Hermes-native first-board factors."""

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

KEYS = ["trade_date", "instrument_id"]
FORMULA_VERSION = "first_board_hermes_native_v1"
MONEY_FLOW_FEATURES = [
    "h4_main_net_to_turnover",
    "h4_main_net_5_to_turnover",
    "h4_large_net_to_turnover",
    "h4_xl_net_concentration",
    "h4_retail_net_to_turnover",
]
LIMIT_FEATURES = ["h5_close_limit_value_to_amount"]
INTRADAY_FEATURES = [
    "h6_volume_weighted_entropy",
    "h6_average_trade_net_inflow_ratio",
    "h6_weighted_close_skew",
]
NATIVE_FEATURES = [*MONEY_FLOW_FEATURES, *LIMIT_FEATURES, *INTRADAY_FEATURES]
REQUIRED_COLUMNS = [*KEYS, *NATIVE_FEATURES]


def _assert_unique(name: str, frame: pd.DataFrame) -> None:
    duplicate_rows = int(frame.duplicated(KEYS).sum())
    if duplicate_rows:
        raise RuntimeError(f"{name} contains {duplicate_rows} duplicate keys")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return numerator / denominator


def _rolling_five_sum_to_mean(
    numerator: pd.Series,
    denominator: pd.Series,
    groups: pd.Series,
) -> pd.Series:
    numerator_sum = (
        numerator.groupby(groups, sort=False)
        .rolling(5, min_periods=5)
        .sum()
        .reset_index(level=0, drop=True)
        .reindex(numerator.index)
    )
    denominator_mean = (
        denominator.groupby(groups, sort=False)
        .rolling(5, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(denominator.index)
    )
    return _safe_ratio(numerator_sum, denominator_mean)


def compute_native_factors(
    raw: pd.DataFrame,
    money_flow: pd.DataFrame,
    limit_events: pd.DataFrame,
    intraday: pd.DataFrame,
) -> pd.DataFrame:
    """Compute nine factors on the complete raw stock-day key space."""
    raw = raw[[*KEYS, "amount"]].copy()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    raw = raw.sort_values(KEYS).reset_index(drop=True)
    _assert_unique("daily_raw", raw)

    money_flow = money_flow.copy()
    money_flow["trade_date"] = pd.to_datetime(money_flow["trade_date"])
    money_flow = money_flow.sort_values(KEYS).reset_index(drop=True)
    _assert_unique("daily_money_flow", money_flow)
    main_net = pd.to_numeric(money_flow["main_net_inflow"], errors="coerce")
    large_net = pd.to_numeric(money_flow["net_inflow_l"], errors="coerce")
    extra_large_net = pd.to_numeric(money_flow["net_inflow_xl"], errors="coerce")
    retail_net = pd.to_numeric(money_flow["retail_net_inflow"], errors="coerce")
    turnover = pd.to_numeric(money_flow["turnover_value"], errors="coerce")
    money_flow["h4_main_net_to_turnover"] = _safe_ratio(main_net, turnover)
    money_flow["h4_main_net_5_to_turnover"] = _rolling_five_sum_to_mean(
        main_net, turnover, money_flow["instrument_id"]
    )
    money_flow["h4_large_net_to_turnover"] = _safe_ratio(large_net, turnover)
    bucket_abs_sum = sum(
        pd.to_numeric(money_flow[column], errors="coerce").abs()
        for column in (
            "net_inflow_s", "net_inflow_m", "net_inflow_l", "net_inflow_xl"
        )
    )
    money_flow["h4_xl_net_concentration"] = _safe_ratio(
        extra_large_net, bucket_abs_sum
    )
    money_flow["h4_retail_net_to_turnover"] = _safe_ratio(retail_net, turnover)
    money_values = money_flow[[*KEYS, *MONEY_FLOW_FEATURES]]

    limit_events = limit_events.loc[
        limit_events["limit_type"].astype(str).str.lower().eq("up")
    ].copy()
    limit_events["trade_date"] = pd.to_datetime(limit_events["trade_date"])
    _assert_unique("daily_limit_events[limit_type=up]", limit_events)
    limit_values = limit_events[[*KEYS, "close_limit_value"]]

    intraday = intraday.copy()
    intraday["trade_date"] = pd.to_datetime(intraday["trade_date"])
    _assert_unique("daily_intraday_factors", intraday)
    intraday_values = intraday[[
        *KEYS,
        "volume_weighted_entropy",
        "average_trade_net_inflow_ratio",
        "weighted_close_skew",
    ]].rename(columns={
        "volume_weighted_entropy": "h6_volume_weighted_entropy",
        "average_trade_net_inflow_ratio": "h6_average_trade_net_inflow_ratio",
        "weighted_close_skew": "h6_weighted_close_skew",
    })

    output = raw.merge(money_values, on=KEYS, how="left", validate="one_to_one")
    output = output.merge(limit_values, on=KEYS, how="left", validate="one_to_one")
    output["h5_close_limit_value_to_amount"] = _safe_ratio(
        output.pop("close_limit_value"), output["amount"]
    )
    output = output.drop(columns="amount").merge(
        intraday_values, on=KEYS, how="left", validate="one_to_one"
    )
    output[NATIVE_FEATURES] = output[NATIVE_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    _assert_unique("Hermes native factor cache", output)
    return output[[*KEYS, *NATIVE_FEATURES]].sort_values(KEYS).reset_index(drop=True)


def _source_reference_audit(
    raw: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
) -> dict[str, int]:
    raw_keys = pd.MultiIndex.from_frame(raw[KEYS])
    result = {}
    for name, source in sources.items():
        source_keys = pd.MultiIndex.from_frame(source[KEYS])
        result[name] = int((~source_keys.isin(raw_keys)).sum())
    return result


def _row_key_hash(frame: pd.DataFrame) -> str:
    import hashlib

    values = pd.util.hash_pandas_object(frame[KEYS], index=False).values.tobytes()
    return hashlib.sha256(values).hexdigest()


def apply_security_repairs(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    source_table: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply exact, source-record-locked symbol-transition repairs."""
    frame = frame.copy()
    audit = []
    for repair in manifest.get("repairs", []):
        if repair.get("source_table") != source_table:
            continue
        expected = set(repair["expected_source_record_ids"])
        mask = (
            frame["instrument_id"].astype(str).eq(repair["old_instrument_id"])
            & frame["trade_date"].between(
                pd.Timestamp(repair["effective_from"]),
                pd.Timestamp(repair["effective_to"]),
            )
        )
        actual = set(frame.loc[mask, "source_record_id"].astype(str))
        if actual != expected:
            raise RuntimeError(
                "security repair source records mismatch for "
                f"{repair['old_instrument_id']}: expected={sorted(expected)}, "
                f"actual={sorted(actual)}"
            )
        frame.loc[mask, "instrument_id"] = repair["new_instrument_id"]
        audit.append({
            "source_table": source_table,
            "old_instrument_id": repair["old_instrument_id"],
            "new_instrument_id": repair["new_instrument_id"],
            "effective_from": repair["effective_from"],
            "effective_to": repair["effective_to"],
            "repaired_rows": int(mask.sum()),
            "source_record_ids": sorted(actual),
            "evidence": repair["evidence"],
        })
    _assert_unique(f"{source_table} after security repairs", frame)
    return frame, audit


def build_native_factor_cache(
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
        fields=[*KEYS, "amount"],
    )
    money = snapshot.table(
        "daily_money_flow", start=start, end=end, cutoff=cutoff,
        fields=[
            *KEYS, "main_net_inflow", "retail_net_inflow", "turnover_value",
            "net_inflow_s", "net_inflow_m", "net_inflow_l", "net_inflow_xl",
        ],
    )
    limits = snapshot.table(
        "daily_limit_events", start=start, end=end, cutoff=cutoff,
        fields=[*KEYS, "limit_type", "close_limit_value"],
    )
    intraday = snapshot.table(
        "daily_intraday_factors", start=start, end=end, cutoff=cutoff,
        fields=[
            *KEYS, "volume_weighted_entropy",
            "average_trade_net_inflow_ratio", "weighted_close_skew",
            "source_record_id",
        ],
    )
    for name, source in {
        "daily_raw": raw,
        "daily_money_flow": money,
        "daily_limit_events": limits,
        "daily_intraday_factors": intraday,
    }.items():
        source["trade_date"] = pd.to_datetime(source["trade_date"])
        if name != "daily_limit_events":
            _assert_unique(name, source)
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
        {"daily_money_flow": money, "daily_limit_events": limits, "daily_intraday_factors": intraday},
    )
    if any(reference_audit.values()):
        raise RuntimeError(f"native factor sources contain orphan keys: {reference_audit}")
    factors = compute_native_factors(raw, money, limits, intraday)
    concentration_outside_range = int(
        (
            factors["h4_xl_net_concentration"].notna()
            & ~factors["h4_xl_net_concentration"].between(-1.0, 1.0)
        ).sum()
    )
    if concentration_outside_range:
        raise RuntimeError(
            "h4_xl_net_concentration contains values outside [-1, 1]: "
            f"{concentration_outside_range}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    factors.to_parquet(temporary, index=False)
    os.replace(temporary, output_path)
    years = factors["trade_date"].dt.year
    yearly_coverage = {
        str(year): {
            feature: float(block[feature].notna().mean())
            for feature in NATIVE_FEATURES
        }
        for year, block in factors.groupby(years)
    }
    metadata = {
        "status": "ready_for_modeling",
        "formula_version": FORMULA_VERSION,
        "snapshot_id": snapshot.metadata.dataset_id,
        "snapshot_fingerprint": snapshot.metadata.fingerprint,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "cutoff": cutoff.isoformat(),
        "rows": len(factors),
        "columns": list(factors.columns),
        "features": NATIVE_FEATURES,
        "source_tables": {
            "daily_raw": [*KEYS, "amount"],
            "daily_money_flow": [
                *KEYS, "main_net_inflow", "retail_net_inflow", "turnover_value",
                "net_inflow_s", "net_inflow_m", "net_inflow_l", "net_inflow_xl",
            ],
            "daily_limit_events": [*KEYS, "limit_type", "close_limit_value"],
            "daily_intraday_factors": [
                *KEYS, "volume_weighted_entropy",
                "average_trade_net_inflow_ratio", "weighted_close_skew",
                "source_record_id",
            ],
        },
        "source_orphan_rows": reference_audit,
        "security_repairs": {
            "manifest": str(repair_manifest) if repair_manifest is not None else None,
            "manifest_sha256": (
                hash_file(repair_manifest) if repair_manifest is not None else None
            ),
            "version": (
                repair_metadata.get("version") if repair_metadata is not None else None
            ),
            "audit": repair_audit,
        },
        "duplicate_key_rows": int(factors.duplicated(KEYS).sum()),
        "null_key_rows": int(factors[KEYS].isna().any(axis=1).sum()),
        "row_key_sha256": _row_key_hash(factors),
        "coverage": {
            feature: {
                "non_null_rows": int(factors[feature].notna().sum()),
                "missing_rows": int(factors[feature].isna().sum()),
                "coverage": float(factors[feature].notna().mean()),
            }
            for feature in NATIVE_FEATURES
        },
        "yearly_coverage": yearly_coverage,
        "range_checks": {
            "infinite_numeric_values": int(
                np.isinf(factors[NATIVE_FEATURES].to_numpy(dtype=float)).sum()
            ),
            "h4_xl_net_concentration_outside_minus1_plus1": concentration_outside_range,
        },
        "sha256": hash_file(output_path),
    }
    metadata_path = output_path.with_suffix(".meta.json")
    metadata_temporary = metadata_path.with_suffix(".json.tmp")
    metadata_temporary.write_text(canonical_json(metadata), encoding="utf-8")
    os.replace(metadata_temporary, metadata_path)
    return metadata


def verify_native_factor_cache(
    cache_path: Path,
    *,
    snapshot_fingerprint: str,
) -> dict[str, Any]:
    metadata_path = cache_path.with_suffix(".meta.json")
    if not cache_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"native factor cache or metadata is missing: {cache_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "ready_for_modeling":
        raise RuntimeError("native factor cache is not ready_for_modeling")
    if metadata.get("formula_version") != FORMULA_VERSION:
        raise RuntimeError("native factor cache formula version mismatch")
    if metadata.get("snapshot_fingerprint") != snapshot_fingerprint:
        raise RuntimeError("native factor cache parent snapshot fingerprint mismatch")
    if metadata.get("columns") != REQUIRED_COLUMNS:
        raise RuntimeError("native factor cache column contract mismatch")
    actual_hash = hash_file(cache_path)
    if metadata.get("sha256") != actual_hash:
        raise RuntimeError("native factor cache content hash mismatch")
    return metadata
