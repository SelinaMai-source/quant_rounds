"""Build the first-board ML panel exclusively from a ZyQuant Hermes snapshot.

The builder intentionally leaves feature NaNs untouched. Imputation, missing
indicators and winsorisation belong inside each training fold, never here.
"""

from __future__ import annotations

import gc
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from zyquant.data import DataSnapshot

from .expansion_factors import (
    EXPANSION_FEATURES,
    verify_expansion_factor_cache,
)
from .native_factors import (
    NATIVE_FEATURES,
    verify_native_factor_cache,
)

KEYS = ["trade_date", "instrument_id"]
BASE_FEATURES = [
    "g0_ret_1d", "g0_turnover", "g0_amount_log", "g0_float_mv_log",
    "g0_upper_shadow", "g0_lower_shadow", "g0_close_open", "g0_amt_vs_5",
    "g1_ret_5", "g1_ret_10", "g1_ret_20", "g1_ma_bias_5",
    "g3_hist_limit_60", "g3_days_since_limit",
    "g4_has_mf", "g7_listing_days_log", "g7_total_mv_log",
]
HERMES_MONEY_FLOW_FEATURES = [
    "h4_main_net_to_turnover",
    "h4_main_net_5_to_turnover",
    "h4_large_net_to_turnover",
]
HERMES_PRIMITIVE_FEATURES = [
    "h_prim_ttl_norm_rebuilt", "h_proxy_px_vol_corr", "h_proxy_path_eff_ohlc",
]
HERMES_FEATURES = [*HERMES_MONEY_FLOW_FEATURES, *HERMES_PRIMITIVE_FEATURES]
ADDITIONAL_HERMES_FEATURES = [
    feature for feature in NATIVE_FEATURES
    if feature not in HERMES_MONEY_FLOW_FEATURES
]
HERMES29_FEATURES = [*HERMES_FEATURES, *ADDITIONAL_HERMES_FEATURES]
HERMES44_FEATURES = [*HERMES29_FEATURES, *EXPANSION_FEATURES]
DROPPED_ORIGINAL_FEATURES = {
    "g3_post_limit_ret_mean": "removed by user: 98% structural missingness",
    "g9_margin_balance_chg_1d": "removed by user: 58% missingness for non-marginable stocks",
}
REPLACED_ORIGINAL_FEATURES = {
    "g4_inst_net_to_amt": "h4_main_net_to_turnover",
    "g4_inst_net_5_to_amt": "h4_main_net_5_to_turnover",
    "g4_big_net_to_amt": "h4_large_net_to_turnover",
}
LABEL_COLUMNS = [
    "next1_date", "next2_date", "next1_open", "next2_close",
    "label_e2_gross", "label_e2_net", "label_reject_reason",
]


def _assert_unique(name: str, frame: pd.DataFrame) -> None:
    bad = int(frame.duplicated(KEYS).sum())
    if bad:
        raise RuntimeError(f"{name} has {bad} duplicate instrument/date keys")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / den


def _stock_rolling(
    series: pd.Series,
    groups: pd.Series,
    window: int,
    min_periods: int,
    op: str,
) -> pd.Series:
    rolled = series.groupby(groups, sort=False).rolling(window, min_periods=min_periods)
    result = getattr(rolled, op)().reset_index(level=0, drop=True)
    return result.reindex(series.index)


def _calendar(snapshot: DataSnapshot, start: date, end: date, cutoff: date) -> pd.DataFrame:
    cal = snapshot.table(
        "trade_calendar", start=start, end=end, cutoff=cutoff,
        fields=["trade_date", "exchange"],
    )
    if cal["exchange"].astype(str).eq("XSHG").any():
        cal = cal.loc[cal["exchange"].astype(str).eq("XSHG")]
    dates = pd.Series(pd.to_datetime(cal["trade_date"]).drop_duplicates().sort_values())
    return pd.DataFrame({"trade_date": dates.to_numpy(), "cal_idx": np.arange(len(dates), dtype=np.int32)})


def _compute_daily_features(raw: pd.DataFrame, adjusted: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    _assert_unique("daily_raw", raw)
    _assert_unique("daily_post_adjusted", adjusted)
    raw = raw.copy()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    adjusted = adjusted[[*KEYS, "close_post"]].copy()
    adjusted["trade_date"] = pd.to_datetime(adjusted["trade_date"])
    raw = raw.merge(adjusted, on=KEYS, how="left", validate="one_to_one")
    raw = raw.merge(calendar, on="trade_date", how="left", validate="many_to_one")
    if raw["cal_idx"].isna().any():
        raise RuntimeError("daily_raw contains dates absent from the XSHG trading calendar")
    raw = raw.sort_values(KEYS).reset_index(drop=True)
    groups = raw["instrument_id"]

    tradable = (~raw["paused"].fillna(True).astype(bool)) & pd.to_numeric(raw["volume"], errors="coerce").gt(0)
    raw["is_close_limit"] = tradable & pd.to_numeric(raw["close"], errors="coerce").ge(
        pd.to_numeric(raw["limit_up"], errors="coerce") - 1e-4
    )
    raw["is_open_limit"] = tradable & pd.to_numeric(raw["open"], errors="coerce").ge(
        pd.to_numeric(raw["limit_up"], errors="coerce") - 1e-4
    )
    raw["is_close_limit_down"] = tradable & pd.to_numeric(raw["close"], errors="coerce").le(
        pd.to_numeric(raw["limit_down"], errors="coerce") + 1e-4
    )
    raw["g0_ret_1d"] = _safe_ratio(raw["close"], raw["pre_close"]) - 1.0
    raw["g0_amount_log"] = np.log1p(pd.to_numeric(raw["amount"], errors="coerce").where(raw["amount"].ge(0)))
    raw["g0_upper_shadow"] = _safe_ratio(
        raw["high"] - raw[["open", "close"]].max(axis=1), raw["pre_close"]
    )
    raw["g0_lower_shadow"] = _safe_ratio(
        raw[["open", "close"]].min(axis=1) - raw["low"], raw["pre_close"]
    )
    raw["g0_close_open"] = _safe_ratio(raw["close"], raw["open"]) - 1.0

    amount_prev = raw["amount"].groupby(groups, sort=False).shift(1)
    amount_prev5 = _stock_rolling(amount_prev, groups, 5, 5, "mean")
    raw["g0_amt_vs_5"] = _safe_ratio(raw["amount"], amount_prev5)
    for window in (5, 10, 20):
        prior = raw["close_post"].groupby(groups, sort=False).shift(window)
        raw[f"g1_ret_{window}"] = _safe_ratio(raw["close_post"], prior) - 1.0
    ma5 = _stock_rolling(raw["close_post"], groups, 5, 5, "mean")
    raw["g1_ma_bias_5"] = _safe_ratio(raw["close_post"], ma5) - 1.0

    prior_limit = raw["is_close_limit"].groupby(groups, sort=False).shift(1).eq(True)
    raw["g3_hist_limit_60"] = _stock_rolling(
        prior_limit.astype(float), groups, 60, 10, "sum"
    )
    row_number = raw.groupby("instrument_id", sort=False).cumcount().astype(float)
    last_limit_row = row_number.where(raw["is_close_limit"]).groupby(
        groups, sort=False
    ).ffill().groupby(groups, sort=False).shift(1)
    raw["g3_days_since_limit"] = (row_number - last_limit_row).fillna(-1.0)
    raw["g7_listing_days_log"] = np.log1p(row_number + 1.0)

    event_cal = raw["cal_idx"].where(raw["is_close_limit"])
    previous_event_cal = event_cal.groupby(groups, sort=False).ffill().groupby(
        groups, sort=False
    ).shift(1)
    no_limit_in_20 = previous_event_cal.isna() | (raw["cal_idx"] - previous_event_cal).gt(20)
    raw["is_first_board_hard"] = raw["is_close_limit"] & no_limit_in_20
    return raw


def _compute_money_flow_features(money: pd.DataFrame) -> pd.DataFrame:
    """Build Hermes-native order-size flow ratios without identity claims."""
    _assert_unique("daily_money_flow", money)
    money = money.copy()
    money["trade_date"] = pd.to_datetime(money["trade_date"])
    money = money.sort_values(KEYS).reset_index(drop=True)
    main_net = pd.to_numeric(money["main_net_inflow"], errors="coerce")
    large_net = pd.to_numeric(money["net_inflow_l"], errors="coerce")
    turnover = pd.to_numeric(money["turnover_value"], errors="coerce")
    money["h4_main_net_to_turnover"] = _safe_ratio(main_net, turnover)
    money["h4_large_net_to_turnover"] = _safe_ratio(large_net, turnover)
    groups = money["instrument_id"]
    main_net_5 = _stock_rolling(main_net, groups, 5, 5, "sum")
    turnover_5_mean = _stock_rolling(turnover, groups, 5, 5, "mean")
    money["h4_main_net_5_to_turnover"] = _safe_ratio(
        main_net_5, turnover_5_mean
    )
    money["g4_has_mf"] = 1.0
    return money[[*KEYS, "g4_has_mf", *HERMES_MONEY_FLOW_FEATURES]]


def _attach_direct_features(
    sample: pd.DataFrame,
    snapshot: DataSnapshot,
    start: date,
    end: date,
    cutoff: date,
    *,
    include_money_flow: bool = True,
) -> pd.DataFrame:
    valuation = snapshot.table(
        "daily_valuation", start=start, end=end, cutoff=cutoff,
        fields=[*KEYS, "turnover_rate", "circulating_market_cap", "market_cap"],
    )
    _assert_unique("daily_valuation", valuation)
    valuation["trade_date"] = pd.to_datetime(valuation["trade_date"])
    valuation = valuation.rename(columns={
        "turnover_rate": "g0_turnover",
        "circulating_market_cap": "_float_mv",
        "market_cap": "_total_mv",
    })
    sample = sample.merge(valuation, on=KEYS, how="left", validate="one_to_one")
    sample["g0_float_mv_log"] = np.log1p(pd.to_numeric(sample.pop("_float_mv"), errors="coerce").where(lambda x: x >= 0))
    sample["g7_total_mv_log"] = np.log1p(pd.to_numeric(sample.pop("_total_mv"), errors="coerce").where(lambda x: x >= 0))

    if include_money_flow:
        money = snapshot.table(
            "daily_money_flow", start=start, end=end, cutoff=cutoff,
            fields=[
                *KEYS, "main_net_inflow", "net_inflow_l", "turnover_value",
            ],
        )
        money = _compute_money_flow_features(money)
        sample = sample.merge(
            money[[*KEYS, "g4_has_mf", *HERMES_MONEY_FLOW_FEATURES]],
            on=KEYS, how="left", validate="one_to_one",
        )
        sample["g4_has_mf"] = sample["g4_has_mf"].fillna(0.0)
    return sample


def _attach_native_factor_cache(
    sample: pd.DataFrame,
    cache_path: Path,
    *,
    sample_start: date,
    sample_end: date,
) -> pd.DataFrame:
    cache = pd.read_parquet(
        cache_path,
        columns=[*KEYS, *NATIVE_FEATURES],
        filters=[
            ("trade_date", ">=", sample_start),
            ("trade_date", "<=", sample_end),
        ],
    )
    cache["trade_date"] = pd.to_datetime(cache["trade_date"])
    _assert_unique("Hermes native factor cache", cache)
    cache["_native_cache_row"] = True
    sample = sample.merge(cache, on=KEYS, how="left", validate="one_to_one")
    sample["g4_has_mf"] = (
        sample[HERMES_MONEY_FLOW_FEATURES].notna().any(axis=1)
        & sample["_native_cache_row"].fillna(False)
    ).astype(float)
    return sample.drop(columns="_native_cache_row")


def _attach_expansion_factor_cache(
    sample: pd.DataFrame,
    cache_path: Path,
    *,
    sample_start: date,
    sample_end: date,
) -> pd.DataFrame:
    cache = pd.read_parquet(
        cache_path,
        columns=[*KEYS, *EXPANSION_FEATURES],
        filters=[
            ("trade_date", ">=", sample_start),
            ("trade_date", "<=", sample_end),
        ],
    )
    cache["trade_date"] = pd.to_datetime(cache["trade_date"])
    _assert_unique("Hermes expansion factor cache", cache)
    return sample.merge(cache, on=KEYS, how="left", validate="one_to_one")


def _attach_labels(sample: pd.DataFrame, raw: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    date_by_idx = pd.Series(calendar["trade_date"].to_numpy(), index=calendar["cal_idx"])
    sample["next1_date"] = (sample["cal_idx"] + 1).map(date_by_idx)
    sample["next2_date"] = (sample["cal_idx"] + 2).map(date_by_idx)
    lookup = raw.set_index(KEYS)[["open", "close", "is_open_limit", "is_close_limit_down", "paused", "volume"]]
    for offset, prefix in ((1, "next1"), (2, "next2")):
        keys = pd.MultiIndex.from_arrays(
            [sample[f"{prefix}_date"], sample["instrument_id"]], names=KEYS
        )
        values = lookup.reindex(keys).reset_index(drop=True)
        sample[f"{prefix}_open"] = values["open"].to_numpy()
        sample[f"{prefix}_close"] = values["close"].to_numpy()
        sample[f"{prefix}_open_limit"] = values["is_open_limit"].to_numpy()
        sample[f"{prefix}_close_limit_down"] = values["is_close_limit_down"].to_numpy()
        sample[f"{prefix}_paused"] = values["paused"].to_numpy()
        sample[f"{prefix}_volume"] = values["volume"].to_numpy()

    buy = pd.to_numeric(sample["next1_open"], errors="coerce")
    sell = pd.to_numeric(sample["next2_close"], errors="coerce")
    missing_horizon = sample["next1_date"].isna() | sample["next2_date"].isna()
    next1_paused = sample["next1_paused"].astype("boolean").fillna(True).astype(bool)
    next2_paused = sample["next2_paused"].astype("boolean").fillna(True).astype(bool)
    buy_limit = sample["next1_open_limit"].astype("boolean").fillna(False).astype(bool)
    sell_limit = sample["next2_close_limit_down"].astype("boolean").fillna(False).astype(bool)
    next1_volume = pd.to_numeric(sample["next1_volume"], errors="coerce").fillna(0)
    next2_volume = pd.to_numeric(sample["next2_volume"], errors="coerce").fillna(0)
    missing_buy = buy.isna() | next1_paused | next1_volume.le(0)
    missing_sell = sell.isna() | next2_paused | next2_volume.le(0)
    gross_ok = ~(missing_horizon | missing_buy | buy_limit | sell.isna())
    net_ok = gross_ok & ~missing_sell & ~sell_limit
    sample["label_e2_gross"] = (sell / buy - 1.0).where(gross_ok)
    buy_net = buy * (1.0 + 0.00123) * (1.0 + 0.0001)
    sell_net = sell * (1.0 - 0.00123) * (1.0 - 0.0001 - 0.0005)
    sample["label_e2_net"] = (sell_net / buy_net - 1.0).where(net_ok)
    reason = np.select(
        [missing_horizon, missing_buy, buy_limit, missing_sell, sell_limit],
        ["insufficient_horizon", "missing_buy_open", "buy_open_limit_up", "missing_next2_close", "sell_close_limit_down"],
        default="ok",
    )
    sample["label_reject_reason"] = reason
    drop = [c for c in sample.columns if c.startswith(("next1_", "next2_"))]
    keep = set(LABEL_COLUMNS)
    return sample.drop(columns=[c for c in drop if c not in keep])


def _drop_missing_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep only rows usable by both published E2 targets; never inspect features."""
    valid = frame[["label_e2_gross", "label_e2_net"]].notna().all(axis=1)
    rejected = frame.loc[~valid, "label_reject_reason"].value_counts(dropna=False)
    audit = {
        "policy": "require_non_null_label_e2_gross_and_label_e2_net",
        "rows_before": len(frame),
        "rows_dropped": int((~valid).sum()),
        "rows_after": int(valid.sum()),
        "dropped_reasons": {str(key): int(value) for key, value in rejected.items()},
    }
    return frame.loc[valid].copy(), audit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit(frame: pd.DataFrame, variants: dict[str, list[str]]) -> dict[str, Any]:
    # Audit the complete active contract.  Deriving this from the emitted
    # variants keeps hermes23 backward compatible while ensuring full29's six
    # additional native factors are covered by the same NaN/inf checks.
    features = list(dict.fromkeys(feature for names in variants.values() for feature in names))
    numeric = frame[features + ["label_e2_gross", "label_e2_net"]]
    inf_count = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
    if inf_count:
        raise RuntimeError(f"dataset still contains {inf_count} infinite numeric values")
    missing = {
        name: {
            "missing_rows": int(frame[name].isna().sum()),
            "missing_rate": float(frame[name].isna().mean()),
            "non_null_rows": int(frame[name].notna().sum()),
        }
        for name in features
    }
    all_null = [name for name, item in missing.items() if item["non_null_rows"] == 0]
    if all_null:
        raise RuntimeError(f"all-null model features: {all_null}")
    gross_count = int(frame["label_e2_gross"].notna().sum())
    net_count = int(frame["label_e2_net"].notna().sum())
    if gross_count / len(frame) < 0.5 or net_count / len(frame) < 0.5:
        raise RuntimeError(
            f"label alignment failed: gross={gross_count}/{len(frame)}, "
            f"net={net_count}/{len(frame)}"
        )
    yearly: dict[str, Any] = {}
    years = pd.to_datetime(frame["trade_date"]).dt.year
    for year in sorted(years.unique()):
        part = frame.loc[years.eq(year)]
        yearly[str(year)] = {
            "rows": len(part),
            "gross_label_coverage": float(part["label_e2_gross"].notna().mean()),
            "net_label_coverage": float(part["label_e2_net"].notna().mean()),
            "feature_missing_rate": {
                name: float(part[name].isna().mean()) for name in features
            },
        }
    return {
        "rows": len(frame),
        "date_min": frame["trade_date"].min().date().isoformat(),
        "date_max": frame["trade_date"].max().date().isoformat(),
        "instruments": int(frame["instrument_id"].nunique()),
        "duplicate_key_rows": int(frame.duplicated(KEYS).sum()),
        "null_key_rows": int(frame[KEYS].isna().any(axis=1).sum()),
        "infinite_numeric_values": inf_count,
        "feature_missing": missing,
        "label_non_null": {
            "label_e2_gross": gross_count,
            "label_e2_net": net_count,
        },
        "label_reject_reasons": {str(k): int(v) for k, v in frame["label_reject_reason"].value_counts(dropna=False).items()},
        "variant_row_key_sha256": {
            name: hashlib.sha256(
                pd.util.hash_pandas_object(frame[KEYS], index=False).values.tobytes()
            ).hexdigest()
            for name in variants
        },
        "yearly": yearly,
    }


def build_dataset(
    snapshot_path: Path,
    primitive_cache: Path,
    output_dir: Path,
    *,
    native_factor_cache: Path | None = None,
    expansion_factor_cache: Path | None = None,
    feature_contract: str = "hermes23",
    history_start: date = date(2010, 1, 1),
    sample_start: date = date(2018, 1, 1),
    sample_end: date = date(2026, 7, 1),
    cutoff: date = date(2026, 7, 24),
) -> dict[str, Any]:
    snapshot = DataSnapshot(snapshot_path, verify_hashes=True)
    if feature_contract not in {"hermes23", "hermes29", "hermes44"}:
        raise ValueError(f"unknown feature contract: {feature_contract}")
    native_cache_metadata = None
    if feature_contract in {"hermes29", "hermes44"}:
        if native_factor_cache is None:
            raise ValueError("hermes29 requires native_factor_cache")
        native_cache_metadata = verify_native_factor_cache(
            native_factor_cache,
            snapshot_fingerprint=snapshot.metadata.fingerprint,
        )
    expansion_cache_metadata = None
    if feature_contract == "hermes44":
        if expansion_factor_cache is None:
            raise ValueError("hermes44 requires expansion_factor_cache")
        expansion_cache_metadata = verify_expansion_factor_cache(
            expansion_factor_cache,
            snapshot_fingerprint=snapshot.metadata.fingerprint,
        )
    calendar = _calendar(snapshot, history_start, cutoff, cutoff)
    raw = snapshot.table(
        "daily_raw", start=history_start, end=cutoff, cutoff=cutoff,
        fields=[*KEYS, "open", "high", "low", "close", "pre_close", "volume", "amount", "paused", "limit_up", "limit_down"],
    )
    adjusted = snapshot.table(
        "daily_post_adjusted", start=history_start, end=cutoff, cutoff=cutoff,
        fields=[*KEYS, "close_post"],
    )
    daily = _compute_daily_features(raw, adjusted, calendar)
    sample = daily.loc[
        daily["is_first_board_hard"]
        & daily["trade_date"].between(pd.Timestamp(sample_start), pd.Timestamp(sample_end))
    ].copy()
    if sample.empty:
        raise RuntimeError("Hermes hard first-board sample is empty")
    sample = _attach_direct_features(
        sample, snapshot, history_start, sample_end, cutoff,
        include_money_flow=feature_contract == "hermes23",
    )
    if feature_contract in {"hermes29", "hermes44"}:
        sample = _attach_native_factor_cache(
            sample,
            native_factor_cache,
            sample_start=sample_start,
            sample_end=sample_end,
        )
    if feature_contract == "hermes44":
        sample = _attach_expansion_factor_cache(
            sample,
            expansion_factor_cache,
            sample_start=sample_start,
            sample_end=sample_end,
        )
    sample = _attach_labels(sample, daily, calendar)
    del adjusted, raw
    gc.collect()

    primitives = pd.read_parquet(
        primitive_cache, columns=[*KEYS, *HERMES_PRIMITIVE_FEATURES]
    )
    primitives["trade_date"] = pd.to_datetime(primitives["trade_date"])
    _assert_unique("Hermes primitive cache", primitives)
    sample = sample.merge(primitives, on=KEYS, how="left", validate="one_to_one")
    active_hermes_features = {
        "hermes23": HERMES_FEATURES,
        "hermes29": HERMES29_FEATURES,
        "hermes44": HERMES44_FEATURES,
    }[feature_contract]
    keep = [
        *KEYS, "cal_idx", "is_first_board_hard", *BASE_FEATURES,
        *active_hermes_features, *LABEL_COLUMNS,
    ]
    candidate_frame = sample[keep].sort_values(KEYS).reset_index(drop=True)
    numeric_cols = [
        *BASE_FEATURES, *active_hermes_features, "label_e2_gross", "label_e2_net"
    ]
    candidate_frame[numeric_cols] = candidate_frame[numeric_cols].replace([np.inf, -np.inf], np.nan)
    if candidate_frame.duplicated(KEYS).any() or candidate_frame[KEYS].isna().any(axis=None):
        raise RuntimeError("candidate dataset has duplicate or null keys")
    frame, label_filter_audit = _drop_missing_labels(candidate_frame)

    variants = {
        "baseline": [*BASE_FEATURES, *HERMES_MONEY_FLOW_FEATURES],
        "ttl": [*BASE_FEATURES, *HERMES_MONEY_FLOW_FEATURES, HERMES_PRIMITIVE_FEATURES[0]],
        "ttl_pxvol": [*BASE_FEATURES, *HERMES_MONEY_FLOW_FEATURES, *HERMES_PRIMITIVE_FEATURES[:2]],
        "ttl_pxvol_path": [*BASE_FEATURES, *HERMES_FEATURES],
    }
    if feature_contract in {"hermes29", "hermes44"}:
        variants["full29"] = [*BASE_FEATURES, *HERMES29_FEATURES]
    if feature_contract == "hermes44":
        variants["full44"] = [*BASE_FEATURES, *HERMES44_FEATURES]
    audit = _audit(frame, variants)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    common = [*KEYS, "cal_idx", "is_first_board_hard", *LABEL_COLUMNS]
    candidate_path = output_dir / "candidate_panel.parquet"
    candidate_frame.to_parquet(candidate_path, index=False)
    outputs["candidate_panel"] = {
        "file": candidate_path.name,
        "purpose": "point-in-time inference universe; labels may be null and must never filter predictions",
        "rows": len(candidate_frame),
        "sha256": _sha256(candidate_path),
    }
    for name, features in variants.items():
        path = output_dir / f"{name}.parquet"
        frame[[*common, *features]].to_parquet(path, index=False)
        outputs[name] = {
            "file": path.name,
            "features": features,
            "rows": len(frame),
            "sha256": _sha256(path),
        }
    full_path = output_dir / "full_panel.parquet"
    frame.to_parquet(full_path, index=False)
    outputs["full_panel"] = {
        "file": full_path.name,
        "rows": len(frame),
        "sha256": _sha256(full_path),
    }
    manifest = {
        "status": "ready_for_modeling",
        "source_policy": "pure_hermes_only",
        "snapshot_id": snapshot.metadata.dataset_id,
        "snapshot_fingerprint": snapshot.metadata.fingerprint,
        "history_start": history_start.isoformat(),
        "sample_start": sample_start.isoformat(),
        "sample_end": sample_end.isoformat(),
        "cutoff": cutoff.isoformat(),
        "feature_contract": feature_contract,
        "base_features": BASE_FEATURES,
        "hermes_features": active_hermes_features,
        "native_factor_cache": (
            {
                "path": str(native_factor_cache),
                "sha256": native_cache_metadata["sha256"],
                "formula_version": native_cache_metadata["formula_version"],
                "parent_snapshot_fingerprint": native_cache_metadata[
                    "snapshot_fingerprint"
                ],
            }
            if native_cache_metadata is not None
            else None
        ),
        "expansion_factor_cache": (
            {
                "path": str(expansion_factor_cache),
                "sha256": expansion_cache_metadata["sha256"],
                "formula_version": expansion_cache_metadata["formula_version"],
                "parent_snapshot_fingerprint": expansion_cache_metadata[
                    "snapshot_fingerprint"
                ],
            }
            if expansion_cache_metadata is not None
            else None
        ),
        "dropped_original_features": DROPPED_ORIGINAL_FEATURES,
        "replaced_original_features": REPLACED_ORIGINAL_FEATURES,
        "hermes_money_flow_contract": {
            "h4_main_net_to_turnover": "main_net_inflow / turnover_value",
            "h4_main_net_5_to_turnover": "rolling_sum(main_net_inflow, 5) / rolling_mean(turnover_value, 5); five observations required",
            "h4_large_net_to_turnover": "net_inflow_l / turnover_value",
            "semantic_note": "Hermes order-size buckets; never labeled as institution or investor identity",
        },
        "missing_policy": "raw NaN retained; inf converted to NaN; no feature-based row drop in the data layer",
        "training_preprocess_contract": "for each fold, drop rows with any active-feature NaN in memory only; never rewrite parquet",
        "candidate_contract": "score candidate_panel using active features only; never use label presence, label value or reject reason to define the prediction universe",
        "labels": {
            "label_e2_gross": "D+1 open to D+2 close; D+1 open limit-up/unavailable rejected; no costs",
            "label_e2_net": "fixed D+1 open to D+2 close; buy/sell tradability enforced; commission 0.0001 each side, sell tax 0.0005, slippage 0.00123 each side",
        },
        "label_filter": label_filter_audit,
        "audit": audit,
        "outputs": outputs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
