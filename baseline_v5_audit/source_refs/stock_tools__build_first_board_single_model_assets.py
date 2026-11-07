#!/usr/bin/env python3
"""Publish the Hermes supplement, factor cache, and single-model panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from first_board_hermes.single_model import (  # noqa: E402
    INDUSTRY_FEATURES,
    KEYS,
    MICROSTRUCTURE_FEATURES,
    MINUTE_EXTENSION_FEATURES,
    SINGLE_MODEL_HISTORICAL29_FEATURES,
    SINGLE_MODEL_HISTORICAL34_FEATURES,
    SINGLE_MODEL_HISTORICAL38_FEATURES,
    SINGLE_MODEL_FEATURES,
    attach_d1_touch_label,
    compute_minute_extension_factors,
    symmetric_change,
)
from zyquant.data import DataSnapshot  # noqa: E402


MICRO_RENAME = {
    "OCVP": "h1_auction_volume_ratio",
    "PCV": "h1_consistent_buy_ratio",
    "NCV": "h1_consistent_sell_ratio",
    "BCVP_15MIN": "h1_last15m_volume_ratio",
    "SMART_MONEY": "h4_smart_money_price",
    "VOI": "h4_volume_order_imbalance",
    "INF_BUY_RATIO_CLOSE": "h4_informed_buy_close_ratio",
    "UP_LIMIT_REACHED_TIMES": "fb_touch_count",
}
MINUTE_PRIMITIVE_FEATURES = [
    "g_prim_ttl_norm",
    "g_prim_px_vol_corr",
    "g_prim_prelimit_accel",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def read_selection(path: Path) -> pd.DataFrame:
    parts = sorted((path / "selection").rglob("*.parquet"))
    if not parts:
        raise RuntimeError(f"minute selection missing under {path}")
    selection = pd.concat([pd.read_parquet(part) for part in parts], ignore_index=True)
    selection["trade_date"] = pd.to_datetime(selection["trade_date"])
    selection["instrument_id"] = selection["instrument_id"].astype(str)
    if selection.duplicated(KEYS).any():
        raise RuntimeError("minute selection has duplicate keys")
    return selection


def publish_microstructure_snapshot(
    extracted: Path,
    selection: pd.DataFrame,
    v7_manifest: dict[str, Any],
    minute_manifest: dict[str, Any],
    output: Path,
    *,
    formal_microstructure_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if output.exists():
        raise RuntimeError(f"immutable microstructure snapshot exists: {output}")
    stage = output.parent / f".{output.name}.staging-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    raw = pd.read_csv(
        extracted / "stock_microstructure.tsv",
        sep="\t",
        dtype={"ticker_symbol": str},
        na_values=[r"\N"],
    )
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    raw["ticker_symbol"] = raw["ticker_symbol"].str.zfill(6)
    if raw.duplicated(["trade_date", "ticker_symbol"]).any():
        raise RuntimeError("extracted microstructure rows contain duplicate keys")
    targets = selection[[*KEYS]].copy()
    targets["ticker_symbol"] = targets["instrument_id"].str.split(".").str[0]
    if targets.duplicated(["trade_date", "ticker_symbol"]).any():
        raise RuntimeError("selection has ambiguous ticker/date mappings")
    stock = targets.merge(
        raw,
        on=["trade_date", "ticker_symbol"],
        how="left",
        validate="one_to_one",
    ).drop(columns="ticker_symbol")
    stock = stock.rename(columns=MICRO_RENAME)
    for name in MICROSTRUCTURE_FEATURES:
        stock[name] = pd.to_numeric(stock[name], errors="coerce")
    coverage = {name: float(stock[name].notna().mean()) for name in MICROSTRUCTURE_FEATURES}
    formal = (
        list(MICROSTRUCTURE_FEATURES)
        if formal_microstructure_features is None
        else list(formal_microstructure_features)
    )
    failed = {name: coverage[name] for name in formal if coverage[name] < 0.98}
    if failed:
        raise RuntimeError(f"microstructure coverage below 98%: {failed}")
    if stock.duplicated(KEYS).any() or len(stock) != len(selection):
        raise RuntimeError("normalized microstructure keys do not match minute selection")

    market = pd.read_csv(
        extracted / "market_limit_stats.tsv", sep="\t", na_values=[r"\N"]
    )
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    if market.duplicated(["trade_date"]).any():
        raise RuntimeError("market limit statistics contain duplicate dates")
    stock_dir = stage / "daily_microstructure"
    market_dir = stage / "daily_market_limit_stats"
    stock_dir.mkdir()
    market_dir.mkdir()
    stock_path = stock_dir / "part-000.parquet"
    market_path = market_dir / "part-000.parquet"
    atomic_parquet(stock, stock_path)
    atomic_parquet(market, market_path)
    manifest = {
        "manifest_schema": "first_board.hermes_microstructure/1",
        "dataset_id": output.name,
        "start_date": str(selection["trade_date"].min().date()),
        "end_date": str(selection["trade_date"].max().date()),
        "parent_v7_id": v7_manifest["dataset_id"],
        "parent_v7_fingerprint": v7_manifest["fingerprint"],
        "parent_minute_snapshot_id": minute_manifest["dataset_id"],
        "parent_minute_snapshot_fingerprint": minute_manifest["fingerprint"],
        "source_tables": [
            "equ_h2l_factor_t1", "equ_h2l_factor_t4", "mkt_limit", "mkt_limit_stats"
        ],
        "rows": {"daily_microstructure": len(stock), "daily_market_limit_stats": len(market)},
        "coverage": coverage,
        "formal_microstructure_features": formal,
        "duplicate_key_rows": 0,
        "unknown_instrument_rows_dropped": 0,
        "files": {
            "daily_microstructure/part-000.parquet": sha256(stock_path),
            "daily_market_limit_stats/part-000.parquet": sha256(market_path),
        },
    }
    manifest["fingerprint"] = payload_hash(manifest)
    (stage / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(stage, output)
    return stock, market, manifest


def load_microstructure_snapshot(
    path: Path,
    v7_manifest: dict[str, Any],
    minute_manifest: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify and reuse the immutable raw supplement when only formulas change."""
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"microstructure manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "parent_v7_id": v7_manifest["dataset_id"],
        "parent_v7_fingerprint": v7_manifest["fingerprint"],
        "parent_minute_snapshot_id": minute_manifest["dataset_id"],
        "parent_minute_snapshot_fingerprint": minute_manifest["fingerprint"],
    }
    mismatched = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatched:
        raise RuntimeError(f"microstructure parent mismatch: {mismatched}")
    relative = "daily_microstructure/part-000.parquet"
    stock_path = path / relative
    if sha256(stock_path) != manifest["files"][relative]:
        raise RuntimeError("microstructure content hash mismatch")
    stock = pd.read_parquet(stock_path)
    if stock.duplicated(KEYS).any():
        raise RuntimeError("published microstructure contains duplicate keys")
    return stock, manifest


def map_pit_industry(frame: pd.DataFrame, membership: pd.DataFrame) -> pd.Series:
    """Map instrument/date rows to the SW L1 membership active on that date."""
    # For 2022+, the V2021 taxonomy is the canonical SW first-level industry.
    members = membership.loc[
        membership["classification"].astype(str).eq("SW_L1:V2021")
    ].copy()
    members["effective_from"] = pd.to_datetime(members["effective_from"])
    members["effective_to"] = pd.to_datetime(members["effective_to"])
    by_instrument = {
        str(instrument): block.sort_values("effective_from")
        for instrument, block in members.groupby("instrument_id", sort=False)
    }
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    for instrument, positions in frame.groupby("instrument_id", sort=False).groups.items():
        history = by_instrument.get(str(instrument))
        if history is None:
            continue
        starts = history["effective_from"].to_numpy(dtype="datetime64[ns]")
        ends = history["effective_to"].to_numpy(dtype="datetime64[ns]")
        values = history["industry_id"].astype(str).to_numpy()
        dates = pd.to_datetime(frame.loc[positions, "trade_date"]).to_numpy(dtype="datetime64[ns]")
        indices = np.searchsorted(starts, dates, side="right") - 1
        valid = indices >= 0
        safe = np.maximum(indices, 0)
        valid &= np.isnat(ends[safe]) | (dates < ends[safe])
        mapped = np.where(valid, values[safe], None)
        result.loc[positions] = mapped
    return result


def compute_industry_features(
    selection: pd.DataFrame,
    limit_events: pd.DataFrame,
    daily_raw: pd.DataFrame,
    membership: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    calendar_dates = pd.Series(pd.to_datetime(calendar["trade_date"]).drop_duplicates().sort_values())
    cal_pos = pd.Series(np.arange(len(calendar_dates)), index=calendar_dates).to_dict()
    close_limits = selection[[*KEYS]].copy()
    close_limits["cal_idx"] = close_limits["trade_date"].map(cal_pos)
    close_limits = close_limits.sort_values(["instrument_id", "trade_date"])
    previous = close_limits.groupby("instrument_id")["cal_idx"].shift(1)
    close_limits["is_first_board"] = previous.isna() | (close_limits["cal_idx"] - previous > 20)
    close_limits["industry_id"] = map_pit_industry(close_limits, membership)

    events = limit_events.loc[
        limit_events["limit_type"].astype(str).str.lower().eq("up"),
        [*KEYS, "close_limit_value"],
    ].copy()
    events["industry_id"] = map_pit_industry(events, membership)
    closed_keys = pd.MultiIndex.from_frame(close_limits[KEYS])
    events["is_close_limit"] = pd.MultiIndex.from_frame(events[KEYS]).isin(closed_keys)
    market_seal = events.groupby("trade_date")["is_close_limit"].mean()
    industry_seal = events.groupby(["trade_date", "industry_id"])["is_close_limit"].mean()

    first = close_limits.loc[close_limits["is_first_board"]].copy()
    counts = first.groupby(["trade_date", "industry_id"])["instrument_id"].transform("size")
    totals = first.groupby("trade_date")["instrument_id"].transform("size")
    first["ind_first_board_count_log"] = np.log1p(counts)
    first["ind_first_board_share"] = counts / totals.replace(0, np.nan)
    keys = pd.MultiIndex.from_frame(first[["trade_date", "industry_id"]])
    first["ind_seal_rate_excess"] = (
        industry_seal.reindex(keys).to_numpy()
        - first["trade_date"].map(market_seal).to_numpy()
    )

    amounts = daily_raw[[*KEYS, "amount"]]
    quality = events[[*KEYS, "close_limit_value"]].merge(
        amounts, on=KEYS, how="left", validate="one_to_one"
    )
    quality["seal_quality"] = pd.to_numeric(
        quality["close_limit_value"], errors="coerce"
    ) / pd.to_numeric(quality["amount"], errors="coerce").replace(0, np.nan)
    first = first.merge(quality[[*KEYS, "seal_quality"]], on=KEYS, how="left", validate="one_to_one")
    first["ind_stock_seal_quality_rank"] = first.groupby(
        ["trade_date", "industry_id"], dropna=False
    )["seal_quality"].rank(method="average", pct=True)
    return first[[*KEYS, *INDUSTRY_FEATURES]]


def compute_market_context(
    selection: pd.DataFrame,
    daily_raw: pd.DataFrame,
    limit_events: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Build diagnostic/market-gate series; these columns never enter the stock model."""
    raw = daily_raw.copy()
    tradable = (
        ~raw["paused"].fillna(True).astype(bool)
        & pd.to_numeric(raw["volume"], errors="coerce").gt(0)
    )
    raw["is_up"] = tradable & pd.to_numeric(raw["close"], errors="coerce").gt(
        pd.to_numeric(raw["pre_close"], errors="coerce")
    )
    up_ratio = raw.loc[tradable].groupby("trade_date")["is_up"].mean()
    closed = selection.groupby("trade_date")["instrument_id"].size()
    touched = (
        limit_events.loc[limit_events["limit_type"].astype(str).str.lower().eq("up")]
        .groupby("trade_date")["instrument_id"].size()
    )
    dates = pd.Series(pd.to_datetime(calendar["trade_date"]).drop_duplicates().sort_values())
    dates = dates.loc[
        dates.between(selection["trade_date"].min(), selection["trade_date"].max())
    ].reset_index(drop=True)
    previous_to_current = dict(zip(dates.iloc[:-1], dates.iloc[1:]))
    prior = selection[[*KEYS]].copy()
    prior["trade_date"] = prior["trade_date"].map(previous_to_current)
    prior = prior.dropna(subset=["trade_date"])
    current_raw = raw[[*KEYS, "is_up"]].rename(columns={"is_up": "prev_limit_is_red"})
    prior = prior.merge(current_raw, on=KEYS, how="left", validate="one_to_one")
    current_closed_keys = pd.MultiIndex.from_frame(selection[KEYS])
    prior["prev_limit_is_relimit"] = pd.MultiIndex.from_frame(prior[KEYS]).isin(
        current_closed_keys
    )
    previous_stats = prior.groupby("trade_date").agg(
        prev_limit_red_rate=("prev_limit_is_red", "mean"),
        prev_limit_relimit_rate=("prev_limit_is_relimit", "mean"),
        prev_limit_count=("instrument_id", "size"),
    )
    context = pd.DataFrame({"trade_date": dates})
    context["market_up_ratio"] = context["trade_date"].map(up_ratio)
    context["market_limit_count"] = context["trade_date"].map(closed).fillna(0).astype(int)
    context["market_limit_count_log"] = np.log1p(context["market_limit_count"])
    context["market_touched_limit_count"] = context["trade_date"].map(touched).fillna(0).astype(int)
    context["market_seal_success_rate"] = (
        context["market_limit_count"]
        / context["market_touched_limit_count"].replace(0, np.nan)
    )
    context["market_failed_limit_rate"] = 1.0 - context["market_seal_success_rate"]
    context = context.join(previous_stats, on="trade_date")
    return context


def build_factor_cache(
    snapshot: DataSnapshot,
    minute_snapshot: Path,
    minute_factor_cache: Path,
    stock_micro: pd.DataFrame,
    output: Path,
    *,
    formal_microstructure_features: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if output.exists():
        raise RuntimeError(f"immutable factor cache exists: {output}")
    selection = read_selection(minute_snapshot)
    minute_factor_manifest = json.loads(
        (minute_factor_cache / "manifest.json").read_text(encoding="utf-8")
    )
    minute_snapshot_manifest = json.loads(
        (minute_snapshot / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        minute_factor_manifest.get("parent_minute_snapshot_fingerprint")
        != minute_snapshot_manifest.get("fingerprint")
    ):
        raise RuntimeError("minute factor cache parent snapshot mismatch")
    minute_factor_path = minute_factor_cache / "factors.parquet"
    if sha256(minute_factor_path) != minute_factor_manifest.get("file_sha256"):
        raise RuntimeError("minute factor cache content hash mismatch")
    primitives = pd.read_parquet(
        minute_factor_path, columns=[*KEYS, *MINUTE_PRIMITIVE_FEATURES]
    )
    primitives["trade_date"] = pd.to_datetime(primitives["trade_date"])
    primitives["instrument_id"] = primitives["instrument_id"].astype(str)
    if len(primitives) != len(selection) or primitives.duplicated(KEYS).any():
        raise RuntimeError("minute primitive cache keys do not match selection")
    minute_parts = []
    by_day = {
        day: block for day, block in selection.groupby(selection["trade_date"].dt.date)
    }
    for path in sorted((minute_snapshot / "minute_bars").glob("trade_date=*/part.parquet")):
        bars = pd.read_parquet(path)
        if bars.empty:
            continue
        day = pd.Timestamp(bars["trade_date"].iloc[0]).date()
        minute_parts.append(compute_minute_extension_factors(bars, by_day[day]))
    minute = pd.concat(minute_parts, ignore_index=True)
    minute["trade_date"] = pd.to_datetime(minute["trade_date"])
    if len(minute) != len(selection) or minute.duplicated(KEYS).any():
        raise RuntimeError("minute extension cache keys do not match selection")

    start = selection["trade_date"].min().date()
    end = selection["trade_date"].max().date()
    cutoff = snapshot.metadata.as_of_date
    limits = snapshot.table(
        "daily_limit_events", start=start, end=end, cutoff=cutoff,
        fields=[
            *KEYS,
            "limit_type",
            "first_limit_value",
            "last_limit_value",
            "close_limit_value",
        ],
    )
    raw = snapshot.table(
        "daily_raw", start=start, end=end, cutoff=cutoff,
        fields=[*KEYS, "open", "close", "pre_close", "volume", "amount", "paused"],
    )
    money = snapshot.table(
        "daily_money_flow", start=start, end=end, cutoff=cutoff,
        fields=[*KEYS, "net_in_close", "turnover_value"],
    )
    membership = snapshot.table(
        "industry_membership", start=date(1990, 1, 1), end=end, cutoff=cutoff,
    )
    calendar = snapshot.table(
        "trade_calendar", start=date(2010, 1, 1), end=end, cutoff=cutoff,
        fields=["trade_date", "exchange"],
    )
    if calendar["exchange"].astype(str).eq("XSHG").any():
        calendar = calendar.loc[calendar["exchange"].astype(str).eq("XSHG")]
    for frame in (limits, raw, money):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    up_limits = limits.loc[limits["limit_type"].astype(str).str.lower().eq("up")]
    if up_limits.duplicated(KEYS).any():
        raise RuntimeError("up-limit events contain duplicate keys")
    industry = compute_industry_features(selection, limits, raw, membership, calendar)
    market_context = compute_market_context(selection, raw, limits, calendar)

    cache = selection[[*KEYS]].merge(minute, on=KEYS, validate="one_to_one")
    cache = cache.merge(primitives, on=KEYS, validate="one_to_one")
    cache = cache.merge(stock_micro[[*KEYS, *MICROSTRUCTURE_FEATURES]], on=KEYS, validate="one_to_one")
    seal_change = up_limits[[*KEYS, "first_limit_value", "last_limit_value"]].copy()
    first_value = pd.to_numeric(seal_change.pop("first_limit_value"), errors="coerce")
    last_value = pd.to_numeric(seal_change.pop("last_limit_value"), errors="coerce")
    seal_change["h5_last_first_limit_value_change_sym"] = symmetric_change(
        last_value, first_value
    )
    cache = cache.merge(seal_change, on=KEYS, how="left", validate="one_to_one")
    close_flow = money[[*KEYS]].copy()
    close_flow["fb_close_net_to_turnover"] = pd.to_numeric(
        money["net_in_close"], errors="coerce"
    ) / pd.to_numeric(money["turnover_value"], errors="coerce").replace(0, np.nan)
    cache = cache.merge(close_flow, on=KEYS, how="left", validate="one_to_one")
    cache = cache.merge(industry, on=KEYS, how="left", validate="one_to_one")
    cache = cache.replace([np.inf, -np.inf], np.nan).sort_values(KEYS).reset_index(drop=True)
    new_features = [
        *MINUTE_PRIMITIVE_FEATURES,
        *MICROSTRUCTURE_FEATURES,
        *MINUTE_EXTENSION_FEATURES,
        "h5_last_first_limit_value_change_sym",
        "fb_close_net_to_turnover",
        *INDUSTRY_FEATURES,
    ]
    coverage = {name: float(cache[name].notna().mean()) for name in new_features}
    formal = (
        list(MICROSTRUCTURE_FEATURES)
        if formal_microstructure_features is None
        else list(formal_microstructure_features)
    )
    failed = {name: coverage[name] for name in formal if coverage[name] < 0.98}
    if failed:
        raise RuntimeError(f"formal microstructure factor coverage below 98%: {failed}")
    stage = output.parent / f".{output.name}.staging-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    path = stage / "factors.parquet"
    market_path = stage / "market_context.parquet"
    atomic_parquet(cache[[*KEYS, *new_features]], path)
    atomic_parquet(market_context, market_path)
    meta = {
        "status": "passed",
        "formula_version": "first_board_single_model_v6",
        "rows": len(cache),
        "features": new_features,
        "coverage": coverage,
        "formal_microstructure_features": formal,
        "snapshot_id": snapshot.metadata.dataset_id,
        "snapshot_fingerprint": snapshot.metadata.fingerprint,
        "minute_snapshot": str(minute_snapshot),
        "minute_snapshot_manifest_sha256": sha256(minute_snapshot / "manifest.json"),
        "minute_factor_cache_sha256": sha256(minute_factor_cache / "factors.parquet"),
        "row_key_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(cache[KEYS], index=False).values.tobytes()
        ).hexdigest(),
        "sha256": sha256(path),
        "market_context_sha256": sha256(market_path),
        "market_context_rows": len(market_context),
        "market_context_features": [
            name for name in market_context.columns if name != "trade_date"
        ],
        "duplicate_key_rows": 0,
    }
    (stage / "single_model_v6.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(stage, output)
    return cache[[*KEYS, *new_features]], meta


def build_panels(
    base_panel: Path,
    cache: pd.DataFrame,
    snapshot: DataSnapshot,
    output: Path,
    *,
    repair_legacy_days_since_limit: bool = False,
    panel_start: date = date(2022, 1, 1),
    panel_end: date | None = None,
    model_features: list[str] | None = None,
    formal_microstructure_features: list[str] | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable panel output exists: {output}")
    start = panel_start
    end = panel_end or snapshot.metadata.as_of_date
    active_features = model_features or list(SINGLE_MODEL_FEATURES)
    market = snapshot.table(
        "daily_raw", start=start, end=end,
        cutoff=snapshot.metadata.as_of_date,
        fields=[*KEYS, "high", "limit_up", "paused", "volume"],
    )
    stage = output.parent / f".{output.name}.staging-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    # The former training panel was filtered on the old exit-return label.  The
    # new D+1 touch target matures earlier, so derive training rows from the full
    # candidate panel and only remove rows whose new label is unavailable.
    candidate = pd.read_parquet(base_panel / "candidate_panel.parquet")
    candidate["trade_date"] = pd.to_datetime(candidate["trade_date"])
    candidate = candidate.loc[
        candidate["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))
    ].copy()
    candidate = candidate.merge(cache, on=KEYS, how="left", validate="one_to_one")
    days_since_audit = None
    if repair_legacy_days_since_limit:
        legacy = pd.to_numeric(candidate["g3_days_since_limit"], errors="coerce")
        if legacy.dropna().lt(0).any():
            raise RuntimeError("legacy days-since-limit repair received negative values")
        candidate["g3_days_since_limit"] = legacy.add(1.0).fillna(-1.0)
        days_since_audit = {
            "legacy_nan_to_minus_one": int(legacy.isna().sum()),
            "legacy_zero_based_rows_shifted": int(legacy.notna().sum()),
            "remaining_nan": int(candidate["g3_days_since_limit"].isna().sum()),
        }
    missing = sorted(set(active_features) - set(candidate.columns))
    if missing:
        raise RuntimeError(f"candidate panel missing single-model features: {missing}")
    candidate, candidate_label_audit = attach_d1_touch_label(candidate, market)
    training = candidate.loc[candidate["label_d1_touch"].notna()].copy()
    formal_new_candidates = [
        *(
            list(MICROSTRUCTURE_FEATURES)
            if formal_microstructure_features is None
            else list(formal_microstructure_features)
        ),
        *MINUTE_EXTENSION_FEATURES,
        "h5_last_first_limit_value_change_sym",
        "fb_close_net_to_turnover",
        *INDUSTRY_FEATURES,
    ]
    formal_new = [name for name in formal_new_candidates if name in active_features]
    yearly_coverage: dict[str, Any] = {}
    for year, block in candidate.groupby(candidate["trade_date"].dt.year):
        values = {name: float(block[name].notna().mean()) for name in formal_new}
        yearly_coverage[str(year)] = values
        failed = {name: value for name, value in values.items() if value < 0.98}
        if failed:
            raise RuntimeError(f"formal new-factor coverage below 98% in {year}: {failed}")

    files: dict[str, Any] = {}
    for filename, frame in (
        ("training_panel.parquet", training),
        ("candidate_panel.parquet", candidate),
    ):
        label_audit = dict(candidate_label_audit)
        label_audit["rows_in_output"] = len(frame)
        fixed = ["trade_date", "instrument_id", "cal_idx", "is_first_board_hard"]
        trailing = [name for name in frame.columns if name not in {*fixed, *active_features}]
        frame = frame[[*fixed, *active_features, *trailing]]
        path = stage / filename
        atomic_parquet(frame, path)
        files[filename] = {
            "rows": len(frame),
            "sha256": sha256(path),
            "touch_label": label_audit,
            "feature_missing": {name: int(frame[name].isna().sum()) for name in active_features},
        }
    manifest = {
        "status": "ready_for_training",
        "feature_contract": (
            "first_board_single_model_historical29_v1"
            if active_features == SINGLE_MODEL_HISTORICAL29_FEATURES
            else "first_board_single_model_historical34_v1"
            if active_features == SINGLE_MODEL_HISTORICAL34_FEATURES
            else "first_board_single_model_historical38_v1"
            if active_features == SINGLE_MODEL_HISTORICAL38_FEATURES
            else "first_board_single_model_v6"
        ),
        "features": active_features,
        "feature_count": len(active_features),
        "snapshot_id": snapshot.metadata.dataset_id,
        "snapshot_fingerprint": snapshot.metadata.fingerprint,
        "base_panel": str(base_panel),
        "base_manifest_sha256": sha256(base_panel / "manifest.json"),
        "missing_policy": "retain in parquet; complete-case delete inside each fold",
        "legacy_days_since_limit_repair": days_since_audit,
        "formal_new_factor_yearly_coverage": yearly_coverage,
        "files": files,
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(stage, output)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--minute-snapshot", type=Path, required=True)
    parser.add_argument("--minute-factor-cache", type=Path, required=True)
    parser.add_argument("--base-panel", type=Path, required=True)
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--microstructure-out", type=Path, required=True)
    parser.add_argument("--factor-out", type=Path, required=True)
    parser.add_argument("--panel-out", type=Path, required=True)
    parser.add_argument("--repair-legacy-days-since-limit", action="store_true")
    parser.add_argument(
        "--feature-contract",
        choices=["full44", "historical38", "historical34", "historical29"],
        default="full44",
    )
    parser.add_argument("--panel-start", type=date.fromisoformat, default=date(2022, 1, 1))
    parser.add_argument("--panel-end", type=date.fromisoformat)
    args = parser.parse_args()
    historical = args.feature_contract in {"historical38", "historical34", "historical29"}
    formal_micro = (
        []
        if args.feature_contract in {"historical34", "historical29"}
        else ["h4_smart_money_price"]
        if historical
        else list(MICROSTRUCTURE_FEATURES)
    )
    model_features = (
        list(SINGLE_MODEL_HISTORICAL29_FEATURES)
        if args.feature_contract == "historical29"
        else
        list(SINGLE_MODEL_HISTORICAL34_FEATURES)
        if args.feature_contract == "historical34"
        else
        list(SINGLE_MODEL_HISTORICAL38_FEATURES)
        if historical
        else list(SINGLE_MODEL_FEATURES)
    )
    snapshot = DataSnapshot(args.snapshot, verify_hashes=True)
    v7_manifest = json.loads((args.snapshot / "manifest.json").read_text())
    minute_manifest = json.loads((args.minute_snapshot / "manifest.json").read_text())
    selection = read_selection(args.minute_snapshot)
    if args.microstructure_out.exists():
        stock, source_manifest = load_microstructure_snapshot(
            args.microstructure_out, v7_manifest, minute_manifest
        )
    else:
        stock, _, source_manifest = publish_microstructure_snapshot(
            args.extracted,
            selection,
            v7_manifest,
            minute_manifest,
            args.microstructure_out,
            formal_microstructure_features=formal_micro,
        )
    cache, factor_meta = build_factor_cache(
        snapshot,
        args.minute_snapshot,
        args.minute_factor_cache,
        stock,
        args.factor_out,
        formal_microstructure_features=formal_micro,
    )
    panel_manifest = build_panels(
        args.base_panel,
        cache,
        snapshot,
        args.panel_out,
        repair_legacy_days_since_limit=args.repair_legacy_days_since_limit,
        panel_start=args.panel_start,
        panel_end=args.panel_end,
        model_features=model_features,
        formal_microstructure_features=formal_micro,
    )
    print(json.dumps({
        "microstructure": source_manifest,
        "factor_cache": factor_meta,
        "panels": panel_manifest,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
