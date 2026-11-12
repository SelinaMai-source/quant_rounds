"""Round10 overlay：B03 replacement + A/C/E/F/G 变体。不改 fused score，除非 E 扰动。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..round2.overlays import overlay_summary
from ..round8.overlays import MIN_KEEP, apply_rank_multipliers
from ..round9.overlays import v3_date_mult
from .allocator import b03_nav, cell_nav, load_alloc, nav_to_mult_custom, sep_bucket
from .contract import (
    A03_CELL,
    A04_CELL,
    B03_CELL,
    B03_OVERLAY,
    BASE_CASH,
    E02_MULTIPLIED,
    PARENT_CODE,
    RANK_NAV,
    TOP1_LOCK,
)

IMPLEMENTED_KINDS = {
    "exact_b03", "b03", "v4", "preregistered",
    "agree_only", "sep_only", "cell_table", "daily_file",
    "perturb", "sep_neighbor", "alloc_neighbor", "capacity",
}


def _meta(name: str, nav: pd.Series, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {
        "round10_kind": name,
        "parent": PARENT_CODE,
        "top1_lock": TOP1_LOCK,
        "e02_multiplied": E02_MULTIPLIED,
        "allocator": name,
        "top2_nav_mean": float(pd.to_numeric(nav, errors="coerce").mean()) if len(nav) else None,
        "top2_zero_days": int((pd.to_numeric(nav, errors="coerce") <= 1e-12).sum()) if len(nav) else 0,
    }
    if extra:
        out.update(extra)
    return out


def _apply(predictions: pd.DataFrame, date_mult: pd.Series, nav: pd.Series, *, top2_nav: float = 0.10):
    top2_extra = nav_to_mult_custom(nav, top2_nav).reindex(date_mult.index).fillna(MIN_KEEP)
    overlayed = apply_rank_multipliers(predictions, date_mult, top2_extra=top2_extra)
    return overlayed, overlay_summary(date_mult)


def _parse_cell(spec: dict[str, Any]) -> dict[tuple[str, str], float]:
    raw = spec.get("cell") or {}
    if raw and isinstance(next(iter(raw.keys())), tuple):
        return {k: float(v) for k, v in raw.items()}
    out = {}
    for key, val in raw.items():
        if isinstance(key, str) and "_" in key:
            a, s = key.split("_", 1)
            out[(a, s)] = float(val)
    return out or dict(B03_CELL)


def _capacity_clip(
    overlayed: pd.DataFrame,
    *,
    capital: float,
    participation: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from ..paths import ROUND10_ROOT

    amount_path = ROUND10_ROOT / "capacity" / "daily_amount.parquet"
    if not amount_path.is_file():
        return overlayed, {"capacity_clip": "NO_AMOUNT_CACHE", "n_clipped": 0}
    amt = pd.read_parquet(amount_path)
    amt["trade_date"] = pd.to_datetime(amt["trade_date"]).dt.normalize()
    amt["instrument_id"] = amt["instrument_id"].astype(str)
    work = overlayed.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
    work["instrument_id"] = work["instrument_id"].astype(str)
    work["_rank"] = work.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    merged = work.merge(
        amt.rename(columns={"trade_date": "signal_date"}),
        on=["signal_date", "instrument_id"],
        how="left",
    )
    nav_path = ROUND10_ROOT / "R10_00_baseline_v4" / "trades.parquet"
    nav_map = {}
    if nav_path.is_file():
        t = pd.read_parquet(nav_path)
        t["signal_date"] = pd.to_datetime(t["signal_date"]).dt.normalize()
        nav_map = (
            t.groupby("signal_date")["entry_portfolio_nav"].median().astype(float).to_dict()
        )
    scale = capital / float(BASE_CASH)
    fracs = {1: float(RANK_NAV[0]), 2: float(RANK_NAV[1])}
    budgets = []
    for _, row in merged.iterrows():
        day = row["signal_date"]
        nav = float(nav_map.get(day, BASE_CASH)) * scale
        rank = int(row["_rank"]) if pd.notna(row["_rank"]) else 0
        frac = fracs.get(rank, 0.0)
        mult = float(row.get("allocation_multiplier") or 0.0)
        budgets.append(nav * frac * mult)
    merged["_budget"] = budgets
    max_notional = participation * pd.to_numeric(merged.get("amount"), errors="coerce")
    ratio = (max_notional / merged["_budget"].replace(0, np.nan)).clip(upper=1.0)
    ratio = ratio.where(merged["_budget"] > 0, 1.0).fillna(1.0)
    clipped = ratio < 0.999
    work.loc[clipped.to_numpy(), "allocation_multiplier"] = (
        work.loc[clipped.to_numpy(), "allocation_multiplier"].astype(float) * ratio.loc[clipped].to_numpy()
    ).clip(MIN_KEEP, 1.0)
    meta = {
        "capacity_clip": "applied",
        "capital": capital,
        "participation": participation,
        "n_rows": int(len(work)),
        "n_clipped": int(clipped.sum()),
        "median_ratio": float(ratio.median()) if len(ratio) else None,
        "amount_coverage": float(pd.to_numeric(merged.get("amount"), errors="coerce").notna().mean()),
    }
    return work.drop(columns=["_rank"], errors="ignore"), meta


def build_overlayed(
    spec: dict[str, Any],
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    kind = spec.get("kind") or "exact_b03"
    cfg: dict[str, Any] = {}
    date_mult, geo = v3_date_mult(predictions)
    idx = date_mult.index
    agree = geo["head_state"].astype(str).reindex(idx)
    sep_p = geo["top2_minus_median_p250"].reindex(idx)

    if kind in {"exact_b03", "b03", "v4"} or (
        kind == "preregistered" and spec.get("allocator") == "B03_conservative"
    ):
        nav = b03_nav(agree, sep_bucket(sep_p))
        overlayed, summary = _apply(predictions, date_mult, nav)
        meta = _meta("exact_b03", nav, summary)
        meta["allocator"] = "B03_conservative"
        return overlayed, meta, cfg

    if kind == "agree_only":
        mapping = spec.get("agree_nav") or {"A": 0.10, "B": 0.05, "C": 0.00}
        nav = agree.map(mapping).astype(float).fillna(0.0)
        overlayed, summary = _apply(predictions, date_mult, nav)
        return overlayed, _meta("agree_only", nav, summary), cfg

    if kind == "sep_only":
        mapping = spec.get("sep_nav") or {"LOW": 0.00, "MID": 0.05, "HIGH": 0.10}
        sep = sep_bucket(sep_p)
        nav = sep.map(mapping).astype(float).fillna(0.0)
        overlayed, summary = _apply(predictions, date_mult, nav)
        return overlayed, _meta("sep_only", nav, summary), cfg

    if kind == "cell_table":
        table = _parse_cell(spec)
        nav = cell_nav(agree, sep_bucket(sep_p), table)
        overlayed, summary = _apply(predictions, date_mult, nav)
        return overlayed, _meta("cell_table", nav, summary), cfg

    if kind == "sep_neighbor":
        low = float(spec.get("low", 0.20))
        mid = float(spec.get("mid", 0.50))
        nav = b03_nav(agree, sep_bucket(sep_p, low=low, mid=mid))
        overlayed, summary = _apply(predictions, date_mult, nav)
        return overlayed, _meta("sep_neighbor", nav, {**summary, "low": low, "mid": mid}), cfg

    if kind == "alloc_neighbor":
        top2_nav = float(spec.get("top2_nav") or 0.10)
        ladder = tuple(spec.get("ladder") or (0.0, 0.05, 0.10))
        raw = b03_nav(agree, sep_bucket(sep_p))
        scaled = raw * (top2_nav / 0.10)
        overlayed, summary = _apply(predictions, date_mult, scaled, top2_nav=top2_nav)
        cfg["rank_name_budget_nav_fractions"] = (float(RANK_NAV[0]), top2_nav)
        return overlayed, _meta("alloc_neighbor", scaled, {**summary, "ladder": ladder, "top2_nav": top2_nav}), cfg

    if kind == "daily_file":
        key = str(spec.get("alloc_code") or spec.get("code") or "")
        if not key:
            raise RuntimeError(f"daily allocator missing alloc_code: {spec}")
        loaded = load_alloc(key).reindex(idx)
        nav = loaded.fillna(0.0)
        overlayed, summary = _apply(predictions, date_mult, nav)
        return overlayed, _meta("daily_file", nav, {**summary, "alloc_code": key}), cfg

    if kind == "perturb":
        scale = float(spec.get("scale") or 0.005)
        seed = int(spec.get("seed") or 17)
        work = predictions.copy()
        work["signal_date"] = pd.to_datetime(work["signal_date"])
        pct = work.groupby("signal_date", sort=False)["score"].rank(method="first", pct=True)
        rng = np.random.default_rng(seed)
        noise = rng.uniform(-scale, scale, size=len(work))
        work["score"] = (pct.to_numpy() + noise).clip(0.0, 1.0)
        nav = b03_nav(agree, sep_bucket(sep_p))
        overlayed, summary = _apply(work, date_mult, nav)
        meta = _meta("perturb", nav, {**summary, "scale": scale, "seed": seed, "score_perturbed": True})
        return overlayed, meta, cfg

    if kind == "capacity":
        nav = b03_nav(agree, sep_bucket(sep_p))
        overlayed, summary = _apply(predictions, date_mult, nav)
        overlayed, clip_meta = _capacity_clip(
            overlayed,
            capital=float(spec.get("capital") or BASE_CASH),
            participation=float(spec.get("participation") or 0.03),
        )
        meta = _meta("capacity", nav, {**summary, **clip_meta})
        return overlayed, meta, cfg

    raise RuntimeError(f"unknown overlay kind: {spec}")


__all__ = ["apply_rank_multipliers", "build_overlayed"]
