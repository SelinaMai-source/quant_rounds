"""Original B2 Top-1 selection with ranked Top-5 fallback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_b2_top1_predictions(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    daily_raw: pd.DataFrame,
    output_dir: Path,
    *,
    candidate_top_n: int = 5,
    market_up_ratio_min: float = 0.45,
    min_open_gap: float = -0.02,
    max_open_gap: float = 0.07,
    min_signal_amount: float = 10_000_000.0,
    buy_slippage_bps: float = 12.3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply D-close market gate and D+1-open B2 filters without label use."""
    pred = predictions.copy()
    pred["signal_date"] = pd.to_datetime(pred["signal_date"])
    pred["instrument_id"] = pred["instrument_id"].astype(str)
    pred = pred.sort_values(
        ["signal_date", "score", "instrument_id"], ascending=[True, False, True]
    )
    pred["rank"] = pred.groupby("signal_date").cumcount() + 1
    pool = pred.loc[pred["rank"].le(candidate_top_n)].copy()

    panel = candidates.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["instrument_id"] = panel["instrument_id"].astype(str)
    panel = panel[["trade_date", "instrument_id", "next1_date"]].rename(
        columns={"trade_date": "signal_date", "next1_date": "entry_date"}
    )
    panel["entry_date"] = pd.to_datetime(panel["entry_date"])
    pool = pool.merge(panel, on=["signal_date", "instrument_id"], how="left", validate="one_to_one")

    raw = daily_raw.copy()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    raw["instrument_id"] = raw["instrument_id"].astype(str)
    raw["ret"] = pd.to_numeric(raw["close"], errors="coerce") / pd.to_numeric(
        raw["pre_close"], errors="coerce"
    ) - 1.0
    market = raw.groupby("trade_date", as_index=False).agg(
        market_up_ratio=("ret", lambda values: float(values.gt(0).mean())),
        market_rows=("instrument_id", "nunique"),
    ).rename(columns={"trade_date": "signal_date"})
    signal = raw[["trade_date", "instrument_id", "close", "amount"]].rename(
        columns={
            "trade_date": "signal_date",
            "close": "signal_close",
            "amount": "signal_amount",
        }
    )
    entry = raw[
        ["trade_date", "instrument_id", "open", "low", "limit_up", "paused", "volume"]
    ].rename(
        columns={
            "trade_date": "entry_date",
            "open": "entry_open",
            "low": "entry_low",
            "limit_up": "entry_limit_up",
            "paused": "entry_paused",
            "volume": "entry_volume",
        }
    )
    pool = pool.merge(market, on="signal_date", how="left", validate="many_to_one")
    pool = pool.merge(signal, on=["signal_date", "instrument_id"], how="left", validate="one_to_one")
    pool = pool.merge(entry, on=["entry_date", "instrument_id"], how="left", validate="one_to_one")

    selected: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    slip = buy_slippage_bps / 10_000.0
    for signal_date, block in pool.groupby("signal_date", sort=True):
        block = block.sort_values("rank")
        if block["market_up_ratio"].isna().all() or float(block["market_up_ratio"].iloc[0]) < market_up_ratio_min:
            for row in block.itertuples(index=False):
                attempts.append(_attempt(row, "market_gate_off", False))
            continue
        for row in block.itertuples(index=False):
            reason = _b2_reason(
                row,
                min_open_gap=min_open_gap,
                max_open_gap=max_open_gap,
                min_signal_amount=min_signal_amount,
                buy_slippage=slip,
            )
            passed = reason == "ok"
            attempts.append(_attempt(row, reason, passed))
            if passed:
                selected.append({
                    "signal_date": signal_date,
                    "instrument_id": str(row.instrument_id),
                    "score": float(row.score),
                    "b2_rank": int(row.rank),
                    "entry_date": row.entry_date,
                    "market_up_ratio": float(row.market_up_ratio),
                })
                break

    result = pd.DataFrame(selected)
    if result.empty:
        raise RuntimeError("B2 filters selected no predictions")
    result = result.sort_values(["signal_date", "instrument_id"]).reset_index(drop=True)
    attempt_frame = pd.DataFrame(attempts)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.parquet"
    attempt_path = output_dir / "b2_attempts.parquet"
    result.to_parquet(prediction_path, index=False)
    attempt_frame.to_parquet(attempt_path, index=False)
    report = {
        "status": "completed",
        "contract": "score Top-5 at D close; market up-ratio gate; first B2-pass name at D+1 open",
        "parameters": {
            "candidate_top_n": candidate_top_n,
            "market_up_ratio_min": market_up_ratio_min,
            "min_open_gap": min_open_gap,
            "max_open_gap": max_open_gap,
            "min_signal_amount": min_signal_amount,
            "buy_slippage_bps": buy_slippage_bps,
        },
        "input_prediction_days": int(pred["signal_date"].nunique()),
        "selected_days": int(result["signal_date"].nunique()),
        "no_selection_days": int(pred["signal_date"].nunique() - result["signal_date"].nunique()),
        "selected_rank_counts": {
            str(key): int(value) for key, value in result["b2_rank"].value_counts().sort_index().items()
        },
        "attempt_reasons": {
            str(key): int(value) for key, value in attempt_frame["reason"].value_counts().items()
        },
        "predictions_sha256": _sha256(prediction_path),
        "attempts_sha256": _sha256(attempt_path),
    }
    (output_dir / "b2_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result, report


def _b2_reason(
    row,
    *,
    min_open_gap: float,
    max_open_gap: float,
    min_signal_amount: float,
    buy_slippage: float,
) -> str:
    required = [row.signal_close, row.signal_amount, row.entry_open, row.entry_volume]
    if any(pd.isna(value) for value in required) or pd.isna(row.entry_date):
        return "missing_bar"
    if bool(row.entry_paused) if not pd.isna(row.entry_paused) else True:
        return "halt_or_untradable"
    if float(row.entry_volume) <= 0:
        return "halt_or_untradable"
    if float(row.signal_amount) < min_signal_amount:
        return "low_liquidity"
    gap = float(row.entry_open) / float(row.signal_close) - 1.0
    if gap > max_open_gap:
        return "gap_too_high"
    if gap < min_open_gap:
        return "gap_too_low"
    if not pd.isna(row.entry_limit_up):
        if float(row.entry_open) >= float(row.entry_limit_up) - 1e-4:
            return "open_limit_up"
        if float(row.entry_open) * (1.0 + buy_slippage) > float(row.entry_limit_up) + 1e-8:
            return "slippage_over_limit"
    return "ok"


def _attempt(row, reason: str, selected: bool) -> dict[str, Any]:
    return {
        "signal_date": row.signal_date,
        "entry_date": row.entry_date,
        "instrument_id": str(row.instrument_id),
        "rank": int(row.rank),
        "score": float(row.score),
        "market_up_ratio": float(row.market_up_ratio) if not pd.isna(row.market_up_ratio) else np.nan,
        "selected": selected,
        "reason": reason,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
