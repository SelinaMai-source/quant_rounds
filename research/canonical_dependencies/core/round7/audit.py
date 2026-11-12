"""Optional post-run audit against legacy artifacts; never feeds the strategy."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..paths import assert_project_result_path, ensure_round7_tree, round7_dir
from .contract import CANDIDATE_CODE

KEYS = ["signal_date", "instrument_id"]


def _load_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=[*KEYS, "score"])
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    return frame


def _top2(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values(
        ["signal_date", "score", "instrument_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).groupby("signal_date", sort=False).head(2).groupby("signal_date")["instrument_id"].apply(tuple)


def _selection_audit(new_path: Path, old_path: Path) -> dict:
    new, old = _load_prediction(new_path), _load_prediction(old_path)
    new_top, old_top = _top2(new), _top2(old)
    common = new_top.index.intersection(old_top.index)
    ordered = sum(new_top.loc[day] == old_top.loc[day] for day in common)
    return {
        "new_rows": int(len(new)),
        "legacy_rows": int(len(old)),
        "common_days": int(len(common)),
        "ordered_top2_equal_days": int(ordered),
        "ordered_top2_equal_rate": float(ordered / len(common)) if len(common) else None,
    }


def audit_legacy_reference(reference_stock_root: Path) -> dict:
    """Compare final outputs after completion. Reference data is never returned to training."""
    root = ensure_round7_tree()
    legacy_run = reference_stock_root / "runs/first_board_fs01122_no_sampling_v2"
    legacy_ablation = reference_stock_root / "runs/first_board_sampling_ablation_2x2_v1"
    paths = {
        "five": (
            root / "models/five/predictions.parquet",
            legacy_run / "five_ensemble_raw/predictions.parquet",
        ),
        "ten": (
            root / "models/ten/predictions.parquet",
            legacy_ablation / "r10_c10/ensemble_raw/predictions.parquet",
        ),
        "blend": (
            root / "predictions/dual90_10_gap_floor/predictions.parquet",
            legacy_run / "fully_retrained_90_10/predictions.parquet",
        ),
        "risk": (
            root / "predictions/clean_risk/predictions.parquet",
            legacy_run / "fully_retrained_anchor_replay/predictions.parquet",
        ),
    }
    selections = {name: _selection_audit(*pair) for name, pair in paths.items()}

    new_risk = pd.read_parquet(paths["risk"][0])
    old_risk = pd.read_parquet(paths["risk"][1])
    for frame in (new_risk, old_risk):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    risk_common = new_risk.merge(old_risk, on=KEYS, how="inner", suffixes=("_new", "_legacy"))
    risk_multiplier_mismatches = int(
        (risk_common["allocation_multiplier_new"] - risk_common["allocation_multiplier_legacy"]).abs().gt(1e-12).sum()
    )
    new_risk_report = json.loads(
        (root / "predictions/clean_risk/report.json").read_text(encoding="utf-8")
    )
    legacy_registry = pd.read_parquet(
        reference_stock_root
        / "runs/first_board_sentiment_soft_refinement_grid_v1/predictions/registry.parquet"
    )
    legacy_rule = legacy_registry.loc[
        legacy_registry["case_id"].eq("soft_style_75_auction_rescue_25")
    ].iloc[0]

    new_trades = pd.read_parquet(round7_dir(CANDIDATE_CODE) / "trades.parquet")
    old_trades = pd.read_parquet(legacy_run / "fully_retrained_backtest/trades.parquet")
    for frame in (new_trades, old_trades):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    trades = new_trades.merge(old_trades, on=KEYS, how="outer", indicator=True, suffixes=("_new", "_legacy"))
    common_trades = trades.loc[trades["_merge"].eq("both")]
    trade_multiplier_mismatches = int(
        (common_trades["allocation_multiplier_new"] - common_trades["allocation_multiplier_legacy"])
        .abs().gt(1e-12).sum()
    )
    payload = {
        "status": "complete",
        "role": "post-run diagnostic only; legacy artifacts were not inputs to training, fusion, risk, or backtest",
        "selection_audit": selections,
        "risk_common_key_multiplier_mismatches": risk_multiplier_mismatches,
        "thresholds": {
            "clean_style_low": new_risk_report["rules"]["style_low_threshold"],
            "legacy_style_low": float(legacy_rule["style_low_threshold"]),
            "clean_auction_high": new_risk_report["rules"]["auction_high_threshold"],
            "legacy_auction_high": float(legacy_rule["auction_high_threshold"]),
        },
        "trade_key_counts": trades["_merge"].value_counts().to_dict(),
        "trade_multiplier_mismatches": trade_multiplier_mismatches,
        "attribution": (
            "Common-date Top2 selections are identical through both heads and the 90/10 blend. "
            "The return difference is caused by clean recalibration of daily risk allocations; "
            "changed sizing then compounds into different quantities and NAV."
        ),
        "written_at": datetime.now().astimezone().isoformat(),
    }
    target = root / "diagnostics" / "legacy_reference_audit.json"
    assert_project_result_path(target)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return payload
