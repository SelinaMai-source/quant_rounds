"""Gap eligibility and deterministic 90/10 dual-head fusion."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ..paths import assert_project_result_path, ensure_round7_tree
from ..stock_readonly import sha256_file
from .contract import GAP_FLOOR, TEN_HEAD, TEN_WEIGHT

KEYS = ["signal_date", "instrument_id"]


def _load_head(name: str) -> pd.DataFrame:
    path = ensure_round7_tree() / "models" / name / "predictions.parquet"
    frame = pd.read_parquet(path, columns=[*KEYS, "score"])
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    if frame.duplicated(KEYS).any():
        raise RuntimeError(f"duplicate {name} prediction keys")
    return frame.rename(columns={"score": f"score_{name}"})


def fuse_predictions(five: pd.DataFrame, ten: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    work = five.merge(ten, on=KEYS, how="inner", validate="one_to_one")
    if len(work) != len(five) or len(work) != len(ten):
        raise RuntimeError("five/ten prediction universes differ")
    work = work.merge(gaps, on=KEYS, how="left", validate="one_to_one")
    work["d1_open_gap"] = pd.to_numeric(work["d1_open_gap"], errors="coerce")
    work = work.loc[np.isfinite(work["d1_open_gap"]) & work["d1_open_gap"].ge(GAP_FLOOR)].copy()
    if work.empty:
        raise RuntimeError("gap eligibility removed all predictions")
    work["pct_five"] = work.groupby("signal_date")["score_five"].rank(method="average", pct=True)
    work["pct_ten"] = work.groupby("signal_date")["score_ten"].rank(method="average", pct=True)
    work["score"] = TEN_WEIGHT * work["pct_ten"] + (1.0 - TEN_WEIGHT) * work["pct_five"]
    output = work[[*KEYS, "score"]].sort_values(
        ["signal_date", "score", "instrument_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if output.duplicated(KEYS).any():
        raise RuntimeError("duplicate blended prediction keys")
    return output


def build_blend(*, resume: bool = True) -> dict:
    root = ensure_round7_tree() / "predictions" / "dual90_10_gap_floor"
    root.mkdir(parents=True, exist_ok=True)
    target = root / "predictions.parquet"
    report_path = root / "report.json"
    assert_project_result_path(target)
    if resume and target.is_file() and report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))

    five_path = ensure_round7_tree() / "models" / "five" / "predictions.parquet"
    ten_path = ensure_round7_tree() / "models" / "ten" / "predictions.parquet"
    five = _load_head("five")
    ten = _load_head("ten")
    panel_path = TEN_HEAD.panel_dir / "candidate_panel.parquet"
    gaps = pd.read_parquet(panel_path, columns=["trade_date", "instrument_id", "d1_open_gap"])
    gaps = gaps.rename(columns={"trade_date": "signal_date"})
    gaps["signal_date"] = pd.to_datetime(gaps["signal_date"])
    gaps["instrument_id"] = gaps["instrument_id"].astype(str)
    if gaps.duplicated(KEYS).any():
        raise RuntimeError("duplicate gap keys")
    output = fuse_predictions(five, ten, gaps)
    output.to_parquet(target, index=False)
    merged_rows = len(five.merge(ten, on=KEYS, how="inner"))
    payload = {
        "status": "complete",
        "ordering": [
            "five-seed rank ensemble per head",
            "finite d1_open_gap >= -0.01",
            "per-head same-day percentile rerank",
            "0.90 ten-bin + 0.10 five-bin",
            "instrument_id ascending stable tie break",
        ],
        "gap_floor": GAP_FLOOR,
        "ten_weight": TEN_WEIGHT,
        "input_rows": int(merged_rows),
        "filtered_rows": int(merged_rows - len(output)),
        "output_rows": int(len(output)),
        "output_days": int(output["signal_date"].nunique()),
        "five_sha256": sha256_file(five_path),
        "ten_sha256": sha256_file(ten_path),
        "candidate_panel_sha256": sha256_file(panel_path),
        "output_sha256": sha256_file(target),
        "written_at": datetime.now().astimezone().isoformat(),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
