#!/usr/bin/env python
"""Build and run the isolated 45-factor model plus causal D+1 open gap."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["trade_date", "instrument_id"]
OPEN_GAP = "d1_open_gap"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def build_panels(
    project_root: Path,
    base_panel: Path,
    snapshot_path: Path,
    output_dir: Path,
) -> tuple[list[str], dict]:
    sys.path.insert(0, str(project_root / "src"))
    from zyquant.data import DataSnapshot

    source_manifest_path = base_panel / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    features = list(source_manifest["features"])
    if not features or OPEN_GAP in features:
        raise RuntimeError("source panel feature contract is invalid")
    snapshot = DataSnapshot(snapshot_path, verify_hashes=True)
    if source_manifest["snapshot_fingerprint"] != snapshot.metadata.fingerprint:
        raise RuntimeError("source panel and snapshot fingerprints differ")

    frames = {
        name: pd.read_parquet(base_panel / name)
        for name in ("training_panel.parquet", "candidate_panel.parquet")
    }
    entry_dates = pd.concat(
        [pd.to_datetime(frame["next1_date"]) for frame in frames.values()],
        ignore_index=True,
    ).dropna()
    entry = snapshot.table(
        "daily_raw",
        start=entry_dates.min().date(),
        end=entry_dates.max().date(),
        cutoff=snapshot.metadata.as_of_date,
        fields=[
            "trade_date", "instrument_id", "open", "pre_close", "volume", "paused",
        ],
    ).rename(
        columns={
            "trade_date": "next1_date",
            "open": "d1_open",
            "pre_close": "d1_pre_close",
            "volume": "d1_volume",
            "paused": "d1_paused",
        }
    )
    entry["next1_date"] = pd.to_datetime(entry["next1_date"])
    entry["instrument_id"] = entry["instrument_id"].astype(str)
    if entry.duplicated(["next1_date", "instrument_id"]).any():
        raise RuntimeError("daily_raw entry keys are duplicated")

    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for name, frame in frames.items():
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["next1_date"] = pd.to_datetime(frame["next1_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        original_key_hash = pd.util.hash_pandas_object(frame[KEYS], index=False).values.tobytes()
        frame = frame.merge(
            entry,
            on=["next1_date", "instrument_id"],
            how="left",
            validate="many_to_one",
            sort=False,
        )
        valid = (
            ~frame["d1_paused"].fillna(True).astype(bool)
            & pd.to_numeric(frame["d1_volume"], errors="coerce").gt(0)
            & pd.to_numeric(frame["d1_open"], errors="coerce").gt(0)
            & pd.to_numeric(frame["d1_pre_close"], errors="coerce").gt(0)
        )
        frame[OPEN_GAP] = np.where(
            valid,
            pd.to_numeric(frame["d1_open"], errors="coerce")
            / pd.to_numeric(frame["d1_pre_close"], errors="coerce")
            - 1.0,
            np.nan,
        )
        frame = frame.drop(columns=["d1_open", "d1_pre_close", "d1_volume", "d1_paused"])
        new_key_hash = pd.util.hash_pandas_object(frame[KEYS], index=False).values.tobytes()
        if original_key_hash != new_key_hash:
            raise RuntimeError(f"{name} key order changed")
        path = output_dir / name
        _atomic_parquet(frame, path)
        post_2022 = frame["trade_date"].ge("2022-01-01")
        outputs[name] = {
            "rows": len(frame),
            "sha256": _sha256(path),
            "post_2022_rows": int(post_2022.sum()),
            "post_2022_open_gap_nan": int(frame.loc[post_2022, OPEN_GAP].isna().sum()),
        }

    contract = [*features, OPEN_GAP]
    manifest = {
        "status": "ready_for_isolated_training",
        "feature_contract": f"{source_manifest.get('feature_contract', 'panel')}_plus_d1_open_gap_v1",
        "feature_count": len(contract),
        "features": contract,
        "new_feature": {
            "name": OPEN_GAP,
            "formula": "D+1 open / D+1 pre_close - 1",
            "available_at": "D+1 09:25 after opening call auction",
            "model_decision_at": "D+1 09:25",
        },
        "execution_assumption": "D+1 official open * (1 + 12.3 bp)",
        "source_panel": str(base_panel),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "snapshot_fingerprint": snapshot.metadata.fingerprint,
        "outputs": outputs,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return contract, manifest


def train(
    project_root: Path,
    panel_dir: Path,
    snapshot_path: Path,
    run_dir: Path,
    features: list[str],
) -> dict:
    sys.path.insert(0, str(project_root / "src"))
    import first_board_hermes.regressor as regressor
    from zyquant.data import DataSnapshot

    regressor.ALL_FEATURES = features
    snapshot = DataSnapshot(snapshot_path, verify_hashes=True)
    calendar = snapshot.table(
        "trade_calendar",
        start=date(2010, 1, 1),
        end=snapshot.metadata.as_of_date,
        cutoff=snapshot.metadata.as_of_date,
        fields=["trade_date", "exchange"],
    )
    if calendar["exchange"].astype(str).eq("XSHG").any():
        calendar = calendar.loc[calendar["exchange"].astype(str).eq("XSHG")]
    _, report = regressor.rolling_session_train_predict(
        pd.read_parquet(panel_dir / "training_panel.parquet"),
        pd.read_parquet(panel_dir / "candidate_panel.parquet"),
        calendar["trade_date"].drop_duplicates().sort_values(),
        run_dir,
        features=features,
        first_prediction_year=2022,
        last_prediction_year=2026,
        train_sessions=240,
        target="label_exit_rule_net",
        label_end_column="actual_exit_date",
        sample_start=date(2022, 1, 1),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--base-panel", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--panel-out", type=Path, required=True)
    parser.add_argument("--run-out", type=Path)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    features, manifest = build_panels(
        args.project_root, args.base_panel, args.snapshot, args.panel_out
    )
    if args.build_only:
        print(json.dumps({"panel": manifest["outputs"]}, ensure_ascii=False, indent=2))
        return 0
    if args.run_out is None:
        raise ValueError("--run-out is required unless --build-only is used")
    report = train(args.project_root, args.panel_out, args.snapshot, args.run_out, features)
    print(json.dumps({
        "panel": manifest["outputs"],
        "prediction_rows": report["prediction_rows"],
        "primary_evaluation_metric": report["primary_evaluation_metric"],
        "oos_regression_metrics": report["oos_regression_metrics"],
        "ranking_diagnostics": report["ranking_diagnostics"],
        "yearly": report["yearly"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
