"""Train the two frozen no-sampling ranking heads from read-only panels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..paths import PREDICTION_START, SIGNAL_END, assert_project_result_path, ensure_round7_tree
from ..stock_readonly import ensure_stock_src_on_path, sha256_file
from .contract import FIVE_HEAD, SEEDS, TEN_HEAD, HeadContract, model_parameters


def _head_root(head: HeadContract) -> Path:
    path = ensure_round7_tree() / "models" / head.name
    path.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(path)
    return path


def _load_panel(head: HeadContract) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_path = head.panel_dir / "training_panel.parquet"
    candidate_path = head.panel_dir / "candidate_panel.parquet"
    required = [
        "trade_date", "instrument_id", *head.features, head.target,
        head.label_end_column,
    ]
    training = pd.read_parquet(training_path, columns=list(dict.fromkeys(required)))
    candidate = pd.read_parquet(
        candidate_path,
        columns=list(dict.fromkeys(["trade_date", "instrument_id", *head.features, head.target])),
    )
    for frame in (training, candidate):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    training[head.label_end_column] = pd.to_datetime(training[head.label_end_column])
    return training, candidate


def _rank_signature(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame[["signal_date", "instrument_id", "score"]].copy()
    ranked["signal_date"] = pd.to_datetime(ranked["signal_date"])
    ranked = ranked.sort_values(
        ["signal_date", "score", "instrument_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    ranked["rank"] = ranked.groupby("signal_date", sort=False).cumcount()
    return ranked[["signal_date", "instrument_id", "rank"]].sort_values(
        ["signal_date", "instrument_id"]
    ).reset_index(drop=True)


def assert_seed_rank_invariance(frames: Iterable[pd.DataFrame]) -> dict:
    items = list(frames)
    if not items:
        raise ValueError("no seed predictions")
    reference = _rank_signature(items[0])
    mismatches = []
    for index, frame in enumerate(items[1:], start=1):
        candidate = _rank_signature(frame)
        same_keys = reference[["signal_date", "instrument_id"]].equals(
            candidate[["signal_date", "instrument_id"]]
        )
        mismatch_rows = (
            int((reference["rank"] != candidate["rank"]).sum())
            if same_keys else max(len(reference), len(candidate))
        )
        mismatches.append({"seed_index": index, "mismatch_rows": mismatch_rows})
    passed = all(row["mismatch_rows"] == 0 for row in mismatches)
    if not passed:
        raise RuntimeError(f"no-sampling seed rank invariance failed: {mismatches}")
    return {"passed": True, "comparisons": mismatches, "rows": int(len(reference))}


def ensemble_rank_pct(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = None
    rank_columns = []
    for index, frame in enumerate(frames):
        column = f"seed_{index}_rank_pct"
        part = frame[["signal_date", "instrument_id", "score"]].copy()
        part["signal_date"] = pd.to_datetime(part["signal_date"])
        part[column] = part.groupby("signal_date")["score"].rank(method="average", pct=True)
        part = part[["signal_date", "instrument_id", column]]
        merged = part if merged is None else merged.merge(
            part, on=["signal_date", "instrument_id"], how="inner", validate="one_to_one"
        )
        rank_columns.append(column)
    if merged is None:
        raise ValueError("no seed predictions")
    merged["score"] = merged[rank_columns].mean(axis=1)
    return merged[["signal_date", "instrument_id", "score"]].sort_values(
        ["signal_date", "score", "instrument_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def train_head(head: HeadContract, *, resume: bool = True) -> dict:
    ensure_stock_src_on_path()
    from first_board_hermes.baseline import RankerConfig, rolling_train_predict

    root = _head_root(head)
    ensemble_path = root / "predictions.parquet"
    report_path = root / "ensemble_report.json"
    if resume and ensemble_path.is_file() and report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))

    training, candidates = _load_panel(head)
    first_year = max(2018, int(candidates["trade_date"].dt.year.min()))
    last_year = int(SIGNAL_END.year)
    allowed = {item.name for item in fields(RankerConfig)}
    frames = []
    seed_reports = []
    for seed in SEEDS:
        seed_dir = root / f"seed{seed}"
        seed_path = seed_dir / "predictions.parquet"
        if resume and seed_path.is_file():
            prediction = pd.read_parquet(seed_path)
            metadata = json.loads((seed_dir / "training_report.json").read_text(encoding="utf-8"))
        else:
            params = model_parameters(seed)
            params["eval_metric"] = head.eval_metric
            params["rank_n_bins"] = head.rank_n_bins
            config = RankerConfig(**{key: value for key, value in params.items() if key in allowed})
            prediction, metadata = rolling_train_predict(
                training,
                candidates,
                seed_dir,
                features=head.features,
                first_prediction_year=first_year,
                last_prediction_year=last_year,
                config=config,
                target=head.target,
                label_end_column=head.label_end_column,
                save_models=True,
            )
        prediction["signal_date"] = pd.to_datetime(prediction["signal_date"])
        prediction = prediction.loc[
            prediction["signal_date"].dt.date.between(PREDICTION_START, SIGNAL_END)
        ].copy()
        frames.append(prediction)
        seed_reports.append({
            "seed": seed,
            "rows": int(len(prediction)),
            "prediction_sha256": sha256_file(seed_path),
            "folds": metadata.get("folds", []),
        })

    invariant = assert_seed_rank_invariance(frames)
    ensemble = ensemble_rank_pct(frames)
    ensemble.to_parquet(ensemble_path, index=False)
    manifest_path = head.panel_dir / "manifest.json"
    payload = {
        "status": "trained_oos",
        "head": head.to_dict(),
        "walk_forward": "exact prior 3 calendar years train, next year predict",
        "model": {**model_parameters(SEEDS[0]), "eval_metric": head.eval_metric},
        "seeds": list(SEEDS),
        "seed_rank_invariance": invariant,
        "rows": int(len(ensemble)),
        "days": int(ensemble["signal_date"].nunique()),
        "panel_manifest_sha256": sha256_file(manifest_path),
        "prediction_sha256": sha256_file(ensemble_path),
        "feature_sha256": hashlib.sha256(json.dumps(head.features).encode()).hexdigest(),
        "seed_reports": seed_reports,
        "written_at": datetime.now().astimezone().isoformat(),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return payload


def train_all(*, resume: bool = True) -> dict:
    return {
        "five": train_head(FIVE_HEAD, resume=resume),
        "ten": train_head(TEN_HEAD, resume=resume),
    }
