"""只读加载 Stock src/runs：挂 src、读 fs00263 集成预测 / 候选面板 / Hermes 快照。

禁止写入 Stock src、runs 及其他课题。本课题目录（含 result/）可写。
Baseline 账本只拷到本课题 result/round1/。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import (
    BASELINE_REPORT_RELATIVE,
    CANDIDATE_PANEL_RELATIVE,
    ENSEMBLE_RELATIVE,
    FS01122_PANEL_RELATIVE,
    FS01122_PREDICTIONS_RELATIVE,
    FS01122_SEED_PRED_RELATIVE,
    FS01122_SEEDS,
    FS01122_STOCK_REPORT_RELATIVE,
    MARKET_END,
    PREDICTION_START,
    SIGNAL_END,
    SNAPSHOT_RELATIVE,
    STOCK_ROOT,
    STOCK_SRC,
    assert_project_result_path,
    round1_dir,
    stock_path,
)

_STOCK_ON_PATH = False


def ensure_stock_src_on_path() -> Path:
    """Insert Stock/src onto sys.path so ``import first_board_hermes`` works."""
    global _STOCK_ON_PATH
    src = STOCK_SRC.resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Stock src not found (read-only): {src}")
    text = str(src)
    if text not in sys.path:
        sys.path.insert(0, text)
    _STOCK_ON_PATH = True
    return src


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensemble_dir() -> Path:
    path = stock_path(ENSEMBLE_RELATIVE)
    if not path.is_dir():
        raise FileNotFoundError(f"missing fs00263 ensemble dir (read-only): {path}")
    return path


def stock_baseline_report_path() -> Path:
    path = stock_path(BASELINE_REPORT_RELATIVE)
    if not path.is_file():
        raise FileNotFoundError(f"missing baseline report.json (read-only): {path}")
    return path


def load_fs00263_ensemble_predictions(
    *,
    start=PREDICTION_START,
    signal_end=SIGNAL_END,
) -> pd.DataFrame:
    """Load n50_d2 five-seed ensemble predictions for fs00263 (read-only)."""
    path = ensemble_dir() / "predictions.parquet"
    frame = pd.read_parquet(path)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    if "instrument_id" in frame:
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    filtered = frame.loc[
        frame["signal_date"].dt.date.between(start, signal_end)
    ].copy()
    return filtered


def load_candidate_panel(columns: list[str] | None = None) -> pd.DataFrame:
    path = stock_path(CANDIDATE_PANEL_RELATIVE)
    if not path.is_file():
        raise FileNotFoundError(f"missing candidate panel (read-only): {path}")
    return pd.read_parquet(path, columns=columns)


def load_stock_baseline_report() -> dict[str, Any]:
    path = stock_baseline_report_path()
    return json.loads(path.read_text(encoding="utf-8"))


def copy_baseline_report_to_project() -> Path:
    """Copy Stock report.json into 课题 result/round1/R1_00_baseline/. Never writes back."""
    source = stock_baseline_report_path()
    dest_dir = round1_dir("R1_00_baseline")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "report.json"
    assert_project_result_path(dest)
    shutil.copy2(source, dest)
    nav_source = source.with_name("nav.parquet")
    if nav_source.is_file():
        nav_dest = dest_dir / "nav.parquet"
        assert_project_result_path(nav_dest)
        shutil.copy2(nav_source, nav_dest)
    provenance = {
        "status": "copied_from_stock_readonly",
        "source": str(source),
        "source_resolved": str(source.resolve()),
        "stock_root": str(STOCK_ROOT),
        "stock_root_resolved": str(STOCK_ROOT.resolve()),
        "source_sha256": sha256_file(source),
        "note": "课题侧副本；未修改 Stock /data1/stock-mxy",
        "trial_id": "fs00263",
        "capacity": "n50_d2",
        "ensemble": "ensemble_rank_pct_5seed",
        "variant": "top1_both_gates_nav30pct_v1",
        "exit_mode": "close_from_d2",
    }
    provenance_path = dest_dir / "provenance.json"
    assert_project_result_path(provenance_path)
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return dest


def load_fs01122_ensemble_predictions(
    *,
    start=PREDICTION_START,
    signal_end=SIGNAL_END,
) -> pd.DataFrame:
    """Load fs01122 prefilter-gap-floor ensemble (read-only Stock)."""
    path = stock_path(FS01122_PREDICTIONS_RELATIVE)
    if not path.is_file():
        raise FileNotFoundError(f"missing fs01122 predictions (read-only): {path}")
    frame = pd.read_parquet(path)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    if "instrument_id" in frame:
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    return frame.loc[frame["signal_date"].dt.date.between(start, signal_end)].copy()


def load_fs01122_seed_predictions(
    *,
    seed: int,
    start=PREDICTION_START,
    signal_end=SIGNAL_END,
    prefilter: bool = True,
) -> pd.DataFrame:
    rel = FS01122_SEED_PRED_RELATIVE / f"seed{int(seed)}"
    if prefilter:
        path = stock_path(rel / "prefilter_gap_floor" / "predictions.parquet")
    else:
        path = stock_path(rel / "predictions.parquet")
    if prefilter and not path.is_file():
        path = stock_path(rel / "predictions.parquet")
    if not path.is_file():
        raise FileNotFoundError(f"missing fs01122 seed predictions (read-only): {path}")
    frame = pd.read_parquet(path, columns=["signal_date", "instrument_id", "score"])
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["seed"] = int(seed)
    return frame.loc[frame["signal_date"].dt.date.between(start, signal_end)].copy()


def load_fs01122_all_seed_predictions(*, prefilter: bool = False) -> pd.DataFrame:
    """Five n100_d2 seeds. C02 用未预滤种子分；C03 用集成分。"""
    return pd.concat(
        [load_fs01122_seed_predictions(seed=seed, prefilter=prefilter) for seed in FS01122_SEEDS],
        ignore_index=True,
    )


def fs01122_stock_report_path() -> Path:
    path = stock_path(FS01122_STOCK_REPORT_RELATIVE)
    if not path.is_file():
        raise FileNotFoundError(f"missing fs01122 rank25_15 report (read-only): {path}")
    return path


def load_fs01122_training_panel(columns: list[str] | None = None) -> pd.DataFrame:
    path = stock_path(FS01122_PANEL_RELATIVE / "training_panel.parquet")
    if not path.is_file():
        raise FileNotFoundError(f"missing fs01122 14-factor panel (read-only): {path}")
    frame = pd.read_parquet(path, columns=columns)
    if "trade_date" in frame.columns:
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    if "instrument_id" in frame.columns:
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    return frame


def load_prepared_market(
    *,
    warmup_ma5: bool = True,
    ma_window: int = 5,
    verify_hashes: bool = False,
):
    """Load Hermes daily_raw and prepare fixed-sleeve lookups (read-only snapshot)."""
    ensure_stock_src_on_path()
    from first_board_hermes.fixed_sleeve_backtest import prepare_fixed_sleeve_market
    from zyquant.data import DataSnapshot

    snapshot_path = stock_path(SNAPSHOT_RELATIVE)
    snapshot = DataSnapshot(snapshot_path, verify_hashes=verify_hashes)
    market_start = PREDICTION_START
    if warmup_ma5:
        market_start = PREDICTION_START - timedelta(days=max(30, ma_window * 5))
    bars = snapshot.table(
        "daily_raw",
        start=market_start,
        end=MARKET_END,
        cutoff=snapshot.metadata.as_of_date,
        fields=[
            "trade_date", "instrument_id", "open", "high", "low", "close",
            "pre_close", "volume", "paused", "limit_up", "limit_down",
        ],
    )
    prepared = prepare_fixed_sleeve_market(bars, ma_window=ma_window)
    return {
        "prepared": prepared,
        "snapshot_id": snapshot.metadata.dataset_id,
        "snapshot_fingerprint": snapshot.metadata.fingerprint,
        "snapshot_path": str(snapshot_path),
        "market_start": market_start,
        "market_end": MARKET_END,
    }
