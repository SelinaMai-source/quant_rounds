"""OOF 与标签成熟。不用 next2_date 当成熟日。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from core.round7.contract import FIVE_FEATURES, GAP_FLOOR

KEYS = ["signal_date", "instrument_id"]


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_components() -> pd.DataFrame:
    """与 v5 fuse_heads 同一股票池，并左连接已有 LGBM OOF。不因缺 LGBM 删股票。"""
    from core.round17.backtest import load_gaps
    from core.round23_3.contract import E0
    from core.paths import RESULT_ROOT

    five = pd.read_parquet(E0 / "five_pred.parquet").rename(columns={"score": "five"})
    ten = pd.read_parquet(E0 / "ten_pred.parquet").rename(columns={"score": "ten"})
    for frame in (five, ten):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    work = five.merge(ten, on=KEYS, how="inner")
    work = work.merge(load_gaps(), on=KEYS, how="left")
    work["d1_open_gap"] = pd.to_numeric(work["d1_open_gap"], errors="coerce")
    work = work.loc[np.isfinite(work["d1_open_gap"]) & work["d1_open_gap"].ge(GAP_FLOOR)].copy()
    work["pct_five"] = work.groupby("signal_date")["five"].rank(method="average", pct=True)
    work["pct_ten"] = work.groupby("signal_date")["ten"].rank(method="average", pct=True)
    work["v5"] = 0.90 * work["pct_ten"] + 0.10 * work["pct_five"]
    root = RESULT_ROOT / "round23_3" / "oof"
    lgbm_five = pd.read_parquet(root / "lgbm_five_trunc4" / "predictions.parquet").rename(columns={"score": "lgbm_five"})
    lgbm_ten = pd.read_parquet(root / "lgbm_ten_trunc5" / "predictions.parquet").rename(columns={"score": "lgbm_ten"})
    for frame in (lgbm_five, lgbm_ten):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    before = len(work)
    work = work.merge(lgbm_five[KEYS + ["lgbm_five"]], on=KEYS, how="left")
    work = work.merge(lgbm_ten[KEYS + ["lgbm_ten"]], on=KEYS, how="left")
    if len(work) != before:
        raise RuntimeError("LGBM merge changed the v5 pool")
    return work


def mature_mask(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """label_available_at <= cutoff。没有 actual_exit_date 的行不能进入训练。"""
    if "next2_date" in frame.columns and "actual_exit_date" not in frame.columns:
        raise RuntimeError("refusing to treat next2_date as maturity")
    mature = pd.to_datetime(frame["actual_exit_date"])
    return mature.notna() & mature.le(pd.Timestamp(cutoff))


def prediction_pool(candidates: pd.DataFrame) -> pd.DataFrame:
    """预测池不因为未来标签缺失而缩小。"""
    return candidates.copy()


def registry_rows() -> list[dict]:
    from core.round23_3.contract import E0
    from core.paths import RESULT_ROOT

    rows = []
    specs = [
        (E0 / "five_pred.parquet", "five", "XGBRanker_v5"),
        (E0 / "ten_pred.parquet", "ten", "XGBRanker_v5"),
        (RESULT_ROOT / "round23_3/oof/lgbm_five_trunc4/predictions.parquet", "five", "LGBMRanker_oof"),
        (RESULT_ROOT / "round23_3/oof/lgbm_ten_trunc5/predictions.parquet", "ten", "LGBMRanker_oof"),
    ]
    for path, head, model in specs:
        frame = pd.read_parquet(path, columns=["signal_date"])
        rows.append({
            "path": str(path),
            "head": head,
            "model": model,
            "sha256": file_sha(path),
            "rows": int(len(frame)),
            "min_signal_date": str(pd.to_datetime(frame["signal_date"]).min().date()),
            "max_signal_date": str(pd.to_datetime(frame["signal_date"]).max().date()),
            "label_maturity_column": "actual_exit_date",
            "forbidden_maturity_proxy": "next2_date",
            "feature_columns_allowed_for_new_rf": ",".join(FIVE_FEATURES),
            "information_lane": "LEGACY_OPEN_INFORMATION_COMPARISON",
        })
    return rows
