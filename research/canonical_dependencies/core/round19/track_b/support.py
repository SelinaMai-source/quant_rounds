"""Track B 共用 IO / 键对齐。只写 result/round19。不改 result/round18 原件。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...paths import ROUND7_ROOT, ROUND18_ROOT, ROUND19_ROOT, assert_project_result_path, ensure_round19_tree
from ...round7.contract import FIVE_FEATURES, FIVE_HEAD, TEN_FEATURES, TEN_HEAD
from ...round17.train import complete_cases
from ...round18.contract import F1_NAME, F2_NAME, F3_NAME, NEIGHBORS
from ..common import dump_json, file_sha256, load_json, write_md
from ..contract import B00, B01, exp_dir

KEYS = ["trade_date", "instrument_id"]
PRED_KEYS = ["signal_date", "instrument_id"]
TB = ROUND19_ROOT / "track_b"
R18_DATA = ROUND18_ROOT / "data"
R7_FIVE_PRED = ROUND7_ROOT / "models" / "five" / "predictions.parquet"
R7_TEN_PRED = ROUND7_ROOT / "models" / "ten" / "predictions.parquet"
R7_FIVE_REPORT = ROUND7_ROOT / "models" / "five" / "ensemble_report.json"
R7_TEN_REPORT = ROUND7_ROOT / "models" / "ten" / "ensemble_report.json"
R18_A01_FIVE_REPORT = ROUND18_ROOT / "candidates" / "R18_A01" / "five" / "ensemble_report.json"
R18_A01_FIVE_SEED17 = ROUND18_ROOT / "candidates" / "R18_A01" / "five" / "seed17" / "training_report.json"
R7_FIVE_SEED17 = ROUND7_ROOT / "models" / "five" / "seed17" / "training_report.json"
R7_TEN_SEED17 = ROUND7_ROOT / "models" / "ten" / "seed17" / "training_report.json"

JOIN_AUDIT_CSV = TB / "audit" / "join_audit.csv"
PARITY_JSON = TB / "parity" / "six_layer.json"
VERDICT_JSON = TB / "bug_verdict.json"


def tb_dir(*parts: str) -> Path:
    ensure_round19_tree()
    path = TB.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(path)
    return path


def cand_dir(code: str) -> Path:
    dest = exp_dir(code)
    dest.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest)
    return dest


def norm_keys(frame: pd.DataFrame, *, date_col: str = "trade_date") -> pd.DataFrame:
    out = frame.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    if "instrument_id" in out.columns:
        out["instrument_id"] = out["instrument_id"].astype(str)
    return out


def pred_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "signal_date" not in out.columns and "trade_date" in out.columns:
        out = out.rename(columns={"trade_date": "signal_date"})
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    out["instrument_id"] = out["instrument_id"].astype(str)
    return out


def load_r7_preds() -> dict[str, pd.DataFrame]:
    five = pred_keys(pd.read_parquet(R7_FIVE_PRED))
    ten = pred_keys(pd.read_parquet(R7_TEN_PRED))
    return {"five": five, "ten": ten}


def load_r18_panels() -> dict[str, pd.DataFrame]:
    paths = {
        "five_training": R18_DATA / "five_training.parquet",
        "five_candidate": R18_DATA / "five_candidate.parquet",
        "ten_training": R18_DATA / "ten_training.parquet",
        "ten_candidate": R18_DATA / "ten_candidate.parquet",
        "five_frozen": R18_DATA / "five_frozen.parquet",
        "ten_frozen": R18_DATA / "ten_frozen.parquet",
    }
    missing = [k for k, p in paths.items() if not p.is_file()]
    if missing:
        raise RuntimeError(f"R18 panels missing: {missing}; refuse to invent")
    out = {k: pd.read_parquet(p) for k, p in paths.items()}
    for key in ("five_training", "five_candidate", "ten_training", "ten_candidate"):
        out[key] = norm_keys(out[key])
    for key in ("five_frozen", "ten_frozen"):
        out[key] = pred_keys(out[key])
    return out


def load_original_panels() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, head in (("five", FIVE_HEAD), ("ten", TEN_HEAD)):
        train = pd.read_parquet(head.panel_dir / "training_panel.parquet")
        cand = pd.read_parquet(head.panel_dir / "candidate_panel.parquet")
        out[f"{name}_training"] = norm_keys(train)
        out[f"{name}_candidate"] = norm_keys(cand)
    return out


def load_b01_preds() -> dict[str, pd.DataFrame]:
    dest = cand_dir(B01)
    five_p = dest / "five" / "predictions.parquet"
    ten_p = dest / "ten" / "predictions.parquet"
    if not five_p.is_file() or not ten_p.is_file():
        raise RuntimeError("R19_B01 predictions missing; run b-fresh-retrain first")
    return {"five": pred_keys(pd.read_parquet(five_p)), "ten": pred_keys(pd.read_parquet(ten_p))}


def overlay_path(code: str) -> Path:
    return ROUND19_ROOT / "weights" / f"overlayed_{code}.parquet"


def load_overlay(code: str) -> pd.DataFrame:
    path = overlay_path(code)
    if path.is_file():
        return pred_keys(pd.read_parquet(path))
    cand = cand_dir(code) / "overlayed.parquet"
    if cand.is_file():
        return pred_keys(pd.read_parquet(cand))
    pred = cand_dir(code) / "predictions.parquet"
    if pred.is_file():
        return pred_keys(pd.read_parquet(pred))
    raise RuntimeError(f"overlay missing for {code}")


def add_rank(frame: pd.DataFrame, *, score="score") -> pd.DataFrame:
    work = pred_keys(frame)
    work["_rank"] = work.groupby("signal_date", sort=False)[score].rank(method="first", ascending=False)
    return work


def topk(frame: pd.DataFrame, k: int, *, score="score") -> pd.DataFrame:
    work = add_rank(frame, score=score)
    return work.loc[work["_rank"] <= k].copy()


def complete_audit(frame: pd.DataFrame, features: list[str], *, target: str, require_target: bool) -> dict[str, Any]:
    _, audit = complete_cases(frame, features, require_target=require_target, target=target)
    return audit


def year_of(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.year.astype(int)


def finite_corr(a: pd.Series, b: pd.Series, *, method: str = "spearman") -> float | None:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 8:
        return None
    if x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return None
    val = x[mask].corr(y[mask], method=method)
    if val is None or not np.isfinite(val):
        return None
    return float(val)


def status_of(ok: bool | None, *, identifiable: bool = True) -> str:
    if not identifiable or ok is None:
        return "NOT_IDENTIFIABLE"
    return "PASS" if ok else "FAIL"


__all__ = [
    "B00",
    "B01",
    "F1_NAME",
    "F2_NAME",
    "F3_NAME",
    "FIVE_FEATURES",
    "FIVE_HEAD",
    "JOIN_AUDIT_CSV",
    "KEYS",
    "NEIGHBORS",
    "PARITY_JSON",
    "PRED_KEYS",
    "R18_A01_FIVE_REPORT",
    "R18_A01_FIVE_SEED17",
    "R18_DATA",
    "R7_FIVE_PRED",
    "R7_FIVE_REPORT",
    "R7_FIVE_SEED17",
    "R7_TEN_PRED",
    "R7_TEN_REPORT",
    "R7_TEN_SEED17",
    "TB",
    "TEN_FEATURES",
    "TEN_HEAD",
    "VERDICT_JSON",
    "add_rank",
    "cand_dir",
    "complete_audit",
    "dump_json",
    "file_sha256",
    "finite_corr",
    "load_b01_preds",
    "load_json",
    "load_original_panels",
    "load_overlay",
    "load_r18_panels",
    "load_r7_preds",
    "norm_keys",
    "overlay_path",
    "pred_keys",
    "status_of",
    "tb_dir",
    "topk",
    "write_md",
    "year_of",
]
