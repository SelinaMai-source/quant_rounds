"""Walk-forward 训练与 OOF RankIC。默认不跑账户 NAV；全引擎回测另开。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND20_ROOT, ROUND21_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, TEN_FEATURES, TEN_HEAD
from ..round20.metrics import icir_of, rank_ic
from ..round20.train import new_columns, train_both_heads, unique_keep
from .common import dump_json, load_json
from .construct import PANEL_PATH as R21_PANEL
from .contract import (
    A00,
    COLSAMPLE,
    FEE_SCHEDULE,
    SKIP_DEPENDENCY,
    SUBSAMPLE,
    XGB_N_JOBS,
    exp_dir,
)
from .engine import copy_a00_from_parity, run_v5_fused_backtest
from .panel import KEYS

DATA = ROUND21_ROOT / "data"
_G: dict[str, Any] = {}


def prepare_panels(*, resume: bool = True) -> dict[str, Any]:
    paths = {
        "five_training": DATA / "five_training.parquet",
        "five_candidate": DATA / "five_candidate.parquet",
        "ten_training": DATA / "ten_training.parquet",
        "ten_candidate": DATA / "ten_candidate.parquet",
        "five_frozen": DATA / "five_frozen.parquet",
        "ten_frozen": DATA / "ten_frozen.parquet",
    }
    if resume and all(p.is_file() for p in paths.values()):
        return {"status": "resumed", "paths": {k: str(v) for k, v in paths.items()}}
    if not R21_PANEL.is_file():
        raise RuntimeError(f"MISSING_ARTIFACT R21 factor_panel: {R21_PANEL}")
    r21 = pd.read_parquet(R21_PANEL)
    r21["trade_date"] = pd.to_datetime(r21["trade_date"])
    r21["instrument_id"] = r21["instrument_id"].astype(str)
    r20 = ROUND20_ROOT / "data"
    DATA.mkdir(parents=True, exist_ok=True)
    for key in ("five_training", "five_candidate", "ten_training", "ten_candidate"):
        src = r20 / f"{key}.parquet"
        if not src.is_file():
            raise RuntimeError(f"MISSING_ARTIFACT R20 panel {src}")
        frame = pd.read_parquet(src)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        extra = [c for c in r21.columns if c not in frame.columns and c not in {"is_first_board_hard"}]
        if extra:
            frame = frame.merge(r21[KEYS + extra], on=KEYS, how="left")
        frame.to_parquet(paths[key], index=False)
    pd.read_parquet(r20 / "five_frozen.parquet").to_parquet(paths["five_frozen"], index=False)
    pd.read_parquet(r20 / "ten_frozen.parquet").to_parquet(paths["ten_frozen"], index=False)
    meta = {
        "status": "prepared",
        "paths": {k: str(v) for k, v in paths.items()},
        "filled_with_zero": False,
        "subsample": SUBSAMPLE,
        "colsample": COLSAMPLE,
    }
    dump_json(DATA / "panels.json", meta)
    return meta


def init_panels() -> None:
    if _G:
        return
    prepare_panels(resume=True)
    for key in ("five_training", "five_candidate", "ten_training", "ten_candidate", "five_frozen", "ten_frozen"):
        path = DATA / f"{key}.parquet"
        frame = pd.read_parquet(path)
        if "trade_date" in frame.columns:
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        if "instrument_id" in frame.columns:
            frame["instrument_id"] = frame["instrument_id"].astype(str)
        if "signal_date" in frame.columns:
            frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        _G[key] = frame


def parse_extras(raw: Any) -> tuple[str, ...]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(unique_keep([str(x) for x in raw if str(x) and str(x) != "nan"]))
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return ()
    return tuple(unique_keep([p.strip() for p in text.replace(",", "|").split("|") if p.strip()]))


def skip_payload(code: str, spec: dict[str, Any], reason: str, status: str) -> dict[str, Any]:
    dest = exp_dir(code)
    dest.mkdir(parents=True, exist_ok=True)
    obs = {
        "code": code,
        "role": spec.get("role"),
        "status": status,
        "reason": reason,
        "formal": bool(spec.get("formal", True)),
        "model": spec.get("model") or "M0",
        "five_extras": list(spec.get("five_extras") or spec.get("extras") or ()),
        "ten_extras": list(spec.get("ten_extras") or spec.get("extras") or ()),
        "cagr": None,
        "sharpe": None,
        "max_drawdown": None,
        "completed_trades": None,
        "nav_end": None,
        "fee_schedule": FEE_SCHEDULE,
        "out_dir": str(dest),
        "looked_at_combo_nav": False,
    }
    dump_json(dest / "summary.json", obs)
    return obs


def oof_rankic(pred: pd.DataFrame, candidates: pd.DataFrame, *, target: str) -> dict[str, Any]:
    work = pred.copy()
    if "signal_date" in work.columns and "trade_date" not in work.columns:
        work["trade_date"] = pd.to_datetime(work["signal_date"])
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work["instrument_id"] = work["instrument_id"].astype(str)
    lab = candidates[KEYS + [target]].copy()
    lab["trade_date"] = pd.to_datetime(lab["trade_date"])
    merged = work.merge(lab, on=KEYS, how="left")
    ics: list[float] = []
    for _, day in merged.groupby("trade_date", sort=False):
        ic = rank_ic(day["score"], day[target])
        if ic is not None:
            ics.append(ic)
    arr = np.array(ics, dtype=float)
    return {
        "oof_rankic_mean": float(np.mean(arr)) if len(arr) else None,
        "oof_rankicir": icir_of(arr) if len(arr) else None,
        "n_oos_days": int(len(arr)),
    }


def _done_oof(dest) -> dict[str, Any] | None:
    path = dest / "oof.json"
    if not path.is_file():
        return None
    payload = load_json(path)
    if payload.get("status") in {"OOF", "SKIP", SKIP_DEPENDENCY} or payload.get("five_oof"):
        return payload
    return None


def train_oof(spec: dict[str, Any], *, resume: bool = True, n_jobs: int = XGB_N_JOBS) -> dict[str, Any]:
    """Walk-forward XGB，只写 OOF 预测 / RankIC，不跑账户 NAV。"""
    code = str(spec["code"])
    dest = exp_dir(code)
    dest.mkdir(parents=True, exist_ok=True)
    if resume:
        existing = _done_oof(dest)
        if existing is not None:
            existing["resumed"] = True
            return existing
    if spec.get("role") == "exact_v5_replay" or code == A00:
        copied = copy_a00_from_parity(dest)
        copied["status"] = "OOF_SKIP_EXACT_REPLAY"
        copied["looked_at_combo_nav"] = True
        dump_json(dest / "oof.json", copied)
        return copied
    extras = parse_extras(spec.get("extras"))
    five_extras = parse_extras(spec.get("five_extras") if spec.get("five_extras") is not None else extras)
    ten_extras = parse_extras(spec.get("ten_extras") if spec.get("ten_extras") is not None else extras)
    force_retrain = bool(spec.get("force_retrain"))
    five_new = new_columns(FIVE_FEATURES, five_extras)
    ten_new = new_columns(TEN_FEATURES, ten_extras)
    if not five_new and not ten_new and not force_retrain:
        return skip_payload(code, spec, "no new columns and not force_retrain", "SKIP_NO_NEW_FEATURES")
    init_panels()
    missing = []
    for name in unique_keep([*five_new, *ten_new]):
        if name in five_new and name not in _G["five_training"].columns:
            missing.append(f"five:{name}")
        if name in ten_new and name not in _G["ten_training"].columns:
            missing.append(f"ten:{name}")
    if missing:
        return skip_payload(code, spec, "missing columns: " + ",".join(missing), "SKIP_MISSING_COLUMN")
    try:
        trained = train_both_heads(
            five_training=_G["five_training"],
            five_candidates=_G["five_candidate"],
            ten_training=_G["ten_training"],
            ten_candidates=_G["ten_candidate"],
            five_extras=five_extras,
            ten_extras=ten_extras,
            output_dir=dest,
            resume=resume,
            five_frozen=_G["five_frozen"],
            ten_frozen=_G["ten_frozen"],
            force_retrain=force_retrain,
            n_jobs=n_jobs,
        )
    except Exception as exc:
        return skip_payload(code, spec, f"{type(exc).__name__}: {exc}", "SKIP_TRAIN_ERROR")
    five_pred = trained["five_pred"]
    ten_pred = trained["ten_pred"]
    five_pred.to_parquet(dest / "five_pred.parquet", index=False)
    ten_pred.to_parquet(dest / "ten_pred.parquet", index=False)
    five_oof = oof_rankic(five_pred, _G["five_candidate"], target=FIVE_HEAD.target)
    ten_oof = oof_rankic(ten_pred, _G["ten_candidate"], target=TEN_HEAD.target)
    payload = {
        "code": code,
        "role": spec.get("role"),
        "status": "OOF",
        "formal": bool(spec.get("formal", True)),
        "force_retrain": force_retrain,
        "five_extras": list(five_extras),
        "ten_extras": list(ten_extras),
        "five_new": trained.get("five_new"),
        "ten_new": trained.get("ten_new"),
        "five_oof": five_oof,
        "ten_oof": ten_oof,
        "subsample": SUBSAMPLE,
        "colsample": COLSAMPLE,
        "filled_with_zero": False,
        "looked_at_combo_nav": False,
        "out_dir": str(dest),
        "independent_oos": False,
    }
    dump_json(dest / "oof.json", payload)
    dump_json(dest / "summary.json", {**payload, "cagr": None, "sharpe": None, "status": "OOF_ONLY"})
    return payload


def run_full_backtest(code: str, spec: dict[str, Any] | None = None, *, resume: bool = True) -> dict[str, Any]:
    dest = exp_dir(code)
    if resume and (dest / "nav.parquet").is_file() and (dest / "summary.json").is_file():
        existing = load_json(dest / "summary.json")
        if existing.get("sharpe") is not None:
            existing["resumed"] = True
            return existing
    if code == A00:
        return copy_a00_from_parity(dest)
    five_p = dest / "five_pred.parquet"
    ten_p = dest / "ten_pred.parquet"
    if not five_p.is_file() or not ten_p.is_file():
        if spec is None:
            return skip_payload(code, {"code": code}, "missing OOF predictions", "SKIP_NO_OOF")
        train_oof(spec, resume=True)
    if not five_p.is_file() or not ten_p.is_file():
        return skip_payload(code, spec or {"code": code}, "OOF train failed", "SKIP_NO_OOF")
    five = pd.read_parquet(five_p)
    ten = pd.read_parquet(ten_p)
    extra = {
        "code": code,
        "role": (spec or {}).get("role"),
        "fee_schedule": FEE_SCHEDULE,
        "validation_hooks": None,
        "subsample": SUBSAMPLE,
        "colsample": COLSAMPLE,
    }
    obs = run_v5_fused_backtest(dest, five, ten, extra=extra, resume=resume)
    oof = load_json(dest / "oof.json")
    payload = {
        **obs,
        "code": code,
        "role": (spec or {}).get("role"),
        "five_oof": oof.get("five_oof"),
        "ten_oof": oof.get("ten_oof"),
        "five_extras": oof.get("five_extras") or (spec or {}).get("five_extras"),
        "ten_extras": oof.get("ten_extras") or (spec or {}).get("ten_extras"),
        "looked_at_combo_nav": True,
        "fee_schedule": FEE_SCHEDULE,
        "validation_hooks": None,
    }
    dump_json(dest / "summary.json", payload)
    return payload


__all__ = [
    "init_panels",
    "oof_rankic",
    "parse_extras",
    "prepare_panels",
    "run_full_backtest",
    "skip_payload",
    "train_oof",
]
