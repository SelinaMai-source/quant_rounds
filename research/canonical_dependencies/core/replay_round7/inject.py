"""把 Round7 双头 90/10 当日分注入 overlay 缓存。C03 禁止复用 fs00263。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import PREDICTION_START, SIGNAL_END, STOCK_ROUND7_ROOT
from ..round2.data import seed_top1_agreement
from .contract import BLEND_PRED, GAP_FLOOR, RISK_PRED, SEEDS, TEN_WEIGHT

_STATE: dict[str, Any] = {}
KEYS = ["signal_date", "instrument_id"]


def _clip_signal(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    out["instrument_id"] = out["instrument_id"].astype(str)
    return out.loc[out["signal_date"].dt.date.between(PREDICTION_START, SIGNAL_END)].copy()


def load_locked_alpha() -> pd.DataFrame:
    if "alpha" not in _STATE:
        if not BLEND_PRED.is_file():
            raise FileNotFoundError(f"missing Round7 blend predictions (read-only): {BLEND_PRED}")
        _STATE["alpha"] = _clip_signal(pd.read_parquet(BLEND_PRED, columns=[*KEYS, "score"]))
    return _STATE["alpha"]


def load_locked_risk() -> pd.DataFrame:
    if "risk" not in _STATE:
        if not RISK_PRED.is_file():
            raise FileNotFoundError(f"missing Round7 clean_risk predictions (read-only): {RISK_PRED}")
        cols = [*KEYS, "score"]
        raw = pd.read_parquet(RISK_PRED)
        keep = [c for c in (*cols, "allocation_multiplier") if c in raw.columns]
        _STATE["risk"] = _clip_signal(raw[keep])
    return _STATE["risk"]


def _load_head_seed(head: str, seed: int) -> pd.DataFrame:
    path = STOCK_ROUND7_ROOT / "models" / head / f"seed{seed}" / "predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing Round7 {head} seed{seed}: {path}")
    frame = pd.read_parquet(path, columns=[*KEYS, "score"])
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    return frame.rename(columns={"score": f"score_{head}"})


def _fuse_seed(five: pd.DataFrame, ten: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    work = five.merge(ten, on=KEYS, how="inner", validate="one_to_one")
    work = work.merge(gaps, on=KEYS, how="left", validate="one_to_one")
    work["d1_open_gap"] = pd.to_numeric(work["d1_open_gap"], errors="coerce")
    work = work.loc[np.isfinite(work["d1_open_gap"]) & work["d1_open_gap"].ge(GAP_FLOOR)].copy()
    work["pct_five"] = work.groupby("signal_date")["score_five"].rank(method="average", pct=True)
    work["pct_ten"] = work.groupby("signal_date")["score_ten"].rank(method="average", pct=True)
    work["score"] = TEN_WEIGHT * work["pct_ten"] + (1.0 - TEN_WEIGHT) * work["pct_five"]
    return work[[*KEYS, "score"]]


def load_locked_seeds() -> pd.DataFrame:
    if "seeds" not in _STATE:
        from .contract import TEN_PANEL

        gaps = pd.read_parquet(
            TEN_PANEL / "candidate_panel.parquet",
            columns=["trade_date", "instrument_id", "d1_open_gap"],
        ).rename(columns={"trade_date": "signal_date"})
        gaps["signal_date"] = pd.to_datetime(gaps["signal_date"])
        gaps["instrument_id"] = gaps["instrument_id"].astype(str)
        parts = []
        for seed in SEEDS:
            fused = _fuse_seed(_load_head_seed("five", seed), _load_head_seed("ten", seed), gaps)
            fused["seed"] = int(seed)
            parts.append(_clip_signal(fused))
        _STATE["seeds"] = pd.concat(parts, ignore_index=True)
    return _STATE["seeds"]


def agreement_frame() -> pd.DataFrame:
    if "agree" not in _STATE:
        _STATE["agree"] = seed_top1_agreement(load_locked_alpha(), load_locked_seeds())
    return _STATE["agree"]


def inject_round7_into_overlay_caches() -> dict[str, Any]:
    ens = load_locked_alpha()
    seeds = load_locked_seeds()
    agree = agreement_frame()
    seed_key = ",".join(str(s) for s in SEEDS)

    from ..round2 import data as r2_data
    from ..round2 import runner as r2_runner
    from ..round3 import overlays as r3_overlays
    from ..round4 import overlays as r4_overlays

    r2_runner._CACHE["agree"] = agree
    r3_overlays._CACHE[f"agree:{seed_key}"] = agree
    r3_overlays._CACHE["agree"] = agree
    r4_overlays._CACHE.clear()
    r4_overlays._CACHE[f"agree:{seed_key}"] = agree

    def _ens(*_a, **_k):
        return ens

    def _seeds(*_a, **_k):
        return seeds

    r3_overlays.load_fs00263_ensemble_predictions = _ens  # type: ignore[attr-defined]
    r3_overlays.load_all_seed_predictions = _seeds  # type: ignore[attr-defined]
    r2_runner.load_fs00263_ensemble_predictions = _ens  # type: ignore[attr-defined]
    r2_runner.load_all_seed_predictions = _seeds  # type: ignore[attr-defined]
    r2_data.load_all_seed_predictions = _seeds  # type: ignore[assignment]
    return {
        "ensemble_rows": int(len(ens)),
        "seed_rows": int(len(seeds)),
        "agree_days": int(len(agree)),
        "c03_source": "round7_dual90_10_current_scores_rolling250",
    }
