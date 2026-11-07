"""把 fs01122 集成分 / 五种子注入旧 overlay 缓存，使 C03 按新分 rolling250。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import FS01122_SEEDS, PREDICTION_START, SIGNAL_END
from ..round2.data import seed_top1_agreement
from ..stock_readonly import (
    load_fs01122_all_seed_predictions,
    load_fs01122_ensemble_predictions,
)

_STATE: dict[str, Any] = {}


def load_locked_predictions() -> pd.DataFrame:
    if "ens" not in _STATE:
        _STATE["ens"] = load_fs01122_ensemble_predictions(
            start=PREDICTION_START, signal_end=SIGNAL_END
        )
    return _STATE["ens"]


def load_locked_seeds() -> pd.DataFrame:
    if "seeds" not in _STATE:
        _STATE["seeds"] = load_fs01122_all_seed_predictions(prefilter=True)
    return _STATE["seeds"]


def agreement_frame() -> pd.DataFrame:
    if "agree" not in _STATE:
        _STATE["agree"] = seed_top1_agreement(load_locked_predictions(), load_locked_seeds())
    return _STATE["agree"]


def inject_fs01122_into_overlay_caches() -> dict[str, Any]:
    """C03 / C02 走 fs01122 分，不再读 fs00263 种子。"""
    ens = load_locked_predictions()
    seeds = load_locked_seeds()
    agree = agreement_frame()
    seed_key = ",".join(str(s) for s in FS01122_SEEDS)

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

    from ..round2 import data as r2_data

    r2_data.load_all_seed_predictions = _seeds  # type: ignore[assignment]
    return {
        "ensemble_rows": int(len(ens)),
        "seed_rows": int(len(seeds)),
        "agree_days": int(len(agree)),
        "c03_source": "fs01122_top1_top2_rolling250",
    }
