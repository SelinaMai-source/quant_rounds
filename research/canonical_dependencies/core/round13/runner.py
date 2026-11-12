"""Round13 回测入口：复用 Round12 runner。产物只写 result/round13/。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..replay_round7.inject import inject_round7_into_overlay_caches, load_locked_alpha
from ..round11.contract import BASELINES
from ..round12.runner import hist_hooks, run_r12_backtest
from .contract import RANK_NAV, TOP_K

_CACHE: dict[str, Any] = {}


def load_market() -> dict[str, Any]:
    if "market" not in _CACHE:
        from ..stock_readonly import load_prepared_market

        inject_round7_into_overlay_caches()
        _CACHE["market"] = load_prepared_market(warmup_ma5=True)
    return _CACHE["market"]


def load_alpha() -> pd.DataFrame:
    if "alpha" not in _CACHE:
        inject_round7_into_overlay_caches()
        _CACHE["alpha"] = load_locked_alpha()
    return _CACHE["alpha"]


def build_version_overlayed(version: str) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    key = f"ov_{version}"
    if key in _CACHE:
        return _CACHE[key]
    pred = load_alpha()
    spec = BASELINES[version]
    kind = (spec.overlay_spec or {}).get("kind")
    if kind == "exact_v2":
        from ..round8.overlays import build_overlayed
    else:
        from ..round9.overlays import build_overlayed
    overlayed, meta, cfg = build_overlayed(dict(spec.overlay_spec), pred)
    _CACHE[key] = (overlayed, meta, cfg)
    return overlayed, meta, cfg


def run_r13_backtest(
    dest,
    *,
    overlayed: pd.DataFrame,
    validation_hooks: dict[str, Any] | None = None,
    initial_cash: float | None = None,
    extra_report: dict[str, Any] | None = None,
    resume: bool = True,
    write: bool = True,
    prepared_market=None,
    market_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loaded = None
    if prepared_market is None or market_meta is None:
        loaded = load_market()
        prepared_market = prepared_market if prepared_market is not None else loaded["prepared"]
        market_meta = market_meta if market_meta is not None else loaded
    extra = {
        "fee_schedule": "HISTORICAL_FEE_CORRECTED",
        "rank_nav": list(RANK_NAV),
        "top_k": TOP_K,
        "evidence_role": "round13_baseline_consolidation",
        "note": "Round13 产物只写 result/round13/。不覆盖 legacy。不自动升 CURRENT v5。",
    }
    if extra_report:
        extra.update(extra_report)
    return run_r12_backtest(
        dest,
        prepared_market=prepared_market,
        market_meta=market_meta,
        overlayed=overlayed,
        validation_hooks=validation_hooks if validation_hooks is not None else hist_hooks(),
        initial_cash=initial_cash,
        extra_report=extra,
        resume=resume,
        write=write,
    )


__all__ = [
    "build_version_overlayed",
    "hist_hooks",
    "load_alpha",
    "load_market",
    "run_r13_backtest",
]
