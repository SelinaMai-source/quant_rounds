"""Finalists 稳健性：n100_d2、leave-one-seed、成本×1.5/×2、入场+10/20/30bp、20 日块 bootstrap 2000。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND5_ROOT, assert_project_result_path, ensure_round5_tree
from ..round2.data import SEEDS
from .registry import R5Spec, TRAIN_SEEDS, V3_OVERLAY, clone_spec, get_spec
from .runner import load_report, run_spec, tracking_row
from .train import train_key


def neighbor_cost_specs(bases: list[str]) -> list[R5Spec]:
    out = []
    for code in bases[:3]:
        spec = get_spec(code) if code in _safe_specs() else None
        if spec is None:
            continue
        for mult in (1.5, 2.0):
            out.append(clone_spec(
                spec,
                code=f"{code}_FEEx{mult:g}",
                title_zh=f"{spec.title_zh} 费用×{mult:g}",
                cost_overrides={"fee_mult": mult},
                stage="robust",
                tags=("robust", "cost"),
                diagnostic=True,
            ))
        for bp in (10, 20, 30):
            out.append(clone_spec(
                spec,
                code=f"{code}_BUY{bp}bp",
                title_zh=f"{spec.title_zh} 入场+{bp}bp",
                cost_overrides={"buy_adverse_bps": bp},
                stage="robust",
                tags=("robust", "cost"),
                diagnostic=True,
            ))
    return out


def _safe_specs():
    from .registry import SPECS
    return SPECS


def seed_leave_specs(bases: list[str]) -> list[R5Spec]:
    out = []
    for code in bases[:3]:
        spec = _safe_specs().get(code)
        if spec is None or spec.alpha_source == "locked_fs00263":
            continue
        for leave in TRAIN_SEEDS:
            kept = tuple(s for s in TRAIN_SEEDS if s != leave)
            key = f"{spec.prediction_key}_loo{leave}"
            out.append(clone_spec(
                spec,
                code=f"{code}_LOO{leave}",
                title_zh=f"{spec.title_zh} 去种子{leave}",
                seeds=kept,
                seed_leave_out=leave,
                prediction_key=key,
                stage="robust",
                tags=("robust", "seed"),
                diagnostic=True,
            ))
    return out


def n100_specs(bases: list[str]) -> list[R5Spec]:
    out = []
    for code in bases[:3]:
        spec = _safe_specs().get(code)
        if spec is None or spec.alpha_source == "locked_fs00263":
            continue
        key = f"{spec.prediction_key}_n100d2"
        out.append(clone_spec(
            spec,
            code=f"{code}_n100",
            title_zh=f"{spec.title_zh} n100_d2",
            n_estimators=100,
            prediction_key=key,
            stage="robust",
            tags=("robust", "n100"),
            diagnostic=True,
        ))
    return out


def block_bootstrap_nav(nav: pd.DataFrame, *, n: int = 2000, block: int = 20, seed: int = 202) -> dict[str, Any]:
    """STATISTICAL RESAMPLING NOT OOS。"""
    ordered = nav.sort_values("trade_date").copy()
    ordered["trade_date"] = pd.to_datetime(ordered["trade_date"])
    daily = ordered.set_index("trade_date")["total_asset"].astype(float).pct_change().dropna().to_numpy()
    if len(daily) < block * 2:
        return {"status": "insufficient", "n": 0}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(daily) / block))
    cagrs = []
    sharpes = []
    years = max((ordered["trade_date"].iloc[-1] - ordered["trade_date"].iloc[0]).days / 365.25, 1 / 365.25)
    for _ in range(n):
        starts = rng.integers(0, max(1, len(daily) - block + 1), size=n_blocks)
        sample = np.concatenate([daily[s:s + block] for s in starts])[: len(daily)]
        wealth = float(np.prod(1.0 + sample))
        cagrs.append(wealth ** (1.0 / years) - 1.0)
        mu, sd = float(sample.mean()), float(sample.std(ddof=1))
        sharpes.append(mu / sd * np.sqrt(252) if sd > 0 else None)
    cagrs = np.asarray(cagrs, dtype=float)
    sh = np.asarray([x for x in sharpes if x is not None], dtype=float)
    return {
        "tag": "STATISTICAL RESAMPLING NOT OOS",
        "n": n,
        "block": block,
        "cagr_p05": float(np.quantile(cagrs, 0.05)),
        "cagr_p50": float(np.quantile(cagrs, 0.50)),
        "cagr_p95": float(np.quantile(cagrs, 0.95)),
        "sharpe_p05": float(np.quantile(sh, 0.05)) if len(sh) else None,
        "sharpe_p50": float(np.quantile(sh, 0.50)) if len(sh) else None,
        "sharpe_p95": float(np.quantile(sh, 0.95)) if len(sh) else None,
    }


def write_robust(
    bases: list[str],
    *,
    predictions=None,
    prepared=None,
    meta=None,
    resume: bool = True,
    run_now: bool = True,
) -> dict[str, Any]:
    from .runner import run_many
    from .train import train_spec

    specs = n100_specs(bases) + seed_leave_specs(bases) + neighbor_cost_specs(bases)
    # train extra keys first
    for spec in specs:
        if spec.alpha_source == "local_retrain" and spec.prediction_key:
            train_spec(spec, resume=resume)
    run_many(specs, predictions=predictions, prepared=prepared, meta=meta, resume=resume, run_now=run_now)
    boots = {}
    for code in ["R5_00_baseline_v3", *bases]:
        nav_path = ROUND5_ROOT / code / "nav.parquet"
        if not nav_path.is_file():
            continue
        boots[code] = block_bootstrap_nav(pd.read_parquet(nav_path))
    payload = {
        "written_at": datetime.now().astimezone().isoformat(),
        "bases": bases,
        "ran": [s.code for s in specs],
        "bootstrap": boots,
        "note": "20 日块 bootstrap 2000 = STATISTICAL RESAMPLING NOT OOS。",
    }
    dest = ensure_round5_tree() / "10_robust" / "robust.json"
    assert_project_result_path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return payload
