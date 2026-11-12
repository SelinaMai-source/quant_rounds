"""Finalists 稳健性：成本×1.5/×2、入场+10/20/30bp、n100_d2、leave-one-seed、20 日块 bootstrap、winner 依赖。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND4_ROOT, ROUND6_ROOT, assert_project_result_path, ensure_round6_tree
from .registry import TRAIN_SEEDS, clone_spec
from .runner import run_many
from .train import train_spec


def neighbor_cost_specs(bases: list[str]) -> list:
    from .registry import SPECS

    out = []
    for code in bases[:5]:
        spec = SPECS.get(code)
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


def seed_leave_specs(bases: list[str]) -> list:
    from .registry import SPECS

    out = []
    for code in bases[:3]:
        spec = SPECS.get(code)
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
                reuse_round5_key=None,
                stage="robust",
                tags=("robust", "seed"),
                diagnostic=True,
            ))
    return out


def n100_specs(bases: list[str]) -> list:
    from .registry import SPECS

    out = []
    for code in bases[:3]:
        spec = SPECS.get(code)
        if spec is None or spec.alpha_source == "locked_fs00263":
            continue
        key = f"{spec.prediction_key}_n100d2"
        out.append(clone_spec(
            spec,
            code=f"{code}_n100",
            title_zh=f"{spec.title_zh} n100_d2",
            n_estimators=100,
            prediction_key=key,
            reuse_round5_key=None,
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


def winner_dependence(trades: pd.DataFrame) -> dict[str, Any]:
    """Top1/5/10% winner 对累计收益的依赖。DIAGNOSTIC。"""
    t = trades.copy()
    ret = None
    for c in ("net_return", "trade_return", "return"):
        if c in t.columns:
            ret = pd.to_numeric(t[c], errors="coerce")
            break
    if ret is None:
        for c in t.columns:
            if "return" in c and "year" not in c:
                ret = pd.to_numeric(t[c], errors="coerce")
                break
    if ret is None:
        return {"status": "no_return_col", "tag": "DIAGNOSTIC"}
    ret = ret.dropna()
    if ret.empty:
        return {"status": "empty", "tag": "DIAGNOSTIC"}
    total = float(ret.sum())
    ordered = ret.sort_values(ascending=False)
    n = len(ordered)
    out = {"tag": "DIAGNOSTIC", "n": int(n), "sum_ret": total}
    for pct in (0.01, 0.05, 0.10):
        k = max(1, int(np.ceil(n * pct)))
        share = float(ordered.iloc[:k].sum() / total) if total else None
        out[f"top{int(pct*100)}pct_share"] = share
    return out


def write_robust(
    bases: list[str],
    *,
    predictions=None,
    prepared=None,
    meta=None,
    resume: bool = True,
    run_now: bool = True,
) -> dict[str, Any]:
    specs = n100_specs(bases) + seed_leave_specs(bases) + neighbor_cost_specs(bases)
    for spec in specs:
        if spec.alpha_source == "local_retrain" and spec.prediction_key:
            train_spec(spec, resume=resume)
    run_many(specs, predictions=predictions, prepared=prepared, meta=meta, resume=resume, run_now=run_now)
    boots = {}
    winners = {}
    for code in ["R6_00_baseline_v3", *bases]:
        nav_path = ROUND6_ROOT / code / "nav.parquet"
        tr_path = ROUND6_ROOT / code / "trades.parquet"
        if nav_path.is_file():
            boots[code] = block_bootstrap_nav(pd.read_parquet(nav_path))
        if tr_path.is_file():
            winners[code] = winner_dependence(pd.read_parquet(tr_path))
    v3_nav = ROUND4_ROOT / "R4_B05" / "nav.parquet"
    if v3_nav.is_file() and "R4_B05" not in boots:
        boots["R4_B05"] = block_bootstrap_nav(pd.read_parquet(v3_nav))
    payload = {
        "written_at": datetime.now().astimezone().isoformat(),
        "bases": bases,
        "ran": [s.code for s in specs],
        "bootstrap": boots,
        "winner_dependence": winners,
        "note": "20 日块 bootstrap 2000 = STATISTICAL RESAMPLING NOT OOS。winner 依赖 DIAGNOSTIC。",
    }
    dest = ensure_round6_tree() / "10_robust" / "robust.json"
    assert_project_result_path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return payload
