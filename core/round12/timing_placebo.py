"""State-timing placebo：打乱 v2/v3/v4 状态后全引擎。保年 / 候选数桶 / 市场风险桶 / 状态频率。"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND11_ROOT, ROUND12_ROOT
from ..round8.overlays import MIN_KEEP, apply_rank_multipliers
from ..round9.allocator import b03_nav, nav_to_mult, sep_bucket
from ..round9.overlays import v3_date_mult
from ..round11.common import daily_from_nav, max_drawdown_of, sharpe_of, worst_window_return
from .common import dump_json, load_json, write_md
from .placebo import _cagr_from_daily, _metrics_pack, _percentile
from .runner import hist_hooks, run_r12_backtest

TIMING_MD = ROUND12_ROOT / "12_timing_placebo.md"
TIMING_JSON = ROUND12_ROOT / "timing_placebo" / "timing.json"
N_TIMING = 500  # 全引擎太贵；契约允许 ≥500
SEED = 20261012
LAYERS = ("v2", "v3", "v4")

_G: dict[str, Any] = {}


def _load_predictions():
    from ..replay_round7.inject import inject_round7_into_overlay_caches, load_locked_alpha

    inject_round7_into_overlay_caches()
    return load_locked_alpha()


def _n_cand(predictions: pd.DataFrame) -> pd.Series:
    p = predictions.copy()
    p["signal_date"] = pd.to_datetime(p["signal_date"]).dt.normalize()
    return p.groupby("signal_date")["instrument_id"].nunique()


def _strata(idx: pd.DatetimeIndex, n_cand: pd.Series, market: pd.Series) -> pd.Series:
    year = pd.Series(idx.year, index=idx).astype(str)
    nc = pd.to_numeric(n_cand.reindex(idx), errors="coerce")
    mk = pd.to_numeric(market.reindex(idx), errors="coerce")
    try:
        n_b = pd.qcut(nc.rank(method="first"), 4, labels=["n1", "n2", "n3", "n4"])
    except ValueError:
        n_b = pd.Series("nX", index=idx)
    try:
        m_b = pd.qcut(mk.rank(method="first"), 3, labels=["m1", "m2", "m3"])
    except ValueError:
        m_b = pd.Series("mX", index=idx)
    return year + "|" + n_b.astype(str) + "|" + m_b.astype(str)


def _permute_labels(labels: pd.Series, strata: pd.Series, rng: np.random.Generator) -> pd.Series:
    out = labels.copy()
    for key, idx in strata.groupby(strata).groups.items():
        pos = list(idx)
        vals = labels.loc[pos].to_numpy()
        rng.shuffle(vals)
        out.loc[pos] = vals
    return out


def build_layer_context(predictions: pd.DataFrame) -> dict[str, Any]:
    date_mult, geo = v3_date_mult(predictions)
    idx = date_mult.index
    agree = geo["head_state"].astype(str).reindex(idx)
    sep = sep_bucket(geo["top2_minus_median_p250"]).reindex(idx)
    n_cand = _n_cand(predictions).reindex(idx)
    strata = _strata(idx, n_cand, date_mult)
    v2_state = (pd.to_numeric(geo["top1_minus_median_p250"], errors="coerce").reindex(idx) <= 0.20).map({True: "LOW", False: "OK"})
    v3_state = agree
    v4_state = agree.astype(str) + "_" + sep.astype(str)
    return {
        "predictions": predictions,
        "date_mult": date_mult,
        "geo": geo,
        "agree": agree,
        "sep": sep,
        "strata": strata,
        "v2_state": v2_state,
        "v3_state": v3_state,
        "v4_state": v4_state,
        "idx": idx,
    }


def overlay_from_shuffled(ctx: dict[str, Any], layer: str, rng: np.random.Generator) -> pd.DataFrame:
    from ..round8.overlays import confidence_from_kind, market_b01a, product_date
    from ..round8.overlays import build_overlayed as r8_build

    pred = ctx["predictions"]
    strata = ctx["strata"]
    if layer == "v4":
        shuffled = _permute_labels(ctx["v4_state"], strata, rng)
        agree = shuffled.str.split("_").str[0]
        sep = shuffled.str.split("_").str[1]
        nav = b03_nav(agree, sep)
        top2 = nav_to_mult(nav).reindex(ctx["idx"]).fillna(MIN_KEEP)
        return apply_rank_multipliers(pred, ctx["date_mult"], top2_extra=top2)
    if layer == "v3":
        # 打乱 disagreement / head_state，保留 v2 date_mult，重做 E02 Top2 乘子
        shuffled = _permute_labels(ctx["v3_state"], strata, rng)
        mapping = {"A": 1.00, "B": 0.75, "C": 0.50}
        top2 = shuffled.map(mapping).astype(float).fillna(1.0)
        top2 = top2.where(top2 > 0, MIN_KEEP).clip(MIN_KEEP, 1.0)
        return apply_rank_multipliers(pred, ctx["date_mult"], top2_extra=top2)
    if layer == "v2":
        shuffled = _permute_labels(ctx["v2_state"], strata, rng)
        conf = shuffled.map({"LOW": 0.50, "OK": 1.00}).astype(float).fillna(1.0)
        market = market_b01a(pred).reindex(ctx["idx"]).fillna(1.0)
        date_mult = product_date(market, conf)
        return apply_rank_multipliers(pred, date_mult)
    raise ValueError(layer)


def _actual_for(layer: str) -> dict[str, Any]:
    paths = {
        "v2": ROUND11_ROOT / "cost" / "R11_V2_CORRECTED" / "nav.parquet",
        "v3": ROUND11_ROOT / "cost" / "R11_V3_CORRECTED" / "nav.parquet",
        "v4": ROUND11_ROOT / "cost" / "R11_V4_CORRECTED" / "nav.parquet",
    }
    path = paths[layer]
    if not path.is_file():
        return {}
    return _metrics_pack(pd.read_parquet(path))


def _init_worker() -> None:
    if _G.get("market") is not None and _G.get("ctx") is not None:
        return
    from ..replay_round7.inject import inject_round7_into_overlay_caches
    from ..stock_readonly import load_prepared_market

    inject_round7_into_overlay_caches()
    loaded = load_prepared_market(warmup_ma5=True)
    pred = _load_predictions()
    _G["market"] = loaded["prepared"]
    _G["meta"] = loaded
    _G["ctx"] = build_layer_context(pred)


def _one(layer: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    ov = overlay_from_shuffled(_G["ctx"], layer, rng)
    ran = run_r12_backtest(
        ROUND12_ROOT / "timing_placebo" / f"_mem_{layer}_{seed}",
        prepared_market=_G["market"],
        market_meta=_G["meta"],
        overlayed=ov,
        validation_hooks=hist_hooks(),
        extra_report={"timing_layer": layer, "seed": seed, "fee_schedule": "HISTORICAL_FEE_CORRECTED"},
        resume=False,
        write=False,
    )
    pack = _metrics_pack(ran["nav"])
    pack["layer"] = layer
    pack["seed"] = seed
    return pack


def run_timing_placebo(*, resume: bool = True, jobs: int = 1, **_: Any) -> dict[str, Any]:
    dest = ROUND12_ROOT / "timing_placebo"
    dest.mkdir(parents=True, exist_ok=True)
    ckpt = dest / "timing_runs.jsonl"
    n_target = N_TIMING
    if resume and TIMING_JSON.is_file():
        existing = load_json(TIMING_JSON)
        if existing.get("status") == "DONE" and min((existing.get("n_by_layer") or {}).values() or [0]) >= 500:
            return existing
    done: dict[tuple[str, int], dict[str, Any]] = {}
    if resume and ckpt.is_file():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done[(rec["layer"], int(rec["seed"]))] = rec
    tasks = [(layer, SEED + i) for layer in LAYERS for i in range(n_target)]
    todo = [t for t in tasks if t not in done]
    if todo:
        _init_worker()
        n_jobs = max(int(jobs), 1)
        if n_jobs == 1:
            with ckpt.open("a", encoding="utf-8") as fh:
                for layer, seed in todo:
                    rec = _one(layer, seed)
                    done[(layer, seed)] = rec
                    fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    fh.flush()
        else:
            import multiprocessing as mp

            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
                futs = {pool.submit(_one, layer, seed): (layer, seed) for layer, seed in todo}
                with ckpt.open("a", encoding="utf-8") as fh:
                    for fut in as_completed(futs):
                        layer, seed = futs[fut]
                        try:
                            rec = fut.result()
                        except Exception as exc:
                            rec = {
                                "layer": layer,
                                "seed": seed,
                                "error": str(exc),
                                "cagr": None,
                                "sharpe": None,
                                "max_drawdown": None,
                                "worst12m": None,
                            }
                        done[(rec["layer"], int(rec["seed"]))] = rec
                        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                        fh.flush()
    by_layer: dict[str, Any] = {}
    for layer in LAYERS:
        runs = [
            done[(layer, SEED + i)]
            for i in range(n_target)
            if (layer, SEED + i) in done and done[(layer, SEED + i)].get("cagr") is not None
        ]
        actual = _actual_for(layer)
        dist = {k: [r.get(k) for r in runs] for k in ("cagr", "sharpe", "max_drawdown", "worst12m")}
        by_layer[layer] = {
            "n": len(runs),
            "actual": actual,
            "percentiles": {k: _percentile(actual.get(k), dist[k]) for k in dist},
            "empirical_p_cagr": (
                float(np.mean(np.array([x for x in dist["cagr"] if x is not None]) >= float(actual["cagr"])))
                if actual.get("cagr") is not None and any(x is not None for x in dist["cagr"]) else None
            ),
        }
    payload = {
        "status": "DONE",
        "n_target": n_target,
        "n_reason": "全引擎太贵，按契约采用 ≥500。",
        "n_by_layer": {k: v["n"] for k, v in by_layer.items()},
        "layers": by_layer,
        "strata": "year × candidate-count quartile × market-mult tertile；状态频率在层内保持",
        "states": {
            "v2": "top1-median LOW/OK confidence",
            "v3": "head_state A/B/C disagreement",
            "v4": "B03 9-cell agree_sep",
        },
    }
    dump_json(TIMING_JSON, payload)
    write_md(
        TIMING_MD,
        "\n".join([
            "# Round 12 · state-timing placebo",
            "",
            f"每层 n = **{n_target}**（全引擎 expensive path ≥500）。",
            "",
            "| 层 | n | 真实 CAGR | CAGR 百分位 | 经验 p(CAGR≥真实) | Sharpe 百分位 |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                f"| {k} | {v['n']} | {(v.get('actual') or {}).get('cagr')} | {(v.get('percentiles') or {}).get('cagr')} | {v.get('empirical_p_cagr')} | {(v.get('percentiles') or {}).get('sharpe')} |"
                for k, v in by_layer.items()
            ],
            "",
            "组合构建前打乱状态；保留年、候选数桶、市场风险桶、状态频率。",
            "",
        ]),
    )
    return payload


def load_timing() -> dict[str, Any]:
    return load_json(TIMING_JSON)


__all__ = ["load_timing", "run_timing_placebo"]
