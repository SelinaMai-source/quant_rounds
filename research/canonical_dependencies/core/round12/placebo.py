"""全引擎 matched placebo：当日合格候选内随机名次。保留 B03 日程 / 费用 / exit。"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND12_ROOT
from ..round8.overlays import MIN_KEEP
from ..round11.common import daily_from_nav, max_drawdown_of, sharpe_of, worst_window_return
from .common import dump_json, load_json, write_md
from .runner import hist_hooks, load_v4_overlayed, run_r12_backtest

PLACEBO_MD = ROUND12_ROOT / "11_matched_placebo.md"
PLACEBO_JSON = ROUND12_ROOT / "placebo" / "placebo.json"
N_PLACEBO = 1000
SEED = 20261012


def _cagr_from_daily(daily: pd.Series) -> float | None:
    x = daily.dropna().astype(float)
    if x.empty:
        return None
    wealth = float((1.0 + x).prod())
    years = max(len(x) / 252.0, 1e-9)
    return wealth ** (1.0 / years) - 1.0


def _metrics_pack(nav: pd.DataFrame) -> dict[str, Any]:
    daily = daily_from_nav(nav)
    w12 = worst_window_return(daily, 252)
    return {
        "cagr": _cagr_from_daily(daily),
        "sharpe": sharpe_of(daily),
        "max_drawdown": max_drawdown_of(daily),
        "worst12m": w12.get("return"),
        "n_days": int(len(daily)),
    }


def _shuffle_overlay(overlayed: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    work = overlayed.copy().reset_index(drop=True)
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
    work["instrument_id"] = work["instrument_id"].astype(str)
    work["_rank"] = work.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    slots = {}
    for day, g in work.groupby("signal_date", sort=False):
        m1 = g.loc[g["_rank"].eq(1), "allocation_multiplier"]
        m2 = g.loc[g["_rank"].eq(2), "allocation_multiplier"]
        slots[day] = (
            float(m1.iloc[0]) if len(m1) else 1.0,
            float(m2.iloc[0]) if len(m2) else MIN_KEEP,
        )
    new_score = np.empty(len(work), dtype=float)
    for _, g in work.groupby("signal_date", sort=False):
        pos = g.index.to_numpy()
        perm = rng.permutation(len(g))
        new_score[pos] = np.arange(len(g), 0, -1, dtype=float)[perm]
    work["score"] = new_score
    work["_new_rank"] = work.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    work["allocation_multiplier"] = MIN_KEEP
    for day, g in work.groupby("signal_date", sort=False):
        w1, w2 = slots[day]
        t1 = g.index[g["_new_rank"].eq(1)]
        t2 = g.index[g["_new_rank"].eq(2)]
        if len(t1):
            work.loc[t1, "allocation_multiplier"] = w1
        if len(t2):
            work.loc[t2, "allocation_multiplier"] = w2
    return work.drop(columns=["_rank", "_new_rank"], errors="ignore")


_G: dict[str, Any] = {}


def _init_worker() -> None:
    if _G.get("market") is not None and _G.get("overlayed") is not None:
        return
    from ..replay_round7.inject import inject_round7_into_overlay_caches
    from ..stock_readonly import load_prepared_market

    inject_round7_into_overlay_caches()
    loaded = load_prepared_market(warmup_ma5=True)
    overlayed, _, _ = load_v4_overlayed()
    _G["market"] = loaded["prepared"]
    _G["meta"] = loaded
    _G["overlayed"] = overlayed.reset_index(drop=True)


def _one_placebo(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    ov = _shuffle_overlay(_G["overlayed"], rng)
    ran = run_r12_backtest(
        ROUND12_ROOT / "placebo" / f"_mem_{seed}",
        prepared_market=_G["market"],
        market_meta=_G["meta"],
        overlayed=ov,
        validation_hooks=hist_hooks(),
        extra_report={"placebo_seed": seed, "fee_schedule": "HISTORICAL_FEE_CORRECTED"},
        resume=False,
        write=False,
    )
    pack = _metrics_pack(ran["nav"])
    pack["seed"] = seed
    pack["completed_trades"] = (ran.get("summary") or {}).get("completed_trades")
    return pack


def _percentile(actual: float | None, dist: list[float | None]) -> float | None:
    xs = [float(x) for x in dist if x is not None and np.isfinite(x)]
    if actual is None or not xs:
        return None
    return float(100.0 * np.mean(np.array(xs) <= float(actual)))


def run_placebo(*, resume: bool = True, jobs: int = 1, **_: Any) -> dict[str, Any]:
    dest = ROUND12_ROOT / "placebo"
    dest.mkdir(parents=True, exist_ok=True)
    ckpt = dest / "placebo_runs.jsonl"
    if resume and PLACEBO_JSON.is_file():
        existing = load_json(PLACEBO_JSON)
        if existing.get("n_placebo", 0) >= N_PLACEBO and existing.get("status") == "DONE":
            return existing
    from pathlib import Path

    from ..stock_readonly import load_prepared_market

    overlayed, _, _ = load_v4_overlayed()
    overlayed = overlayed.reset_index(drop=True)
    loaded = load_prepared_market(warmup_ma5=True)
    _G["market"] = loaded["prepared"]
    _G["meta"] = loaded
    _G["overlayed"] = overlayed
    actual_nav = ROUND12_ROOT / "parity" / "R12_01_CORRECTED_V4" / "nav.parquet"
    if not actual_nav.is_file():
        actual_nav = Path("/home/mxy/Stock/课题/一进二策略/result/round11/cost/R11_V4_CORRECTED/nav.parquet")
    actual = _metrics_pack(pd.read_parquet(actual_nav))
    done: dict[int, dict[str, Any]] = {}
    if resume and ckpt.is_file():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done[int(rec["seed"])] = rec
    seeds = [SEED + i for i in range(N_PLACEBO)]
    todo = [s for s in seeds if s not in done]
    if todo:
        n_jobs = max(int(jobs), 1)
        if n_jobs == 1:
            with ckpt.open("a", encoding="utf-8") as fh:
                for seed in todo:
                    rec = _one_placebo(seed)
                    done[seed] = rec
                    fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    fh.flush()
        else:
            import multiprocessing as mp

            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
                futs = {pool.submit(_one_placebo, seed): seed for seed in todo}
                with ckpt.open("a", encoding="utf-8") as fh:
                    for fut in as_completed(futs):
                        seed = futs[fut]
                        try:
                            rec = fut.result()
                        except Exception as exc:
                            rec = {"seed": seed, "error": str(exc), "cagr": None, "sharpe": None, "max_drawdown": None, "worst12m": None}
                        done[int(rec["seed"])] = rec
                        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                        fh.flush()
    runs = [done[s] for s in seeds if s in done and done[s].get("cagr") is not None]
    dist = {k: [r.get(k) for r in runs] for k in ("cagr", "sharpe", "max_drawdown", "worst12m")}
    # also compare v1-v4 legacy points if available
    versions = {"R12_01": actual}
    for key, path in (
        ("v1", ROUND12_ROOT.parents[0] / "round11/cost/R11_V1_CORRECTED/nav.parquet"),
        ("v2", ROUND12_ROOT.parents[0] / "round11/cost/R11_V2_CORRECTED/nav.parquet"),
        ("v3", ROUND12_ROOT.parents[0] / "round11/cost/R11_V3_CORRECTED/nav.parquet"),
        ("v4", ROUND12_ROOT.parents[0] / "round11/cost/R11_V4_CORRECTED/nav.parquet"),
    ):
        if path.is_file():
            versions[key] = _metrics_pack(pd.read_parquet(path))
    percentiles = {
        name: {k: _percentile(pack.get(k), dist[k]) for k in dist}
        for name, pack in versions.items()
    }
    payload = {
        "status": "DONE",
        "method": "FULL_ENGINE_MATCHED_PLACEBO",
        "n_placebo": len(runs),
        "target_n": N_PLACEBO,
        "seed": SEED,
        "actual_R12_01": actual,
        "percentiles": percentiles,
        "dist_summary": {
            k: {
                "p05": float(np.nanpercentile([x for x in vs if x is not None], 5)) if any(x is not None for x in vs) else None,
                "p50": float(np.nanpercentile([x for x in vs if x is not None], 50)) if any(x is not None for x in vs) else None,
                "p95": float(np.nanpercentile([x for x in vs if x is not None], 95)) if any(x is not None for x in vs) else None,
            }
            for k, vs in dist.items()
        },
        "keeps": ["signal_days", "candidate_set", "B03_exposure_calendar", "risk_layer", "engine", "fees", "exit"],
        "note": "只在当日合格候选内随机名次。这是对 Round11 隔离 unit-fill 的升级。",
    }
    dump_json(PLACEBO_JSON, payload)
    write_md(
        PLACEBO_MD,
        "\n".join([
            "# Round 12 · 全引擎 matched placebo",
            "",
            f"n = **{len(runs)}**（目标 {N_PLACEBO}）。方法 = FULL_ENGINE + corrected cost。",
            "",
            f"- R12_01 CAGR 百分位 = **{(percentiles.get('R12_01') or {}).get('cagr')}**",
            f"- R12_01 Sharpe 百分位 = **{(percentiles.get('R12_01') or {}).get('sharpe')}**",
            f"- R12_01 MaxDD 百分位 = **{(percentiles.get('R12_01') or {}).get('max_drawdown')}**",
            f"- R12_01 Worst12M 百分位 = **{(percentiles.get('R12_01') or {}).get('worst12m')}**",
            "",
            "保留交易日、候选集、B03 暴露日程、风险层、引擎、费用、exit。",
            "",
        ]),
    )
    return payload


def load_placebo() -> dict[str, Any]:
    return load_json(PLACEBO_JSON)


__all__ = ["load_placebo", "run_placebo"]
