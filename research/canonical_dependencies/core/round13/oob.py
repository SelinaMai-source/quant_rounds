"""Winner-vs-ensemble OOB、全引擎 placebo、timing placebo（含 PnL/暴露）。"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND13_ROOT
from ..round11.common import daily_from_nav, max_drawdown_of, sharpe_of, worst_window_return
from ..round12.placebo import _cagr_from_daily, _metrics_pack, _percentile, _shuffle_overlay
from ..round12.runner import hist_hooks
from .candidates import load_candidates, load_daily_map
from .common import cagr_of, dump_json, load_json, write_md
from .contract import (
    BLOCK_DAYS,
    CANDIDATE_IDS,
    COMPLEXITY_ORDER,
    OOB_N,
    PLACEBO_N,
    R13_00,
    TIMING_N,
    overlayed_path,
)
from .freeze import require_frozen
from .runner import run_r13_backtest
from .weights import load_overlayed, load_weights_table

OOB_JSON = ROUND13_ROOT / "oob" / "oob.json"
OOB_MD = ROUND13_ROOT / "07_oob.md"
PLACEBO_JSON = ROUND13_ROOT / "placebo" / "placebo.json"
PLACEBO_MD = ROUND13_ROOT / "08_placebo.md"
TIMING_JSON = ROUND13_ROOT / "timing" / "timing.json"
TIMING_MD = ROUND13_ROOT / "09_timing.md"

_G: dict[str, Any] = {}
CONTENDERS = ("v2", "v3", "v4")


def run_oob(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and OOB_JSON.is_file():
        existing = load_json(OOB_JSON)
        if existing.get("status") == "DONE" and int(existing.get("n") or 0) >= 500:
            return existing
    dailies = load_daily_map()
    aligned = pd.concat({k: dailies[k] for k in [*CONTENDERS, *CANDIDATE_IDS] if k in dailies}, axis=1).dropna()
    n = len(aligned)
    block = BLOCK_DAYS
    n_blocks = n // block
    if n_blocks < 8:
        raise RuntimeError("OOB too few blocks")
    rng = np.random.default_rng(20261013)
    recs = {cid: [] for cid in CANDIDATE_IDS}
    winners = []
    n_use = OOB_N
    for _ in range(n_use):
        perm = rng.permutation(n_blocks)
        split = n_blocks // 2
        train_b = perm[:split]
        test_b = perm[split:]

        def _take(blocks_idx):
            idx = np.concatenate([np.arange(b * block, min((b + 1) * block, n)) for b in blocks_idx])
            return aligned.iloc[idx]

        train = _take(train_b)
        test = _take(test_b)
        is_sr = {k: sharpe_of(train[k]) or -9.0 for k in CONTENDERS}
        winner = max(is_sr, key=is_sr.get)
        winners.append(winner)
        w_oos = sharpe_of(test[winner]) or -9.0
        for cid in CANDIDATE_IDS:
            a_oos = sharpe_of(test[cid]) or -9.0
            recs[cid].append(a_oos - w_oos)
    payload_ids = {}
    for cid in CANDIDATE_IDS:
        arr = np.array(recs[cid], dtype=float)
        payload_ids[cid] = {
            "p_gt_winner": float(np.mean(arr > 0)),
            "median_delta": float(np.median(arr)),
            "p05": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
            "mean_delta": float(np.mean(arr)),
        }
    from collections import Counter

    payload = {
        "status": "DONE",
        "n": n_use,
        "block": block,
        "n_blocks": n_blocks,
        "train_share": 0.5,
        "winner_freq": dict(Counter(winners)),
        "by_id": payload_ids,
        "note": "TRAIN 只在 v2/v3/v4 中选 Sharpe 胜者；OOS 比固定 A01–A05。",
    }
    dump_json(OOB_JSON, payload)
    lines = [
        "# Round 13 · Winner-vs-ensemble OOB",
        "",
        f"N=**{n_use}** 20 日块。IS 胜者频率={payload['winner_freq']}。",
        "",
        "| ID | P(Axx>winner) | median ΔSharpe | 5% | 95% |",
        "|---|---:|---:|---:|---:|",
    ]
    for cid in CANDIDATE_IDS:
        r = payload_ids[cid]
        lines.append(f"| `{cid}` | {r['p_gt_winner']:.3f} | {r['median_delta']:.4f} | {r['p05']:.4f} | {r['p95']:.4f} |")
    write_md(OOB_MD, "\n".join(lines) + "\n")
    return payload


def _init_placebo(code: str) -> None:
    from .runner import load_market

    loaded = load_market()
    _G["market"] = loaded["prepared"]
    _G["meta"] = loaded
    _G["overlayed"] = load_overlayed(code).reset_index(drop=True)
    _G["code"] = code


def _one_placebo(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    ov = _shuffle_overlay(_G["overlayed"], rng)
    ran = run_r13_backtest(
        ROUND13_ROOT / "placebo" / f"_mem_{_G['code']}_{seed}",
        overlayed=ov,
        validation_hooks=hist_hooks(),
        extra_report={"placebo_seed": seed, "parent": _G["code"]},
        resume=False,
        write=False,
        prepared_market=_G["market"],
        market_meta=_G["meta"],
    )
    pack = _metrics_pack(ran["nav"])
    pack["seed"] = seed
    pack["code"] = _G["code"]
    return pack


def _placebo_one_code(code: str, *, resume: bool, jobs: int, overlay_key: str) -> dict[str, Any]:
    dest = ROUND13_ROOT / "placebo" / code
    dest.mkdir(parents=True, exist_ok=True)
    ckpt = dest / "runs.jsonl"
    nav_path = (ROUND13_ROOT / "parity" / R13_00 / "nav.parquet") if code in {R13_00, "v4"} else (
        ROUND13_ROOT / "candidates" / code / "nav.parquet"
    )
    if code == "v4":
        nav_path = ROUND13_ROOT / "parity" / R13_00 / "nav.parquet"
    actual = _metrics_pack(pd.read_parquet(nav_path)) if nav_path.is_file() else {}
    done: dict[int, dict[str, Any]] = {}
    if resume and ckpt.is_file():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done[int(rec["seed"])] = rec
    seed0 = 20261013
    seeds = [seed0 + i for i in range(PLACEBO_N)]
    todo = [s for s in seeds if s not in done]
    if todo:
        _init_placebo(overlay_key)
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
    return {
        "n_placebo": len(runs),
        "target_n": PLACEBO_N,
        "actual": actual,
        "percentiles": {k: _percentile(actual.get(k), dist[k]) for k in dist},
        "dist_summary": {
            k: {
                "p05": float(np.nanpercentile([x for x in vs if x is not None], 5)) if any(x is not None for x in vs) else None,
                "p50": float(np.nanpercentile([x for x in vs if x is not None], 50)) if any(x is not None for x in vs) else None,
                "p95": float(np.nanpercentile([x for x in vs if x is not None], 95)) if any(x is not None for x in vs) else None,
            }
            for k, vs in dist.items()
        },
    }


def run_placebo(*, resume: bool = True, jobs: int = 1, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and PLACEBO_JSON.is_file():
        existing = load_json(PLACEBO_JSON)
        if existing.get("status") == "DONE" and int(existing.get("min_n") or 0) >= PLACEBO_N:
            return existing
    targets = [("v4", "v4"), *((cid, cid) for cid in CANDIDATE_IDS)]
    by_code = {}
    for label, ov_key in targets:
        nav_code = R13_00 if label == "v4" else label
        by_code[label] = _placebo_one_code(nav_code, resume=resume, jobs=jobs, overlay_key=ov_key)
    # fix: v4 already stored as R13_00 if we called with R13_00. Keep both keys.
    payload = {
        "status": "DONE",
        "method": "FULL_ENGINE_MATCHED_PLACEBO",
        "min_n": min((v.get("n_placebo") or 0) for v in by_code.values()) if by_code else 0,
        "by_code": by_code,
        "keeps": ["signal_days", "candidate_set", "exposure_calendar", "engine", "fees", "exit"],
        "note": "当日合格候选内随机名次。finalist 目标 ≥1000。",
    }
    dump_json(PLACEBO_JSON, payload)
    lines = [
        "# Round 13 · 全引擎 matched placebo",
        "",
        f"目标 n=**{PLACEBO_N}**。",
        "",
        "| ID | n | CAGR pct | Sharpe pct | MaxDD pct | Worst12M pct |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for code, row in by_code.items():
        p = row.get("percentiles") or {}
        lines.append(
            f"| `{code}` | {row.get('n_placebo')} | {p.get('cagr')} | {p.get('sharpe')} | "
            f"{p.get('max_drawdown')} | {p.get('worst12m')} |"
        )
    write_md(PLACEBO_MD, "\n".join(lines) + "\n")
    return payload


def _exposure_series(code: str) -> pd.Series:
    w = load_weights_table()
    src = "v4" if code in {R13_00, "v4"} else code
    sub = w.loc[w["source"].astype(str).eq(src)].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    sub["signal_date"] = pd.to_datetime(sub["signal_date"]).dt.normalize()
    return sub.groupby("signal_date")["target_nav_fraction"].sum().sort_index()


def _pnl_per_exposure(daily: pd.Series, expo: pd.Series) -> dict[str, Any]:
    aligned = pd.concat([daily.rename("r"), expo.rename("e")], axis=1).dropna()
    if aligned.empty:
        return {"pnl_per_unit_exposure": None, "mean_ret": None, "mean_exposure": None}
    mean_e = float(aligned["e"].mean())
    return {
        "mean_ret": float(aligned["r"].mean()),
        "mean_exposure": mean_e,
        "pnl_per_unit_exposure": float(aligned["r"].sum() / aligned["e"].sum()) if aligned["e"].sum() else None,
        "mean_ret_over_mean_exposure": (float(aligned["r"].mean()) / mean_e) if mean_e else None,
    }


def _strata_from_weights(idx: pd.DatetimeIndex) -> pd.Series:
    from ..round2.data import load_regime_frame
    from .weights import load_weights_table

    w = load_weights_table()
    w = w.loc[w["source"].eq("v4")].copy()
    w["signal_date"] = pd.to_datetime(w["signal_date"]).dt.normalize()
    n_cand = w.groupby("signal_date")["instrument_id"].nunique().reindex(idx)
    year = pd.Series(idx.year, index=idx).astype(str)
    try:
        n_b = pd.qcut(n_cand.rank(method="first"), 4, labels=["n1", "n2", "n3", "n4"])
    except ValueError:
        n_b = pd.Series("nX", index=idx)
    try:
        regime = load_regime_frame()
        mk = pd.to_numeric(regime["up_ratio"], errors="coerce").reindex(idx)
        m_b = pd.qcut(mk.rank(method="first"), 3, labels=["m1", "m2", "m3"])
    except Exception:
        m_b = pd.Series("mX", index=idx)
    return year + "|" + pd.Series(n_b, index=idx).astype(str) + "|" + pd.Series(m_b, index=idx).astype(str)


def _one_timing(seed: int) -> dict[str, Any]:
    from .weights import inject_weights

    rng = np.random.default_rng(seed)
    shuffled = _G["a01"].copy()
    top2 = shuffled.loc[shuffled["engine_rank"].eq(2)].copy()
    dates = top2["signal_date"].to_numpy()
    new_w = top2["target_nav_fraction"].to_numpy().copy()
    new_m = top2["allocation_multiplier"].to_numpy().copy()
    st = _G["strata"].reindex(pd.DatetimeIndex(pd.to_datetime(dates))).astype(str)
    for key in pd.unique(st.fillna("NA")):
        pos = np.where(st.fillna("NA").to_numpy() == key)[0]
        if len(pos) < 2:
            continue
        perm = rng.permutation(len(pos))
        new_w[pos] = new_w[pos][perm]
        new_m[pos] = new_m[pos][perm]
    top2 = top2.copy()
    top2["target_nav_fraction"] = new_w
    top2["allocation_multiplier"] = new_m
    shuf = pd.concat([shuffled.loc[shuffled["engine_rank"].eq(1)], top2], ignore_index=True)
    ov = inject_weights(_G["base_v4"], shuf)
    loaded = _G["loaded"]
    ran = run_r13_backtest(
        _G["dest"] / f"_mem_{seed}",
        overlayed=ov,
        validation_hooks=hist_hooks(),
        extra_report={"timing_seed": seed},
        resume=False,
        write=False,
        prepared_market=loaded["prepared"],
        market_meta=loaded,
    )
    pack = _metrics_pack(ran["nav"])
    daily = daily_from_nav(ran["nav"])
    expo = shuf.groupby("signal_date")["target_nav_fraction"].sum().sort_index()
    expo.index = pd.to_datetime(expo.index).normalize()
    pack.update(_pnl_per_exposure(daily, expo))
    pack["seed"] = seed
    return pack


def run_timing(*, resume: bool = True, jobs: int = 1, **_: Any) -> dict[str, Any]:
    """打乱 v2/v3/v4 Top2 分歧模式，重建 A01，检验 ensemble 是否只是更平滑暴露。"""
    require_frozen()
    if resume and TIMING_JSON.is_file():
        existing = load_json(TIMING_JSON)
        if existing.get("status") == "DONE" and int(existing.get("n") or 0) >= 500:
            return existing
    from .runner import build_version_overlayed, load_market
    from .weights import load_weights_table

    w = load_weights_table()
    base_v4, _, _ = build_version_overlayed("v4")
    loaded = load_market()
    a01 = w.loc[w["source"].eq("A01")].copy()
    a01["signal_date"] = pd.to_datetime(a01["signal_date"]).dt.normalize()
    idx = pd.DatetimeIndex(sorted(a01["signal_date"].unique()))
    strata = _strata_from_weights(idx)
    daily_map = load_daily_map()
    actual_expo = _pnl_per_exposure(daily_map["A01"], _exposure_series("A01"))
    v4_expo = _pnl_per_exposure(daily_map["v4"], _exposure_series("v4"))

    dest = ROUND13_ROOT / "timing"
    dest.mkdir(parents=True, exist_ok=True)
    ckpt = dest / "timing_runs.jsonl"
    done: dict[int, dict[str, Any]] = {}
    if resume and ckpt.is_file():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("error") or rec.get("cagr") is None:
                continue
            done[int(rec["seed"])] = rec
    seeds = [20261014 + i for i in range(TIMING_N)]
    todo = [s for s in seeds if s not in done]
    _G["a01"] = a01
    _G["strata"] = strata
    _G["base_v4"] = base_v4
    _G["loaded"] = loaded
    _G["dest"] = dest

    if todo:
        n_jobs = max(int(jobs), 1)
        if n_jobs == 1:
            with ckpt.open("a", encoding="utf-8") as fh:
                for seed in todo:
                    rec = _one_timing(seed)
                    done[seed] = rec
                    fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    fh.flush()
        else:
            import multiprocessing as mp

            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
                futs = {pool.submit(_one_timing, seed): seed for seed in todo}
                with ckpt.open("a", encoding="utf-8") as fh:
                    for fut in as_completed(futs):
                        seed = futs[fut]
                        try:
                            rec = fut.result()
                        except Exception as exc:
                            rec = {"seed": seed, "error": str(exc), "cagr": None, "sharpe": None}
                        done[int(rec["seed"])] = rec
                        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                        fh.flush()
    runs = [done[s] for s in seeds if s in done and done[s].get("cagr") is not None]
    actual = _metrics_pack(pd.read_parquet(ROUND13_ROOT / "candidates" / "A01" / "nav.parquet"))
    dist_cagr = [r.get("cagr") for r in runs]
    dist_pnl = [r.get("pnl_per_unit_exposure") for r in runs]
    payload = {
        "status": "DONE",
        "n": len(runs),
        "target_n": TIMING_N,
        "strata": "year × candidate-count quartile × up_ratio tertile",
        "actual_A01": actual,
        "actual_pnl_exposure": actual_expo,
        "v4_pnl_exposure": v4_expo,
        "percentiles": {
            "cagr": _percentile(actual.get("cagr"), dist_cagr),
            "sharpe": _percentile(actual.get("sharpe"), [r.get("sharpe") for r in runs]),
            "pnl_per_unit_exposure": _percentile(actual_expo.get("pnl_per_unit_exposure"), dist_pnl),
        },
        "empirical_p_cagr": (
            float(np.mean(np.array([x for x in dist_cagr if x is not None]) >= float(actual["cagr"])))
            if actual.get("cagr") is not None and any(x is not None for x in dist_cagr) else None
        ),
        "note": "打乱 A01 的 Top2 权重时间（层内保持频率）。若真实点不优于 shuffled，则 ensemble 可能只是更平滑暴露。",
    }
    dump_json(TIMING_JSON, payload)
    write_md(
        TIMING_MD,
        "\n".join([
            "# Round 13 · timing placebo",
            "",
            f"n=**{len(runs)}**（目标 {TIMING_N}）。层 = 年 × 候选数桶 × 风险桶。",
            f"- A01 真实 CAGR 百分位 = **{(payload['percentiles'] or {}).get('cagr')}** empir_p={payload.get('empirical_p_cagr')}",
            f"- A01 PnL/单位暴露 = {actual_expo.get('pnl_per_unit_exposure')}；v4 = {v4_expo.get('pnl_per_unit_exposure')}",
            f"- shuffled PnL/暴露 百分位 = {(payload['percentiles'] or {}).get('pnl_per_unit_exposure')}",
            "",
        ]),
    )
    return payload


def load_oob() -> dict[str, Any]:
    return load_json(OOB_JSON)


def load_placebo() -> dict[str, Any]:
    return load_json(PLACEBO_JSON)


def load_timing() -> dict[str, Any]:
    return load_json(TIMING_JSON)


__all__ = ["load_oob", "load_placebo", "load_timing", "run_oob", "run_placebo", "run_timing"]
