"""a-placebo：C4 N=1000 全引擎匹配随机。禁止 ledger shuffle。对照不得晋级。"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from ...paths import ROUND19_ROOT, assert_project_result_path
from ...round12.runner import hist_hooks, run_r12_backtest
from ...round15.runner import apply_top2_veto
from ...round16_5.common import empirical_p, nav_end
from ...round16_5.contract import ENGINE_CHECK_SEEDS, MASK_CHECK_N, MATCHING_CONSTRAINT_FAILURE
from ...round16_5.placebo import _breadth_bucket, _draw_dates
from ...round17.backtest import cached_market, cached_overlay_v4
from ..common import daily_from_nav, dump_json, load_json, write_md
from ..contract import A_C4, PERM_N, PERM_SEED
from .engine import o01_signal_dates
from .support import (
    CHECK_JSON,
    EVENTS_JSON,
    PLACEBO_CKPT,
    PLACEBO_JSON,
    idx_dates,
    load_nav_trades,
    pack_daily,
    require_c0,
    require_c1,
    resume_ok,
    skip_result,
    ta_md,
)

_G: dict[str, Any] = {}


def build_spec() -> dict[str, Any]:
    events = load_json(EVENTS_JSON)
    eligible = idx_dates(events.get("eligible_dates") or [])
    signals = idx_dates(o01_signal_dates())
    treat = idx_dates(events.get("planned_treat_dates") or [])
    if len(treat) == 0:
        treat = eligible.intersection(signals)
    strata = pd.Series(eligible.year.astype(str), index=eligible)
    buckets = _breadth_bucket(eligible)
    cell = strata.astype(str) + "|" + buckets.astype(str)
    year_n = {int(y): int((treat.year == int(y)).sum()) for y in sorted(set(eligible.year))}
    cell_n: dict[str, int] = {}
    treat_cell = pd.Series(treat.year.astype(str), index=treat) + "|" + _breadth_bucket(treat).astype(str)
    for key, n in treat_cell.value_counts().items():
        cell_n[str(key)] = int(n)
    secondary_ok = True
    for key, nneed in cell_n.items():
        if nneed > int((cell == key).sum()):
            secondary_ok = False
            break
    return {
        "eligible": eligible,
        "treat": treat,
        "year_n": year_n,
        "cell_n": cell_n,
        "cell": cell,
        "secondary_ok": secondary_ok,
        "layer": "year_x_b01a" if secondary_ok else "year_only",
        "matching_status": "OK" if secondary_ok else MATCHING_CONSTRAINT_FAILURE,
    }


def _metrics_from_nav(nav: pd.DataFrame) -> dict[str, Any]:
    pack = pack_daily(daily_from_nav(nav))
    pack["nav_end"] = nav_end(nav)
    return pack


def _one_perm(seed: int) -> dict[str, Any]:
    spec = _G["spec"]
    rng = np.random.default_rng(int(seed))
    dates = _draw_dates(spec, rng)
    ov = apply_top2_veto(_G["overlayed"], dates)
    ran = run_r12_backtest(
        ROUND19_ROOT / "track_a" / "placebo" / f"_mem_{seed}",
        prepared_market=_G["market"],
        market_meta=_G["meta"],
        overlayed=ov,
        validation_hooks=hist_hooks(),
        extra_report={"placebo_seed": int(seed), "family": A_C4, "fee_schedule": "HISTORICAL_FEE_CORRECTED"},
        resume=False,
        write=False,
    )
    pack = _metrics_from_nav(ran["nav"])
    pack["seed"] = int(seed)
    pack["n_dates"] = int(len(dates))
    pack["completed_trades"] = (ran.get("summary") or {}).get("completed_trades")
    pack["status"] = "ok"
    return pack


def _load_ckpt() -> dict[int, dict[str, Any]]:
    done: dict[int, dict[str, Any]] = {}
    if not PLACEBO_CKPT.is_file():
        return done
    for line in PLACEBO_CKPT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("seed") is None:
            continue
        done[int(rec["seed"])] = rec
    return done


def run_engine_check(*, spec: dict[str, Any], n_check: int = 3) -> dict[str, Any]:
    v4_nav = pd.read_parquet(require_c0() / "nav.parquet")
    v01_nav = pd.read_parquet(require_c1() / "nav.parquet")
    v4_daily = daily_from_nav(v4_nav)
    v01_daily = daily_from_nav(v01_nav)
    ov = cached_overlay_v4()
    loaded = cached_market()
    rows = []
    disagree = 0
    for seed in list(ENGINE_CHECK_SEEDS)[:n_check]:
        rng = np.random.default_rng(int(seed))
        dates = _draw_dates(spec, rng)
        ov_v = apply_top2_veto(ov, dates)
        ran = run_r12_backtest(
            ROUND19_ROOT / "track_a" / "placebo" / f"_check_{seed}",
            prepared_market=loaded["prepared"],
            market_meta=loaded,
            overlayed=ov_v,
            validation_hooks=hist_hooks(),
            extra_report={"engine_check_seed": int(seed)},
            resume=False,
            write=False,
        )
        canon = daily_from_nav(ran["nav"])
        dummy = v4_daily.copy()
        hit = dummy.index.isin(pd.DatetimeIndex(dates))
        dummy.loc[hit] = v01_daily.reindex(dummy.index).loc[hit]
        aligned = pd.concat({"c": canon, "swap": dummy}, axis=1).dropna()
        swap_max = float((aligned["c"] - aligned["swap"]).abs().max()) if len(aligned) else None
        ok_swap = swap_max is not None and swap_max < 1e-10
        if not ok_swap:
            disagree += 1
        rows.append({"seed": int(seed), "n_dates": int(len(dates)), "swap_max_abs_daily": swap_max, "swap_matches": ok_swap})
    use_canonical = True
    payload = {
        "status": "DONE",
        "n_check": int(len(rows)),
        "n_disagree": int(disagree),
        "use_canonical": True,
        "reason": "optimized vs canonical 不一致则以 canonical 为准；C4 全程 apply_top2_veto + run_r12_backtest(write=False)。",
        "rows": rows,
        "not_ledger_shuffle": True,
    }
    dump_json(CHECK_JSON, payload)
    return payload


def run_placebo(*, resume: bool = True, jobs: int = 8, **_: Any) -> dict[str, Any]:
    existing = load_json(PLACEBO_JSON) if resume else {}
    if existing.get("status") == "DONE" and not existing.get("incomplete") and int(existing.get("n") or 0) >= PERM_N:
        return existing
    events = load_json(EVENTS_JSON)
    if not events.get("eligible_dates"):
        payload = skip_result("a-placebo", "ELIGIBLE_DATES_MISSING", status="BLOCKED")
        dump_json(PLACEBO_JSON, payload)
        return payload
    try:
        spec = build_spec()
        check = run_engine_check(spec=spec)
        n0 = pd.read_parquet(require_c0() / "nav.parquet")
        n1 = pd.read_parquet(require_c1() / "nav.parquet")
    except FileNotFoundError as exc:
        payload = skip_result("a-placebo", str(exc), status="BLOCKED")
        dump_json(PLACEBO_JSON, payload)
        return payload
    v00 = pack_daily(daily_from_nav(n0))
    v01 = pack_daily(daily_from_nav(n1))
    seeds = [PERM_SEED + i for i in range(PERM_N)]
    done = _load_ckpt() if resume else {}
    todo = [s for s in seeds if s not in done or done[s].get("status") != "ok"]
    _G["overlayed"] = cached_overlay_v4().reset_index(drop=True)
    loaded = cached_market()
    _G["market"] = loaded["prepared"]
    _G["meta"] = loaded
    _G["spec"] = spec
    PLACEBO_CKPT.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(PLACEBO_CKPT)
    if todo:
        n_jobs = max(int(jobs), 1)
        if n_jobs == 1:
            with PLACEBO_CKPT.open("a", encoding="utf-8") as fh:
                for seed in todo:
                    rec = _one_perm(seed)
                    done[seed] = rec
                    fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    fh.flush()
        else:
            import multiprocessing as mp

            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
                futs = {pool.submit(_one_perm, seed): seed for seed in todo}
                with PLACEBO_CKPT.open("a", encoding="utf-8") as fh:
                    for fut in as_completed(futs):
                        seed = futs[fut]
                        try:
                            rec = fut.result()
                        except Exception as exc:
                            rec = {
                                "seed": seed,
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                                "cagr": None,
                                "sharpe": None,
                            }
                        done[int(rec["seed"])] = rec
                        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                        fh.flush()
    runs = [done[s] for s in seeds if s in done and done[s].get("status") == "ok" and done[s].get("sharpe") is not None]
    n = len(runs)
    sharpes = [float(r["sharpe"]) for r in runs]
    cagrs = [float(r["cagr"]) for r in runs if r.get("cagr") is not None]
    nav_ends = [float(r["nav_end"]) for r in runs if r.get("nav_end") is not None]
    o01_sh = float(v01.get("sharpe") or 0)
    o01_cagr = float(v01.get("cagr") or 0)
    o01_nav = nav_end(n1)
    n_beat_sh = int(sum(1 for x in sharpes if x >= o01_sh - 1e-15))
    n_beat_cagr = int(sum(1 for x in cagrs if x >= o01_cagr - 1e-15))
    n_beat_nav = int(sum(1 for x in nav_ends if o01_nav is not None and x >= float(o01_nav) - 1e-6))
    payload = {
        "status": "DONE" if n else "NO_RUNS",
        "code": A_C4,
        "not_candidate": True,
        "formal": False,
        "method": "FULL_ENGINE_YEAR_STRATIFIED_TOP2_VETO",
        "not_ledger_swap": True,
        "n": n,
        "n_target": PERM_N,
        "incomplete": n < PERM_N,
        "seed": PERM_SEED,
        "jobs_last": int(jobs),
        "matching": {
            "status": spec["matching_status"],
            "layer": spec["layer"],
            "year_n": spec["year_n"],
            "n_eligible": int(len(spec["eligible"])),
            "n_treat_matched": int(len(spec["treat"])),
        },
        "engine_check": {k: check.get(k) for k in ("n_check", "n_disagree", "use_canonical", "reason")},
        "c0": v00,
        "c1": v01,
        "p_empirical_sharpe": empirical_p(n_beat_sh, n),
        "p_empirical_cagr": empirical_p(n_beat_cagr, n),
        "p_empirical_nav_end": empirical_p(n_beat_nav, n),
        "n_beat_or_tie_sharpe": n_beat_sh,
        "n_beat_or_tie_cagr": n_beat_cagr,
        "n_beat_or_tie_nav_end": n_beat_nav,
        "percentile_sharpe": float(100.0 * np.mean(np.array(sharpes) <= o01_sh)) if sharpes else None,
        "formula": "(1 + #placebos beating or tying O01) / (1 + N)",
        "never_p_zero": True,
        "dist": {
            "sharpe": {
                "mean": float(np.mean(sharpes)) if sharpes else None,
                "p05": float(np.percentile(sharpes, 5)) if sharpes else None,
                "p50": float(np.percentile(sharpes, 50)) if sharpes else None,
                "p95": float(np.percentile(sharpes, 95)) if sharpes else None,
            }
        },
        "o01_sharpe": o01_sh,
        "o01_cagr": o01_cagr,
        "ckpt": str(PLACEBO_CKPT),
    }
    dump_json(PLACEBO_JSON, payload)
    write_md(
        ta_md("placebo.md"),
        "\n".join(
            [
                "# Round19 Track A · a-placebo C4",
                "",
                f"N=`{n}` / `{PERM_N}` 全引擎。p_sharpe=`{payload['p_empirical_sharpe']}` p_cagr=`{payload['p_empirical_cagr']}`。",
                f"分层=`{spec['layer']}` matching=`{spec['matching_status']}` eligible=`{len(spec['eligible'])}` treat=`{len(spec['treat'])}`。",
                "禁止账本置换。C4 不得晋级。canonical 为准。",
                "",
            ]
        ),
    )
    return payload
