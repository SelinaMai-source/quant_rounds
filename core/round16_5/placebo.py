"""V02 全引擎年分层随机 Top2 veto。禁止账本日收益置换。"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_5_ROOT, assert_project_result_path
from ..round8.overlays import MIN_KEEP
from ..round15.runner import apply_top2_veto
from .common import (
    cagr_of,
    daily_from_nav,
    dump_json,
    empirical_p,
    load_json,
    max_drawdown_of,
    nav_end,
    pack_daily,
    pct,
    sharpe_of,
    write_md,
)
from .contract import (
    ENGINE_CHECK_SEEDS,
    MASK_CHECK_N,
    MATCHING_CONSTRAINT_FAILURE,
    PERM_N,
    PERM_SEED,
    V00,
    V01,
    V02,
    parity_dir,
)
from .engine import cached_market, cached_overlay_v4, o01_signal_dates, run_engine, with_rank
from .freeze import require_frozen
from .attribution import load_attribution

PLACEBO_JSON = ROUND16_5_ROOT / "placebo" / "placebo.json"
CHECK_JSON = ROUND16_5_ROOT / "placebo" / "engine_check.json"
CHECK_MD = ROUND16_5_ROOT / "04_placebo_engine_check.md"
PLACEBO_MD = ROUND16_5_ROOT / "05_V02_full_engine_placebo.md"
CKPT = ROUND16_5_ROOT / "placebo" / "placebo_runs.jsonl"

_G: dict[str, Any] = {}


def _breadth_bucket(idx: pd.DatetimeIndex) -> pd.Series:
    loaded = cached_market()
    prepared = loaded["prepared"]
    up = getattr(prepared, "market_up_ratio", None)
    if up is None:
        return pd.Series("mX", index=idx)
    s = pd.to_numeric(pd.Series(up), errors="coerce")
    s.index = pd.to_datetime(s.index).normalize()
    x = s.reindex(idx)
    out = pd.Series("m0", index=idx)
    out = out.where(~x.ge(0.45), "m1")
    out = out.where(~x.ge(0.50), "m2")
    out = out.where(~x.ge(0.60), "m3")
    out = out.where(x.notna(), "mX")
    return out.astype(str)


def build_eligible() -> dict[str, Any]:
    attr = load_attribution()
    eligible = pd.DatetimeIndex(pd.to_datetime(attr.get("eligible_dates") or [])).normalize().unique().sort_values()
    signals = pd.DatetimeIndex(pd.to_datetime(o01_signal_dates())).normalize()
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
        npool = int((cell == key).sum())
        if nneed > npool:
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


def _draw_dates(spec: dict[str, Any], rng: np.random.Generator) -> pd.DatetimeIndex:
    eligible: pd.DatetimeIndex = spec["eligible"]
    picked = []
    if spec["layer"] == "year_x_b01a":
        cell: pd.Series = spec["cell"]
        for key, nneed in spec["cell_n"].items():
            pool = eligible[cell.eq(key)]
            n = min(int(nneed), len(pool))
            if n <= 0:
                continue
            chosen = rng.choice(pool.to_numpy(), size=n, replace=False)
            picked.extend(list(chosen))
    else:
        for year, nneed in spec["year_n"].items():
            pool = eligible[eligible.year == int(year)]
            n = min(int(nneed), len(pool))
            if n <= 0:
                continue
            chosen = rng.choice(pool.to_numpy(), size=n, replace=False)
            picked.extend(list(chosen))
    if not picked:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(picked)).normalize().unique().sort_values()


def _metrics_from_nav(nav: pd.DataFrame) -> dict[str, Any]:
    daily = daily_from_nav(nav)
    pack = pack_daily(daily)
    pack["nav_end"] = nav_end(nav)
    return pack


def _optimized_ledger_swap(v4_daily: pd.Series, cand_daily: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    dummy = v4_daily.copy()
    hit = dummy.index.isin(pd.DatetimeIndex(dates))
    aligned = cand_daily.reindex(dummy.index)
    dummy.loc[hit] = aligned.loc[hit]
    return dummy


def _optimized_subtract_top2(v4_nav: pd.DataFrame, v4_trades: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    t = with_rank(v4_trades)
    t2 = t.loc[t["_rank"].eq(2)].copy()
    t2["signal_date"] = pd.to_datetime(t2["signal_date"]).dt.normalize()
    t2["actual_exit_date"] = pd.to_datetime(t2["actual_exit_date"]).dt.normalize()
    hit = t2.loc[t2["signal_date"].isin(pd.DatetimeIndex(dates))]
    nav = v4_nav.copy()
    nav["trade_date"] = pd.to_datetime(nav["trade_date"]).dt.normalize()
    nav = nav.sort_values("trade_date")
    asset = nav.set_index("trade_date")["total_asset"].astype(float)
    adj = asset.copy()
    for _, row in hit.iterrows():
        exit_d = row["actual_exit_date"]
        net = float(row["net"] or 0.0)
        if exit_d in adj.index:
            adj.loc[adj.index >= exit_d] = adj.loc[adj.index >= exit_d] - net
    daily = adj.pct_change().dropna()
    daily.name = "ret"
    return daily


def _init_worker(spec_lite: dict[str, Any] | None = None) -> None:
    if _G.get("overlayed") is not None:
        return
    ov = cached_overlay_v4()
    loaded = cached_market()
    _G["overlayed"] = ov.reset_index(drop=True)
    _G["market"] = loaded["prepared"]
    _G["meta"] = loaded
    if spec_lite is not None:
        _G["spec_lite"] = spec_lite


def _one_perm(seed: int) -> dict[str, Any]:
    spec = _G["spec"]
    rng = np.random.default_rng(int(seed))
    dates = _draw_dates(spec, rng)
    ov = apply_top2_veto(_G["overlayed"], dates)
    from ..round12.runner import hist_hooks, run_r12_backtest

    ran = run_r12_backtest(
        ROUND16_5_ROOT / "placebo" / f"_mem_{seed}",
        prepared_market=_G["market"],
        market_meta=_G["meta"],
        overlayed=ov,
        validation_hooks=hist_hooks(),
        extra_report={"placebo_seed": int(seed), "family": V02, "fee_schedule": "HISTORICAL_FEE_CORRECTED"},
        resume=False,
        write=False,
    )
    nav = ran["nav"]
    pack = _metrics_from_nav(nav)
    pack["seed"] = int(seed)
    pack["n_dates"] = int(len(dates))
    pack["completed_trades"] = (ran.get("summary") or {}).get("completed_trades")
    pack["status"] = "ok"
    return pack


def _load_ckpt() -> dict[int, dict[str, Any]]:
    done: dict[int, dict[str, Any]] = {}
    if not CKPT.is_file():
        return done
    for line in CKPT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("seed") is None:
            continue
        done[int(rec["seed"])] = rec
    return done


def run_engine_check(*, spec: dict[str, Any], n_check: int = MASK_CHECK_N) -> dict[str, Any]:
    v4_nav = pd.read_parquet(parity_dir(V00) / "nav.parquet")
    v4_tr = pd.read_parquet(parity_dir(V00) / "trades.parquet")
    v01_nav = pd.read_parquet(parity_dir(V01) / "nav.parquet")
    v4_daily = daily_from_nav(v4_nav)
    v01_daily = daily_from_nav(v01_nav)
    ov = cached_overlay_v4()
    loaded = cached_market()
    rows = []
    disagree = 0
    from ..round12.runner import hist_hooks, run_r12_backtest

    for i, seed in enumerate(list(ENGINE_CHECK_SEEDS)[:n_check]):
        rng = np.random.default_rng(int(seed))
        dates = _draw_dates(spec, rng)
        ov_v = apply_top2_veto(ov, dates)
        ran = run_r12_backtest(
            ROUND16_5_ROOT / "placebo" / f"_check_{seed}",
            prepared_market=loaded["prepared"],
            market_meta=loaded,
            overlayed=ov_v,
            validation_hooks=hist_hooks(),
            extra_report={"engine_check_seed": int(seed)},
            resume=False,
            write=False,
        )
        canon_daily = daily_from_nav(ran["nav"])
        swap = _optimized_ledger_swap(v4_daily, v01_daily, dates)
        sub = _optimized_subtract_top2(v4_nav, v4_tr, dates)
        aligned = pd.concat({"c": canon_daily, "swap": swap, "sub": sub}, axis=1).dropna()
        swap_max = float((aligned["c"] - aligned["swap"]).abs().max()) if len(aligned) else None
        sub_max = float((aligned["c"] - aligned["sub"]).abs().max()) if len(aligned) else None
        ok_swap = swap_max is not None and swap_max < 1e-10
        ok_sub = sub_max is not None and sub_max < 1e-10
        if not (ok_swap and ok_sub):
            disagree += 1
        rows.append({
            "seed": int(seed),
            "n_dates": int(len(dates)),
            "canon_sharpe": sharpe_of(canon_daily),
            "swap_max_abs_daily": swap_max,
            "sub_max_abs_daily": sub_max,
            "swap_matches": ok_swap,
            "sub_matches": ok_sub,
        })
    use_canonical = disagree > 0 or not rows
    payload = {
        "status": "DONE",
        "n_check": int(len(rows)),
        "n_disagree": int(disagree),
        "use_canonical": bool(use_canonical),
        "reason": (
            "optimized replay (ledger swap / subtract-Top2) differs from run_r15/run_r12 canonical engine"
            if use_canonical
            else "optimized matches canonical"
        ),
        "rows": rows,
        "note": "不一致则 1000 perm 全程用 canonical apply_top2_veto + run_r12_backtest(write=False)。",
    }
    dump_json(CHECK_JSON, payload)
    write_md(
        CHECK_MD,
        "\n".join([
            "# Round 16.5 · 04 placebo 引擎校验",
            "",
            f"n_check=`{len(rows)}` disagree=`{disagree}` use_canonical=`{use_canonical}`。",
            f"{payload['reason']}。",
            "",
        ]),
    )
    return payload


def run_placebo(*, resume: bool = True, jobs: int = 1, **_: Any) -> dict[str, Any]:
    require_frozen()
    attr = load_attribution()
    if not attr.get("eligible_dates"):
        raise RuntimeError("ATTRIBUTION_MISSING STOP: run attribution before placebo")
    spec = build_eligible()
    check = run_engine_check(spec=spec)
    if not check.get("use_canonical", True):
        # still use canonical; optimized match is unexpected
        pass
    v01 = pack_daily(daily_from_nav(pd.read_parquet(parity_dir(V01) / "nav.parquet")))
    v00 = pack_daily(daily_from_nav(pd.read_parquet(parity_dir(V00) / "nav.parquet")))
    seeds = [PERM_SEED + i for i in range(PERM_N)]
    done = _load_ckpt() if resume else {}
    todo = [s for s in seeds if s not in done or done[s].get("status") != "ok"]
    _G["overlayed"] = cached_overlay_v4().reset_index(drop=True)
    loaded = cached_market()
    _G["market"] = loaded["prepared"]
    _G["meta"] = loaded
    _G["spec"] = spec
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(CKPT)
    if todo:
        n_jobs = max(int(jobs), 1)
        if n_jobs == 1:
            with CKPT.open("a", encoding="utf-8") as fh:
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
                with CKPT.open("a", encoding="utf-8") as fh:
                    for fut in as_completed(futs):
                        seed = futs[fut]
                        try:
                            rec = fut.result()
                        except Exception as exc:
                            rec = {"seed": seed, "status": "error", "error": f"{type(exc).__name__}: {exc}", "cagr": None, "sharpe": None}
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
    o01_nav = nav_end(pd.read_parquet(parity_dir(V01) / "nav.parquet"))
    n_beat_sh = int(sum(1 for x in sharpes if x >= o01_sh - 1e-15))
    n_beat_cagr = int(sum(1 for x in cagrs if x >= o01_cagr - 1e-15))
    n_beat_nav = int(sum(1 for x in nav_ends if o01_nav is not None and x >= float(o01_nav) - 1e-6))
    p_sh = empirical_p(n_beat_sh, n)
    p_cagr = empirical_p(n_beat_cagr, n)
    p_nav = empirical_p(n_beat_nav, n)
    pct_sh = float(100.0 * np.mean(np.array(sharpes) <= o01_sh)) if sharpes else None
    payload = {
        "status": "DONE" if n else "NO_RUNS",
        "method": "FULL_ENGINE_YEAR_STRATIFIED_TOP2_VETO",
        "not_ledger_swap": True,
        "n": n,
        "n_target": PERM_N,
        "n_ok": n,
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
        "v00": v00,
        "v01": v01,
        "p_empirical_sharpe": p_sh,
        "p_empirical_cagr": p_cagr,
        "p_empirical_nav_end": p_nav,
        "n_beat_or_tie_sharpe": n_beat_sh,
        "n_beat_or_tie_cagr": n_beat_cagr,
        "n_beat_or_tie_nav_end": n_beat_nav,
        "percentile_sharpe": pct_sh,
        "formula": "(1 + #placebos beating or tying O01) / (1 + N)",
        "never_p_zero": True,
        "dist": {
            "sharpe": {
                "mean": float(np.mean(sharpes)) if sharpes else None,
                "p05": float(np.percentile(sharpes, 5)) if sharpes else None,
                "p50": float(np.percentile(sharpes, 50)) if sharpes else None,
                "p95": float(np.percentile(sharpes, 95)) if sharpes else None,
            },
            "cagr": {
                "mean": float(np.mean(cagrs)) if cagrs else None,
                "p50": float(np.percentile(cagrs, 50)) if cagrs else None,
            },
        },
        "o01_sharpe": o01_sh,
        "o01_cagr": o01_cagr,
    }
    dump_json(PLACEBO_JSON, payload)
    write_md(
        PLACEBO_MD,
        "\n".join([
            "# Round 16.5 · 05 V02 全引擎 placebo",
            "",
            f"N=`{n}` / 目标 `{PERM_N}`。方法=**FULL_ENGINE** apply_top2_veto + run_r12_backtest。",
            f"分层=`{spec['layer']}` matching=`{spec['matching_status']}` eligible=`{len(spec['eligible'])}` treat_matched=`{len(spec['treat'])}`。",
            f"经验 p Sharpe=`{p_sh}` CAGR=`{p_cagr}` NAV_end=`{p_nav}`（公式 `(1+#beat_or_tie)/(1+N)`，禁止 p=0）。",
            f"O01 Sharpe 百分位=`{pct_sh}`。engine_check use_canonical=`{check.get('use_canonical')}`。",
            "",
            "V02 不是候选。",
            "",
        ]),
    )
    return payload


def load_placebo() -> dict[str, Any]:
    return load_json(PLACEBO_JSON)
