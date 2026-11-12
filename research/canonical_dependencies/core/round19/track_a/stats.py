"""a-stats：块 bootstrap N=5000、HAC、SPA、LOYO、CVaR、+20/+30bp。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...round11.stats import hac_mean_pvalue, spa_test
from ...round16_5.common import empirical_p, holm_adjust
from ..common import daily_from_nav, dump_json, load_json, write_md
from ..contract import BLOCK_DAYS, BOOT_N, COST_BPS, R16_SPA_P, SPA_BOOT
from .engine import cached_overlay_v4, run_overlayed
from .support import (
    A_C0,
    A_C1,
    A_C3,
    ATTR_JSON,
    BOOT_JSON,
    CONC_JSON,
    EQ_JSON,
    PLACEBO_JSON,
    SPA_JSON,
    STAT_JSON,
    STRESS_JSON,
    load_nav_trades,
    overlayed_path,
    pack_daily,
    pack_obs_from_dir,
    parity_dir,
    require_c0,
    require_c1,
    resume_ok,
    skip_result,
    ta_md,
)


def _cvar(daily: pd.Series, q: float = 0.05) -> float | None:
    x = daily.dropna().astype(float)
    if len(x) < 20:
        return None
    thr = float(np.quantile(x, q))
    tail = x[x <= thr]
    if tail.empty:
        return None
    return float(tail.mean())


def _loyo(d0: pd.Series, d1: pd.Series) -> dict[str, Any]:
    aligned = pd.concat({"c0": d0, "c1": d1}, axis=1).dropna()
    rows = []
    for y in sorted(set(aligned.index.year)):
        mask = aligned.index.year != int(y)
        sub = aligned.loc[mask]
        p0 = pack_daily(sub["c0"])
        p1 = pack_daily(sub["c1"])
        rows.append(
            {
                "left_out_year": int(y),
                "c0_cagr": p0.get("cagr"),
                "c1_cagr": p1.get("cagr"),
                "cagr_inc": (p1.get("cagr") or 0) - (p0.get("cagr") or 0),
                "sharpe_inc": (p1.get("sharpe") or 0) - (p0.get("sharpe") or 0),
            }
        )
    flips = [r for r in rows if (r.get("cagr_inc") or 0) <= 0]
    return {"rows": rows, "n_sign_flip": int(len(flips)), "status": "DONE"}


def _resample(mat: np.ndarray, starts: np.ndarray, block: int) -> np.ndarray:
    n = mat.shape[0]
    pieces = [mat[s : s + block] for s in starts]
    return np.concatenate(pieces, axis=0)[:n]


def run_bootstrap(dmap: dict[str, pd.Series]) -> dict[str, Any]:
    cols = [c for c in (A_C0, A_C1, A_C3) if c in dmap]
    aligned = pd.concat({c: dmap[c] for c in cols}, axis=1).dropna()
    mat = aligned.to_numpy(float)
    n = mat.shape[0]
    block = int(BLOCK_DAYS)
    n_blocks = max(1, n - block + 1)
    rng = np.random.default_rng(20261016)
    pair_idx = [(a, b) for a, b in ((A_C1, A_C0), (A_C1, A_C3), (A_C3, A_C0)) if a in cols and b in cols]
    pair_means = {f"{a}-{b}": [] for a, b in pair_idx}
    sharpes = []
    for _ in range(int(BOOT_N)):
        starts = rng.integers(0, n_blocks, size=int(np.ceil(n / block)))
        resampled = _resample(mat, starts, block)
        sh = []
        for j in range(resampled.shape[1]):
            s = pd.Series(resampled[:, j])
            vol = float(s.std(ddof=1))
            sh.append(float(s.mean() / vol * np.sqrt(252)) if vol > 0 else None)
        sharpes.append(sh)
        for a, b in pair_idx:
            ia, ib = cols.index(a), cols.index(b)
            pair_means[f"{a}-{b}"].append(float(np.mean(resampled[:, ia] - resampled[:, ib])))
    boot = np.array(sharpes, dtype=float)
    pairs = {}
    for a, b in pair_idx:
        xs = np.array(pair_means[f"{a}-{b}"], dtype=float)
        n_le0 = int(np.sum(xs <= 0))
        pairs[f"{a}-{b}"] = {
            "mean": float(np.mean(xs)) if len(xs) else None,
            "p10": float(np.percentile(xs, 10)) if len(xs) else None,
            "p90": float(np.percentile(xs, 90)) if len(xs) else None,
            "p_mean_le_0": empirical_p(n_le0, len(xs)),
        }
    holm = holm_adjust({k: v.get("p_mean_le_0") for k, v in pairs.items()})
    for k, v in pairs.items():
        v["p_holm"] = holm.get(k)
    payload = {
        "status": "DONE",
        "n": int(len(boot)),
        "requested": int(BOOT_N),
        "block_days": block,
        "ids": cols,
        "pairs": pairs,
        "sample_size": n,
    }
    dump_json(BOOT_JSON, payload)
    return payload


def run_spa_local(d0: pd.Series, d1: pd.Series, d3: pd.Series | None) -> dict[str, Any]:
    alts = pd.DataFrame({"C1": d1})
    if d3 is not None:
        alts["C3"] = d3
    spa = spa_test(d0, alts, n_boot=SPA_BOOT)
    if spa.get("status") == "TOO_SHORT" and d3 is None:
        # 单备选：C1 vs C0，Hansen 块 bootstrap
        aligned = pd.concat({"b": d0, "a": d1}, axis=1).dropna()
        d = (aligned["a"] - aligned["b"]).to_numpy()
        rng = np.random.default_rng(17)
        n = len(d)
        block = 20
        n_blocks = int(np.ceil(n / block))
        boot_max = []
        obs = float(np.mean(d))
        for _ in range(SPA_BOOT):
            starts = rng.integers(0, max(n - block, 1), size=n_blocks)
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
            boot_max.append(float(np.mean(d[idx])))
        spa = {
            "status": "DONE",
            "observed_max_mean_delta": obs,
            "p_spa": float(np.mean(np.array(boot_max) >= obs)),
            "n_alts": 1,
            "note": "单备选 C1 vs C0 Hansen 块 bootstrap；不是独立新样本。",
        }
    spa["round16_spa_p_retained"] = R16_SPA_P
    spa["methodology_note"] = (
        f"Round16 family SPA p={R16_SPA_P} 必须保留。本轮是同一历史第三次检验，不是独立 forward。"
    )
    payload = {"status": spa.get("status") or "DONE", "spa": spa, "round16_spa_p": R16_SPA_P}
    dump_json(SPA_JSON, payload)
    return payload


def run_stress(*, resume: bool = True) -> dict[str, Any]:
    existing = load_json(STRESS_JSON)
    if resume and existing.get("status") == "DONE":
        return existing
    out: dict[str, Any] = {}
    for code, ov_fallback in ((A_C0, cached_overlay_v4()), (A_C1, None)):
        path = overlayed_path(code)
        if path.is_file():
            ov = pd.read_parquet(path)
        elif ov_fallback is not None:
            ov = ov_fallback
        else:
            out[code] = {"status": "NO_OVERLAY", "skip": True}
            continue
        rec = {}
        for bps in COST_BPS:
            dest = STRESS_JSON.parent / f"{code}_plus{int(bps)}bp"
            ran = run_overlayed(
                dest,
                ov,
                extra={"code": code, "extra_buy_bps": bps, "fee_schedule": "HISTORICAL_FEE_CORRECTED"},
                resume=resume,
                write=True,
                extra_buy_bps=bps,
            )
            rec[f"plus{int(bps)}"] = ran.get("observed") or pack_obs_from_dir(dest)
        out[code] = rec

    def _cagr(code, key):
        return ((out.get(code) or {}).get(key) or {}).get("cagr")

    pack0 = pack_obs_from_dir(parity_dir(A_C0))
    pack1 = pack_obs_from_dir(parity_dir(A_C1))
    base_sign = 1 if ((pack1.get("cagr") or 0) - (pack0.get("cagr") or 0)) > 0 else -1
    sign_flip = {}
    for key in ("plus20", "plus30"):
        v0, v1 = _cagr(A_C0, key), _cagr(A_C1, key)
        if v0 is None or v1 is None:
            sign_flip[key] = None
        else:
            sign_flip[key] = bool((1 if (float(v1) - float(v0)) > 0 else -1) != base_sign)
    payload = {
        "status": "DONE",
        "results": out,
        "sign_flip": sign_flip,
        "note": "主评价 HISTORICAL_FEE_CORRECTED。不得重复计算已含 slippage。",
    }
    dump_json(STRESS_JSON, payload)
    return payload


def run_stats(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    existing = resume_ok(STAT_JSON) if resume else None
    if existing is not None:
        return existing
    try:
        _, n0 = load_nav_trades(require_c0())
        _, n1 = load_nav_trades(require_c1())
    except FileNotFoundError as exc:
        payload = skip_result("a-stats", str(exc), status="BLOCKED")
        dump_json(STAT_JSON, payload)
        return payload
    d0 = daily_from_nav(n0)
    d1 = daily_from_nav(n1)
    dmap = {A_C0: d0, A_C1: d1}
    d3 = None
    eq = load_json(EQ_JSON)
    c3_dir = eq.get("c3", {}).get("out_dir")
    if c3_dir:
        from pathlib import Path

        p = Path(c3_dir) / "nav.parquet"
        if p.is_file():
            d3 = daily_from_nav(pd.read_parquet(p))
            dmap[A_C3] = d3
    aligned = pd.concat({"c0": d0, "c1": d1}, axis=1).dropna()
    diff = (aligned["c1"] - aligned["c0"]).to_numpy()
    hac = hac_mean_pvalue(diff, lags=int(BLOCK_DAYS))
    boot = run_bootstrap(dmap)
    spa = run_spa_local(d0, d1, d3)
    loyo = _loyo(d0, d1)
    try:
        stress = run_stress(resume=resume)
    except Exception as exc:
        stress = skip_result("a-stats-stress", f"{type(exc).__name__}:{exc}")
    placebo = load_json(PLACEBO_JSON)
    conc = load_json(CONC_JSON)
    payload = {
        "status": "DONE",
        "bootstrap": {k: boot.get(k) for k in ("n", "pairs", "requested", "block_days")},
        "hac": hac,
        "spa": spa.get("spa"),
        "round16_spa_p": R16_SPA_P,
        "loyo": loyo,
        "cvar05": {"c0": _cvar(d0), "c1": _cvar(d1)},
        "stress_sign_flip": (stress.get("sign_flip") if isinstance(stress, dict) else None),
        "placebo_n": placebo.get("n"),
        "placebo_p_sharpe": placebo.get("p_empirical_sharpe"),
        "concentration_depends_few": conc.get("depends_on_few_extremes"),
        "note": "SPA 0.496 来自 Round16 family，必须披露。本轮不是独立 forward。",
    }
    dump_json(STAT_JSON, payload)
    write_md(
        ta_md("stats.md"),
        "\n".join(
            [
                "# Round19 Track A · a-stats",
                "",
                f"bootstrap N=`{boot.get('n')}` HAC p=`{hac.get('p_two_sided')}` SPA p=`{(spa.get('spa') or {}).get('p_spa')}` Round16 SPA=`{R16_SPA_P}`。",
                f"LOYO sign_flip=`{loyo.get('n_sign_flip')}` CVaR5 C0=`{payload['cvar05']['c0']}` C1=`{payload['cvar05']['c1']}`。",
                f"+20/+30 sign_flip=`{payload['stress_sign_flip']}` C4 p=`{placebo.get('p_empirical_sharpe')}`。",
                "",
            ]
        ),
    )
    return payload
