"""配对统计：Delta Sharpe = Sharpe_c - Sharpe_b；SPA 优先 arch。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

BLOCK = 20
BOOT_N = 5000


def _sharpe(returns: np.ndarray) -> float | None:
    if len(returns) < 2:
        return None
    vol = float(np.std(returns, ddof=1))
    if vol <= 0:
        return None
    return float(np.mean(returns) / vol * np.sqrt(252.0))


def _cagr(returns: np.ndarray, years: float) -> float | None:
    if years <= 0 or len(returns) == 0:
        return None
    wealth = float(np.prod(1.0 + returns))
    if wealth <= 0:
        return None
    return float(wealth ** (1.0 / years) - 1.0)


def _maxdd(returns: np.ndarray) -> float | None:
    if len(returns) == 0:
        return None
    nav = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    return float(dd.min())


def delta_sharpe_of_paths(candidate: np.ndarray, baseline: np.ndarray) -> float | None:
    sc = _sharpe(candidate)
    sb = _sharpe(baseline)
    if sc is None or sb is None:
        return None
    return float(sc - sb)


def paired_block_bootstrap_v2(
    candidate: pd.Series,
    baseline: pd.Series,
    *,
    n_boot: int = BOOT_N,
    block: int = BLOCK,
    seed: int = 17,
    calendar: pd.DatetimeIndex | None = None,
) -> dict[str, Any]:
    left = candidate.copy()
    right = baseline.copy()
    left.index = pd.to_datetime(left.index)
    right.index = pd.to_datetime(right.index)
    if calendar is not None:
        cal = pd.DatetimeIndex(pd.to_datetime(calendar))
        missing_c = [str(d.date()) for d in cal.difference(left.index)]
        missing_b = [str(d.date()) for d in cal.difference(right.index)]
        if missing_c or missing_b:
            return {
                "status": "BLOCKED",
                "reason": "calendar_mismatch",
                "missing_candidate_dates": missing_c[:50],
                "missing_baseline_dates": missing_b[:50],
                "n_missing_candidate": len(missing_c),
                "n_missing_baseline": len(missing_b),
            }
        left = left.reindex(cal)
        right = right.reindex(cal)
    pair = pd.concat([left.rename("c"), right.rename("b")], axis=1, join="inner")
    if pair.isna().any().any():
        return {
            "status": "BLOCKED",
            "reason": "na_in_aligned_calendar",
            "n": int(len(pair)),
            "n_na": int(pair.isna().any(axis=1).sum()),
            "note": "禁止 dropna 静默缩短评价窗",
        }
    c = pair["c"].to_numpy(dtype=float)
    b = pair["b"].to_numpy(dtype=float)
    n = len(pair)
    if n < block * 3:
        return {"status": "NOT_IDENTIFIABLE", "n": n, "reason": "series shorter than 3 blocks"}
    years = n / 252.0
    point = {
        "delta_mean": float(c.mean() - b.mean()),
        "delta_sharpe": delta_sharpe_of_paths(c, b),
        "delta_cagr": None if _cagr(c, years) is None or _cagr(b, years) is None else float(_cagr(c, years) - _cagr(b, years)),
        "delta_maxdd": None if _maxdd(c) is None or _maxdd(b) is None else float(_maxdd(c) - _maxdd(b)),
    }
    rng = np.random.default_rng(seed)
    boot = {k: np.empty(n_boot, dtype=float) for k in ("delta_mean", "delta_sharpe", "delta_cagr", "delta_maxdd")}
    width = max(int(block), 1)
    for i in range(n_boot):
        n_blocks = int(np.ceil(n / width))
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([np.arange(int(s), int(s) + width) % n for s in starts])[:n]
        bc, bb = c[idx], b[idx]
        boot["delta_mean"][i] = float(bc.mean() - bb.mean())
        ds = delta_sharpe_of_paths(bc, bb)
        boot["delta_sharpe"][i] = np.nan if ds is None else ds
        dc = _cagr(bc, years)
        db = _cagr(bb, years)
        boot["delta_cagr"][i] = np.nan if dc is None or db is None else float(dc - db)
        mdc, mdb = _maxdd(bc), _maxdd(bb)
        boot["delta_maxdd"][i] = np.nan if mdc is None or mdb is None else float(mdc - mdb)
    out = {
        "status": "OK",
        "n": n,
        "block": int(block),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "evaluation_years_fixed": years,
        "point": point,
        "ci95": {},
        "note": "Delta Sharpe = Sharpe_c - Sharpe_b；不是差额序列的 Sharpe",
    }
    for key, arr in boot.items():
        clean = arr[np.isfinite(arr)]
        if len(clean) == 0:
            out["ci95"][key] = None
            continue
        lo, hi = np.quantile(clean, [0.025, 0.975])
        out["ci95"][key] = {"low": float(lo), "high": float(hi)}
    return out


def spa_hansen_or_blocked(
    returns: pd.DataFrame,
    *,
    baseline: str,
    reps: int = 5000,
    block_size: int = 20,
    seed: int = 17,
) -> dict[str, Any]:
    if baseline not in returns.columns:
        return {"status": "BLOCKED_STAT_TEST", "reason": "missing baseline"}
    frame = returns.copy()
    if frame.isna().any().any():
        return {"status": "BLOCKED_STAT_TEST", "reason": "na_present_no_silent_dropna", "n_na_rows": int(frame.isna().any(axis=1).sum())}
    base = frame[baseline].to_numpy(dtype=float)
    names = []
    models = []
    for col in frame.columns:
        if col == baseline:
            continue
        series = frame[col].to_numpy(dtype=float)
        if float(np.max(np.abs(series - base))) <= 1e-12:
            continue
        if any(float(np.max(np.abs(series - prev))) <= 1e-12 for prev in models):
            continue
        names.append(col)
        models.append(series)
    if not names:
        return {"status": "NOT_IDENTIFIABLE", "reason": "no distinct alternatives"}
    try:
        from arch.bootstrap import SPA

        losses = np.column_stack([-m for m in models])
        bench = -base
        spa = SPA(bench, losses, reps=reps, block_size=block_size, studentize=True, seed=seed)
        spa.compute()
        return {
            "status": "OK",
            "method": "arch.bootstrap.SPA",
            "arch_version": __import__("arch").__version__,
            "bootstrap": "circular block",
            "reps": reps,
            "block_size": block_size,
            "studentize": True,
            "seed": seed,
            "candidates": names,
            "n_alternatives": len(names),
            "loss_definition": "negative daily net return",
            "pvalues": {
                "lower": float(spa.pvalues[0]),
                "consistent": float(spa.pvalues[1]),
                "upper": float(spa.pvalues[2]),
            },
        }
    except Exception as exc:
        return {
            "status": "BLOCKED_STAT_TEST",
            "reason": str(exc),
            "method_not_used": "self_written_reality_check_spa_is_not_hansen_spa",
            "candidates": names,
            "n_alternatives": len(names),
        }
