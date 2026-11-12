"""日频截面 IC / RankIC / HAC / 块 bootstrap。薄日跳过，不把 IC 记 0。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from .contract import BLOCK_DAYS, BOOT_N, IC_MIN_VALID_NAMES_PER_DAY


def finite_pair(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray] | None:
    a = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < IC_MIN_VALID_NAMES_PER_DAY:
        return None
    a, b = a[mask], b[mask]
    if int(np.unique(a).size) < 2 or int(np.unique(b).size) < 2:
        return None
    return a, b


def pearson_ic(x: pd.Series, y: pd.Series) -> float | None:
    pair = finite_pair(x, y)
    if pair is None:
        return None
    a, b = pair
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if den < 1e-18:
        return None
    val = float(np.dot(a, b) / den)
    return val if np.isfinite(val) else None


def rank_ic(x: pd.Series, y: pd.Series) -> float | None:
    pair = finite_pair(x, y)
    if pair is None:
        return None
    a = pd.Series(pair[0]).rank(method="average")
    b = pd.Series(pair[1]).rank(method="average")
    return pearson_ic(a, b)


def hac_mean(x: np.ndarray, lags: int = BLOCK_DAYS) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < lags + 5:
        return {"status": "TOO_SHORT", "n": int(len(x)), "lags": int(lags)}
    mu = float(x.mean())
    n = len(x)
    xc = x - mu
    gamma0 = float(np.dot(xc, xc) / n)
    acc = gamma0
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        acc += 2.0 * w * float(np.dot(xc[lag:], xc[:-lag]) / n)
    se = float(np.sqrt(max(acc, 1e-18) / n))
    z = mu / se if se > 0 else np.nan
    p = float(2.0 * (1.0 - norm.cdf(abs(z)))) if np.isfinite(z) else None
    return {
        "status": "DONE",
        "mean": mu,
        "hac_se": se,
        "z": float(z) if np.isfinite(z) else None,
        "p_two_sided": p,
        "lags": int(lags),
        "n": int(n),
    }


def block_bootstrap_mean(
    x: np.ndarray,
    *,
    block: int = BLOCK_DAYS,
    n: int = BOOT_N,
    seed: int = 17,
) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    t = len(x)
    if t < block * 2:
        return {"status": "TOO_SHORT", "n_obs": int(t), "n": int(n), "block": int(block)}
    n_blocks = int(np.ceil(t / block))
    rng = np.random.default_rng(int(seed))
    starts = rng.integers(0, t - block + 1, size=(int(n), n_blocks), dtype=np.int64)
    offs = np.arange(block, dtype=np.int64)
    idx = (starts[..., None] + offs).reshape(int(n), -1)[:, :t]
    samples = x[idx].mean(axis=1)
    return {
        "status": "DONE",
        "n": int(n),
        "block": int(block),
        "n_obs": int(t),
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "p05": float(np.percentile(samples, 5)),
        "p95": float(np.percentile(samples, 95)),
        "p_mean_gt0": float(np.mean(samples > 0)),
        "p_mean_lt0": float(np.mean(samples < 0)),
    }


def icir_of(x: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return None
    sd = float(np.std(x, ddof=1))
    if sd < 1e-18:
        return None
    return float(np.mean(x) / sd)


def same_sign_years(yearly: dict[str, float | None], years: tuple[int, ...]) -> int:
    signs = []
    for y in years:
        v = yearly.get(str(int(y)))
        if v is None or not np.isfinite(v) or v == 0:
            continue
        signs.append(float(np.sign(v)))
    return int(abs(sum(signs))) if signs else 0


def qcut_labels(values: np.ndarray, n_bins: int) -> np.ndarray | None:
    s = pd.Series(values)
    if s.nunique(dropna=True) < 2:
        return None
    try:
        q = pd.qcut(s, n_bins, labels=False, duplicates="drop")
    except ValueError:
        return None
    out = q.to_numpy()
    if np.all(pd.isna(q)):
        return None
    return out.astype(float)


def seed_for(*parts: object) -> int:
    import hashlib

    blob = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    return 17 + (int(digest[:8], 16) % 100_000)


__all__ = [
    "block_bootstrap_mean",
    "finite_pair",
    "hac_mean",
    "icir_of",
    "pearson_ic",
    "qcut_labels",
    "rank_ic",
    "same_sign_years",
    "seed_for",
]
