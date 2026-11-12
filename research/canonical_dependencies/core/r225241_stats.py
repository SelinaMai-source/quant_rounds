"""预注册统计。区间不是后验。SPA 不用标签，剔除基线并去掉完全相同的净值路径。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

BLOCK = 20
BOOT_N = 5000
SCOPES = (
    "FULL_HISTORY_FIRST_TO_SECOND",
    "STRICT_PIT_SUBPERIOD_FIRST_TO_SECOND",
    "SECOND_TO_THIRD_INDEPENDENT",
    "EXECUTION_SCENARIO",
    "DIAGNOSTIC_ONLY",
)


def assert_scope(scope: str) -> str:
    if scope not in SCOPES:
        raise ValueError(f"illegal evaluation scope: {scope}")
    return scope


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    width = max(int(block), 1)
    n_blocks = int(np.ceil(n / width))
    starts = rng.integers(0, n, size=n_blocks)
    pieces = [np.arange(int(s), int(s) + width) % n for s in starts]
    return np.concatenate(pieces)[:n]


def paired_block_bootstrap(
    left: pd.Series,
    right: pd.Series,
    *,
    n_boot: int = BOOT_N,
    block: int = BLOCK,
    seed: int = 17,
) -> dict[str, Any]:
    """配对日收益的块自助。返回区间，不叫后验。"""
    pair = pd.concat([left.rename("left"), right.rename("right")], axis=1, join="inner").dropna()
    if len(pair) < block * 3:
        return {"status": "NOT_IDENTIFIABLE", "n": int(len(pair)), "reason": "series shorter than 3 blocks"}
    diff = (pair["left"] - pair["right"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        take = block_indices(len(diff), block, rng)
        means[i] = float(diff[take].mean())
    point = float(diff.mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "status": "OK",
        "n": int(len(diff)),
        "block": int(block),
        "n_boot": int(n_boot),
        "mean_excess": point,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "share_boot_mean_gt_0": float(np.mean(means > 0)),
        "interpretation": "block bootstrap interval, not a posterior",
    }


def reality_check_spa(
    returns: pd.DataFrame,
    *,
    baseline: str,
    n_boot: int = BOOT_N,
    block: int = BLOCK,
    seed: int = 17,
) -> dict[str, Any]:
    """白氏现实检验式块自助。排除基线，去掉与基线或彼此完全相同的净值日收益。"""
    if baseline not in returns.columns:
        return {"status": "NOT_IDENTIFIABLE", "reason": "missing baseline"}
    frame = returns.dropna(how="any").astype(float)
    if len(frame) < block * 3:
        return {"status": "NOT_IDENTIFIABLE", "reason": "short return panel", "n": int(len(frame))}
    base = frame[baseline]
    names = []
    kept = []
    for col in frame.columns:
        if col == baseline:
            continue
        series = frame[col]
        if float(np.max(np.abs(series.to_numpy() - base.to_numpy()))) <= 1e-12:
            continue
        duplicate = False
        for prev in kept:
            if float(np.max(np.abs(series.to_numpy() - prev))) <= 1e-12:
                duplicate = True
                break
        if duplicate:
            continue
        names.append(col)
        kept.append(series.to_numpy(dtype=float))
    if not names:
        return {"status": "NOT_IDENTIFIABLE", "reason": "no distinct alternative NAV", "n": int(len(frame))}
    excess = np.column_stack(kept) - base.to_numpy(dtype=float)[:, None]
    n, k = excess.shape
    mean = excess.mean(axis=0)
    std = excess.std(axis=0, ddof=1)
    usable = std > 1e-12
    if not bool(np.any(usable)):
        return {"status": "NOT_IDENTIFIABLE", "reason": "zero variance differentials", "candidates": names}
    mean = mean[usable]
    std = std[usable]
    excess = excess[:, usable]
    names = [name for name, ok in zip(names, usable) if ok]
    stat = np.sqrt(n) * mean / std
    sample = float(np.max(stat))
    centered = excess - mean
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        take = block_indices(n, block, rng)
        draw = centered[take]
        bmean = draw.mean(axis=0)
        bstd = draw.std(axis=0, ddof=1)
        bstd = np.where(bstd > 1e-12, bstd, np.nan)
        bstat = np.sqrt(n) * bmean / bstd
        boot[i] = float(np.nanmax(bstat))
    pvalue = float(np.mean(boot >= sample))
    best = names[int(np.argmax(stat))]
    return {
        "status": "OK",
        "method": "WHITE_REALITY_CHECK_BLOCK_BOOT",
        "not_a_posterior": True,
        "baseline_excluded": baseline,
        "candidates": names,
        "best_candidate": best,
        "sample_stat": sample,
        "p_value": pvalue,
        "n": int(n),
        "block": int(block),
        "n_boot": int(n_boot),
        "used_labels": False,
    }


def ndcg_at_k(frame: pd.DataFrame, *, k: int, grade: str = "grade", score: str = "score", date: str = "signal_date") -> dict[str, Any]:
    from sklearn.metrics import ndcg_score

    scores: list[float] = []
    n_days = 0
    for _, block in frame.groupby(date, sort=True):
        block = block.dropna(subset=[grade, score])
        if len(block) < 2:
            continue
        n_days += 1
        y_true = block[grade].to_numpy(dtype=float)[None, :]
        y_score = block[score].to_numpy(dtype=float)[None, :]
        if np.nanmax(y_true) <= 0:
            scores.append(0.0 if np.allclose(y_true, y_true[0, 0]) else float(ndcg_score(y_true, y_score, k=k)))
            continue
        scores.append(float(ndcg_score(y_true, y_score, k=k)))
    if not scores:
        return {"status": "NOT_IDENTIFIABLE", "k": k, "n_days": 0}
    return {"status": "OK", "k": int(k), "n_days": int(n_days), "ndcg": float(np.mean(scores))}


def grade_edges(values: np.ndarray, n_grades: int = 5) -> np.ndarray:
    clean = values[np.isfinite(values)]
    if len(clean) < n_grades * 5:
        return np.array([], dtype=float)
    probs = np.linspace(0, 1, n_grades + 1)[1:-1]
    edges = np.unique(np.quantile(clean, probs))
    return edges.astype(float)


def apply_grades(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(edges) == 0:
        return out
    mask = np.isfinite(values)
    out[mask] = np.digitize(values[mask], edges)
    return out


def topk_utility(selected_net: np.ndarray) -> dict[str, float | None]:
    clean = selected_net[np.isfinite(selected_net)]
    if len(clean) == 0:
        return {"o3": None, "o4": None, "n": 0}
    downside = np.maximum(0.0, -clean)
    o3 = float(clean.mean())
    o4 = float(o3 - 0.5 * downside.mean())
    return {"o3": o3, "o4": o4, "n": int(len(clean))}


def daily_spearman(frame: pd.DataFrame, left: str, right: str, date: str = "signal_date") -> float | None:
    vals = []
    for _, block in frame.groupby(date, sort=False):
        if len(block) < 5:
            continue
        a = pd.to_numeric(block[left], errors="coerce")
        b = pd.to_numeric(block[right], errors="coerce")
        ok = a.notna() & b.notna()
        if int(ok.sum()) < 5 or a[ok].nunique() < 2 or b[ok].nunique() < 2:
            continue
        vals.append(float(a[ok].corr(b[ok], method="spearman")))
    if not vals:
        return None
    return float(np.mean(vals))


def c4_window_counts(sign: pd.Series) -> dict[str, int]:
    """复现 shift(1).rolling(5, min_periods=3)，按行滚动，不插入日历空档。"""
    shifted = pd.to_numeric(sign, errors="coerce").shift(1)
    counts = shifted.rolling(5, min_periods=3).count()
    valid = counts[shifted.rolling(5, min_periods=3).mean().notna()]
    return {
        "n_valid_3": int((valid == 3).sum()),
        "n_valid_4": int((valid == 4).sum()),
        "n_valid_5": int((valid == 5).sum()),
        "n_valid": int(valid.shape[0]),
    }


def rolling_uses_calendar_index(index: pd.DatetimeIndex) -> bool:
    """若窗口按行而不是按日历天，缺日不会被插进窗口。"""
    series = pd.Series(np.arange(len(index), dtype=float), index=index)
    rolled = series.rolling("5D").count()
    row = series.rolling(5, min_periods=3).count()
    return not bool(np.allclose(rolled.fillna(-1), row.fillna(-1)))


def permute_within_date(frame: pd.DataFrame, columns: list[str], date: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for _, block in frame.groupby(date, sort=False):
        order = np.arange(len(block))
        rng.shuffle(order)
        shuffled = block.copy()
        shuffled[columns] = block[columns].to_numpy()[order]
        parts.append(shuffled)
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()


def ranking_groups_contiguous(dates: pd.Series) -> bool:
    values = pd.to_datetime(dates).to_numpy()
    if len(values) == 0:
        return False
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(values)]
    seen = set()
    for start, end in zip(starts, ends):
        key = values[start]
        if key in seen:
            return False
        if not np.all(values[start:end] == key):
            return False
        seen.add(key)
    return True


def purge_mask(label_end: pd.Series, inner_start: pd.Timestamp) -> pd.Series:
    end = pd.to_datetime(pd.Series(label_end))
    return end.notna() & (end <= pd.Timestamp(inner_start))
