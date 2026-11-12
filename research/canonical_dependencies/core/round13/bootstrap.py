"""Block bootstrap、CPCV、Family SPA。复用 R11 函数，不新发明 CPCV。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND13_ROOT
from ..round11.common import sharpe_of
from ..round11.stats import cpcv_oos_sharpe, spa_test
from .candidates import load_daily_map
from .common import cagr_of, dump_json, load_json, write_md
from .contract import BLOCK_DAYS, BOOT_N, BOOT_N_MIN, CANDIDATE_IDS, SPA_BOOT, STAT_IDS
from .freeze import require_frozen

BOOT_JSON = ROUND13_ROOT / "bootstrap" / "bootstrap.json"
BOOT_MD = ROUND13_ROOT / "04_bootstrap.md"
CPCV_JSON = ROUND13_ROOT / "cpcv" / "cpcv.json"
CPCV_MD = ROUND13_ROOT / "05_cpcv.md"
SPA_JSON = ROUND13_ROOT / "spa" / "spa.json"
SPA_MD = ROUND13_ROOT / "06_spa.md"
IDS = list(STAT_IDS)


def _block_starts(n: int, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n / block))
    return rng.integers(0, max(n - block, 1) + (0 if n > block else 0), size=(n_boot, n_blocks))


def _resample(mat: np.ndarray, starts: np.ndarray, block: int) -> np.ndarray:
    n = mat.shape[0]
    pieces = [mat[s : s + block] for s in starts]
    sample = np.concatenate(pieces, axis=0)[:n]
    return sample


def run_bootstrap(*, resume: bool = True, n_boot: int | None = None, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and BOOT_JSON.is_file():
        existing = load_json(BOOT_JSON)
        if existing.get("status") == "DONE" and int(existing.get("n") or 0) >= BOOT_N_MIN:
            return existing
    dailies = load_daily_map()
    aligned = pd.concat({k: dailies[k] for k in IDS if k in dailies}, axis=1).dropna()
    n = len(aligned)
    n_use = int(n_boot or BOOT_N)
    if n < BLOCK_DAYS * 4:
        raise RuntimeError("bootstrap series too short")
    rng = np.random.default_rng(20261012)
    try:
        starts = rng.integers(0, n - BLOCK_DAYS + 1, size=(n_use, int(np.ceil(n / BLOCK_DAYS))))
    except ValueError:
        n_use = BOOT_N_MIN
        starts = rng.integers(0, n - BLOCK_DAYS + 1, size=(n_use, int(np.ceil(n / BLOCK_DAYS))))
    mat = aligned.to_numpy(dtype=float)
    cols = list(aligned.columns)
    sharpe_boot = np.empty((n_use, len(cols)), dtype=float)
    cagr_boot = np.empty((n_use, len(cols)), dtype=float)
    for i, row in enumerate(starts):
        sample = _resample(mat, row, BLOCK_DAYS)
        for j in range(len(cols)):
            x = sample[:, j]
            vol = float(x.std(ddof=1))
            sharpe_boot[i, j] = float(x.mean() / vol * np.sqrt(252)) if vol > 0 else np.nan
            wealth = float(np.prod(1.0 + x))
            years = max(len(x) / 252.0, 1e-9)
            cagr_boot[i, j] = wealth ** (1.0 / years) - 1.0

    def _rank_stats(boot: np.ndarray) -> dict[str, Any]:
        ranks = pd.DataFrame(boot, columns=cols).rank(axis=1, ascending=False, method="average")
        out = {}
        for c in cols:
            r = ranks[c].to_numpy()
            v = boot[:, cols.index(c)]
            out[c] = {
                "rank_mean": float(np.nanmean(r)),
                "rank_std": float(np.nanstd(r, ddof=1)),
                "p10": float(np.nanpercentile(v, 10)),
                "p90": float(np.nanpercentile(v, 90)),
                "top1_freq": float(np.nanmean(r <= 1.0 + 1e-9)),
                "top3_freq": float(np.nanmean(r <= 3.0 + 1e-9)),
            }
        return out

    payload = {
        "status": "DONE",
        "n": n_use,
        "block": BLOCK_DAYS,
        "n_days": n,
        "ids": cols,
        "sharpe": _rank_stats(sharpe_boot),
        "cagr": _rank_stats(cagr_boot),
        "note": "20 日块 bootstrap。N 目标 5000，下限 2000。",
    }
    dump_json(BOOT_JSON, payload)
    lines = [
        "# Round 13 · Block bootstrap",
        "",
        f"N=**{n_use}** block={BLOCK_DAYS} 日。",
        "",
        "| ID | Sharpe rank mean | rank std | p10 | p90 | top1 | top3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cols:
        r = payload["sharpe"][c]
        lines.append(
            f"| `{c}` | {r['rank_mean']:.3f} | {r['rank_std']:.3f} | {r['p10']:.4f} | {r['p90']:.4f} | "
            f"{r['top1_freq']:.3f} | {r['top3_freq']:.3f} |"
        )
    write_md(BOOT_MD, "\n".join(lines) + "\n")
    return payload


def run_cpcv(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and CPCV_JSON.is_file():
        existing = load_json(CPCV_JSON)
        if existing.get("status") == "DONE":
            return existing
    dailies = load_daily_map()
    by_id = {}
    for sid in IDS:
        if sid not in dailies:
            continue
        raw = cpcv_oos_sharpe(dailies[sid])
        # extras from the same contract: recompute combo list stats if raw has mean/median
        extra = {}
        if raw.get("status") == "DONE":
            import itertools
            from ..round11.stats import EMBARGO, MIN_BLOCK, _blocks

            daily = dailies[sid].dropna().sort_index()
            blocks = _blocks(daily.index)
            combos = list(itertools.combinations(range(len(blocks)), 2))
            oos = []
            for test_idx in combos:
                test_index = blocks[test_idx[0]]
                for j in test_idx[1:]:
                    test_index = test_index.append(blocks[j])
                sh = sharpe_of(daily.loc[test_index])
                if sh is not None:
                    oos.append(sh)
            arr = np.array(oos, dtype=float)
            extra = {
                "oos_sharpe_p10": float(np.nanpercentile(arr, 10)) if len(arr) else None,
                "oos_sharpe_worst": float(np.nanmin(arr)) if len(arr) else None,
                "share_sharpe_gt0": float(np.mean(arr > 0)) if len(arr) else None,
                "share_sharpe_gt1": float(np.mean(arr > 1.0)) if len(arr) else None,
                "embargo_days": EMBARGO,
                "min_block": MIN_BLOCK,
            }
        by_id[sid] = {**raw, **extra}
    payload = {
        "status": "DONE",
        "not_forward_oos": True,
        "by_id": by_id,
        "note": "原样 round11.cpcv_oos_sharpe；额外报 p10/worst 与 Sharpe>0/>1 比例。不是 forward OOS。",
    }
    dump_json(CPCV_JSON, payload)
    lines = [
        "# Round 13 · CPCV",
        "",
        "embargo=12，块≥60。**不是 forward OOS**。",
        "",
        "| ID | mean | median | p10 | worst | P(SR>0) | P(SR>1) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sid, row in by_id.items():
        lines.append(
            f"| `{sid}` | {row.get('oos_sharpe_mean')} | {row.get('oos_sharpe_median')} | "
            f"{row.get('oos_sharpe_p10')} | {row.get('oos_sharpe_worst')} | "
            f"{row.get('share_sharpe_gt0')} | {row.get('share_sharpe_gt1')} |"
        )
    write_md(CPCV_MD, "\n".join(lines) + "\n")
    return payload


def run_spa(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and SPA_JSON.is_file():
        existing = load_json(SPA_JSON)
        if existing.get("status") == "DONE":
            return existing
    dailies = load_daily_map()
    bench = dailies["v4"]
    alts = pd.concat({k: dailies[k] for k in CANDIDATE_IDS if k in dailies}, axis=1)
    family = spa_test(bench, alts, n_boot=SPA_BOOT, seed=17)
    aligned = pd.concat([bench.rename("b"), alts], axis=1).dropna()
    d = aligned.drop(columns=["b"]).sub(aligned["b"], axis=0)
    mean_d = d.mean()
    rng = np.random.default_rng(17)
    n = len(d)
    block = BLOCK_DAYS
    n_blocks = int(np.ceil(n / block))
    boot = []
    obs_means = mean_d.to_numpy()
    center = np.maximum(obs_means, 0.0)
    for _ in range(SPA_BOOT):
        starts = rng.integers(0, max(n - block, 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        sample = d.to_numpy()[idx]
        star = sample.mean(axis=0) - center
        boot.append(star)
    boot = np.vstack(boot)
    boot_max = boot.max(axis=1)
    obs_max = float(obs_means.max())
    p_hansen = float(np.mean(boot_max >= obs_max - np.maximum(obs_means.max(), 0.0) + obs_max * 0))
    # consistent Hansen: compare boot max of (mean* - max(mean,0)) vs max(mean)
    p_hansen = float(np.mean(boot_max >= obs_max))
    raw = {}
    for i, col in enumerate(d.columns):
        raw[col] = {
            "mean_delta": float(obs_means[i]),
            "raw_p": float(np.mean(boot[:, i] >= obs_means[i])),
        }
    # family-adjusted: SPA p as the family test; Holm on raw
    ps = sorted(((k, v["raw_p"]) for k, v in raw.items()), key=lambda x: x[1])
    m = len(ps)
    holm = {}
    for i, (k, p) in enumerate(ps):
        holm[k] = min(1.0, (m - i) * p)
    payload = {
        "status": "DONE",
        "h0": "A01–A05 相对 corrected v4 无更优",
        "spa_test_reuse": family,
        "hansen_recentered": {
            "observed_max_mean_delta": obs_max,
            "p_spa": p_hansen,
            "n_boot": SPA_BOOT,
            "n_alts": int(d.shape[1]),
        },
        "raw": raw,
        "holm_adjusted_p": holm,
        "family_p": p_hansen,
        "note": "Hansen-style block bootstrap SPA + Holm。不是因果证明。",
    }
    dump_json(SPA_JSON, payload)
    lines = [
        "# Round 13 · Family SPA",
        "",
        f"H0 = A01–A05 相对 corrected v4 无更优。Hansen p=**{p_hansen}**。reuse spa_test p={family.get('p_spa')}。",
        "",
        "| ID | mean delta | raw p | Holm p |",
        "|---|---:|---:|---:|",
    ]
    for col in d.columns:
        lines.append(f"| `{col}` | {raw[col]['mean_delta']} | {raw[col]['raw_p']} | {holm.get(col)} |")
    write_md(SPA_MD, "\n".join(lines) + "\n")
    return payload


def load_bootstrap() -> dict[str, Any]:
    return load_json(BOOT_JSON)


def load_cpcv() -> dict[str, Any]:
    return load_json(CPCV_JSON)


def load_spa() -> dict[str, Any]:
    return load_json(SPA_JSON)


__all__ = ["load_bootstrap", "load_cpcv", "load_spa", "run_bootstrap", "run_cpcv", "run_spa"]
