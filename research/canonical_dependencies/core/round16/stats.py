"""20 日 block bootstrap N=5000（最低 2000）、HAC、CPCV、family SPA、LOYO。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_ROOT
from ..round11.common import daily_from_nav, sharpe_of
from ..round11.stats import cpcv_oos_sharpe, spa_test
from .backtest import load_backtest
from .common import dump_json, load_json, write_md
from .contract import BLOCK_DAYS, BOOT_N, BOOT_N_MIN, OPTION_IDS, R16_00, SPA_BOOT, experiment_dir, parity_dir
from .freeze import require_frozen

STAT_JSON = ROUND16_ROOT / "stats" / "stats.json"
BOOT_JSON = ROUND16_ROOT / "bootstrap" / "bootstrap.json"
CPCV_JSON = ROUND16_ROOT / "cpcv" / "cpcv.json"
SPA_JSON = ROUND16_ROOT / "spa" / "spa.json"
STAT_MD = ROUND16_ROOT / "11_statistical_validation.md"


def _nav_path(code: str) -> Path | None:
    if code in {R16_00, "v4"}:
        p = parity_dir(R16_00) / "nav.parquet"
        return p if p.is_file() else None
    p = experiment_dir(code) / "nav.parquet"
    return p if p.is_file() else None


def load_daily_map() -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for code in ("v4", *OPTION_IDS):
        path = _nav_path(code)
        if path is None:
            continue
        out[code] = daily_from_nav(pd.read_parquet(path))
    return out


def _resample(mat: np.ndarray, starts: np.ndarray, block: int) -> np.ndarray:
    n = mat.shape[0]
    pieces = [mat[s : s + block] for s in starts]
    return np.concatenate(pieces, axis=0)[:n]


def _hac_mean(diff: pd.Series, lags: int = 20) -> dict[str, Any]:
    x = diff.dropna().to_numpy(float)
    if len(x) < lags + 5:
        return {"n": int(len(x)), "mean": None, "t_hac": None}
    mean = float(x.mean())
    n = len(x)
    gamma0 = float(np.dot(x - mean, x - mean) / n)
    acc = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1)
        ck = float(np.dot(x[k:] - mean, x[:-k] - mean) / n)
        acc += 2 * w * ck
    se = float(np.sqrt(max(acc, 0.0) / n))
    t = mean / se if se > 0 else None
    return {"n": n, "mean": mean, "t_hac": t, "lags": lags}


def run_stats(**_: Any) -> dict[str, Any]:
    require_frozen()
    dmap = load_daily_map()
    bt = load_backtest()
    ids = ["v4"]
    for cid in OPTION_IDS:
        row = (bt.get("results") or {}).get(cid) or {}
        if not row.get("skip") and row.get("cagr") is not None and cid in dmap:
            ids.append(cid)
    boot_n = int(BOOT_N)
    cols = [c for c in ids if c in dmap]
    if len(cols) < 1:
        payload = {"status": "SKIP_NO_ALTS", "ids": cols}
        dump_json(STAT_JSON, payload)
        dump_json(BOOT_JSON, payload)
        dump_json(CPCV_JSON, payload)
        dump_json(SPA_JSON, payload)
        write_md(STAT_MD, "# Round16 · 11 统计\n\nSKIP_NO_ALTS。\n")
        return payload
    aligned = pd.concat({c: dmap[c] for c in cols}, axis=1).dropna()
    mat = aligned.to_numpy(float)
    n = mat.shape[0]
    block = int(BLOCK_DAYS)
    n_blocks = max(1, n - block + 1)
    rng = np.random.default_rng(20261016)
    n_eff = boot_n
    sharpes = []
    try:
        for i in range(boot_n):
            starts = rng.integers(0, n_blocks, size=int(np.ceil(n / block)))
            resampled = _resample(mat, starts, block)
            sh = []
            for j in range(resampled.shape[1]):
                s = pd.Series(resampled[:, j])
                sh.append(sharpe_of(s))
            sharpes.append(sh)
            if i + 1 == BOOT_N_MIN:
                n_eff = max(n_eff, BOOT_N_MIN)
    except MemoryError:
        n_eff = max(len(sharpes), BOOT_N_MIN)
        sharpes = sharpes[:n_eff]
    boot = np.array(sharpes, dtype=float)
    boot_payload = {
        "status": "DONE",
        "n": int(len(boot)),
        "requested": boot_n,
        "min_required": BOOT_N_MIN,
        "block_days": block,
        "ids": cols,
        "mean_sharpe": {c: float(np.nanmean(boot[:, i])) for i, c in enumerate(cols)},
        "p10": {c: float(np.nanpercentile(boot[:, i], 10)) for i, c in enumerate(cols)},
        "sample_size": n,
        "coverage": [str(aligned.index.min().date()), str(aligned.index.max().date())],
    }
    dump_json(BOOT_JSON, boot_payload)
    cpcv = {}
    for c in cols:
        rec = cpcv_oos_sharpe(dmap[c])
        rec["not_forward_oos"] = True
        rec["median"] = rec.get("oos_sharpe_median")
        cpcv[c] = rec
    dump_json(CPCV_JSON, {"status": "DONE", "by_id": cpcv, "not_forward": True})
    spa = {"status": "SKIP_NO_ALTS"}
    alts = [c for c in cols if c != "v4"]
    if len(alts) >= 1 and "v4" in dmap:
        try:
            alt_df = pd.concat({c: dmap[c] for c in alts}, axis=1)
            spa = spa_test(dmap["v4"], alt_df, n_boot=SPA_BOOT)
            spa["family"] = "option"
            spa["n_alts"] = len(alts)
        except Exception as exc:
            spa = {"status": "FAIL", "error": type(exc).__name__}
    dump_json(SPA_JSON, spa)
    hac = {}
    if "v4" in dmap:
        for c in alts:
            diff = dmap[c] - dmap["v4"]
            hac[c] = _hac_mean(diff)
    loyo = {}
    for c in cols:
        s = dmap[c]
        by = s.groupby(s.index.year).apply(lambda x: float((1 + x).prod() - 1))
        loyo[c] = {"median": float(by.median()) if len(by) else None, "worst": float(by.min()) if len(by) else None, "by_year": {str(k): float(v) for k, v in by.items()}}
    payload = {
        "status": "DONE",
        "bootstrap": boot_payload,
        "cpcv": {c: (cpcv[c].get("median") if isinstance(cpcv[c], dict) else None) for c in cols},
        "spa": spa,
        "hac": hac,
        "loyo": loyo,
        "ids": cols,
        "multiple_testing": "family SPA on option candidates vs v4",
    }
    dump_json(STAT_JSON, payload)
    write_md(
        STAT_MD,
        "\n".join([
            "# Round16 · 11 统计检验",
            "",
            f"bootstrap N=`{boot_payload.get('n')}` block=`{block}` ids=`{cols}`。",
            f"CPCV not_forward=`True`。SPA=`{(spa or {}).get('status')}`。",
            f"HAC=`{hac}`",
            "",
            "不得把 CPCV 当 forward OOS。",
            "",
        ]),
    )
    return payload


def load_stats() -> dict[str, Any]:
    return load_json(STAT_JSON)


def load_bootstrap() -> dict[str, Any]:
    return load_json(BOOT_JSON)


def load_cpcv() -> dict[str, Any]:
    return load_json(CPCV_JSON)


def load_spa() -> dict[str, Any]:
    return load_json(SPA_JSON)
