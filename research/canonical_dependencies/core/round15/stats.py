"""Block bootstrap、CPCV、Family SPA。期权一组、Top1 一组。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND15_ROOT
from ..round11.common import daily_from_nav, sharpe_of
from ..round11.stats import cpcv_oos_sharpe, spa_test
from .common import dump_json, load_json, write_md
from .contract import (
    ALPHA_IDS,
    BLOCK_DAYS,
    BOOT_N,
    BOOT_N_MIN,
    OPTION_IDS,
    R15_00,
    SPA_BOOT,
    TOP1_IDS,
    experiment_dir,
    parity_dir,
)
from .freeze import require_frozen
from .options import load_options
from .top1 import load_top1

STAT_JSON = ROUND15_ROOT / "stats" / "stats.json"
BOOT_JSON = ROUND15_ROOT / "bootstrap" / "bootstrap.json"
CPCV_JSON = ROUND15_ROOT / "cpcv" / "cpcv.json"
SPA_JSON = ROUND15_ROOT / "spa" / "spa.json"
BOOT_MD = ROUND15_ROOT / "10_bootstrap.md"
CPCV_MD = ROUND15_ROOT / "11_cpcv.md"
SPA_MD = ROUND15_ROOT / "12_spa.md"


def _nav_path(code: str) -> Path | None:
    if code in {R15_00, "v4"}:
        p = parity_dir(R15_00) / "nav.parquet"
        return p if p.is_file() else None
    p = experiment_dir(code) / "nav.parquet"
    if p.is_file():
        return p
    p2 = ROUND15_ROOT / "execution" / code / "nav.parquet"
    return p2 if p2.is_file() else None


def load_daily_map() -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for code in (R15_00, "v4", *ALPHA_IDS):
        path = _nav_path("v4" if code == "v4" else code)
        if path is None:
            continue
        key = "v4" if code in {R15_00, "v4"} else code
        if key in out:
            continue
        out[key] = daily_from_nav(pd.read_parquet(path))
    return out


def _resample(mat: np.ndarray, starts: np.ndarray, block: int) -> np.ndarray:
    n = mat.shape[0]
    pieces = [mat[s : s + block] for s in starts]
    return np.concatenate(pieces, axis=0)[:n]


def _rank_stats(boot: np.ndarray, cols: list[str]) -> dict[str, Any]:
    ranks = pd.DataFrame(boot, columns=cols).rank(axis=1, ascending=False, method="average")
    out = {}
    for c in cols:
        r = ranks[c].to_numpy()
        v = boot[:, cols.index(c)]
        out[c] = {
            "rank_mean": float(np.nanmean(r)),
            "rank_std": float(np.nanstd(r, ddof=1)) if len(r) > 1 else None,
            "p10": float(np.nanpercentile(v, 10)),
            "p90": float(np.nanpercentile(v, 90)),
            "top1_freq": float(np.nanmean(r <= 1.0 + 1e-9)),
        }
    return out


def _eligible_ids() -> list[str]:
    ids = ["v4"]
    opt = (load_options().get("results") or {})
    top = (load_top1().get("results") or {})
    for cid in OPTION_IDS:
        row = opt.get(cid) or {}
        if not row.get("skip") and row.get("cagr") is not None:
            ids.append(cid)
    for cid in TOP1_IDS:
        row = top.get(cid) or {}
        if not row.get("skip") and row.get("cagr") is not None:
            ids.append(cid)
    return ids


def run_stats(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and STAT_JSON.is_file():
        existing = load_json(STAT_JSON)
        if existing.get("status") == "DONE":
            return existing
    dailies = load_daily_map()
    ids = [k for k in _eligible_ids() if k in dailies]
    boot_payload: dict[str, Any]
    if len(ids) < 2:
        boot_payload = {
            "status": "SKIP_TOO_FEW",
            "n": 0,
            "ids": ids,
            "note": "除 v4 外无正式候选日收益，bootstrap/SPA 无 family 可测。",
        }
        dump_json(BOOT_JSON, boot_payload)
        write_md(BOOT_MD, "# Round 15 · Bootstrap\n\n除 v4 外无候选。SKIP。\n")
        cpcv_payload = {"status": "DONE", "not_forward_oos": True, "by_id": {}}
        if "v4" in dailies:
            raw = cpcv_oos_sharpe(dailies["v4"])
            extra = {}
            if raw.get("status") == "DONE":
                extra = {
                    "oos_sharpe_p10": raw.get("oos_sharpe_p05"),
                    "oos_sharpe_median": raw.get("oos_sharpe_median"),
                    "oos_sharpe_mean": raw.get("oos_sharpe_mean"),
                    "oos_sharpe_worst": raw.get("oos_sharpe_p05"),
                }
            cpcv_payload["by_id"]["v4"] = {**raw, **extra}
        dump_json(CPCV_JSON, cpcv_payload)
        write_md(CPCV_MD, "# Round 15 · CPCV\n\nembargo=12，块≥60。**不是 forward OOS**。仅 v4。\n")
        spa_payload = {
            "status": "SKIP_NO_ALTS",
            "option_family": {"status": "SKIP"},
            "top1_family": {"status": "SKIP"},
            "note": "无期权/Top1 正式候选，SPA 不跑。",
        }
        dump_json(SPA_JSON, spa_payload)
        write_md(SPA_MD, "# Round 15 · SPA\n\n无候选 family。SKIP。\n")
        payload = {"status": "DONE", "bootstrap": boot_payload, "cpcv": cpcv_payload, "spa": spa_payload}
        dump_json(STAT_JSON, payload)
        return payload

    aligned = pd.concat({k: dailies[k] for k in ids}, axis=1).dropna()
    n = len(aligned)
    n_use = BOOT_N if n >= BLOCK_DAYS * 4 else BOOT_N_MIN
    rng = np.random.default_rng(20261013)
    starts = rng.integers(0, max(n - BLOCK_DAYS, 1) + 1, size=(n_use, int(np.ceil(n / BLOCK_DAYS))))
    mat = aligned.to_numpy(dtype=float)
    cols = list(aligned.columns)
    sharpe_boot = np.empty((n_use, len(cols)), dtype=float)
    for i, row in enumerate(starts):
        sample = _resample(mat, row, BLOCK_DAYS)
        for j in range(len(cols)):
            x = sample[:, j]
            vol = float(x.std(ddof=1))
            sharpe_boot[i, j] = float(x.mean() / vol * np.sqrt(252)) if vol > 0 else np.nan
    boot_payload = {
        "status": "DONE",
        "n": n_use,
        "block": BLOCK_DAYS,
        "n_days": n,
        "ids": cols,
        "sharpe": _rank_stats(sharpe_boot, cols),
    }
    dump_json(BOOT_JSON, boot_payload)
    lines = ["# Round 15 · Block bootstrap", "", f"N=**{n_use}** block={BLOCK_DAYS}。", ""]
    write_md(BOOT_MD, "\n".join(lines))

    by_id = {}
    for sid in ids:
        raw = cpcv_oos_sharpe(dailies[sid])
        extra = {}
        if raw.get("status") == "DONE":
            extra = {
                "oos_sharpe_p10": raw.get("oos_sharpe_p05"),
                "oos_sharpe_worst": raw.get("oos_sharpe_p05"),
            }
        by_id[sid] = {**raw, **extra}
    cpcv_payload = {"status": "DONE", "not_forward_oos": True, "by_id": by_id}
    dump_json(CPCV_JSON, cpcv_payload)
    write_md(CPCV_MD, "# Round 15 · CPCV\n\nembargo=12，块≥60。**不是 forward OOS**。\n")

    bench = dailies["v4"]
    opt_alts = [c for c in OPTION_IDS if c in dailies]
    top_alts = [c for c in TOP1_IDS if c in dailies]
    option_spa = {"status": "SKIP_NO_ALTS"}
    top_spa = {"status": "SKIP_NO_ALTS"}
    if opt_alts:
        option_spa = spa_test(bench, pd.concat({k: dailies[k] for k in opt_alts}, axis=1), n_boot=SPA_BOOT, seed=17)
        option_spa["family"] = "option"
        option_spa["ids"] = opt_alts
    if top_alts:
        top_spa = spa_test(bench, pd.concat({k: dailies[k] for k in top_alts}, axis=1), n_boot=SPA_BOOT, seed=19)
        top_spa["family"] = "top1"
        top_spa["ids"] = top_alts
    spa_payload = {
        "status": "DONE",
        "option_family": option_spa,
        "top1_family": top_spa,
        "note": "两组 family 分开。Hansen-style block bootstrap SPA vs v4。",
    }
    dump_json(SPA_JSON, spa_payload)
    write_md(
        SPA_MD,
        f"# Round 15 · SPA\n\noption p={(option_spa or {}).get('p_spa')} top1 p={(top_spa or {}).get('p_spa')}\n",
    )
    payload = {"status": "DONE", "bootstrap": boot_payload, "cpcv": cpcv_payload, "spa": spa_payload}
    dump_json(STAT_JSON, payload)
    return payload


def load_stats() -> dict[str, Any]:
    return load_json(STAT_JSON)


def load_bootstrap() -> dict[str, Any]:
    return load_json(BOOT_JSON)


def load_cpcv() -> dict[str, Any]:
    return load_json(CPCV_JSON)


def load_spa() -> dict[str, Any]:
    return load_json(SPA_JSON)


__all__ = [
    "BOOT_N_MIN",
    "load_bootstrap",
    "load_cpcv",
    "load_daily_map",
    "load_spa",
    "load_stats",
    "run_stats",
    "sharpe_of",
]
