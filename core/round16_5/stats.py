"""同窗绩效、20 日块 bootstrap N=5000、CPCV、family SPA、前向。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import MARKET_END, ROUND16_5_ROOT
from ..round11.stats import cpcv_oos_sharpe, spa_test
from .common import (
    cagr_of,
    daily_from_nav,
    dump_json,
    empirical_p,
    holm_adjust,
    load_json,
    max_drawdown_of,
    pack_daily,
    sharpe_of,
    write_md,
)
from .contract import (
    BLOCK_DAYS,
    BOOT_N,
    BOOT_N_MIN,
    EMBARGO,
    FORWARD_OOS_NOT_AVAILABLE,
    INCONCLUSIVE_LOW_N,
    MIN_BLOCK,
    R16_SPA_P,
    SPA_BOOT,
    V00,
    V01,
    V03,
    V04,
    experiment_dir,
    parity_dir,
)
from .freeze import require_frozen
from .attribution import load_attribution
from .placebo import load_placebo

STAT_JSON = ROUND16_5_ROOT / "stats" / "stats.json"
BOOT_JSON = ROUND16_5_ROOT / "bootstrap" / "bootstrap.json"
CPCV_JSON = ROUND16_5_ROOT / "cpcv" / "cpcv.json"
SPA_JSON = ROUND16_5_ROOT / "spa" / "spa.json"
FWD_JSON = ROUND16_5_ROOT / "forward" / "forward.json"
WIN_MD = ROUND16_5_ROOT / "11_same_window.md"
STAT_MD = ROUND16_5_ROOT / "13_statistical_validation.md"
FWD_MD = ROUND16_5_ROOT / "14_forward_oos.md"


def _nav_path(code: str):
    if code in {V00, V01}:
        p = parity_dir(code) / "nav.parquet"
        return p if p.is_file() else None
    p = experiment_dir(code) / "nav.parquet"
    return p if p.is_file() else None


def load_daily_map() -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for code in (V00, V01, V03, V04):
        path = _nav_path(code)
        if path is None:
            continue
        out[code] = daily_from_nav(pd.read_parquet(path))
    return out


def _resample(mat: np.ndarray, starts: np.ndarray, block: int) -> np.ndarray:
    n = mat.shape[0]
    pieces = [mat[s : s + block] for s in starts]
    return np.concatenate(pieces, axis=0)[:n]


def _window_metrics(dmap: dict[str, pd.Series], mask: pd.Index, label: str) -> dict[str, Any]:
    rec = {"window": label, "n": int(len(mask))}
    if len(mask) < 40:
        rec["flag"] = INCONCLUSIVE_LOW_N
    for code, s in dmap.items():
        sub = s.reindex(mask).dropna()
        rec[code] = pack_daily(sub)
    return rec


def run_same_window(dmap: dict[str, pd.Series]) -> dict[str, Any]:
    attr = load_attribution()
    aligned = pd.concat(dmap, axis=1).dropna()
    full_idx = aligned.index
    obs = pd.DatetimeIndex(pd.to_datetime(attr.get("treat_in_window") or [])).normalize()
    from .engine import option_observable_dates

    opt = option_observable_dates().intersection(full_idx)
    windows = {
        "FULL_RESEARCH_WINDOW": _window_metrics(dmap, full_idx, "FULL_RESEARCH_WINDOW"),
        "OPTION_VALID_SIGNAL_WINDOW": _window_metrics(dmap, opt, "OPTION_VALID_SIGNAL_WINDOW"),
        "ACTUALLY_TREATED_DAYS": _window_metrics(dmap, obs.intersection(full_idx), "ACTUALLY_TREATED_DAYS"),
    }
    yearly = []
    for y in range(2018, 2027):
        idx = full_idx[full_idx.year == y]
        row = {"year": y, "n": int(len(idx))}
        if len(idx) < 30:
            row["flag"] = INCONCLUSIVE_LOW_N
        for code, s in dmap.items():
            row[code] = pack_daily(s.reindex(idx).dropna())
        yearly.append(row)
    payload = {"status": "DONE", "windows": windows, "yearly": yearly}
    write_md(
        WIN_MD,
        "\n".join([
            "# Round 16.5 · 11 同窗绩效",
            "",
            f"FULL n=`{windows['FULL_RESEARCH_WINDOW']['n']}` OPTION n=`{windows['OPTION_VALID_SIGNAL_WINDOW']['n']}` TREATED n=`{windows['ACTUALLY_TREATED_DAYS']['n']}`。",
            "",
        ]),
    )
    return payload


def run_bootstrap(dmap: dict[str, pd.Series]) -> dict[str, Any]:
    cols = [c for c in (V00, V01, V03, V04) if c in dmap]
    aligned = pd.concat({c: dmap[c] for c in cols}, axis=1).dropna()
    mat = aligned.to_numpy(float)
    n = mat.shape[0]
    block = int(BLOCK_DAYS)
    n_blocks = max(1, n - block + 1)
    rng = np.random.default_rng(20261016)
    boot_n = int(BOOT_N)
    sharpes = []
    pair_defs = [("V01-V00", V01, V00), ("V03-V00", V03, V00), ("V04-V00", V04, V00), ("V01-V03", V01, V03), ("V04-V03", V04, V03), ("V04-V01", V04, V01)]
    pair_idx = [(a, b) for _, a, b in pair_defs if a in cols and b in cols]
    pair_means = {f"{a}-{b}": [] for a, b in pair_idx}
    try:
        for i in range(boot_n):
            starts = rng.integers(0, n_blocks, size=int(np.ceil(n / block)))
            resampled = _resample(mat, starts, block)
            sh = []
            for j in range(resampled.shape[1]):
                sh.append(sharpe_of(pd.Series(resampled[:, j])))
            sharpes.append(sh)
            for a, b in pair_idx:
                ia, ib = cols.index(a), cols.index(b)
                diff = resampled[:, ia] - resampled[:, ib]
                pair_means[f"{a}-{b}"].append(float(np.mean(diff)))
            if i + 1 == BOOT_N_MIN and len(sharpes) >= BOOT_N_MIN:
                pass
    except MemoryError:
        sharpes = sharpes[: max(len(sharpes), BOOT_N_MIN)]
    boot = np.array(sharpes, dtype=float)
    pairs = {}
    for a, b in pair_idx:
        xs = np.array(pair_means[f"{a}-{b}"], dtype=float)
        # 单侧：mean_diff > 0
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
        "requested": boot_n,
        "min_required": BOOT_N_MIN,
        "block_days": block,
        "ids": cols,
        "mean_sharpe": {c: float(np.nanmean(boot[:, i])) for i, c in enumerate(cols)},
        "p10_sharpe": {c: float(np.nanpercentile(boot[:, i], 10)) for i, c in enumerate(cols)},
        "pairs": pairs,
        "sample_size": n,
        "coverage": [str(aligned.index.min().date()), str(aligned.index.max().date())],
    }
    dump_json(BOOT_JSON, payload)
    return payload


def run_cpcv(dmap: dict[str, pd.Series]) -> dict[str, Any]:
    by = {}
    for c, s in dmap.items():
        rec = cpcv_oos_sharpe(s)
        rec["not_forward_oos"] = True
        rec["embargo_blocks"] = EMBARGO
        rec["min_block"] = MIN_BLOCK
        rec["median"] = rec.get("oos_sharpe_median")
        by[c] = rec
    payload = {"status": "DONE", "by_id": by, "not_forward": True, "note": "CPCV 不是 forward OOS。"}
    dump_json(CPCV_JSON, payload)
    return payload


def run_spa(dmap: dict[str, pd.Series]) -> dict[str, Any]:
    alts = [c for c in (V01, V04) if c in dmap]
    spa = {"status": "SKIP_NO_ALTS"}
    if len(alts) >= 1 and V00 in dmap:
        alt_df = pd.concat({c: dmap[c] for c in alts}, axis=1)
        spa = spa_test(dmap[V00], alt_df, n_boot=SPA_BOOT)
        spa["family"] = "round16_5_v01_v04"
        spa["n_alts"] = len(alts)
        spa["round16_spa_p_retained"] = R16_SPA_P
        spa["methodology_note"] = (
            "Round16 SPA p=0.496 是 O01+O02 vs v4 的 family SPA，账本对照时代。"
            "本轮 family 为 V01/V04 vs V00，全引擎确认性样本，不是独立新样本。"
            "PRIOR_SELECTION_EXISTS=TRUE，不得声称独立确认。"
        )
    ctrl = None
    if V03 in dmap and V00 in dmap:
        ctrl = spa_test(dmap[V00], pd.concat({V03: dmap[V03]}, axis=1), n_boot=SPA_BOOT)
        ctrl["family"] = "v03_control_only"
    payload = {"status": spa.get("status") or "DONE", "spa": spa, "v03_control_spa": ctrl, "round16_spa_p": R16_SPA_P}
    dump_json(SPA_JSON, payload)
    return payload


def run_forward(**_: Any) -> dict[str, Any]:
    require_frozen()
    payload = {
        "status": FORWARD_OOS_NOT_AVAILABLE,
        "cutoff": str(MARKET_END),
        "checked_after": "2026-07-24",
        "inbox_incomplete": True,
        "forward_validation_supported": False,
        "live_validated": False,
        "note": f"研究快照截止 {MARKET_END}。冻结后再查 2026-07-24 之后无完整未参与研究新数据。不得把 CPCV 当 forward OOS。",
    }
    dump_json(FWD_JSON, payload)
    write_md(FWD_MD, f"# Round 16.5 · 14 Forward OOS\n\n**{FORWARD_OOS_NOT_AVAILABLE}**。{payload['note']}\n")
    return payload


def run_stats(**_: Any) -> dict[str, Any]:
    require_frozen()
    dmap = load_daily_map()
    windows = run_same_window(dmap)
    boot = run_bootstrap(dmap)
    cpcv = run_cpcv(dmap)
    spa = run_spa(dmap)
    placebo = load_placebo()
    payload = {
        "status": "DONE",
        "windows": windows,
        "bootstrap": {k: boot.get(k) for k in ("n", "pairs", "mean_sharpe", "p10_sharpe")},
        "cpcv": {k: cpcv.get(k) for k in ("status", "not_forward")},
        "spa": spa.get("spa"),
        "round16_spa_p": R16_SPA_P,
        "placebo_n": placebo.get("n"),
        "placebo_p_sharpe": placebo.get("p_empirical_sharpe"),
    }
    dump_json(STAT_JSON, payload)
    write_md(
        STAT_MD,
        "\n".join([
            "# Round 16.5 · 13 统计验证",
            "",
            f"bootstrap N=`{boot.get('n')}` CPCV not_forward=True SPA p=`{(spa.get('spa') or {}).get('p_spa')}`。",
            f"Round16 SPA p=`{R16_SPA_P}` 保留。V02 N=`{placebo.get('n')}` p_sharpe=`{placebo.get('p_empirical_sharpe')}`。",
            "PRIOR_SELECTION_EXISTS=TRUE。不得声称独立确认。",
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


def load_forward() -> dict[str, Any]:
    return load_json(FWD_JSON)
