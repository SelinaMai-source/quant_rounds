"""PSR / DSR / MinTRL / CPCV / SPA / staircase。两套费用都跑。"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as spstats

from ..paths import RESULT_ROOT, ROUND11_ROOT
from .common import daily_from_nav, dump_json, load_legacy_dailies, moments, sharpe_of, stage_dir, write_md
from .contract import BASELINES
from .cost import load_cost_status
from .alphaforge import load_alphaforge
from .placebo import load_placebo
from .trials import load_trials

STATS_MD = ROUND11_ROOT / "10_dsr_pbo_cpcv.md"
EULER = 0.5772156649015329
EMBARGO = 12
MIN_BLOCK = 60


def psr(sr: float, n: int, skew: float, kurt: float, sr_star: float = 0.0) -> float | None:
    if n <= 2 or not np.isfinite(sr):
        return None
    var = (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr) / (n - 1)
    if var <= 0:
        return None
    return float(spstats.norm.cdf((sr - sr_star) / np.sqrt(var)))


def expected_max_sr(n_obs: int, n_trials: int, skew: float, kurt: float, sr: float) -> float | None:
    if n_obs <= 2 or n_trials < 2:
        return None
    var = (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr) / (n_obs - 1)
    if var <= 0:
        return None
    z1 = float(spstats.norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(spstats.norm.ppf(1.0 - np.exp(-1.0) / n_trials))
    return float(np.sqrt(var) * ((1.0 - EULER) * z1 + EULER * z2))


def dsr(sr: float, n: int, n_trials: int, skew: float, kurt: float) -> dict[str, Any]:
    sr_star = expected_max_sr(n, n_trials, skew, kurt, sr)
    prob = psr(sr, n, skew, kurt, sr_star=sr_star or 0.0) if sr_star is not None else None
    return {"sr": sr, "sr_star": sr_star, "dsr": prob, "n_obs": n, "n_trials": n_trials}


def min_trl(sr: float, skew: float, kurt: float, *, sr_star: float = 0.0, alpha: float = 0.95) -> float | None:
    if sr <= sr_star or not np.isfinite(sr):
        return None
    z = float(spstats.norm.ppf(alpha))
    num = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    return float(1.0 + num * (z / (sr - sr_star)) ** 2)


def _blocks(index: pd.DatetimeIndex, min_len: int = MIN_BLOCK) -> list[pd.DatetimeIndex]:
    n = len(index)
    n_blocks = max(6, n // max(min_len, 1))
    while n_blocks > 4 and n // n_blocks < min_len:
        n_blocks -= 1
    edges = np.linspace(0, n, n_blocks + 1, dtype=int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a >= min_len:
            out.append(index[a:b])
    return out


def cpcv_oos_sharpe(daily: pd.Series, *, n_test_blocks: int = 2) -> dict[str, Any]:
    daily = daily.dropna().sort_index()
    blocks = _blocks(daily.index)
    if len(blocks) < 4:
        return {"status": "TOO_FEW_BLOCKS", "n_blocks": len(blocks)}
    combos = list(itertools.combinations(range(len(blocks)), n_test_blocks))
    oos = []
    for test_idx in combos:
        test_pos = set(test_idx)
        test_index = blocks[test_idx[0]]
        for j in test_idx[1:]:
            test_index = test_index.append(blocks[j])
        # purge / embargo around each test block
        mask = pd.Series(True, index=daily.index)
        for j in test_idx:
            b = blocks[j]
            start = b[0]
            end = b[-1]
            lo = start - pd.tseries.offsets.BDay(EMBARGO)
            hi = end + pd.tseries.offsets.BDay(EMBARGO)
            mask.loc[(mask.index >= lo) & (mask.index <= hi)] = False
            # keep test days for OOS evaluation
            mask.loc[b] = True
        oos_ret = daily.loc[test_index]
        # drop embargo-adjacent contamination already handled by not using IS here
        sh = sharpe_of(oos_ret)
        if sh is not None:
            oos.append(sh)
    arr = np.array(oos, dtype=float)
    return {
        "status": "DONE",
        "not_forward_oos": True,
        "note": "CPCV 不是 forward OOS。purge/embargo≈12 交易日（持有≤10 + 标签窗）。块≥60日。",
        "n_blocks": len(blocks),
        "n_combos": len(combos),
        "embargo_days": EMBARGO,
        "min_block": MIN_BLOCK,
        "oos_sharpe_mean": float(np.nanmean(arr)) if len(arr) else None,
        "oos_sharpe_median": float(np.nanmedian(arr)) if len(arr) else None,
        "oos_sharpe_p05": float(np.nanpercentile(arr, 5)) if len(arr) else None,
        "oos_sharpe_p95": float(np.nanpercentile(arr, 95)) if len(arr) else None,
        "n_valid": int(np.isfinite(arr).sum()) if len(arr) else 0,
    }


def selection_pbo(matrix: pd.DataFrame, *, n_chunks: int = 8) -> dict[str, Any]:
    """TRUE_SELECTION_BASED_PBO：IS 选胜，看 OOS 相对秩。"""
    matrix = matrix.dropna(how="all")
    if matrix.shape[1] < 8 or matrix.shape[0] < MIN_BLOCK * 4:
        return {
            "status": "TRUE_PBO_NOT_FULLY_IDENTIFIABLE",
            "reason": f"matrix cols={matrix.shape[1]} rows={matrix.shape[0]}",
            "label": "TRUE_PBO_NOT_FULLY_IDENTIFIABLE",
        }
    n = len(matrix)
    edges = np.linspace(0, n, n_chunks + 1, dtype=int)
    chunks = [matrix.iloc[a:b] for a, b in zip(edges[:-1], edges[1:]) if b > a]
    if len(chunks) < 4:
        return {"status": "TRUE_PBO_NOT_FULLY_IDENTIFIABLE", "label": "TRUE_PBO_NOT_FULLY_IDENTIFIABLE"}
    half = len(chunks) // 2
    logits = []
    under = 0
    n_combo = 0
    for is_idx in itertools.combinations(range(len(chunks)), half):
        oos_idx = [i for i in range(len(chunks)) if i not in is_idx]
        is_df = pd.concat([chunks[i] for i in is_idx], axis=0)
        oos_df = pd.concat([chunks[i] for i in oos_idx], axis=0)
        is_sr = is_df.apply(lambda s: sharpe_of(s.dropna()) or -9.0)
        winner = str(is_sr.idxmax())
        oos_sr = oos_df.apply(lambda s: sharpe_of(s.dropna()) or -9.0)
        ranks = oos_sr.rank(ascending=False, method="average")
        rel = float((ranks[winner] - 1.0) / max(len(ranks) - 1, 1))
        rel = min(max(rel, 1e-6), 1.0 - 1e-6)
        logits.append(float(np.log(rel / (1.0 - rel))))
        if ranks[winner] > (len(ranks) + 1) / 2.0:
            under += 1
        n_combo += 1
        if n_combo >= 200:
            break
    arr = np.array(logits, dtype=float)
    return {
        "status": "DONE",
        "label": "TRUE_SELECTION_BASED_PBO",
        "pbo": float(under / n_combo) if n_combo else None,
        "mean_logit": float(np.mean(arr)) if len(arr) else None,
        "n_combos": n_combo,
        "n_strats": int(matrix.shape[1]),
        "n_chunks": len(chunks),
        "note": "IS 选 Sharpe 胜者，统计其 OOS 秩劣于中位的比例。",
    }


def loss_proxy(daily: pd.Series, blocks: int = 8) -> dict[str, Any]:
    daily = daily.dropna()
    if len(daily) < blocks * MIN_BLOCK:
        return {"status": "TOO_SHORT", "label": "PROBABILITY_OF_LOSS_PROXY"}
    edges = np.linspace(0, len(daily), blocks + 1, dtype=int)
    loss = 0
    n = 0
    for a, b in zip(edges[:-1], edges[1:]):
        sh = sharpe_of(daily.iloc[a:b])
        if sh is None:
            continue
        n += 1
        if sh < 0:
            loss += 1
    return {
        "status": "DONE",
        "label": "PROBABILITY_OF_LOSS_PROXY",
        "p_loss": float(loss / n) if n else None,
        "n_folds": n,
        "note": "这不是 selection-based PBO。",
    }


def hac_mean_pvalue(x: np.ndarray, lags: int = 20) -> dict[str, Any]:
    x = x[np.isfinite(x)]
    if len(x) < lags + 5:
        return {"status": "TOO_SHORT"}
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
    p = float(2.0 * (1.0 - spstats.norm.cdf(abs(z)))) if np.isfinite(z) else None
    return {"mean": mu, "hac_se": se, "z": z, "p_two_sided": p, "lags": lags}


def staircase(dailies: dict[str, pd.Series]) -> dict[str, Any]:
    out = {}
    pairs = (("v2", "v1"), ("v3", "v2"), ("v4", "v3"))
    for a, b in pairs:
        aligned = pd.concat([dailies[a].rename("a"), dailies[b].rename("b")], axis=1).dropna()
        delta = (aligned["a"] - aligned["b"]).to_numpy()
        out[f"{a}_minus_{b}"] = hac_mean_pvalue(delta)
    return out


def spa_test(benchmark: pd.Series, alts: pd.DataFrame, *, n_boot: int = 500, seed: int = 17) -> dict[str, Any]:
    aligned = pd.concat([benchmark.rename("b"), alts], axis=1).dropna()
    if aligned.shape[1] < 3 or len(aligned) < 80:
        return {"status": "TOO_SHORT"}
    d = aligned.drop(columns=["b"]).sub(aligned["b"], axis=0)
    mean_d = d.mean()
    rng = np.random.default_rng(seed)
    n = len(d)
    block = 20
    n_blocks = int(np.ceil(n / block))
    boot_max = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        sample = d.to_numpy()[idx]
        boot_max.append(float(np.max(sample.mean(axis=0))))
    obs = float(mean_d.max())
    p = float(np.mean(np.array(boot_max) >= obs))
    return {
        "status": "DONE",
        "observed_max_mean_delta": obs,
        "p_spa": p,
        "n_alts": int(d.shape[1]),
        "note": "Hansen-style block bootstrap SPA vs v4；不是因果证明。",
    }


def _load_daily_matrix(max_cols: int = 80) -> pd.DataFrame:
    index_path = ROUND11_ROOT / "trials" / "daily_matrix_index.json"
    if not index_path.is_file():
        return pd.DataFrame()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    cols = {}
    for row in payload.get("rows") or []:
        path = Path(row["nav_path"])
        if not path.is_file():
            continue
        try:
            s = daily_from_nav(pd.read_parquet(path))
        except Exception:
            continue
        key = f"{row.get('source_round')}::{row.get('trial_id')}"
        cols[key] = s
        if len(cols) >= max_cols:
            break
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).sort_index()


def _corrected_dailies() -> dict[str, pd.Series]:
    cost = load_cost_status()
    out = {}
    for key, row in (cost.get("corrected") or {}).items():
        path = row.get("nav_path")
        if path and Path(path).is_file():
            out[key] = daily_from_nav(pd.read_parquet(path))
    return out


def _pack_for(daily: pd.Series, n_trials_set: dict[str, int], label: str) -> dict[str, Any]:
    mom = moments(daily)
    sr_ann = sharpe_of(daily)
    n_days = int(mom["n"])
    daily_sr = float(mom["mean"] / mom["std"]) if mom["std"] not in (0, None) and mom["std"] == mom["std"] else float("nan")
    n_years = max(n_days / 252.0, 2.0)
    # Bailey：SR 与 n 必须同频。主结果用日频 SR + 日数；另报年化 SR + 年数，避免把 1.81 配 2075 天造成 DSR=1 虚高。
    dsr_daily = {
        name: dsr(float(daily_sr or 0), n_days, int(nt), mom["skew"], mom["kurt"])
        for name, nt in n_trials_set.items()
    }
    dsr_ann_years = {
        name: dsr(float(sr_ann or 0), int(round(n_years)), int(nt), mom["skew"], mom["kurt"])
        for name, nt in n_trials_set.items()
    }
    return {
        "label": label,
        "sharpe": sr_ann,
        "daily_sr": daily_sr,
        "n_years": n_years,
        "moments": mom,
        "psr_zero_daily": psr(float(daily_sr or 0), n_days, mom["skew"], mom["kurt"], 0.0),
        "psr_zero_ann_nyears": psr(float(sr_ann or 0), int(round(n_years)), mom["skew"], mom["kurt"], 0.0),
        "min_trl_psr95_daily": min_trl(float(daily_sr or 0), mom["skew"], mom["kurt"]),
        "dsr": dsr_daily,
        "dsr_ann_nyears": dsr_ann_years,
        "dsr_convention": "dsr=Bailey daily-SR + n_days; dsr_ann_nyears=annualized SR + n_years",
        "cpcv": cpcv_oos_sharpe(daily),
        "loss_proxy": loss_proxy(daily),
    }


def run_stats(**_: Any) -> dict[str, Any]:
    dest = stage_dir("stats")
    trials = load_trials()
    af = load_alphaforge()
    n_set = {
        "documented": int(trials.get("n_documented") or 20),
        "full_discovered": int(trials.get("n_full_discovered") or 40),
        "conservative_upper": int(trials.get("n_conservative_upper") or 80),
    }
    if min(n_set.values()) <= 4:
        raise RuntimeError("n_trials<=4 forbidden")
    legacy = load_legacy_dailies()
    corrected = _corrected_dailies()
    matrix = _load_daily_matrix()
    pbo_sel = selection_pbo(matrix)
    pbo_sel["universe_status"] = "TRUE_PBO_NOT_FULLY_IDENTIFIABLE"
    pbo_sel["universe_note"] = (
        "可恢复日收益矩阵只是 documented 子集；更早未落盘 search 不可恢复。"
        "对 recover 子集算 selection-PBO，但对完整研究宇宙必须标 TRUE_PBO_NOT_FULLY_IDENTIFIABLE。"
    )
    af_name = af.get("pbo_name") or af.get("math_label") or "LOCAL_EQUIVALENT_NOT_ALPHA_FORGE"
    payload = {
        "status": "DONE",
        "n_trials": n_set,
        "legacy_v4": _pack_for(legacy["v4"], n_set, "legacy_v4"),
        "corrected_v4": _pack_for(corrected["v4"], n_set, "corrected_v4") if "v4" in corrected else {},
        "true_pbo": pbo_sel,
        "alphaforge_pbo_name": af_name,
        "proxy_vs_true": {
            "TRUE_PBO": pbo_sel.get("label"),
            "PROXY": "PROBABILITY_OF_LOSS_PROXY",
            "alphaforge": af_name,
        },
        "staircase_legacy": staircase(legacy),
        "staircase_corrected": staircase(corrected) if len(corrected) == 4 else {"status": "NEED_CORRECTED"},
        "spa_legacy": spa_test(legacy["v4"], pd.DataFrame({k: v for k, v in legacy.items() if k != "v4"})),
        "not_forward_oos": True,
        "post_hoc_diagnostic": True,
    }
    if "v4" in corrected and len(corrected) >= 2:
        payload["spa_corrected"] = spa_test(
            corrected["v4"],
            pd.DataFrame({k: v for k, v in corrected.items() if k != "v4"}),
        )
    dump_json(dest / "stats.json", payload)
    dsr_doc = ((payload.get("corrected_v4") or {}).get("dsr") or {}).get("full_discovered") or {}
    dsr_ann = ((payload.get("corrected_v4") or {}).get("dsr_ann_nyears") or {}).get("full_discovered") or {}
    cpcv = (payload.get("corrected_v4") or {}).get("cpcv") or {}
    lines = [
        "# Round 11 · DSR / PBO / CPCV / SPA",
        "",
        f"trial 三档：documented={n_set['documented']} / discovered={n_set['full_discovered']} / upper={n_set['conservative_upper']}。**禁止 n=4**。",
        "",
        "DSR 主结果 = Bailey **日频 SR + 日数**（与 n 同频）。`dsr_ann_nyears` = 年化 SR + 年数；禁止把年化 1.81 配 n=2075。",
        "",
        "## 关键",
        "",
        f"- corrected v4 DSR daily-freq (full discovered) = **{dsr_doc.get('dsr')}**；SR*={dsr_doc.get('sr_star')}；daily SR={dsr_doc.get('sr')}",
        f"- corrected v4 DSR ann+nyears (full discovered) = **{dsr_ann.get('dsr')}**；SR*={dsr_ann.get('sr_star')}",
        f"- TRUE_PBO (recovered subset) = **{pbo_sel.get('pbo')}** label=`{pbo_sel.get('label')}` n_strats={pbo_sel.get('n_strats')}",
        f"- 完整宇宙：`{pbo_sel.get('universe_status')}`",
        f"- PROXY P(loss) = **{((payload.get('corrected_v4') or {}).get('loss_proxy') or {}).get('p_loss')}**",
        f"- AlphaForge 命名 = `{af_name}`",
        f"- CPCV OOS Sharpe mean = **{cpcv.get('oos_sharpe_mean')}**（**不是 forward OOS**）",
        "",
        pbo_sel.get("note") or "",
        "",
        cpcv.get("note") or "",
        "",
    ]
    write_md(STATS_MD, "\n".join(lines))
    return payload


def load_stats() -> dict[str, Any]:
    path = ROUND11_ROOT / "stats" / "stats.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


__all__ = ["load_stats", "run_stats"]
