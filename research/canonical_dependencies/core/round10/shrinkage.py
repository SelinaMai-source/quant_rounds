"""C01–C03：9 cell shrinkage，k=50，3Y walk-forward，阈值只用训练折。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND10_ROOT, assert_project_result_path, ensure_round10_tree
from .allocator import write_alloc
from .contract import SHRINK_K, WF_PREDICT_YEARS
from .daily import load_daily, persist_daily, train_mask
from .registry import specs_for_stage
from .runner import run_codes


def _cell_stats(block: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    yy = pd.to_numeric(y.reindex(block.index), errors="coerce").dropna()
    n = int(len(yy))
    if n == 0:
        return {"n": 0, "mu": np.nan, "median": np.nan, "win": np.nan, "large_loss": np.nan, "se": np.nan}
    std = float(yy.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "mu": float(yy.mean()),
        "median": float(yy.median()),
        "win": float((yy > 0).mean()),
        "large_loss": float((yy <= -0.08).mean()),
        "se": float(std / np.sqrt(n)) if n > 0 else np.nan,
    }


def _shrink(mu_c: float, n_c: float, mu_g: float, k: float = SHRINK_K) -> float:
    if not np.isfinite(mu_c):
        return float(mu_g) if np.isfinite(mu_g) else 0.0
    n = float(n_c)
    return (n / (n + k)) * float(mu_c) + (k / (n + k)) * float(mu_g)


def _map_edge(value: float, cut: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 0.0
    if value > cut:
        return 0.10
    return 0.05


def build_shrinkage_allocs(daily: pd.DataFrame) -> dict[str, Any]:
    ensure_round10_tree()
    idx = daily.index
    y_raw = pd.to_numeric(daily.get("top2_ret"), errors="coerce")
    y_stress = y_raw - 0.0020
    out = {
        "R10_C01": pd.Series(0.0, index=idx, dtype=float),
        "R10_C02": pd.Series(0.0, index=idx, dtype=float),
        "R10_C03": pd.Series(0.0, index=idx, dtype=float),
    }
    fold_log: list[dict[str, Any]] = []
    cells = [(a, s) for a in ("A", "B", "C") for s in ("LOW", "MID", "HIGH")]
    for year in WF_PREDICT_YEARS:
        tr = train_mask(daily, year)
        te = daily.index.year == year
        if int(tr.sum()) < 30:
            from .allocator import b03_nav

            nav = b03_nav(daily.loc[te, "agreement"], daily.loc[te, "sep_bucket"])
            for code in out:
                out[code].loc[te] = nav.to_numpy()
            fold_log.append({"year": year, "fallback": "B03", "n_train": int(tr.sum())})
            continue
        for ycol, code, conservative in (
            (y_raw, "R10_C01", False),
            (y_raw, "R10_C02", True),
            (y_stress, "R10_C03", False),
        ):
            stats = {}
            mus = []
            ns = []
            for a, s in cells:
                mask = tr & daily["agreement"].eq(a) & daily["sep_bucket"].eq(s)
                st = _cell_stats(daily.loc[mask], ycol)
                stats[f"{a}_{s}"] = st
                if st["n"] > 0 and np.isfinite(st["mu"]):
                    mus.append(st["mu"])
                    ns.append(st["n"])
            mu_g = float(np.average(mus, weights=ns)) if mus else float(ycol.loc[tr].mean())
            shrink = {}
            pos = []
            for a, s in cells:
                st = stats[f"{a}_{s}"]
                mu_s = _shrink(st["mu"], st["n"], mu_g)
                if conservative:
                    se = st["se"] if np.isfinite(st.get("se", np.nan)) else 0.0
                    mu_s = mu_s - 0.5 * float(se)
                shrink[f"{a}_{s}"] = mu_s
                if mu_s > 0:
                    pos.append(mu_s)
            cut = float(np.median(pos)) if pos else 0.0
            mapped = []
            for a, s in zip(daily.loc[te, "agreement"].astype(str), daily.loc[te, "sep_bucket"].astype(str)):
                mapped.append(_map_edge(shrink.get(f"{a}_{s}", 0.0), cut))
            out[code].loc[te] = mapped
            fold_log.append({
                "year": year, "code": code, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                "mu_global": mu_g, "cut": cut, "shrink": shrink, "stats": stats,
            })
    for code, series in out.items():
        write_alloc(code, idx, series)
        series.to_frame("top2_nav_frac").to_parquet(
            ROUND10_ROOT / "shrinkage" / f"{code}.parquet"
        )
    payload = {"k": SHRINK_K, "years": list(WF_PREDICT_YEARS), "folds": fold_log}
    path = ROUND10_ROOT / "metrics" / "shrinkage_walkforward.json"
    assert_project_result_path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    return payload


def run_shrinkage(*, resume: bool = True, jobs: int = 1, experiment: str = "") -> dict[str, Any]:
    daily = load_daily()
    persist_daily(daily)
    wf = build_shrinkage_allocs(daily)
    codes = [c.strip() for c in experiment.split(",") if c.strip()] if experiment else [s.code for s in specs_for_stage("shrinkage")]
    bt = run_codes(codes, resume=resume, jobs=jobs)
    from .docs import write_named_md

    write_named_md(
        "03_shrinkage_allocator.md",
        "shrinkage",
        codes,
        "k=50 固定。3Y walk-forward，阈值只用训练折正值 cell 的 median。C03 训练效用 = Top2 净收益 − 20bp。\n"
        f"折数 {len(wf.get('folds') or [])}。\n",
    )
    return {"walkforward": {"k": wf.get("k"), "n_folds": len(wf.get("folds") or [])}, "bt": {k: v.get("status") for k, v in (bt.get("jobs") or {}).items()}, "failed": bt.get("failed") or []}


__all__ = ["build_shrinkage_allocs", "run_shrinkage"]
