"""新因子 vs 真实 14/13。显式 signed/abs 字段。TRAIN_FOLD 官方。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_2_ROOT
from ..round20.panel import fold_masks
from .common import dump_json, write_md
from .contract import CLUSTER_SPEARMAN_ABS, FIVE_FEATURES, FIVE_HEAD, KEYS, MD, NEAREST_LEGACY_NOT_IDENTIFIABLE, PREDICT_YEARS, TEN_FEATURES
from .panels import load_research_panel, pit_safe_factors

CORR_JSON = ROUND21_2_ROOT / "correlation" / "correlation.json"
MATRIX_CSV = ROUND21_2_ROOT / "new_old_correlation.csv"
FOLD_CSV = ROUND21_2_ROOT / "correlation" / "fold_nearest_legacy_mapping.csv"
REPLACE_CSV = ROUND21_2_ROOT / "factor_replacement_matrix.csv"


def _spearman(a: pd.Series, b: pd.Series) -> float | None:
    x, y = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    if int(mask.sum()) < 5:
        return None
    rho = x.loc[mask].corr(y.loc[mask], method="spearman")
    return float(rho) if rho is not None and np.isfinite(rho) else None


def _pearson(a: pd.Series, b: pd.Series) -> float | None:
    x, y = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    if int(mask.sum()) < 5:
        return None
    rho = x.loc[mask].corr(y.loc[mask], method="pearson")
    return float(rho) if rho is not None and np.isfinite(rho) else None


def run_correlate(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and CORR_JSON.is_file() and MATRIX_CSV.is_file() and FOLD_CSV.is_file():
        from .common import load_json

        existing = load_json(CORR_JSON)
        if existing.get("status") == "DONE":
            return existing
    panel = load_research_panel(resume=True)
    new_factors = [f for f in pit_safe_factors(resume=True) if f in panel.columns]
    old_cols = list(dict.fromkeys([*FIVE_FEATURES, *TEN_FEATURES]))
    matrix_rows = []
    fold_rows = []
    nearest_by_factor: dict[str, Any] = {}
    # 按日一次 Spearman 矩阵，避免 23×14×日 次逐对 corr
    daily_map: dict[str, dict[str, list[float]]] = {f: {o: [] for o in old_cols} for f in new_factors}
    use_cols = [c for c in [*new_factors, *old_cols] if c in panel.columns]
    for _, g in panel.groupby("trade_date", sort=False):
        if len(g) < 5:
            continue
        sub = g[use_cols]
        if sub.nunique(dropna=True).le(1).all():
            continue
        with np.errstate(invalid="ignore"):
            mat = sub.corr(method="spearman")
        for feat in new_factors:
            if feat not in mat.columns:
                continue
            for old in old_cols:
                if old not in mat.columns:
                    continue
                rho = mat.loc[feat, old]
                if rho is not None and np.isfinite(rho):
                    daily_map[feat][old].append(float(rho))
    for feat in new_factors:
        hist = []
        for old in old_cols:
            spear = _spearman(panel[feat], panel[old])
            pear = _pearson(panel[feat], panel[old])
            dvals = daily_map[feat][old]
            rec = {
                "new_factor": feat,
                "legacy_factor": old,
                "full_sample_signed_rho": spear,
                "full_sample_abs_rho": None if spear is None else abs(spear),
                "full_sample_pearson": pear,
                "mean_daily_rho": float(np.mean(dvals)) if dvals else None,
                "median_daily_rho": float(np.median(dvals)) if dvals else None,
                "mean_absolute_daily_rho": float(np.mean(np.abs(dvals))) if dvals else None,
                "n_daily": len(dvals),
                "scope": "FULL_HISTORY_DIAGNOSTIC",
            }
            hist.append(rec)
            matrix_rows.append(rec)
        fold_nearest = {}
        last_j = None
        changed = False
        for y in PREDICT_YEARS:
            train_m, _ = fold_masks(panel, predict_year=y, label=FIVE_HEAD.target)
            tr = panel.loc[train_m]
            scored = []
            for old in old_cols:
                rho = _spearman(tr[feat], tr[old])
                if rho is None:
                    continue
                scored.append((abs(rho), rho, old))
            scored.sort(reverse=True)
            if not scored:
                rec = {"new_factor": feat, "predict_year": y, "nearest_legacy": None, "fold_signed_rho": None, "fold_abs_rho": None, "status": NEAREST_LEGACY_NOT_IDENTIFIABLE}
            else:
                ab, signed, jstar = scored[0]
                rec = {
                    "new_factor": feat,
                    "predict_year": y,
                    "nearest_legacy": jstar,
                    "fold_signed_rho": signed,
                    "fold_abs_rho": ab,
                    "status": "OK",
                    "cluster": [s[2] for s in scored if s[0] >= CLUSTER_SPEARMAN_ABS],
                    "top3": [s[2] for s in scored[:3]],
                }
            fold_rows.append(rec)
            fold_nearest[str(y)] = rec
            if last_j and rec.get("nearest_legacy") and rec["nearest_legacy"] != last_j:
                changed = True
            if rec.get("nearest_legacy"):
                last_j = rec["nearest_legacy"]
        counts: dict[str, int] = {}
        for r in fold_nearest.values():
            if r.get("nearest_legacy"):
                counts[r["nearest_legacy"]] = counts.get(r["nearest_legacy"], 0) + 1
        majority = max(counts, key=counts.get) if counts else None
        maj_signed = [r["fold_signed_rho"] for r in fold_nearest.values() if r.get("nearest_legacy") == majority and r.get("fold_signed_rho") is not None]
        maj_abs = [r["fold_abs_rho"] for r in fold_nearest.values() if r.get("nearest_legacy") == majority and r.get("fold_abs_rho") is not None]
        nearest_by_factor[feat] = {
            "majority_legacy": majority,
            "fold_changed": changed,
            "mean_fold_signed_rho_majority": float(np.mean(maj_signed)) if maj_signed else None,
            "mean_fold_abs_rho_majority": float(np.mean(maj_abs)) if maj_abs else None,
            "fold_nearest": fold_nearest,
            "do_not_use_full_history_to_pick_drop": True,
        }
    MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix_rows).to_csv(MATRIX_CSV, index=False)
    flat = []
    for r in fold_rows:
        flat.append({**{k: v for k, v in r.items() if k not in {"top3", "cluster"}}, "top3": "|".join(r.get("top3") or []), "cluster": "|".join(r.get("cluster") or [])})
    pd.DataFrame(flat).to_csv(FOLD_CSV, index=False)
    dump_json(ROUND21_2_ROOT / "correlation" / "fold_nearest.json", nearest_by_factor)
    replace_rows = []
    for feat, rec in nearest_by_factor.items():
        replace_rows.append(
            {
                "new_factor": feat,
                "nearest_legacy": rec.get("majority_legacy"),
                "mean_fold_signed_rho": rec.get("mean_fold_signed_rho_majority"),
                "mean_fold_abs_rho": rec.get("mean_fold_abs_rho_majority"),
                "fold_changed": rec.get("fold_changed"),
                "official": "TRAIN_FOLD",
            }
        )
    pd.DataFrame(replace_rows).to_csv(REPLACE_CSV, index=False)
    payload = {
        "status": "DONE",
        "official": "TRAIN_FOLD_CORRELATION",
        "n_new": len(new_factors),
        "new_factors": new_factors,
        "nearest": nearest_by_factor,
        "fold_changed_factors": [k for k, v in nearest_by_factor.items() if v.get("fold_changed")],
        "field_names_explicit": True,
    }
    dump_json(CORR_JSON, payload)
    write_md(
        MD["09"],
        "\n".join(
            [
                "# Round21.2 · 09 相关与冗余",
                "",
                "字段：full_sample_signed_rho / full_sample_abs_rho / fold_signed_rho / fold_abs_rho / mean_daily_rho / median_daily_rho / mean_absolute_daily_rho。",
                "官方最近邻 = 训练窗 |Spearman| 最大。阈值 0.85 只分层。",
                f"折间变化：`{payload['fold_changed_factors']}`",
                "",
            ]
        ),
    )
    nlines = ["# Round21.2 · 11 最近邻替换映射", "", "| new | year | nearest | fold_signed | fold_abs |", "|---|---:|---|---:|---:|"]
    for r in fold_rows:
        nlines.append(f"| `{r.get('new_factor')}` | {r.get('predict_year')} | `{r.get('nearest_legacy')}` | {r.get('fold_signed_rho')} | {r.get('fold_abs_rho')} |")
    write_md(MD["11"], "\n".join(nlines))
    return payload
