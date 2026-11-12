"""新因子 vs 真实 14/13。FULL_HISTORY 诊断；TRAIN_FOLD 主。j*=argmax |Spearman|。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_1_ROOT
from ..round20.panel import fold_masks
from .common import dump_json, load_json, write_md
from .contract import (
    CLUSTER_SPEARMAN_ABS,
    CORR_MD,
    FIVE_FEATURES,
    FIVE_HEAD,
    KEYS,
    LABELS,
    NEAREST_LEGACY_NOT_IDENTIFIABLE,
    NEAREST_MD,
    PREDICT_YEARS,
    TEN_FEATURES,
)
from .panels import load_research_panel, pit_safe_factors

CORR_JSON = ROUND21_1_ROOT / "correlation" / "correlation.json"
MATRIX_CSV = ROUND21_1_ROOT / "correlation" / "new_legacy_correlation_matrix.csv"
DAILY_PARQ = ROUND21_1_ROOT / "correlation" / "daily_cross_sectional_correlation.parquet"
FOLD_CSV = ROUND21_1_ROOT / "correlation" / "fold_nearest_legacy_mapping.csv"


def _spearman(a: pd.Series, b: pd.Series) -> float | None:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    if int(mask.sum()) < 5:
        return None
    rho = x.loc[mask].corr(y.loc[mask], method="spearman")
    if rho is None or not np.isfinite(rho):
        return None
    return float(rho)


def _pearson(a: pd.Series, b: pd.Series) -> float | None:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    if int(mask.sum()) < 5:
        return None
    rho = x.loc[mask].corr(y.loc[mask], method="pearson")
    if rho is None or not np.isfinite(rho):
        return None
    return float(rho)


def _daily_cs_all(frame: pd.DataFrame, new_cols: list[str], old_cols: list[str]) -> pd.DataFrame:
    """每日一张 Spearman 相关阵，避免 新×旧×日 次独立 corr。"""
    cols = list(dict.fromkeys([*new_cols, *old_cols]))
    work = frame[["trade_date", *cols]].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    rows = []
    for day, g in work.groupby("trade_date", sort=True):
        n_day = int(len(g))
        if n_day < 5:
            continue
        with np.errstate(invalid="ignore"):
            cm = g[cols].corr(method="spearman")
        for new in new_cols:
            for old in old_cols:
                rho = cm.loc[new, old] if new in cm.index and old in cm.columns else np.nan
                both = int(
                    (
                        np.isfinite(g[new].to_numpy(dtype=float))
                        & np.isfinite(g[old].to_numpy(dtype=float))
                    ).sum()
                )
                if both < 5 or rho is None or (isinstance(rho, float) and not np.isfinite(rho)):
                    rho = None
                else:
                    rho = float(rho)
                rows.append(
                    {
                        "trade_date": pd.Timestamp(day),
                        "new_factor": new,
                        "legacy_factor": old,
                        "spearman": rho,
                        "n_matched": both,
                        "n_day": n_day,
                        "overlap_coverage": both / n_day if n_day else None,
                    }
                )
    return pd.DataFrame(rows)


def run_correlate(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and CORR_JSON.is_file() and MATRIX_CSV.is_file() and FOLD_CSV.is_file() and DAILY_PARQ.is_file():
        existing = load_json(CORR_JSON)
        if existing.get("status") == "DONE":
            return existing

    panel = load_research_panel(resume=True)
    new_factors = pit_safe_factors(resume=True)
    old_cols = list(dict.fromkeys([*FIVE_FEATURES, *TEN_FEATURES]))
    missing_old = [c for c in old_cols if c not in panel.columns]
    if missing_old:
        raise RuntimeError(f"research panel missing real 14/13: {missing_old}")

    matrix_rows = []
    fold_rows = []
    nearest_by_factor: dict[str, Any] = {}
    usable = [f for f in new_factors if f in panel.columns]
    print(f"[round21.1] correlate daily-cs {len(usable)} x {len(old_cols)}", flush=True)
    daily_all = _daily_cs_all(panel, usable, old_cols)
    print(f"[round21.1] correlate daily-cs rows={len(daily_all)}", flush=True)

    for feat in usable:
        hist = []
        for old in old_cols:
            pear = _pearson(panel[feat], panel[old])
            spear = _spearman(panel[feat], panel[old])
            both = int(
                (
                    np.isfinite(pd.to_numeric(panel[feat], errors="coerce").to_numpy(dtype=float))
                    & np.isfinite(pd.to_numeric(panel[old], errors="coerce").to_numpy(dtype=float))
                ).sum()
            )
            rec = {
                "new_factor": feat,
                "legacy_factor": old,
                "scope": "FULL_HISTORY_DIAGNOSTIC",
                "pearson": pear,
                "spearman": spear,
                "abs_spearman": None if spear is None else abs(spear),
                "matched_n": both,
                "overlap_coverage": both / len(panel) if len(panel) else None,
                "independent_oos": False,
            }
            hist.append(rec)
            matrix_rows.append(rec)
        if not daily_all.empty:
            daily_feat = daily_all.loc[daily_all["new_factor"].eq(feat)]
            for old in old_cols:
                sub = daily_feat.loc[daily_feat["legacy_factor"].eq(old)]
                vals = pd.to_numeric(sub["spearman"], errors="coerce")
                matrix_rows.append(
                    {
                        "new_factor": feat,
                        "legacy_factor": old,
                        "scope": "FULL_HISTORY_DAILY_CS",
                        "mean_daily_spearman": float(vals.mean()) if vals.notna().any() else None,
                        "median_daily_spearman": float(vals.median()) if vals.notna().any() else None,
                        "n_days": int(vals.notna().sum()),
                        "independent_oos": False,
                    }
                )

        fold_nearest = {}
        fold_changed = False
        last_j = None
        for y in PREDICT_YEARS:
            train_m, _ = fold_masks(panel, predict_year=y, label=FIVE_HEAD.target)
            tr = panel.loc[train_m]
            scored = []
            for old in old_cols:
                rho = _spearman(tr[feat], tr[old])
                if rho is None:
                    continue
                scored.append((abs(rho), rho, old, int(
                    (
                        np.isfinite(pd.to_numeric(tr[feat], errors="coerce").to_numpy(dtype=float))
                        & np.isfinite(pd.to_numeric(tr[old], errors="coerce").to_numpy(dtype=float))
                    ).sum()
                )))
            scored.sort(reverse=True)
            if not scored:
                jstar, signed, ab, n_m = None, None, None, 0
                status = NEAREST_LEGACY_NOT_IDENTIFIABLE
            else:
                ab, signed, jstar, n_m = scored[0]
                status = "OK"
            top3 = [{"legacy": s[2], "abs_spearman": s[0], "spearman": s[1], "n": s[3]} for s in scored[:3]]
            cluster = [s[2] for s in scored if s[0] >= CLUSTER_SPEARMAN_ABS]
            rec = {
                "new_factor": feat,
                "predict_year": y,
                "train_years": [y - 3, y - 1],
                "nearest_legacy": jstar,
                "spearman": signed,
                "abs_spearman": ab,
                "matched_n": n_m,
                "status": status,
                "top3": top3,
                "cluster_abs_ge_0_85": cluster,
                "official": True,
            }
            fold_rows.append(rec)
            fold_nearest[str(y)] = rec
            if last_j is not None and jstar is not None and jstar != last_j:
                fold_changed = True
            if jstar is not None:
                last_j = jstar

        # 主实验最近邻：折间不同必须报告，禁止挑全历史最好
        official = [fold_nearest[str(y)] for y in PREDICT_YEARS if fold_nearest.get(str(y), {}).get("nearest_legacy")]
        counts: dict[str, int] = {}
        for r in official:
            counts[r["nearest_legacy"]] = counts.get(r["nearest_legacy"], 0) + 1
        majority = max(counts, key=counts.get) if counts else None
        nearest_by_factor[feat] = {
            "fold_nearest": {k: {"nearest_legacy": v.get("nearest_legacy"), "abs_spearman": v.get("abs_spearman"), "status": v.get("status")} for k, v in fold_nearest.items()},
            "fold_changed": fold_changed,
            "majority_legacy": majority,
            "per_fold_required": fold_changed,
            "full_history_top1": max(
                (r for r in hist if r.get("abs_spearman") is not None),
                key=lambda r: r["abs_spearman"],
                default={},
            ).get("legacy_factor"),
            "do_not_use_full_history_to_pick_drop": True,
        }

    MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix_rows).to_csv(MATRIX_CSV, index=False)
    if not daily_all.empty:
        daily_all.to_parquet(DAILY_PARQ, index=False)
    fold_flat = []
    for r in fold_rows:
        fold_flat.append({k: v for k, v in r.items() if k != "top3"})
        fold_flat[-1]["top3"] = "|".join(
            f"{t['legacy']}:{t['abs_spearman']}" for t in (r.get("top3") or [])
        )
        fold_flat[-1]["cluster"] = "|".join(r.get("cluster_abs_ge_0_85") or [])
    pd.DataFrame(fold_flat).to_csv(FOLD_CSV, index=False)

    payload = {
        "status": "DONE",
        "universe": "first_board_candidates trade_date x instrument_id",
        "legacy_source": "core/round7/contract.py FIVE_FEATURES 14 / TEN_FEATURES 13",
        "full_history_is_diagnostic_only": True,
        "official": "TRAIN_FOLD_CORRELATION",
        "j_star_rule": "argmax |Spearman|; NaN not ranked as 0",
        "threshold_0_85_diagnostic_only": True,
        "top1_replace_only": True,
        "n_new": len(new_factors),
        "new_factors": new_factors,
        "nearest": nearest_by_factor,
        "fold_changed_factors": [k for k, v in nearest_by_factor.items() if v.get("fold_changed")],
        "matrix_csv": str(MATRIX_CSV),
        "daily_parquet": str(DAILY_PARQ),
        "fold_csv": str(FOLD_CSV),
    }
    dump_json(CORR_JSON, payload)
    dump_json(ROUND21_1_ROOT / "correlation" / "fold_nearest.json", nearest_by_factor)
    lines = [
        "# Round21.1 · 03 新因子 vs 真实 14/13 相关",
        "",
        "对齐：一进二首板候选 `trade_date × instrument_id`。旧列仅 `FIVE_FEATURES`/`TEN_FEATURES`。",
        "A=`FULL_HISTORY_DIAGNOSTIC` 禁止当独立 OOS。B=`TRAIN_FOLD_CORRELATION` 官方。",
        f"`j* = argmax |Spearman|`。阈值 {CLUSTER_SPEARMAN_ABS} 只诊断。主实验只替换 Top1。",
        "",
        f"折间最近邻变化：`{payload['fold_changed_factors']}`",
        "",
    ]
    write_md(CORR_MD, "\n".join(lines))
    nlines = [
        "# Round21.1 · 04 折内最近邻映射",
        "",
        "预测年 Y 只用 Y 之前训练日。样本不足 → `NEAREST_LEGACY_NOT_IDENTIFIABLE`。",
        "",
        "| new | year | nearest | |ρ| | n | cluster≥0.85 |",
        "|---|---:|---|---:|---:|---|",
    ]
    for r in fold_rows:
        nlines.append(
            f"| `{r['new_factor']}` | {r['predict_year']} | `{r['nearest_legacy']}` | "
            f"{r['abs_spearman']} | {r['matched_n']} | {r.get('cluster_abs_ge_0_85')} |"
        )
    write_md(NEAREST_MD, "\n".join(nlines))
    return payload
