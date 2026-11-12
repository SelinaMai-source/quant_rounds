"""Fold 内 Ridge 残差 Conditional IC。禁止全样本正交。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..round20.metrics import icir_of, rank_ic
from ..round20.panel import fold_masks
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, TEN_FEATURES, TEN_HEAD
from .common import dump_json, finite_or_none, load_json, write_md
from .contract import COND_MD, KEYS, PREDICT_YEARS, RIDGE_ALPHA
from .panels import load_research_panel, pit_safe_factors

from ..paths import ROUND21_1_ROOT

COND_JSON = ROUND21_1_ROOT / "conditional" / "conditional.json"
COND_CSV = ROUND21_1_ROOT / "conditional" / "conditional_ic.csv"


def _fold_map_safe() -> dict[str, Any]:
    return load_json(ROUND21_1_ROOT / "correlation" / "fold_nearest.json")


def _residual(x_tr, y_tr, x_te, y_te) -> np.ndarray | None:
    if len(x_tr) < 30 or x_tr.shape[1] == 0 or len(x_te) == 0:
        return None
    scaler = StandardScaler()
    try:
        xt = scaler.fit_transform(x_tr)
        xe = scaler.transform(x_te)
    except ValueError:
        return None
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(xt, y_tr)
    return y_te - model.predict(xe)


def _ics(dates: pd.Series, score: np.ndarray, label: np.ndarray) -> list[float]:
    work = pd.DataFrame({"d": pd.to_datetime(dates).to_numpy(), "s": score, "y": label})
    out = []
    for _, day in work.groupby("d", sort=False):
        ic = rank_ic(day["s"], day["y"])
        if ic is not None:
            out.append(ic)
    return out


def run_conditional(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and COND_JSON.is_file() and COND_CSV.is_file():
        existing = load_json(COND_JSON)
        if existing.get("status") == "DONE":
            return existing
    panel = load_research_panel(resume=True)
    factors = pit_safe_factors(resume=True)
    nearest = _fold_map_safe()
    rows = []
    for feat in factors:
        if feat not in panel.columns:
            continue
        maj = (nearest.get(feat) or {}).get("majority_legacy")
        for label, locked, head in (
            (FIVE_HEAD.target, list(FIVE_FEATURES), "five"),
            (TEN_HEAD.target, list(TEN_FEATURES), "ten"),
        ):
            if label not in panel.columns:
                continue
            raw_ics, cond_ics, resid_ics = [], [], []
            for y in PREDICT_YEARS:
                train_m, pred_m = fold_masks(panel, predict_year=y, label=label)
                tr = panel.loc[train_m]
                te = panel.loc[pred_m]
                if feat not in te.columns or label not in te.columns:
                    continue
                raw = _ics(te["trade_date"], te[feat].to_numpy(dtype=float), te[label].to_numpy(dtype=float))
                raw_ics.extend(raw)
                ctrl = [maj] if maj and maj in te.columns and maj in tr.columns else []
                if ctrl:
                    trc = tr[[feat, *ctrl, label]].replace([np.inf, -np.inf], np.nan).dropna()
                    tec = te[[feat, *ctrl, label, "trade_date"]].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(trc) >= 30 and len(tec) >= 5:
                        resid = _residual(
                            trc[ctrl].to_numpy(dtype=float),
                            trc[feat].to_numpy(dtype=float),
                            tec[ctrl].to_numpy(dtype=float),
                            tec[feat].to_numpy(dtype=float),
                        )
                        if resid is not None:
                            cond_ics.extend(_ics(tec["trade_date"], resid, tec[label].to_numpy(dtype=float)))
                use_old = [c for c in locked if c in tr.columns and c in te.columns]
                trc = tr[[feat, *use_old, label]].replace([np.inf, -np.inf], np.nan).dropna()
                tec = te[[feat, *use_old, label, "trade_date"]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(trc) >= 30 and len(tec) >= 5:
                    resid = _residual(
                        trc[use_old].to_numpy(dtype=float),
                        trc[feat].to_numpy(dtype=float),
                        tec[use_old].to_numpy(dtype=float),
                        tec[feat].to_numpy(dtype=float),
                    )
                    if resid is not None:
                        resid_ics.extend(_ics(tec["trade_date"], resid, tec[label].to_numpy(dtype=float)))
            # top-tail: 每日新因子 top20%
            tail = []
            for _, day in panel.groupby("trade_date", sort=False):
                s = pd.to_numeric(day[feat], errors="coerce")
                yv = pd.to_numeric(day[label], errors="coerce")
                ok = np.isfinite(s) & np.isfinite(yv)
                if int(ok.sum()) < 8:
                    continue
                q = np.nanquantile(s[ok], 0.80)
                pick = ok & (s >= q)
                if int(pick.sum()) < 3:
                    continue
                ic = rank_ic(s[pick], yv[pick])
                if ic is not None:
                    tail.append(ic)
            rec = {
                "factor_id": feat,
                "head": head,
                "label": label,
                "nearest_legacy": maj,
                "raw_ic": float(np.mean(raw_ics)) if raw_ics else None,
                "raw_icir": icir_of(np.array(raw_ics, dtype=float)) if raw_ics else None,
                "conditional_ic": float(np.mean(cond_ics)) if cond_ics else None,
                "conditional_icir": icir_of(np.array(cond_ics, dtype=float)) if cond_ics else None,
                "residual_ic_all_baseline": float(np.mean(resid_ics)) if resid_ics else None,
                "top_tail_conditional_ic": float(np.mean(tail)) if tail else None,
                "n_raw_days": len(raw_ics),
                "n_cond_days": len(cond_ics),
                "do_not_auto_drop_if_cond_zero": True,
                "fit": "train_fold_ridge_only",
            }
            rows.append(rec)
            dest = ROUND21_1_ROOT / "factors" / feat
            dest.mkdir(parents=True, exist_ok=True)
            dump_json(dest / "conditional.json", rec)
    frame = pd.DataFrame(rows)
    COND_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(COND_CSV, index=False)
    payload = {"status": "DONE", "n": len(rows), "csv": str(COND_CSV), "rows": rows}
    dump_json(COND_JSON, payload)
    lines = [
        "# Round21.1 · 06 Conditional IC",
        "",
        "Train-fold Ridge 残差。禁止全样本正交。Conditional IC≈0 不自动删除。",
        "",
        "| factor | head | raw IC | cond IC | residual IC | top-tail | nearest |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['factor_id']}` | {r['head']} | {r['raw_ic']} | {r['conditional_ic']} | "
            f"{r['residual_ic_all_baseline']} | {r['top_tail_conditional_ic']} | `{r['nearest_legacy']}` |"
        )
    write_md(COND_MD, "\n".join(lines))
    return payload
