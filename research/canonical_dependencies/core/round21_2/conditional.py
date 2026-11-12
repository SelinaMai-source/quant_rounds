"""训练折 Ridge 残差 Conditional IC。禁止全样本正交。"""

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
from .contract import KEYS, MD, PREDICT_YEARS, RIDGE_ALPHA
from .panels import load_research_panel, pit_safe_factors

from ..paths import ROUND21_2_ROOT

COND_JSON = ROUND21_2_ROOT / "conditional" / "conditional.json"
COND_CSV = ROUND21_2_ROOT / "conditional" / "conditional_ic.csv"


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
    factors = [f for f in pit_safe_factors(resume=True) if f in panel.columns]
    nearest = load_json(ROUND21_2_ROOT / "correlation" / "fold_nearest.json")
    rows = []
    for feat in factors:
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
                raw = _ics(te["trade_date"], te[feat].to_numpy(dtype=float), te[label].to_numpy(dtype=float))
                raw_ics.extend(raw)
                cols = [c for c in locked if c in tr.columns and c in te.columns]
                if maj and maj in cols:
                    base_cols = [maj]
                else:
                    base_cols = cols[:1] if cols else []
                if base_cols and feat in tr.columns:
                    xtr = tr[base_cols].to_numpy(dtype=float)
                    xte = te[base_cols].to_numpy(dtype=float)
                    ytr = pd.to_numeric(tr[feat], errors="coerce").to_numpy(dtype=float)
                    yte = pd.to_numeric(te[feat], errors="coerce").to_numpy(dtype=float)
                    mask_tr = np.isfinite(xtr).all(axis=1) & np.isfinite(ytr)
                    mask_te = np.isfinite(xte).all(axis=1) & np.isfinite(yte)
                    resid = _residual(xtr[mask_tr], ytr[mask_tr], xte[mask_te], yte[mask_te])
                    if resid is not None:
                        lab_te = pd.to_numeric(te.loc[mask_te, label], errors="coerce").to_numpy(dtype=float)
                        dates = te.loc[mask_te, "trade_date"]
                        resid_ics.extend(_ics(dates, resid, lab_te))
                cond_ics.extend(raw)
            rec = {
                "factor_id": feat,
                "head": head,
                "label": label,
                "nearest_legacy": maj,
                "raw_rankic": float(np.mean(raw_ics)) if raw_ics else None,
                "conditional_ic": float(np.mean(resid_ics)) if resid_ics else None,
                "conditional_icir": icir_of(pd.Series(resid_ics)) if resid_ics else None,
                "n_raw_days": len(raw_ics),
                "n_resid_days": len(resid_ics),
                "full_sample_orthogonal_forbidden": True,
            }
            rows.append(rec)
            dest = ROUND21_2_ROOT / "factors" / feat
            dest.mkdir(parents=True, exist_ok=True)
            dump_json(dest / f"conditional_{head}.json", rec)
    COND_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(COND_CSV, index=False)
    payload = {"status": "DONE", "n": len(rows), "rows": rows, "ridge_alpha": RIDGE_ALPHA}
    dump_json(COND_JSON, payload)
    write_md(
        MD["10"],
        "\n".join(
            [
                "# Round21.2 · 10 Conditional IC",
                "",
                "只在训练折拟合 Ridge，预测折残差 RankIC。禁止全样本正交。",
                f"N=`{len(rows)}` alpha=`{RIDGE_ALPHA}`",
                "",
            ]
        ),
    )
    return payload
