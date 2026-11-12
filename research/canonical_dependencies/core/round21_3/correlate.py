"""相关只对真实 14/13 + zoo 参考池。TRAIN_FOLD 选最近邻。|ρ| 低时 E3 不得称去冗余。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_3_ROOT
from ..round7.contract import FIVE_FEATURES, TEN_FEATURES
from .common import dump_json, write_md
from .contract import CLUSTER_SPEARMAN_ABS, FEATURE_RECOMPOSITION, KEYS, MD, NEW_FACTOR_IDS, PREDICT_YEARS
from .panels_local import load_research_panel
from .pit import run_pit

CORR_JSON = ROUND21_3_ROOT / "correlation" / "correlation.json"
FOLD_JSON = ROUND21_3_ROOT / "correlation" / "fold_nearest.json"
FOLD_CSV = ROUND21_3_ROOT / "correlation" / "fold_nearest.csv"


def _spearman(a: pd.Series, b: pd.Series) -> float | None:
    mask = a.notna() & b.notna()
    if int(mask.sum()) < 30:
        return None
    if a.loc[mask].nunique() < 2 or b.loc[mask].nunique() < 2:
        return None
    return float(a.loc[mask].corr(b.loc[mask], method="spearman"))


def run_correlate(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and CORR_JSON.is_file() and FOLD_CSV.is_file() and MD["09"].is_file():
        from .common import load_json

        existing = load_json(CORR_JSON)
        if existing.get("status") == "DONE":
            return existing
    pit = run_pit(resume=True)
    panel = load_research_panel(resume=True)
    legacy = [c for c in list(FIVE_FEATURES) + list(TEN_FEATURES) if c in panel.columns]
    legacy = list(dict.fromkeys(legacy))
    factors = [c for c in NEW_FACTOR_IDS if c in panel.columns]
    dups = set(pit.get("duplicates") or [])
    factors = [c for c in factors if c not in dups]
    fold_rows = []
    nearest: dict[str, Any] = {}
    for feat in factors:
        mapping = {}
        rhos = []
        for y in PREDICT_YEARS:
            sub = panel.loc[panel["trade_date"].dt.year.eq(y)]
            best_j, best_abs, best_s = None, -1.0, None
            for j in legacy:
                rho = _spearman(pd.to_numeric(sub[feat], errors="coerce"), pd.to_numeric(sub[j], errors="coerce"))
                if rho is None:
                    continue
                if abs(rho) > best_abs:
                    best_abs, best_j, best_s = abs(rho), j, rho
            if best_j:
                mapping[str(y)] = best_j
                rhos.append(best_s)
                fold_rows.append({"new_factor": feat, "predict_year": y, "nearest_legacy": best_j, "fold_signed_rho": best_s, "fold_abs_rho": best_abs, "cluster": best_j})
        counts: dict[str, int] = {}
        for v in mapping.values():
            counts[v] = counts.get(v, 0) + 1
        majority = max(counts, key=counts.get) if counts else None
        mean_s = float(np.mean(rhos)) if rhos else None
        nearest[feat] = {
            "majority_legacy": majority,
            "mean_fold_signed": mean_s,
            "mean_fold_abs": float(np.mean(np.abs(rhos))) if rhos else None,
            "fold_changed": len(set(mapping.values())) > 1,
            "mapping": mapping,
            "recomposition": bool(mean_s is not None and abs(mean_s) < 0.20),
            "label": FEATURE_RECOMPOSITION if (mean_s is not None and abs(mean_s) < 0.20) else "NEAR",
        }
    FOLD_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(FOLD_CSV, index=False)
    dump_json(FOLD_JSON, {"nearest": nearest})
    high = [k for k, v in nearest.items() if (v.get("mean_fold_abs") or 0) >= CLUSTER_SPEARMAN_ABS]
    payload = {
        "status": "DONE",
        "nearest": nearest,
        "high_corr": high,
        "fold_changed_factors": [k for k, v in nearest.items() if v.get("fold_changed")],
        "universe": "real_14_13",
        "low_rho_not_deredundancy": True,
    }
    dump_json(CORR_JSON, payload)
    write_md(
        MD["09"],
        f"# Round21.3 · 09 相关\n\n只对真实 14/13。|ρ| 低不得称去冗余（{FEATURE_RECOMPOSITION}）。高相关=`{high}`。\n",
    )
    write_md(MD["11"], f"# Round21.3 · 11 最近邻\n\nTRAIN_FOLD `j*=argmax |Spearman|`。nearest=`{ {k: v.get('majority_legacy') for k, v in nearest.items()} }`\n")
    return payload
