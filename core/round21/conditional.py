"""Fold 内 Ridge 残差 Conditional IC。禁止全历史 g()。锁定 14/13 LOFO 诊断不得删列。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..paths import ROUND21_ROOT
from ..round7.contract import FIVE_FEATURES, TEN_FEATURES
from ..round20.metrics import icir_of, rank_ic
from ..round20.panel import fold_masks
from .common import dump_json, load_json, write_md
from .contract import (
    BLOCKED_PIT,
    CONDITIONAL_MD,
    IC_YEARS,
    PCA_EXPLAINED_VARIANCE,
    RIDGE_ALPHA,
)
from .freeze import require_frozen
from .library import build_research_panel
from .panel import ic_window
from .pit import run_pit
from .redundancy import KEEP_CSV, run_redundancy

COND_JSON = ROUND21_ROOT / "conditional" / "conditional.json"
COND_CSV = ROUND21_ROOT / "conditional" / "conditional_ic.csv"
ORTHO_CSV = ROUND21_ROOT / "conditional" / "ortho_pca.csv"
LOFO_CSV = ROUND21_ROOT / "conditional" / "lofo.csv"
FACTORS_DIR = ROUND21_ROOT / "factors"

HEADS = (
    ("label_e2_net", tuple(FIVE_FEATURES)),
    ("label_exit_rule_net", tuple(TEN_FEATURES)),
)


def _complete(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    use = [c for c in cols if c in frame.columns]
    work = frame[use].replace([np.inf, -np.inf], np.nan)
    return frame.loc[work.notna().all(axis=1)].copy()


def _fit_residual(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> np.ndarray | None:
    if len(x_train) < 30 or x_train.shape[1] == 0 or len(x_test) == 0:
        return None
    scaler = StandardScaler()
    try:
        xt = scaler.fit_transform(x_train)
        xe = scaler.transform(x_test)
    except ValueError:
        return None
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(xt, y_train)
    return y_test - model.predict(xe)


def _ridge_score(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray | None:
    if len(x_train) < 30 or x_train.shape[1] == 0 or len(x_test) == 0:
        return None
    scaler = StandardScaler()
    try:
        xt = scaler.fit_transform(x_train)
        xe = scaler.transform(x_test)
    except ValueError:
        return None
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(xt, y_train)
    return model.predict(xe)


def _daily_rankic(dates: pd.Series, score: np.ndarray, label: np.ndarray) -> list[float]:
    work = pd.DataFrame({"d": pd.to_datetime(dates).to_numpy(), "s": score, "y": label})
    out: list[float] = []
    for _, day in work.groupby("d", sort=False):
        ic = rank_ic(day["s"], day["y"])
        if ic is not None:
            out.append(ic)
    return out


def run_conditional(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and COND_JSON.is_file() and COND_CSV.is_file() and CONDITIONAL_MD.is_file():
        existing = load_json(COND_JSON)
        if existing.get("status") == "DONE":
            return existing
    red = run_redundancy(resume=True)
    keep_frame = pd.read_csv(KEEP_CSV)
    keep_map = {str(r["name"]): r for r in keep_frame.to_dict(orient="records")}
    pit = run_pit(resume=True)
    pit_map = {r["factor_id"]: r for r in (pit.get("rows") or [])}
    panel = ic_window(build_research_panel(resume=True))
    new_names = [str(x) for x in keep_frame["name"].tolist() if str(x) in panel.columns]
    years = [y for y in IC_YEARS if int((panel["trade_date"].dt.year == y).sum()) > 0]

    cond_rows: list[dict[str, Any]] = []
    for lab, locked in HEADS:
        lock = [c for c in locked if c in panel.columns]
        for feat in new_names:
            if feat in lock:
                continue
            oos: list[float] = []
            n_folds = 0
            skipped = 0
            for y in years:
                train_m, pred_m = fold_masks(panel, predict_year=y, label=lab)
                cols = [*lock, feat, lab]
                tr = _complete(panel.loc[train_m], cols)
                te = _complete(panel.loc[pred_m], cols)
                if len(tr) < 80 or len(te) < 20:
                    skipped += 1
                    continue
                resid = _fit_residual(
                    tr[lock].to_numpy(dtype=float),
                    pd.to_numeric(tr[feat], errors="coerce").to_numpy(dtype=float),
                    te[lock].to_numpy(dtype=float),
                    pd.to_numeric(te[feat], errors="coerce").to_numpy(dtype=float),
                )
                if resid is None:
                    skipped += 1
                    continue
                n_folds += 1
                oos.extend(_daily_rankic(te["trade_date"], resid, pd.to_numeric(te[lab], errors="coerce").to_numpy(dtype=float)))
            arr = np.array(oos, dtype=float)
            row = {
                "factor": feat,
                "label": lab,
                "n_oos_days": int(len(arr)),
                "n_folds": n_folds,
                "n_folds_skipped": skipped,
                "cond_rankic_mean": float(np.mean(arr)) if len(arr) else None,
                "cond_rankicir": icir_of(arr) if len(arr) else None,
                "raw_action": (keep_map.get(feat) or {}).get("action"),
                "family": (keep_map.get(feat) or {}).get("family"),
                "pit_status": (pit_map.get(feat) or {}).get("pit_status"),
                "blocked_pit": (pit_map.get(feat) or {}).get("pit_status") == BLOCKED_PIT,
                "full_history_fit": False,
                "ridge_alpha": RIDGE_ALPHA,
                "controls": ",".join(lock),
            }
            cond_rows.append(row)
    cond = pd.DataFrame(cond_rows)
    COND_CSV.parent.mkdir(parents=True, exist_ok=True)
    cond.to_csv(COND_CSV, index=False)

    for feat in new_names:
        sub = cond.loc[cond["factor"].eq(feat)]
        dump_json(
            FACTORS_DIR / feat / "conditional.json",
            {
                "factor_id": feat,
                "full_history_fit": False,
                "ridge_alpha": RIDGE_ALPHA,
                "rows": sub.to_dict(orient="records"),
            },
        )

    families = sorted({str(v) for v in keep_frame["family"].dropna().unique()})
    ortho_rows: list[dict[str, Any]] = []
    for fam in families:
        members = [
            str(n)
            for n in keep_frame.loc[keep_frame["family"].eq(fam), "name"].tolist()
            if str(n) in panel.columns
            and str((keep_map.get(n) or {}).get("action", "")).startswith("KEEP")
            and str((keep_map.get(n) or {}).get("pit_status")) == "PIT_PASS"
        ]
        if not members:
            continue
        for lab, locked in HEADS:
            oos_pruned: list[float] = []
            oos_ortho: list[float] = []
            oos_pca: list[float] = []
            n_comp: list[int] = []
            n_folds = 0
            for y in years:
                train_m, pred_m = fold_masks(panel, predict_year=y, label=lab)
                cols = [*members, lab]
                tr = _complete(panel.loc[train_m], cols)
                te = _complete(panel.loc[pred_m], cols)
                if len(tr) < 80 or len(te) < 20:
                    continue
                n_folds += 1
                ics = []
                for m in members:
                    ric = rank_ic(tr[m], tr[lab])
                    ics.append((abs(ric or 0.0), ric or 0.0, m))
                ics.sort(reverse=True)
                chosen: list[str] = []
                for _, _, m in ics:
                    ok = True
                    for c in chosen:
                        rho = pd.to_numeric(tr[m], errors="coerce").corr(pd.to_numeric(tr[c], errors="coerce"), method="spearman")
                        if rho is not None and np.isfinite(rho) and abs(float(rho)) >= 0.85:
                            ok = False
                            break
                    if ok:
                        chosen.append(m)
                    if len(chosen) >= 3:
                        break
                if chosen:
                    oos_pruned.extend(
                        _daily_rankic(
                            te["trade_date"],
                            pd.to_numeric(te[chosen[0]], errors="coerce").to_numpy(dtype=float),
                            pd.to_numeric(te[lab], errors="coerce").to_numpy(dtype=float),
                        )
                    )
                ordered = [m for _, _, m in ics]
                if ordered:
                    yte = pd.to_numeric(te[lab], errors="coerce").to_numpy(dtype=float)
                    oos_ortho.extend(
                        _daily_rankic(
                            te["trade_date"],
                            pd.to_numeric(te[ordered[0]], errors="coerce").to_numpy(dtype=float),
                            yte,
                        )
                    )
                    basis_tr = pd.to_numeric(tr[ordered[0]], errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
                    basis_te = pd.to_numeric(te[ordered[0]], errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
                    last_resid = None
                    for m in ordered[1:]:
                        last_resid = _fit_residual(
                            basis_tr,
                            pd.to_numeric(tr[m], errors="coerce").to_numpy(dtype=float),
                            basis_te,
                            pd.to_numeric(te[m], errors="coerce").to_numpy(dtype=float),
                        )
                        basis_tr = np.column_stack([basis_tr, pd.to_numeric(tr[m], errors="coerce").to_numpy(dtype=float)])
                        basis_te = np.column_stack([basis_te, pd.to_numeric(te[m], errors="coerce").to_numpy(dtype=float)])
                    if last_resid is not None:
                        oos_ortho.extend(_daily_rankic(te["trade_date"], last_resid, yte))
                Xtr = tr[members].to_numpy(dtype=float)
                Xte = te[members].to_numpy(dtype=float)
                if Xtr.shape[1] >= 1 and len(Xtr) > Xtr.shape[1]:
                    scaler = StandardScaler()
                    try:
                        Ztr = scaler.fit_transform(Xtr)
                        Zte = scaler.transform(Xte)
                        pca = PCA(n_components=PCA_EXPLAINED_VARIANCE, svd_solver="full")
                        pca.fit(Ztr)
                        Tte = pca.transform(Zte)
                        n_comp.append(int(Tte.shape[1]))
                        oos_pca.extend(_daily_rankic(te["trade_date"], Tte[:, 0], pd.to_numeric(te[lab], errors="coerce").to_numpy(dtype=float)))
                    except Exception:
                        pass

            def _pack(xs: list[float]) -> tuple[float | None, float | None, int]:
                arr = np.array(xs, dtype=float)
                arr = arr[np.isfinite(arr)]
                if len(arr) == 0:
                    return None, None, 0
                return float(np.mean(arr)), icir_of(arr), int(len(arr))

            p_mean, p_ir, p_n = _pack(oos_pruned)
            o_mean, o_ir, o_n = _pack(oos_ortho)
            c_mean, c_ir, c_n = _pack(oos_pca)
            scored = [
                ("pruned", p_ir if p_ir is not None else -1e9, p_mean),
                ("ortho", o_ir if o_ir is not None else -1e9, o_mean),
                ("pca", c_ir if c_ir is not None else -1e9, c_mean),
            ]
            scored.sort(key=lambda t: (t[1], abs(t[2] or 0)), reverse=True)
            winner = scored[0][0] if any(s[1] > -1e8 for s in scored) else "NONE"
            ortho_rows.append(
                {
                    "family": fam,
                    "label": lab,
                    "n_members_keep": len(members),
                    "members": ",".join(members),
                    "n_folds": n_folds,
                    "pruned_rankic": p_mean,
                    "pruned_icir": p_ir,
                    "pruned_n": p_n,
                    "ortho_rankic": o_mean,
                    "ortho_icir": o_ir,
                    "ortho_n": o_n,
                    "pca_rankic": c_mean,
                    "pca_icir": c_ir,
                    "pca_n": c_n,
                    "pca_median_n_comp": float(np.median(n_comp)) if n_comp else None,
                    "pca_explained_variance": PCA_EXPLAINED_VARIANCE,
                    "winner": winner,
                    "full_history_fit": False,
                }
            )
    ortho = pd.DataFrame(ortho_rows)
    ortho.to_csv(ORTHO_CSV, index=False)

    lofo_rows: list[dict[str, Any]] = []
    for lab, locked in HEADS:
        lock = [c for c in locked if c in panel.columns]
        full_ics: list[float] = []
        drop_ics: dict[str, list[float]] = {c: [] for c in lock}
        n_folds = 0
        for y in years:
            train_m, pred_m = fold_masks(panel, predict_year=y, label=lab)
            cols = [*lock, lab]
            tr = _complete(panel.loc[train_m], cols)
            te = _complete(panel.loc[pred_m], cols)
            if len(tr) < 80 or len(te) < 20:
                continue
            n_folds += 1
            ytr = pd.to_numeric(tr[lab], errors="coerce").to_numpy(dtype=float)
            yte = pd.to_numeric(te[lab], errors="coerce").to_numpy(dtype=float)
            full = _ridge_score(tr[lock].to_numpy(dtype=float), ytr, te[lock].to_numpy(dtype=float))
            if full is not None:
                full_ics.extend(_daily_rankic(te["trade_date"], full, yte))
            for drop in lock:
                use = [c for c in lock if c != drop]
                sc = _ridge_score(tr[use].to_numpy(dtype=float), ytr, te[use].to_numpy(dtype=float))
                if sc is not None:
                    drop_ics[drop].extend(_daily_rankic(te["trade_date"], sc, yte))
        full_arr = np.array(full_ics, dtype=float)
        full_mean = float(np.mean(full_arr)) if len(full_arr) else None
        full_ir = icir_of(full_arr) if len(full_arr) else None
        for feat in lock:
            arr = np.array(drop_ics[feat], dtype=float)
            mean = float(np.mean(arr)) if len(arr) else None
            delta = (full_mean - mean) if (full_mean is not None and mean is not None) else None
            lofo_rows.append(
                {
                    "feature": feat,
                    "label": lab,
                    "n_locked": len(lock),
                    "n_folds": n_folds,
                    "full_rankic": full_mean,
                    "full_icir": full_ir,
                    "lofo_rankic": mean,
                    "lofo_icir": icir_of(arr) if len(arr) else None,
                    "delta_rankic_full_minus_lofo": delta,
                    "delete_from_v5": False,
                    "diagnostic_only": True,
                    "note": "Ridge 排序代理。不得从锁定 v5 删列。",
                }
            )
    lofo = pd.DataFrame(lofo_rows)
    lofo.to_csv(LOFO_CSV, index=False)

    leaders = (
        cond.loc[cond["blocked_pit"].eq(False) & cond["cond_rankicir"].notna()]
        .assign(abs_ir=lambda d: d["cond_rankicir"].abs())
        .sort_values("abs_ir", ascending=False)
        .head(10)
        .drop(columns=["abs_ir"])
        .to_dict(orient="records")
        if len(cond)
        else []
    )
    payload = {
        "status": "DONE",
        "full_history_fit": False,
        "ridge_alpha": RIDGE_ALPHA,
        "pca_explained_variance": PCA_EXPLAINED_VARIANCE,
        "lofo_delete_from_v5": False,
        "n_conditional_rows": int(len(cond)),
        "n_ortho_rows": int(len(ortho)),
        "n_lofo_rows": int(len(lofo)),
        "leaders": leaders,
        "ortho_winner_counts": ortho["winner"].value_counts().to_dict() if len(ortho) else {},
        "csv": str(COND_CSV),
        "ortho_csv": str(ORTHO_CSV),
        "lofo_csv": str(LOFO_CSV),
        "redundancy_n_keep": red.get("n_keep"),
        "note": "Ridge 残差 / PCA 均 fold 内拟合。禁止全历史 g()。LOFO 不得删 v5。BLOCKED_PIT 不晋级。",
    }
    dump_json(COND_JSON, payload)
    lines = [
        "# 06 Conditional IC / 正交 / PCA / LOFO",
        "",
        f"Ridge α=`{RIDGE_ALPHA}`，预测年 Y ← 训 Y-3..Y-1，**禁止全历史拟合**。",
        f"PCA `explained_variance_ratio={PCA_EXPLAINED_VARIANCE}`，不搜阈值。",
        "",
        "## Conditional RankIC 领先（排除 BLOCKED_PIT）",
        "",
        "| factor | label | cond RankIC | cond ICIR | n_oos_days | action |",
        "|---|---|---:|---:|---:|---|",
    ]
    for rec in leaders:
        lines.append(
            f"| `{rec['factor']}` | {rec['label']} | {rec.get('cond_rankic_mean')} | "
            f"{rec.get('cond_rankicir')} | {rec.get('n_oos_days')} | {rec.get('raw_action')} |"
        )
    lines += ["", f"conditional：`{COND_CSV}`", f"ortho/pca：`{ORTHO_CSV}`", f"lofo：`{LOFO_CSV}`", ""]
    write_md(CONDITIONAL_MD, "\n".join(lines))
    return payload


__all__ = ["COND_CSV", "COND_JSON", "LOFO_CSV", "ORTHO_CSV", "run_conditional"]
