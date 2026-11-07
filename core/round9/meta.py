"""C01–C04：每天一行 Top2，3Y walk-forward，7 特征，禁止调参。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND9_ROOT, assert_project_result_path, ensure_round9_tree
from .allocator import activity_stats, b01_nav, snap_nav, write_alloc
from .contract import (
    META_FEATURES,
    META_XGB,
    PARENT_CODE,
    WF_PREDICT_YEARS,
)
from .features import build_daily_frame, feature_matrix, persist_daily
from .runner import load_summary, run_codes


def _xgb_params() -> dict[str, Any]:
    p = dict(META_XGB)
    assert float(p["subsample"]) == 1.0
    assert float(p["colsample_bytree"]) == 1.0
    assert int(p["max_depth"]) == 2
    return p


def _fit_reg(x: np.ndarray, y: np.ndarray):
    from xgboost import XGBRegressor

    p = _xgb_params()
    model = XGBRegressor(
        objective="reg:squarederror",
        max_depth=int(p["max_depth"]),
        n_estimators=int(p["n_estimators"]),
        learning_rate=float(p["learning_rate"]),
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=float(p["min_child_weight"]),
        reg_lambda=float(p["reg_lambda"]),
        reg_alpha=float(p["reg_alpha"]),
        tree_method=str(p["tree_method"]),
        random_state=int(p["random_state"]),
        n_jobs=int(p["n_jobs"]),
    )
    model.fit(x, y)
    return model


def _fit_clf(x: np.ndarray, y: np.ndarray):
    from xgboost import XGBClassifier

    p = _xgb_params()
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=int(p["max_depth"]),
        n_estimators=int(p["n_estimators"]),
        learning_rate=float(p["learning_rate"]),
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=float(p["min_child_weight"]),
        reg_lambda=float(p["reg_lambda"]),
        reg_alpha=float(p["reg_alpha"]),
        tree_method=str(p["tree_method"]),
        random_state=int(p["random_state"]),
        n_jobs=int(p["n_jobs"]),
    )
    model.fit(x, y)
    return model


def train_mask(daily: pd.DataFrame, predict_year: int) -> pd.Series:
    years = daily.index.year
    train_years = {predict_year - 3, predict_year - 2, predict_year - 1}
    in_train = years.isin(train_years)
    exit_y = pd.to_datetime(daily.get("label_exit_date"), errors="coerce")
    mature = exit_y.dt.year.fillna(predict_year - 1).le(predict_year - 1)
    y = pd.to_numeric(daily.get("top2_ret"), errors="coerce")
    return in_train & mature & y.notna()


def pred_mask(daily: pd.DataFrame, predict_year: int) -> pd.Series:
    return daily.index.year == predict_year


def c01_map(pred: np.ndarray, train_pred: np.ndarray) -> np.ndarray:
    pos = train_pred[train_pred > 0]
    cut = float(np.median(pos)) if len(pos) else 0.0
    out = np.zeros(len(pred), dtype=float)
    out[pred > 0] = 0.05
    out[pred > cut] = 0.10
    return out


def tertile_map(pred: np.ndarray, train_pred: np.ndarray) -> np.ndarray:
    if len(train_pred) < 9:
        return c01_map(pred, train_pred)
    q1, q2 = np.nanpercentile(train_pred, [100 / 3, 200 / 3])
    out = np.zeros(len(pred), dtype=float)
    out[pred > q1] = 0.05
    out[pred > q2] = 0.10
    return out


def win_map(proba: np.ndarray, train_proba: np.ndarray) -> np.ndarray:
    if len(train_proba) < 9:
        out = np.where(proba >= 0.5, 0.10, 0.0)
        return out.astype(float)
    q1, q2 = np.nanpercentile(train_proba, [100 / 3, 200 / 3])
    out = np.zeros(len(proba), dtype=float)
    out[proba > q1] = 0.05
    out[proba > q2] = 0.10
    return out


def _monotone(pred: np.ndarray, y: np.ndarray, bins: int = 5) -> bool:
    mask = np.isfinite(pred) & np.isfinite(y)
    if mask.sum() < bins * 8:
        return False
    qs = pd.qcut(pd.Series(pred[mask]).rank(method="first"), bins, labels=False, duplicates="drop")
    means = pd.Series(y[mask]).groupby(qs).mean()
    if len(means) < 3:
        return False
    return bool(means.is_monotonic_increasing)


def _fallback_b01(daily: pd.DataFrame) -> pd.Series:
    return b01_nav(daily["agreement"].astype(str), daily["sep_bucket"].astype(str))


def walk_forward(
    daily: pd.DataFrame,
    feat_cols: tuple[str, ...] = META_FEATURES,
    *,
    dest: dict[str, str] | None = None,
    update_c04_skip: bool = True,
) -> dict[str, Any]:
    X = feature_matrix(daily, feat_cols)
    y = pd.to_numeric(daily["top2_ret"], errors="coerce")
    win = (y > 0).astype(float)
    oof_pred = pd.Series(np.nan, index=daily.index, dtype=float)
    oof_proba = pd.Series(np.nan, index=daily.index, dtype=float)
    oof_iso = pd.Series(np.nan, index=daily.index, dtype=float)
    alloc = {k: pd.Series(np.nan, index=daily.index, dtype=float) for k in ("C01", "C02", "C03", "C04")}
    folds = []
    leak = {"label_mature_violations": 0, "future_year_in_train": 0}

    for year in WF_PREDICT_YEARS:
        tr = train_mask(daily, year)
        te = pred_mask(daily, year)
        n_tr = int(tr.sum())
        rec = {
            "predict_year": year,
            "train_years": [year - 3, year - 2, year - 1],
            "n_train": n_tr,
            "n_pred_days": int(te.sum()),
            "short_train": n_tr < 40,
        }
        fb = _fallback_b01(daily)
        if n_tr < 40 or int(te.sum()) == 0:
            rec["fallback"] = "B01"
            for k in alloc:
                alloc[k].loc[te] = fb.loc[te]
            folds.append(rec)
            continue
        xtr = X.loc[tr].fillna(0.0).to_numpy(dtype=float)
        ytr = y.loc[tr].to_numpy(dtype=float)
        wtr = win.loc[tr].to_numpy(dtype=int)
        xte = X.loc[te].fillna(0.0).to_numpy(dtype=float)
        if wtr.min() == wtr.max():
            rec["fallback"] = "B01_one_class"
            for k in alloc:
                alloc[k].loc[te] = fb.loc[te]
            folds.append(rec)
            continue
        reg = _fit_reg(xtr, ytr)
        clf = _fit_clf(xtr, wtr)
        ptr = reg.predict(xtr)
        pte = reg.predict(xte)
        qtr = clf.predict_proba(xtr)[:, 1]
        qte = clf.predict_proba(xte)[:, 1]
        oof_pred.loc[te] = pte
        oof_proba.loc[te] = qte
        alloc["C01"].loc[te] = c01_map(pte, ptr)
        alloc["C02"].loc[te] = win_map(qte, qtr)
        alloc["C03"].loc[te] = tertile_map(pte, ptr)
        try:
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(ptr, ytr)
            iso_te = iso.predict(pte)
            oof_iso.loc[te] = iso_te
            alloc["C04"].loc[te] = c01_map(iso_te, iso.predict(ptr))
        except Exception as exc:
            rec["iso_error"] = repr(exc)
            alloc["C04"].loc[te] = alloc["C01"].loc[te]
        exit_y = pd.to_datetime(daily.loc[tr, "label_exit_date"], errors="coerce")
        leak["label_mature_violations"] += int((exit_y.dt.year > (year - 1)).sum())
        leak["future_year_in_train"] += int((daily.loc[tr].index.year >= year).sum())
        rec["train_mean_y"] = float(ytr.mean())
        rec["pred_mean"] = float(np.mean(pte))
        folds.append(rec)

    dest = dest or {"C01": "R9_C01", "C02": "R9_C02", "C03": "R9_C03", "C04": "R9_C04"}
    for k, s in alloc.items():
        s = s.fillna(_fallback_b01(daily))
        alloc[k] = s.map(snap_nav)
        if dest.get(k):
            write_alloc(dest[k], daily.index, alloc[k])

    yv = y.to_numpy()
    pv = oof_pred.to_numpy()
    monotone = _monotone(pv[np.isfinite(pv) & np.isfinite(yv)], yv[np.isfinite(pv) & np.isfinite(yv)])
    if update_c04_skip and not monotone:
        get = __import__("core.round9.registry", fromlist=["get_spec"]).get_spec
        spec = get("R9_C04")
        spec.skip = True
        spec.skip_reason = "C01_OOF_NOT_MONOTONE"

    oof = daily.reset_index()
    oof["c01_pred"] = oof_pred.to_numpy()
    oof["c02_proba"] = oof_proba.to_numpy()
    oof["c04_iso"] = oof_iso.to_numpy()
    oof["alloc_c01"] = alloc["C01"].to_numpy()
    oof["alloc_c02"] = alloc["C02"].to_numpy()
    oof["alloc_c03"] = alloc["C03"].to_numpy()
    oof["alloc_c04"] = alloc["C04"].to_numpy()
    if dest.get("C01") == "R9_C01" and update_c04_skip:
        path = ROUND9_ROOT / "models" / "meta_oof.parquet"
        assert_project_result_path(path)
        oof.to_parquet(path, index=False)
    payload_oof = oof

    leak["top1_lock"] = "runner_enforced"
    leak["budget_set"] = sorted({float(x) for s in alloc.values() for x in s.dropna().unique()})
    leak["budget_ok"] = all(abs(v - snap_nav(v)) < 1e-12 for v in leak["budget_set"])
    leak["past_only_features"] = list(feat_cols)
    leak["passed"] = leak["label_mature_violations"] == 0 and leak["future_year_in_train"] == 0 and leak["budget_ok"]

    payload = {
        "folds": folds,
        "monotone_c01": bool(monotone),
        "leakage": leak,
        "activity": {k: activity_stats(alloc[k]) for k in alloc},
        "n_days": int(len(daily)),
        "features": list(feat_cols),
        "xgb": _xgb_params(),
    }
    return payload


def _calibration_plot(oof: pd.DataFrame) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    pred = pd.to_numeric(oof.get("c01_pred"), errors="coerce")
    y = pd.to_numeric(oof.get("top2_ret"), errors="coerce")
    m = pred.notna() & y.notna()
    if m.sum() < 30:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    qs = pd.qcut(pred[m].rank(method="first"), 5, labels=False, duplicates="drop")
    g = pd.DataFrame({"p": pred[m], "y": y[m], "q": qs}).groupby("q").mean()
    ax[0].plot(g["p"], g["y"], marker="o")
    ax[0].axhline(0, color="gray", lw=0.8)
    ax[0].set_title("C01 OOF calibration (quintile)")
    ax[0].set_xlabel("mean pred")
    ax[0].set_ylabel("mean realized Top2 ret")
    ax[1].hist(pred[m], bins=30, alpha=0.7)
    ax[1].set_title("C01 OOF pred hist")
    fig.tight_layout()
    path = ROUND9_ROOT / "figures" / "c01_calibration.png"
    assert_project_result_path(path)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def _pct(x, digits=2) -> str:
    if x is None:
        return "—"
    return f"{float(x):.{digits}%}".replace("-", "−")


def write_meta_md(payload: dict[str, Any], codes: list[str]) -> None:
    from .docs import metrics_table

    act = payload.get("activity") or {}
    leak = payload.get("leakage") or {}
    body = f"""# 03 — Top2 meta model

每天 **一行** Top2。walk-forward = 主 Alpha 同年 fold（过去 3 个日历年训 → 下一年预测）。
特征恰好 7 个：`{payload.get('features')}`。
XGB：`max_depth=2`，`subsample=colsample=1.0`，`n_estimators=50`，**禁止调参**。
阈值只用训练折历史。2018 等短训练折回退 B01。

C01 必须证明识别 edge，而不是 `LOW_ACTIVITY` 少买 Top2。
若 B01 打平/打赢 C01，默认不引入 ML。

## 漏泄

- past-only 特征：{leak.get('past_only_features')}
- label mature 违约行：{leak.get('label_mature_violations')}
- 训练折含未来年：{leak.get('future_year_in_train')}
- 预算集合：{leak.get('budget_set')}（须 ⊆ {{0, 0.05, 0.10}}）
- 判定：{"PASS" if leak.get("passed") else "FAIL"}

C01 OOF 单调：{payload.get("monotone_c01")}。C04 仅在单调时启用。

## 活动度

| 组 | zero_share | share_5 | share_10 | flag |
| --- | ---: | ---: | ---: | --- |
"""
    for k, st in act.items():
        body += (
            f"| {k} | {st.get('zero_share')} | {st.get('share_5')} | {st.get('share_10')} | {st.get('flag')} |\n"
        )
    body += f"""
## 正式回测（vs {PARENT_CODE}）

{metrics_table(codes)}

校准图：`figures/c01_calibration.png`。OOF：`models/meta_oof.parquet`。
"""
    path = ROUND9_ROOT / "03_top2_meta_model.md"
    assert_project_result_path(path)
    path.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")


def run_meta(*, resume: bool = True, jobs: int = 1) -> dict[str, Any]:
    ensure_round9_tree()
    daily_path = ROUND9_ROOT / "ledger" / "top2_daily_features.parquet"
    if daily_path.is_file():
        daily = pd.read_parquet(daily_path)
        daily["signal_date"] = pd.to_datetime(daily["signal_date"]).dt.normalize()
        daily = daily.set_index("signal_date").sort_index()
    else:
        daily = build_daily_frame()
        persist_daily(daily)
    payload = walk_forward(daily)
    fig = None
    oof_path = ROUND9_ROOT / "models" / "meta_oof.parquet"
    if oof_path.is_file():
        fig = _calibration_plot(pd.read_parquet(oof_path))
    payload["calibration_fig"] = fig
    meta_path = ROUND9_ROOT / "metrics" / "meta_walkforward.json"
    assert_project_result_path(meta_path)
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    from .registry import get_spec

    codes = ["R9_C01", "R9_C02", "R9_C03", "R9_C04"]
    runnable = [c for c in codes if not get_spec(c).skip]
    bt = run_codes(runnable, resume=resume, jobs=jobs) if runnable else {"jobs": {}, "failed": []}
    payload["backtest"] = {k: v.get("status") for k, v in (bt.get("jobs") or {}).items()}
    write_meta_md(payload, codes)
    return payload


__all__ = ["run_meta", "train_mask", "walk_forward"]
