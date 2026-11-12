"""V04 时序正交化 Put Skew。唯一新 Alpha。不搜残差阈值。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_5_ROOT
from ..round15.runner import apply_top2_veto
from .common import daily_from_nav, dump_json, load_json, pack_daily, write_md
from .contract import (
    PCT_MIN_OBS,
    PCT_WINDOW,
    RIDGE_ALPHA,
    SKEW_Q,
    V00,
    V01,
    V04,
    V04_FIT_MIN,
    V04_FIT_WINDOW,
    experiment_dir,
    parity_dir,
)
from .engine import cached_overlay_v4, put_skew_series, run_engine, save_overlay
from .features import equity_panel, ridge_fit, ridge_predict, standardize_train
from .freeze import require_frozen

ORT_JSON = ROUND16_5_ROOT / "orthogonal" / "orthogonal.json"
ORT_MD = ROUND16_5_ROOT / "07_V04_orthogonal_put_skew.md"
RESID_PATH = ROUND16_5_ROOT / "orthogonal" / "residual_put_skew.parquet"


def _rolling_residual(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    idx = y.index.intersection(X.index)
    y = y.reindex(idx)
    X = X.reindex(idx)
    resid = pd.Series(index=idx, dtype=float)
    values_y = y.to_numpy(float)
    values_x = X.to_numpy(float)
    n = len(idx)
    win = V04_FIT_WINDOW
    minp = V04_FIT_MIN
    for i in range(n):
        lo = max(0, i - win)
        sl = slice(lo, i)  # strictly before t
        if (i - lo) < minp:
            continue
        ytr = values_y[sl]
        Xtr = values_x[sl]
        mask = np.isfinite(ytr) & np.isfinite(Xtr).all(axis=1)
        if int(mask.sum()) < minp:
            continue
        ytr = ytr[mask]
        Xtr = Xtr[mask]
        Xt = values_x[i : i + 1]
        if not np.isfinite(values_y[i]) or not np.isfinite(Xt).all():
            continue
        Xtr_s, Xt_s, _, _ = standardize_train(Xtr, Xt)
        beta = ridge_fit(Xtr_s, ytr, alpha=RIDGE_ALPHA)
        pred = float(ridge_predict(Xt_s, beta)[0])
        resid.iloc[i] = float(values_y[i] - pred)
    return resid


def run_orthogonal(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    dest = experiment_dir(V04)
    if resume and (dest / "report.json").is_file() and ORT_JSON.is_file():
        existing = load_json(ORT_JSON)
        if existing.get("status") == "DONE":
            return existing
    y = put_skew_series()
    panel = equity_panel()
    X = panel.reindex(y.index)
    feat_cols = ["market_up_ratio", "B01A", "candidate_count", "index_RV20"]
    resid = _rolling_residual(y, X[feat_cols])
    RESID_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"trade_date": resid.index, "residual_put_skew": resid.values}).to_parquet(RESID_PATH, index=False)
    past = resid.shift(1)
    past_q = past.rolling(PCT_WINDOW, min_periods=PCT_MIN_OBS).quantile(SKEW_Q)
    hit = past > past_q
    dates = pd.DatetimeIndex(hit.index[hit.fillna(False)]).normalize().unique().sort_values()
    ov = apply_top2_veto(cached_overlay_v4(), dates)
    save_overlay(V04, ov)
    ran = run_engine(
        dest,
        ov,
        extra={
            "code": V04,
            "round": "16.5",
            "role": "orthogonal_residual_put_skew",
            "n_events": int(len(dates)),
            "ridge_alpha": RIDGE_ALPHA,
            "fit_window": V04_FIT_WINDOW,
            "fit_min": V04_FIT_MIN,
        },
        resume=resume,
        write=True,
    )
    obs = ran.get("observed") or {}
    v00 = pack_daily(daily_from_nav(pd.read_parquet(parity_dir(V00) / "nav.parquet")))
    v01 = pack_daily(daily_from_nav(pd.read_parquet(parity_dir(V01) / "nav.parquet")))
    n_resid = int(resid.notna().sum())
    payload = {
        "status": "DONE",
        "is_candidate": True,
        "n_events": int(len(dates)),
        "n_residual_days": n_resid,
        "ridge_alpha": RIDGE_ALPHA,
        "fit_window": V04_FIT_WINDOW,
        "fit_min": V04_FIT_MIN,
        "percentile": {"window": PCT_WINDOW, "min": PCT_MIN_OBS, "q": SKEW_Q, "shift": 1},
        "no_threshold_search": True,
        "features": feat_cols,
        "standardize": "train_only",
        "observed": obs,
        "v00": v00,
        "v01": v01,
        "worse_than_v01": (
            (obs.get("sharpe") is not None and v01.get("sharpe") is not None and float(obs["sharpe"]) < float(v01["sharpe"]))
        ),
        "dates": [str(x.date()) for x in dates],
        "residual_path": str(RESID_PATH),
        "out_dir": str(dest),
        "note": "V04 更差则如实报告，不换第二套残差模型。",
    }
    dump_json(ORT_JSON, payload)
    write_md(
        ORT_MD,
        "\n".join([
            "# Round 16.5 · 07 V04 正交化 Put Skew",
            "",
            "唯一新 Alpha。`PUT_SKEW25 = f(X_equity)+ε`，Ridge α=1.0，窗 504/最少 252，t 之前拟合。残差过去 252/最少 126 的 90 分位，shift(1) → Top2 MIN_KEEP。",
            f"n_events=`{len(dates)}` n_residual_days=`{n_resid}`。",
            f"CAGR=`{obs.get('cagr')}` Sharpe=`{obs.get('sharpe')}` MaxDD=`{obs.get('max_drawdown')}` 笔数=`{obs.get('completed_trades')}`。",
            f"worse_than_v01=`{payload['worse_than_v01']}`。不换第二套残差。",
            "",
        ]),
    )
    return payload


def load_orthogonal() -> dict[str, Any]:
    return load_json(ORT_JSON)
