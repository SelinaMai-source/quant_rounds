"""V03 股票风险 veto 对照。活动匹配阈值只用过去训练窗。不能升 v5。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_5_ROOT
from ..round15.runner import apply_top2_veto
from .common import daily_from_nav, dump_json, load_json, pack_daily, write_md
from .contract import (
    ACTIVITY_NOT_FULLY_MATCHED,
    V00,
    V01,
    V03,
    V03_RIDGE_ALPHA,
    experiment_dir,
    overlayed_path,
    parity_dir,
)
from .engine import cached_overlay_v4, o01_signal_dates, run_engine, save_overlay
from .features import equity_panel, ridge_fit, ridge_predict, standardize_train, v4_top2_labels
from .freeze import require_frozen
from .attribution import load_attribution

EQ_JSON = ROUND16_5_ROOT / "equity" / "equity.json"
EQ_MD = ROUND16_5_ROOT / "06_V03_equity_risk_veto.md"


def _jaccard(a, b) -> float | None:
    a = set(pd.DatetimeIndex(pd.to_datetime(list(a))).normalize())
    b = set(pd.DatetimeIndex(pd.to_datetime(list(b))).normalize())
    if not a and not b:
        return None
    return float(len(a & b) / len(a | b))


def run_equity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    dest = experiment_dir(V03)
    if resume and (dest / "report.json").is_file() and EQ_JSON.is_file():
        existing = load_json(EQ_JSON)
        if existing.get("status") == "DONE":
            return existing
    panel = equity_panel()
    labels = v4_top2_labels()
    labels["signal_date"] = pd.to_datetime(labels["signal_date"]).dt.normalize()
    feat_cols = ["market_up_ratio", "B01A", "candidate_count", "index_RV20"]
    m = labels.merge(panel.reset_index().rename(columns={"index": "signal_date"}), on="signal_date", how="left")
    m = m.sort_values("signal_date")
    attr = load_attribution()
    eligible = pd.DatetimeIndex(pd.to_datetime(attr.get("eligible_dates") or [])).normalize()
    o01_dates = pd.DatetimeIndex(pd.to_datetime(o01_signal_dates())).normalize()
    o01_elig = eligible.intersection(o01_dates)

    scores = pd.Series(index=pd.DatetimeIndex(m["signal_date"]), dtype=float)
    thresholds = {}
    years = pd.to_datetime(m["signal_date"]).dt.year
    for test_year in sorted(set(years)):
        train_mask = years < test_year
        test_mask = years == test_year
        # 成熟标签：退出日所在年 < test_year
        mature = pd.to_datetime(m["actual_exit_date"]).dt.year < test_year
        use = train_mask & mature
        if int(use.sum()) < 40 or int(test_mask.sum()) < 5:
            continue
        Xtr_raw = m.loc[use, feat_cols].to_numpy(float)
        ytr = (-pd.to_numeric(m.loc[use, "top2_net"], errors="coerce")).to_numpy(float)
        ytr = np.where(np.isfinite(ytr), ytr, 0.0)
        Xte_raw = m.loc[test_mask, feat_cols].to_numpy(float)
        Xtr, Xte, _, _ = standardize_train(Xtr_raw, Xte_raw)
        beta = ridge_fit(Xtr, ytr, alpha=V03_RIDGE_ALPHA)
        pred_tr = ridge_predict(Xtr, beta)
        pred_te = ridge_predict(Xte, beta)
        train_dates = pd.DatetimeIndex(pd.to_datetime(m.loc[use, "signal_date"]).dt.normalize())
        n_target = int(train_dates.isin(o01_elig).sum())
        if n_target <= 0:
            n_target = max(1, int(round(0.1 * len(pred_tr))))
        q = 1.0 - min(0.99, max(0.01, n_target / max(len(pred_tr), 1)))
        thr = float(np.quantile(pred_tr, q))
        thresholds[int(test_year)] = {"threshold": thr, "n_target_train": n_target, "n_train": int(use.sum()), "q": q}
        scores.loc[pd.DatetimeIndex(pd.to_datetime(m.loc[test_mask, "signal_date"]).dt.normalize())] = pred_te

    veto = []
    by_year_activity = {}
    for test_year, rec in thresholds.items():
        year_idx = scores.index[scores.index.year == test_year]
        year_idx = year_idx.intersection(eligible) if len(eligible) else year_idx
        sc = scores.reindex(year_idx).dropna()
        picked = sc.index[sc >= rec["threshold"]]
        n_o01 = int((o01_elig.year == test_year).sum())
        by_year_activity[int(test_year)] = {
            "n_v03": int(len(picked)),
            "n_o01": n_o01,
            "n_eligible": int((eligible.year == test_year).sum()) if len(eligible) else int(len(year_idx)),
        }
        veto.extend(list(picked))
    veto_idx = pd.DatetimeIndex(pd.to_datetime(veto)).normalize().unique().sort_values()
    matched = True
    for y, rec in by_year_activity.items():
        if rec["n_o01"] > 0 and abs(rec["n_v03"] - rec["n_o01"]) > max(2, 0.25 * rec["n_o01"]):
            matched = False
    jac = _jaccard(veto_idx, o01_elig)
    ov = apply_top2_veto(cached_overlay_v4(), veto_idx)
    save_overlay(V03, ov)
    ran = run_engine(
        dest,
        ov,
        extra={"code": V03, "round": "16.5", "role": "equity_control_not_candidate", "n_veto": int(len(veto_idx))},
        resume=resume,
        write=True,
    )
    obs = ran.get("observed") or {}
    v00 = pack_daily(daily_from_nav(pd.read_parquet(parity_dir(V00) / "nav.parquet")))
    v01 = pack_daily(daily_from_nav(pd.read_parquet(parity_dir(V01) / "nav.parquet")))
    payload = {
        "status": "DONE",
        "not_candidate": True,
        "n_veto": int(len(veto_idx)),
        "n_o01_eligible": int(len(o01_elig)),
        "jaccard": jac,
        "overlap": int(len(set(veto_idx) & set(o01_elig))),
        "activity_matched": bool(matched),
        "activity_flag": None if matched else ACTIVITY_NOT_FULLY_MATCHED,
        "by_year": by_year_activity,
        "thresholds": thresholds,
        "features": feat_cols,
        "alpha": V03_RIDGE_ALPHA,
        "mature_labels": True,
        "full_sample_threshold": False,
        "observed": obs,
        "v00": v00,
        "v01": v01,
        "veto_dates": [str(x.date()) for x in veto_idx],
        "out_dir": str(dest),
    }
    dump_json(EQ_JSON, payload)
    write_md(
        EQ_MD,
        "\n".join([
            "# Round 16.5 · 06 V03 股票风险 veto",
            "",
            f"**不是候选**。Ridge α=`{V03_RIDGE_ALPHA}` 特征=`{feat_cols}` 成熟标签、训练窗阈值。",
            f"n_veto=`{len(veto_idx)}` Jaccard vs O01=`{jac}` activity_matched=`{matched}` flag=`{payload['activity_flag']}`。",
            f"CAGR=`{obs.get('cagr')}` Sharpe=`{obs.get('sharpe')}` MaxDD=`{obs.get('max_drawdown')}` 笔数=`{obs.get('completed_trades')}`。",
            "",
        ]),
    )
    return payload


def load_equity() -> dict[str, Any]:
    return load_json(EQ_JSON)
