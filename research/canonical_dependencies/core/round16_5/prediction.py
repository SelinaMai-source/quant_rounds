"""Top2 亏损时序 OOS 预测。equity-only vs equity+skew。纠正 Round16 只看 top1_net。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_5_ROOT
from .common import dump_json, load_json, write_md
from .contract import LOSS_NET_LT_0, SEVERE_LOSS_RET
from .engine import put_skew_series
from .features import equity_panel, ridge_fit, ridge_predict, standardize_train, v4_top2_labels
from .freeze import require_frozen

PRED_JSON = ROUND16_5_ROOT / "prediction" / "prediction.json"
PRED_MD = ROUND16_5_ROOT / "10_prediction.md"


def _cls_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) < 20:
        return {"n": int(len(y)), "auc": None, "brier": None, "logloss": None}
    brier = float(np.mean((p - y) ** 2))
    pr = np.clip(p, 1e-6, 1 - 1e-6)
    logloss = float(-np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr)))
    auc = None
    order = np.argsort(p)
    ys = y[order]
    pos = ys.sum()
    neg = len(ys) - pos
    if pos > 0 and neg > 0:
        ranks = np.arange(1, len(ys) + 1)
        auc = float((ranks[ys == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
    # calibration 10 bins
    bins = np.clip((p * 10).astype(int), 0, 9)
    cal = []
    for b in range(10):
        m = bins == b
        if m.sum() >= 5:
            cal.append({"bin": b, "mean_pred": float(p[m].mean()), "mean_y": float(y[m].mean()), "n": int(m.sum())})
    return {"n": int(len(y)), "auc": auc, "brier": brier, "logloss": logloss, "calibration": cal}


def _oos_ridge_prob(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    idx = X.index.intersection(y.dropna().index)
    X = X.loc[idx]
    y = y.loc[idx]
    pred = pd.Series(index=idx, dtype=float)
    years = pd.DatetimeIndex(idx).year
    for test_year in sorted(set(years)):
        train = years < test_year
        test = years == test_year
        if int(train.sum()) < 40 or int(test.sum()) < 8:
            continue
        Xtr, Xte, _, _ = standardize_train(X.loc[train].to_numpy(float), X.loc[test].to_numpy(float))
        yt = y.loc[train].to_numpy(float)
        beta = ridge_fit(Xtr, yt, alpha=1.0)
        raw = ridge_predict(Xte, beta)
        pred.loc[test] = 1.0 / (1.0 + np.exp(-np.clip(raw, -20, 20)))
    return pred


def run_prediction(**_: Any) -> dict[str, Any]:
    require_frozen()
    labels = v4_top2_labels()
    labels["signal_date"] = pd.to_datetime(labels["signal_date"]).dt.normalize()
    panel = equity_panel()
    skew = put_skew_series().rename("PUT_SKEW25")
    m = labels.set_index("signal_date").sort_index()
    eq = panel.reindex(m.index)
    # PIT: 用 D 可观测状态预测 D 的 Top2（D+1 开盘成交）。skew 用 shift(1) 与 O01 一致。
    Xeq = eq[["market_up_ratio", "B01A", "candidate_count", "index_RV20"]].copy()
    Xplus = Xeq.copy()
    Xplus["PUT_SKEW25"] = skew.shift(1).reindex(m.index)
    y_loss = (pd.to_numeric(m["top2_net"], errors="coerce") < LOSS_NET_LT_0).astype(float)
    y_sev = (pd.to_numeric(m["top2_ret"], errors="coerce") < SEVERE_LOSS_RET).astype(float)
    results = {}
    for name, y in (("top2_loss_net_lt_0", y_loss), ("top2_severe_loss", y_sev)):
        peq = _oos_ridge_prob(Xeq, y)
        ppl = _oos_ridge_prob(Xplus, y)
        results[name] = {
            "equity_only": _cls_metrics(y.reindex(peq.index).to_numpy(), peq.to_numpy()),
            "equity_plus_skew": _cls_metrics(y.reindex(ppl.index).to_numpy(), ppl.to_numpy()),
            "n_oos": int(peq.notna().sum()),
        }
    payload = {
        "status": "DONE",
        "target": "Top2 loss (not top1_net)",
        "pre_registered": {"loss_net_lt": LOSS_NET_LT_0, "severe_loss_ret": SEVERE_LOSS_RET},
        "in_sample_t_only": False,
        "results": results,
        "note": "年留出 Ridge-logit 概率。Round16 增量主要看了 top1_net，本轮纠正为 Top2 亏损。",
    }
    dump_json(PRED_JSON, payload)
    loss = results.get("top2_loss_net_lt_0") or {}
    write_md(
        PRED_MD,
        "\n".join([
            "# Round 16.5 · 10 Top2 亏损预测",
            "",
            f"equity-only AUC=`{(loss.get('equity_only') or {}).get('auc')}` Brier=`{(loss.get('equity_only') or {}).get('brier')}`。",
            f"equity+skew AUC=`{(loss.get('equity_plus_skew') or {}).get('auc')}` Brier=`{(loss.get('equity_plus_skew') or {}).get('brier')}`。",
            "目标=Top2 亏损，不是 top1_net。",
            "",
        ]),
    )
    return payload


def load_prediction() -> dict[str, Any]:
    return load_json(PRED_JSON)
