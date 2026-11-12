"""a-prediction：Top2 亏损 OOS。AUC+Brier+LogLoss，禁止只挑 AUC。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...round16_5.contract import LOSS_NET_LT_0, SEVERE_LOSS_RET
from ...round16_5.features import equity_panel, ridge_fit, ridge_predict, standardize_train
from ...round16_5.prediction import _cls_metrics, _oos_ridge_prob
from .engine import put_skew_series
from .equity import _c0_top2_labels
from ..common import dump_json, write_md
from .support import PRED_JSON, resume_ok, skip_result, ta_md


def run_prediction(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    existing = resume_ok(PRED_JSON) if resume else None
    if existing is not None:
        return existing
    try:
        labels = _c0_top2_labels()
        panel = equity_panel()
        skew = put_skew_series().rename("PUT_SKEW25")
    except Exception as exc:
        payload = skip_result("a-prediction", f"{type(exc).__name__}:{exc}")
        dump_json(PRED_JSON, payload)
        return payload
    labels["signal_date"] = pd.to_datetime(labels["signal_date"]).dt.normalize()
    m = labels.set_index("signal_date").sort_index()
    eq = panel.reindex(m.index)
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
    loss = results.get("top2_loss_net_lt_0") or {}
    eq_m = loss.get("equity_only") or {}
    pl_m = loss.get("equity_plus_skew") or {}
    payload = {
        "status": "DONE",
        "target": "Top2 loss (not top1_net)",
        "pre_registered": {"loss_net_lt": LOSS_NET_LT_0, "severe_loss_ret": SEVERE_LOSS_RET},
        "in_sample_t_only": False,
        "metrics_required": ["auc", "brier", "logloss"],
        "results": results,
        "increment": {
            "auc": (None if eq_m.get("auc") is None or pl_m.get("auc") is None else float(pl_m["auc"]) - float(eq_m["auc"])),
            "brier": (None if eq_m.get("brier") is None or pl_m.get("brier") is None else float(pl_m["brier"]) - float(eq_m["brier"])),
            "logloss": (None if eq_m.get("logloss") is None or pl_m.get("logloss") is None else float(pl_m["logloss"]) - float(eq_m["logloss"])),
        },
        "note": "必须同时报 AUC/Brier/LogLoss。skew 用 shift(1)。",
    }
    dump_json(PRED_JSON, payload)
    write_md(
        ta_md("prediction.md"),
        "\n".join(
            [
                "# Round19 Track A · a-prediction",
                "",
                f"equity-only AUC=`{eq_m.get('auc')}` Brier=`{eq_m.get('brier')}` LogLoss=`{eq_m.get('logloss')}`。",
                f"equity+skew AUC=`{pl_m.get('auc')}` Brier=`{pl_m.get('brier')}` LogLoss=`{pl_m.get('logloss')}`。",
                "禁止只挑 AUC。目标=Top2 亏损。",
                "",
            ]
        ),
    )
    return payload
