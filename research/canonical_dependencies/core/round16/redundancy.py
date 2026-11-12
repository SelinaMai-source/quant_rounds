"""训练期 residualization，禁止全样本。判定 REDUNDANT / PARTIALLY_INCREMENTAL / INCREMENTAL / INCONCLUSIVE。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_ROOT
from .common import dump_json, load_json, write_md
from .factors import load_factor_daily
from .freeze import require_frozen
from .incremental import _oos_predict, _v4_daily_pnl, load_incremental

RED_JSON = ROUND16_ROOT / "redundancy" / "redundancy.json"
RED_MD = ROUND16_ROOT / "07_equity_option_redundancy.md"


def run_redundancy(**_: Any) -> dict[str, Any]:
    require_frozen()
    daily = load_factor_daily()
    pnl = _v4_daily_pnl()
    if daily.empty or pnl.empty:
        payload = {"status": "SKIP", "verdict": "INCONCLUSIVE", "full_sample_residualization": False}
        dump_json(RED_JSON, payload)
        write_md(RED_MD, "# Round16 · 07 冗余\n\nSKIP。\n")
        return payload
    f = daily.copy()
    f["trade_date"] = pd.to_datetime(f["trade_date"]).dt.normalize()
    p = pnl.copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"]).dt.normalize()
    m = f.merge(p, on="trade_date", how="inner").sort_values("trade_date").set_index("trade_date")
    eq = pd.DataFrame({
        "eq_n": pd.to_numeric(m["n"], errors="coerce").shift(1),
        "eq_score": pd.to_numeric(m["mean_score"], errors="coerce").shift(1),
        "eq_top1": pd.to_numeric(m["top1_net"], errors="coerce").shift(1),
    })
    opt_cols = [c for c in ("ATM_IV30", "PUT_SKEW25", "PUT_WING_CONVEXITY", "IV_RV_SPREAD") if c in m.columns]
    residuals = {}
    pred_power = {}
    for col in opt_cols:
        y = pd.to_numeric(m[col], errors="coerce")
        yhat, meta = _oos_predict(eq, y)
        resid = (y - yhat).rename("resid")
        residuals[col] = float(resid.dropna().std()) if resid.notna().any() else None
        # 残差对未来 top1_net 的 OOS
        tgt = pd.to_numeric(m["top1_net"], errors="coerce")
        Xr = pd.DataFrame({"r": resid.shift(1)})
        phat, _ = _oos_predict(Xr, tgt)
        aligned = pd.concat({"y": tgt, "p": phat}, axis=1).dropna()
        if len(aligned) > 30:
            r = float(np.corrcoef(aligned["y"], aligned["p"])[0, 1])
        else:
            r = None
        pred_power[col] = {"oos_corr_top1": r, "n": int(len(aligned)), "n_folds": meta.get("n_folds")}
    corrs = [v["oos_corr_top1"] for v in pred_power.values() if v.get("oos_corr_top1") is not None]
    mean_abs = float(np.mean(np.abs(corrs))) if corrs else None
    if mean_abs is None:
        verdict = "INCONCLUSIVE"
    elif mean_abs < 0.03:
        verdict = "REDUNDANT"
    elif mean_abs < 0.08:
        verdict = "PARTIALLY_INCREMENTAL"
    else:
        verdict = "INCREMENTAL"
    inc = load_incremental()
    payload = {
        "status": "DONE",
        "verdict": verdict,
        "mean_abs_oos_corr": mean_abs,
        "pred_power": pred_power,
        "residual_std": residuals,
        "full_sample_residualization": False,
        "incremental_crosscheck": inc.get("incremental"),
        "note": "训练期拟合 f(equity)，后续生成 OptionResidual。不是全样本残差。",
    }
    dump_json(RED_JSON, payload)
    write_md(
        RED_MD,
        "\n".join([
            "# Round 16 · 07 股-期冗余",
            "",
            f"判定=**{verdict}** mean_abs_oos_corr=`{mean_abs}`。full_sample=`False`。",
            f"pred_power=`{pred_power}`",
            "",
            "显著性 ≠ 策略收益显著性。",
            "",
        ]),
    )
    return payload


def load_redundancy() -> dict[str, Any]:
    return load_json(RED_JSON)
