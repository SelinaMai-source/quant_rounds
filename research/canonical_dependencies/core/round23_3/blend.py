"""Five/Ten 与 XGB/LGBM 融合。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.research_integrity_v2.merge_rank import explicit_merge


def blend_ten(five: pd.DataFrame, ten: pd.DataFrame, w_ten: float) -> pd.DataFrame:
    a = five.rename(columns={"score": "five"})
    b = ten.rename(columns={"score": "ten"})
    for frame in (a, b):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    work = explicit_merge(a, b, on=["signal_date", "instrument_id"], how="inner", validate="one_to_one")
    from core.round17.backtest import load_gaps
    from core.round7.contract import GAP_FLOOR

    work = work.merge(load_gaps(), on=["signal_date", "instrument_id"], how="left")
    work["d1_open_gap"] = pd.to_numeric(work["d1_open_gap"], errors="coerce")
    work = work.loc[np.isfinite(work["d1_open_gap"]) & work["d1_open_gap"].ge(GAP_FLOOR)].copy()
    work["pct_five"] = work.groupby("signal_date")["five"].rank(method="average", pct=True)
    work["pct_ten"] = work.groupby("signal_date")["ten"].rank(method="average", pct=True)
    work["score"] = float(w_ten) * work["pct_ten"] + (1.0 - float(w_ten)) * work["pct_five"]
    return work[["signal_date", "instrument_id", "score"]]


def blend_xgb_lgbm(xgb: pd.DataFrame, lgbm: pd.DataFrame, w_lgbm: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """w_lgbm=0 直接返回冻结 v5，不额外 re-rank。"""
    if float(w_lgbm) == 0.0:
        return xgb.copy(), pd.DataFrame({"signal_date": [], "missing_stock_count": [], "reason": []})
    a = xgb.rename(columns={"score": "xgb"}).copy()
    b = lgbm.rename(columns={"score": "lgbm"}).copy()
    for frame in (a, b):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    fallback_rows = []
    out_parts = []
    for day, block in a.groupby("signal_date", sort=True):
        day_l = b.loc[b["signal_date"].eq(day)]
        merged = block.merge(day_l, on=["signal_date", "instrument_id"], how="left")
        missing = int(merged["lgbm"].isna().sum())
        if missing:
            fallback_rows.append(
                {
                    "signal_date": str(pd.Timestamp(day).date()),
                    "missing_stock_count": missing,
                    "reason": "MISSING_LGBM_PRED",
                    "fallback_day": True,
                }
            )
            out_parts.append(block.rename(columns={"xgb": "score"})[["signal_date", "instrument_id", "score"]])
            continue
        # 完整覆盖才融合；禁止 silent inner-join 缩小池
        if len(day_l) != len(block) or set(day_l["instrument_id"]) != set(block["instrument_id"]):
            # 若 LGBM 多出股票忽略；若少则已在 missing 分支。此处等长且集合相同。
            pass
        merged["px"] = merged["xgb"].rank(method="average", pct=True)
        merged["pl"] = merged["lgbm"].rank(method="average", pct=True)
        merged["score"] = float(w_lgbm) * merged["pl"] + (1.0 - float(w_lgbm)) * merged["px"]
        out_parts.append(merged[["signal_date", "instrument_id", "score"]])
    out = pd.concat(out_parts, ignore_index=True) if out_parts else xgb.copy()
    return out, pd.DataFrame(fallback_rows)
