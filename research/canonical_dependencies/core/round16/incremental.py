"""EQUITY_ONLY vs EQUITY_PLUS_OPTION 时序 OOS 增量。禁止只用 in-sample t。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_ROOT
from .common import dump_json, load_json, write_md
from .contract import R16_00, parity_dir
from .factors import load_factor_daily, load_factors
from .freeze import require_frozen

INC_JSON = ROUND16_ROOT / "incremental" / "incremental.json"
INC_MD = ROUND16_ROOT / "06_option_factor_incrementality.md"


def _v4_daily_pnl() -> pd.DataFrame:
    path = parity_dir(R16_00) / "trades.parquet"
    if not path.is_file():
        from ..round15.contract import R15_00, parity_dir as p15
        path = p15(R15_00) / "trades.parquet"
    if not path.is_file():
        return pd.DataFrame()
    t = pd.read_parquet(path)
    t["signal_date"] = pd.to_datetime(t["signal_date"]).dt.normalize()
    t["_rank"] = t.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    t["net"] = pd.to_numeric(t.get("sell_proceeds"), errors="coerce") - pd.to_numeric(t.get("buy_cost"), errors="coerce")
    daily = t.groupby("signal_date").agg(
        n=("instrument_id", "size"),
        top1_net=("net", lambda s: s.iloc[0] if len(s) else np.nan),
        total_net=("net", "sum"),
        mean_score=("score", "mean"),
    )
    t2 = t.loc[t["_rank"].eq(2)].groupby("signal_date")["net"].sum().rename("top2_net")
    daily = daily.join(t2, how="left")
    daily["left_tail"] = (daily["total_net"] < daily["total_net"].quantile(0.10)).astype(float)
    daily["big_loss"] = (daily["total_net"] < 0).astype(float)
    return daily.reset_index().rename(columns={"signal_date": "trade_date"})


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) < 20:
        return {"n": int(len(y)), "r2": None, "auc": None, "brier": None, "logloss": None}
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None
    # classification if y in {0,1}
    binary = set(np.unique(np.round(y, 6))) <= {0.0, 1.0}
    auc = brier = logloss = None
    if binary:
        brier = float(np.mean((p - y) ** 2))
        pr = np.clip(p, 1e-6, 1 - 1e-6)
        logloss = float(-np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr)))
        order = np.argsort(p)
        ys = y[order]
        pos = ys.sum()
        neg = len(ys) - pos
        if pos > 0 and neg > 0:
            ranks = np.arange(1, len(ys) + 1)
            auc = float((ranks[ys == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
    return {"n": int(len(y)), "r2": r2, "auc": auc, "brier": brier, "logloss": logloss}


def _oos_predict(X: pd.DataFrame, y: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    """Leave-future-year-out linear. 禁止全样本。"""
    idx = X.index.intersection(y.dropna().index)
    X = X.loc[idx].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = y.loc[idx]
    pred = pd.Series(index=idx, dtype=float)
    years = pd.DatetimeIndex(idx).year
    coefs = []
    for test_year in sorted(set(years)):
        train = years < test_year
        test = years == test_year
        if train.sum() < 40 or test.sum() < 10:
            continue
        Xt = X.loc[train].to_numpy()
        yt = y.loc[train].to_numpy()
        Xt = np.column_stack([np.ones(len(Xt)), Xt])
        try:
            beta, *_ = np.linalg.lstsq(Xt, yt, rcond=None)
        except np.linalg.LinAlgError:
            continue
        coefs.append({"year": int(test_year), "beta": [float(v) for v in beta]})
        Xs = np.column_stack([np.ones(int(test.sum())), X.loc[test].to_numpy()])
        pred.loc[test] = Xs @ beta
    return pred, {"n_folds": len(coefs), "coefs": coefs, "in_sample_t_only": False}


def run_incremental(**_: Any) -> dict[str, Any]:
    require_frozen()
    fac = load_factors()
    daily_f = load_factor_daily()
    pnl = _v4_daily_pnl()
    if fac.get("skip") or daily_f.empty or pnl.empty:
        payload = {
            "status": "SKIP_IV_SURFACE" if fac.get("skip") else "SKIP_NO_ALIGN",
            "incremental": "INCONCLUSIVE",
            "in_sample_t_only": False,
        }
        dump_json(INC_JSON, payload)
        write_md(INC_MD, "# Round16 · 06 增量\n\nSKIP。不编造 R2。\n")
        return payload
    f = daily_f.copy()
    f["trade_date"] = pd.to_datetime(f["trade_date"]).dt.normalize()
    p = pnl.copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"]).dt.normalize()
    m = p.merge(f, on="trade_date", how="inner").sort_values("trade_date")
    m = m.set_index("trade_date")
    eq_cols = ["n", "mean_score"]
    for c in eq_cols:
        if c not in m.columns:
            m[c] = np.nan
    m["eq_n"] = pd.to_numeric(m["n"], errors="coerce").shift(1)
    m["eq_score"] = pd.to_numeric(m["mean_score"], errors="coerce").shift(1)
    opt_cols = [c for c in ("ATM_IV30", "PUT_SKEW25", "PUT_WING_CONVEXITY", "IV_RV_SPREAD") if c in m.columns]
    for c in opt_cols:
        m[c] = pd.to_numeric(m[c], errors="coerce").shift(1)
    targets = {
        "top1_net": m.get("top1_net"),
        "top2_net": m.get("top2_net"),
        "total_net": m.get("total_net"),
        "left_tail": m.get("left_tail"),
        "big_loss": m.get("big_loss"),
    }
    Xeq = m[["eq_n", "eq_score"]].copy()
    Xplus = m[["eq_n", "eq_score", *opt_cols]].copy()
    results = {}
    for name, y in targets.items():
        if y is None:
            continue
        peq, meta_eq = _oos_predict(Xeq, y)
        ppl, meta_pl = _oos_predict(Xplus, y)
        meq = _metrics(y.reindex(peq.index).to_numpy(), peq.to_numpy())
        mpl = _metrics(y.reindex(ppl.index).to_numpy(), ppl.to_numpy())
        inc_r2 = None
        if meq.get("r2") is not None and mpl.get("r2") is not None:
            inc_r2 = float(mpl["r2"] - meq["r2"])
        results[name] = {
            "equity_only": meq,
            "equity_plus_option": mpl,
            "incremental_r2": inc_r2,
            "n_folds": meta_pl.get("n_folds"),
        }
    inc_vals = [v.get("incremental_r2") for v in results.values() if v.get("incremental_r2") is not None]
    mean_inc = float(np.mean(inc_vals)) if inc_vals else None
    verdict = "INCONCLUSIVE"
    if mean_inc is not None:
        if mean_inc > 0.01:
            verdict = "YES"
        elif mean_inc > 0:
            verdict = "INCONCLUSIVE"
        else:
            verdict = "NO"
    payload = {
        "status": "DONE",
        "results": results,
        "mean_incremental_r2": mean_inc,
        "incremental": verdict,
        "in_sample_t_only": False,
        "n_aligned": int(len(m)),
        "option_cols": opt_cols,
        "note": "chronological year-holdout OLS。不是 in-sample t。",
    }
    dump_json(INC_JSON, payload)
    write_md(
        INC_MD,
        "\n".join([
            "# Round 16 · 06 期权因子增量",
            "",
            f"aligned_days=`{payload['n_aligned']}` mean_incremental_R2=`{mean_inc}` 判定=`{verdict}`。",
            f"results=`{results}`",
            "",
            "EQUITY_ONLY = 滞后候选数/均分。EQUITY_PLUS_OPTION 加 ATM_IV30/skew/convexity/IV_RV。时序 OOS。",
            "",
        ]),
    )
    return payload


def load_incremental() -> dict[str, Any]:
    return load_json(INC_JSON)
