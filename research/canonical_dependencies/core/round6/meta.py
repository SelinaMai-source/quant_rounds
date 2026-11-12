"""F：外折 OOF meta。禁止 in-sample Top1 进训练集。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from ..round2.data import SEEDS, load_all_seed_predictions, load_regime_frame
from ..round4.ecology import build_ecology_frame
from .overlays import margin_frame_from_scores
from .registry import LOCKED_FEATURES
from .train import LABEL_END, load_panel

COMPLETE_YEARS = tuple(range(2018, 2026))
META_FEATURE_NAMES = (
    "margin", "agree_n", "n_cand", "disp",
    "b01a", "relimit", "top1_f0", "top1_f1", "top1_f2",
    "top1_f3", "top1_f4", "top1_f5",
)


def daily_top1(preds: pd.DataFrame) -> pd.DataFrame:
    ranked = preds.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True])
    return ranked.groupby("signal_date", sort=True).head(1).copy()


def seed_agreement(preds: pd.DataFrame) -> pd.Series:
    """当日种子 Top1 与集成 Top1 一致数。锁分种子只读。"""
    ens = daily_top1(preds)[["signal_date", "instrument_id"]].rename(columns={"instrument_id": "ens_id"})
    try:
        seeds = load_all_seed_predictions()
    except Exception:
        return pd.Series(np.nan, index=pd.to_datetime(ens["signal_date"]))
    seeds["signal_date"] = pd.to_datetime(seeds["signal_date"])
    top = (
        seeds.sort_values(["seed", "signal_date", "score", "instrument_id"], ascending=[True, True, False, True])
        .groupby(["seed", "signal_date"], sort=True)
        .head(1)[["seed", "signal_date", "instrument_id"]]
    )
    joined = top.merge(ens, on="signal_date", how="left")
    joined["ok"] = joined["instrument_id"] == joined["ens_id"]
    agree = joined.groupby("signal_date")["ok"].sum()
    return agree


def _b01a_series() -> pd.Series:
    from ..round2.overlays import breadth_tier_multiplier

    up = load_regime_frame()["up_ratio"]
    return breadth_tier_multiplier(up, 0.45, 0.50, 0.60)


def build_meta_frame(primary: pd.DataFrame) -> pd.DataFrame:
    """每个信号日一行：仅用当日及过去信息。"""
    preds = primary.copy()
    preds["signal_date"] = pd.to_datetime(preds["signal_date"])
    preds["instrument_id"] = preds["instrument_id"].astype(str)
    top = daily_top1(preds)
    margin = margin_frame_from_scores(preds, use_intraday_rank=True)
    n_cand = preds.groupby("signal_date")["instrument_id"].size().rename("n_cand")
    disp = preds.groupby("signal_date")["score"].std().rename("disp")
    agree = seed_agreement(preds).rename("agree_n")
    b01a = _b01a_series()
    try:
        eco = build_ecology_frame()
        relimit = eco["continue_rate"] if eco is not None and "continue_rate" in eco.columns else None
    except Exception:
        relimit = None

    panel = load_panel(LOCKED_FEATURES, "label_exit_rule_net")
    lab = panel.copy()
    lab["trade_date"] = pd.to_datetime(lab["trade_date"])
    lab["instrument_id"] = lab["instrument_id"].astype(str)
    feat_cols = [c for c in LOCKED_FEATURES if c in lab.columns]
    joined = top.merge(
        lab[["trade_date", "instrument_id", "label_exit_rule_net", LABEL_END, *feat_cols]].rename(
            columns={"trade_date": "signal_date"}
        ),
        on=["signal_date", "instrument_id"],
        how="left",
    )
    out = joined.set_index("signal_date").sort_index()
    out["margin"] = margin.reindex(out.index)["top1_top2_margin"]
    out["n_cand"] = n_cand.reindex(out.index)
    out["disp"] = disp.reindex(out.index)
    out["agree_n"] = agree.reindex(out.index)
    out["b01a"] = b01a.reindex(out.index)
    if relimit is not None:
        rel = relimit.copy()
        rel.index = pd.to_datetime(rel.index)
        out["relimit"] = rel.reindex(out.index)
    else:
        out["relimit"] = np.nan
    for i, col in enumerate(LOCKED_FEATURES):
        out[f"top1_f{i}"] = pd.to_numeric(out.get(col), errors="coerce")
    out["y_ret"] = pd.to_numeric(out.get("label_exit_rule_net"), errors="coerce")
    out["y_win"] = (out["y_ret"] > 0).astype(float)
    out["year"] = out.index.year
    # 行是 walk-forward Top1：y 是该笔事后收益，不能要求 exit < signal_date。
    # 成熟约束在 fit_oof_meta：训练样本 exit < 预测年 1 月 1 日。
    return out


def fit_oof_meta(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """外折：只用预测年之前、且标签已成熟的 OOF Top1。禁止 in-sample / 未来年。"""
    work = frame.copy()
    feat_cols = [c for c in META_FEATURE_NAMES if c in work.columns]
    x_all = work[feat_cols].replace([np.inf, -np.inf], np.nan)
    med = x_all.median()
    x_all = x_all.fillna(med)
    pred_ret = pd.Series(index=work.index, dtype=float)
    pred_win = pd.Series(index=work.index, dtype=float)
    folds = []
    years = sorted(int(y) for y in work["year"].unique() if 2018 <= int(y) <= 2026)
    exits = pd.to_datetime(work[LABEL_END], errors="coerce") if LABEL_END in work.columns else None
    for year in years:
        cutoff = pd.Timestamp(year, 1, 1)
        past = work["year"] < year
        if exits is not None:
            past = past & exits.lt(cutoff)
        train = work.loc[past]
        test = work.loc[work["year"] == year]
        if (train["year"] >= year).any():
            raise RuntimeError("outer-fold leakage")
        y = pd.to_numeric(train["y_ret"], errors="coerce")
        mask = y.notna()
        if mask.sum() < 30 or test.empty:
            pred_ret.loc[test.index] = np.nan
            pred_win.loc[test.index] = np.nan
            folds.append({"year": year, "status": "empty", "train_rows": int(mask.sum())})
            continue
        xt = x_all.loc[train.index[mask]]
        xv = x_all.loc[test.index]
        reg = Ridge(alpha=1.0)
        reg.fit(xt, y[mask].to_numpy())
        pred_ret.loc[test.index] = reg.predict(xv)
        yw = pd.to_numeric(train.loc[mask, "y_win"], errors="coerce")
        clf = LogisticRegression(max_iter=200)
        try:
            clf.fit(xt, yw.to_numpy())
            pred_win.loc[test.index] = clf.predict_proba(xv)[:, 1]
        except Exception:
            pred_win.loc[test.index] = 0.5
        folds.append({"year": year, "train_rows": int(mask.sum()), "test_rows": int(len(test))})
    return pred_ret, pred_win, {"folds": folds, "features": feat_cols, "oof_only": True}


def meta_multiplier(pred_ret: pd.Series, pred_win: pd.Series, kind: str) -> pd.Series:
    """meta≤1。F01 硬门 10/30；F02 软门不 skip；F03 胜率底 20%×0.5。"""
    idx = pred_ret.index.union(pred_win.index)
    r = pd.to_numeric(pred_ret.reindex(idx), errors="coerce")
    w = pd.to_numeric(pred_win.reindex(idx), errors="coerce")
    # PIT：分位只用过去（expanding）
    rp = r.expanding(min_periods=40).rank(pct=True)
    out = pd.Series(1.0, index=idx)
    if kind == "F01":
        out = out.where(~rp.le(0.30), 0.5)
        out = out.where(~rp.le(0.10), 0.0)
        out = out.where(rp.notna(), 1.0)
    elif kind == "F02":
        out = out.where(~rp.le(0.30), 0.5)
        out = out.where(~rp.le(0.10), 0.5)  # 软门不 skip
        out = out.where(rp.notna(), 1.0)
    elif kind == "F03":
        out = out.where(~w.lt(0.20), 0.5)
        out = out.where(w.notna(), 1.0)
    else:
        raise ValueError(kind)
    return out.clip(0.0, 1.0)


def top1_agreement_rate(a: pd.DataFrame, b: pd.DataFrame) -> float:
    ta = daily_top1(a)[["signal_date", "instrument_id"]]
    tb = daily_top1(b)[["signal_date", "instrument_id"]]
    m = ta.merge(tb, on="signal_date", suffixes=("_a", "_b"))
    if m.empty:
        return float("nan")
    return float((m["instrument_id_a"] == m["instrument_id_b"]).mean())


def context_series() -> dict[str, pd.Series]:
    """H：breadth / 连板率 / 首板数 / 候选数，只作 context。"""
    regime = load_regime_frame()
    out: dict[str, pd.Series] = {}
    if "up_ratio" in regime.columns:
        out["breadth"] = pd.to_numeric(regime["up_ratio"], errors="coerce")
    if "con_ratio" in regime.columns:
        out["relimit"] = pd.to_numeric(regime["con_ratio"], errors="coerce")
    try:
        eco = build_ecology_frame()
        if "continue_rate" in eco.columns:
            s = pd.to_numeric(eco["continue_rate"], errors="coerce")
            s.index = pd.to_datetime(s.index)
            out["relimit"] = s
        if "fb_count" in eco.columns:
            s = pd.to_numeric(eco["fb_count"], errors="coerce")
            s.index = pd.to_datetime(s.index)
            out["fb_count"] = s
    except Exception:
        pass
    panel = load_panel(LOCKED_FEATURES, "label_exit_rule_net")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    n = panel.groupby("trade_date")["instrument_id"].size()
    n.index = pd.to_datetime(n.index)
    out["n_cand"] = n
    return out


def context_multiplier(series: pd.Series, *, low: float = 0.20, high: float = 0.80) -> pd.Series:
    """PIT rolling250：两端半仓，不做 MoE。"""
    x = pd.to_numeric(series, errors="coerce").sort_index()
    p = x.rolling(250, min_periods=40).rank(pct=True)
    out = pd.Series(1.0, index=p.index)
    out = out.where(~p.le(low), 0.5)
    out = out.where(~p.ge(high), 0.5)
    return out.where(p.notna(), 1.0).clip(0.0, 1.0)
