"""Top1 归因 + T01/T02。D+1 竞价不得进 D-close 模型。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND15_ROOT, stock_path, CANDIDATE_PANEL_RELATIVE
from ..round8.overlays import MIN_KEEP
from .common import dump_json, load_json, observed_from_report, pct, write_md
from .contract import (
    BANNED_DPLUS1,
    R15_00,
    SEAL_D_FEATURES,
    T01,
    T01_MIN_N,
    T02,
    T02_MIN_N,
    T02_MIN_OOS_R2,
    experiment_dir,
    overlayed_path,
    parity_dir,
)
from .freeze import load_skips, require_frozen
from .runner import apply_top1_skip, build_version_overlayed, hist_hooks, load_market, run_r15_backtest

TOP1_JSON = ROUND15_ROOT / "top1" / "top1.json"
ATTR_MD = ROUND15_ROOT / "top1" / "top1_alpha_attribution.md"
TOP1_MD = ROUND15_ROOT / "07_top1.md"


def _v4_trades() -> pd.DataFrame:
    path = parity_dir(R15_00) / "trades.parquet"
    t = pd.read_parquet(path)
    t["signal_date"] = pd.to_datetime(t["signal_date"]).dt.normalize()
    t["instrument_id"] = t["instrument_id"].astype(str)
    t["_rank"] = t.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    t["net"] = pd.to_numeric(t.get("sell_proceeds"), errors="coerce") - pd.to_numeric(t.get("buy_cost"), errors="coerce")
    t["gross_ret"] = pd.to_numeric(t.get("ret"), errors="coerce")
    t["buy_cost"] = pd.to_numeric(t.get("buy_cost"), errors="coerce")
    t["sell_proceeds"] = pd.to_numeric(t.get("sell_proceeds"), errors="coerce")
    t["year"] = t["signal_date"].dt.year
    return t


def _cost_hurdle(top1: pd.DataFrame) -> float:
    """训练期锁定：2018–2021 的中位 |净边| 中，成本占名义的尺度。用亏损侧 25% 作为不下单门槛。"""
    train = top1.loc[top1["year"].between(2018, 2021)]
    net = pd.to_numeric(train["net"], errors="coerce").dropna()
    if len(net) < T01_MIN_N:
        net = pd.to_numeric(top1["net"], errors="coerce").dropna()
    if net.empty:
        return 0.0
    # 预测净边低于该门槛则跳过：取训练期净边的 0 与成本拖累。用 0 作为执行成本要求（净边须为正）。
    # 更严：训练期负净边的中位绝对值作为成本要求。
    neg = net.loc[net < 0]
    if len(neg) >= 20:
        return float(abs(neg.median()))
    return 0.0


def _attach_d_features(top: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    candidates = [
        stock_path("data/first_board_hermes/aligned_original_candidate_full44_plus_d1_open_gap_v1/candidate_panel.parquet"),
        stock_path(str(CANDIDATE_PANEL_RELATIVE)),
    ]
    panel_path = next((p for p in candidates if p.is_file()), None)
    if panel_path is None:
        return top, []
    try:
        import pyarrow.parquet as pq

        have = set(pq.ParquetFile(str(panel_path)).schema.names)
    except Exception:
        have = set(pd.read_parquet(panel_path).columns)
    keep = [c for c in SEAL_D_FEATURES if c in have and c not in BANNED_DPLUS1]
    date_col = "trade_date" if "trade_date" in have else ("signal_date" if "signal_date" in have else None)
    if date_col is None or not keep:
        return top, []
    cols = [date_col, "instrument_id", *keep]
    panel = pd.read_parquet(panel_path, columns=cols)
    panel["signal_date"] = pd.to_datetime(panel[date_col]).dt.normalize()
    panel["instrument_id"] = panel["instrument_id"].astype(str)
    merged = top.merge(panel[ ["signal_date", "instrument_id", *keep] ], on=["signal_date", "instrument_id"], how="left")
    return merged, keep


def _oof_predict(frame: pd.DataFrame, ycol: str, xcols: list[str]) -> tuple[pd.Series, dict[str, Any]]:
    """按年留出的线性 OOF。无 sklearn 时用 numpy lstsq。"""
    y = pd.to_numeric(frame[ycol], errors="coerce")
    X = frame[xcols].apply(pd.to_numeric, errors="coerce")
    pred = pd.Series(np.nan, index=frame.index)
    years = sorted(frame["year"].dropna().unique())
    r2s = []
    for y0 in years:
        tr = frame["year"] < y0
        te = frame["year"] == y0
        if tr.sum() < T01_MIN_N or te.sum() < 10:
            continue
        Xtr = X.loc[tr].fillna(X.loc[tr].median())
        ytr = y.loc[tr]
        mask = ytr.notna() & np.isfinite(Xtr).all(axis=1)
        Xtr, ytr = Xtr.loc[mask], ytr.loc[mask]
        if len(Xtr) < T01_MIN_N:
            continue
        A = np.column_stack([np.ones(len(Xtr)), Xtr.to_numpy(dtype=float)])
        try:
            coef, *_ = np.linalg.lstsq(A, ytr.to_numpy(dtype=float), rcond=None)
        except np.linalg.LinAlgError:
            continue
        Xte = X.loc[te].fillna(X.loc[tr].median())
        B = np.column_stack([np.ones(len(Xte)), Xte.to_numpy(dtype=float)])
        yhat = B @ coef
        pred.loc[te] = yhat
        yte = y.loc[te]
        ok = yte.notna() & np.isfinite(yhat)
        if ok.sum() >= 10:
            yt = yte.loc[ok].to_numpy(dtype=float)
            yp = yhat[ok.to_numpy()]
            ss_res = float(np.sum((yt - yp) ** 2))
            ss_tot = float(np.sum((yt - yt.mean()) ** 2))
            r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    meta = {
        "oos_r2_mean": float(np.mean(r2s)) if r2s else None,
        "oos_r2_by_fold": r2s,
        "n_pred": int(pred.notna().sum()),
        "xcols": xcols,
        "banned_dplus1": list(BANNED_DPLUS1),
    }
    return pred, meta


def _attribution(top1: pd.DataFrame, top2: pd.DataFrame, feat_cols: list[str]) -> str:
    def pack(df: pd.DataFrame) -> dict[str, Any]:
        net = pd.to_numeric(df["net"], errors="coerce")
        gr = pd.to_numeric(df["gross_ret"], errors="coerce")
        return {
            "n": int(len(df)),
            "net_sum": float(net.sum()) if net.notna().any() else None,
            "net_mean": float(net.mean()) if net.notna().any() else None,
            "gross_mean": float(gr.mean()) if gr.notna().any() else None,
            "win_rate": float((net > 0).mean()) if len(df) else None,
            "left_tail_net": float(net.quantile(0.05)) if net.notna().any() else None,
        }
    a1 = pack(top1)
    a2 = pack(top2)
    lines = [
        "# Top1 alpha attribution",
        "",
        "D-day 封板/尾盘压力可用现有首板特征。**D+1 竞价不得进 D-close 模型。**",
        "",
        f"Top1 n={a1['n']} net_sum={a1['net_sum']} net_mean={a1['net_mean']} win={a1['win_rate']} left5={a1['left_tail_net']}",
        f"Top2 n={a2['n']} net_sum={a2['net_sum']} net_mean={a2['net_mean']} win={a2['win_rate']} left5={a2['left_tail_net']}",
        "",
        f"D 特征列 = {feat_cols or '（面板缺列，仅用 score）'}",
        f"禁止列 = {list(BANNED_DPLUS1)}",
        "",
        "## 失败模式",
        "",
        "净边为负的 Top1 交易视为失败样本；不把 D+1 开盘缺口当 D-close 可交易信息。",
        "",
    ]
    if "score" in top1.columns:
        q = pd.qcut(pd.to_numeric(top1["score"], errors="coerce"), 4, duplicates="drop")
        grp = top1.groupby(q, observed=False)["net"].mean()
        lines += ["## 按 score 四分位的 Top1 净边均值", "", str(grp.to_dict()), ""]
    return "\n".join(lines)


def run_top1(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and TOP1_JSON.is_file():
        existing = load_json(TOP1_JSON)
        if existing.get("status") in {"DONE", "PARTIAL"}:
            return existing
    skips = load_skips()
    trades = _v4_trades()
    top1 = trades.loc[trades["_rank"].eq(1)].copy()
    top2 = trades.loc[trades["_rank"].eq(2)].copy()
    top1, feat_cols = _attach_d_features(top1)
    top2, _ = _attach_d_features(top2)
    write_md(ATTR_MD, _attribution(top1, top2, feat_cols))

    xcols = ["score", *feat_cols]
    xcols = [c for c in xcols if c in top1.columns]
    pred, oof_meta = _oof_predict(top1, "net", xcols)
    top1["pred_net"] = pred
    hurdle = _cost_hurdle(top1)
    skip_dates = top1.loc[top1["pred_net"].notna() & (top1["pred_net"] < hurdle), "signal_date"]
    t01 = {"code": T01, "skip": False, "eligible": True}
    t02 = {"code": T02, "skip": True, "status": "SKIP_T02", "eligible": False}

    loaded = load_market()
    ov4, _, _ = build_version_overlayed("v4")
    results: dict[str, Any] = {}

    if skips.get("SKIP_T01") or oof_meta.get("n_pred", 0) < T01_MIN_N:
        t01 = {"code": T01, "skip": True, "status": "SKIP_T01", "eligible": False, "reason": "insufficient OOF preds"}
        results[T01] = t01
    else:
        ov = apply_top1_skip(ov4, skip_dates)
        overlayed_path(T01).parent.mkdir(parents=True, exist_ok=True)
        ov.to_parquet(overlayed_path(T01), index=False)
        ran = run_r15_backtest(
            experiment_dir(T01),
            overlayed=ov,
            validation_hooks=hist_hooks(),
            extra_report={"candidate": T01, "family": "top1", "n_skip_days": int(skip_dates.nunique())},
            resume=resume,
            prepared_market=loaded["prepared"],
            market_meta=loaded,
        )
        t01 = observed_from_report(ran.get("report") or {})
        t01.update({
            "code": T01,
            "status": ran.get("status"),
            "skip": False,
            "eligible": True,
            "n_skip_days": int(pd.Series(skip_dates).nunique()),
            "hurdle": hurdle,
            "nav_path": str(experiment_dir(T01) / "nav.parquet"),
            "keep_original_rank": True,
        })
        results[T01] = t01

    r2 = oof_meta.get("oos_r2_mean")
    t02_ok = (
        not skips.get("SKIP_T02")
        and r2 is not None
        and float(r2) >= T02_MIN_OOS_R2
        and int(oof_meta.get("n_pred") or 0) >= T02_MIN_N
    )
    if not t02_ok:
        t02 = {
            "code": T02,
            "skip": True,
            "status": "SKIP_T02",
            "eligible": False,
            "reason": f"OOF R2={r2} n={oof_meta.get('n_pred')} threshold R2>={T02_MIN_OOS_R2} n>={T02_MIN_N}",
            "no_reliable_counterfactual_for_non_top1": True,
        }
        results[T02] = t02
    else:
        # 有限 rerank：仅在当日已交易 Top1/Top2 之间按 OOF 净边互换。
        top2b, _ = _attach_d_features(trades.loc[trades["_rank"].eq(2)].copy())
        pred2, _ = _oof_predict(top2b, "net", [c for c in xcols if c in top2b.columns] or ["score"])
        top2b["pred_net"] = pred2
        a = top1[["signal_date", "instrument_id", "pred_net"]].rename(
            columns={"instrument_id": "id1", "pred_net": "p1"}
        )
        b = top2b[["signal_date", "instrument_id", "pred_net"]].rename(
            columns={"instrument_id": "id2", "pred_net": "p2"}
        )
        both = a.merge(b, on="signal_date", how="inner")
        swap_days = both.loc[both["p1"].notna() & both["p2"].notna() & (both["p2"] > both["p1"]), "signal_date"]
        work = ov4.copy()
        work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
        work["_rank"] = work.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
        # 互换：把这两行的 allocation_multiplier 对调，保持 MIN_KEEP 约束。
        swap_set = set(pd.to_datetime(swap_days).normalize())
        rows = []
        for dt, g in work.groupby("signal_date", sort=False):
            gg = g.copy()
            if dt in swap_set:
                m1 = gg["_rank"].eq(1)
                m2 = gg["_rank"].eq(2)
                if m1.any() and m2.any():
                    w1 = float(gg.loc[m1, "allocation_multiplier"].iloc[0])
                    w2 = float(gg.loc[m2, "allocation_multiplier"].iloc[0])
                    gg.loc[m1, "allocation_multiplier"] = max(w2, MIN_KEEP)
                    gg.loc[m2, "allocation_multiplier"] = max(w1, MIN_KEEP)
                    # 交换 score 排序键会改 head(top_k) 身份：对调 score 使 engine rank 交换
                    s1 = float(gg.loc[m1, "score"].iloc[0])
                    s2 = float(gg.loc[m2, "score"].iloc[0])
                    gg.loc[m1, "score"] = s2 + 1e-12
                    gg.loc[m2, "score"] = s1
            rows.append(gg)
        ovt = pd.concat(rows, ignore_index=True).drop(columns=["_rank"], errors="ignore")
        overlayed_path(T02).parent.mkdir(parents=True, exist_ok=True)
        ovt.to_parquet(overlayed_path(T02), index=False)
        ran = run_r15_backtest(
            experiment_dir(T02),
            overlayed=ovt,
            validation_hooks=hist_hooks(),
            extra_report={"candidate": T02, "family": "top1", "n_swap_days": int(pd.Series(swap_days).nunique())},
            resume=resume,
            prepared_market=loaded["prepared"],
            market_meta=loaded,
        )
        t02 = observed_from_report(ran.get("report") or {})
        t02.update({
            "code": T02,
            "status": ran.get("status"),
            "skip": False,
            "eligible": True,
            "n_swap_days": int(pd.Series(swap_days).nunique()),
            "nav_path": str(experiment_dir(T02) / "nav.parquet"),
        })
        results[T02] = t02

    payload = {
        "status": "DONE",
        "results": results,
        "oof": oof_meta,
        "hurdle": hurdle,
        "n_top1": int(len(top1)),
        "attribution_path": str(ATTR_MD),
        "dplus1_in_d_close_model": False,
    }
    dump_json(TOP1_JSON, payload)
    lines = [
        "# Round 15 · Top1",
        "",
        f"归因：`{ATTR_MD}`。D+1 竞价未进 D-close 模型。",
        f"OOF R2={oof_meta.get('oos_r2_mean')} n_pred={oof_meta.get('n_pred')} hurdle={hurdle}",
        "",
        "| ID | status | CAGR | Sharpe | MaxDD |",
        "|---|---|---:|---:|---:|",
    ]
    for code in (T01, T02):
        row = results.get(code) or {}
        lines.append(
            f"| `{code}` | {row.get('status')} | {pct(row.get('cagr'))} | {row.get('sharpe')} | {pct(row.get('max_drawdown'))} |"
        )
    write_md(TOP1_MD, "\n".join(lines) + "\n")
    return payload


def load_top1() -> dict[str, Any]:
    return load_json(TOP1_JSON)


__all__ = ["load_top1", "run_top1"]
