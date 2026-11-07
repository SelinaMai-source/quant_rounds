"""Five/Ten OOF 诊断、排序不稳、Top1/Top2 变化日。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...round7.contract import FIVE_HEAD, TEN_HEAD
from .retrain import run_fresh_retrain
from .support import (
    add_rank,
    dump_json,
    finite_corr,
    load_b01_preds,
    load_original_panels,
    load_r7_preds,
    pred_keys,
    tb_dir,
    write_md,
    year_of,
)


def _ensure(**kw: Any) -> None:
    if not (tb_dir("retrain") / "fresh_retrain.json").is_file():
        run_fresh_retrain(**kw)


def _ndcg_at(block: pd.DataFrame, k: int, *, score="score", y="label") -> float | None:
    try:
        from sklearn.metrics import ndcg_score
    except ImportError:
        return None
    s = pd.to_numeric(block[score], errors="coerce")
    t = pd.to_numeric(block[y], errors="coerce")
    mask = s.notna() & t.notna()
    if int(mask.sum()) < 2:
        return None
    rel = t[mask].rank(method="average").to_numpy(dtype=float)
    val = float(ndcg_score([rel], [s[mask].to_numpy()], k=k, ignore_ties=False))
    return val if np.isfinite(val) else None


def _attach_labels(pred: pd.DataFrame, panel: pd.DataFrame, target: str) -> pd.DataFrame:
    work = pred_keys(pred)
    lab = panel.copy()
    lab["trade_date"] = pd.to_datetime(lab["trade_date"])
    lab["instrument_id"] = lab["instrument_id"].astype(str)
    cols = ["trade_date", "instrument_id", target]
    if target not in lab.columns:
        return work
    merged = work.merge(
        lab[cols].rename(columns={"trade_date": "signal_date", target: "label"}),
        on=["signal_date", "instrument_id"],
        how="left",
    )
    return merged


def _head_metrics(merged: pd.DataFrame) -> dict[str, Any]:
    if merged.empty or "label" not in merged.columns:
        return {"n": 0, "rank_ic": None, "ndcg@1": None, "ndcg@2": None, "top1_precision": None, "top2_precision": None}
    ics, n1, n2, p1, p2 = [], [], [], [], []
    ranked = add_rank(merged)
    for _, day in ranked.groupby("signal_date", sort=True):
        ic = finite_corr(day["score"], day["label"])
        if ic is not None:
            ics.append(ic)
        v1 = _ndcg_at(day, 1)
        if v1 is not None:
            n1.append(v1)
        v2 = _ndcg_at(day, 2)
        if v2 is not None:
            n2.append(v2)
        lab = pd.to_numeric(day["label"], errors="coerce")
        if lab.notna().sum() < 2:
            continue
        truth_rank = lab.rank(method="first", ascending=False)
        pred_top1 = day.loc[day["_rank"].eq(1)]
        if len(pred_top1):
            idx = pred_top1.index[0]
            p1.append(float(truth_rank.loc[idx] == 1.0) if pd.notna(truth_rank.loc[idx]) else 0.0)
        pred_top2 = set(day.loc[day["_rank"] <= 2, "instrument_id"].astype(str))
        true_top2 = set(day.loc[truth_rank <= 2, "instrument_id"].astype(str))
        if pred_top2 and true_top2:
            p2.append(len(pred_top2 & true_top2) / 2.0)
    return {
        "n": int(len(merged)),
        "n_days": int(merged["signal_date"].nunique()),
        "rank_ic": float(np.mean(ics)) if ics else None,
        "ndcg@1": float(np.mean(n1)) if n1 else None,
        "ndcg@2": float(np.mean(n2)) if n2 else None,
        "top1_precision": float(np.mean(p1)) if p1 else None,
        "top2_precision": float(np.mean(p2)) if p2 else None,
    }


def _by_year(merged: pd.DataFrame) -> dict[str, Any]:
    out = {}
    work = merged.copy()
    work["year"] = year_of(work["signal_date"])
    for year, block in work.groupby("year", sort=True):
        out[str(int(year))] = _head_metrics(block)
    return out


def run_heads(**kw: Any) -> dict[str, Any]:
    _ensure(**kw)
    r7 = load_r7_preds()
    b01 = load_b01_preds()
    orig = load_original_panels()
    rows = []
    yearly = {}
    for head, target in (("five", FIVE_HEAD.target), ("ten", TEN_HEAD.target)):
        panel = orig[f"{head}_candidate"]
        for tag, pred in (("B00", r7[head]), ("B01", b01[head])):
            merged = _attach_labels(pred, panel, target)
            met = _head_metrics(merged)
            met.update({"code": tag, "head": head})
            rows.append(met)
            yearly[f"{tag}_{head}"] = _by_year(merged)
    payload = {
        "status": "done",
        "rows": rows,
        "yearly": yearly,
        "diagnostic_not_eligible": True,
        "note": "OOF NDCG / Rank IC / Top1/2 precision. Diagnostic only.",
    }
    dump_json(tb_dir("heads") / "heads.json", payload)
    lines = [
        "# Track B · Five/Ten OOF",
        "",
        "| code | head | Rank IC | NDCG@1 | NDCG@2 | Top1 prec | Top2 prec | n |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['code']} | {r['head']} | {r.get('rank_ic')} | {r.get('ndcg@1')} | {r.get('ndcg@2')} | "
            f"{r.get('top1_precision')} | {r.get('top2_precision')} | {r.get('n')} |"
        )
    write_md(tb_dir("heads") / "08_heads.md", "\n".join(lines) + "\n")
    return payload


def run_top_change(**kw: Any) -> dict[str, Any]:
    _ensure(**kw)
    r7 = load_r7_preds()
    b01 = load_b01_preds()
    records = []
    summary = {}
    for head in ("five", "ten"):
        a = add_rank(r7[head])
        b = add_rank(b01[head])
        t1a = a.loc[a["_rank"].eq(1), ["signal_date", "instrument_id"]].rename(columns={"instrument_id": "b00"})
        t1b = b.loc[b["_rank"].eq(1), ["signal_date", "instrument_id"]].rename(columns={"instrument_id": "b01"})
        j1 = t1a.merge(t1b, on="signal_date", how="outer")
        j1["changed"] = j1["b00"].fillna("") != j1["b01"].fillna("")
        t2a = a.loc[a["_rank"] <= 2].groupby("signal_date")["instrument_id"].apply(lambda s: tuple(sorted(s.astype(str))))
        t2b = b.loc[b["_rank"] <= 2].groupby("signal_date")["instrument_id"].apply(lambda s: tuple(sorted(s.astype(str))))
        t2 = pd.concat([t2a.rename("b00"), t2b.rename("b01")], axis=1)
        t2["changed"] = t2["b00"] != t2["b01"]
        # Kendall on common keys, sampled by date
        inner = pred_keys(r7[head]).merge(
            pred_keys(b01[head]),
            on=["signal_date", "instrument_id"],
            how="inner",
            suffixes=("_b00", "_b01"),
        )
        taus = []
        for _, day in inner.groupby("signal_date"):
            if len(day) < 3:
                continue
            tau = day["score_b00"].corr(day["score_b01"], method="kendall")
            if tau is not None and np.isfinite(tau):
                taus.append(float(tau))
        n_days = int(j1["signal_date"].nunique())
        n_top1 = int(j1["changed"].sum())
        n_top2 = int(t2["changed"].sum())
        summary[head] = {
            "n_days": n_days,
            "n_top1_changed": n_top1,
            "top1_change_rate": float(n_top1 / n_days) if n_days else None,
            "n_top2_set_changed": n_top2,
            "top2_change_rate": float(n_top2 / max(len(t2), 1)),
            "mean_kendall_tau": float(np.mean(taus)) if taus else None,
        }
        changed = j1.loc[j1["changed"]].copy()
        changed["signal_date"] = pd.to_datetime(changed["signal_date"]).dt.strftime("%Y-%m-%d")
        changed["head"] = head
        records.extend(changed.to_dict(orient="records"))
    frame = pd.DataFrame(records)
    dest = tb_dir("heads") / "top1_change_days.csv"
    frame.to_csv(dest, index=False)
    payload = {
        "status": "done",
        "summary": summary,
        "n_change_rows": int(len(frame)),
        "path": str(dest),
        "diagnostic_not_eligible": True,
    }
    dump_json(tb_dir("heads") / "top_change.json", payload)
    write_md(
        tb_dir("heads") / "09_top_change.md",
        "# Track B · Top1/Top2 变化日\n\n"
        f"five: `{summary.get('five')}`\n\nten: `{summary.get('ten')}`\n\npath=`{dest}`\n",
    )
    return payload


__all__ = ["run_heads", "run_top_change"]
