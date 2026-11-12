"""B03 选股 vs ALPHA-INDUCED ALLOCATION EFFECT。完整引擎 counterfactual。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...round17.backtest import run_canonical
from ...round18.backtest import summarize
from ..common import slim_obs
from ..contract import HISTORICAL_FEE_CORRECTED, NON_ADDITIVE_PATH_DEPENDENCE
from .retrain import run_fresh_retrain
from .support import (
    B00,
    B01,
    add_rank,
    cand_dir,
    dump_json,
    load_json,
    load_overlay,
    tb_dir,
    write_md,
)


def _ensure(**kw: Any) -> None:
    if not (tb_dir("retrain") / "fresh_retrain.json").is_file():
        run_fresh_retrain(**kw)


def _rank_alloc_map(overlay: pd.DataFrame) -> pd.DataFrame:
    work = add_rank(overlay)
    if "allocation_multiplier" not in work.columns:
        work["allocation_multiplier"] = 1.0
    return work[["signal_date", "_rank", "allocation_multiplier"]].drop_duplicates(
        ["signal_date", "_rank"], keep="first"
    )


def transfer_alloc(score_src: pd.DataFrame, alloc_src: pd.DataFrame) -> pd.DataFrame:
    """Keep score_src names/ranks; apply alloc_src multipliers by (date, rank slot)."""
    scored = add_rank(score_src)
    mapping = _rank_alloc_map(alloc_src)
    out = scored.merge(mapping, on=["signal_date", "_rank"], how="left", suffixes=("", "_map"))
    col = "allocation_multiplier_map" if "allocation_multiplier_map" in out.columns else "allocation_multiplier"
    if "allocation_multiplier" in score_src.columns and col == "allocation_multiplier":
        # mapping already named allocation_multiplier after merge
        pass
    out["allocation_multiplier"] = pd.to_numeric(out[col], errors="coerce").fillna(1.0).clip(1e-12, 1.0)
    keep = ["signal_date", "instrument_id", "score", "allocation_multiplier"]
    extra = [c for c in ("score_five", "score_ten") if c in out.columns]
    return out[keep + extra]


def _same_selection_diff_alloc(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, Any]:
    ta = add_rank(a)
    tb = add_rank(b)
    t2a = ta.loc[ta["_rank"] <= 2].groupby("signal_date")["instrument_id"].apply(
        lambda s: tuple(sorted(map(str, s)))
    )
    t2b = tb.loc[tb["_rank"] <= 2].groupby("signal_date")["instrument_id"].apply(
        lambda s: tuple(sorted(map(str, s)))
    )
    both = pd.concat([t2a.rename("a"), t2b.rename("b")], axis=1).dropna()
    same = both.loc[both["a"] == both["b"]]
    if "allocation_multiplier" not in ta.columns or same.empty:
        return {"n_same_selection_days": int(len(same)), "n_same_sel_diff_alloc": None}
    ma = _rank_alloc_map(a).rename(columns={"allocation_multiplier": "m_a"})
    mb = _rank_alloc_map(b).rename(columns={"allocation_multiplier": "m_b"})
    jm = ma.merge(mb, on=["signal_date", "_rank"], how="inner")
    jm = jm.loc[jm["signal_date"].isin(same.index)]
    diff = ~np.isclose(
        pd.to_numeric(jm["m_a"], errors="coerce"),
        pd.to_numeric(jm["m_b"], errors="coerce"),
        rtol=0,
        atol=1e-12,
    )
    n_days = int(jm.loc[diff, "signal_date"].nunique()) if len(jm) else 0
    return {
        "n_same_selection_days": int(len(same)),
        "n_same_sel_diff_alloc": n_days,
        "same_sel_diff_alloc_rate": float(n_days / max(len(same), 1)),
    }


def _run_cf(code: str, overlayed: pd.DataFrame, *, resume: bool) -> dict[str, Any]:
    dest = tb_dir("heads") / "cf" / code
    dest.mkdir(parents=True, exist_ok=True)
    if resume and (dest / "report.json").is_file() and (dest / "nav.parquet").is_file():
        report = load_json(dest / "report.json")
        return slim_obs(summarize({"status": "resumed", "report": report, "out_dir": str(dest)}))
    ran = run_canonical(
        dest,
        overlayed,
        extra={
            "code": code,
            "role": "allocation_counterfactual",
            "fee_schedule": HISTORICAL_FEE_CORRECTED,
            "diagnostic_not_eligible": True,
        },
        resume=resume,
        write=True,
    )
    return slim_obs(ran.get("observed") or summarize(ran))


def run_allocation(*, resume: bool = True, **kw: Any) -> dict[str, Any]:
    _ensure(resume=resume, **kw)
    ov0 = load_overlay(B00)
    ov1 = load_overlay(B01)
    sel = _same_selection_diff_alloc(ov0, ov1)
    t1a = add_rank(ov0)
    t1b = add_rank(ov1)
    j1 = t1a.loc[t1a["_rank"].eq(1), ["signal_date", "instrument_id"]].merge(
        t1b.loc[t1b["_rank"].eq(1), ["signal_date", "instrument_id"]],
        on="signal_date",
        how="inner",
        suffixes=("_b00", "_b01"),
    )
    top1_agree = float((j1["instrument_id_b00"] == j1["instrument_id_b01"]).mean()) if len(j1) else None
    t2a = t1a.loc[t1a["_rank"] <= 2, ["signal_date", "instrument_id"]]
    t2b = t1b.loc[t1b["_rank"] <= 2, ["signal_date", "instrument_id"]]
    both = t2a.merge(t2b, on=["signal_date", "instrument_id"], how="inner")
    top2_overlap = float(len(both) / max(len(t2a), 1))

    cf_sel1_alloc0 = transfer_alloc(ov1, ov0)
    cf_sel0_alloc1 = transfer_alloc(ov0, ov1)
    cf_a = _run_cf("R19_B_CF_SEL_B01_ALLOC_B00", cf_sel1_alloc0, resume=resume)
    cf_b = _run_cf("R19_B_CF_SEL_B00_ALLOC_B01", cf_sel0_alloc1, resume=resume)
    b00 = slim_obs((load_json(cand_dir(B00) / "summary.json") or {}).get("observed") or {})
    b01 = slim_obs((load_json(cand_dir(B01) / "summary.json") or {}).get("observed") or {})

    def _d(a: dict[str, Any], b: dict[str, Any], key: str) -> float | None:
        try:
            return float(a.get(key)) - float(b.get(key))
        except (TypeError, ValueError):
            return None

    d_sel = _d(cf_a, b00, "sharpe")  # B01 names + B00 alloc vs B00
    d_alloc = _d(cf_b, b00, "sharpe")  # B00 names + B01 alloc vs B00
    d_full = _d(b01, b00, "sharpe")
    additive = None
    path_dep = False
    if d_sel is not None and d_alloc is not None and d_full is not None:
        additive = float(d_sel + d_alloc)
        path_dep = abs(additive - d_full) > 0.02
    payload = {
        "status": "done",
        "top1_agree": top1_agree,
        "top2_overlap": top2_overlap,
        "selection_vs_alloc": sel,
        "B00": b00,
        "B01": b01,
        "CF_SEL_B01_ALLOC_B00": cf_a,
        "CF_SEL_B00_ALLOC_B01": cf_b,
        "delta_sharpe": {
            "full_b01_minus_b00": d_full,
            "selection_only": d_sel,
            "allocation_only": d_alloc,
            "sum_of_parts": additive,
        },
        "path_dependence": path_dep,
        "path_dependence_label": NON_ADDITIVE_PATH_DEPENDENCE if path_dep else None,
        "note": (
            "B03 格子随排序变化 = ALPHA-INDUCED ALLOCATION EFFECT。"
            "CF 用 (date, rank) 乘子迁移，不改公式、不搜权重。"
        ),
        "diagnostic_not_eligible": True,
    }
    dump_json(tb_dir("heads") / "allocation.json", payload)
    write_md(
        tb_dir("heads") / "10_allocation.md",
        "# Track B · B03 选股 vs ALPHA-INDUCED ALLOCATION\n\n"
        f"Top1 agree=`{top1_agree}` Top2 overlap=`{top2_overlap}`\n\n"
        f"- same-selection days=`{sel.get('n_same_selection_days')}` "
        f"same-sel-diff-alloc=`{sel.get('n_same_sel_diff_alloc')}`\n"
        f"- ΔSharpe full=`{d_full}` selection=`{d_sel}` allocation=`{d_alloc}` sum=`{additive}`\n"
        f"- path_dependence=`{path_dep}` label=`{payload['path_dependence_label']}`\n"
        f"- CF_SEL_B01_ALLOC_B00 Sharpe=`{cf_a.get('sharpe')}` trades=`{cf_a.get('completed_trades')}`\n"
        f"- CF_SEL_B00_ALLOC_B01 Sharpe=`{cf_b.get('sharpe')}` trades=`{cf_b.get('completed_trades')}`\n",
    )
    return payload


__all__ = ["run_allocation", "transfer_alloc"]
