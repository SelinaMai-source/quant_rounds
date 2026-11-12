"""Pair-conditional：先 PIT 诊断，稳定跨年才实跑 B01–B03。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND10_ROOT, assert_project_result_path, ensure_round10_tree
from .allocator import write_alloc
from .contract import INDUSTRY_SNAPSHOT, PARENT_CODE
from .daily import load_daily, persist_daily
from .registry import get_spec, specs_for_stage
from .runner import run_codes


def _pit_industry(dates: pd.DatetimeIndex, instruments: pd.Series) -> pd.Series:
    if not INDUSTRY_SNAPSHOT.exists():
        return pd.Series(pd.NA, index=instruments.index, dtype="object")
    files = list(INDUSTRY_SNAPSHOT.rglob("*.parquet"))
    if not files:
        return pd.Series(pd.NA, index=instruments.index, dtype="object")
    raw = pd.read_parquet(files[0])
    raw["instrument_id"] = raw["instrument_id"].astype(str)
    for col in ("effective_from", "effective_to", "known_at"):
        raw[col] = pd.to_datetime(raw[col], errors="coerce")
    prefer = raw["classification"].astype(str).eq("SW_L1:V2021")
    raw = raw.sort_values(["instrument_id", "known_at"])
    out = []
    want = pd.DataFrame({
        "signal_date": pd.to_datetime(dates).normalize(),
        "instrument_id": instruments.astype(str).to_numpy(),
    })
    by_id = {k: g for k, g in raw.groupby("instrument_id", sort=False)}
    for day, iid in zip(want["signal_date"], want["instrument_id"]):
        g = by_id.get(iid)
        if g is None:
            out.append(pd.NA)
            continue
        ok = (g["known_at"] <= day) & (g["effective_from"] <= day) & (g["effective_to"].isna() | (g["effective_to"] > day))
        hit = g.loc[ok]
        if hit.empty:
            out.append(pd.NA)
            continue
        pref = hit.loc[hit["classification"].astype(str).eq("SW_L1:V2021")]
        use = pref if not pref.empty else hit
        out.append(str(use.iloc[-1]["industry_id"]))
    return pd.Series(out, index=instruments.index, dtype="object")


def _attach_returns(daily: pd.DataFrame) -> pd.DataFrame:
    from .contract import HERMES_DAILY_ADJ, HERMES_VALUATION

    ids = sorted(set(daily["top1_id"].astype(str)) | set(daily["top2_id"].astype(str)))
    out = daily.copy()
    if HERMES_VALUATION.exists():
        try:
            import pyarrow.dataset as ds

            val = ds.dataset(str(HERMES_VALUATION), format="parquet").to_table(
                columns=["trade_date", "instrument_id", "circulating_market_cap", "turnover_rate"],
                filter=ds.field("instrument_id").isin(ids[:4000]),
            ).to_pandas()
            val["trade_date"] = pd.to_datetime(val["trade_date"]).dt.normalize()
            val["instrument_id"] = val["instrument_id"].astype(str)
            for rank, prefix in ((1, "top1"), (2, "top2")):
                keys = out.reset_index()[["signal_date", f"{prefix}_id"]].rename(
                    columns={"signal_date": "trade_date", f"{prefix}_id": "instrument_id"}
                )
                m = keys.merge(val, on=["trade_date", "instrument_id"], how="left")
                out[f"{prefix}_market_cap"] = pd.to_numeric(m["circulating_market_cap"], errors="coerce").to_numpy()
                out[f"{prefix}_turnover"] = pd.to_numeric(m["turnover_rate"], errors="coerce").to_numpy()
        except Exception as exc:
            out.attrs["valuation_error"] = repr(exc)
    if "top1_market_cap" in out.columns:
        out["market_cap_ratio"] = out["top2_market_cap"] / out["top1_market_cap"].replace(0, np.nan)
        out["turnover_ratio"] = out["top2_turnover"] / out["top1_turnover"].replace(0, np.nan)
    if HERMES_DAILY_ADJ.exists():
        try:
            import pyarrow.dataset as ds

            adj = ds.dataset(str(HERMES_DAILY_ADJ), format="parquet").to_table(
                columns=["trade_date", "instrument_id", "close_post"],
                filter=ds.field("instrument_id").isin(ids),
            ).to_pandas()
            adj["trade_date"] = pd.to_datetime(adj["trade_date"]).dt.normalize()
            adj["instrument_id"] = adj["instrument_id"].astype(str)
            adj = adj.sort_values(["instrument_id", "trade_date"])
            adj["ret"] = adj.groupby("instrument_id")["close_post"].pct_change()
            wide = adj.pivot(index="trade_date", columns="instrument_id", values="ret")
            c20, c60, p20_1, p20_2 = [], [], [], []
            for day, r in out.iterrows():
                d = pd.Timestamp(day)
                hist20 = wide.loc[(wide.index < d)].tail(20)
                hist60 = wide.loc[(wide.index < d)].tail(60)
                a, b = str(r["top1_id"]), str(r["top2_id"])
                def _corr(hist):
                    if a not in hist.columns or b not in hist.columns:
                        return np.nan
                    pair = hist[[a, b]].dropna()
                    if len(pair) < 8:
                        return np.nan
                    return float(pair[a].corr(pair[b]))
                c20.append(_corr(hist20))
                c60.append(_corr(hist60))
                if a in hist20.columns:
                    p20_1.append(float((1 + hist20[a].dropna()).prod() - 1) if hist20[a].notna().any() else np.nan)
                else:
                    p20_1.append(np.nan)
                if b in hist20.columns:
                    p20_2.append(float((1 + hist20[b].dropna()).prod() - 1) if hist20[b].notna().any() else np.nan)
                else:
                    p20_2.append(np.nan)
            out["pair_corr_20"] = c20
            out["pair_corr_60"] = c60
            out["top1_past20_return"] = p20_1
            out["top2_past20_return"] = p20_2
        except Exception as exc:
            out.attrs["adj_error"] = repr(exc)
    return out


def _year_sign_stable(daily: pd.DataFrame, mask_a: pd.Series, mask_b: pd.Series, col: str = "top2_ret") -> dict[str, Any]:
    y = pd.to_numeric(daily.get(col), errors="coerce")
    by = {}
    for year in range(2018, 2027):
        ya = y.loc[mask_a & (daily.index.year == year)].dropna()
        yb = y.loc[mask_b & (daily.index.year == year)].dropna()
        by[str(year)] = {
            "n_a": int(len(ya)), "n_b": int(len(yb)),
            "mean_a": float(ya.mean()) if len(ya) else None,
            "mean_b": float(yb.mean()) if len(yb) else None,
            "delta": (float(ya.mean() - yb.mean()) if len(ya) >= 10 and len(yb) >= 10 else None),
        }
    deltas = [v["delta"] for v in by.values() if v["delta"] is not None]
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    return {
        "by_year": by,
        "n_years": len(deltas),
        "pos_years": pos,
        "neg_years": neg,
        "stable_same_worse": neg >= 4 and pos <= 2,
        "stable_same_better": pos >= 4 and neg <= 2,
        "mean_a": float(y.loc[mask_a].mean()) if y.loc[mask_a].notna().any() else None,
        "mean_b": float(y.loc[mask_b].mean()) if y.loc[mask_b].notna().any() else None,
        "n_a": int(mask_a.sum()),
        "n_b": int(mask_b.sum()),
    }


def run_pair(*, resume: bool = True, jobs: int = 1, experiment: str = "") -> dict[str, Any]:
    ensure_round10_tree()
    daily = load_daily()
    persist_daily(daily)
    daily = daily.copy()
    daily["top1_industry"] = _pit_industry(daily.index, daily["top1_id"])
    daily["top2_industry"] = _pit_industry(daily.index, daily["top2_id"])
    daily["same_industry_L1"] = (
        daily["top1_industry"].notna()
        & daily["top2_industry"].notna()
        & daily["top1_industry"].astype(str).eq(daily["top2_industry"].astype(str))
    )
    daily["same_industry_L2"] = pd.NA
    pit_cov = float(daily["top1_industry"].notna().mean())
    daily = _attach_returns(daily)
    panel_path = ROUND10_ROOT / "diagnostics" / "top1_top2_pair_panel.parquet"
    assert_project_result_path(panel_path)
    daily.reset_index().to_parquet(panel_path, index=False)

    known = daily["top1_industry"].notna() & daily["top2_industry"].notna()
    industry = _year_sign_stable(daily, daily["same_industry_L1"] & known, (~daily["same_industry_L1"]) & known)
    corr = daily.get("pair_corr_20")
    corr_diag = {"status": "MISSING"}
    if corr is not None and pd.to_numeric(corr, errors="coerce").notna().sum() >= 80:
        q = pd.to_numeric(corr, errors="coerce")
        cuts = q.quantile([0.33, 0.67])
        low, high = q.le(cuts.iloc[0]), q.gt(cuts.iloc[1])
        mid = ~low & ~high & q.notna()
        means = {
            "low": float(daily.loc[low, "top2_ret"].mean()) if daily.loc[low, "top2_ret"].notna().any() else None,
            "mid": float(daily.loc[mid, "top2_ret"].mean()) if daily.loc[mid, "top2_ret"].notna().any() else None,
            "high": float(daily.loc[high, "top2_ret"].mean()) if daily.loc[high, "top2_ret"].notna().any() else None,
        }
        mono_down = (
            means["low"] is not None and means["mid"] is not None and means["high"] is not None
            and means["low"] >= means["mid"] >= means["high"]
        )
        high_vs_low = _year_sign_stable(daily, high, low)
        corr_diag = {
            "means": means,
            "mono_down": mono_down,
            "high_vs_low": high_vs_low,
            "cuts": [float(cuts.iloc[0]), float(cuts.iloc[1])],
            "n": int(q.notna().sum()),
        }

    rel = {"status": "CHECK"}
    gap = pd.to_numeric(daily.get("score_gap"), errors="coerce")
    if gap.notna().sum() >= 80:
        cut = float(gap.quantile(0.80))
        weak = gap.ge(cut)  # top2 much weaker than top1
        rel = _year_sign_stable(daily, weak, ~weak & gap.notna())
        rel["cut80"] = cut

    skip_b01 = pit_cov < 0.70 or not (industry.get("stable_same_worse") or industry.get("stable_same_better"))
    skip_b02 = not bool((corr_diag.get("high_vs_low") or {}).get("stable_same_worse") or (corr_diag.get("high_vs_low") or {}).get("stable_same_better") or corr_diag.get("mono_down"))
    skip_b03 = not (rel.get("stable_same_worse") or rel.get("stable_same_better"))

    b03 = None
    try:
        from .allocator import b03_nav
        b03 = b03_nav(daily["agreement"], daily["sep_bucket"])
    except Exception:
        b03 = pd.Series(0.0, index=daily.index)

    decisions = {}
    if skip_b01:
        get_spec("R10_B01").skip = True
        get_spec("R10_B01").skip_reason = "SKIP_B01"
        decisions["R10_B01"] = "SKIP_B01"
    else:
        get_spec("R10_B01").skip = False
        get_spec("R10_B01").skip_reason = ""
        nav = b03.copy()
        if industry.get("stable_same_worse"):
            nav = nav.where(~daily["same_industry_L1"], 0.0)
            decisions["R10_B01"] = "same_L1_veto"
        else:
            nav = nav.where(daily["same_industry_L1"] | ~known, nav.clip(upper=0.05))
            decisions["R10_B01"] = "cross_L1_cap5"
        write_alloc("R10_B01", daily.index, nav)

    if skip_b02:
        get_spec("R10_B02").skip = True
        get_spec("R10_B02").skip_reason = "SKIP_B02"
        decisions["R10_B02"] = "SKIP_B02"
    else:
        get_spec("R10_B02").skip = False
        get_spec("R10_B02").skip_reason = ""
        q = pd.to_numeric(daily.get("pair_corr_20"), errors="coerce")
        cuts = q.quantile([0.33, 0.67])
        high = q.gt(cuts.iloc[1])
        nav = b03.copy()
        if (corr_diag.get("high_vs_low") or {}).get("stable_same_worse") or corr_diag.get("mono_down"):
            nav = nav.where(~high, 0.5 * nav)
            decisions["R10_B02"] = "high_corr_half"
        else:
            low = q.le(cuts.iloc[0])
            nav = nav.where(~low, 0.5 * nav)
            decisions["R10_B02"] = "low_corr_half"
        write_alloc("R10_B02", daily.index, nav)

    if skip_b03:
        get_spec("R10_B03").skip = True
        get_spec("R10_B03").skip_reason = "SKIP_B03"
        decisions["R10_B03"] = "SKIP_B03"
    else:
        get_spec("R10_B03").skip = False
        get_spec("R10_B03").skip_reason = ""
        cut = float(rel.get("cut80") or gap.quantile(0.80))
        weak = gap.ge(cut)
        nav = b03.copy()
        if rel.get("stable_same_worse"):
            nav = nav.where(~weak, 0.0)
            decisions["R10_B03"] = "large_gap_veto"
        else:
            nav = nav.where(weak, nav.clip(upper=0.05))
            decisions["R10_B03"] = "small_gap_cap5"
        write_alloc("R10_B03", daily.index, nav)

    codes = [c.strip() for c in experiment.split(",") if c.strip()] if experiment else [s.code for s in specs_for_stage("pair")]
    bt = run_codes(codes, resume=resume, jobs=jobs)
    payload = {
        "parent": PARENT_CODE,
        "pit_l1_coverage": pit_cov,
        "l2": "NOT_AVAILABLE",
        "industry": industry,
        "corr": corr_diag,
        "relative": rel,
        "decisions": decisions,
        "failed": bt.get("failed") or [],
        "jobs": {k: v.get("status") for k, v in (bt.get("jobs") or {}).items()},
    }
    path = ROUND10_ROOT / "metrics" / "pair_diagnostics.json"
    assert_project_result_path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    from .docs import write_named_md

    extra = (
        f"PIT L1 覆盖 {pit_cov:.1%}。L2 = NOT_AVAILABLE，禁止申万回填。\n"
        f"B 决策：{decisions}。\n"
        "不允许预设分散更好；先看跨年符号是否稳定。\n"
    )
    write_named_md("02_pair_conditional_utility.md", "pair", codes, extra)
    return payload


__all__ = ["run_pair"]
