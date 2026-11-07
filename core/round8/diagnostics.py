"""几何 / 分歧 / 左尾诊断。不改 Alpha。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND8_ROOT, assert_project_result_path, ensure_round8_tree, round8_dir
from .contract import ANCHOR_DIR, PARENT_CODE
from .geometry import build_geometry, eligible_heads
from .overlays import exact_v2_date_mult, market_b01a
from ..replay_round7.inject import load_locked_alpha

DECILES = list(range(10))


def _write(name: str, payload: Any) -> None:
    path = ROUND8_ROOT / "metrics" / name
    assert_project_result_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, pd.DataFrame):
        payload.to_parquet(path, index=False if "signal_date" in payload.columns else True)
    else:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _trade_rank_returns() -> pd.DataFrame:
    trades = pd.read_parquet(ANCHOR_DIR / "trades.parquet")
    trades["signal_date"] = pd.to_datetime(trades["signal_date"])
    trades["ret"] = pd.to_numeric(trades["ret"], errors="coerce")
    trades["rank"] = trades.groupby("signal_date")["score"].rank(method="first", ascending=False)
    return trades


def _decile_stats(geo: pd.DataFrame, trades: pd.DataFrame, col: str) -> dict[str, Any]:
    pcol = f"{col}_p250" if f"{col}_p250" in geo.columns else col
    work = geo[[pcol]].copy()
    work["decile"] = pd.qcut(work[pcol].rank(method="first"), 10, labels=False, duplicates="drop")
    t = trades.merge(work.reset_index(), on="signal_date", how="left")
    out = {}
    for d, block in t.groupby("decile"):
        if pd.isna(d):
            continue
        top1 = block.loc[block["rank"].eq(1), "ret"]
        top2 = block.loc[block["rank"].eq(2), "ret"]
        out[str(int(d))] = {
            "n": int(len(block)),
            "top1_mean": float(top1.mean()) if len(top1) else None,
            "top2_mean": float(top2.mean()) if len(top2) else None,
            "win_rate": float((block["ret"] > 0).mean()),
            "large_loss_rate": float((block["ret"] <= -0.05).mean()),
            "right_tail_rate": float((block["ret"] >= 0.10).mean()),
            "mean_holding": float(pd.to_numeric(block.get("actual_exit_date"), errors="coerce").pipe(
                lambda s: (pd.to_datetime(block["actual_exit_date"]) - pd.to_datetime(block["entry_date"])).dt.days.mean()
            )) if "actual_exit_date" in block else None,
        }
    return out


def run_geometry(*, resume: bool = True) -> dict[str, Any]:
    ensure_round8_tree()
    geo = build_geometry()
    geo_out = geo.reset_index()
    _write("geometry_daily.parquet", geo_out)
    trades = _trade_rank_returns()
    features = [
        "top1_minus_median",
        "top2_minus_median",
        "top1_minus_top2",
        "top1_z_robust",
        "iqr",
        "candidate_count",
    ]
    deciles = {feat: _decile_stats(geo, trades, feat) for feat in features}
    # A vs B: high top1-median = strong top1 or weak others?
    p = geo["top1_minus_median_p250"]
    high = geo.loc[p.ge(0.80)]
    low = geo.loc[p.le(0.20)]
    interpretation = {
        "high_p80_mean_top1_score": float(high["top1_score"].mean()) if len(high) else None,
        "high_p80_mean_median": float(high["median"].mean()) if len(high) else None,
        "high_p80_mean_iqr": float(high["iqr"].mean()) if len(high) else None,
        "high_p80_mean_n": float(high["candidate_count"].mean()) if len(high) else None,
        "low_p20_mean_top1_score": float(low["top1_score"].mean()) if len(low) else None,
        "low_p20_mean_median": float(low["median"].mean()) if len(low) else None,
        "low_p20_mean_iqr": float(low["iqr"].mean()) if len(low) else None,
        "low_p20_mean_n": float(low["candidate_count"].mean()) if len(low) else None,
        "corr_sep_vs_top1": float(geo["top1_minus_median"].corr(geo["top1_score"])),
        "corr_sep_vs_median": float(geo["top1_minus_median"].corr(geo["median"])),
        "corr_sep_vs_iqr": float(geo["top1_minus_median"].corr(geo["iqr"])),
        "n_small_candidate_days": int(geo["small_candidate_day"].sum()),
        "n_days": int(len(geo)),
    }
    payload = {
        "parent": PARENT_CODE,
        "score_definition": "fused 90/10 after same-day percentile; geometry on fused raw values",
        "deciles": deciles,
        "high_sep_means": interpretation,
    }
    _write("geometry_diagnostics.json", payload)
    return payload


def run_top2_marginal() -> dict[str, Any]:
    """组合增量：有 Top2 的 v2 相对 Top1-only，不是单票均值。"""
    v2_nav = pd.read_parquet(ANCHOR_DIR / "nav.parquet")
    top1_path = round8_dir("R8_D_TOP1_ONLY") / "nav.parquet"
    if not top1_path.is_file():
        return {"status": "missing_top1_only"}
    t1 = pd.read_parquet(top1_path)

    def _daily(nav: pd.DataFrame) -> pd.Series:
        n = nav.copy()
        n["trade_date"] = pd.to_datetime(n["trade_date"])
        n = n.sort_values("trade_date")
        return n.set_index("trade_date")["total_asset"].astype(float).pct_change().dropna()

    a = _daily(v2_nav).rename("with_top2")
    b = _daily(t1).rename("top1_only")
    aligned = pd.concat([a, b], axis=1).dropna()
    aligned["incremental"] = aligned["with_top2"] - aligned["top1_only"]
    wealth_inc = float((1.0 + aligned["incremental"]).prod())
    years = max(len(aligned) / 252.0, 1e-9)
    # rolling drawdown contribution: path DD with vs without
    def _dd(r):
        nav = (1.0 + r).cumprod()
        return float((nav / nav.cummax() - 1.0).min())

    payload = {
        "definition": "incremental_pnl = portfolio_with_Top2 - Top1_only (daily return difference)",
        "n_days": int(len(aligned)),
        "mean_daily_incremental": float(aligned["incremental"].mean()),
        "std_daily_incremental": float(aligned["incremental"].std(ddof=1)) if len(aligned) > 1 else None,
        "ann_incremental": float(wealth_inc ** (1.0 / years) - 1.0),
        "maxdd_with_top2": _dd(aligned["with_top2"]),
        "maxdd_top1_only": _dd(aligned["top1_only"]),
        "incremental_dd_delta": _dd(aligned["with_top2"]) - _dd(aligned["top1_only"]),
        "pct_days_top2_helps": float((aligned["incremental"] > 0).mean()),
        "note": "正增量表示 Top2 sleeve 提高组合收益；负值表示 Top2 在组合层面有负边际效用",
    }
    _write("top2_marginal_utility.json", payload)
    return payload


def run_disagreement_diag() -> dict[str, Any]:
    geo = build_geometry()
    trades = _trade_rank_returns()
    t = trades.merge(geo.reset_index()[["signal_date", "head_state", "rank_corr"]], on="signal_date", how="left")
    states = {}
    for st, block in t.groupby("head_state"):
        top1 = block.loc[block["rank"].eq(1), "ret"]
        top2 = block.loc[block["rank"].eq(2), "ret"]
        states[str(st)] = {
            "n_trades": int(len(block)),
            "n_days": int(block["signal_date"].nunique()),
            "top1_mean": float(top1.mean()) if len(top1) else None,
            "top2_mean": float(top2.mean()) if len(top2) else None,
            "large_loss_rate": float((block["ret"] <= -0.05).mean()),
            "right_tail_rate": float((block["ret"] >= 0.10).mean()),
            "mean_rank_corr": float(block["rank_corr"].mean()) if block["rank_corr"].notna().any() else None,
        }
    payload = {"states": states, "day_counts": geo["head_state"].value_counts(dropna=False).to_dict()}
    _write("disagreement_diagnostics.json", payload)
    return payload


def _drawdown_episodes(nav: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
    n = nav.copy()
    n["trade_date"] = pd.to_datetime(n["trade_date"])
    n = n.sort_values("trade_date")
    asset = n.set_index("trade_date")["total_asset"].astype(float)
    peak = asset.cummax()
    dd = asset / peak - 1.0
    episodes = []
    in_dd = False
    start = None
    trough = None
    trough_val = 0.0
    for dt, val in dd.items():
        if val < -1e-12:
            if not in_dd:
                in_dd = True
                start = dt
                trough = dt
                trough_val = val
            elif val < trough_val:
                trough = dt
                trough_val = val
        elif in_dd:
            episodes.append({
                "start": str(pd.Timestamp(start).date()),
                "trough": str(pd.Timestamp(trough).date()),
                "end": str(pd.Timestamp(dt).date()),
                "depth": float(trough_val),
            })
            in_dd = False
    if in_dd:
        episodes.append({
            "start": str(pd.Timestamp(start).date()),
            "trough": str(pd.Timestamp(trough).date()),
            "end": str(pd.Timestamp(dd.index[-1]).date()),
            "depth": float(trough_val),
        })
    episodes = sorted(episodes, key=lambda x: x["depth"])[:top_n]
    return episodes


def run_left_tail(*, resume: bool = True) -> dict[str, Any]:
    ensure_round8_tree()
    nav = pd.read_parquet(ANCHOR_DIR / "nav.parquet")
    geo = build_geometry()
    trades = _trade_rank_returns()
    episodes = _drawdown_episodes(nav, 5)
    n = nav.copy()
    n["trade_date"] = pd.to_datetime(n["trade_date"])
    asset = n.set_index("trade_date")["total_asset"].astype(float)
    peak = asset.cummax()
    dd = asset / peak - 1.0
    in_dd = dd.lt(-0.08)
    geo2 = geo.copy()
    geo2["in_dd"] = in_dd.reindex(geo2.index).fillna(False)
    # map signal_date to next trading day-ish: use same calendar
    compare_cols = [
        "top1_minus_median", "top2_minus_median", "top1_minus_top2",
        "candidate_count", "rank_corr",
    ]
    cmp = {}
    for col in compare_cols:
        a = pd.to_numeric(geo2.loc[~geo2["in_dd"], col], errors="coerce")
        b = pd.to_numeric(geo2.loc[geo2["in_dd"], col], errors="coerce")
        cmp[col] = {
            "normal_mean": float(a.mean()) if a.notna().any() else None,
            "dd_mean": float(b.mean()) if b.notna().any() else None,
        }
    if "head_state" in geo2:
        cmp["head_state"] = {
            "normal": geo2.loc[~geo2["in_dd"], "head_state"].value_counts(normalize=True).to_dict(),
            "dd": geo2.loc[geo2["in_dd"], "head_state"].value_counts(normalize=True).to_dict(),
        }
    t = trades.copy()
    t["in_dd"] = t["signal_date"].map(geo2["in_dd"]).fillna(False)
    rank_pnl = {}
    for flag, name in ((False, "normal"), (True, "dd")):
        sub = t.loc[t["in_dd"].eq(flag)]
        rank_pnl[name] = {
            "top1_mean": float(sub.loc[sub["rank"].eq(1), "ret"].mean()) if (sub["rank"] == 1).any() else None,
            "top2_mean": float(sub.loc[sub["rank"].eq(2), "ret"].mean()) if (sub["rank"] == 2).any() else None,
            "top1_sum": float(sub.loc[sub["rank"].eq(1), "ret"].sum()) if (sub["rank"] == 1).any() else None,
            "top2_sum": float(sub.loc[sub["rank"].eq(2), "ret"].sum()) if (sub["rank"] == 2).any() else None,
        }
    # H03 audit: limit-down / ecology in DD
    h03 = {"decision": "SKIP_H03", "reason": "no stable targeted market state beyond B01a already in v2"}
    try:
        from ..round4.ecology import build_ecology_frame
        from ..round2.runner import regime_frame

        eco = build_ecology_frame()
        regime = regime_frame()
        # join on signal_date
        extra = {}
        if "up_ratio" in regime.columns:
            up = regime["up_ratio"]
            extra["up_ratio_normal"] = float(up.reindex(geo2.index)[~geo2["in_dd"]].mean())
            extra["up_ratio_dd"] = float(up.reindex(geo2.index)[geo2["in_dd"]].mean())
        h03["observed"] = extra
        # do not invent a 2022-fitted rule
        h03["decision"] = "SKIP_H03"
    except Exception as exc:
        h03["error"] = repr(exc)

    payload = {
        "episodes": episodes,
        "compare": cmp,
        "rank_pnl": rank_pnl,
        "h03": h03,
    }
    _write("left_tail_audit.json", payload)
    return payload


def dump_contract_facts() -> dict[str, Any]:
    geo = build_geometry()
    alpha = load_locked_alpha()
    date_mult, meta = exact_v2_date_mult(alpha)
    mkt = market_b01a(alpha)
    heads = eligible_heads()
    n_by = geo["candidate_count"].value_counts().sort_index()
    return {
        "market_layer": "B01a breadth_tier 45/50/60 → 0 / 0.50 / 0.75 / 1.00",
        "includes_B01a": True,
        "clean_risk": False,
        "used_locked_round7_risk": False,
        "top_k": 2,
        "rank_nav": [0.3, 0.1],
        "percentile": {
            "feature": "top1_minus_median on fused score",
            "fused_score": "same-day percentile then 0.90 ten + 0.10 five",
            "daily_stat": "raw (top1 - median) of fused scores, not re-percentiled within day",
            "pit": "rolling 250 rank(pct=True), min_periods=50",
            "rule": "p250 <= 0.20 → confidence 0.5 else 1.0; NaN warmup → 1.0",
        },
        "n_signal_days": int(len(geo)),
        "small_candidate_days": {
            "definition": "n<=3",
            "n_days": int((geo["candidate_count"] <= 3).sum()),
            "n1": int((geo["candidate_count"] == 1).sum()),
            "n2": int((geo["candidate_count"] == 2).sum()),
            "note": "n=1 ⇒ top1-median=0; empty day skipped; p250 NaN ⇒ multiplier 1.0",
        },
        "overlay_meta": meta,
        "market_skip_days": int((mkt <= 1e-6).sum()),
        "combine": "market × confidence (product), same multiplier on Top1 and Top2",
        "score_rows": int(len(alpha)),
        "eligible_head_rows": int(len(heads)),
        "candidate_count_hist": {str(k): int(v) for k, v in n_by.items()},
    }


__all__ = [
    "dump_contract_facts",
    "run_disagreement_diag",
    "run_geometry",
    "run_left_tail",
    "run_top2_marginal",
]
