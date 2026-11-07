"""20 张图。缺数据不造假图。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_5_ROOT, assert_project_result_path
from .common import daily_from_nav, load_json
from .contract import V00, V01, V03, V04, experiment_dir, parity_dir

FIG_DIR = ROUND16_5_ROOT / "figures"


def _setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(FIG_DIR / "dummy.png")
    return plt


def _missing(plt, name: str, msg: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True, fontsize=11)
    ax.set_title(name)
    path = FIG_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _save(fig, plt, name: str) -> str:
    path = FIG_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def make_figures() -> dict[str, Any]:
    plt = _setup()
    out: dict[str, str] = {}
    dmap = {}
    for code, path in (
        (V00, parity_dir(V00) / "nav.parquet"),
        (V01, parity_dir(V01) / "nav.parquet"),
        (V03, experiment_dir(V03) / "nav.parquet"),
        (V04, experiment_dir(V04) / "nav.parquet"),
    ):
        if path.is_file():
            dmap[code] = daily_from_nav(pd.read_parquet(path))

    def wealth(s):
        return (1.0 + s).cumprod()

    if V00 in dmap and V01 in dmap:
        fig, ax = plt.subplots(figsize=(9, 4))
        for c, s in dmap.items():
            ax.plot(s.index, wealth(s), label=c, lw=1.2)
        ax.legend(fontsize=8)
        ax.set_title("NAV wealth V00/V01/V03/V04")
        ax.grid(True, alpha=0.3)
        out["fig01_nav"] = _save(fig, plt, "fig01_nav.png")
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(dmap[V01].index, (dmap[V01] - dmap[V00].reindex(dmap[V01].index)).fillna(0), lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title("Daily return difference V01-V00")
        out["fig02_daily_diff"] = _save(fig, plt, "fig02_daily_diff.png")
    else:
        out["fig01_nav"] = _missing(plt, "fig01_nav.png", "NAV missing")
        out["fig02_daily_diff"] = _missing(plt, "fig02_daily_diff.png", "NAV missing")

    attr = load_json(ROUND16_5_ROOT / "attribution" / "attribution.json")
    recon = attr.get("reconciliation") or {}
    if recon:
        labels = ["signal_all", "in_window", "eligible", "changed", "avoided", "no_effect"]
        vals = [
            recon.get("n_signal_all_including_2015") or 0,
            recon.get("n_signal_in_backtest_window") or 0,
            recon.get("n_eligible_top2_nonzero_budget_observable") or 0,
            recon.get("n_changed_fills") or 0,
            recon.get("n_avoided_trades") or 0,
            recon.get("n_no_effect_days") or 0,
        ]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, vals)
        ax.set_title("Event count waterfall 332 vs 298")
        ax.tick_params(axis="x", rotation=20)
        out["fig03_event_waterfall"] = _save(fig, plt, "fig03_event_waterfall.png")
    else:
        out["fig03_event_waterfall"] = _missing(plt, "fig03_event_waterfall.png", "attribution missing")

    yearly = attr.get("yearly") or []
    if yearly:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        xs = [r["year"] for r in yearly]
        ax.bar([x - 0.15 for x in xs], [r.get("n_signal_in_window") or 0 for r in yearly], width=0.3, label="signal")
        ax.bar([x + 0.15 for x in xs], [r.get("n_avoided") or 0 for r in yearly], width=0.3, label="avoided")
        ax.legend()
        ax.set_title("Yearly veto / avoided 2018-2026YTD")
        out["fig04_yearly_veto"] = _save(fig, plt, "fig04_yearly_veto.png")
    else:
        out["fig04_yearly_veto"] = _missing(plt, "fig04_yearly_veto.png", "yearly missing")

    csv = ROUND16_5_ROOT / "attribution" / "O01_treated_trades.csv"
    if csv.is_file():
        df = pd.read_csv(csv)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sub = df.loc[df.get("avoided").astype(bool) if "avoided" in df.columns else df.index]
        ax.hist(pd.to_numeric(sub.get("v4_net"), errors="coerce").dropna(), bins=30)
        ax.set_title("Avoided Top2 trade PnL (v4 net)")
        out["fig05_avoided_pnl_hist"] = _save(fig, plt, "fig05_avoided_pnl_hist.png")
    else:
        out["fig05_avoided_pnl_hist"] = _missing(plt, "fig05_avoided_pnl_hist.png", "treated trades CSV missing")

    conc = attr.get("concentration") or {}
    if conc:
        ks = [1, 3, 5, 10]
        shares = [((conc.get(str(k)) or {}).get("share_of_total_diff")) or 0 for k in ks]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar([str(k) for k in ks], shares)
        ax.set_title("Share of NAV daily-diff from top K days")
        out["fig06_concentration"] = _save(fig, plt, "fig06_concentration.png")
    else:
        out["fig06_concentration"] = _missing(plt, "fig06_concentration.png", "concentration missing")

    leave = attr.get("leave_one_event_top10_daily_diff") or {}
    if leave:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        keys = list(leave.keys())[:10]
        incs = [(leave[k] or {}).get("cagr_inc_adj") or 0 for k in keys]
        ax.barh(keys, incs)
        ax.axvline(attr.get("cagr_increment") or 0, color="r", ls="--", label="full increment")
        ax.set_title("Leave-one-event CAGR increment (daily-diff diagnostic)")
        out["fig07_leave_one_event"] = _save(fig, plt, "fig07_leave_one_event.png")
    else:
        out["fig07_leave_one_event"] = _missing(plt, "fig07_leave_one_event.png", "leave-one-event missing")

    ckpt = ROUND16_5_ROOT / "placebo" / "placebo_runs.jsonl"
    sharpes = []
    cagrs = []
    if ckpt.is_file():
        import json

        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("sharpe") is not None:
                sharpes.append(float(rec["sharpe"]))
            if rec.get("cagr") is not None:
                cagrs.append(float(rec["cagr"]))
    placebo = load_json(ROUND16_5_ROOT / "placebo" / "placebo.json")
    o01_sh = placebo.get("o01_sharpe")
    o01_cagr = placebo.get("o01_cagr")
    if sharpes:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(sharpes, bins=40, alpha=0.85)
        if o01_sh is not None:
            ax.axvline(float(o01_sh), color="r", label="O01/V01")
        ax.legend()
        ax.set_title(f"V02 full-engine Sharpe N={len(sharpes)}")
        out["fig08_v02_sharpe_hist"] = _save(fig, plt, "fig08_v02_sharpe_hist.png")
    else:
        out["fig08_v02_sharpe_hist"] = _missing(plt, "fig08_v02_sharpe_hist.png", "placebo runs not finished")
    if cagrs:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(cagrs, bins=40, alpha=0.85)
        if o01_cagr is not None:
            ax.axvline(float(o01_cagr), color="r", label="O01/V01")
        ax.legend()
        ax.set_title(f"V02 full-engine CAGR N={len(cagrs)}")
        out["fig09_v02_cagr_hist"] = _save(fig, plt, "fig09_v02_cagr_hist.png")
    else:
        out["fig09_v02_cagr_hist"] = _missing(plt, "fig09_v02_cagr_hist.png", "placebo runs not finished")

    eq = load_json(ROUND16_5_ROOT / "equity" / "equity.json")
    years = list((eq.get("by_year") or {}).keys())
    if years:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ys = [int(y) for y in years]
        ax.bar([y - 0.15 for y in ys], [eq["by_year"][str(y)]["n_o01"] if str(y) in eq["by_year"] else eq["by_year"][y]["n_o01"] for y in years], width=0.3, label="O01")
        ax.bar([y + 0.15 for y in ys], [eq["by_year"][str(y)]["n_v03"] if str(y) in eq["by_year"] else eq["by_year"][y]["n_v03"] for y in years], width=0.3, label="V03")
        ax.legend()
        ax.set_title(f"V03 vs O01 activity Jaccard={eq.get('jaccard')}")
        out["fig10_v03_overlap"] = _save(fig, plt, "fig10_v03_overlap.png")
    else:
        out["fig10_v03_overlap"] = _missing(plt, "fig10_v03_overlap.png", "V03 not run")

    rp = ROUND16_5_ROOT / "orthogonal" / "residual_put_skew.parquet"
    if rp.is_file():
        r = pd.read_parquet(rp)
        r["trade_date"] = pd.to_datetime(r["trade_date"])
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(r["trade_date"], r["residual_put_skew"], lw=0.7)
        ax.set_title("V04 residual Put Skew")
        out["fig11_v04_residual"] = _save(fig, plt, "fig11_v04_residual.png")
    else:
        out["fig11_v04_residual"] = _missing(plt, "fig11_v04_residual.png", "V04 residual missing")

    if dmap:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        labels = list(dmap)
        sh = [float(dmap[c].mean() / dmap[c].std(ddof=1) * np.sqrt(252)) if dmap[c].std(ddof=1) > 0 else 0 for c in labels]
        ax.bar(labels, sh)
        ax.set_title("Same-window Sharpe")
        ax.tick_params(axis="x", rotation=20)
        out["fig12_same_window_sharpe"] = _save(fig, plt, "fig12_same_window_sharpe.png")
    else:
        out["fig12_same_window_sharpe"] = _missing(plt, "fig12_same_window_sharpe.png", "dailies missing")

    stress = load_json(ROUND16_5_ROOT / "stress" / "stress.json")
    res = stress.get("results") or {}
    if res:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        codes = [c for c in (V00, V01, V03, V04) if c in res and not res[c].get("skip")]
        x = np.arange(len(codes))
        c20 = [((res[c] or {}).get("plus20") or {}).get("cagr") or 0 for c in codes]
        c30 = [((res[c] or {}).get("plus30") or {}).get("cagr") or 0 for c in codes]
        ax.bar(x - 0.15, c20, width=0.3, label="+20bp")
        ax.bar(x + 0.15, c30, width=0.3, label="+30bp")
        ax.set_xticks(x)
        ax.set_xticklabels(codes, rotation=20)
        ax.legend()
        ax.set_title("Cost stress CAGR")
        out["fig13_cost_stress"] = _save(fig, plt, "fig13_cost_stress.png")
    else:
        out["fig13_cost_stress"] = _missing(plt, "fig13_cost_stress.png", "stress missing")

    boot = load_json(ROUND16_5_ROOT / "bootstrap" / "bootstrap.json")
    if boot.get("mean_sharpe"):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ks = list(boot["mean_sharpe"])
        ax.bar(ks, [boot["mean_sharpe"][k] for k in ks], yerr=[np.nan] * len(ks))
        ax.set_title("Bootstrap mean Sharpe")
        ax.tick_params(axis="x", rotation=20)
        out["fig14_bootstrap"] = _save(fig, plt, "fig14_bootstrap.png")
    else:
        out["fig14_bootstrap"] = _missing(plt, "fig14_bootstrap.png", "bootstrap missing")

    cpcv = load_json(ROUND16_5_ROOT / "cpcv" / "cpcv.json")
    by = cpcv.get("by_id") or {}
    if by:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ks = list(by)
        med = [(by[k] or {}).get("median") or (by[k] or {}).get("oos_sharpe_median") or 0 for k in ks]
        ax.bar(ks, med)
        ax.set_title("CPCV median Sharpe (NOT forward OOS)")
        ax.tick_params(axis="x", rotation=20)
        out["fig15_cpcv"] = _save(fig, plt, "fig15_cpcv.png")
    else:
        out["fig15_cpcv"] = _missing(plt, "fig15_cpcv.png", "cpcv missing")

    if V00 in dmap:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        for c, s in dmap.items():
            yr = s.groupby(s.index.year).apply(lambda x: float((1 + x).prod() - 1))
            ax.plot(yr.index.astype(int), yr.values, marker="o", label=c)
        ax.legend(fontsize=8)
        ax.set_title("Yearly returns")
        ax.grid(True, alpha=0.3)
        out["fig16_yearly_returns"] = _save(fig, plt, "fig16_yearly_returns.png")
        fig, ax = plt.subplots(figsize=(9, 3.5))
        for c, s in dmap.items():
            nav = wealth(s)
            dd = nav / nav.cummax() - 1.0
            ax.plot(dd.index, dd.values, label=c, lw=1)
        ax.legend(fontsize=8)
        ax.set_title("Drawdown")
        out["fig17_drawdown"] = _save(fig, plt, "fig17_drawdown.png")
    else:
        out["fig16_yearly_returns"] = _missing(plt, "fig16_yearly_returns.png", "NAV missing")
        out["fig17_drawdown"] = _missing(plt, "fig17_drawdown.png", "NAV missing")

    pred = load_json(ROUND16_5_ROOT / "prediction" / "prediction.json")
    cal = (((pred.get("results") or {}).get("top2_loss_net_lt_0") or {}).get("equity_plus_skew") or {}).get("calibration") or []
    if cal:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.scatter([c["mean_pred"] for c in cal], [c["mean_y"] for c in cal])
        ax.set_title("Top2 loss calibration equity+skew")
        ax.set_xlabel("mean pred")
        ax.set_ylabel("mean y")
        out["fig18_calibration"] = _save(fig, plt, "fig18_calibration.png")
    else:
        out["fig18_calibration"] = _missing(plt, "fig18_calibration.png", "prediction calibration missing")

    quote = load_json(ROUND16_5_ROOT / "quote" / "quote.json")
    if quote.get("bid_ask") == "BID_ASK_NOT_AVAILABLE" or quote.get("verdict") in {"LAST_PRICE_PROXY_ONLY", "QUOTE_QUALITY_FAIL", None}:
        out["fig19_quote"] = _missing(
            plt,
            "fig19_quote.png",
            f"BID_ASK_NOT_AVAILABLE / verdict={quote.get('verdict')}. IV=last_close_only. No fabricated bid/ask figure.",
        )
    else:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(["bilateral"], [((quote.get("chain") or {}).get("bilateral_rate") or 0)])
        ax.set_title("Quote bilateral rate")
        out["fig19_quote"] = _save(fig, plt, "fig19_quote.png")

    out["fig20_forward"] = _missing(
        plt,
        "fig20_forward.png",
        "FORWARD_OOS_NOT_AVAILABLE after 2026-07-24. CPCV is not forward. PRIOR_SELECTION_EXISTS=TRUE.",
    )
    return {"n": len(out), "paths": out}
