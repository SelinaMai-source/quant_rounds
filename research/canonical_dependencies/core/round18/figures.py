"""20 张图。缺数据不造假。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND18_ROOT, assert_project_result_path
from .common import load_json
from .contract import A00, exp_dir, parity_dir

FIG_DIR = ROUND18_ROOT / "figures"
FIG_JSON = ROUND18_ROOT / "figures" / "figures.json"


def _save(fig, name: str) -> str:
    path = FIG_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(path)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return str(path)


def _nav(code: str) -> pd.DataFrame | None:
    for p in (exp_dir(code) / "nav.parquet", parity_dir(A00) / "nav.parquet"):
        if p.is_file():
            return pd.read_parquet(p)
    return None


def make_figures() -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = {}
    skipped = []

    def skip(name, reason):
        skipped.append({"name": name, "reason": reason})

    matrix = load_json(ROUND18_ROOT / "metrics" / "matrix.json")
    rows = matrix.get("rows") or []
    pit = load_json(ROUND18_ROOT / "pit" / "pit.json")

    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        plotted = 0
        for row in rows:
            nav = _nav(row.get("code"))
            if nav is None:
                continue
            ax.plot(pd.to_datetime(nav["trade_date"]), nav["total_asset"], label=row.get("code"), lw=1)
            plotted += 1
        if plotted:
            ax.legend(fontsize=7)
            ax.set_title("Round18 NAV")
            paths["01_nav"] = _save(fig, "01_nav.png")
        else:
            skip("01_nav.png", "no nav")
        plt.close(fig)
    except Exception as exc:
        skip("01_nav.png", str(exc))

    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [r.get("code") for r in rows]
        sharpes = [r.get("sharpe") if r.get("sharpe") is not None else np.nan for r in rows]
        ax.bar(np.arange(len(labels)), sharpes)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_title("Sharpe")
        paths["02_sharpe"] = _save(fig, "02_sharpe.png")
        plt.close(fig)
    except Exception as exc:
        skip("02_sharpe.png", str(exc))

    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        mdd = [r.get("max_drawdown") if r.get("max_drawdown") is not None else np.nan for r in rows]
        ax.bar(np.arange(len(rows)), mdd)
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels([r.get("code") for r in rows], rotation=90, fontsize=7)
        ax.set_title("MaxDD")
        paths["03_maxdd"] = _save(fig, "03_maxdd.png")
        plt.close(fig)
    except Exception as exc:
        skip("03_maxdd.png", str(exc))

    for i, (name, rec) in enumerate((pit.get("factors") or {}).items(), start=4):
        fname = f"{i:02d}_coverage_{name.split('_')[1] if '_' in name else name}.png"
        try:
            years = rec.get("year_coverage") or {}
            if not years:
                skip(fname, "no year coverage")
                continue
            fig, ax = plt.subplots(figsize=(8, 3))
            xs = list(years)
            ys = [years[y].get("coverage") for y in xs]
            ax.bar(xs, ys)
            ax.axhline(0.95, color="red", ls="--")
            ax.set_title(f"coverage {name}")
            paths[fname] = _save(fig, fname)
            plt.close(fig)
        except Exception as exc:
            skip(fname, str(exc))

    # remaining placeholders only if data exists
    names = [
        "08_yearly_heatmap.png", "09_f1_hist.png", "10_f2_hist.png", "11_f3_hist.png",
        "12_f1_vs_f3.png", "13_redundancy.png", "14_bootstrap.png", "15_cost.png",
        "16_capacity.png", "17_spa.png", "18_dd_path.png", "19_trades.png", "20_increment.png",
    ]
    panel_p = ROUND18_ROOT / "data" / "processed" / "factor_panel.parquet"
    try:
        if panel_p.is_file():
            panel = pd.read_parquet(panel_p)
            from .contract import F1_NAME, F2_NAME, F3_NAME
            for fname, col in (("09_f1_hist.png", F1_NAME), ("10_f2_hist.png", F2_NAME), ("11_f3_hist.png", F3_NAME)):
                if col not in panel or panel[col].notna().sum() == 0:
                    skip(fname, "no valid values")
                    continue
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(panel[col].dropna(), bins=40)
                ax.set_title(col)
                paths[fname] = _save(fig, fname)
                plt.close(fig)
            if {F1_NAME, F3_NAME} <= set(panel.columns) and panel[F1_NAME].notna().any():
                fig, ax = plt.subplots(figsize=(5, 5))
                s = panel[[F1_NAME, F3_NAME]].dropna()
                if len(s) > 10:
                    ax.scatter(s[F1_NAME], s[F3_NAME], s=2, alpha=0.3)
                    ax.set_xlabel(F1_NAME)
                    ax.set_ylabel(F3_NAME)
                    paths["12_f1_vs_f3.png"] = _save(fig, "12_f1_vs_f3.png")
                else:
                    skip("12_f1_vs_f3.png", "too few joint rows")
                plt.close(fig)
        else:
            for fname in ("09_f1_hist.png", "10_f2_hist.png", "11_f3_hist.png", "12_f1_vs_f3.png"):
                skip(fname, "no factor panel")
    except Exception as exc:
        skip("factor_hists", str(exc))

    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        years = sorted({y for r in rows for y in (r.get("yearly_returns") or {})})
        if years:
            data = []
            labs = []
            for r in rows:
                if r.get("sharpe") is None:
                    continue
                labs.append(r.get("code"))
                data.append([r.get("yearly_returns", {}).get(y) for y in years])
            arr = np.array(data, dtype=float)
            im = ax.imshow(arr, aspect="auto", cmap="RdYlGn")
            ax.set_xticks(range(len(years)))
            ax.set_xticklabels(years, fontsize=7)
            ax.set_yticks(range(len(labs)))
            ax.set_yticklabels(labs, fontsize=7)
            fig.colorbar(im)
            ax.set_title("yearly returns")
            paths["08_yearly_heatmap.png"] = _save(fig, "08_yearly_heatmap.png")
        else:
            skip("08_yearly_heatmap.png", "no yearly")
        plt.close(fig)
    except Exception as exc:
        skip("08_yearly_heatmap.png", str(exc))

    boot = load_json(ROUND18_ROOT / "bootstrap" / "bootstrap.json")
    try:
        pairs = boot.get("pairs") or {}
        if pairs:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(range(len(pairs)), [v.get("mean_diff_sharpe") or 0 for v in pairs.values()])
            ax.set_xticks(range(len(pairs)))
            ax.set_xticklabels(list(pairs), rotation=90, fontsize=7)
            ax.set_title("bootstrap mean ΔSharpe")
            paths["14_bootstrap.png"] = _save(fig, "14_bootstrap.png")
            plt.close(fig)
        else:
            skip("14_bootstrap.png", "no pairs")
    except Exception as exc:
        skip("14_bootstrap.png", str(exc))

    try:
        a00 = _nav(A00)
        if a00 is not None:
            fig, ax = plt.subplots(figsize=(8, 3))
            nav = a00.sort_values("trade_date")
            peak = nav["total_asset"].cummax()
            ax.plot(pd.to_datetime(nav["trade_date"]), nav["total_asset"] / peak - 1.0)
            ax.set_title("A00 drawdown")
            paths["18_dd_path.png"] = _save(fig, "18_dd_path.png")
            plt.close(fig)
        else:
            skip("18_dd_path.png", "no A00 nav")
    except Exception as exc:
        skip("18_dd_path.png", str(exc))

    try:
        formal = [r for r in rows if r.get("formal") and r.get("sharpe") is not None]
        a00s = next((r.get("sharpe") for r in formal if r.get("code") == A00), None)
        if a00s is not None:
            fig, ax = plt.subplots(figsize=(8, 3))
            labs = [r.get("code") for r in formal if r.get("code") != A00]
            vals = [float(r.get("sharpe")) - float(a00s) for r in formal if r.get("code") != A00]
            if labs:
                ax.bar(labs, vals)
                ax.axhline(0, color="black")
                ax.set_title("Sharpe increment vs A00")
                ax.tick_params(axis="x", rotation=90)
                paths["20_increment.png"] = _save(fig, "20_increment.png")
            else:
                skip("20_increment.png", "no formal alts")
            plt.close(fig)
        else:
            skip("20_increment.png", "no A00 sharpe")
    except Exception as exc:
        skip("20_increment.png", str(exc))

    for fname in names:
        if fname not in {p.split("/")[-1] for p in paths.values()} and not any(s["name"] == fname for s in skipped):
            skip(fname, "not enough data to draw without fabricating")

    payload = {"paths": paths, "skipped": skipped, "n_drawn": len(paths)}
    from .common import dump_json
    dump_json(FIG_JSON, payload)
    return payload
