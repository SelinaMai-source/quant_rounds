"""Round10 必要图。缺数据就跳过，不造假。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import ROUND10_ROOT, assert_project_result_path, ensure_round10_tree, round10_dir
from .contract import R10_00
from .daily import load_daily


def _save(fig, name: str) -> None:
    path = ROUND10_ROOT / "figures" / name
    assert_project_result_path(path)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    fig.clf()


def write_figures() -> dict[str, str]:
    ensure_round10_tree()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = {}
    nav_path = round10_dir(R10_00) / "nav.parquet"
    if nav_path.is_file():
        nav = pd.read_parquet(nav_path)
        nav["trade_date"] = pd.to_datetime(nav["trade_date"])
        nav = nav.sort_values("trade_date")
        asset = nav.set_index("trade_date")["total_asset"].astype(float)
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(asset.index, asset.values)
        ax[0].set_title("baseline_v4 equity")
        dd = asset / asset.cummax() - 1.0
        ax[1].plot(dd.index, dd.values, color="crimson")
        ax[1].set_title("drawdown")
        _save(fig, "01_v4_equity_dd.png")
        written["equity"] = "01_v4_equity_dd.png"
        plt.close(fig)
    try:
        daily = load_daily()
        cells = daily.groupby(["agreement", "sep_bucket"])["top2_ret"].mean().unstack()
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(cells.reindex(index=["A", "B", "C"], columns=["LOW", "MID", "HIGH"]).to_numpy(), cmap="RdYlGn")
        ax.set_xticks(range(3), ["LOW", "MID", "HIGH"])
        ax.set_yticks(range(3), ["A", "B", "C"])
        ax.set_title("Top2 mean ret by cell")
        fig.colorbar(im, ax=ax)
        _save(fig, "02_cell_heatmap.png")
        written["cell"] = "02_cell_heatmap.png"
        plt.close(fig)
        if "pair_corr_20" in daily.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            x = pd.to_numeric(daily["pair_corr_20"], errors="coerce")
            y = pd.to_numeric(daily["top2_ret"], errors="coerce")
            ax.scatter(x, y, s=8, alpha=0.4)
            ax.set_title("pair corr20 vs Top2 ret")
            _save(fig, "04_pair_corr.png")
            written["corr"] = "04_pair_corr.png"
            plt.close(fig)
        if "same_industry_L1" in daily.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            daily.boxplot(column="top2_ret", by="same_industry_L1", ax=ax)
            ax.set_title("same L1 vs Top2 ret")
            plt.suptitle("")
            _save(fig, "05_same_industry.png")
            written["industry"] = "05_same_industry.png"
            plt.close(fig)
    except Exception:
        pass
    boot = ROUND10_ROOT / "bootstrap" / "summary.json"
    if boot.is_file():
        import json
        payload = json.loads(boot.read_text(encoding="utf-8"))
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (code, row) in enumerate((payload.get("bootstrap") or {}).items()):
            ax.bar(i, float(row.get("p_delta_ret_gt0") or 0))
        ax.set_xticks(range(len(payload.get("bootstrap") or {})), list((payload.get("bootstrap") or {}).keys()), rotation=45, ha="right")
        ax.set_title("P(ΔCAGR>0) vs v4")
        ax.axhline(0.70, color="gray", ls="--")
        _save(fig, "16_bootstrap_p.png")
        written["boot"] = "16_bootstrap_p.png"
        plt.close(fig)
    return written


__all__ = ["write_figures"]
