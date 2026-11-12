"""Round 3 图：净值 / 回撤 / 分年 / 滚动12M / 散点 / stress / 邻域。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..metrics import clip_nav_for_metrics
from ..paths import ROUND3_ROOT, assert_project_result_path, ensure_round3_tree


def _save(fig, name: str) -> Path:
    dest = ensure_round3_tree() / "figures" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest)
    fig.savefig(dest, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return dest


def _nav(code: str) -> pd.DataFrame | None:
    path = ROUND3_ROOT / code / "nav.parquet"
    if not path.is_file():
        return None
    return clip_nav_for_metrics(pd.read_parquet(path))


def plot_all(codes: list[str]) -> list[str]:
    written = []
    written.append(str(plot_nav(codes)))
    written.append(str(plot_drawdown(codes)))
    written.append(str(plot_yearly(codes)))
    written.append(str(plot_rolling12(codes)))
    written.append(str(plot_scatter(codes)))
    written += [str(p) for p in plot_heatmaps()]
    return [p for p in written if p]


def plot_nav(codes: list[str]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    for code in codes:
        nav = _nav(code)
        if nav is None or nav.empty:
            continue
        ax.plot(pd.to_datetime(nav["trade_date"]), nav["total_asset"] / 300_000.0, label=code, linewidth=1.1)
    ax.set_title("NAV (eval window, start=1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, "nav.png")


def plot_drawdown(codes: list[str]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    for code in codes:
        nav = _nav(code)
        if nav is None or nav.empty:
            continue
        asset = nav["total_asset"].astype(float)
        dd = asset / asset.cummax() - 1.0
        ax.plot(pd.to_datetime(nav["trade_date"]), dd, label=code, linewidth=1.1)
    ax.set_title("Drawdown")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, "drawdown.png")


def plot_yearly(codes: list[str]) -> Path:
    rows = []
    for code in codes:
        report = ROUND3_ROOT / code / "report.json"
        if not report.is_file():
            continue
        import json

        yearly = json.loads(report.read_text(encoding="utf-8")).get("metrics", {}).get("yearly_returns") or {}
        for year, val in yearly.items():
            if year == "2017":
                continue
            rows.append({"code": code, "year": year, "ret": float(val)})
    fig, ax = plt.subplots(figsize=(11, 5))
    if rows:
        frame = pd.DataFrame(rows)
        pivot = frame.pivot(index="year", columns="code", values="ret")
        pivot.plot(kind="bar", ax=ax)
    ax.set_title("Calendar-year return (2026=YTD, not a full year)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, "yearly.png")


def plot_rolling12(codes: list[str]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    for code in codes:
        nav = _nav(code)
        if nav is None or len(nav) < 260:
            continue
        asset = nav["total_asset"].astype(float)
        rolled = asset / asset.shift(252) - 1.0
        ax.plot(pd.to_datetime(nav["trade_date"]), rolled, label=code, linewidth=1.1)
    ax.set_title("Rolling 12M return")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, "rolling12m.png")


def plot_scatter(codes: list[str]) -> Path:
    import json

    xs, ys, labels = [], [], []
    for code in codes:
        path = ROUND3_ROOT / code / "report.json"
        if not path.is_file():
            continue
        m = json.loads(path.read_text(encoding="utf-8")).get("metrics") or {}
        if m.get("max_drawdown") is None or m.get("annualized_return") is None:
            continue
        xs.append(float(m["max_drawdown"]))
        ys.append(float(m["annualized_return"]))
        labels.append(code)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("MaxDD")
    ax.set_ylabel("CAGR")
    ax.set_title("CAGR–MaxDD")
    ax.grid(True, alpha=0.3)
    return _save(fig, "cagr_mdd.png")


def plot_heatmaps() -> list[Path]:
    import json

    out = []
    stress = ROUND3_ROOT / "12_stress" / "stress.json"
    if stress.is_file():
        payload = json.loads(stress.read_text(encoding="utf-8"))
        keys = ["normal", "cost_slip20", "cost_fee20", "open_buy30", "size_nav20", "drop2024"]
        rows = []
        idx = []
        for item in payload.get("rows") or []:
            idx.append(item["code"])
            rows.append([
                ((item.get(k) or {}).get("cagr") if isinstance(item.get(k), dict) else None)
                for k in keys
            ])
        if rows:
            fig, ax = plt.subplots(figsize=(9, 4))
            data = pd.DataFrame(rows, index=idx, columns=keys)
            im = ax.imshow(data.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn")
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(keys, rotation=30, ha="right")
            ax.set_yticks(range(len(idx)))
            ax.set_yticklabels(idx)
            fig.colorbar(im, ax=ax, label="CAGR")
            ax.set_title("Stress CAGR (worst scenario)")
            out.append(_save(fig, "stress_heat.png"))
    neighbor = ROUND3_ROOT / "06_neighbor" / "neighbor.json"
    if neighbor.is_file():
        payload = json.loads(neighbor.read_text(encoding="utf-8"))
        labels, cagrs = [], []
        for name, block in payload.items():
            center = block.get("center") or {}
            if center.get("cagr") is not None:
                labels.append(f"{name}/center")
                cagrs.append(float(center["cagr"]))
            for nb in block.get("neighbors") or []:
                if nb.get("cagr") is not None:
                    labels.append(f"{name}/{nb['code']}")
                    cagrs.append(float(nb["cagr"]))
        if cagrs:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.imshow([cagrs], aspect="auto", cmap="RdYlGn")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
            ax.set_yticks([])
            ax.set_title("Neighborhood CAGR")
            out.append(_save(fig, "neighbor_heat.png"))
    return out
