"""Round 4 图：净值、回撤、分年、滚动12M、Alpha vs Beta、分差校准、2D 热力、生态分位。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..metrics import clip_nav_for_metrics
from ..paths import ROUND4_ROOT, assert_project_result_path, ensure_round4_tree


def _save(fig, name: str) -> Path:
    dest = ensure_round4_tree() / "figures" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(dest)
    fig.savefig(dest, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return dest


def _nav(code: str) -> pd.DataFrame | None:
    path = ROUND4_ROOT / code / "nav.parquet"
    if not path.is_file():
        return None
    return clip_nav_for_metrics(pd.read_parquet(path))


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
        ax.plot(pd.to_datetime(nav["trade_date"]), asset / asset.cummax() - 1.0, label=code, linewidth=1.1)
    ax.set_title("Drawdown")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, "drawdown.png")


def plot_yearly(codes: list[str]) -> Path:
    rows = []
    for code in codes:
        report = ROUND4_ROOT / code / "report.json"
        if not report.is_file():
            continue
        m = json.loads(report.read_text(encoding="utf-8")).get("metrics") or {}
        yearly = m.get("yearly_returns") or {}
        for y, v in yearly.items():
            rows.append({"code": code, "year": y, "ret": v})
    fig, ax = plt.subplots(figsize=(11, 5))
    if rows:
        frame = pd.DataFrame(rows)
        years = sorted(frame["year"].unique())
        x = np.arange(len(years))
        width = 0.8 / max(len(codes), 1)
        for i, code in enumerate(codes):
            sl = frame.loc[frame["code"] == code].drop_duplicates("year").set_index("year")["ret"]
            vals = [float(sl.loc[y]) if y in sl.index else 0.0 for y in years]
            ax.bar(x + i * width, vals, width=width, label=code)
        ax.set_xticks(x)
        ax.set_xticklabels(years)
    ax.set_title("Yearly return (2026=YTD)")
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, "yearly.png")


def plot_rolling12(codes: list[str]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    for code in codes:
        nav = _nav(code)
        if nav is None or nav.empty:
            continue
        ret = nav.set_index(pd.to_datetime(nav["trade_date"]))["total_asset"].pct_change()
        roll = (1.0 + ret).rolling(252).apply(lambda x: float(np.prod(x) - 1.0), raw=True)
        ax.plot(roll.index, roll, label=code, linewidth=1.0)
    ax.set_title("Rolling 12M")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, "rolling12.png")


def plot_alpha_beta() -> Path | None:
    path = ROUND4_ROOT / "01_attr" / "attribution.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    ev = payload.get("event_beta_vs_alpha") or {}
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["v2 2024", "Event Beta", "等权指数", "Selection Alpha"]
    vals = [
        ev.get("v2_2024") or 0,
        ev.get("event_beta") or 0,
        ev.get("equal_weight_index_2024") or 0,
        ev.get("selection_alpha") or 0,
    ]
    ax.bar(labels, vals)
    ax.set_title("2024 Event Beta vs Selection Alpha")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, "alpha_beta.png")


def plot_margin_cal() -> Path | None:
    from .overlays import agreement_frame
    from ..stock_readonly import load_fs00263_ensemble_predictions

    trades_path = ROUND4_ROOT / "R4_00_baseline_v2" / "trades.parquet"
    if not trades_path.is_file():
        return None
    trades = pd.read_parquet(trades_path)
    trades["signal_date"] = pd.to_datetime(trades["signal_date"])
    agree = agreement_frame(load_fs00263_ensemble_predictions())
    merged = trades.merge(agree.reset_index().rename(columns={"index": "signal_date"}), on="signal_date", how="left")
    p = pd.to_numeric(merged["margin_p250"], errors="coerce")
    ret = pd.to_numeric(merged["ret"], errors="coerce")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(p, ret, s=8, alpha=0.35)
    ax.set_xlabel("margin p250")
    ax.set_ylabel("trade ret")
    ax.set_title("Margin calibration (v2 trades)")
    ax.grid(True, alpha=0.3)
    return _save(fig, "margin_cal.png")


def plot_2d_heat() -> Path | None:
    from .overlays import _agree_bucket, _margin_bucket, agreement_frame
    from ..stock_readonly import load_fs00263_ensemble_predictions

    trades_path = ROUND4_ROOT / "R4_00_baseline_v2" / "trades.parquet"
    if not trades_path.is_file():
        return None
    trades = pd.read_parquet(trades_path)
    trades["signal_date"] = pd.to_datetime(trades["signal_date"])
    agree = agreement_frame(load_fs00263_ensemble_predictions())
    merged = trades.merge(agree.reset_index().rename(columns={"index": "signal_date"}), on="signal_date", how="left")
    merged["a"] = _agree_bucket(merged["agree_n"])
    merged["m"] = _margin_bucket(merged["margin_p250"])
    merged["ret"] = pd.to_numeric(merged["ret"], errors="coerce")
    pivot = merged.pivot_table(index="a", columns="m", values="ret", aggfunc="mean")
    pivot = pivot.reindex(index=["le3", "eq4", "eq5"], columns=["low", "mid", "high"])
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(3), ["low", "mid", "high"])
    ax.set_yticks(range(3), ["≤3", "4", "5"])
    ax.set_title("2D mean trade ret")
    fig.colorbar(im, ax=ax)
    for i in range(3):
        for j in range(3):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
    return _save(fig, "twodim_heat.png")


def plot_eco_pct() -> Path | None:
    from .ecology import build_ecology_frame

    eco = build_ecology_frame()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    series = [
        ("continue_rate", "昨日首板今日连板率"),
        ("oc_median_ewma20", "开收中位数 EWMA20"),
        ("big_loss_rate", "大亏率"),
    ]
    for ax, (col, title) in zip(axes, series):
        if col not in eco:
            continue
        ax.plot(eco.index, pd.to_numeric(eco[col], errors="coerce"), linewidth=0.9)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    return _save(fig, "ecology.png")


def plot_all(codes: list[str]) -> list[str]:
    written = []
    for fn, args in (
        (plot_nav, (codes,)),
        (plot_drawdown, (codes,)),
        (plot_yearly, (codes,)),
        (plot_rolling12, (codes,)),
    ):
        try:
            written.append(str(fn(*args)))
        except Exception:
            continue
    for fn in (plot_alpha_beta, plot_margin_cal, plot_2d_heat, plot_eco_pct):
        try:
            p = fn()
            if p is not None:
                written.append(str(p))
        except Exception:
            continue
    return [p for p in written if p]
