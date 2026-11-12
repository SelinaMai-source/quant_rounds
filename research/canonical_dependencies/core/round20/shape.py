"""Q1–Q5 收益形态 + Top-tail。样本不够不强制 Q10。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND20_ROOT
from .common import dump_json, load_json, write_md
from .contract import F2_NAME, IC_MIN_VALID_NAMES_PER_DAY, LABELS, SHAPE_MD
from .freeze import require_frozen
from .panel import build_diag_panel, ic_window, load_registry
from .pit import run_pit

SHAPE_JSON = ROUND20_ROOT / "shape" / "shape.json"
SHAPE_CSV = ROUND20_ROOT / "shape" / "shape.csv"


def _monotonic(vals: list[float | None]) -> str:
    xs = [v for v in vals if v is not None and np.isfinite(v)]
    if len(xs) < 3:
        return "INSUFFICIENT"
    inc = all(xs[i] <= xs[i + 1] + 1e-12 for i in range(len(xs) - 1))
    dec = all(xs[i] >= xs[i + 1] - 1e-12 for i in range(len(xs) - 1))
    if inc and dec:
        return "FLAT"
    if inc:
        return "INCREASING"
    if dec:
        return "DECREASING"
    return "NON_MONOTONIC"


def run_shape(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and SHAPE_JSON.is_file() and SHAPE_CSV.is_file() and SHAPE_MD.is_file():
        existing = load_json(SHAPE_JSON)
        if existing.get("status") == "DONE":
            return existing
    pit = {r["name"]: r for r in (run_pit(resume=True).get("rows") or [])}
    panel = ic_window(build_diag_panel(resume=True))
    names = [str(x) for x in load_registry()["name"].tolist()]
    day_ids, _unique_dates = pd.factorize(panel["trade_date"], sort=True)
    order = np.argsort(day_ids, kind="mergesort")
    sorted_ids = day_ids[order]
    bounds = np.append(np.flatnonzero(np.diff(sorted_ids, prepend=-1)), len(order))
    groups = {int(sorted_ids[a]): order[a:b] for a, b in zip(bounds[:-1], bounds[1:])}
    label_arrs = {lab: pd.to_numeric(panel[lab], errors="coerce").to_numpy(dtype=float) for lab in LABELS if lab in panel.columns}

    rows: list[dict[str, Any]] = []
    highlights: list[dict[str, Any]] = []
    for feat in names:
        if feat not in panel.columns:
            continue
        fx = pd.to_numeric(panel[feat], errors="coerce").to_numpy(dtype=float)
        for lab, y in label_arrs.items():
            q_means = {i: [] for i in range(5)}
            n_days_q5 = 0
            n_days_q10 = 0
            n_days_skip_thin = 0
            n_days_skip_q10 = 0
            top_spreads: list[float] = []
            top2_spreads: list[float] = []
            n_bins_seen: list[int] = []
            for idx in groups.values():
                xv = fx[idx]
                yv = y[idx]
                mask = np.isfinite(xv) & np.isfinite(yv)
                n_valid = int(mask.sum())
                if n_valid < IC_MIN_VALID_NAMES_PER_DAY:
                    n_days_skip_thin += 1
                    continue
                x1, y1 = xv[mask], yv[mask]
                if int(np.unique(x1).size) < 2:
                    continue
                order_x = np.argsort(x1, kind="mergesort")
                ranks = np.empty(n_valid, dtype=float)
                ranks[order_x] = (np.arange(n_valid, dtype=float) + 1.0) / float(n_valid)
                q5 = np.minimum((ranks * 5.0).astype(int), 4)
                bins = int(np.unique(q5).size)
                n_bins_seen.append(bins)
                if bins >= 2:
                    n_days_q5 += 1
                    for b in range(5):
                        sel = y1[q5 == b]
                        if len(sel):
                            q_means[b].append(float(np.mean(sel)))
                # Q10 / top-tail 仅当当日有效数 ≥10；不够不强制
                if n_valid >= 10:
                    n_days_q10 += 1
                    top = y1[ranks >= 0.90]
                    bot = y1[ranks <= 0.10]
                    if len(top) and len(bot):
                        top_spreads.append(float(np.mean(top) - np.mean(bot)))
                else:
                    n_days_skip_q10 += 1
                top2 = y1[order_x[-2:]]
                rest = y1[order_x[:-2]] if n_valid > 2 else np.array([])
                if len(top2) and len(rest):
                    top2_spreads.append(float(np.mean(top2) - np.mean(rest)))
            q_avg = [float(np.mean(q_means[i])) if q_means[i] else None for i in range(5)]
            mono = _monotonic(q_avg)
            q5q1 = None
            if q_avg[0] is not None and q_avg[4] is not None:
                q5q1 = q_avg[4] - q_avg[0]
            rec = {
                "factor": feat,
                "label": lab,
                "family": (pit.get(feat) or {}).get("family"),
                "q1": q_avg[0],
                "q2": q_avg[1],
                "q3": q_avg[2],
                "q4": q_avg[3],
                "q5": q_avg[4],
                "q5_minus_q1": q5q1,
                "monotonicity": mono,
                "n_days_q5": n_days_q5,
                "n_days_q10": n_days_q10,
                "n_days_skip_thin": n_days_skip_thin,
                "n_days_q10_not_forced": n_days_skip_q10,
                "top_tail_mean_spread": float(np.mean(top_spreads)) if top_spreads else None,
                "top2_minus_rest": float(np.mean(top2_spreads)) if top2_spreads else None,
                "median_bins_q5": float(np.median(n_bins_seen)) if n_bins_seen else None,
                "f2": feat == F2_NAME,
                "forced_q10": False,
            }
            rows.append(rec)
            if rec["top_tail_mean_spread"] is not None or rec["q5_minus_q1"] is not None:
                highlights.append(rec)
    frame = pd.DataFrame(rows)
    SHAPE_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SHAPE_CSV, index=False)
    nonmono = frame.loc[frame["monotonicity"].eq("NON_MONOTONIC") & frame["f2"].eq(False)]
    best_q = (
        frame.loc[frame["f2"].eq(False) & frame["q5_minus_q1"].notna()]
        .assign(abs_spread=lambda d: d["q5_minus_q1"].abs())
        .sort_values("abs_spread", ascending=False)
        .head(8)
        .drop(columns=["abs_spread"])
    )
    best_tail = (
        frame.loc[frame["f2"].eq(False) & frame["top_tail_mean_spread"].notna()]
        .assign(abs_spread=lambda d: d["top_tail_mean_spread"].abs())
        .sort_values("abs_spread", ascending=False)
        .head(8)
        .drop(columns=["abs_spread"])
    )
    payload = {
        "status": "DONE",
        "forced_q10": False,
        "n_rows": int(len(frame)),
        "n_non_monotonic": int(len(nonmono)),
        "best_q5_minus_q1": best_q.to_dict(orient="records"),
        "best_top_tail": best_tail.to_dict(orient="records"),
        "csv": str(SHAPE_CSV),
        "note": "Q1–Q5；当日有效数<10 不强制 Q10。薄日跳过。F2 进表但不可晋级。",
    }
    dump_json(SHAPE_JSON, payload)
    lines = [
        "# 04 Shape / Top-tail",
        "",
        "Q1–Q5。样本不够**不强制 Q10**。Top-tail = 当日 rank pct 顶 10% 减底 10%（仅 n≥10 的日）。",
        f"non_monotonic 因子×标签=`{payload['n_non_monotonic']}`。",
        "",
        "## Q5−Q1 |spread| 最大（排除 F2）",
        "",
        "| factor | label | Q1 | Q5 | Q5−Q1 | mono |",
        "|---|---|---:|---:|---:|---|",
    ]
    for rec in payload["best_q5_minus_q1"]:
        lines.append(
            f"| `{rec['factor']}` | {rec['label']} | {rec['q1']:.4f} | {rec['q5']:.4f} | "
            f"{rec['q5_minus_q1']:.4f} | {rec['monotonicity']} |"
        )
    lines += [
        "",
        "## Top-tail 最大（排除 F2）",
        "",
        "| factor | label | top-tail | top2−rest | n_days_q10 |",
        "|---|---|---:|---:|---:|",
    ]
    for rec in payload["best_top_tail"]:
        tt = rec.get("top_tail_mean_spread")
        t2 = rec.get("top2_minus_rest")
        lines.append(
            f"| `{rec['factor']}` | {rec['label']} | {tt:.4f} | "
            f"{'—' if t2 is None else f'{t2:.4f}'} | {rec['n_days_q10']} |"
        )
    lines += ["", f"csv：`{SHAPE_CSV}`", ""]
    write_md(SHAPE_MD, "\n".join(lines))
    return payload


__all__ = ["SHAPE_CSV", "SHAPE_JSON", "run_shape"]
