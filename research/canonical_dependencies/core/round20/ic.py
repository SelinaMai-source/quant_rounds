"""双标签日频截面 IC / RankIC / 年频 / HAC / 块 bootstrap。薄日跳过，不记 IC=0。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND20_ROOT
from .common import dump_json, finite_or_none, load_json, write_md
from .contract import (
    BLOCK_DAYS,
    BOOT_N,
    COMPLETE_IC_YEARS,
    F2_NAME,
    IC_MD,
    IC_MIN_VALID_NAMES_PER_DAY,
    LABELS,
)
from .freeze import require_frozen
from .metrics import block_bootstrap_mean, hac_mean, icir_of, pearson_ic, rank_ic, same_sign_years, seed_for
from .panel import build_diag_panel, ic_window, load_registry
from .pit import PIT_JSON, run_pit

IC_JSON = ROUND20_ROOT / "ic" / "ic.json"
IC_CSV = ROUND20_ROOT / "ic" / "ic_summary.csv"
DAILY_PATH = ROUND20_ROOT / "ic" / "daily_ic.parquet"


def _summarize_series(daily: pd.DataFrame, factor: str, label: str, metric: str) -> dict[str, Any]:
    sub = daily.loc[daily["factor"].eq(factor) & daily["label"].eq(label), ["trade_date", metric, "n_valid"]]
    series = pd.to_numeric(sub[metric], errors="coerce")
    used = series[np.isfinite(series.to_numpy(dtype=float))]
    years = pd.to_datetime(sub.loc[used.index, "trade_date"]).dt.year
    yearly: dict[str, float | None] = {}
    for y in COMPLETE_IC_YEARS:
        block = used.loc[years.eq(y)]
        yearly[str(int(y))] = float(block.mean()) if len(block) else None
    arr = used.to_numpy(dtype=float)
    hac = hac_mean(arr, lags=BLOCK_DAYS)
    boot = block_bootstrap_mean(arr, block=BLOCK_DAYS, n=BOOT_N, seed=seed_for(factor, label, metric))
    mean = float(np.mean(arr)) if len(arr) else None
    return {
        "factor": factor,
        "label": label,
        "metric": metric,
        "n_days_used": int(len(arr)),
        "n_days_listed": int(sub["trade_date"].nunique()) if len(sub) else 0,
        "mean": mean,
        "std": float(np.std(arr, ddof=1)) if len(arr) >= 2 else None,
        "icir": icir_of(arr),
        "yearly": yearly,
        "yearly_icir": icir_of(np.array([v for v in yearly.values() if v is not None], dtype=float)),
        "same_sign_years": same_sign_years(yearly, COMPLETE_IC_YEARS),
        "hac": hac,
        "bootstrap": boot,
    }


def run_ic(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and IC_JSON.is_file() and IC_CSV.is_file() and DAILY_PATH.is_file() and IC_MD.is_file():
        existing = load_json(IC_JSON)
        if existing.get("status") == "DONE":
            return existing
    pit = run_pit(resume=True)
    pit_rows = {r["name"]: r for r in pit.get("rows") or []}
    panel = ic_window(build_diag_panel(resume=True))
    reg = load_registry()
    names = [str(x) for x in reg["name"].tolist()]
    day_ids, unique_dates = pd.factorize(panel["trade_date"], sort=True)
    unique_dates = pd.DatetimeIndex(unique_dates)
    n_dates = int(len(unique_dates))
    order = np.argsort(day_ids, kind="mergesort")
    sorted_ids = day_ids[order]
    bounds = np.append(np.flatnonzero(np.diff(sorted_ids, prepend=-1)), len(order))
    groups = {int(sorted_ids[a]): order[a:b] for a, b in zip(bounds[:-1], bounds[1:])}

    label_arrs = {lab: pd.to_numeric(panel[lab], errors="coerce").to_numpy(dtype=float) for lab in LABELS if lab in panel.columns}
    daily_rows: list[dict[str, Any]] = []
    skip_counts = {
        "thin": 0,
        "degenerate": 0,
        "label_missing": 0,
        "factor_missing": 0,
        "recorded": 0,
        "ic_recorded_as_zero": 0,
    }
    for feat in names:
        if feat not in panel.columns:
            continue
        fx = pd.to_numeric(panel[feat], errors="coerce").to_numpy(dtype=float)
        for lab, y in label_arrs.items():
            for di, idx in groups.items():
                xv = fx[idx]
                yv = y[idx]
                mask = np.isfinite(xv) & np.isfinite(yv)
                n_valid = int(mask.sum())
                if n_valid < IC_MIN_VALID_NAMES_PER_DAY:
                    skip_counts["thin"] += 1
                    continue
                x1, y1 = xv[mask], yv[mask]
                if int(np.unique(x1).size) < 2 or int(np.unique(y1).size) < 2:
                    skip_counts["degenerate"] += 1
                    continue
                xs = pd.Series(x1)
                ys = pd.Series(y1)
                ic = pearson_ic(xs, ys)
                ric = rank_ic(xs, ys)
                if ic is None and ric is None:
                    skip_counts["degenerate"] += 1
                    continue
                # 跳过的日不写 0；只有真正算出的值才落盘
                daily_rows.append(
                    {
                        "trade_date": unique_dates[di],
                        "factor": feat,
                        "label": lab,
                        "n_valid": n_valid,
                        "ic": ic,
                        "rankic": ric,
                    }
                )
                skip_counts["recorded"] += 1
                if ic == 0.0:
                    skip_counts["ic_recorded_as_zero"] += 1
    daily = pd.DataFrame(daily_rows)
    DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if daily.empty:
        daily = pd.DataFrame(columns=["trade_date", "factor", "label", "n_valid", "ic", "rankic"])
    daily.to_parquet(DAILY_PATH, index=False)

    summaries: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for feat in names:
        rec_pit = pit_rows.get(feat) or {}
        for lab in LABELS:
            for metric in ("ic", "rankic"):
                part = daily.loc[daily["factor"].eq(feat) & daily["label"].eq(lab)] if len(daily) else daily
                summ = _summarize_series(daily, feat, lab, metric) if len(part) else {
                    "factor": feat,
                    "label": lab,
                    "metric": metric,
                    "n_days_used": 0,
                    "mean": None,
                    "icir": None,
                    "yearly": {str(y): None for y in COMPLETE_IC_YEARS},
                    "hac": {"status": "TOO_SHORT"},
                    "bootstrap": {"status": "TOO_SHORT"},
                    "same_sign_years": 0,
                    "yearly_icir": None,
                }
                if rec_pit.get("market_level") and summ["n_days_used"] == 0:
                    summ["skip_reason"] = "NO_WITHIN_DAY_VARIATION; 不把 IC 记 0"
                summ["pit_status"] = rec_pit.get("pit_status")
                summ["promotion_eligible"] = rec_pit.get("promotion_eligible")
                summ["family"] = rec_pit.get("family")
                summaries.append(summ)
                boot = summ.get("bootstrap") or {}
                hac = summ.get("hac") or {}
                flat_rows.append(
                    {
                        "factor": feat,
                        "family": rec_pit.get("family"),
                        "label": lab,
                        "metric": metric,
                        "n_days_used": summ.get("n_days_used"),
                        "mean": summ.get("mean"),
                        "std": summ.get("std"),
                        "icir": summ.get("icir"),
                        "yearly_icir": summ.get("yearly_icir"),
                        "same_sign_years": summ.get("same_sign_years"),
                        "hac_se": hac.get("hac_se"),
                        "hac_p": hac.get("p_two_sided"),
                        "boot_p05": boot.get("p05"),
                        "boot_p95": boot.get("p95"),
                        "boot_p_mean_gt0": boot.get("p_mean_gt0"),
                        "pit_status": rec_pit.get("pit_status"),
                        "promotion_eligible": rec_pit.get("promotion_eligible"),
                        "f2": feat == F2_NAME,
                        "skip_reason": summ.get("skip_reason"),
                    }
                )
    flat = pd.DataFrame(flat_rows)
    IC_CSV.parent.mkdir(parents=True, exist_ok=True)
    flat.to_csv(IC_CSV, index=False)

    def _best(metric: str, label: str) -> list[dict[str, Any]]:
        sub = flat.loc[
            flat["metric"].eq(metric)
            & flat["label"].eq(label)
            & flat["f2"].eq(False)
            & flat["mean"].notna()
        ].copy()
        sub["abs_icir"] = pd.to_numeric(sub["icir"], errors="coerce").abs()
        sub = sub.sort_values(["abs_icir", "mean"], ascending=[False, False])
        return sub.head(8).drop(columns=["abs_icir"]).to_dict(orient="records")

    best = {
        f"{metric}|{lab}": _best(metric, lab) for metric in ("ic", "rankic") for lab in LABELS
    }
    payload = {
        "status": "DONE",
        "min_valid_names_per_day": IC_MIN_VALID_NAMES_PER_DAY,
        "do_not_record_ic_as_zero_when_skipped": True,
        "n_dates_window": n_dates,
        "n_daily_rows": int(len(daily)),
        "skip_counts": skip_counts,
        "block_days": BLOCK_DAYS,
        "bootstrap_n": BOOT_N,
        "best": best,
        "csv": str(IC_CSV),
        "daily": str(DAILY_PATH),
        "summaries": summaries,
        "note": "薄日与退化日跳过，不写 IC=0。F2 进审计表但不可晋级。市场层因子截面 IC 为 NO_WITHIN_DAY_VARIATION。",
    }
    dump_json(IC_JSON, payload)

    def _fmt(rec: dict[str, Any]) -> str:
        return (
            f"| `{rec.get('factor')}` | {rec.get('mean'):.4f} | {finite_or_none(rec.get('icir'))} | "
            f"{rec.get('n_days_used')} | {rec.get('hac_p')} |"
        )

    lines = [
        "# 03 双标签 IC / RankIC",
        "",
        f"日截面最少有效股票=`{IC_MIN_VALID_NAMES_PER_DAY}`。不足则跳过该日，**不把 IC 记 0**。",
        f"HAC lags=`{BLOCK_DAYS}`；块 bootstrap N=`{BOOT_N}` block=`{BLOCK_DAYS}`。",
        f"n_daily_rows=`{len(daily)}` skip_thin=`{skip_counts['thin']}` skip_degenerate=`{skip_counts['degenerate']}`。",
        "",
        "## 最佳 RankIC（label_e2_net，排除 F2）",
        "",
        "| factor | mean | ICIR | n_days | HAC p |",
        "|---|---:|---:|---:|---:|",
    ]
    for rec in best.get("rankic|label_e2_net") or []:
        if rec.get("mean") is None:
            continue
        lines.append(_fmt(rec))
    lines += [
        "",
        "## 最佳 RankIC（label_exit_rule_net，排除 F2）",
        "",
        "| factor | mean | ICIR | n_days | HAC p |",
        "|---|---:|---:|---:|---:|",
    ]
    for rec in best.get("rankic|label_exit_rule_net") or []:
        if rec.get("mean") is None:
            continue
        lines.append(_fmt(rec))
    lines += ["", f"csv：`{IC_CSV}`", f"daily：`{DAILY_PATH}`", ""]
    write_md(IC_MD, "\n".join(lines))
    return payload


__all__ = ["DAILY_PATH", "IC_CSV", "IC_JSON", "run_ic"]
