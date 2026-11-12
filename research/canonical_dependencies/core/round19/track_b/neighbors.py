"""F1/F2/F3 相对近邻因子的条件增量。不改公式、不重训。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...paths import ROUND18_ROOT
from ...round18.contract import F1_NAME, F2_NAME, F3_NAME, NEIGHBORS
from ...round18.factors import load_factor_panel
from .retrain import run_fresh_retrain
from .support import (
    FIVE_HEAD,
    KEYS,
    dump_json,
    finite_corr,
    load_json,
    load_original_panels,
    load_r18_panels,
    tb_dir,
    write_md,
    year_of,
)

PAIRS = {
    F1_NAME: ("h5_close_limit_value_to_amount", "fb_close_seal_streak_norm"),
    F2_NAME: ("h4_large_net_to_turnover", "h4_retail_net_to_turnover"),
    F3_NAME: ("h5_close_limit_value_to_amount", "g0_float_mv_log"),
}


def _ensure(**kw: Any) -> None:
    if not (tb_dir("retrain") / "fresh_retrain.json").is_file():
        run_fresh_retrain(**kw)


def _partial_rank_ic(y: pd.Series, x: pd.Series, controls: pd.DataFrame) -> float | None:
    work = pd.DataFrame({"y": pd.to_numeric(y, errors="coerce"), "x": pd.to_numeric(x, errors="coerce")})
    for c in controls.columns:
        work[c] = pd.to_numeric(controls[c], errors="coerce")
    work = work.dropna()
    if len(work) < 30:
        return None
    yr = work["y"].rank(method="average")
    xr = work["x"].rank(method="average")
    cr = work[list(controls.columns)].rank(method="average")
    # residualize via OLS on ranks
    cmat = np.column_stack([np.ones(len(work)), cr.to_numpy(dtype=float)])
    try:
        beta_y, *_ = np.linalg.lstsq(cmat, yr.to_numpy(dtype=float), rcond=None)
        beta_x, *_ = np.linalg.lstsq(cmat, xr.to_numpy(dtype=float), rcond=None)
    except np.linalg.LinAlgError:
        return None
    ry = yr.to_numpy(dtype=float) - cmat @ beta_y
    rx = xr.to_numpy(dtype=float) - cmat @ beta_x
    if np.std(ry) < 1e-12 or np.std(rx) < 1e-12:
        return None
    val = float(np.corrcoef(rx, ry)[0, 1])
    return val if np.isfinite(val) else None


def run_neighbors(**kw: Any) -> dict[str, Any]:
    _ensure(**kw)
    orig = load_original_panels()
    r18 = load_r18_panels()
    try:
        factors = load_factor_panel()
        factors["trade_date"] = pd.to_datetime(factors["trade_date"])
        factors["instrument_id"] = factors["instrument_id"].astype(str)
    except Exception as exc:
        factors = None
        factor_err = str(exc)
    else:
        factor_err = None
    panel = orig["five_candidate"].copy()
    if factors is not None:
        panel = panel.merge(factors, on=KEYS, how="left")
    else:
        # R18 merged already has F columns
        extra = r18["five_candidate"][KEYS + [c for c in (F1_NAME, F2_NAME, F3_NAME) if c in r18["five_candidate"].columns]]
        panel = panel.merge(extra, on=KEYS, how="left")
    target = FIVE_HEAD.target
    rows = []
    for fname, neighbors in PAIRS.items():
        if fname not in panel.columns:
            rows.append({"factor": fname, "status": "NOT_IDENTIFIABLE", "reason": "factor column missing"})
            continue
        item: dict[str, Any] = {
            "factor": fname,
            "contract_neighbors": list(NEIGHBORS.get(fname, ())),
            "pairs": [],
            "status": "done",
        }
        raw_ic = finite_corr(panel[fname], panel[target]) if target in panel.columns else None
        item["rank_ic_vs_label"] = raw_ic
        present = [c for c in neighbors if c in panel.columns]
        missing = [c for c in neighbors if c not in panel.columns]
        item["neighbors_present"] = present
        item["neighbors_missing"] = missing
        for neigh in neighbors:
            if neigh not in panel.columns:
                item["pairs"].append({"neighbor": neigh, "spearman": None, "status": "NOT_IDENTIFIABLE"})
                continue
            item["pairs"].append({
                "neighbor": neigh,
                "spearman": finite_corr(panel[fname], panel[neigh]),
                "status": "done",
            })
        if present and target in panel.columns:
            item["partial_rank_ic_vs_label_given_neighbors"] = _partial_rank_ic(
                panel[target], panel[fname], panel[present]
            )
        else:
            item["partial_rank_ic_vs_label_given_neighbors"] = None
            if missing:
                item["status"] = "PARTIAL"
        rows.append(item)
    matrix = load_json(ROUND18_ROOT / "metrics" / "matrix.json")
    r18_obs = []
    for rec in (matrix.get("rows") or []):
        if rec.get("code") in {"R18_A00", "R18_A01", "R18_A02", "R18_A03", "R18_D01", "R18_D02", "R18_D03", "R18_D04", "R18_D05", "R18_D06"}:
            r18_obs.append({
                "code": rec.get("code"),
                "role": rec.get("role"),
                "sharpe": rec.get("sharpe"),
                "cagr": rec.get("cagr"),
                "formal": rec.get("formal"),
            })
    payload = {
        "status": "done" if factor_err is None else "PARTIAL",
        "factor_panel_error": factor_err,
        "rows": rows,
        "r18_single_factor_observed": r18_obs,
        "note": (
            "条件增量 = 控制近邻后的 residual Rank IC。不是新公式搜索。"
            "R18 已观测到加 F 列后正式组全面弱于 A00；本表只解释是否只是近邻变换。"
        ),
        "diagnostic_not_eligible": True,
        "n_rows": int(len(panel)),
        "years": sorted({int(y) for y in year_of(panel["trade_date"]).tolist()}) if len(panel) else [],
    }
    dump_json(tb_dir("heads") / "neighbors.json", payload)
    lines = [
        "# Track B · 近邻条件增量",
        "",
        "| factor | Rank IC vs label | partial IC | neighbor | Spearman |",
        "|---|---:|---:|---|---:|",
    ]
    for r in rows:
        pairs = r.get("pairs") or [{"neighbor": "-", "spearman": None}]
        for i, p in enumerate(pairs):
            lines.append(
                f"| {r.get('factor') if i == 0 else ''} | {r.get('rank_ic_vs_label') if i == 0 else ''} | "
                f"{r.get('partial_rank_ic_vs_label_given_neighbors') if i == 0 else ''} | "
                f"{p.get('neighbor')} | {p.get('spearman')} |"
            )
    write_md(tb_dir("heads") / "11_neighbors.md", "\n".join(lines) + "\n")
    return payload


__all__ = ["run_neighbors"]
