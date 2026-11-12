"""训练折内相关与最近邻。全历史相关只作诊断。低相关只允许 Add。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND22_ROOT
from ..round7.contract import FIVE_HEAD
from .common import dump_json, load_json, write_csv, write_md
from .contract import (
    DUPLICATE_EXPRESSION,
    DUPLICATE_RHO,
    EXPLORATORY_DIAGNOSTIC,
    FIVE_FEATURES,
    KEYS,
    MD,
    NOT_APPLICABLE,
    REDUNDANCY_RHO,
    REPEATED_HISTORICAL_TESTING,
    TEN_FEATURES,
)

CORR_CSV = ROUND22_ROOT / "correlation" / "correlation.csv"


def _spear(a: pd.Series, b: pd.Series) -> float | None:
    m = a.notna() & b.notna()
    if int(m.sum()) < 30:
        return None
    val = a.loc[m].corr(b.loc[m], method="spearman")
    if val is None or not np.isfinite(val):
        return None
    return float(val)


def _mean_daily(frame: pd.DataFrame, left: str, right: str) -> float | None:
    vals = []
    for _, block in frame.groupby("trade_date", sort=False):
        ic = _spear(block[left], block[right])
        if ic is not None:
            vals.append(ic)
    if len(vals) < 20:
        return None
    return float(np.mean(vals))


def run_correlate(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and CORR_CSV.is_file() and (ROUND22_ROOT / "correlation" / "correlation.json").is_file():
        existing = load_json(ROUND22_ROOT / "correlation" / "correlation.json")
        if existing.get("status") == "DONE":
            return existing
    pit = load_json(ROUND22_ROOT / "pit" / "pit.json")
    panel = pd.read_parquet(ROUND22_ROOT / "data" / "factor_panel.parquet")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["instrument_id"] = panel["instrument_id"].astype(str)
    base_path = FIVE_HEAD.panel_dir / "candidate_panel.parquet"
    base_cols = [c for c in [*KEYS, *FIVE_FEATURES] if c]
    names = set(pd.read_parquet(base_path, columns=[]).columns) if False else None
    import pyarrow.parquet as pq

    schema = set(pq.read_schema(base_path).names)
    use = [c for c in [*KEYS, *FIVE_FEATURES] if c in schema]
    base = pd.read_parquet(base_path, columns=use)
    base["trade_date"] = pd.to_datetime(base["trade_date"])
    base["instrument_id"] = base["instrument_id"].astype(str)
    merged = panel.merge(base, on=KEYS, how="left")
    rows = []
    for spec in pit.get("rows") or []:
        fid = spec["factor_id"]
        rec: dict[str, Any] = {
            "factor_id": fid,
            "profile": spec.get("profile"),
            "pit_status": spec.get("pit_status"),
            "model_eligible": bool(spec.get("model_eligible")),
            "label": spec.get("label") or "",
            "testing": REPEATED_HISTORICAL_TESTING,
        }
        if fid not in merged.columns or spec.get("pit_status") != "PIT_PASS":
            rec.update({"five_nearest": None, "ten_nearest": None, "five_abs_rho": None, "ten_abs_rho": None, "five_action": NOT_APPLICABLE, "ten_action": NOT_APPLICABLE, "duplicate_hit": spec.get("duplicate_of") or ""})
            rows.append(rec)
            continue
        fold_stats = []
        for year in range(2018, 2027):
            mask = merged["trade_date"].dt.year.between(year - 3, year - 1)
            block = merged.loc[mask]
            if block.empty:
                continue
            five_best = (None, 0.0)
            ten_best = (None, 0.0)
            five_rhos: dict[str, float] = {}
            ten_rhos: dict[str, float] = {}
            for col in FIVE_FEATURES:
                if col not in block.columns:
                    continue
                rho = _spear(block[fid], block[col])
                if rho is None:
                    continue
                five_rhos[col] = float(rho)
                if abs(rho) >= abs(five_best[1]):
                    five_best = (col, rho)
                if col in TEN_FEATURES:
                    ten_rhos[col] = float(rho)
                    if abs(rho) >= abs(ten_best[1]):
                        ten_best = (col, rho)
            fold_stats.append(
                {
                    "year": year,
                    "five_feature": five_best[0],
                    "five_rho": five_best[1],
                    "ten_feature": ten_best[0],
                    "ten_rho": ten_best[1],
                    "five_rhos": five_rhos,
                    "ten_rhos": ten_rhos,
                    "mean_daily_five": _mean_daily(block, fid, five_best[0]) if five_best[0] else None,
                }
            )

        def _mode(key_name: str) -> str | None:
            votes: dict[str, int] = {}
            for item in fold_stats:
                name = item.get(key_name)
                if name:
                    votes[name] = votes.get(name, 0) + 1
            if not votes:
                return None
            return sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        def _median_of(rho_key: str, name: str | None) -> float | None:
            if not name:
                return None
            vals = [abs(float(item[rho_key][name])) for item in fold_stats if name in (item.get(rho_key) or {})]
            return float(np.median(vals)) if vals else None

        five_name = _mode("five_feature")
        ten_name = _mode("ten_feature")
        # 门限用众数最近邻自己的折内 |ρ| 中位数，不用各折最大值的中位数。
        five_med = _median_of("five_rhos", five_name)
        ten_med = _median_of("ten_rhos", ten_name)
        duplicate_hit = ""
        if five_med is not None and five_med >= DUPLICATE_RHO and five_name:
            duplicate_hit = five_name
            rec["model_eligible"] = False
            rec["pit_status"] = DUPLICATE_EXPRESSION
        rec.update(
            {
                "five_nearest": five_name,
                "ten_nearest": ten_name,
                "five_abs_rho": five_med,
                "ten_abs_rho": ten_med,
                "five_action": "REMOVE_REPLACE_ALLOWED" if five_med is not None and five_med >= REDUNDANCY_RHO and five_name in FIVE_FEATURES else (EXPLORATORY_DIAGNOSTIC if five_med is not None and five_med < REDUNDANCY_RHO else NOT_APPLICABLE),
                "ten_action": "REMOVE_REPLACE_ALLOWED" if ten_med is not None and ten_med >= REDUNDANCY_RHO and ten_name in TEN_FEATURES else (EXPLORATORY_DIAGNOSTIC if ten_med is not None and ten_med < REDUNDANCY_RHO else NOT_APPLICABLE),
                "duplicate_hit": duplicate_hit,
                "n_folds": len(fold_stats),
            }
        )
        if duplicate_hit:
            rec["five_action"] = NOT_APPLICABLE
            rec["ten_action"] = NOT_APPLICABLE
        dump_json(ROUND22_ROOT / "factors" / fid / "redundancy.json", {"folds": fold_stats, "summary": rec})
        rows.append(rec)
    write_csv(CORR_CSV, rows)
    payload = {
        "status": "DONE",
        "n": len(rows),
        "n_high_rho": int(sum(r.get("five_action") == "REMOVE_REPLACE_ALLOWED" or r.get("ten_action") == "REMOVE_REPLACE_ALLOWED" for r in rows)),
        "n_still_eligible": int(sum(bool(r.get("model_eligible")) for r in rows)),
        "rho_threshold": REDUNDANCY_RHO,
        "full_sample_used_for_gate": False,
    }
    dump_json(ROUND22_ROOT / "correlation" / "correlation.json", payload | {"rows": rows})
    write_md(
        MD["13"],
        "\n".join(
            [
                "# Round22 · 13 折内相关",
                "",
                "相关在每个预测年的训练窗（Y-3…Y-1）内计算。删除列用折内最近邻的众数，不用全历史相关。",
                f"仍可训练={payload['n_still_eligible']}。达到 |ρ|≥{REDUNDANCY_RHO} 的头={payload['n_high_rho']}。",
                "低相关删除只标 EXPLORATORY_DIAGNOSTIC，不进正式袖套。缺最近邻则该头 NOT_APPLICABLE。",
                f"标记 `{REPEATED_HISTORICAL_TESTING}`。",
                "",
            ]
        ),
    )
    return payload
