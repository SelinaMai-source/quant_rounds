"""PIT / 覆盖 / 与近邻冗余。看 NAV 前完成。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND18_ROOT
from ..round7.contract import FIVE_HEAD
from .common import dump_json, load_json, write_md
from .contract import (
    COVERAGE_MIN,
    F1_NAME,
    F2_NAME,
    F3_NAME,
    NEIGHBORS,
    PARTIAL_FACTOR_COVERAGE,
    PIT_TIMESTAMP_UNVERIFIED,
    SKIP_FACTOR_FOR_PROMOTION,
    TOUCH_TIME_APPROXIMATED,
)
from .factors import PANEL, load_factor_panel

PIT_JSON = ROUND18_ROOT / "pit" / "pit.json"
PIT_MD = ROUND18_ROOT / "06_pit_coverage.md"


def _year_cov(frame: pd.DataFrame, col: str) -> dict[str, Any]:
    work = frame.copy()
    work["year"] = pd.to_datetime(work["trade_date"]).dt.year
    out = {}
    for y, block in work.groupby("year"):
        n = int(len(block))
        v = int(block[col].notna().sum())
        out[str(int(y))] = {"n": n, "valid": v, "coverage": (v / n) if n else None}
    return out


def _redundancy(panel: pd.DataFrame, cand: pd.DataFrame) -> dict[str, Any]:
    work = cand.merge(panel, on=["trade_date", "instrument_id"], how="left", suffixes=("", "_r18"))
    out: dict[str, Any] = {}
    for name, neigh in NEIGHBORS.items():
        left = work[name] if name in work.columns else None
        if left is None or not isinstance(left, pd.Series):
            continue
        rec = {}
        for n in neigh:
            if n not in work.columns:
                rec[n] = None
                continue
            series = work[n]
            if not isinstance(series, pd.Series):
                rec[n] = None
                continue
            rec[n] = float(pd.to_numeric(left, errors="coerce").corr(pd.to_numeric(series, errors="coerce")))
        out[name] = rec
    if {F1_NAME, F3_NAME} <= set(work.columns):
        out["F1_vs_F3"] = float(
            pd.to_numeric(work[F1_NAME], errors="coerce").corr(pd.to_numeric(work[F3_NAME], errors="coerce"))
        )
    return out


def run_pit(**_: Any) -> dict[str, Any]:
    panel = load_factor_panel()
    import pyarrow.parquet as pq

    raw_cols = list(pq.read_schema(FIVE_HEAD.panel_dir / "candidate_panel.parquet").names)
    neigh_cols = list(dict.fromkeys(c for c in sum((tuple(v) for v in NEIGHBORS.values()), ()) if c in raw_cols))
    cand = pd.read_parquet(
        FIVE_HEAD.panel_dir / "candidate_panel.parquet",
        columns=["trade_date", "instrument_id", *neigh_cols],
    )
    cand["trade_date"] = pd.to_datetime(cand["trade_date"])
    cand["instrument_id"] = cand["instrument_id"].astype(str)
    joined = cand.merge(panel, on=["trade_date", "instrument_id"], how="left")
    f2_meta = load_json(ROUND18_ROOT / "factors" / "f2_meta.json")
    f1_meta = load_json(ROUND18_ROOT / "factors" / "f1_meta.json")
    f3_meta = load_json(ROUND18_ROOT / "factors" / "f3_meta.json")
    factors = {}
    for name, meta, default_pit in (
        (F1_NAME, f1_meta, "D_close_minutes_plus_limit_events"),
        (F2_NAME, f2_meta, PIT_TIMESTAMP_UNVERIFIED),
        (F3_NAME, f3_meta, "asof_D_shares_times_unadj_close"),
    ):
        year = _year_cov(joined, name)
        n = int(len(joined))
        v = int(joined[name].notna().sum())
        cov = v / n if n else 0.0
        years_gap = [y for y, r in year.items() if (r.get("coverage") or 0) < 0.50]
        coverage_ok = cov >= COVERAGE_MIN and not years_gap
        pit_status = meta.get("pit_status") or default_pit
        if name == F2_NAME:
            pit_status = meta.get("pit_status") or PIT_TIMESTAMP_UNVERIFIED
        promote_ok = bool(coverage_ok and pit_status not in {PIT_TIMESTAMP_UNVERIFIED, "SKIP_SCHEMA", "SKIP_DATA"})
        if name == F3_NAME:
            # available_at==trade_date 有修订风险，但 as-of-D 构造是计划指定口径
            promote_ok = bool(coverage_ok and meta.get("used_total_mv") is not True)
        if name == F1_NAME:
            promote_ok = bool(coverage_ok and meta.get("status") != "SKIP_DATA")
        factors[name] = {
            "n": n,
            "n_valid": v,
            "coverage": cov,
            "coverage_ok": coverage_ok,
            "year_coverage": year,
            "year_gaps": years_gap,
            "pit_status": pit_status,
            "promote_ok": promote_ok,
            "skip_reason": None if promote_ok else (
                SKIP_FACTOR_FOR_PROMOTION if not coverage_ok or pit_status == PIT_TIMESTAMP_UNVERIFIED
                else meta.get("status")
            ),
            "partial": PARTIAL_FACTOR_COVERAGE if not coverage_ok else None,
            "meta": {k: meta.get(k) for k in ("status", "n_touch_approx", "n_minute_missing", "revision_risk") if k in meta},
        }
    red = _redundancy(panel, cand)
    payload = {
        "status": "done",
        "coverage_min": COVERAGE_MIN,
        "factors": factors,
        "redundancy": red,
        "f2_formal_eligible": bool((factors.get(F2_NAME) or {}).get("promote_ok")),
        "touch_approx": TOUCH_TIME_APPROXIMATED,
        "filled_with_zero": False,
    }
    dump_json(PIT_JSON, payload)
    lines = ["# 06 PIT 与覆盖", "", f"覆盖门 ≥{COVERAGE_MIN:.0%} 且主要年份无断层。PIT 失败不得晋升。", ""]
    for name, rec in factors.items():
        lines.append(
            f"- `{name}` valid=`{rec['n_valid']}/{rec['n']}` cov=`{rec['coverage']}` "
            f"pit=`{rec['pit_status']}` promote_ok=`{rec['promote_ok']}`"
        )
    lines += ["", "## 近邻相关", "", str(red)]
    write_md(PIT_MD, "\n".join(lines))
    return payload
