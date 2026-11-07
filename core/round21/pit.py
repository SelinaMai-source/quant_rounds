"""逐因子 schema / 覆盖 / 单位 / PIT。E 无公告时点 → BLOCKED_PIT。缺失不填 0。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_ROOT
from .common import dump_json, load_json, write_md
from .construct import PANEL_PATH, UNITS, run_construct
from .contract import (
    BLOCKED_DATA,
    BLOCKED_PIT,
    COMPLETE_IC_YEARS,
    DIAGNOSTIC_ONLY,
    PIT_ELIGIBLE_MODEL,
    PIT_MIN_FINITE,
    PIT_YEAR_GAP,
)
from .freeze import require_frozen
from .hypotheses import HYPOTHESES

PIT_JSON = ROUND21_ROOT / "pit" / "pit.json"
PIT_CSV = ROUND21_ROOT / "pit" / "pit.csv"
PIT_MD = ROUND21_ROOT / "03_pit.md"
FACTORS_DIR = ROUND21_ROOT / "factors"


def _year_cov(frame: pd.DataFrame, col: str) -> dict[str, Any]:
    years = pd.to_datetime(frame["trade_date"]).dt.year
    s = pd.to_numeric(frame[col], errors="coerce") if col in frame.columns else pd.Series(np.nan, index=frame.index)
    out: dict[str, Any] = {}
    for y, idx in years.groupby(years).groups.items():
        block = s.loc[idx]
        n = int(len(block))
        v = int(np.isfinite(block.to_numpy(dtype=float)).sum())
        out[str(int(y))] = {"n": n, "valid": v, "coverage": (v / n) if n else None}
    return out


def _schema(fid: str, series: pd.Series, card: dict[str, Any]) -> dict[str, Any]:
    finite = pd.to_numeric(series, errors="coerce")
    finite = finite[np.isfinite(finite.to_numpy(dtype=float))]
    return {
        "factor_id": fid,
        "dtype": "float64",
        "nullable": True,
        "fillna_zero": False,
        "inputs": card.get("inputs"),
        "n_unique_finite": int(finite.nunique()) if len(finite) else 0,
        "min": float(finite.min()) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None,
        "mean": float(finite.mean()) if len(finite) else None,
    }


def run_pit(*, resume: bool = True, jobs: int = 8, **kwargs: Any) -> dict[str, Any]:
    require_frozen()
    if resume and PIT_JSON.is_file() and PIT_CSV.is_file() and PIT_MD.is_file():
        existing = load_json(PIT_JSON)
        if existing.get("status") == "DONE":
            return existing
    construct = run_construct(resume=True, jobs=jobs, **kwargs)
    if not PANEL_PATH.is_file():
        raise RuntimeError("construct panel missing")
    panel = pd.read_parquet(PANEL_PATH)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    cards = {c["factor_id"]: c for c in HYPOTHESES}
    rows: list[dict[str, Any]] = []
    for card in HYPOTHESES:
        fid = card["factor_id"]
        present = fid in panel.columns
        s = pd.to_numeric(panel[fid], errors="coerce") if present else pd.Series(np.nan, index=panel.index)
        n = int(len(panel))
        v = int(np.isfinite(s.to_numpy(dtype=float)).sum()) if present else 0
        cov = (v / n) if n else 0.0
        year = _year_cov(panel, fid) if present else {}
        years_gap = [
            y for y, r in year.items() if int(y) in COMPLETE_IC_YEARS and (r.get("coverage") or 0) < PIT_YEAR_GAP
        ]
        finite = s[np.isfinite(s.to_numpy(dtype=float))] if present else pd.Series(dtype=float)
        nunique = int(finite.nunique(dropna=True)) if len(finite) else 0
        day_nu = None
        market_level = False
        if present and nunique > 1:
            tmp = pd.DataFrame({"d": panel["trade_date"], "x": s})
            tmp = tmp.loc[np.isfinite(tmp["x"].to_numpy(dtype=float))]
            if len(tmp):
                day_nu = float(tmp.groupby("d")["x"].nunique().median())
                market_level = bool(day_nu <= 1.0)
        schema = _schema(fid, s, card)
        units = UNITS.get(fid) or {}
        if card.get("blocked") == BLOCKED_PIT or card["family"] == "E":
            pit_status = BLOCKED_PIT
            promote_ok = False
            skip_reason = "block_trading has TRADE_DATE only; no ANN/available_at"
            eligible = DIAGNOSTIC_ONLY
        elif not present:
            pit_status = BLOCKED_DATA
            promote_ok = False
            skip_reason = "column absent; no fill-0"
            eligible = DIAGNOSTIC_ONLY
        elif v == 0:
            pit_status = BLOCKED_DATA
            promote_ok = False
            skip_reason = "all missing; no fill-0"
            eligible = DIAGNOSTIC_ONLY
        elif nunique <= 1:
            pit_status = "PIT_FAIL_DEGENERATE"
            promote_ok = False
            skip_reason = "degenerate constant; no fill-0"
            eligible = DIAGNOSTIC_ONLY
        elif cov < PIT_MIN_FINITE:
            pit_status = "PIT_FAIL_COVERAGE"
            promote_ok = False
            skip_reason = f"finite_share={cov:.4f} < {PIT_MIN_FINITE}"
            eligible = DIAGNOSTIC_ONLY
        elif years_gap:
            pit_status = "PIT_FAIL_YEAR_GAP"
            promote_ok = False
            skip_reason = f"year coverage < {PIT_YEAR_GAP} in {years_gap}"
            eligible = DIAGNOSTIC_ONLY
        elif market_level:
            pit_status = "PIT_FAIL_MARKET_LEVEL"
            promote_ok = False
            skip_reason = "within-day unique <= 1 (market-level)"
            eligible = DIAGNOSTIC_ONLY
        else:
            pit_status = "PIT_PASS"
            promote_ok = True
            skip_reason = None
            eligible = PIT_ELIGIBLE_MODEL
        rec = {
            "factor_id": fid,
            "family": card["family"],
            "formula": card.get("formula"),
            "pit_rule": card.get("pit_rule"),
            "pit_status": pit_status,
            "model_eligible": eligible,
            "promote_ok": promote_ok,
            "skip_reason": skip_reason,
            "n": n,
            "n_valid": v,
            "coverage": cov,
            "n_unique": nunique,
            "median_within_day_unique": day_nu,
            "market_level": market_level,
            "year_coverage": year,
            "years_gap": years_gap,
            "schema": schema,
            "units": units,
            "filled_with_zero": False,
            "available_at_rule": {
                "A_B_C": "D close",
                "D": "next_trading_day(ANN_DT); valuation.available_at",
                "E": "TRADE_DATE diagnostic only",
                "F": "daily_margin.available_at",
            }[card["family"] if card["family"] in {"D", "E", "F"} else "A_B_C"],
        }
        dest = FACTORS_DIR / fid
        dest.mkdir(parents=True, exist_ok=True)
        dump_json(dest / "schema.json", schema)
        dump_json(
            dest / "coverage.json",
            {"n": n, "n_valid": v, "coverage": cov, "year": year, "years_gap": years_gap},
        )
        dump_json(dest / "units.json", units)
        dump_json(dest / "pit.json", rec)
        rows.append(rec)
    frame = pd.DataFrame(
        [
            {
                k: r[k]
                for k in (
                    "factor_id",
                    "family",
                    "pit_status",
                    "model_eligible",
                    "promote_ok",
                    "n",
                    "n_valid",
                    "coverage",
                    "n_unique",
                    "skip_reason",
                )
            }
            for r in rows
        ]
    )
    PIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PIT_CSV, index=False)
    payload = {
        "status": "DONE",
        "round": 21,
        "n_factors": len(rows),
        "pit_pass": [r["factor_id"] for r in rows if r["pit_status"] == "PIT_PASS"],
        "blocked_pit": [r["factor_id"] for r in rows if r["pit_status"] == BLOCKED_PIT],
        "blocked_data": [r["factor_id"] for r in rows if r["pit_status"] == BLOCKED_DATA],
        "fail_coverage": [r["factor_id"] for r in rows if r["pit_status"] == "PIT_FAIL_COVERAGE"],
        "fail_other": [
            r["factor_id"]
            for r in rows
            if r["pit_status"] not in {"PIT_PASS", BLOCKED_PIT, BLOCKED_DATA, "PIT_FAIL_COVERAGE"}
        ],
        "min_finite": PIT_MIN_FINITE,
        "filled_with_zero": False,
        "construct": {"n_rows": construct.get("n_rows"), "computed": construct.get("computed")},
        "rows": rows,
    }
    dump_json(PIT_JSON, payload)
    lines = [
        "# Round21 · PIT / schema / 覆盖 / 单位",
        "",
        f"PIT_PASS=`{payload['pit_pass']}`",
        f"BLOCKED_PIT=`{payload['blocked_pit']}`",
        f"BLOCKED_DATA=`{payload['blocked_data']}`",
        f"PIT_FAIL_COVERAGE=`{payload['fail_coverage']}`",
        f"other=`{payload['fail_other']}`",
        "",
        f"min_finite=`{PIT_MIN_FINITE}` filled_with_zero=`False`。E 家族无公告时点，禁模型。",
        "",
    ]
    for r in rows:
        lines.append(
            f"- `{r['factor_id']}` {r['pit_status']} cov=`{r['coverage']:.4f}` "
            f"valid=`{r['n_valid']}` eligible=`{r['model_eligible']}` units=`{r['units']}`"
        )
    write_md(PIT_MD, "\n".join(lines) + "\n")
    return payload
