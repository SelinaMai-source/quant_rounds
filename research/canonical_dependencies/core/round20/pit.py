"""对盘点 78 因子做 PIT / 覆盖。F2 不得晋级。缺失不填 0。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND20_ROOT
from .common import dump_json, load_json, write_md
from .contract import (
    COMPLETE_IC_YEARS,
    F1_NAME,
    F2_NAME,
    F3_NAME,
    FIVE_FEATURES,
    PIT_MD,
    PIT_MIN_FINITE,
    PIT_TIMESTAMP_UNVERIFIED,
    PIT_YEAR_GAP,
    R18_PIT_JSON,
    TEN_FEATURES,
)
from .freeze import require_frozen
from .inventory import NO_NEW_DEPENDENCY
from .panel import build_diag_panel, factor_names, load_registry

PIT_JSON = ROUND20_ROOT / "pit" / "pit.json"
PIT_CSV = ROUND20_ROOT / "pit" / "pit.csv"


def _year_cov(frame: pd.DataFrame, col: str) -> dict[str, Any]:
    years = pd.to_datetime(frame["trade_date"]).dt.year
    out: dict[str, Any] = {}
    s = pd.to_numeric(frame[col], errors="coerce") if col in frame.columns else pd.Series(np.nan, index=frame.index)
    for y, idx in years.groupby(years).groups.items():
        block = s.loc[idx]
        n = int(len(block))
        v = int(np.isfinite(block.to_numpy(dtype=float)).sum())
        out[str(int(y))] = {"n": n, "valid": v, "coverage": (v / n) if n else None}
    return out


def run_pit(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and PIT_JSON.is_file() and PIT_CSV.is_file() and PIT_MD.is_file():
        existing = load_json(PIT_JSON)
        if existing.get("status") == "DONE":
            return existing
    panel = build_diag_panel(resume=True)
    reg = load_registry()
    names = factor_names(reg)
    r18_prior = load_json(R18_PIT_JSON)
    r18_factors = (r18_prior.get("factors") or {}) if r18_prior else {}
    rows: list[dict[str, Any]] = []
    for rec in reg.to_dict(orient="records"):
        name = str(rec["name"])
        present = name in panel.columns
        s = pd.to_numeric(panel[name], errors="coerce") if present else pd.Series(dtype=float)
        n = int(len(panel))
        v = int(np.isfinite(s.to_numpy(dtype=float)).sum()) if present else 0
        cov = (v / n) if n else 0.0
        year = _year_cov(panel, name) if present else {}
        years_gap = [y for y, r in year.items() if int(y) in COMPLETE_IC_YEARS and (r.get("coverage") or 0) < PIT_YEAR_GAP]
        finite = s[np.isfinite(s.to_numpy(dtype=float))] if present else pd.Series(dtype=float)
        nunique = int(finite.nunique(dropna=True)) if len(finite) else 0
        # within-day variation
        day_nu = None
        market_level = False
        if present and nunique > 1:
            tmp = pd.DataFrame({"d": panel["trade_date"], "x": s})
            tmp = tmp.loc[np.isfinite(tmp["x"].to_numpy(dtype=float))]
            if len(tmp):
                day_nu = float(tmp.groupby("d")["x"].nunique().median())
                market_level = bool(day_nu <= 1.0)
        source = str(rec.get("source") or "")
        inherited = str(rec.get("pit") or "")
        if name == F2_NAME:
            pit_status = PIT_TIMESTAMP_UNVERIFIED
            promote_ok = False
            skip_reason = "PIT_TIMESTAMP_UNVERIFIED; 不得进可晋级模型"
        elif not present:
            pit_status = "MISSING_COLUMN"
            promote_ok = False
            skip_reason = "column absent after merge; no fill-0"
        elif nunique <= 1:
            pit_status = "PIT_FAIL_DEGENERATE"
            promote_ok = False
            skip_reason = "degenerate constant; no fill-0"
        elif cov < PIT_MIN_FINITE:
            pit_status = "PIT_FAIL_COVERAGE"
            promote_ok = False
            skip_reason = f"finite_share={cov:.4f} < {PIT_MIN_FINITE}"
        elif years_gap:
            pit_status = "PIT_FAIL_YEAR_GAP"
            promote_ok = False
            skip_reason = f"year coverage < {PIT_YEAR_GAP} in {years_gap}"
        elif source == "baseline_five":
            pit_status = "PIT_INHERITED_LOCKED"
            promote_ok = False
            skip_reason = "locked 14/13; 不是新假设，不作为新因子晋级"
        elif name in {F1_NAME, F3_NAME}:
            prior = (r18_factors.get(name) or {})
            inherited = str(prior.get("pit_status") or inherited or "")
            pit_status = "PIT_PASS"
            promote_ok = bool(cov >= PIT_MIN_FINITE and inherited != PIT_TIMESTAMP_UNVERIFIED)
            skip_reason = None if promote_ok else "R18 prior gate"
        else:
            pit_status = "PIT_PASS"
            promote_ok = True
            skip_reason = None
        if name == F2_NAME:
            promote_ok = False
        rows.append(
            {
                "name": name,
                "family": rec.get("family"),
                "source": source,
                "hypothesis_status": rec.get("hypothesis_status"),
                "present": present,
                "n": n,
                "n_valid": v,
                "coverage": cov,
                "nunique": nunique,
                "median_day_nunique": day_nu,
                "market_level": market_level,
                "year_gaps": years_gap,
                "inherited_pit": inherited,
                "pit_status": pit_status,
                "promotion_eligible": bool(promote_ok) and name != F2_NAME and source != "baseline_five",
                "f2_not_promotion_eligible": name == F2_NAME,
                "f1_f3_not_rebranded": name in {F1_NAME, F3_NAME},
                "in_five": name in FIVE_FEATURES,
                "in_ten": name in TEN_FEATURES,
                "no_new_dependency": name in NO_NEW_DEPENDENCY,
                "skip_reason": skip_reason,
                "available_at": rec.get("available_at"),
                "filled_with_zero": False,
            }
        )
    frame = pd.DataFrame(rows)
    PIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PIT_CSV, index=False)
    n_pass = int(frame["pit_status"].isin(["PIT_PASS", "PIT_INHERITED_LOCKED"]).sum())
    n_f2 = int((frame["name"] == F2_NAME).sum())
    n_eligible = int(frame["promotion_eligible"].sum())
    counts = frame["pit_status"].value_counts().to_dict()
    payload = {
        "status": "DONE",
        "n_total": int(len(frame)),
        "n_pass_or_locked": n_pass,
        "n_promotion_eligible": n_eligible,
        "n_f2": n_f2,
        "f2_promotion_eligible": False,
        "f1_f3_rebranded": False,
        "filled_with_zero": False,
        "min_finite": PIT_MIN_FINITE,
        "pit_status_counts": {str(k): int(v) for k, v in counts.items()},
        "market_level": frame.loc[frame["market_level"], "name"].tolist(),
        "fail": frame.loc[~frame["pit_status"].isin(["PIT_PASS", "PIT_INHERITED_LOCKED", PIT_TIMESTAMP_UNVERIFIED]), "name"].tolist(),
        "n_pit_pass": int((frame["pit_status"] == "PIT_PASS").sum()),
        "n_locked": int((frame["pit_status"] == "PIT_INHERITED_LOCKED").sum()),
        "n_unverified": int((frame["pit_status"] == PIT_TIMESTAMP_UNVERIFIED).sum()),
        "csv": str(PIT_CSV),
        "rows": rows,
        "note": "F2 PIT_TIMESTAMP_UNVERIFIED，不得晋级。F1/F3 不换名。锁定 14/13 不是新假设。",
    }
    dump_json(PIT_JSON, payload)
    lines = [
        "# 02 PIT",
        "",
        "对盘点 78 因子逐列覆盖 / 退化 / 日内变异。缺失不填 0。F2 不得晋级。F1/F3 不换名。",
        "",
        f"- n_total=`{payload['n_total']}` pass_or_locked=`{n_pass}` promotion_eligible=`{n_eligible}`",
        f"- F2 `{F2_NAME}` pit=`{PIT_TIMESTAMP_UNVERIFIED}` promotion_eligible=`False`",
        f"- F1/F3 rebranded=`False`",
        f"- filled_with_zero=`False` min_finite=`{PIT_MIN_FINITE}`",
        "",
        "## 状态计数",
        "",
    ]
    for k, v in sorted(counts.items()):
        lines.append(f"- `{k}` = {int(v)}")
    lines += ["", "## 市场层（日内 nunique 中位数 ≤1，截面 IC 将跳过）", ""]
    if payload["market_level"]:
        for n in payload["market_level"]:
            lines.append(f"- `{n}`")
    else:
        lines.append("无。")
    lines += ["", "## 明细（节选失败 / F2 / 锁定）", "", "| name | pit | eligible | coverage | reason |", "|---|---|---|---:|---|"]
    show = frame.loc[
        frame["name"].isin({F1_NAME, F2_NAME, F3_NAME})
        | frame["source"].eq("baseline_five")
        | ~frame["pit_status"].isin(["PIT_PASS", "PIT_INHERITED_LOCKED"])
    ]
    for rec in show.to_dict(orient="records"):
        lines.append(
            f"| `{rec['name']}` | `{rec['pit_status']}` | {rec['promotion_eligible']} | "
            f"{rec['coverage']:.4f} | {rec.get('skip_reason') or ''} |"
        )
    lines += ["", f"csv：`{PIT_CSV}`", ""]
    write_md(PIT_MD, "\n".join(lines))
    return payload


__all__ = ["PIT_CSV", "PIT_JSON", "run_pit"]
