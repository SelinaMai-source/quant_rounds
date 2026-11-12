"""PIT / 覆盖。正式晋级只接受 STRICT_PIT。V3 DUPLICATE 不伪造成新因子。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_3_ROOT
from .common import dump_json, write_md
from .construct import CONSTRUCT_JSON, PANEL_PATH, run_construct
from .contract import BLOCKED_PIT, DUPLICATE, KEYS, MD, NEW_FACTOR_IDS, PIT_MIN_FINITE, RESEARCH_ONLY, STRICT_PIT
from .hypotheses import HYPOTHESES

PIT_JSON = ROUND21_3_ROOT / "pit" / "pit.json"
PIT_CSV = ROUND21_3_ROOT / "new_factor_PIT.csv"


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


def run_pit(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and PIT_JSON.is_file() and PIT_CSV.is_file() and MD["05"].is_file():
        from .common import load_json

        existing = load_json(PIT_JSON)
        if existing.get("status") == "DONE":
            return existing
    run_construct(resume=True)
    from .common import load_json

    construct = load_json(CONSTRUCT_JSON)
    dups = {r["new"]: r for r in (construct.get("duplicates") or []) if r.get("verdict") == DUPLICATE}
    if not PANEL_PATH.is_file():
        raise RuntimeError("construct panel missing")
    panel = pd.read_parquet(PANEL_PATH)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    rows = []
    pit_pass = []
    blocked = []
    for fid in NEW_FACTOR_IDS:
        present = fid in panel.columns
        s = pd.to_numeric(panel[fid], errors="coerce") if present else pd.Series(np.nan, index=panel.index)
        n = int(len(s))
        v = int(np.isfinite(s.to_numpy(dtype=float)).sum())
        cov = (v / n) if n else None
        ycov = _year_cov(panel, fid) if present else {}
        year_gap = False
        covs = [x.get("coverage") for x in ycov.values() if x.get("coverage") is not None]
        if covs and (max(covs) - min(covs)) > 0.50:
            year_gap = True
        status = STRICT_PIT
        reason = ""
        if fid in dups:
            status = DUPLICATE
            reason = f"formula ≡ {dups[fid].get('old')}"
        elif not present or v == 0:
            status = BLOCKED_PIT
            reason = "not computed or all-NA"
            blocked.append(fid)
        elif fid.startswith("r21_u"):
            # 触板事件：未开板不填 0，覆盖按 50% 事件门槛
            status = STRICT_PIT if (cov or 0) >= 0.50 else RESEARCH_ONLY
            if status == RESEARCH_ONLY:
                reason = f"minute/touch coverage {cov:.3f} < 0.50"
        elif cov is not None and cov < PIT_MIN_FINITE:
            status = RESEARCH_ONLY
            reason = f"coverage {cov:.3f} < {PIT_MIN_FINITE}"
            late = [ycov[str(y)]["coverage"] for y in range(2021, 2027) if str(y) in ycov and ycov[str(y)].get("coverage") is not None]
            if late and (sum(late) / len(late)) >= 0.75:
                reason += "; STRICT_PIT_SUBPERIOD_EVIDENCE 2021+ (do not vs full-history v5)"
        elif year_gap and fid.startswith("r21_h"):
            status = RESEARCH_ONLY
            reason = "year gap"
        if status == STRICT_PIT:
            pit_pass.append(fid)
        rows.append(
            {
                "factor_id": fid,
                "present": present,
                "n": n,
                "valid": v,
                "coverage": cov,
                "year_coverage": ycov,
                "pit_status": status,
                "reason": reason,
            }
        )
    pd.DataFrame(rows).to_csv(PIT_CSV, index=False)
    payload = {
        "status": "DONE",
        "n_pass": len(pit_pass),
        "pit_pass": pit_pass,
        "blocked": blocked,
        "duplicates": list(dups),
        "rows": rows,
        "formal_requires": STRICT_PIT,
    }
    dump_json(PIT_JSON, payload)
    write_md(MD["05"], f"# Round21.3 · 05 PIT\n\nPIT_PASS=`{pit_pass}` blocked=`{blocked}` dups=`{list(dups)}` 全池 80% 不是唯一门槛。\n")
    return payload
