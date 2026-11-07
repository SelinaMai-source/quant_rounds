"""PIT / 覆盖。正式晋级只接受 STRICT_PIT。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_4_ROOT
from .common import dump_json, write_md
from .construct import CONSTRUCT_JSON, PANEL_PATH, run_construct
from .contract import BLOCKED_PIT, DUPLICATE, MD, PIT_MIN_FINITE, RESEARCH_ONLY, STRICT_PIT, WXYZ_FACTOR_IDS
from .hypotheses import HYPOTHESES

PIT_JSON = ROUND21_4_ROOT / "pit" / "pit.json"
PIT_CSV = ROUND21_4_ROOT / "new_factor_PIT.csv"


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
    if resume and PIT_JSON.is_file() and PIT_CSV.is_file() and MD["22"].is_file():
        from .common import load_json

        existing = load_json(PIT_JSON)
        if existing.get("status") == "DONE":
            return existing
    run_construct(resume=True)
    from .common import load_json

    construct = load_json(CONSTRUCT_JSON)
    dups = {r["new"]: r for r in (construct.get("duplicates") or []) if r.get("verdict") == DUPLICATE}
    if not PANEL_PATH.is_file():
        payload = {"status": "BLOCKED", "reason": "construct panel missing", "pit_pass": [], "blocked": list(WXYZ_FACTOR_IDS)}
        dump_json(PIT_JSON, payload)
        return payload
    panel = pd.read_parquet(PANEL_PATH)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    rows = []
    pit_pass = []
    blocked = []
    for fid in WXYZ_FACTOR_IDS:
        present = fid in panel.columns
        s = pd.to_numeric(panel[fid], errors="coerce") if present else pd.Series(np.nan, index=panel.index)
        n = int(len(s))
        v = int(np.isfinite(s.to_numpy(dtype=float)).sum())
        cov = (v / n) if n else None
        ycov = _year_cov(panel, fid) if present else {}
        status = STRICT_PIT
        reason = ""
        if fid in dups:
            status = DUPLICATE
            reason = f"formula ≡ {dups[fid].get('old')}"
        elif not present or v == 0:
            status = BLOCKED_PIT
            reason = "not computed or all-NA"
            blocked.append(fid)
        elif cov is not None and cov < PIT_MIN_FINITE:
            status = RESEARCH_ONLY
            reason = f"coverage {cov:.3f} < {PIT_MIN_FINITE}"
        if status == STRICT_PIT:
            pit_pass.append(fid)
        rows.append({"factor_id": fid, "n": n, "valid": v, "coverage": cov, "pit_status": status, "reason": reason, "year": ycov})
    pd.DataFrame([{k: r[k] for k in ("factor_id", "n", "valid", "coverage", "pit_status", "reason")} for r in rows]).to_csv(PIT_CSV, index=False)
    payload = {"status": "DONE", "n_pass": len(pit_pass), "pit_pass": pit_pass, "blocked": blocked, "rows": rows, "duplicates": list(dups)}
    dump_json(PIT_JSON, payload)
    write_md(MD["22"], f"# Round21.4 · 22 PIT\n\nPASS=`{pit_pass}` blocked=`{blocked}` N_pass=`{len(pit_pass)}`\n")
    return payload
