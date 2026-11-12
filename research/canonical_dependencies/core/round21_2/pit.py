"""PIT / 覆盖。全池 80% 不是唯一门槛。禁止填 0。正式晋级只接受 STRICT_PIT。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_2_ROOT
from .common import dump_json, write_md
from .construct import PANEL_PATH, run_construct
from .contract import (
    BLOCKED_DATA,
    BLOCKED_PIT,
    COMPLETE,
    KEYS,
    MD,
    NEW_FACTOR_IDS,
    PIT_ELIGIBLE_MODEL,
    PIT_MIN_FINITE,
    RESEARCH_ONLY,
    STRICT_PIT_VALID,
)
from .hypotheses import HYPOTHESES

PIT_JSON = ROUND21_2_ROOT / "pit" / "pit.json"
PIT_CSV = ROUND21_2_ROOT / "new_factor_PIT.csv"


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
    if not PANEL_PATH.is_file():
        raise RuntimeError("construct panel missing")
    panel = pd.read_parquet(PANEL_PATH)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    cards = {c["factor_id"]: c for c in HYPOTHESES}
    rows = []
    for fid in NEW_FACTOR_IDS:
        present = fid in panel.columns
        s = pd.to_numeric(panel[fid], errors="coerce") if present else pd.Series(np.nan, index=panel.index)
        strict_col = None
        for cand in (f"{fid}_strict", f"{fid.split('_')[0]}_{fid.split('_')[1]}_strict", f"r21_{fid.split('_')[1]}_strict"):
            if cand in panel.columns:
                strict_col = cand
                break
        # G 用 r21_g1_strict 等
        fam = fid.split("_")[1]
        gstrict = f"r21_{fam}_strict"
        if fam.startswith("g") and f"r21_{fam}_strict" in panel.columns:
            strict_col = f"r21_{fam}_strict"
        if f"r21_{fid[4:6]}_strict" in panel.columns:
            strict_col = f"r21_{fid[4:6]}_strict"
        # explicit map
        smap = {
            "r21_g1_avg_ticket_value": "r21_g1_strict",
            "r21_g2_large_ticket_asym": "r21_g2_strict",
            "r21_g3_order_count_imbalance": "r21_g3_strict",
            "r21_g4_large_ticket_conc_gap": "r21_g4_strict",
            "r21_h1_late_flow_dominance": "r21_h1_strict",
            "r21_h2_large_small_flow_div": "r21_h1_strict",
            "r21_h3_flow_dir_disagree": "r21_h1_strict",
            "r21_h4_large_flow_persist": "r21_h1_strict",
        }
        if fid in smap and smap[fid] in panel.columns:
            strict_col = smap[fid]
        n = int(len(panel))
        finite = np.isfinite(s.to_numpy(dtype=float))
        v = int(finite.sum())
        cov = (v / n) if n else 0.0
        if strict_col and strict_col in panel.columns:
            sm = pd.to_numeric(panel[strict_col], errors="coerce").fillna(0).gt(0.5)
            strict_s = s.where(sm)
            sv = int(np.isfinite(strict_s.to_numpy(dtype=float)).sum())
            strict_cov = sv / n if n else 0.0
            appl_n = int(sm.sum())
            cond_cov = (sv / appl_n) if appl_n else None
            true_miss = 1.0 - (sv / appl_n) if appl_n else 1.0
        else:
            strict_s = s
            sv = v
            strict_cov = cov
            appl_n = v
            cond_cov = 1.0 if v else None
            true_miss = 0.0 if present else 1.0
        year = _year_cov(panel.assign(**{fid: strict_s}), fid) if present else {}
        years_ok = [y for y, rec in year.items() if (rec.get("coverage") or 0) >= 0.5]
        card = cards.get(fid) or {}
        if not present:
            status = BLOCKED_DATA
            eligible = False
        elif fid.startswith("r21_h") and strict_cov < PIT_MIN_FINITE:
            # 回填导致 STRICT 样本缩短
            status = RESEARCH_ONLY if cov >= PIT_MIN_FINITE else "PIT_FAIL_COVERAGE"
            eligible = False
        elif fid.startswith("r21_g") and strict_cov < PIT_MIN_FINITE and cov >= PIT_MIN_FINITE:
            status = RESEARCH_ONLY
            eligible = False
        elif strict_cov >= PIT_MIN_FINITE and len(years_ok) >= 6:
            status = "PIT_PASS"
            eligible = True
        elif strict_cov >= 0.5 and cond_cov is not None and cond_cov >= 0.9 and appl_n >= 0.3 * n:
            status = "PIT_PASS"
            eligible = True
        elif not present:
            status = BLOCKED_DATA
            eligible = False
        else:
            status = "PIT_FAIL_COVERAGE"
            eligible = False
        rows.append(
            {
                "factor_id": fid,
                "family": card.get("family"),
                "present": present,
                "universe_coverage": cov,
                "strict_coverage": strict_cov,
                "conditional_coverage": cond_cov,
                "true_missing_rate": true_miss,
                "applicable_n": appl_n,
                "pit_status": status,
                "eligible": eligible,
                "deployment": "XGB_NATIVE_NA_PLUS_FLAG",
                "filled_with_zero": False,
                "year": year,
                "strict_pit": STRICT_PIT_VALID if eligible else status,
            }
        )
        d = ROUND21_2_ROOT / "factors" / fid
        d.mkdir(parents=True, exist_ok=True)
        dump_json(d / "pit.json", rows[-1])
    PIT_CSV_PARENT = PIT_CSV.parent
    PIT_CSV_PARENT.mkdir(parents=True, exist_ok=True)
    slim = [{k: r[k] for k in r if k != "year"} for r in rows]
    pd.DataFrame(slim).to_csv(PIT_CSV, index=False)
    pass_ids = [r["factor_id"] for r in rows if r["pit_status"] == "PIT_PASS"]
    blocked = [r["factor_id"] for r in rows if r["pit_status"] in {BLOCKED_DATA, BLOCKED_PIT}]
    payload = {
        "status": "DONE",
        "n": len(rows),
        "pit_pass": pass_ids,
        "n_pass": len(pass_ids),
        "blocked": blocked,
        "rows": rows,
        "whole_universe_80_not_sole_gate": True,
        "filled_with_zero": False,
    }
    dump_json(PIT_JSON, payload)
    mask_cols = KEYS[:]
    masks = panel[KEYS].copy()
    for r in rows:
        fid = r["factor_id"]
        if fid not in panel.columns:
            continue
        val = pd.to_numeric(panel[fid], errors="coerce")
        masks[f"{fid}__applicability"] = val.notna().astype("int8")
        masks[f"{fid}__missingness"] = val.isna().astype("int8")
        sc = None
        for cand in (f"{fid}_strict", f"r21_{fid.split('_')[1]}_strict"):
            if cand in panel.columns:
                sc = cand
                break
        if sc:
            masks[f"{fid}__strict_pit"] = pd.to_numeric(panel[sc], errors="coerce").fillna(0).gt(0.5).astype("int8")
        else:
            masks[f"{fid}__strict_pit"] = val.notna().astype("int8")
        mask_cols.extend([f"{fid}__applicability", f"{fid}__missingness", f"{fid}__strict_pit"])
    from .construct import MASK_PATH

    MASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    masks.to_parquet(MASK_PATH, index=False)
    payload["coverage_masks"] = str(MASK_PATH)
    dump_json(PIT_JSON, payload)
    lines = [
        "# Round21.2 · 05 PIT 与覆盖",
        "",
        f"PIT_PASS N=`{len(pass_ids)}` `{pass_ids}`",
        f"BLOCKED=`{blocked}`",
        "全池 80% 不是唯一门槛。禁止填 0。正式晋级只接受 STRICT_PIT。",
        "",
        "| factor | universe | strict | cond | true_miss | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['factor_id']}` | {r['universe_coverage']} | {r['strict_coverage']} | {r['conditional_coverage']} | {r['true_missing_rate']} | {r['pit_status']} |"
        )
    write_md(MD["05"], "\n".join(lines))
    return payload
