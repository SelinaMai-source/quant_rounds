"""evaluate_factor Gate0–13 + IC/Rolling。未过门 SKIPPED/FAIL。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_4_ROOT
from ..round20.metrics import icir_of, pearson_ic, rank_ic
from .common import dump_json, finite_or_none, load_json, write_md
from .construct import CONSTRUCT_JSON, PANEL_PATH
from .contract import KEYS, LABELS, MD, WXYZ_FACTOR_IDS
from .evaluate import evaluate_factor
from .hypotheses import HYPOTHESES
from .pit import PIT_JSON, run_pit

EV_JSON = ROUND21_4_ROOT / "evidence" / "evidence.json"
IC_CSV = ROUND21_4_ROOT / "new_factor_IC.csv"
ROLL_CSV = ROUND21_4_ROOT / "rolling_factor_evidence.csv"
GATES = (
    "FactorDefinition",
    "DataAudit",
    "PIT",
    "IC",
    "Rolling",
    "Shape",
    "TopTail",
    "Redundancy",
    "ConditionalIC",
    "Combination",
    "ModelIncrementality",
    "PortfolioEvidence",
    "PromotionGate",
    "HEAD_SPECIFIC_FOUR_WAY",
    "DATA_RECOVERY_BEFORE_COVERAGE_REJECT",
)


def _daily_ic(frame: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(frame[label], errors="coerce")
    x = pd.to_numeric(frame[col], errors="coerce")
    ready = pd.to_datetime(frame.get("actual_exit_date" if label == "label_exit_rule_net" else "next2_date"), errors="coerce")
    for day, idx in frame.groupby("trade_date", sort=True).groups.items():
        xv = x.loc[idx].to_numpy(dtype=float)
        yv = y.loc[idx].to_numpy(dtype=float)
        mask = np.isfinite(xv) & np.isfinite(yv)
        if int(mask.sum()) < 5:
            continue
        if int(np.unique(xv[mask]).size) < 2 or int(np.unique(yv[mask]).size) < 2:
            continue
        xs, ys = pd.Series(xv[mask]), pd.Series(yv[mask])
        rows.append(
            {
                "trade_date": pd.Timestamp(day),
                "n_valid": int(mask.sum()),
                "ic": pearson_ic(xs, ys),
                "rankic": rank_ic(xs, ys),
                "ready_at": pd.Timestamp(ready.loc[idx].max()) if ready.loc[idx].notna().any() else pd.NaT,
            }
        )
    return pd.DataFrame(rows)


def _roll(daily: pd.DataFrame, window: int) -> dict[str, Any]:
    if daily.empty or "rankic" not in daily.columns:
        return {"window": window, "rankic_mean": None, "icir": None, "pos_ratio": None}
    s = pd.to_numeric(daily["rankic"], errors="coerce").dropna()
    if len(s) < max(20, window // 4):
        return {"window": window, "rankic_mean": float(s.mean()) if len(s) else None, "icir": icir_of(s) if len(s) else None, "pos_ratio": float((s > 0).mean()) if len(s) else None, "note": "short"}
    return {"window": window, "rankic_mean": float(s.mean()), "icir": icir_of(s), "pos_ratio": float((s > 0).mean())}


def run_evidence(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and EV_JSON.is_file() and MD["23"].is_file():
        existing = load_json(EV_JSON)
        if existing.get("status") == "DONE":
            return existing
    pit = run_pit(resume=True)
    construct = load_json(CONSTRUCT_JSON)
    if not PANEL_PATH.is_file():
        payload = {"status": "BLOCKED", "reason": "no panel", "n_factory": 0}
        dump_json(EV_JSON, payload)
        return payload
    panel = pd.read_parquet(PANEL_PATH)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    from ..round21.panel import load_candidate_keys

    keys = load_candidate_keys()
    for col in LABELS + ("next2_date", "actual_exit_date"):
        if col in keys.columns and col not in panel.columns:
            panel = panel.merge(keys[["trade_date", "instrument_id", col]], on=KEYS, how="left")
    ic_rows = []
    roll_rows = []
    factory = []
    dest_root = ROUND21_4_ROOT / "factors"
    for fid in WXYZ_FACTOR_IDS:
        fdir = dest_root / fid
        fdir.mkdir(parents=True, exist_ok=True)
        if fid not in panel.columns:
            dump_json(fdir / "construct.json", {"status": "MISSING"})
            factory.append(evaluate_factor(fid, through="PromotionGate", persist=True))
            continue
        s = pd.to_numeric(panel[fid], errors="coerce")
        dump_json(fdir / "construct.json", {"status": "OK", "n_finite": int(np.isfinite(s.to_numpy(dtype=float)).sum())})
        prow = next((r for r in (pit.get("rows") or []) if r.get("factor_id") == fid), {})
        dump_json(fdir / "pit.json", {"status": prow.get("pit_status") or "MISSING", "pit_status": prow.get("pit_status"), "coverage": prow.get("coverage")})
        for label in LABELS:
            if label not in panel.columns:
                continue
            daily = _daily_ic(panel, fid, label)
            mean_ic = finite_or_none(daily["rankic"].mean()) if not daily.empty else None
            ic_rows.append({"factor_id": fid, "label": label, "rankic_mean": mean_ic, "n_days": int(len(daily)), "icir": icir_of(pd.to_numeric(daily["rankic"], errors="coerce")) if not daily.empty else None})
            dump_json(fdir / "ic.json", {"label": label, "rankic_mean": mean_ic, "n_days": int(len(daily))})
            for w in (60, 120, 252):
                rr = _roll(daily, w)
                roll_rows.append({"factor_id": fid, "label": label, **rr})
            dump_json(fdir / "rolling.json", {"windows": [60, 120, 252]})
            if not daily.empty:
                dump_json(fdir / "shape.json", {"skew": finite_or_none(s.skew()), "status": "OK"})
                dump_json(fdir / "top_tail.json", {"p90_minus_p10": finite_or_none(s.quantile(0.9) - s.quantile(0.1)), "status": "OK"})
        factory.append(evaluate_factor(fid, through="PromotionGate", persist=True))
    if ic_rows:
        pd.DataFrame(ic_rows).to_csv(IC_CSV, index=False)
    if roll_rows:
        pd.DataFrame(roll_rows).to_csv(ROLL_CSV, index=False)
    from .correlate import run_correlate

    corr = run_correlate(resume=True)
    payload = {"status": "DONE", "n_factory": len(factory), "gates": list(GATES), "ic_csv": str(IC_CSV), "rows": ic_rows, "n_nearest": corr.get("n")}
    dump_json(EV_JSON, payload)
    write_md(MD["23"], f"# Round21.4 · 23 证据\n\nfactory N=`{len(factory)}` IC rows=`{len(ic_rows)}`\n")
    return payload
