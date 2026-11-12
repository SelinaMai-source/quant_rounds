"""evaluate_factor Gate0–13。不编造 Score。未过门 SKIPPED/FAIL。写 factors/<id>/ 产物后调用工厂。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_3_ROOT
from ..round20.metrics import icir_of, pearson_ic, rank_ic
from .common import dump_json, finite_or_none, load_json, write_md
from .construct import CONSTRUCT_JSON, PANEL_PATH
from .contract import KEYS, LABELS, MD, NEW_FACTOR_IDS
from .evaluate import evaluate_factor
from .hypotheses import HYPOTHESES
from .panels_local import load_research_panel
from .pit import PIT_JSON, run_pit

EV_JSON = ROUND21_3_ROOT / "evidence" / "evidence.json"
IC_CSV = ROUND21_3_ROOT / "new_factor_IC.csv"
ROLL_CSV = ROUND21_3_ROOT / "rolling_factor_evidence.csv"
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
        return {
            "window": window,
            "rankic_mean": float(s.mean()) if len(s) else None,
            "icir": icir_of(s) if len(s) else None,
            "pos_ratio": float((s > 0).mean()) if len(s) else None,
            "note": "short",
        }
    roll = s.rolling(window, min_periods=max(10, window // 5)).mean()
    return {
        "window": window,
        "rankic_mean": float(s.mean()),
        "roll_last": float(roll.dropna().iloc[-1]) if roll.notna().any() else None,
        "icir": icir_of(s),
        "pos_ratio": float((s > 0).mean()),
    }


def _write_package(fid: str, panel: pd.DataFrame, pit_row: dict[str, Any], ic_rows: list[dict[str, Any]]) -> dict[str, Any]:
    dest = ROUND21_3_ROOT / "factors" / fid
    dest.mkdir(parents=True, exist_ok=True)
    card = next((c for c in HYPOTHESES if c["factor_id"] == fid), {})
    construct = load_json(CONSTRUCT_JSON)
    present = fid in panel.columns
    dump_json(dest / "construct.json", {"factor_id": fid, "present": present, "card": card, "status": "PASS" if present else "FAIL"})
    dump_json(dest / "pit.json", {**pit_row, "status": pit_row.get("pit_status")})
    ic_e2 = next((r for r in ic_rows if r.get("label") == "label_e2_net"), {})
    dump_json(dest / "ic.json", {"status": "PASS" if ic_e2.get("raw_rankic") is not None else "SKIPPED", **ic_e2})
    dump_json(dest / "rolling.json", {"status": "PASS" if ic_e2.get("icir") is not None else "SKIPPED", "icir": ic_e2.get("icir")})
    shape = {"status": "SKIPPED", "quintile": None}
    toptail = {"status": "SKIPPED"}
    if present and "label_e2_net" in panel.columns:
        x = pd.to_numeric(panel[fid], errors="coerce")
        y = pd.to_numeric(panel["label_e2_net"], errors="coerce")
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) > 200:
            xx, yy = x[mask], y[mask]
            bins = pd.qcut(xx.rank(method="first"), 5, labels=False, duplicates="drop")
            means = yy.groupby(bins).mean()
            q = {int(i): float(means.loc[i]) for i in means.index}
            top = float(yy.loc[bins == bins.max()].mean()) if len(means) else None
            bot = float(yy.loc[bins == bins.min()].mean()) if len(means) else None
            shape = {"status": "PASS", "quintile": q}
            toptail = {"status": "PASS", "top": top, "bottom": bot, "spread": None if top is None or bot is None else top - bot}
    dump_json(dest / "shape.json", shape)
    dump_json(dest / "top_tail.json", toptail)
    dump_json(dest / "redundancy.json", {"status": "PENDING_CORRELATE"})
    dump_json(dest / "conditional.json", {"status": "PENDING"})
    dump_json(dest / "combination.json", {"status": "SKIPPED", "reason": "budget in interact"})
    dump_json(dest / "incrementality.json", {"status": "PENDING_TRAIN"})
    dump_json(dest / "portfolio.json", {"status": "PENDING_BACKTEST", "rows": []})
    dump_json(
        dest / "factory.json",
        {
            "factor_id": fid,
            "n_gates": 14,
            "recovery_before_reject": True,
            "duplicates": construct.get("duplicates"),
        },
    )
    try:
        factory = evaluate_factor(fid, through="PromotionGate", persist=True)
    except Exception as exc:
        factory = {"factor_id": fid, "status": "FAIL", "reason": str(exc)}
    return {
        "factor_id": fid,
        "steps": {k: (load_json(dest / f) or {}).get("status") for k, f in (
            ("DataAudit", "construct.json"),
            ("PIT", "pit.json"),
            ("IC", "ic.json"),
            ("Rolling", "rolling.json"),
            ("Shape", "shape.json"),
            ("TopTail", "top_tail.json"),
        )},
        "PromotionGate": factory.get("PromotionGate") or {"status": factory.get("status"), "reason": factory.get("reason")},
        "evaluate_status": factory.get("status"),
    }


def run_evidence(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and EV_JSON.is_file() and IC_CSV.is_file() and MD["06"].is_file():
        existing = load_json(EV_JSON)
        if existing.get("status") == "DONE" and existing.get("n_factory"):
            return existing
    pit = run_pit(resume=True)
    panel = load_research_panel(resume=True)
    pit_map = {r["factor_id"]: r for r in (pit.get("rows") or [])}
    ic_rows = []
    roll_rows = []
    for fid in NEW_FACTOR_IDS:
        if fid not in panel.columns:
            continue
        for lab in LABELS:
            if lab not in panel.columns:
                continue
            daily = _daily_ic(panel, fid, lab)
            dest = ROUND21_3_ROOT / "factors" / fid
            dest.mkdir(parents=True, exist_ok=True)
            if not daily.empty:
                daily.to_parquet(dest / f"daily_ic_{lab}.parquet", index=False)
            s = pd.to_numeric(daily["rankic"], errors="coerce").dropna() if not daily.empty else pd.Series(dtype=float)
            ic_rows.append(
                {
                    "factor_id": fid,
                    "label": lab,
                    "raw_ic": finite_or_none(pd.to_numeric(daily["ic"], errors="coerce").mean()) if not daily.empty else None,
                    "raw_rankic": finite_or_none(s.mean()) if len(s) else None,
                    "icir": icir_of(s) if len(s) else None,
                    "n_days": int(len(s)),
                }
            )
            for w, name in ((60, "ROUND21_INTERNAL_W60"), (120, "ROUND21_INTERNAL_W120"), (252, "ROUND21_INTERNAL_W252")):
                rr = _roll(daily, w)
                rr.update({"factor_id": fid, "label": lab, "window_name": name, "namespace": "ROUND21_INTERNAL"})
                roll_rows.append(rr)
    pd.DataFrame(ic_rows).to_csv(IC_CSV, index=False)
    pd.DataFrame(roll_rows).to_csv(ROLL_CSV, index=False)
    factories = []
    for fid in NEW_FACTOR_IDS:
        factories.append(_write_package(fid, panel, pit_map.get(fid) or {}, [r for r in ic_rows if r["factor_id"] == fid]))
    best = sorted(
        [r for r in ic_rows if r.get("label") == "label_e2_net" and r.get("raw_rankic") is not None],
        key=lambda r: abs(float(r["raw_rankic"])),
        reverse=True,
    )[:5]
    payload = {
        "status": "DONE",
        "n_factory": len(factories),
        "gates": list(GATES),
        "best_abs_rankic_e2": best,
        "rows": ic_rows,
        "factories": factories,
        "no_invented_score": True,
        "called_evaluate_factor": True,
    }
    dump_json(EV_JSON, payload)
    write_md(MD["06"], f"# Round21.3 · 06 单因子证据\n\nfactory N=`{len(factories)}` gates=0–13。已调用 `evaluate_factor`。禁止编造 Score。\n")
    write_md(MD["08"], "# Round21.3 · 08 非线性 / Top-tail\n\n五分位。不要求完美单调。见 factors/<id>/shape.json。\n")
    write_md(MD["10"], "# Round21.3 · 10 条件 IC\n\n条件切片见 factors/<id>/conditional.json。未切片标 PENDING/SKIPPED。\n")
    return payload
