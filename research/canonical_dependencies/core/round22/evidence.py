"""单因子 RankIC。样本过少标 NOT_IDENTIFIABLE。两个标签分开。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND22_ROOT
from ..round21.panel import load_candidate_keys
from .common import dump_json, load_json, write_csv, write_md
from .contract import KEYS, LABELS, MD, NOT_IDENTIFIABLE, PIT_PASS

IC_CSV = ROUND22_ROOT / "ic" / "ic_results.csv"


def _daily_ic(frame: pd.DataFrame, factor: str, label: str) -> dict[str, Any]:
    work = frame[[ "trade_date", factor, label]].copy()
    work[factor] = pd.to_numeric(work[factor], errors="coerce")
    work[label] = pd.to_numeric(work[label], errors="coerce")
    ics = []
    by_year: dict[str, list[float]] = {}
    for day, block in work.groupby("trade_date", sort=False):
        sub = block.dropna()
        if len(sub) < 5:
            continue
        ic = sub[factor].corr(sub[label], method="spearman")
        if ic is None or not np.isfinite(ic):
            continue
        ics.append(float(ic))
        by_year.setdefault(str(pd.Timestamp(day).year), []).append(float(ic))
    arr = np.array(ics, dtype=float)
    if len(arr) < 60:
        return {"status": NOT_IDENTIFIABLE, "n_days": int(len(arr)), "mean": None, "icir": None, "by_year": {}}
    return {
        "status": "OK",
        "n_days": int(len(arr)),
        "mean": float(arr.mean()),
        "icir": float(arr.mean() / arr.std(ddof=1)) if arr.std(ddof=1) > 0 else None,
        "by_year": {y: float(np.mean(v)) for y, v in by_year.items()},
    }


def run_evidence(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and IC_CSV.is_file() and MD["12"].is_file():
        existing = load_json(ROUND22_ROOT / "ic" / "ic.json")
        if existing.get("status") == "DONE":
            return existing
    pit = load_json(ROUND22_ROOT / "pit" / "pit.json")
    rows_in = pit.get("rows") or []
    panel = pd.read_parquet(ROUND22_ROOT / "data" / "factor_panel.parquet")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["instrument_id"] = panel["instrument_id"].astype(str)
    labels = load_candidate_keys()
    keep = [c for c in [*KEYS, *LABELS] if c in labels.columns]
    base = panel.merge(labels[keep], on=KEYS, how="left")
    out_rows = []
    for row in rows_in:
        fid = row["factor_id"]
        record = {"factor_id": fid, "pit_status": row.get("pit_status"), "profile": row.get("profile")}
        if fid not in base.columns or row.get("pit_status") != PIT_PASS:
            record["status"] = "SKIPPED"
            record["reason"] = "没有数值或未过 PIT，不把阻断写成 IC"
            out_rows.append(record)
            continue
        for label in LABELS:
            if label not in base.columns:
                record[label] = {"status": NOT_IDENTIFIABLE, "reason": "label missing"}
                continue
            ic = _daily_ic(base, fid, label)
            record[label] = ic
            dest = ROUND22_ROOT / "factors" / fid
            dump_json(dest / f"ic_{label}.json", ic)
        record["status"] = "DONE"
        out_rows.append(record)
        dump_json(ROUND22_ROOT / "factors" / fid / "ic.json", record)
    payload = {"status": "DONE", "n": len(out_rows), "n_ic": int(sum(r.get("status") == "DONE" for r in out_rows))}
    dump_json(ROUND22_ROOT / "ic" / "ic.json", payload | {"rows": out_rows})
    flat = []
    for r in out_rows:
        flat.append(
            {
                "factor_id": r["factor_id"],
                "profile": r.get("profile"),
                "status": r.get("status"),
                "e2_mean": (r.get("label_e2_net") or {}).get("mean") if isinstance(r.get("label_e2_net"), dict) else None,
                "e2_icir": (r.get("label_e2_net") or {}).get("icir") if isinstance(r.get("label_e2_net"), dict) else None,
                "e2_days": (r.get("label_e2_net") or {}).get("n_days") if isinstance(r.get("label_e2_net"), dict) else None,
                "e2_status": (r.get("label_e2_net") or {}).get("status") if isinstance(r.get("label_e2_net"), dict) else r.get("reason"),
                "exit_mean": (r.get("label_exit_rule_net") or {}).get("mean") if isinstance(r.get("label_exit_rule_net"), dict) else None,
                "exit_icir": (r.get("label_exit_rule_net") or {}).get("icir") if isinstance(r.get("label_exit_rule_net"), dict) else None,
                "exit_days": (r.get("label_exit_rule_net") or {}).get("n_days") if isinstance(r.get("label_exit_rule_net"), dict) else None,
                "exit_status": (r.get("label_exit_rule_net") or {}).get("status") if isinstance(r.get("label_exit_rule_net"), dict) else None,
            }
        )
    write_csv(IC_CSV, flat)
    write_md(MD["12"], f"# Round22 · 12 单因子证据\n\nIC 行数={payload['n_ic']}。日样本少于 60 标 `{NOT_IDENTIFIABLE}`。Five 标签 `label_e2_net` 与 Ten 标签 `label_exit_rule_net` 分开。\n")
    return payload
