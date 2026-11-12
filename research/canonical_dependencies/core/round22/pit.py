"""PIT、覆盖、日截面方差、常数日。不把阻断写成通过。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND22_ROOT
from .common import dump_json, load_json, write_csv, write_md
from .contract import (
    BLOCKED_ACCESS,
    BLOCKED_DATA,
    BLOCKED_PIT,
    BLOCKED_SEMANTICS,
    COMPOSITE_OF_KNOWN_FACTORS,
    CONSTANT_DAY_SHARE,
    CROSS_SECTIONALLY_DEGENERATE,
    DUPLICATE_EXPRESSION,
    FACTOR_SPECS,
    MD,
    PIT_MIN_FINITE,
    PIT_PASS,
    STRICT_PIT_SUBPERIOD_RESEARCH,
)

PIT_JSON = ROUND22_ROOT / "pit" / "pit.json"
PIT_CSV = ROUND22_ROOT / "pit" / "pit_coverage.csv"


def _screen(series: pd.Series, dates: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce")
    finite = float(values.notna().mean()) if len(values) else 0.0
    day = pd.DataFrame({"trade_date": dates, "v": values}).dropna()
    if day.empty:
        return {"finite_share": finite, "median_cs_var": 0.0, "constant_day_share": 1.0, "n_days": 0}
    stats = day.groupby("trade_date")["v"].agg(["std", "count", "nunique"])
    usable = stats.loc[stats["count"] >= 5]
    if usable.empty:
        return {"finite_share": finite, "median_cs_var": 0.0, "constant_day_share": 1.0, "n_days": int(stats.shape[0])}
    var = usable["std"].fillna(0).pow(2)
    constant = (usable["std"].fillna(0) < 1e-8) | (usable["nunique"] <= 1)
    return {
        "finite_share": finite,
        "median_cs_var": float(var.median()),
        "constant_day_share": float(constant.mean()),
        "n_days": int(usable.shape[0]),
    }


def run_pit(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and PIT_JSON.is_file() and PIT_CSV.is_file():
        existing = load_json(PIT_JSON)
        if existing.get("status") == "DONE":
            return existing
    panel_path = ROUND22_ROOT / "data" / "factor_panel.parquet"
    if not panel_path.is_file():
        raise RuntimeError("factor panel missing")
    panel = pd.read_parquet(panel_path)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    eq_path = ROUND22_ROOT / "construct" / "formula_equivalence.csv"
    eq = pd.read_csv(eq_path) if eq_path.is_file() else pd.DataFrame()
    eq_map = {str(r["factor_id"]): r for r in eq.to_dict(orient="records")} if not eq.empty else {}
    rows = []
    for spec in FACTOR_SPECS:
        fid = spec["factor_id"]
        prior = eq_map.get(fid) or {}
        prior_status = str(prior.get("status") or "")
        screen = _screen(panel[fid], panel["trade_date"]) if fid in panel.columns else {"finite_share": 0.0, "median_cs_var": 0.0, "constant_day_share": 1.0, "n_days": 0}
        label = spec.get("label")
        if prior_status in {DUPLICATE_EXPRESSION, BLOCKED_DATA, BLOCKED_ACCESS, BLOCKED_PIT, BLOCKED_SEMANTICS, STRICT_PIT_SUBPERIOD_RESEARCH}:
            status = prior_status
            reason = str(prior.get("reason") or prior_status)
            model_ok = False
        elif screen["constant_day_share"] >= CONSTANT_DAY_SHARE or screen["median_cs_var"] <= 0:
            status = CROSS_SECTIONALLY_DEGENERATE
            reason = "日截面方差过低或常数日比例过高"
            model_ok = False
        elif screen["finite_share"] < PIT_MIN_FINITE:
            status = "COVERAGE_BELOW_GATE"
            reason = f"有限值占比 {screen['finite_share']:.3f} < {PIT_MIN_FINITE}"
            if fid.startswith("r22_e"):
                reason += "；停牌样本可能过稀，不下强结论。从未停牌保持 NA。"
            model_ok = False
        else:
            status = PIT_PASS
            reason = "PIT 窗口为 D 及以前已可得字段，覆盖与截面方差通过"
            model_ok = spec["profile"] == "A" and prior_status not in {STRICT_PIT_SUBPERIOD_RESEARCH}
        if label == COMPOSITE_OF_KNOWN_FACTORS and status == PIT_PASS:
            reason += "；COMPOSITE_OF_KNOWN_FACTORS，可测增量但不是新的原始信息"
        rows.append(
            {
                "factor_id": fid,
                "family": spec["family"],
                "profile": spec["profile"],
                "pit_status": status,
                "reason": reason,
                "model_eligible": bool(model_ok and status == PIT_PASS),
                "label": label or "",
                "duplicate_of": prior.get("duplicate_of") or "",
                **screen,
            }
        )
    write_csv(PIT_CSV, rows)
    summary = {
        "status": "DONE",
        "n": len(rows),
        "n_pit_pass": int(sum(r["pit_status"] == PIT_PASS for r in rows)),
        "n_model_eligible": int(sum(r["model_eligible"] for r in rows)),
        "n_duplicate": int(sum(r["pit_status"] == DUPLICATE_EXPRESSION for r in rows)),
        "n_blocked": int(sum(str(r["pit_status"]).startswith("BLOCKED") or r["pit_status"] in {BLOCKED_ACCESS, BLOCKED_DATA, BLOCKED_PIT, BLOCKED_SEMANTICS} for r in rows)),
        "n_degenerate": int(sum(r["pit_status"] == CROSS_SECTIONALLY_DEGENERATE for r in rows)),
        "n_coverage_fail": int(sum(r["pit_status"] == "COVERAGE_BELOW_GATE" for r in rows)),
        "counts": pd.Series([r["pit_status"] for r in rows]).value_counts().to_dict(),
    }
    dump_json(PIT_JSON, {"summary": summary, "rows": rows})
    write_md(
        MD["11"],
        "\n".join(
            [
                "# Round22 · 11 PIT / 覆盖 / 常数",
                "",
                f"提案={summary['n']} PIT_PASS={summary['n_pit_pass']} 可进全历史模型={summary['n_model_eligible']}。",
                f"重复={summary['n_duplicate']} 阻断={summary['n_blocked']} 退化={summary['n_degenerate']} 覆盖不足={summary['n_coverage_fail']}。",
                f"计数=`{summary['counts']}`。",
                "",
                "覆盖门槛 0.80 在经济门冻结时已经写进训练门。这里不按 Sharpe 加减。",
                "",
            ]
        ),
    )
    return summary
