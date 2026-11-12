"""20 日块 bootstrap 2000（paired 同块）；Δ 只对 v4。"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..paths import ROUND10_ROOT, assert_project_result_path, ensure_round10_tree, round10_dir
from ..round8.robustness import block_bootstrap_delta
from .contract import R10_00
from .registry import SPECS, register_finalist_stress
from .runner import load_summary, run_codes


def _daily(nav: pd.DataFrame) -> pd.Series:
    n = nav.copy()
    n["trade_date"] = pd.to_datetime(n["trade_date"])
    return n.sort_values("trade_date").set_index("trade_date")["total_asset"].astype(float).pct_change().dropna()


def pick_finalists() -> list[str]:
    from .docs import v4_anchor_metrics, v5_gate_row

    v4 = v4_anchor_metrics()
    scored = []
    for code, spec in SPECS.items():
        if spec.diagnostic or spec.post_hoc or spec.skip or spec.stage in {
            "parity", "robustness", "perturbation", "attribution", "neighborhood", "execution", "capacity",
        }:
            continue
        s = load_summary(code)
        if not s or s.get("cagr") is None:
            continue
        scored.append((code, float(s["sharpe"]), float(s["cagr"]), v5_gate_row(code, s)))
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    out = []
    for code, _, _, gate in scored:
        if code not in out:
            out.append(code)
        if len(out) >= 5:
            break
    if R10_00 not in out:
        out = [R10_00] + out
    return out[:6]


def run_robustness_stage(*, resume: bool = True, jobs: int = 1) -> dict[str, Any]:
    ensure_round10_tree()
    finals = pick_finalists()
    extra = register_finalist_stress([c for c in finals if c != R10_00])
    if extra:
        run_codes(extra, resume=resume, jobs=jobs)
    base = _daily(pd.read_parquet(round10_dir(R10_00) / "nav.parquet"))
    boot = {}
    for code in finals:
        nav_path = round10_dir(code) / "nav.parquet"
        if not nav_path.is_file():
            continue
        boot[code] = block_bootstrap_delta(_daily(pd.read_parquet(nav_path)), base)
        path = ROUND10_ROOT / "bootstrap" / f"{code}.json"
        assert_project_result_path(path)
        path.write_text(json.dumps(boot[code], ensure_ascii=False, indent=2) + "\n")
    payload = {
        "finalists": finals,
        "stress": extra,
        "bootstrap": boot,
        "note": "paired 20-day block bootstrap n=2000 same blocks; Δ vs baseline_v4 only",
    }
    out = ROUND10_ROOT / "bootstrap" / "summary.json"
    assert_project_result_path(out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    return payload


__all__ = ["pick_finalists", "run_robustness_stage"]
