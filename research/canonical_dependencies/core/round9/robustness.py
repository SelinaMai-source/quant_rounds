"""20 日块 bootstrap 2000（paired 同块）；费用 ×1.5/×2；买侧不利。"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..paths import ROUND9_ROOT, assert_project_result_path, ensure_round9_tree, round9_dir
from ..round8.robustness import block_bootstrap_delta
from .contract import ANCHOR_DIR, R9_00
from .registry import SPECS, register_winner_stress, specs_for_stage
from .runner import load_summary, run_codes


def winner_dependence(code: str) -> dict[str, Any]:
    from ..round8.robustness import winner_dependence as _r8
    from ..paths import round9_dir

    path = round9_dir(code) / "trades.parquet"
    if path.is_file():
        # 复用 round8 口径，但指向 round9 trades
        import pandas as pd
        from ..round8.robustness import winner_dependence as _

        t = pd.read_parquet(path)
        dollar = pd.to_numeric(t.get("sell_proceeds"), errors="coerce") - pd.to_numeric(t.get("buy_cost"), errors="coerce")
        if dollar.isna().all():
            dollar = pd.to_numeric(t["ret"], errors="coerce")
        total = float(dollar.sum())
        ordered = dollar.sort_values(ascending=False)
        n = len(ordered)

        def share(k: float, tail: str) -> float:
            import numpy as np

            m = max(1, int(np.ceil(n * k)))
            if tail == "top":
                return float(ordered.head(m).sum() / total) if total else float("nan")
            return float(ordered.tail(m).sum() / total) if total else float("nan")

        return {
            "code": code,
            "role": "DIAGNOSTIC_NOT_REALISTIC",
            "top1pct": share(0.01, "top"),
            "top5pct": share(0.05, "top"),
            "top10pct": share(0.10, "top"),
        }
    return {"status": "missing", "code": code}


def _daily(nav: pd.DataFrame) -> pd.Series:
    n = nav.copy()
    n["trade_date"] = pd.to_datetime(n["trade_date"])
    return n.sort_values("trade_date").set_index("trade_date")["total_asset"].astype(float).pct_change().dropna()


def pick_candidate() -> str | None:
    from .docs import v4_gate_row

    best = None
    best_key = (-9.0, -9.0)
    for code, spec in SPECS.items():
        if spec.diagnostic or spec.post_hoc or spec.skip:
            continue
        if spec.stage in {"robustness", "parity", "forensics"}:
            continue
        s = load_summary(code)
        if not s or s.get("sharpe") is None:
            continue
        row = v4_gate_row(code, s)
        if not row.get("meets"):
            continue
        key = (float(s["sharpe"]), float(s["cagr"]))
        if key > best_key:
            best_key = key
            best = code
    if best:
        return best
    # 没过门槛也挑一个 formal 对照做压力，仍不升 v4
    for code in ("R9_B01", "R9_C01", "R9_B03"):
        if load_summary(code) and load_summary(code).get("cagr") is not None:
            return code
    return None


def run_robustness_stage(*, resume: bool = True, jobs: int = 1) -> dict[str, Any]:
    ensure_round9_tree()
    codes = [s.code for s in specs_for_stage("robustness") if not s.post_hoc]
    cand = pick_candidate()
    extra = register_winner_stress(cand) if cand else []
    run_list = codes + extra
    bt = run_codes(run_list, resume=resume, jobs=jobs) if run_list else {"jobs": {}, "failed": []}
    base = _daily(pd.read_parquet(round9_dir(R9_00) / "nav.parquet"))
    boot = {R9_00: block_bootstrap_delta(base, base)}
    depend = {R9_00: winner_dependence(R9_00)}
    finals = [R9_00]
    if cand:
        finals.append(cand)
    for code in finals:
        nav_path = round9_dir(code) / "nav.parquet"
        if not nav_path.is_file():
            if code == R9_00:
                nav_path = ANCHOR_DIR / "nav.parquet"
            else:
                continue
        boot[code] = block_bootstrap_delta(_daily(pd.read_parquet(nav_path)), base)
        depend[code] = winner_dependence(code)
        path = ROUND9_ROOT / "bootstrap" / f"{code}.json"
        assert_project_result_path(path)
        path.write_text(json.dumps(boot[code], ensure_ascii=False, indent=2) + "\n")
    payload = {
        "candidate": cand,
        "bootstrap": boot,
        "winner_dependence": depend,
        "note": "paired 20-day block bootstrap n=2000 same blocks; winner dependence DIAGNOSTIC",
    }
    out = ROUND9_ROOT / "bootstrap" / "summary.json"
    assert_project_result_path(out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    from .docs import write_named_md

    extra_md = (
        "块长 20 日，2000 次，candidate 与 v3 **同一块**。费用 ×1.5/×2；买侧 +10/20/30/50bp。\n"
        f"candidate=`{cand}`。winner dependence = DIAGNOSTIC / NOT REALISTIC。\n"
    )
    write_named_md(
        "09_robustness.md",
        "robustness",
        [c for c in run_list if c in SPECS],
        extra_md,
    )
    return payload


__all__ = ["pick_candidate", "run_robustness_stage"]
