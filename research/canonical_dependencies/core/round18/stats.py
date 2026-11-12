"""成本 / 容量 / 分年 / bootstrap / SPA。只对正式组。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import MARKET_END, ROUND18_ROOT
from ..round11.common import daily_from_nav, sharpe_of
from ..round17.backtest import apply_b03, run_canonical
from .common import dump_json, load_json, write_md
from .contract import (
    A00,
    BLOCK_DAYS,
    BOOT_N,
    CAPACITY_NAV,
    COST_BPS,
    FORWARD_OOS_NOT_AVAILABLE,
    PRIOR_RESEARCH_SELECTION_EXISTS,
    SPA_BOOT,
    exp_dir,
    parity_dir,
)

STRESS_JSON = ROUND18_ROOT / "stress" / "stress.json"
CAP_JSON = ROUND18_ROOT / "capacity" / "capacity.json"
BOOT_JSON = ROUND18_ROOT / "bootstrap" / "bootstrap.json"
SPA_JSON = ROUND18_ROOT / "spa" / "spa.json"
FWD_JSON = ROUND18_ROOT / "forward" / "forward.json"
STATS_JSON = ROUND18_ROOT / "stats" / "stats.json"


def _nav(code: str) -> pd.DataFrame | None:
    for path in (exp_dir(code) / "nav.parquet", parity_dir(A00) / "nav.parquet"):
        if path.is_file():
            return pd.read_parquet(path)
    return None


def _pred(code: str) -> pd.DataFrame | None:
    path = exp_dir(code) / "predictions.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    return None


def _formal_rows() -> list[dict[str, Any]]:
    matrix = load_json(ROUND18_ROOT / "metrics" / "matrix.json")
    return [r for r in (matrix.get("rows") or []) if r.get("formal") and r.get("sharpe") is not None]


def run_stats(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    rows = _formal_rows()
    a00 = next((r for r in rows if r.get("code") == A00), None) or load_json(ROUND18_ROOT / "parity" / "parity.json").get("observed") or {}
    stress = {}
    sign_flip = False
    for row in rows:
        code = row["code"]
        pred = _pred(code)
        rec = {"code": code, "base_sharpe": row.get("sharpe"), "base_cagr": row.get("cagr")}
        if pred is None and code == A00:
            from ..round17.backtest import cached_overlay_v4
            pred = cached_overlay_v4()
        if pred is None:
            rec["status"] = "SKIP_DATA"
            stress[code] = rec
            continue
        overlayed, _ = (pred, None) if "allocation_multiplier" in pred.columns else apply_b03(pred)
        for bps in COST_BPS:
            dest = ROUND18_ROOT / "stress" / f"{code}_plus{int(bps)}bp"
            ran = run_canonical(dest, overlayed, extra={"stress_bps": bps}, resume=resume, write=True, extra_buy_bps=float(bps))
            rec[f"plus{int(bps)}"] = ran.get("observed")
            sh = (ran.get("observed") or {}).get("sharpe")
            if row.get("sharpe") is not None and sh is not None and (float(row["sharpe"]) > 0) != (float(sh) > 0):
                sign_flip = True
        stress[code] = rec
    dump_json(STRESS_JSON, {"rows": stress, "sign_flip": sign_flip})
    write_md(ROUND18_ROOT / "09_cost_stress.md", f"# 09 成本压力\n\nsign_flip=`{sign_flip}`\n")

    cap = {}
    cap_flip = False
    for row in rows:
        code = row["code"]
        pred = _pred(code)
        if pred is None and code == A00:
            from ..round17.backtest import cached_overlay_v4
            pred = cached_overlay_v4()
        if pred is None:
            continue
        overlayed, _ = (pred, None) if "allocation_multiplier" in pred.columns else apply_b03(pred)
        rec = {}
        for nav0 in CAPACITY_NAV:
            dest = ROUND18_ROOT / "capacity" / f"{code}_{int(nav0)}"
            ran = run_canonical(dest, overlayed, extra={"capacity": nav0}, resume=resume, write=True, initial_cash=float(nav0))
            rec[str(int(nav0))] = ran.get("observed")
            sh = (ran.get("observed") or {}).get("sharpe")
            if row.get("sharpe") is not None and sh is not None and (float(row["sharpe"]) > 0) != (float(sh) > 0):
                cap_flip = True
        cap[code] = rec
    dump_json(CAP_JSON, {"rows": cap, "sign_flip": cap_flip})
    write_md(ROUND18_ROOT / "10_capacity.md", f"# 10 容量\n\nsign_flip=`{cap_flip}`\n")

    boot = {"n": BOOT_N, "block_days": BLOCK_DAYS, "pairs": {}, "prior_research_selection_exists": PRIOR_RESEARCH_SELECTION_EXISTS}
    a00_nav = _nav(A00)
    a00_d = daily_from_nav(a00_nav) if a00_nav is not None else None
    rng = np.random.default_rng(20261018)
    for row in rows:
        code = row["code"]
        if code == A00 or a00_d is None:
            continue
        nav = _nav(code)
        if nav is None:
            continue
        d = daily_from_nav(nav)
        joined = pd.concat([a00_d.rename("a00"), d.rename("alt")], axis=1, join="inner").dropna()
        if len(joined) < BLOCK_DAYS * 4:
            continue
        n = len(joined)
        n_blocks = max(2, n // BLOCK_DAYS)
        diffs = []
        idx = np.arange(n)
        blocks = np.array_split(idx, n_blocks)
        for _ in range(BOOT_N):
            take = rng.integers(0, len(blocks), size=len(blocks))
            samp = np.concatenate([blocks[i] for i in take])
            sa = sharpe_of(joined["alt"].iloc[samp])
            sb = sharpe_of(joined["a00"].iloc[samp])
            if sa is None or sb is None:
                continue
            diffs.append(float(sa) - float(sb))
        arr = np.asarray(diffs, dtype=float)
        boot["pairs"][f"{code}-{A00}"] = {
            "mean_diff_sharpe": float(arr.mean()) if len(arr) else None,
            "p_mean_le_0": float((arr <= 0).mean()) if len(arr) else None,
            "n": int(len(arr)),
        }
    dump_json(BOOT_JSON, boot)
    write_md(ROUND18_ROOT / "11_bootstrap.md", f"# 11 Bootstrap\n\nN=`{BOOT_N}` block=`{BLOCK_DAYS}`\n")

    alts = []
    labels = []
    for row in rows:
        if row.get("code") == A00:
            continue
        nav = _nav(row["code"])
        if nav is None:
            continue
        alts.append(daily_from_nav(nav))
        labels.append(row["code"])
    spa = {"n_series": len(alts), "prior_research_selection_exists": PRIOR_RESEARCH_SELECTION_EXISTS, "boot": SPA_BOOT}
    if alts and a00_d is not None:
        from ..round11.stats import spa_test

        mat = pd.concat([s.rename(l) for s, l in zip(alts, labels)], axis=1)
        bench = a00_d.reindex(mat.index)
        try:
            spa["spa"] = spa_test(bench, mat, n_boot=SPA_BOOT)
        except Exception as exc:
            spa["spa"] = {"status": "EXECUTION_FAILED", "reason": str(exc)}
    else:
        spa["spa"] = {"status": "SKIP_DATA", "reason": "no alternative daily series"}
    dump_json(SPA_JSON, spa)
    write_md(ROUND18_ROOT / "12_spa.md", f"# 12 SPA / Holm / max-stat\n\nPRIOR_RESEARCH_SELECTION_EXISTS=`TRUE` n_series=`{spa['n_series']}`\n")

    yearly = {r.get("code"): r.get("yearly_returns") for r in rows}
    write_md(ROUND18_ROOT / "13_year_robustness.md", "# 13 分年\n\n" + str(yearly) + "\n")

    fwd = {
        "status": FORWARD_OOS_NOT_AVAILABLE,
        "cutoff": str(MARKET_END),
        "note": f"研究快照截止 {MARKET_END}。冻结后再查完整 panel；2026-07-25 之后无完整未参与研究新数据。",
    }
    dump_json(FWD_JSON, fwd)
    write_md(ROUND18_ROOT / "14_forward_oos.md", f"# 14 Forward OOS\n\n**{FORWARD_OOS_NOT_AVAILABLE}**。{fwd['note']}\n")

    from .attribution import run_attribution

    attr = run_attribution()
    payload = {
        "stress": {"sign_flip": sign_flip},
        "capacity": {"sign_flip": cap_flip},
        "bootstrap": boot,
        "spa": spa,
        "forward": fwd,
        "yearly": yearly,
        "attribution": attr,
    }
    dump_json(STATS_JSON, payload)
    return payload


def run_forward(**_: Any) -> dict[str, Any]:
    if FWD_JSON.is_file():
        return load_json(FWD_JSON)
    return run_stats()
