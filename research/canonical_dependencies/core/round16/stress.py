"""成本压力：主评价 HISTORICAL_FEE_CORRECTED；+20bp / +30bp adverse buy。同窗 vs v4。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND16_ROOT
from ..round12.runner import hist_hooks
from ..round15.runner import load_market, run_r15_backtest
from .backtest import load_backtest
from .common import dump_json, load_json, observed_from_report, pct, write_md
from .contract import OPTION_IDS, R16_00, experiment_dir, overlayed_path, parity_dir
from .freeze import require_frozen
from ..round15.runner import build_version_overlayed

ST_JSON = ROUND16_ROOT / "stress" / "stress.json"
ST_MD = ROUND16_ROOT / "10_execution_cost_stress.md"


def _run_cost(code: str, ov, dest, loaded, bps: float, resume: bool) -> dict[str, Any]:
    ran = run_r15_backtest(
        dest,
        overlayed=ov,
        validation_hooks=hist_hooks(extra_buy_bps=bps),
        extra_report={"fee_schedule": "HISTORICAL_FEE_CORRECTED", "extra_buy_bps": bps, "candidate": code},
        resume=resume,
        prepared_market=loaded["prepared"],
        market_meta=loaded,
    )
    obs = observed_from_report(ran.get("report") or {})
    obs.update({"code": code, "bps": bps, "status": ran.get("status")})
    return obs


def run_stress(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    bt = load_backtest()
    results = bt.get("results") or {}
    loaded = load_market()
    ov4, _, _ = build_version_overlayed("v4")
    out: dict[str, Any] = {"v4": {}}
    for bps in (20.0, 30.0):
        dest = ROUND16_ROOT / "stress" / f"v4_plus{int(bps)}bp"
        out["v4"][f"plus{int(bps)}"] = _run_cost("v4", ov4, dest, loaded, bps, resume)
    for code in OPTION_IDS:
        row = results.get(code) or {}
        if row.get("skip"):
            out[code] = {"status": row.get("status"), "skip": True}
            continue
        path = overlayed_path(code)
        if not path.is_file():
            out[code] = {"status": "NO_OVERLAY", "skip": True}
            continue
        import pandas as pd
        ov = pd.read_parquet(path)
        rec = {}
        for bps in (20.0, 30.0):
            dest = ROUND16_ROOT / "stress" / f"{code}_plus{int(bps)}bp"
            rec[f"plus{int(bps)}"] = _run_cost(code, ov, dest, loaded, bps, resume)
        out[code] = rec
    payload = {"status": "DONE", "results": out, "note": "不得重复计算已含 slippage。同窗 vs v4。"}
    dump_json(ST_JSON, payload)
    lines = ["# Round16 · 10 成本压力", "", "主评价 HISTORICAL_FEE_CORRECTED。另 +20bp / +30bp adverse buy。", ""]
    write_md(ST_MD, "\n".join(lines) + f"\n{out}\n")
    return payload


def load_stress() -> dict[str, Any]:
    return load_json(ST_JSON)
