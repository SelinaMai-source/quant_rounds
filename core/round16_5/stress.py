"""同窗成本压力 +20/+30bp。不得重复计算已含 slippage。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND16_5_ROOT
from .common import dump_json, load_json, observed_from_report, pct, write_md
from .contract import COST_BPS, V00, V01, V03, V04, overlayed_path, parity_dir
from .engine import cached_overlay_v4, run_engine
from .freeze import require_frozen

ST_JSON = ROUND16_5_ROOT / "stress" / "stress.json"
ST_MD = ROUND16_5_ROOT / "12_cost_stress.md"


def _ov(code: str):
    path = overlayed_path(code)
    if path.is_file():
        return pd.read_parquet(path)
    if code == V00:
        return cached_overlay_v4()
    p2 = overlayed_path(code)
    return pd.read_parquet(p2) if p2.is_file() else None


def run_stress(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    out: dict[str, Any] = {}
    for code in (V00, V01, V03, V04):
        ov = _ov(code)
        if ov is None:
            out[code] = {"status": "NO_OVERLAY", "skip": True}
            continue
        rec = {}
        for bps in COST_BPS:
            dest = ROUND16_5_ROOT / "stress" / f"{code}_plus{int(bps)}bp"
            ran = run_engine(
                dest,
                ov,
                extra={"code": code, "extra_buy_bps": bps, "fee_schedule": "HISTORICAL_FEE_CORRECTED"},
                resume=resume,
                write=True,
                extra_buy_bps=bps,
            )
            rec[f"plus{int(bps)}"] = ran.get("observed") or observed_from_report(ran.get("report") or {})
        out[code] = rec
    sign_flip = {}
    def _cagr(code, key):
        row = ((out.get(code) or {}).get(key) or {}).get("cagr")
        return row
    for key in ("plus20", "plus30"):
        v0 = _cagr(V00, key)
        v1 = _cagr(V01, key)
        if v0 is None or v1 is None:
            sign_flip[key] = None
        else:
            base_sign = 1 if (float((load_json(parity_dir(V01) / "summary.json") or {}).get("cagr") or 0) - float((load_json(parity_dir(V00) / "summary.json") or {}).get("cagr") or 0)) > 0 else -1
            stress_sign = 1 if (float(v1) - float(v0)) > 0 else -1
            sign_flip[key] = bool(stress_sign != base_sign)
    payload = {
        "status": "DONE",
        "results": out,
        "sign_flip": sign_flip,
        "note": "主评价 HISTORICAL_FEE_CORRECTED。不得重复计算已含 slippage。同窗 vs V00。",
    }
    dump_json(ST_JSON, payload)
    write_md(
        ST_MD,
        "\n".join([
            "# Round 16.5 · 12 成本压力",
            "",
            f"sign_flip=`{sign_flip}`。",
            f"V00 +20 CAGR=`{_cagr(V00, 'plus20')}` V01 +20 CAGR=`{_cagr(V01, 'plus20')}`。",
            f"V00 +30 CAGR=`{_cagr(V00, 'plus30')}` V01 +30 CAGR=`{_cagr(V01, 'plus30')}`。",
            "",
        ]),
    )
    return payload


def load_stress() -> dict[str, Any]:
    return load_json(ST_JSON)
