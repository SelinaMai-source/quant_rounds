"""A3 vs A0 43 元逐交易对账。禁止只引用 summary.json。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND21_4_ROOT
from ..round11.common import daily_from_nav
from .common import dump_json, finite_or_none, load_json, write_md
from .contract import MD, V5_NAV_END_EXACT, exp_dir

RECON_CSV = ROUND21_4_ROOT / "L1_NAV_RECONCILIATION.csv"
RECON_JSON = ROUND21_4_ROOT / "nav" / "a3_recon.json"
TRADE_CSV = ROUND21_4_ROOT / "attribution" / "a3_trade_bridge.csv"


def _nav(code: str) -> pd.DataFrame | None:
    path = exp_dir(code) / "nav.parquet"
    if not path.is_file():
        return None
    nav = pd.read_parquet(path)
    nav["trade_date"] = pd.to_datetime(nav["trade_date"])
    return nav.sort_values("trade_date")


def _trades(code: str) -> pd.DataFrame:
    path = exp_dir(code) / "trades.parquet"
    if not path.is_file():
        return pd.DataFrame()
    t = pd.read_parquet(path)
    for c in ("trade_date", "signal_date", "entry_date", "exit_date"):
        if c in t.columns:
            t[c] = pd.to_datetime(t[c], errors="coerce")
    return t


def run_recon43(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and RECON_CSV.is_file() and RECON_JSON.is_file() and MD["09"].is_file():
        existing = load_json(RECON_JSON)
        if existing.get("status") == "DONE":
            return existing
    a0 = _nav("R21_4_A0")
    a3 = _nav("R21_4_A3")
    a6 = _nav("R21_4_A6")
    if a0 is None or a3 is None:
        payload = {"status": "AUDIT_FAILURE", "missing": [c for c, n in (("R21_4_A0", a0), ("R21_4_A3", a3)) if n is None]}
        dump_json(RECON_JSON, payload)
        return payload
    merged = a0[["trade_date", "total_asset"]].rename(columns={"total_asset": "a0"}).merge(
        a3[["trade_date", "total_asset"]].rename(columns={"total_asset": "a3"}), on="trade_date", how="outer"
    )
    if a6 is not None:
        merged = merged.merge(a6[["trade_date", "total_asset"]].rename(columns={"total_asset": "a6"}), on="trade_date", how="outer")
    merged = merged.sort_values("trade_date")
    merged["d_a3_a0"] = merged["a3"] - merged["a0"]
    merged["abs_d"] = merged["d_a3_a0"].abs()
    fork = merged.loc[merged["abs_d"].fillna(0) > 0.01]
    first_fork = None if fork.empty else str(pd.Timestamp(fork.iloc[0]["trade_date"]).date())
    last = merged.iloc[-1]
    t0 = _trades("R21_4_A0")
    t3 = _trades("R21_4_A3")
    bridge = pd.DataFrame()
    if not t0.empty and not t3.empty:
        k0 = set(zip(t0.get("instrument_id", pd.Series(dtype=str)).astype(str), pd.to_datetime(t0.get("entry_date", t0.get("trade_date"))).astype(str)))
        k3 = set(zip(t3.get("instrument_id", pd.Series(dtype=str)).astype(str), pd.to_datetime(t3.get("entry_date", t3.get("trade_date"))).astype(str)))
        only3 = k3 - k0
        only0 = k0 - k3
        bridge = pd.DataFrame(
            [{"side": "A3_only", "instrument_id": a, "entry": b} for a, b in sorted(only3)]
            + [{"side": "A0_only", "instrument_id": a, "entry": b} for a, b in sorted(only0)]
        )
        if not bridge.empty:
            bridge.to_csv(TRADE_CSV, index=False)
    merged.to_csv(RECON_CSV, index=False)
    delta_end = finite_or_none(last.get("d_a3_a0"))
    payload = {
        "status": "DONE",
        "first_fork": first_fork,
        "n_rows": int(len(merged)),
        "n_fork_days": int(len(fork)),
        "a0_nav_end": finite_or_none(last.get("a0")),
        "a3_nav_end": finite_or_none(last.get("a3")),
        "a6_nav_end": finite_or_none(last.get("a6")) if "a6" in merged.columns else None,
        "delta_a3_a0": delta_end,
        "v5_nav": V5_NAV_END_EXACT,
        "n_trade_diff": int(len(bridge)),
        "trade_bridge": str(TRADE_CSV) if not bridge.empty else None,
        "csv": str(RECON_CSV),
        "verdict": "ECONOMICALLY_IMMATERIAL" if delta_end is not None and abs(delta_end) < 100 else "MATERIAL_OR_LARGE",
        "note": "A3 是 10% Five 头噪声；+43 元预先不达标。",
    }
    dump_json(RECON_JSON, payload)
    write_md(
        MD["09"],
        "\n".join(
            [
                "# Round21.4 · 09 43 元对账",
                "",
                f"first_fork=`{first_fork}` ΔNAV A3−A0=`{delta_end}` n_fork_days=`{payload['n_fork_days']}` n_trade_diff=`{payload['n_trade_diff']}`",
                f"A0 NAV=`{payload['a0_nav_end']}` A3=`{payload['a3_nav_end']}` A6=`{payload['a6_nav_end']}`",
                f"verdict=`{payload['verdict']}` CSV=`{RECON_CSV}`",
                "",
            ]
        ),
    )
    return payload
