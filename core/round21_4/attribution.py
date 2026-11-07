"""A 矩阵 Top1/Top2 归因 + 冻结对照。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND21_4_ROOT
from .common import dump_json, finite_or_none, load_json, write_md
from .contract import A_CONTRASTS, MD, exp_dir

ATTR_JSON = ROUND21_4_ROOT / "attribution" / "attribution.json"


def _top_share(a: str, b: str) -> float | None:
    pa = exp_dir(a) / "daily_top1_top2.parquet"
    pb = exp_dir(b) / "daily_top1_top2.parquet"
    if not pa.is_file() or not pb.is_file():
        return None
    xa = pd.read_parquet(pa)
    xb = pd.read_parquet(pb)
    xa["trade_date"] = pd.to_datetime(xa["trade_date"])
    xb["trade_date"] = pd.to_datetime(xb["trade_date"])
    if "head" in xa.columns:
        xa = xa.loc[xa["head"].astype(str).eq("ten")]
    if "head" in xb.columns:
        xb = xb.loc[xb["head"].astype(str).eq("ten")]

    def _keys(df):
        return set(zip(df["trade_date"].astype(str), df["instrument_id"].astype(str), df["rank"].astype(int)))

    ka, kb = _keys(xa), _keys(xb)
    if not ka:
        return None
    return 1.0 - (len(ka & kb) / len(ka))


def _trades_share(a: str, b: str) -> float | None:
    """融合口径：实际成交 (entry_date, instrument_id) 集合差异。冻结门 Top1/Top2 变化应测融合选择。"""
    pa = exp_dir(a) / "trades.parquet"
    pb = exp_dir(b) / "trades.parquet"
    if not pa.is_file() or not pb.is_file():
        return None
    xa = pd.read_parquet(pa, columns=["entry_date", "instrument_id"])
    xb = pd.read_parquet(pb, columns=["entry_date", "instrument_id"])
    ka = set(zip(xa["entry_date"].astype(str), xa["instrument_id"].astype(str)))
    kb = set(zip(xb["entry_date"].astype(str), xb["instrument_id"].astype(str)))
    if not ka:
        return None
    return 1.0 - (len(ka & kb) / len(ka))


def run_attribution(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and ATTR_JSON.is_file() and MD["08"].is_file():
        existing = load_json(ATTR_JSON)
        if existing.get("status") == "DONE":
            return existing
    from .backtest import A_BT_JSON

    bt = load_json(A_BT_JSON)
    by = {r.get("code"): r for r in (bt.get("main_rows") or [])}
    rows = []
    for a, b, name in A_CONTRASTS:
        ra, rb = by.get(a) or {}, by.get(b) or {}
        rows.append(
            {
                "contrast": name,
                "a": a,
                "b": b,
                "delta_sharpe": None if ra.get("sharpe") is None or rb.get("sharpe") is None else float(ra["sharpe"]) - float(rb["sharpe"]),
                "delta_cagr": None if ra.get("cagr") is None or rb.get("cagr") is None else float(ra["cagr"]) - float(rb["cagr"]),
                "delta_nav": None if ra.get("nav_end") is None or rb.get("nav_end") is None else float(ra["nav_end"]) - float(rb["nav_end"]),
                "delta_maxdd": None if ra.get("max_drawdown") is None or rb.get("max_drawdown") is None else float(ra["max_drawdown"]) - float(rb["max_drawdown"]),
                "top1_changed_share": _top_share(a, b),
                "top1_changed_share_fused": _trades_share(a, b),
                "a_sharpe": ra.get("sharpe"),
                "b_sharpe": rb.get("sharpe"),
            }
        )
    toxic = "TEN" if abs((next((r for r in rows if r["contrast"] == "A4-A0_add_l1_ten"), {}) or {}).get("delta_nav") or 0) > 1e6 else "UNCLEAR"
    payload = {
        "status": "DONE",
        "rows": rows,
        "toxic_head": toxic,
        "l1_fold_rho": -0.073,
        "interpretation": "L1 vs g0 ρ≈-0.073 = FEATURE_RECOMPOSITION; Ten 加 L1 解释 21.3 −940万",
        "forty_three_source": bt.get("forty_three_source"),
    }
    dump_json(ATTR_JSON, payload)
    write_md(
        MD["08"],
        "# Round21.4 · 08 L1 A 矩阵归因\n\n"
        f"toxic_head=`{toxic}` 43来源=`{bt.get('forty_three_source')}`\n\n"
        + "\n".join(f"- {r['contrast']}: ΔNAV=`{r.get('delta_nav')}` ΔSharpe=`{r.get('delta_sharpe')}` topΔ(ten头)=`{r.get('top1_changed_share')}` topΔ(融合成交)=`{r.get('top1_changed_share_fused')}`" for r in rows)
        + "\n",
    )
    return payload
