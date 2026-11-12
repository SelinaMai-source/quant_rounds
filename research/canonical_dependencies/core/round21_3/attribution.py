"""Top1/Top2 归因 + 四路对照。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND21_3_ROOT
from .backtest import BT_JSON, L1_BT_JSON
from .common import dump_json, finite_or_none, load_json, write_md
from .contract import L1_E0, L1_E1, L1_E2, L1_E3, MD, exp_dir
from .correlate import CORR_JSON

ATTR_JSON = ROUND21_3_ROOT / "attribution" / "attribution.json"


def _top_share(a: str, b: str) -> float | None:
    pa = exp_dir(a) / "daily_top1_top2.parquet"
    pb = exp_dir(b) / "daily_top1_top2.parquet"
    if not pa.is_file() or not pb.is_file():
        return None
    xa = pd.read_parquet(pa)
    xb = pd.read_parquet(pb)
    xa["trade_date"] = pd.to_datetime(xa["trade_date"])
    xb["trade_date"] = pd.to_datetime(xb["trade_date"])
    xa = xa.loc[xa["head"].astype(str).eq("ten")] if "head" in xa.columns else xa
    xb = xb.loc[xb["head"].astype(str).eq("ten")] if "head" in xb.columns else xb

    def _keys(df):
        return set(zip(df["trade_date"].astype(str), df["instrument_id"].astype(str), df["rank"].astype(int)))

    ka, kb = _keys(xa), _keys(xb)
    if not ka:
        return None
    return 1.0 - (len(ka & kb) / len(ka))


def run_attribution(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and ATTR_JSON.is_file() and MD["16"].is_file():
        existing = load_json(ATTR_JSON)
        if existing.get("status") == "DONE":
            return existing
    bt = load_json(BT_JSON)
    l1 = load_json(L1_BT_JSON)
    corr = load_json(CORR_JSON)
    nearest = corr.get("nearest") or {}
    by = {r.get("code"): r for r in (bt.get("main_rows") or [])}
    for r in l1.get("main_rows") or []:
        by.setdefault(r.get("code"), r)
    rows = []
    # L1 强制
    e0, e1, e2, e3 = by.get(L1_E0) or {}, by.get(L1_E1) or {}, by.get(L1_E2) or {}, by.get(L1_E3) or {}
    rows.append(
        {
            "new_factor": "r21_l1_known_float_chg",
            "nearest_legacy": "g0_amount_log",
            "fold_signed": -0.073198324455432,
            "fold_abs": 0.073198324455432,
            "delta_remove_sharpe": None if e1.get("sharpe") is None or e0.get("sharpe") is None else float(e1["sharpe"]) - float(e0["sharpe"]),
            "delta_add_sharpe": None if e2.get("sharpe") is None or e0.get("sharpe") is None else float(e2["sharpe"]) - float(e0["sharpe"]),
            "delta_replace_sharpe": None if e3.get("sharpe") is None or e0.get("sharpe") is None else float(e3["sharpe"]) - float(e0["sharpe"]),
            "delta_new_given_remove_sharpe": None if e3.get("sharpe") is None or e1.get("sharpe") is None else float(e3["sharpe"]) - float(e1["sharpe"]),
            "delta_replace_cagr": None if e3.get("cagr") is None or e0.get("cagr") is None else float(e3["cagr"]) - float(e0["cagr"]),
            "delta_replace_maxdd": None if e3.get("max_drawdown") is None or e0.get("max_drawdown") is None else float(e3["max_drawdown"]) - float(e0["max_drawdown"]),
            "top1_changed_share": _top_share(L1_E3, L1_E0),
            "interpretation": l1.get("forty_three_source") or "PENDING",
        }
    )
    # 其他 E3
    for r in bt.get("main_rows") or []:
        code = str(r.get("code") or "")
        if not code.startswith("R21_3_E3_"):
            continue
        fid = r.get("factor_id") or code.replace("R21_3_E3_", "")
        if fid == "r21_l1_known_float_chg":
            continue
        e2c = by.get(f"R21_3_E2_{fid}") or {}
        e1c = by.get(f"R21_3_E1_{fid}") or {}
        nb = nearest.get(fid) or {}
        rows.append(
            {
                "new_factor": fid,
                "nearest_legacy": nb.get("majority_legacy"),
                "fold_signed": nb.get("mean_fold_signed"),
                "fold_abs": nb.get("mean_fold_abs"),
                "delta_remove_sharpe": None if e1c.get("sharpe") is None or e0.get("sharpe") is None else float(e1c["sharpe"]) - float(e0.get("sharpe") or 0),
                "delta_add_sharpe": None if e2c.get("sharpe") is None or e0.get("sharpe") is None else float(e2c["sharpe"]) - float(e0.get("sharpe") or 0),
                "delta_replace_sharpe": None if r.get("sharpe") is None or e0.get("sharpe") is None else float(r["sharpe"]) - float(e0.get("sharpe") or 0),
                "top1_changed_share": _top_share(code, L1_E0),
            }
        )
    pd.DataFrame(rows).to_csv(ROUND21_3_ROOT / "attribution" / "four_way_delta.csv", index=False)
    payload = {"status": "DONE", "rows": rows}
    dump_json(ATTR_JSON, payload)
    write_md(MD["16"], f"# Round21.3 · 16 Top1/Top2 归因\n\nL1 四路对照已写入。rows=`{len(rows)}`\n")
    return payload
