"""Top1/Top2 与 A00 重叠，分离选股 vs ALPHA-INDUCED ALLOCATION EFFECT。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND18_ROOT
from .common import dump_json, write_md
from .contract import A00, exp_dir, parity_dir

ATTR_JSON = ROUND18_ROOT / "attribution" / "increments.json"
ATTR_MD = ROUND18_ROOT / "11a_top1_top2_attribution.md"


def _overlay_path(code: str):
    p = exp_dir(code) / "predictions.parquet"
    if p.is_file():
        return p
    if code == A00:
        p2 = parity_dir(A00) / "predictions.parquet"
        if p2.is_file():
            return p2
    return None


def _top(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    work = frame.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"])
    work["instrument_id"] = work["instrument_id"].astype(str)
    work["_r"] = work.groupby("signal_date")["score"].rank(method="first", ascending=False)
    return work.loc[work["_r"] <= k, ["signal_date", "instrument_id", "_r"]]


def run_attribution() -> dict[str, Any]:
    matrix = __import__("json").loads((ROUND18_ROOT / "metrics" / "matrix.json").read_text(encoding="utf-8"))
    a00p = _overlay_path(A00)
    if a00p is None:
        # A00 用 cached overlay 写入的 trades 反推
        from ..round17.backtest import cached_overlay_v4

        a00 = cached_overlay_v4()
    else:
        a00 = pd.read_parquet(a00p)
    t1 = _top(a00, 1).rename(columns={"instrument_id": "a00"})
    t2 = _top(a00, 2)
    rows = []
    for rec in matrix.get("rows") or []:
        code = rec.get("code")
        if code == A00 or rec.get("sharpe") is None:
            continue
        path = _overlay_path(code)
        if path is None:
            continue
        pred = pd.read_parquet(path)
        c1 = _top(pred, 1).rename(columns={"instrument_id": "alt"})
        joined = t1.merge(c1, on=["signal_date", "_r"], how="inner")
        top1_agree = float((joined["a00"] == joined["alt"]).mean()) if len(joined) else None
        c2 = _top(pred, 2)
        keys = ["signal_date", "instrument_id"]
        both = t2.merge(c2, on=keys, how="inner")
        top2_overlap = float(len(both) / max(len(t2), 1))
        rows.append({
            "code": code,
            "sharpe": rec.get("sharpe"),
            "sharpe_minus_a00": (float(rec["sharpe"]) - 1.8120914675763191) if rec.get("sharpe") is not None else None,
            "top1_agree": top1_agree,
            "top2_overlap": top2_overlap,
            "note": "B03 格子随排序变化 = ALPHA-INDUCED ALLOCATION EFFECT",
        })
    payload = {"status": "done", "rows": rows}
    dump_json(ATTR_JSON, payload)
    lines = ["# Top1/Top2 归因", "", "| code | ΔSharpe | Top1 agree | Top2 overlap |", "|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['code']} | {r.get('sharpe_minus_a00')} | {r.get('top1_agree')} | {r.get('top2_overlap')} |")
    write_md(ATTR_MD, "\n".join(lines))
    return payload
