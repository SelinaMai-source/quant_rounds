"""R9_A01 Top2=0 + 日度边际效用账本 + A/B/C × separation 2D。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND9_ROOT, assert_project_result_path, ensure_round9_tree, round9_dir
from .allocator import b01_nav, b02_nav, b03_nav, write_alloc
from .contract import B01_CELL, DISAGREE_LABEL, PARENT_CODE, R9_00
from .features import build_daily_frame, persist_daily
from .runner import load_summary, run_codes


def _pct(x, digits=2, signed=True) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    txt = f"{float(x):.{digits}%}"
    if signed and float(x) > 0:
        txt = "+" + txt
    return txt.replace("-", "−")


def _cell_table(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for agree in ("A", "B", "C"):
        for sep in ("LOW", "MID", "HIGH"):
            block = daily.loc[daily["agreement"].eq(agree) & daily["sep_bucket"].eq(sep)]
            n = int(len(block))
            inc = pd.to_numeric(block.get("incremental_ret"), errors="coerce")
            ret = pd.to_numeric(block.get("top2_ret"), errors="coerce")
            flag = ""
            if n < 30:
                flag = "N<30_NO_MAP"
            elif n < 50:
                flag = "LOW_SAMPLE"
            rows.append({
                "agreement": agree,
                "agreement_alias": DISAGREE_LABEL[agree],
                "sep_bucket": sep,
                "n": n,
                "mean_incremental": float(inc.mean()) if inc.notna().any() else None,
                "median_incremental": float(inc.median()) if inc.notna().any() else None,
                "pct_inc_gt0": float((inc > 0).mean()) if inc.notna().any() else None,
                "mean_top2_ret": float(ret.mean()) if ret.notna().any() else None,
                "top2_win": float((ret > 0).mean()) if ret.notna().any() else None,
                "b01_nav": float(B01_CELL[(agree, sep)]),
                "flag": flag,
            })
    return pd.DataFrame(rows)


def write_forensics_md(payload: dict[str, Any], cells: pd.DataFrame) -> None:
    a01 = payload.get("a01") or {}
    v3 = payload.get("v3") or {}
    mu = payload.get("marginal") or {}
    lines = [
        "| agreement | alias | sep | n | mean Δret | Top2 mean ret | Top2 win | B01 NAV | flag |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, r in cells.iterrows():
        lines.append(
            f"| {r['agreement']} | {r['agreement_alias']} | {r['sep_bucket']} | {int(r['n'])} | "
            f"{_pct(r.get('mean_incremental'), 4)} | {_pct(r.get('mean_top2_ret'))} | "
            f"{_pct(r.get('top2_win'), signed=False) if r.get('top2_win') is not None else '—'} | "
            f"{float(r['b01_nav']):.0%} | {r.get('flag') or ''} |"
        )
    body = f"""# 01 — Top2 边际效用 forensics

Parent = **baseline_v3 = {PARENT_CODE}**。`R9_A01` = 同一套 v3 Top1，Top2=0。
增量定义：日收益 `v3 − A01`（组合层面，不是单票均值）。
分离度 = `top2_minus_median_p250`：LOW≤p20 / MID p20–p50 / HIGH>p50。
分歧沿用 E02 的 **A/B/C**（HIGH=A，MEDIUM=B，LOW=C）。
N&lt;50 标 `LOW_SAMPLE`；N&lt;30 **不得进正式 mapping**。B01 九格是预注册经济表，不是全样本填表。

## A01 vs v3

| 组 | CAGR | Sharpe | MaxDD | 笔数 |
| --- | ---: | ---: | ---: | ---: |
| `R9_00` / v3 | {_pct(v3.get('cagr'))} | {v3.get('sharpe')} | {_pct(v3.get('max_drawdown'))} | {v3.get('completed_trades')} |
| `R9_A01` Top2=0 | {_pct(a01.get('cagr'))} | {a01.get('sharpe')} | {_pct(a01.get('max_drawdown'))} | {a01.get('completed_trades')} |

组合日增量：n={mu.get('n_days')}；均值={mu.get('mean_daily_incremental')}；
年化增量≈{mu.get('ann_incremental')}；Top2 有正增量的天数占比={mu.get('pct_days_top2_helps')}。

## 2D：A/B/C × LOW/MID/HIGH

{chr(10).join(lines)}

日度账本：`result/round9/ledger/top2_marginal_ledger.parquet`。
"""
    path = ROUND9_ROOT / "01_top2_marginal_forensics.md"
    assert_project_result_path(path)
    path.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")


def run_forensics(*, resume: bool = True, jobs: int = 1) -> dict[str, Any]:
    ensure_round9_tree()
    out = run_codes(["R9_A01"], resume=resume, jobs=jobs)
    daily = build_daily_frame(v3_code=R9_00, a01_code="R9_A01")
    persist_daily(daily, "top2_daily_features.parquet")
    write_alloc("R9_B01", daily.index, b01_nav(daily["agreement"].astype(str), daily["sep_bucket"].astype(str)))
    write_alloc("R9_B02", daily.index, b02_nav(daily["agreement"].astype(str), daily["sep_bucket"].astype(str)))
    write_alloc("R9_B03", daily.index, b03_nav(daily["agreement"].astype(str), daily["sep_bucket"].astype(str)))
    ledger = daily.reset_index()
    path = ROUND9_ROOT / "ledger" / "top2_marginal_ledger.parquet"
    assert_project_result_path(path)
    ledger.to_parquet(path, index=False)

    v3_nav = pd.read_parquet(round9_dir(R9_00) / "nav.parquet")
    a01_nav = pd.read_parquet(round9_dir("R9_A01") / "nav.parquet")
    def _r(nav):
        n = nav.copy()
        n["trade_date"] = pd.to_datetime(n["trade_date"])
        s = n.sort_values("trade_date").set_index("trade_date")["total_asset"].astype(float).pct_change().dropna()
        return s
    a = _r(v3_nav).rename("v3")
    b = _r(a01_nav).rename("a01")
    aligned = pd.concat([a, b], axis=1).dropna()
    aligned["inc"] = aligned["v3"] - aligned["a01"]
    wealth = float((1.0 + aligned["inc"]).prod())
    years = max(len(aligned) / 252.0, 1e-9)
    marginal = {
        "n_days": int(len(aligned)),
        "mean_daily_incremental": float(aligned["inc"].mean()),
        "std_daily_incremental": float(aligned["inc"].std(ddof=1)) if len(aligned) > 1 else None,
        "ann_incremental": float(wealth ** (1.0 / years) - 1.0),
        "pct_days_top2_helps": float((aligned["inc"] > 0).mean()),
    }
    cells = _cell_table(daily)
    cells_path = ROUND9_ROOT / "metrics" / "top2_2d_cells.json"
    assert_project_result_path(cells_path)
    cells_path.write_text(cells.to_json(orient="records", force_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload = {
        "a01": load_summary("R9_A01") or {},
        "v3": load_summary(R9_00) or {},
        "marginal": marginal,
        "n_signal_days": int(len(daily)),
        "jobs": {k: v.get("status") for k, v in (out.get("jobs") or {}).items()},
    }
    (ROUND9_ROOT / "metrics" / "forensics_meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    write_forensics_md(payload, cells)
    return payload


__all__ = ["run_forensics"]
