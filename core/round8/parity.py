"""R8_00 必须逐笔复现 R4_E_top1_minus_median。失败则 STOP。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND8_ROOT, assert_project_result_path, ensure_round8_tree, round8_dir
from ..replay_round7.inject import inject_round7_into_overlay_caches, load_locked_alpha
from ..stock_readonly import load_prepared_market
from .contract import (
    ANCHOR_DIR,
    PARITY_ABS_NAV,
    PARENT_CODE,
    R8_00,
    TARGET_CAGR,
    TARGET_MAXDD,
    TARGET_N_TRADES,
    TARGET_SHARPE,
    TARGET_TOTAL_RETURN,
)
from .overlays import exact_v2_date_mult
from .registry import get_spec
from .runner import run_spec

IDENTITY_KEYS = (
    "signal_date",
    "instrument_id",
    "entry_date",
    "actual_exit_date",
    "initial_quantity",
)
NUMERIC_FIELDS = {
    "score": 1e-12,
    "allocation_multiplier": 1e-12,
    "target_name_budget": 1e-4,
    "entry_portfolio_nav": 1e-4,
    "entry_price": 1e-8,
    "exit_price": 1e-8,
    "entry_reference_price": 1e-8,
    "exit_reference_price": 1e-8,
    "buy_cost": 1e-6,
    "sell_proceeds": 1e-6,
    "ret": 1e-10,
    "buy_commission": 1e-6,
    "stamp_tax": 1e-6,
}
PARITY_MD = ROUND8_ROOT / "01_baseline_v2_parity.md"


def _norm_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("signal_date", "entry_date", "actual_exit_date", "planned_exit_date"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col]).dt.normalize()
    if "instrument_id" in out.columns:
        out["instrument_id"] = out["instrument_id"].astype(str)
    if "initial_quantity" in out.columns:
        out["initial_quantity"] = pd.to_numeric(out["initial_quantity"], errors="coerce").fillna(0).astype(int)
    if "score" in out.columns:
        out["_rank"] = out.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    return out


def compare_ledgers(
    local_trades: pd.DataFrame,
    anchor_trades: pd.DataFrame,
    local_nav: pd.DataFrame,
    anchor_nav: pd.DataFrame,
) -> dict[str, Any]:
    lt = _norm_frame(local_trades)
    st = _norm_frame(anchor_trades)
    keys = [c for c in IDENTITY_KEYS if c in lt.columns and c in st.columns]
    lt_i = lt.sort_values(keys).reset_index(drop=True)
    st_i = st.sort_values(keys).reset_index(drop=True)
    field_mismatches: dict[str, int] = {}
    samples: list[Any] = []

    id_mismatch = 0
    if len(lt_i) != len(st_i):
        id_mismatch = abs(len(lt_i) - len(st_i))
        samples.append({"reason": "len", "r8": int(len(lt_i)), "anchor": int(len(st_i))})
    elif keys:
        diff = lt_i[keys].ne(st_i[keys])
        id_mismatch = int(diff.any(axis=1).sum())
        if id_mismatch:
            bad = diff.any(axis=1)
            sample = pd.concat(
                [lt_i.loc[bad, keys].head(8), st_i.loc[bad, keys].head(8)],
                axis=1,
                keys=["r8", "anchor"],
            )
            samples = json.loads(sample.to_json(orient="records", date_format="iso"))

    if id_mismatch == 0 and len(lt_i) == len(st_i):
        for col, tol in NUMERIC_FIELDS.items():
            if col not in lt_i.columns or col not in st_i.columns:
                continue
            a = pd.to_numeric(lt_i[col], errors="coerce")
            b = pd.to_numeric(st_i[col], errors="coerce")
            both_na = a.isna() & b.isna()
            delta = (a - b).abs()
            bad = (~both_na) & (delta > tol)
            n_bad = int(bad.sum())
            if n_bad:
                field_mismatches[col] = n_bad
                if len(samples) < 8:
                    rows = lt_i.loc[bad, keys + [col]].head(3).copy()
                    rows["anchor"] = b.loc[bad].head(3).to_numpy()
                    samples.extend(json.loads(rows.to_json(orient="records", date_format="iso")))
        if "_rank" in lt_i.columns and "_rank" in st_i.columns:
            n_rank = int((lt_i["_rank"] != st_i["_rank"]).sum())
            if n_rank:
                field_mismatches["rank"] = n_rank

    ln = local_nav.copy()
    sn = anchor_nav.copy()
    ln["trade_date"] = pd.to_datetime(ln["trade_date"]).dt.normalize()
    sn["trade_date"] = pd.to_datetime(sn["trade_date"]).dt.normalize()
    merged = ln.merge(sn, on="trade_date", suffixes=("_r8", "_anchor"))
    nav_abs = (merged["total_asset_r8"].astype(float) - merged["total_asset_anchor"].astype(float)).abs()
    nav_max = float(nav_abs.max()) if len(nav_abs) else float("inf")
    n_field = int(sum(field_mismatches.values()))
    passed = (
        id_mismatch == 0
        and n_field == 0
        and nav_max <= PARITY_ABS_NAV
        and len(lt_i) == TARGET_N_TRADES
        and len(st_i) == TARGET_N_TRADES
    )
    return {
        "trade_rows_r8": int(len(lt_i)),
        "trade_rows_anchor": int(len(st_i)),
        "identity_mismatches": int(id_mismatch),
        "field_mismatches": field_mismatches,
        "field_mismatch_rows": n_field,
        "nav_rows_compared": int(len(merged)),
        "nav_max_abs": nav_max,
        "sample_mismatches": samples[:8],
        "passed": bool(passed),
    }


def _enrich_parity_table(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    t["signal_date"] = pd.to_datetime(t["signal_date"])
    t["rank"] = t.groupby("signal_date")["score"].rank(method="first", ascending=False)
    try:
        from .geometry import build_geometry, eligible_heads

        geo = build_geometry()
        heads = eligible_heads()
        t = t.merge(geo.reset_index(), on="signal_date", how="left")
        head_keys = heads[["signal_date", "instrument_id", "five_score", "ten_score", "fused_score"]].copy()
        head_keys["signal_date"] = pd.to_datetime(head_keys["signal_date"])
        t = t.merge(head_keys, on=["signal_date", "instrument_id"], how="left", suffixes=("", "_head"))
    except Exception as exc:  # enrichment must not veto ledger pass
        t["enrich_error"] = repr(exc)
    alpha = load_locked_alpha()
    date_mult, _ = exact_v2_date_mult(alpha)
    t["fused_score"] = t["score"]
    t["market_x_confidence"] = t["signal_date"].map(date_mult)
    nav = pd.to_numeric(t.get("entry_portfolio_nav"), errors="coerce")
    mult = pd.to_numeric(t.get("allocation_multiplier"), errors="coerce")
    t["Top1_budget"] = np.where(t["rank"].eq(1), nav * 0.30 * mult, np.nan)
    t["Top2_budget"] = np.where(t["rank"].eq(2), nav * 0.10 * mult, np.nan)
    keep = [
        "signal_date", "instrument_id", "rank",
        "five_score", "ten_score", "fused_score",
        "candidate_count", "top1_score", "top2_score", "median", "top1_minus_median",
        "market_x_confidence", "allocation_multiplier",
        "target_name_budget", "Top1_budget", "Top2_budget",
        "entry_date", "entry_price", "initial_quantity",
        "actual_exit_date", "exit_price", "buy_cost", "ret",
        "entry_portfolio_nav",
    ]
    cols = [c for c in keep if c in t.columns]
    return t[cols]


def _pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def write_parity_markdown(payload: dict[str, Any]) -> None:
    ensure_round8_tree()
    obs = payload.get("observed") or {}
    tgt = payload.get("targets") or {}
    ledger = payload.get("ledger") or {}
    status = payload.get("status")
    fields = ledger.get("field_mismatches") or {}
    field_rows = "\n".join(f"| `{k}` | {v} |" for k, v in fields.items()) or "| （无） | 0 |"
    body = f"""# Round 8 · R8_00 vs R4_E 逐笔 parity

锚点：`{PARENT_CODE}` @ `{ANCHOR_DIR}`。
正式账本：`local_executor` → Stock `fixed_sleeve_backtest`。

## 结果：**{status}**

{"复现成功，后续 B/C/D/… 可以开工。" if payload.get("passed") else "**STOP。禁止发明后续 stage。** 先看 mismatch，再改 R8_00 契约/实现。"}

## 指标

| 指标 | R4_E（目标） | R8_00（观察） | 判定 |
|---|---:|---:|---|
| CAGR | {_pct(tgt.get("cagr"))} | {_pct(obs.get("cagr"))} | {"PASS" if payload.get("metric_ok") else "FAIL"} |
| Sharpe | {float(tgt.get("sharpe") or 0):.3f} | {float(obs.get("sharpe") or 0):.3f} | 同上 |
| MaxDD | {_pct(tgt.get("max_drawdown"))} | {_pct(obs.get("max_drawdown"))} | 同上 |
| 笔数 | {tgt.get("trades")} | {obs.get("trades")} | 同上 |
| 累计 | {tgt.get("total_return")} | {obs.get("total_return")} | — |

精确值：目标 CAGR=`{tgt.get("cagr")}` Sharpe=`{tgt.get("sharpe")}` MaxDD=`{tgt.get("max_drawdown")}`；
观察 CAGR=`{obs.get("cagr")}` Sharpe=`{obs.get("sharpe")}` MaxDD=`{obs.get("max_drawdown")}`。

## 逐笔字段

核对：`signal_date, instrument_id, rank, score, allocation_multiplier, target_name_budget, entry/exit, cost, NAV`。

| 项 | 值 |
|---|---|
| R8 笔数 | {ledger.get("trade_rows_r8")} |
| 锚点笔数 | {ledger.get("trade_rows_anchor")} |
| 身份键 mismatch | {ledger.get("identity_mismatches")} |
| 数值字段 mismatch 行合计 | {ledger.get("field_mismatch_rows")} |
| NAV 对比行 | {ledger.get("nav_rows_compared")} |
| NAV max abs | {ledger.get("nav_max_abs")} |
| NAV 容差 | {PARITY_ABS_NAV:g} |
| ledger.passed | {ledger.get("passed")} |
| metric_ok | {payload.get("metric_ok")} |

分字段 mismatch 行数：

| 字段 | 行数 |
|---|---:|
{field_rows}

样本 mismatch（最多 8 条）：

```json
{json.dumps(ledger.get("sample_mismatches") or [], ensure_ascii=False, indent=2, default=str)}
```

enrich 表：`result/round8/{R8_00}/parity_fields.parquet`（含 median / five / ten / 日度乘子；enrich 失败不否决 ledger）。
"""
    assert_project_result_path(PARITY_MD)
    PARITY_MD.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")


def run_parity(*, resume: bool = True) -> dict[str, Any]:
    ensure_round8_tree()
    inject_round7_into_overlay_caches()
    dest = round8_dir(R8_00)
    dest.mkdir(parents=True, exist_ok=True)
    loaded = load_prepared_market(warmup_ma5=True)
    payload_run = run_spec(get_spec(R8_00), prepared_market=loaded["prepared"], market_meta=loaded, resume=resume)
    local_trades = pd.read_parquet(dest / "trades.parquet")
    local_nav = pd.read_parquet(dest / "nav.parquet")
    anchor_trades = pd.read_parquet(ANCHOR_DIR / "trades.parquet")
    anchor_nav = pd.read_parquet(ANCHOR_DIR / "nav.parquet")
    cmp = compare_ledgers(local_trades, anchor_trades, local_nav, anchor_nav)
    report = json.loads((dest / "report.json").read_text(encoding="utf-8"))
    metrics = report.get("metrics") or {}
    metric_ok = (
        abs(float(metrics.get("annualized_return", 0)) - TARGET_CAGR) < 1e-8
        and abs(float(metrics.get("sharpe_zero_rate", 0)) - TARGET_SHARPE) < 1e-8
        and abs(float(metrics.get("max_drawdown", 0)) - TARGET_MAXDD) < 1e-8
        and int(metrics.get("completed_trades", -1)) == TARGET_N_TRADES
    )
    passed = bool(cmp["passed"] and metric_ok)
    try:
        table = _enrich_parity_table(local_trades)
        table_path = dest / "parity_fields.parquet"
        assert_project_result_path(table_path)
        table.to_parquet(table_path, index=False)
    except Exception as exc:
        (dest / "parity_fields_error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
    out = {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "anchor": PARENT_CODE,
        "anchor_dir": str(ANCHOR_DIR),
        "targets": {
            "cagr": TARGET_CAGR,
            "sharpe": TARGET_SHARPE,
            "max_drawdown": TARGET_MAXDD,
            "trades": TARGET_N_TRADES,
            "total_return": TARGET_TOTAL_RETURN,
        },
        "observed": {
            "cagr": metrics.get("annualized_return"),
            "sharpe": metrics.get("sharpe_zero_rate"),
            "max_drawdown": metrics.get("max_drawdown"),
            "trades": metrics.get("completed_trades"),
            "total_return": metrics.get("total_return"),
        },
        "metric_ok": metric_ok,
        "ledger": cmp,
        "run": {k: payload_run.get(k) for k in ("status", "code", "out_dir")},
    }
    path = dest / "parity.json"
    assert_project_result_path(path)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    write_parity_markdown(out)
    return out


def load_parity_status() -> dict[str, Any]:
    path = round8_dir(R8_00) / "parity.json"
    if not path.is_file():
        return {"passed": False, "status": "missing"}
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["compare_ledgers", "load_parity_status", "run_parity", "write_parity_markdown"]
