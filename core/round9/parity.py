"""R9_00 必须逐笔复现 R8_E02。失败则 STOP，禁止发明后续 stage。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..paths import assert_project_result_path, ensure_round9_tree, round9_dir
from ..replay_round7.inject import inject_round7_into_overlay_caches
from ..stock_readonly import load_prepared_market
from .contract import (
    ANCHOR_DIR,
    DISAGREE_ALIAS,
    DISAGREE_LABEL,
    DISAGREE_MULT,
    PARITY_ABS_NAV,
    PARITY_MD,
    PARENT_CODE,
    R9_00,
    RANK_NAV,
    TARGET_CAGR,
    TARGET_LOYO_MED,
    TARGET_LOYO_WORST,
    TARGET_MAXDD,
    TARGET_N_TRADES,
    TARGET_SHARPE,
    TARGET_TOTAL_RETURN,
)
from .geometry import build_geometry, eligible_heads
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
EXTRA_NUMERIC = {
    "five_rank": 0.0,
    "ten_rank": 0.0,
    "fused_rank": 0.0,
    "engine_rank": 0.0,
    "Top2_budget": 1e-4,
    "Top1_budget": 1e-4,
}


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
        samples.append({"reason": "len", "r9": int(len(lt_i)), "anchor": int(len(st_i))})
    elif keys:
        diff = lt_i[keys].ne(st_i[keys])
        id_mismatch = int(diff.any(axis=1).sum())
        if id_mismatch:
            bad = diff.any(axis=1)
            sample = pd.concat(
                [lt_i.loc[bad, keys].head(8), st_i.loc[bad, keys].head(8)],
                axis=1,
                keys=["r9", "anchor"],
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
    merged = ln.merge(sn, on="trade_date", suffixes=("_r9", "_anchor"))
    nav_abs = (merged["total_asset_r9"].astype(float) - merged["total_asset_anchor"].astype(float)).abs()
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
        "trade_rows_r9": int(len(lt_i)),
        "trade_rows_anchor": int(len(st_i)),
        "identity_mismatches": int(id_mismatch),
        "field_mismatches": field_mismatches,
        "field_mismatch_rows": n_field,
        "nav_rows_compared": int(len(merged)),
        "nav_max_abs": nav_max,
        "sample_mismatches": samples[:8],
        "passed": bool(passed),
    }


def _enrich(trades: pd.DataFrame) -> pd.DataFrame:
    t = _norm_frame(trades)
    geo = build_geometry()
    heads = eligible_heads().copy()
    heads["signal_date"] = pd.to_datetime(heads["signal_date"]).dt.normalize()
    heads["instrument_id"] = heads["instrument_id"].astype(str)
    heads["five_rank"] = heads.groupby("signal_date")["five_score"].rank(method="first", ascending=False)
    heads["ten_rank"] = heads.groupby("signal_date")["ten_score"].rank(method="first", ascending=False)
    heads["fused_rank"] = heads.groupby("signal_date")["fused_score"].rank(method="first", ascending=False)
    geo_r = geo.reset_index()
    geo_r["signal_date"] = pd.to_datetime(geo_r["signal_date"]).dt.normalize()
    t = t.merge(
        heads[["signal_date", "instrument_id", "five_score", "ten_score", "fused_score", "five_rank", "ten_rank", "fused_rank"]],
        on=["signal_date", "instrument_id"],
        how="left",
        suffixes=("", "_head"),
    )
    t = t.merge(
        geo_r[[
            "signal_date", "head_state", "top1_id", "top2_id",
            "five_top1", "ten_top1", "candidate_count",
            "top1_minus_median", "top2_minus_median_p250",
        ]],
        on="signal_date",
        how="left",
    )
    t["agreement"] = t["head_state"].astype(str)
    t["agreement_alias"] = t["agreement"].map(DISAGREE_LABEL)
    t["disagree_mult"] = t["agreement"].map(DISAGREE_MULT).astype(float)
    nav = pd.to_numeric(t.get("entry_portfolio_nav"), errors="coerce")
    mult = pd.to_numeric(t.get("allocation_multiplier"), errors="coerce")
    iid = t["instrument_id"].astype(str)
    t["engine_rank"] = np.where(
        iid.eq(t["top1_id"].astype(str)),
        1,
        np.where(iid.eq(t["top2_id"].astype(str)), 2, np.nan),
    )
    t["Top1_budget"] = np.where(t["engine_rank"].eq(1), nav * float(RANK_NAV[0]) * mult, np.nan)
    t["Top2_budget"] = np.where(t["engine_rank"].eq(2), nav * float(RANK_NAV[1]) * mult, np.nan)
    t["trade_rank"] = t["_rank"]
    return t


def _top1_lock_check(local: pd.DataFrame, anchor: pd.DataFrame) -> dict[str, Any]:
    """每天 geometry Top1 票必须等于 E02，否则 FAIL_TOP1_MUTATION。"""
    def _daily_top1(frame: pd.DataFrame) -> pd.Series:
        work = frame.copy()
        work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
        if "top1_id" in work.columns:
            return (
                work.drop_duplicates("signal_date")
                .set_index("signal_date")["top1_id"]
                .astype(str)
            )
        work["_rank"] = work.groupby("signal_date")["score"].rank(method="first", ascending=False)
        return work.loc[work["_rank"].eq(1)].drop_duplicates("signal_date").set_index("signal_date")["instrument_id"].astype(str)

    a = _daily_top1(local)
    b = _daily_top1(anchor)
    idx = a.index.union(b.index)
    bad = a.reindex(idx).fillna("") != b.reindex(idx).fillna("")
    n_bad = int(bad.sum())
    samples = []
    if n_bad:
        for day in idx[bad][:8]:
            samples.append({
                "signal_date": str(pd.Timestamp(day).date()),
                "r9": a.get(day, ""),
                "e02": b.get(day, ""),
            })
    status = "PASS" if n_bad == 0 else "FAIL_TOP1_MUTATION"
    return {
        "status": status,
        "n_days": int(len(idx)),
        "mismatches": n_bad,
        "samples": samples,
        "passed": n_bad == 0,
    }


def _budget_check(enriched: pd.DataFrame) -> dict[str, Any]:
    t = enriched.copy()
    nav = pd.to_numeric(t["entry_portfolio_nav"], errors="coerce")
    mult = pd.to_numeric(t["allocation_multiplier"], errors="coerce")
    budget = pd.to_numeric(t["target_name_budget"], errors="coerce")
    rank = pd.to_numeric(t["engine_rank"], errors="coerce")
    expected = pd.Series(np.nan, index=t.index, dtype=float)
    expected = expected.where(~rank.eq(1), nav * float(RANK_NAV[0]) * mult)
    expected = expected.where(~rank.eq(2), nav * float(RANK_NAV[1]) * mult)
    delta = (budget - expected).abs()
    mask = expected.notna() & budget.notna()
    n_bad = int((mask & (delta > 1e-4)).sum())
    n_top2 = int(rank.eq(2).sum())
    n_top2_budget = int(t["Top2_budget"].notna().sum())
    return {
        "n_compared": int(mask.sum()),
        "n_bad": n_bad,
        "n_engine_rank2": n_top2,
        "n_top2_budget": n_top2_budget,
        "max_abs": float(delta.loc[mask].max()) if mask.any() else 0.0,
        "passed": n_bad == 0,
    }


def _agreement_check(enriched: pd.DataFrame) -> dict[str, Any]:
    state = enriched["head_state"].astype(str)
    bad_state = ~state.isin(["A", "B", "C"]) & state.notna() & ~state.eq("nan")
    n_missing = int(state.isin(["", "nan", "None"]).sum() + state.isna().sum())
    counts = {k: int((state == k).sum()) for k in ("A", "B", "C")}
    alias_ok = all(DISAGREE_ALIAS[k] in DISAGREE_MULT for k in ("HIGH", "MEDIUM", "LOW"))
    return {
        "counts": counts,
        "n_missing": n_missing,
        "n_invalid": int(bad_state.sum()),
        "alias": dict(DISAGREE_ALIAS),
        "mult": dict(DISAGREE_MULT),
        "passed": int(bad_state.sum()) == 0 and alias_ok,
    }


def compare_enriched(local: pd.DataFrame, anchor: pd.DataFrame) -> dict[str, Any]:
    keys = [c for c in IDENTITY_KEYS if c in local.columns and c in anchor.columns]
    lt = local.sort_values(keys).reset_index(drop=True)
    st = anchor.sort_values(keys).reset_index(drop=True)
    field_mismatches: dict[str, int] = {}
    samples: list[Any] = []
    if len(lt) != len(st):
        return {
            "passed": False,
            "reason": "len",
            "r9": int(len(lt)),
            "e02": int(len(st)),
            "field_mismatches": {"len": abs(len(lt) - len(st))},
        }
    for col, tol in EXTRA_NUMERIC.items():
        if col not in lt.columns or col not in st.columns:
            field_mismatches[col] = -1
            continue
        a = pd.to_numeric(lt[col], errors="coerce")
        b = pd.to_numeric(st[col], errors="coerce")
        both_na = a.isna() & b.isna()
        if tol == 0.0:
            bad = (~both_na) & ((a.fillna(-1) != b.fillna(-1)) | (a.isna() != b.isna()))
        else:
            bad = (~both_na) & ((a - b).abs() > tol)
        n_bad = int(bad.sum())
        if n_bad:
            field_mismatches[col] = n_bad
            if len(samples) < 8:
                rows = lt.loc[bad, keys + [col]].head(3).copy()
                rows["e02"] = b.loc[bad].head(3).to_numpy()
                samples.extend(json.loads(rows.to_json(orient="records", date_format="iso")))
    if "head_state" in lt.columns and "head_state" in st.columns:
        n_state = int((lt["head_state"].astype(str) != st["head_state"].astype(str)).sum())
        if n_state:
            field_mismatches["head_state"] = n_state
    passed = len(field_mismatches) == 0
    return {
        "passed": passed,
        "field_mismatches": field_mismatches,
        "sample_mismatches": samples[:8],
    }


def _pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def write_parity_markdown(payload: dict[str, Any]) -> None:
    ensure_round9_tree()
    obs = payload.get("observed") or {}
    tgt = payload.get("targets") or {}
    ledger = payload.get("ledger") or {}
    extra = payload.get("extra") or {}
    status = payload.get("status")
    fields = ledger.get("field_mismatches") or {}
    extra_fields = (extra.get("enriched") or {}).get("field_mismatches") or {}
    field_rows = "\n".join(f"| `{k}` | {v} |" for k, v in fields.items()) or "| （无） | 0 |"
    extra_rows = "\n".join(f"| `{k}` | {v} |" for k, v in extra_fields.items()) or "| （无） | 0 |"
    top1 = extra.get("top1_lock") or {}
    budget = extra.get("budget") or {}
    agree = extra.get("agreement") or {}
    body = f"""# Round 9 · R9_00 vs R8_E02 逐笔 parity

锚点：`{PARENT_CODE}` @ `{ANCHOR_DIR}`。
正式账本：`local_executor` → Stock `fixed_sleeve_backtest`。
Parent = **baseline_v3 = R8_E02**。分歧状态是 **A/B/C**（HIGH=A，MEDIUM=B，LOW=C）。
E02 **只乘 Top2**：A→1.00 / B→0.75 / C→0.50；Top1 **LOCK**。

## 结果：**{status}**

{"复现成功。后续 stage 必须等本 PHASE 结束、manifest 冻结后再开。" if payload.get("passed") else "**STOP。禁止发明后续 stage。** 先看 mismatch，再改 R9_00 契约/实现。本阶段不冻结 manifest。"}

## 指标

| 指标 | R8_E02（目标） | R9_00（观察） | 判定 |
|---|---:|---:|---|
| CAGR | {_pct(tgt.get("cagr"))} | {_pct(obs.get("cagr"))} | {"PASS" if payload.get("metric_ok") else "FAIL"} |
| Sharpe | {float(tgt.get("sharpe") or 0):.3f} | {float(obs.get("sharpe") or 0):.3f} | 同上 |
| MaxDD | {_pct(tgt.get("max_drawdown"))} | {_pct(obs.get("max_drawdown"))} | 同上 |
| 笔数 | {tgt.get("trades")} | {obs.get("trades")} | 同上 |
| 累计 | {tgt.get("total_return")} | {obs.get("total_return")} | — |
| LOYO med | {_pct(tgt.get("loyo_median"))} | {_pct(obs.get("loyo_median"))} | — |
| LOYO worst | {_pct(tgt.get("loyo_worst"))} | {_pct(obs.get("loyo_worst"))} | — |

精确值：目标 CAGR=`{tgt.get("cagr")}` Sharpe=`{tgt.get("sharpe")}` MaxDD=`{tgt.get("max_drawdown")}` 笔数=`{tgt.get("trades")}`；
观察 CAGR=`{obs.get("cagr")}` Sharpe=`{obs.get("sharpe")}` MaxDD=`{obs.get("max_drawdown")}` 笔数=`{obs.get("trades")}`。

## 逐笔字段（身份 + 价格费用 NAV）

核对：`signal_date, instrument_id, rank, score, allocation_multiplier, target_name_budget, entry/exit, cost, NAV`。

| 项 | 值 |
|---|---|
| R9 笔数 | {ledger.get("trade_rows_r8") or ledger.get("trade_rows_r9")} |
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

## 计划要求的额外核对

### Top1 LOCK

`{top1.get("status")}`；天数 {top1.get("n_days")}；mismatch {top1.get("mismatches")}。

### five / ten / fused rank 与 agreement A/B/C

| 字段 | mismatch 行 |
|---|---:|
{extra_rows}

agreement 计数（R9 enrich）：A={agree.get("counts", {}).get("A")} B={agree.get("counts", {}).get("B")} C={agree.get("counts", {}).get("C")}；
invalid={agree.get("n_invalid")} missing={agree.get("n_missing")}。
别名固定：HIGH=A，MEDIUM=B，LOW=C。乘子只作用于 **fused rank=2**：A=1.00 B=0.75 C=0.50。

### Top2 最终预算

fused rank=2 与 geometry `top2_id` 对齐后，期望 `entry_portfolio_nav × 0.10 × allocation_multiplier`。
比较行 {budget.get("n_compared")}；坏行 {budget.get("n_bad")}；max abs {budget.get("max_abs")}；
engine rank2 笔数 {budget.get("n_engine_rank2") or budget.get("n_fused_rank2")}。判定：{"PASS" if budget.get("passed") else "FAIL"}。

样本 mismatch（最多 8 条）：

```json
{json.dumps(ledger.get("sample_mismatches") or extra.get("enriched", {}).get("sample_mismatches") or [], ensure_ascii=False, indent=2, default=str)}
```

enrich 表：`result/round9/{R9_00}/parity_fields.parquet`（含 five/ten/fused rank、agreement A/B/C、Top1/Top2 预算）。
"""
    assert_project_result_path(PARITY_MD)
    PARITY_MD.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")


def run_parity(*, resume: bool = True) -> dict[str, Any]:
    ensure_round9_tree()
    inject_round7_into_overlay_caches()
    dest = round9_dir(R9_00)
    dest.mkdir(parents=True, exist_ok=True)
    loaded = load_prepared_market(warmup_ma5=True)
    payload_run = run_spec(get_spec(R9_00), prepared_market=loaded["prepared"], market_meta=loaded, resume=resume)
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
    extra: dict[str, Any] = {}
    try:
        local_en = _enrich(local_trades)
        anchor_en = _enrich(anchor_trades)
        extra["enriched"] = compare_enriched(local_en, anchor_en)
        extra["top1_lock"] = _top1_lock_check(local_en, anchor_en)
        extra["budget"] = _budget_check(local_en)
        extra["agreement"] = _agreement_check(local_en)
        extra["anchor_budget"] = _budget_check(anchor_en)
        extra["anchor_agreement"] = _agreement_check(anchor_en)
        table_path = dest / "parity_fields.parquet"
        assert_project_result_path(table_path)
        keep = [
            "signal_date", "instrument_id", "trade_rank",
            "five_score", "ten_score", "fused_score",
            "five_rank", "ten_rank", "fused_rank", "engine_rank",
            "head_state", "agreement", "agreement_alias", "disagree_mult",
            "top1_id", "top2_id", "five_top1", "ten_top1",
            "candidate_count", "allocation_multiplier",
            "target_name_budget", "Top1_budget", "Top2_budget",
            "entry_date", "entry_price", "initial_quantity",
            "actual_exit_date", "exit_price", "buy_cost", "ret",
            "entry_portfolio_nav",
        ]
        cols = [c for c in keep if c in local_en.columns]
        local_en[cols].to_parquet(table_path, index=False)
        extra_ok = (
            bool(extra["enriched"].get("passed"))
            and bool(extra["top1_lock"].get("passed"))
            and bool(extra["agreement"].get("passed"))
            and bool(extra["budget"].get("passed"))
        )
    except Exception as exc:
        extra = {"error": repr(exc)}
        extra_ok = False
        (dest / "parity_fields_error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
    passed = bool(cmp["passed"] and metric_ok and extra_ok)
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
            "loyo_median": TARGET_LOYO_MED,
            "loyo_worst": TARGET_LOYO_WORST,
        },
        "observed": {
            "cagr": metrics.get("annualized_return"),
            "sharpe": metrics.get("sharpe_zero_rate"),
            "max_drawdown": metrics.get("max_drawdown"),
            "trades": metrics.get("completed_trades"),
            "total_return": metrics.get("total_return"),
            "loyo_median": metrics.get("loyo_median"),
            "loyo_worst": metrics.get("loyo_worst"),
        },
        "metric_ok": metric_ok,
        "extra_ok": extra_ok,
        "ledger": cmp,
        "extra": extra,
        "run": {k: payload_run.get(k) for k in ("status", "code", "out_dir")},
    }
    path = dest / "parity.json"
    assert_project_result_path(path)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    write_parity_markdown(out)
    return out


def load_parity_status() -> dict[str, Any]:
    path = round9_dir(R9_00) / "parity.json"
    if not path.is_file():
        return {"passed": False, "status": "missing"}
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["load_parity_status", "run_parity", "write_parity_markdown"]
