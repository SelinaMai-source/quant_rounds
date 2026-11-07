"""v1–v4 对已有账本逐笔/逐日对齐。失败 = REPRODUCIBILITY_FAIL，停止后续 stage。"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import pandas as pd

from ..paths import ROUND11_ROOT, assert_project_result_path, ensure_round11_tree
from ..replay_round7.inject import inject_round7_into_overlay_caches
from ..stock_readonly import load_prepared_market, sha256_file
from .contract import (
    BASELINES,
    BaselineSpec,
    PARITY_ABS_NAV,
    PARITY_MD,
    PARITY_METRIC_ABS,
    SNAPSHOT_FINGERPRINT,
    SNAPSHOT_ID,
    artifact_file_hashes,
    replay_dir,
    source_tree_sha256,
)
from .runner import run_backtest_in_memory, run_spec

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
CHEAP_RUN_SECONDS = 180.0
PARITY_JSON = ROUND11_ROOT / "parity" / "parity.json"


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
    *,
    target_n_trades: int,
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
        samples.append({"reason": "len", "local": int(len(lt_i)), "anchor": int(len(st_i))})
    elif keys:
        diff = lt_i[keys].ne(st_i[keys])
        id_mismatch = int(diff.any(axis=1).sum())
        if id_mismatch:
            bad = diff.any(axis=1)
            sample = pd.concat(
                [lt_i.loc[bad, keys].head(8), st_i.loc[bad, keys].head(8)],
                axis=1,
                keys=["local", "anchor"],
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
    merged = ln.merge(sn, on="trade_date", suffixes=("_local", "_anchor"))
    nav_abs = (merged["total_asset_local"].astype(float) - merged["total_asset_anchor"].astype(float)).abs()
    nav_max = float(nav_abs.max()) if len(nav_abs) else float("inf")
    n_field = int(sum(field_mismatches.values()))
    passed = (
        id_mismatch == 0
        and n_field == 0
        and nav_max <= PARITY_ABS_NAV
        and len(lt_i) == int(target_n_trades)
        and len(st_i) == int(target_n_trades)
    )
    return {
        "trade_rows_local": int(len(lt_i)),
        "trade_rows_anchor": int(len(st_i)),
        "identity_mismatches": int(id_mismatch),
        "field_mismatches": field_mismatches,
        "field_mismatch_rows": n_field,
        "nav_rows_compared": int(len(merged)),
        "nav_max_abs": nav_max,
        "sample_mismatches": samples[:8],
        "passed": bool(passed),
    }


def _observed_metrics(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics") or {}
    return {
        "cagr": metrics.get("annualized_return"),
        "sharpe": metrics.get("sharpe_zero_rate"),
        "max_drawdown": metrics.get("max_drawdown"),
        "completed_trades": metrics.get("completed_trades"),
        "total_return": metrics.get("total_return"),
        "loyo_median": metrics.get("loyo_median"),
        "loyo_worst": metrics.get("loyo_worst"),
    }


def _metrics_ok(observed: dict[str, Any], targets: dict[str, Any]) -> bool:
    try:
        return (
            abs(float(observed.get("cagr") or 0) - float(targets["cagr"])) < PARITY_METRIC_ABS
            and abs(float(observed.get("sharpe") or 0) - float(targets["sharpe"])) < PARITY_METRIC_ABS
            and abs(float(observed.get("max_drawdown") or 0) - float(targets["max_drawdown"])) < PARITY_METRIC_ABS
            and int(observed.get("completed_trades") or -1) == int(targets["completed_trades"])
        )
    except (TypeError, ValueError, KeyError):
        return False


def _series_hash(nav: pd.DataFrame, trades: pd.DataFrame) -> dict[str, str]:
    n = nav.copy()
    n["trade_date"] = pd.to_datetime(n["trade_date"]).dt.normalize()
    arr = n.sort_values("trade_date")["total_asset"].astype("float64").to_numpy()
    t = _norm_frame(trades)
    keys = [c for c in IDENTITY_KEYS if c in t.columns]
    ident = hashlib.sha256()
    ident.update(t.sort_values(keys)[keys].to_csv(index=False).encode("utf-8"))
    return {
        "nav_total_asset": hashlib.sha256(arr.tobytes()).hexdigest(),
        "trade_identity": ident.hexdigest(),
    }


def _load_anchor(spec: BaselineSpec) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    trades = pd.read_parquet(spec.anchor_dir / "trades.parquet")
    nav = pd.read_parquet(spec.anchor_dir / "nav.parquet")
    report = json.loads((spec.anchor_dir / "report.json").read_text(encoding="utf-8"))
    return trades, nav, report


def _run_determinism(
    spec: BaselineSpec,
    *,
    prepared_market,
    first_nav: pd.DataFrame,
    first_trades: pd.DataFrame,
    first_elapsed: float,
    n_runs: int,
) -> dict[str, Any]:
    hashes = [_series_hash(first_nav, first_trades)]
    extra_elapsed: list[float] = []
    cheap = first_elapsed <= CHEAP_RUN_SECONDS
    planned = int(n_runs) if cheap else 1
    if planned < 1:
        planned = 1
    for _ in range(1, planned):
        t0 = time.perf_counter()
        result = run_backtest_in_memory(spec, prepared_market=prepared_market)
        extra_elapsed.append(time.perf_counter() - t0)
        hashes.append(_series_hash(result["nav"], result["trades"]))
    unique = {json.dumps(item, sort_keys=True) for item in hashes}
    return {
        "requested_runs": int(n_runs),
        "actual_runs": len(hashes),
        "cheap_enough_for_3": cheap,
        "first_elapsed_sec": first_elapsed,
        "extra_elapsed_sec": extra_elapsed,
        "hashes": hashes,
        "all_identical": len(unique) == 1,
        "mode": "three_runs" if len(hashes) >= 3 else "hash_existing_plus_one_replay",
    }


def _pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def write_parity_markdown(payload: dict[str, Any]) -> None:
    ensure_round11_tree()
    status = payload.get("status")
    passed = bool(payload.get("passed"))
    tree = payload.get("source_tree_sha256") or {}
    rows = []
    for key in ("v1", "v2", "v3", "v4"):
        item = (payload.get("baselines") or {}).get(key) or {}
        tgt = item.get("targets") or {}
        obs = item.get("observed") or {}
        ledger = item.get("ledger") or {}
        det = item.get("determinism") or {}
        rows.append(
            "| {ver} | {code} | {tcagr} | {ocagr} | {tsh:.4f} | {osh:.4f} | {tn} | {on} | {nav} | {det} | {st} |".format(
                ver=key,
                code=item.get("code"),
                tcagr=_pct(tgt.get("cagr")),
                ocagr=_pct(obs.get("cagr")),
                tsh=float(tgt.get("sharpe") or 0),
                osh=float(obs.get("sharpe") or 0),
                tn=tgt.get("completed_trades"),
                on=obs.get("completed_trades"),
                nav=ledger.get("nav_max_abs"),
                det=det.get("mode"),
                st=item.get("status"),
            )
        )
    body = f"""# Round 11 · v1–v4 可复现 / parity

正式账本：`local_executor` → Stock `fixed_sleeve_backtest`。
复现产物只写 `result/round11/parity/`，**不覆盖** legacy。
容差：指标 `{PARITY_METRIC_ABS:g}`，NAV abs≤`{PARITY_ABS_NAV:g}`。非浮点误差 = `REPRODUCIBILITY_FAIL`。

## 结果：**{status}**

{"四条基线逐笔/逐日对齐通过。本 PHASE 到此结束，不自动开 cost / placebo / AlphaForge / dossier。" if passed else "**REPRODUCIBILITY_FAIL。停止后续 stage。** 不为过关改策略。"}

snapshot = `{SNAPSHOT_ID}` / `{SNAPSHOT_FINGERPRINT}`。
source_tree_sha256 = `{tree.get("tree")}`（无 git commit）。

## 对照表

| 版本 | 锚点 | 目标 CAGR | 观察 CAGR | 目标 Sharpe | 观察 Sharpe | 目标笔数 | 观察笔数 | NAV max abs | 确定性 | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
{chr(10).join(rows)}

## 分版本

"""
    for key in ("v1", "v2", "v3", "v4"):
        item = (payload.get("baselines") or {}).get(key) or {}
        tgt = item.get("targets") or {}
        obs = item.get("observed") or {}
        ledger = item.get("ledger") or {}
        det = item.get("determinism") or {}
        fields = ledger.get("field_mismatches") or {}
        field_rows = "\n".join(f"| `{k}` | {v} |" for k, v in fields.items()) or "| （无） | 0 |"
        body += f"""### {key} `{item.get("code")}` → `{item.get("replay_code")}`

锚点：`{item.get("anchor_dir")}`。
精确值：目标 CAGR=`{tgt.get("cagr")}` Sharpe=`{tgt.get("sharpe")}` MaxDD=`{tgt.get("max_drawdown")}` 笔数=`{tgt.get("completed_trades")}`；
观察 CAGR=`{obs.get("cagr")}` Sharpe=`{obs.get("sharpe")}` MaxDD=`{obs.get("max_drawdown")}` 笔数=`{obs.get("completed_trades")}`。

| 项 | 值 |
|---|---|
| 身份键 mismatch | {ledger.get("identity_mismatches")} |
| 数值字段 mismatch 行合计 | {ledger.get("field_mismatch_rows")} |
| NAV 对比行 | {ledger.get("nav_rows_compared")} |
| NAV max abs | {ledger.get("nav_max_abs")} |
| ledger.passed | {ledger.get("passed")} |
| metric_ok | {item.get("metric_ok")} |
| 既有 artifact sha256 | `{item.get("anchor_sha256")}` |
| 复现 nav/trades sha256 | `{item.get("replay_sha256")}` |
| 确定性 runs | {det.get("actual_runs")} / 请求 {det.get("requested_runs")} |
| 确定性 identical | {det.get("all_identical")} |
| 确定性 mode | {det.get("mode")} |

分字段 mismatch：

| 字段 | 行数 |
|---|---:|
{field_rows}

样本 mismatch：

```json
{json.dumps(ledger.get("sample_mismatches") or [], ensure_ascii=False, indent=2, default=str)}
```

"""
    body += """## 规则

- 只允许浮点误差。身份键、笔数、排序不一致一律 FAIL。
- 确定性：同一 snapshot 连跑 3 次（单次 ≤180s 才跑满）；否则至少哈希既有 artifact + 一次重放。
- 失败不得开后续 stage，不得改策略凑过关。
"""
    assert_project_result_path(PARITY_MD)
    PARITY_MD.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")


def run_parity(
    *,
    resume: bool = True,
    determinism_runs: int = 3,
    prepared_market=None,
    market_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .freeze import require_frozen

    ensure_round11_tree()
    frozen = require_frozen()
    inject_round7_into_overlay_caches()
    if prepared_market is None:
        loaded = load_prepared_market(warmup_ma5=True)
        prepared_market = loaded["prepared"]
        market_meta = loaded
    baselines: dict[str, Any] = {}
    all_pass = True
    for key, spec in BASELINES.items():
        dest = replay_dir(spec)
        dest.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        ran = run_spec(spec, prepared_market=prepared_market, market_meta=market_meta, resume=resume)
        elapsed = time.perf_counter() - t0
        local_trades = pd.read_parquet(dest / "trades.parquet")
        local_nav = pd.read_parquet(dest / "nav.parquet")
        anchor_trades, anchor_nav, anchor_report = _load_anchor(spec)
        ledger = compare_ledgers(
            local_trades,
            anchor_trades,
            local_nav,
            anchor_nav,
            target_n_trades=int(spec.targets["completed_trades"]),
        )
        report = json.loads((dest / "report.json").read_text(encoding="utf-8"))
        observed = _observed_metrics(report)
        metric_ok = _metrics_ok(observed, spec.targets)
        snap_ok = (
            (report.get("snapshot_id") or (market_meta or {}).get("snapshot_id")) == SNAPSHOT_ID
            and (report.get("snapshot_fingerprint") or (market_meta or {}).get("snapshot_fingerprint"))
            == SNAPSHOT_FINGERPRINT
        )
        det = _run_determinism(
            spec,
            prepared_market=prepared_market,
            first_nav=local_nav,
            first_trades=local_trades,
            first_elapsed=elapsed,
            n_runs=determinism_runs,
        )
        passed = bool(ledger["passed"] and metric_ok and snap_ok and det.get("all_identical"))
        if not passed:
            all_pass = False
        baselines[key] = {
            "code": spec.code,
            "replay_code": spec.replay_code,
            "anchor_dir": str(spec.anchor_dir),
            "out_dir": str(dest),
            "status": "PASS" if passed else "REPRODUCIBILITY_FAIL",
            "passed": passed,
            "targets": dict(spec.targets),
            "observed": observed,
            "anchor_metrics": _observed_metrics(anchor_report),
            "metric_ok": metric_ok,
            "snapshot_ok": snap_ok,
            "ledger": ledger,
            "determinism": det,
            "anchor_sha256": artifact_file_hashes(spec.anchor_dir),
            "replay_sha256": artifact_file_hashes(dest),
            "run": {k: ran.get(k) for k in ("status", "code", "out_dir")},
            "elapsed_sec": elapsed,
        }
        (dest / "parity.json").write_text(
            json.dumps(baselines[key], ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    status = "PASS" if all_pass else "REPRODUCIBILITY_FAIL"
    out = {
        "status": status,
        "passed": all_pass,
        "fail_code": None if all_pass else "REPRODUCIBILITY_FAIL",
        "manifest": frozen,
        "source_tree_sha256": source_tree_sha256(),
        "baselines": baselines,
    }
    assert_project_result_path(PARITY_JSON)
    PARITY_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    write_parity_markdown(out)
    return out


def load_parity_status() -> dict[str, Any]:
    if not PARITY_JSON.is_file():
        return {"passed": False, "status": "missing"}
    return json.loads(PARITY_JSON.read_text(encoding="utf-8"))


__all__ = ["compare_ledgers", "load_parity_status", "run_parity", "write_parity_markdown"]
