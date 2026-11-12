"""R1_00 与 Stock rank25_15 逐笔 parity。不过则停。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..local_executor import fs01122_top2_fixed_sleeve_config, run_local_fixed_sleeve_backtest, write_backtest_artifacts
from ..paths import (
    MARKET_END,
    PREDICTION_START,
    assert_project_result_path,
    ensure_replay_fs01122_tree,
    replay_fs01122_dir,
)
from ..round4.runner import attach_round4_fields
from ..stock_readonly import (
    fs01122_stock_report_path,
    load_prepared_market,
)
from .contract import TARGET_MAXDD, TARGET_N_TRADES, TARGET_TOTAL_RETURN
from .inject import load_locked_predictions


TRADE_KEYS = (
    "signal_date",
    "instrument_id",
    "entry_date",
    "actual_exit_date",
    "initial_quantity",
)


def _norm_trades(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("signal_date", "entry_date", "actual_exit_date"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col]).dt.normalize()
    if "instrument_id" in out.columns:
        out["instrument_id"] = out["instrument_id"].astype(str)
    if "initial_quantity" in out.columns:
        out["initial_quantity"] = pd.to_numeric(out["initial_quantity"], errors="coerce").fillna(0).astype(int)
    cols = [c for c in TRADE_KEYS if c in out.columns]
    return out[cols].sort_values(cols).reset_index(drop=True)


def compare_ledgers(local_trades: pd.DataFrame, stock_trades: pd.DataFrame,
                    local_nav: pd.DataFrame, stock_nav: pd.DataFrame) -> dict[str, Any]:
    lt = _norm_trades(local_trades)
    st = _norm_trades(stock_trades)
    n_mismatch = int(len(lt) != len(st))
    mismatches = []
    if list(lt.columns) == list(st.columns) and len(lt) == len(st):
        diff = lt.ne(st)
        n_mismatch = int(diff.any(axis=1).sum())
        if n_mismatch:
            bad = diff.any(axis=1)
            sample = pd.concat([lt.loc[bad].head(8), st.loc[bad].head(8)], axis=1, keys=["local", "stock"])
            mismatches = json.loads(sample.to_json(orient="records", date_format="iso"))
    else:
        mismatches = [{"reason": "schema_or_len", "local": len(lt), "stock": len(st)}]

    ln = local_nav.copy()
    sn = stock_nav.copy()
    ln["trade_date"] = pd.to_datetime(ln["trade_date"]).dt.normalize()
    sn["trade_date"] = pd.to_datetime(sn["trade_date"]).dt.normalize()
    merged = ln.merge(sn, on="trade_date", suffixes=("_local", "_stock"))
    nav_abs = (merged["total_asset_local"].astype(float) - merged["total_asset_stock"].astype(float)).abs()
    nav_max = float(nav_abs.max()) if len(nav_abs) else float("inf")
    passed = n_mismatch == 0 and nav_max <= 1e-4 and len(lt) == TARGET_N_TRADES
    return {
        "trade_rows_local": int(len(lt)),
        "trade_rows_stock": int(len(st)),
        "trade_mismatches": n_mismatch,
        "nav_rows_compared": int(len(merged)),
        "nav_max_abs": nav_max,
        "sample_mismatches": mismatches,
        "passed": bool(passed),
    }


def run_parity(*, resume: bool = True, prepared_market=None, market_meta=None) -> dict[str, Any]:
    ensure_replay_fs01122_tree()
    dest = replay_fs01122_dir("round1", "R1_00_baseline")
    dest.mkdir(parents=True, exist_ok=True)
    report_path = dest / "report.json"
    parity_md = replay_fs01122_dir("parity") / "00_parity.md"
    assert_project_result_path(report_path)

    stock_report_path = fs01122_stock_report_path()
    stock_report = json.loads(stock_report_path.read_text(encoding="utf-8"))
    stock_trades = pd.read_parquet(stock_report_path.with_name("trades.parquet"))
    stock_nav = pd.read_parquet(stock_report_path.with_name("nav.parquet"))

    if resume and report_path.is_file() and (dest / "trades.parquet").is_file():
        local_report = json.loads(report_path.read_text(encoding="utf-8"))
        local_trades = pd.read_parquet(dest / "trades.parquet")
        local_nav = pd.read_parquet(dest / "nav.parquet")
        cmp = compare_ledgers(local_trades, stock_trades, local_nav, stock_nav)
        payload = {"status": "resumed", "compare": cmp, "report": local_report}
    else:
        preds = load_locked_predictions()
        if prepared_market is None:
            loaded = load_prepared_market(warmup_ma5=True)
            prepared_market = loaded["prepared"]
            market_meta = loaded
        config = fs01122_top2_fixed_sleeve_config(exit_mode="close_from_d2")
        result = run_local_fixed_sleeve_backtest(
            preds,
            prepared_market,
            config=config,
            start_date=PREDICTION_START,
            end_date=MARKET_END,
        )
        metrics = result["report"].setdefault("metrics", {})
        attach_round4_fields(result["nav"], result["trades"], metrics)
        extra = {
            "status": "completed_and_validated",
            "experiment_code": "R1_00_baseline",
            "trial_id": "fs01122",
            "capacity": "n100_d2",
            "ensemble": "ensemble_prefilter_gap_floor",
            "variant": "rank25_15",
            "contract": "top_k=2 rank_nav=(0.25,0.15) open_gap_min=-0.01 close_from_d2",
            "predictions_file": str(stock_report.get("predictions_file")),
            "predictions_sha256": stock_report.get("predictions_sha256"),
            "stock_rank25_15": str(stock_report_path),
            "note": (
                "fs01122 Top2 25/15 parity。正式数字走 Stock fixed_sleeve。"
                "d1_open_gap 是 execution-time feature，与 fs00263 D-close 不可混比。"
            ),
        }
        if market_meta:
            extra["snapshot_id"] = market_meta.get("snapshot_id")
            extra["snapshot_fingerprint"] = market_meta.get("snapshot_fingerprint")
        written = write_backtest_artifacts(result, dest, extra_report=extra)
        local_report = written["report"]
        cmp = compare_ledgers(result["trades"], stock_trades, result["nav"], stock_nav)
        payload = {"status": "ran", "compare": cmp, "report": local_report}

    m = (payload["report"].get("metrics") or {})
    tot = float(m.get("total_return") or 0)
    ntr = int(m.get("completed_trades") or 0)
    mdd = float(m.get("max_drawdown") or 0)
    metric_ok = (
        abs(tot - TARGET_TOTAL_RETURN) < 1e-6
        and ntr == TARGET_N_TRADES
        and abs(mdd - TARGET_MAXDD) < 1e-8
    )
    passed = bool(payload["compare"]["passed"] and metric_ok)
    payload["passed"] = passed
    payload["targets"] = {
        "total_return": TARGET_TOTAL_RETURN,
        "completed_trades": TARGET_N_TRADES,
        "max_drawdown": TARGET_MAXDD,
        "got_total_return": tot,
        "got_trades": ntr,
        "got_max_drawdown": mdd,
    }

    md = [
        "# fs01122 Top2 parity（R1_00 vs Stock rank25_15）",
        "",
        f"- Stock 账本：`{stock_report_path}`",
        f"- 课题产物：`{dest}`",
        f"- 目标：累计 **+943.69%** / **1961** 笔 / MaxDD **−27.50%**",
        f"- 本地：累计 {tot:+.4%} / {ntr} 笔 / MaxDD {mdd:.4%}",
        f"- 逐笔 mismatch：{payload['compare']['trade_mismatches']}；NAV max abs：{payload['compare']['nav_max_abs']}",
        f"- **结果：{'PASS' if passed else 'FAIL'}**",
        "",
        "契约：`top_k=2`，`rank_name_budget_nav_fractions=(0.25, 0.15)`，`open_gap_min=-0.01`（无 +7% 上沿），`close_from_d2`，初始 30 万，双袖套。",
        "",
        "已知语义差：`d1_open_gap` 进了 ranker 特征（D+1 信息）。与 fs00263 D-close 不可混比。",
        "",
    ]
    if not passed:
        md.append("Parity 失败，按计划停止后续 overlay / 训练。")
        md.append("")
        md.append("```json")
        md.append(json.dumps(payload["compare"], ensure_ascii=False, indent=2, default=str))
        md.append("```")
    parity_md.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(parity_md)
    parity_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    (dest / "parity.json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "report"}, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    payload["parity_md"] = str(parity_md)
    return payload
