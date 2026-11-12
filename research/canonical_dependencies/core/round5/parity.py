"""门禁 0：R5_00_baseline_v3 必须与 result/round4/R4_B05 逐笔一致。不过就停。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..metrics import CAGR_ABS_TOL, METRIC_VERSION, SHARPE_ABS_TOL
from ..paths import ROUND4_ROOT, ROUND5_ROOT, assert_project_result_path, ensure_round5_tree
from .registry import (
    BASELINE_V3,
    BASELINE_V3_ALIAS,
    V3_CAGR,
    V3_MDD,
    V3_SHARPE,
    V3_TRADES,
    get_spec,
)
from .runner import run_spec

TRADE_KEYS = (
    "signal_date", "instrument_id", "entry_date", "actual_exit_date",
    "initial_quantity", "exit_quantity",
)
MULT_KEYS = ("allocation_multiplier",)
QTY_TOL = 1e-9
FEE_TOL = 1e-4
NAV_TOL = 1e-4
MULT_TOL = 1e-8


def _write(path: Path, text: str) -> Path:
    assert_project_result_path(path)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return path


def _pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _load_r4() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    root = ROUND4_ROOT / BASELINE_V3
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    trades = pd.read_parquet(root / "trades.parquet")
    nav = pd.read_parquet(root / "nav.parquet")
    return report, trades, nav


def _norm_trades(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("signal_date", "entry_date", "actual_exit_date", "planned_exit_date"):
        if col in out:
            out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
    if "instrument_id" in out:
        out["instrument_id"] = out["instrument_id"].astype(str)
    sort_cols = [c for c in ("signal_date", "instrument_id", "entry_date") if c in out]
    return out.sort_values(sort_cols).reset_index(drop=True)


def _norm_nav(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.strftime("%Y-%m-%d")
    return out.sort_values("trade_date").reset_index(drop=True)


def compare_ledgers(
    r5_trades: pd.DataFrame,
    r4_trades: pd.DataFrame,
    r5_nav: pd.DataFrame,
    r4_nav: pd.DataFrame,
) -> dict[str, Any]:
    a = _norm_trades(r5_trades)
    b = _norm_trades(r4_trades)
    na = _norm_nav(r5_nav)
    nb = _norm_nav(r4_nav)
    keys = [k for k in TRADE_KEYS if k in a.columns and k in b.columns]
    n_ok = len(a) == len(b)
    mismatches: list[dict[str, Any]] = []
    if n_ok and keys:
        for col in keys:
            if col in {"initial_quantity", "exit_quantity"}:
                diff = (pd.to_numeric(a[col], errors="coerce") - pd.to_numeric(b[col], errors="coerce")).abs()
                bad = diff > QTY_TOL
            else:
                bad = a[col].astype(str) != b[col].astype(str)
            if bad.any():
                sample = a.loc[bad, keys].head(8).to_dict("records")
                mismatches.append({"col": col, "n": int(bad.sum()), "sample": sample})
    for col in MULT_KEYS:
        if n_ok and col in a.columns and col in b.columns:
            diff = (pd.to_numeric(a[col], errors="coerce") - pd.to_numeric(b[col], errors="coerce")).abs()
            if (diff > MULT_TOL).any():
                mismatches.append({"col": col, "n": int((diff > MULT_TOL).sum()), "max_abs": float(diff.max())})
    fee_cols = [c for c in (
        "buy_cost", "sell_proceeds", "buy_commission", "buy_transfer_fee",
        "sell_commission", "sell_transfer_fee", "stamp_tax",
    ) if c in a.columns and c in b.columns]
    fee_ok = True
    if n_ok:
        for col in fee_cols:
            diff = (pd.to_numeric(a[col], errors="coerce") - pd.to_numeric(b[col], errors="coerce")).abs()
            if (diff > FEE_TOL).any():
                fee_ok = False
                mismatches.append({"col": col, "n": int((diff > FEE_TOL).sum()), "max_abs": float(diff.max())})
    nav_ok = len(na) == len(nb)
    nav_max = None
    if nav_ok and "total_asset" in na and "total_asset" in nb:
        dates_ok = (na["trade_date"] == nb["trade_date"]).all()
        adiff = (pd.to_numeric(na["total_asset"]) - pd.to_numeric(nb["total_asset"])).abs()
        nav_max = float(adiff.max()) if len(adiff) else 0.0
        nav_ok = bool(dates_ok and nav_max <= NAV_TOL)
        if not dates_ok:
            mismatches.append({"col": "nav_trade_date", "n": int((na["trade_date"] != nb["trade_date"]).sum())})
        if nav_max is not None and nav_max > NAV_TOL:
            mismatches.append({"col": "total_asset", "max_abs": nav_max})
    elif not nav_ok:
        mismatches.append({"col": "nav_rows", "r5": int(len(na)), "r4": int(len(nb))})

    identity_ok = n_ok and not any(m.get("col") in TRADE_KEYS for m in mismatches)
    passed = bool(n_ok and identity_ok and fee_ok and nav_ok and len(mismatches) == 0)
    return {
        "n_r5": int(len(a)),
        "n_r4": int(len(b)),
        "n_match": n_ok,
        "trade_identity_ok": identity_ok,
        "fee_ok": fee_ok,
        "nav_ok": nav_ok,
        "nav_max_abs": nav_max,
        "mismatches": mismatches,
        "passed": passed,
    }


def write_parity_markdown(
    *,
    predictions=None,
    prepared=None,
    meta=None,
    resume: bool = True,
    run_now: bool = True,
) -> dict[str, Any]:
    ensure_round5_tree()
    spec = get_spec(BASELINE_V3_ALIAS)
    ran = run_spec(
        spec, predictions=predictions, prepared_market=prepared, market_meta=meta,
        resume=resume, run_now=run_now,
    )
    r4_report, r4_trades, r4_nav = _load_r4()
    r5_dir = ROUND5_ROOT / BASELINE_V3_ALIAS
    r5_report = json.loads((r5_dir / "report.json").read_text(encoding="utf-8")) if (r5_dir / "report.json").is_file() else {}
    r5_trades = pd.read_parquet(r5_dir / "trades.parquet") if (r5_dir / "trades.parquet").is_file() else pd.DataFrame()
    r5_nav = pd.read_parquet(r5_dir / "nav.parquet") if (r5_dir / "nav.parquet").is_file() else pd.DataFrame()

    cmp = compare_ledgers(r5_trades, r4_trades, r5_nav, r4_nav)
    r4m = r4_report.get("metrics") or {}
    r5m = r5_report.get("metrics") or {}
    cagr_ok = r5m.get("annualized_return") is not None and abs(
        float(r5m["annualized_return"]) - float(r4m.get("annualized_return") or V3_CAGR)
    ) <= CAGR_ABS_TOL
    sharpe_ok = r5m.get("sharpe_zero_rate") is not None and abs(
        float(r5m["sharpe_zero_rate"]) - float(r4m.get("sharpe_zero_rate") or V3_SHARPE)
    ) <= SHARPE_ABS_TOL
    mdd_ok = r5m.get("max_drawdown") is not None and abs(
        float(r5m["max_drawdown"]) - float(r4m.get("max_drawdown") or V3_MDD)
    ) <= 1e-5
    trades_ok = int(r5m.get("completed_trades") or 0) == int(r4m.get("completed_trades") or V3_TRADES)
    display_ok = bool(cagr_ok and sharpe_ok and mdd_ok and trades_ok)
    passed = bool(cmp["passed"] and display_ok)

    dest = ROUND5_ROOT / "00_parity"
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "written_at": datetime.now().astimezone().isoformat(),
        "passed": passed,
        "metric_version": METRIC_VERSION,
        "r5_status": ran.get("status"),
        "compare": cmp,
        "display": {
            "cagr_ok": cagr_ok, "sharpe_ok": sharpe_ok, "mdd_ok": mdd_ok, "trades_ok": trades_ok,
            "r5": {
                "cagr": r5m.get("annualized_return"),
                "sharpe": r5m.get("sharpe_zero_rate"),
                "max_drawdown": r5m.get("max_drawdown"),
                "completed_trades": r5m.get("completed_trades"),
                "cagr_ex2024": r5m.get("cagr_ex2024"),
            },
            "r4": {
                "cagr": r4m.get("annualized_return"),
                "sharpe": r4m.get("sharpe_zero_rate"),
                "max_drawdown": r4m.get("max_drawdown"),
                "completed_trades": r4m.get("completed_trades"),
                "cagr_ex2024": r4m.get("cagr_ex2024"),
            },
        },
        "target": "23.86 / 1.454 / −14.28 / 877",
        "note": "复用锁分 fs00263 + R4_B05 overlay，不重训。",
    }
    assert_project_result_path(dest / "parity.json")
    (dest / "parity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )

    mismatch_txt = "无" if not cmp["mismatches"] else json.dumps(cmp["mismatches"][:8], ensure_ascii=False, default=str)
    md = f"""# 00 baseline_v3 逐笔对齐

- 生成时间：{payload['written_at']}
- 对照：`result/round4/R4_B05/` ↔ `result/round5/R5_00_baseline_v3/`
- `metric_version`：`{METRIC_VERSION}`
- 分数：锁分 fs00263 五种子集成（不重训）+ R4_B05 overlay
- 结论：**{'通过' if passed else '未通过，停'}**

## 指标

| | R4_B05 | R5_00 | 对齐 |
|---|---:|---:|:---:|
| CAGR | {_pct(r4m.get('annualized_return'))} | {_pct(r5m.get('annualized_return'))} | {'是' if cagr_ok else '否'} |
| Sharpe | {float(r4m.get('sharpe_zero_rate') or 0):.3f} | {float(r5m.get('sharpe_zero_rate') or 0):.3f} | {'是' if sharpe_ok else '否'} |
| MaxDD | {_pct(r4m.get('max_drawdown'))} | {_pct(r5m.get('max_drawdown'))} | {'是' if mdd_ok else '否'} |
| 去2024 | {_pct(r4m.get('cagr_ex2024'))} | {_pct(r5m.get('cagr_ex2024'))} | — |
| 笔数 | {r4m.get('completed_trades')} | {r5m.get('completed_trades')} | {'是' if trades_ok else '否'} |

目标复现：23.86 / 1.454 / −14.28 / 877。

## 逐笔

| 项 | 结果 |
|---|---|
| 笔数 | R5={cmp['n_r5']} R4={cmp['n_r4']} {'对齐' if cmp['n_match'] else '不对'} |
| 买卖日 / 标的 / 股数 | {'对齐' if cmp['trade_identity_ok'] else '不对'} |
| 费用 | {'对齐' if cmp['fee_ok'] else '不对'} |
| 日 NAV | {'对齐' if cmp['nav_ok'] else '不对'}（max abs={cmp.get('nav_max_abs')}） |

不一致样本：{mismatch_txt}

不过就停，不准进入诊断 / 训练。
"""
    _write(ROUND5_ROOT / "00_baseline_v3_parity.md", md)
    return payload


def load_parity_status() -> dict[str, Any]:
    path = ROUND5_ROOT / "00_parity" / "parity.json"
    if not path.is_file():
        return {"passed": False, "reason": "parity 未写"}
    return json.loads(path.read_text(encoding="utf-8"))
