"""R13_00 / CTRL_V2 / CTRL_V3 精确 parity，以及 A01–A05 单组合 corrected 回测。"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..paths import ROUND13_ROOT
from ..round11.parity import compare_ledgers
from ..round12.runner import hist_hooks
from .common import dump_json, load_json, observed_from_report, pct, write_md
from .contract import (
    CANDIDATE_IDS,
    CTRL_V2_TARGETS,
    CTRL_V3_TARGETS,
    PARITY_ABS_NAV,
    PARITY_METRIC_ABS,
    R11_V2_CORRECTED_DIR,
    R11_V3_CORRECTED_DIR,
    R12_01_DIR,
    R12_01_TARGETS,
    R13_00,
    R13_CTRL_V2,
    R13_CTRL_V3,
    experiment_dir,
    overlayed_path,
    parity_dir,
)
from .freeze import require_frozen
from .runner import build_version_overlayed, load_market, run_r13_backtest
from .weights import load_overlayed

PARITY_MD = ROUND13_ROOT / "01_anchor_parity.md"
PARITY_JSON = ROUND13_ROOT / "parity" / "parity.json"
CAND_MD = ROUND13_ROOT / "02_candidates.md"
CAND_JSON = ROUND13_ROOT / "candidates" / "candidates.json"


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


def _summarize(ran: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    report = ran.get("report") or {}
    obs = observed_from_report(report)
    obs.update({
        "code": ran.get("code"),
        "status": ran.get("status"),
        "out_dir": ran.get("out_dir"),
        "nav_path": str((experiment_dir(str(ran.get("code"))) / "nav.parquet")) if ran.get("code") else None,
    })
    if extra:
        obs.update(extra)
    return obs


def _one_parity(
    code: str,
    overlayed: pd.DataFrame,
    anchor_dir,
    targets: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    dest = parity_dir(code)
    loaded = load_market()
    ran = run_r13_backtest(
        dest,
        overlayed=overlayed,
        validation_hooks=hist_hooks(),
        extra_report={"parity_anchor": str(anchor_dir), "fee_schedule": "HISTORICAL_FEE_CORRECTED"},
        resume=resume,
        prepared_market=loaded["prepared"],
        market_meta=loaded,
    )
    local_trades = pd.read_parquet(dest / "trades.parquet")
    local_nav = pd.read_parquet(dest / "nav.parquet")
    anchor_trades = pd.read_parquet(anchor_dir / "trades.parquet")
    anchor_nav = pd.read_parquet(anchor_dir / "nav.parquet")
    ledger = compare_ledgers(
        local_trades,
        anchor_trades,
        local_nav,
        anchor_nav,
        target_n_trades=int(targets["completed_trades"]),
    )
    report = json.loads((dest / "report.json").read_text(encoding="utf-8"))
    observed = observed_from_report(report)
    metric_ok = _metrics_ok(observed, targets)
    passed = bool(ledger["passed"] and metric_ok)
    return {
        "code": code,
        "anchor_dir": str(anchor_dir),
        "out_dir": str(dest),
        "status": "PASS" if passed else "REPRODUCIBILITY_FAIL",
        "passed": passed,
        "targets": dict(targets),
        "observed": observed,
        "metric_ok": metric_ok,
        "ledger": {k: ledger.get(k) for k in ("passed", "nav_max_abs", "n_local", "n_anchor", "n_mismatch") if k in ledger or True},
        "run": {k: ran.get(k) for k in ("status", "code", "out_dir")},
        "fee_schedule": "HISTORICAL_FEE_CORRECTED",
        "nav_path": str(dest / "nav.parquet"),
    }


def run_parity(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    ov2, _, _ = build_version_overlayed("v2")
    ov3, _, _ = build_version_overlayed("v3")
    ov4, _, _ = build_version_overlayed("v4")
    items = {
        R13_00: _one_parity(R13_00, ov4, R12_01_DIR, R12_01_TARGETS, resume=resume),
        R13_CTRL_V2: _one_parity(R13_CTRL_V2, ov2, R11_V2_CORRECTED_DIR, CTRL_V2_TARGETS, resume=resume),
        R13_CTRL_V3: _one_parity(R13_CTRL_V3, ov3, R11_V3_CORRECTED_DIR, CTRL_V3_TARGETS, resume=resume),
    }
    all_pass = all(v.get("passed") for v in items.values())
    payload = {
        "status": "PASS" if all_pass else "REPRODUCIBILITY_FAIL",
        "passed": all_pass,
        "fail_code": None if all_pass else "REPRODUCIBILITY_FAIL",
        "nav_abs_tol": PARITY_ABS_NAV,
        "metric_abs_tol": PARITY_METRIC_ABS,
        "anchors": items,
        "note": "parity 失败必须 STOP，不得为过关改策略或改容差。",
    }
    dump_json(PARITY_JSON, payload)
    lines = [
        "# Round 13 · 锚点 parity",
        "",
        f"结果：**{payload['status']}**。指标容差 `{PARITY_METRIC_ABS:g}`，NAV abs≤`{PARITY_ABS_NAV:g}`。",
        "",
        "| ID | 锚点 | 目标 CAGR | 观察 CAGR | 目标 Sharpe | 观察 Sharpe | 目标笔数 | 观察笔数 | 判定 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in items.values():
        tgt = item["targets"]
        obs = item["observed"]
        lines.append(
            f"| `{item['code']}` | `{item['anchor_dir']}` | {pct(tgt.get('cagr'))} | {pct(obs.get('cagr'))} | "
            f"{float(tgt.get('sharpe') or 0):.6f} | {float(obs.get('sharpe') or 0):.6f} | "
            f"{tgt.get('completed_trades')} | {obs.get('completed_trades')} | {item['status']} |"
        )
    lines += ["", "失败 = `REPRODUCIBILITY_FAIL`，停止后续 stage。", ""]
    write_md(PARITY_MD, "\n".join(lines))
    if not all_pass:
        raise RuntimeError("REPRODUCIBILITY_FAIL STOP")
    return payload


def load_parity_status() -> dict[str, Any]:
    if not PARITY_JSON.is_file():
        return {"passed": False, "status": "missing"}
    return load_json(PARITY_JSON)


def run_candidates(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    parity = load_parity_status()
    if not parity.get("passed"):
        raise RuntimeError("REPRODUCIBILITY_FAIL_OR_MISSING STOP")
    loaded = load_market()
    results: dict[str, Any] = {}
    for cid in CANDIDATE_IDS:
        ov = load_overlayed(cid)
        dest = experiment_dir(cid)
        ran = run_r13_backtest(
            dest,
            overlayed=ov,
            validation_hooks=hist_hooks(),
            extra_report={
                "fee_schedule": "HISTORICAL_FEE_CORRECTED",
                "candidate": cid,
                "unified_single_portfolio": True,
                "never_average_two_navs": True,
            },
            resume=resume,
            prepared_market=loaded["prepared"],
            market_meta=loaded,
        )
        extra = {
            "nav_path": str(dest / "nav.parquet"),
            "overlayed_path": str(overlayed_path(cid)),
            "unified_single_portfolio": True,
        }
        results[cid] = _summarize(ran, extra)
        results[cid]["nav_path"] = str(dest / "nav.parquet")
    # also index controls from parity dirs
    for code in (R13_00, R13_CTRL_V2, R13_CTRL_V3):
        p = parity_dir(code) / "report.json"
        if p.is_file():
            report = json.loads(p.read_text(encoding="utf-8"))
            results[code] = observed_from_report(report)
            results[code]["code"] = code
            results[code]["nav_path"] = str(parity_dir(code) / "nav.parquet")
            results[code]["status"] = "parity"
    payload = {
        "status": "DONE",
        "results": results,
        "never_average_two_navs": True,
        "n_candidates": len(CANDIDATE_IDS),
        "extra_candidates": False,
    }
    dump_json(CAND_JSON, payload)
    lines = [
        "# Round 13 · A01–A05 corrected-cost",
        "",
        "单组合回测。**禁止**双 NAV 相加/平均。主合同 = HISTORICAL_FEE_CORRECTED。",
        "",
        "| ID | CAGR | Sharpe | MaxDD | 笔数 | LOYO med | LOYO worst |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for code in (R13_00, R13_CTRL_V2, R13_CTRL_V3, *CANDIDATE_IDS):
        row = results.get(code) or {}
        lines.append(
            f"| `{code}` | {pct(row.get('cagr'))} | {row.get('sharpe')} | {pct(row.get('max_drawdown'))} | "
            f"{row.get('completed_trades')} | {pct(row.get('loyo_median'))} | {pct(row.get('loyo_worst'))} |"
        )
    lines += ["", "无 60/40 搜索，无 A04b。", ""]
    write_md(CAND_MD, "\n".join(lines))
    return payload


def load_candidates() -> dict[str, Any]:
    return load_json(CAND_JSON)


def load_daily_map() -> dict[str, pd.Series]:
    from ..round11.common import daily_from_nav

    out: dict[str, pd.Series] = {}
    mapping = {
        "v2": parity_dir(R13_CTRL_V2) / "nav.parquet",
        "v3": parity_dir(R13_CTRL_V3) / "nav.parquet",
        "v4": parity_dir(R13_00) / "nav.parquet",
        R13_00: parity_dir(R13_00) / "nav.parquet",
        R13_CTRL_V2: parity_dir(R13_CTRL_V2) / "nav.parquet",
        R13_CTRL_V3: parity_dir(R13_CTRL_V3) / "nav.parquet",
    }
    for cid in CANDIDATE_IDS:
        mapping[cid] = experiment_dir(cid) / "nav.parquet"
    for key, path in mapping.items():
        if path.is_file():
            out[key] = daily_from_nav(pd.read_parquet(path))
    return out


__all__ = [
    "load_candidates",
    "load_daily_map",
    "load_parity_status",
    "run_candidates",
    "run_parity",
]
