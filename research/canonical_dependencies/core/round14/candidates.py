"""A01/B01/B02/B03/C01 单组合回测 + CTRL_EXP 对照。"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..paths import ROUND14_ROOT
from ..round12.runner import hist_hooks
from .common import dump_json, load_json, observed_from_report, pct, write_md
from .contract import (
    CANDIDATE_IDS,
    CONTROL_IDS,
    DE_RISKING_ONLY,
    FAIL_ALPHA_MUTATION,
    R14_00,
    RISK_ALLOCATION_INCREMENT,
    control_dir,
    ctrl_code,
    experiment_dir,
    overlayed_path,
    parity_dir,
)
from .freeze import require_frozen
from .parity import load_parity_status
from .runner import load_market, run_r14_backtest
from .weights import load_overlayed, load_weights_audit, load_weights_table, run_weights

CAND_JSON = ROUND14_ROOT / "candidates" / "candidates.json"
CAND_MD = ROUND14_ROOT / "03_candidates.md"
CTRL_JSON = ROUND14_ROOT / "controls" / "controls.json"
CTRL_MD = ROUND14_ROOT / "04_controls.md"


def _summarize(ran: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    report = ran.get("report") or {}
    obs = observed_from_report(report)
    obs.update({
        "code": ran.get("code"),
        "status": ran.get("status"),
        "out_dir": ran.get("out_dir"),
    })
    if extra:
        obs.update(extra)
    return obs


def _run_one(code: str, dest, *, resume: bool, loaded) -> dict[str, Any]:
    ov = load_overlayed(code)
    ran = run_r14_backtest(
        dest,
        overlayed=ov,
        validation_hooks=hist_hooks(),
        extra_report={
            "fee_schedule": "HISTORICAL_FEE_CORRECTED",
            "candidate": code,
            "unified_single_portfolio": True,
        },
        resume=resume,
        prepared_market=loaded["prepared"],
        market_meta=loaded,
    )
    extra = {
        "nav_path": str(dest / "nav.parquet"),
        "overlayed_path": str(overlayed_path(code)),
    }
    out = _summarize(ran, extra)
    out["nav_path"] = str(dest / "nav.parquet")
    return out


def run_candidates(*, resume: bool = True, jobs: int = 1, **_: Any) -> dict[str, Any]:
    require_frozen()
    if not load_parity_status().get("passed"):
        raise RuntimeError("REPRODUCIBILITY_FAIL_OR_MISSING STOP")
    w = run_weights(resume=resume)
    failed = list(w.get("failed_identity") or [])
    loaded = load_market()
    results: dict[str, Any] = {}
    for cid in CANDIDATE_IDS:
        if cid in failed:
            results[cid] = {
                "code": cid,
                "status": FAIL_ALPHA_MUTATION,
                "eligible": False,
            }
            continue
        results[cid] = _run_one(cid, experiment_dir(cid), resume=resume, loaded=loaded)
        results[cid]["eligible"] = True
    p = parity_dir(R14_00) / "report.json"
    if p.is_file():
        report = json.loads(p.read_text(encoding="utf-8"))
        results[R14_00] = observed_from_report(report)
        results[R14_00]["code"] = R14_00
        results[R14_00]["nav_path"] = str(parity_dir(R14_00) / "nav.parquet")
        results[R14_00]["status"] = "parity"
    payload = {
        "status": "DONE",
        "results": results,
        "n_candidates": len(CANDIDATE_IDS),
        "extra_candidates": False,
        "failed_identity": failed,
        "jobs_requested": jobs,
    }
    dump_json(CAND_JSON, payload)
    lines = [
        "# Round 14 · 候选 corrected-cost",
        "",
        "单组合回测。主合同 = HISTORICAL_FEE_CORRECTED。仅 A01/B01/B02/B03/C01。",
        "",
        "| ID | CAGR | Sharpe | MaxDD | 笔数 | LOYO med | LOYO worst |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for code in (R14_00, *CANDIDATE_IDS):
        row = results.get(code) or {}
        lines.append(
            f"| `{code}` | {pct(row.get('cagr'))} | {row.get('sharpe')} | {pct(row.get('max_drawdown'))} | "
            f"{row.get('completed_trades')} | {pct(row.get('loyo_median'))} | {pct(row.get('loyo_worst'))} |"
        )
    lines += ["", "无笛卡尔积，无新阈值搜索。", ""]
    write_md(CAND_MD, "\n".join(lines))
    return payload


def _mean_exposure(code: str) -> float | None:
    w = load_weights_table()
    if w.empty:
        return None
    sub = w.loc[w["source"].eq(code)]
    if sub.empty:
        return None
    daily = sub.groupby("signal_date")["target_nav_fraction"].sum()
    return float(daily.mean()) if len(daily) else None


def _unit_pnl(nav_path, exposure: float | None) -> float | None:
    from .common import daily_from_nav

    if not nav_path or exposure is None or exposure <= 0:
        return None
    from pathlib import Path

    p = Path(nav_path)
    if not p.is_file():
        return None
    daily = daily_from_nav(pd.read_parquet(p))
    if daily.empty:
        return None
    return float(daily.sum() / exposure)


def run_controls(*, resume: bool = True, jobs: int = 1, **_: Any) -> dict[str, Any]:
    require_frozen()
    cand = load_candidates()
    if cand.get("status") != "DONE":
        raise RuntimeError("CANDIDATES_MISSING STOP")
    loaded = load_market()
    results: dict[str, Any] = {}
    for cid in CANDIDATE_IDS:
        code = ctrl_code(cid)
        results[code] = _run_one(code, control_dir(code), resume=resume, loaded=loaded)
    comparisons: dict[str, Any] = {}
    cres = cand.get("results") or {}
    v4 = cres.get(R14_00) or {}
    e_v4 = _mean_exposure("v4")
    for cid in CANDIDATE_IDS:
        c = cres.get(cid) or {}
        k = ctrl_code(cid)
        ctrl = results.get(k) or {}
        e_c = _mean_exposure(cid)
        e_k = _mean_exposure(k)
        unit_c = _unit_pnl(c.get("nav_path"), e_c)
        unit_k = _unit_pnl(ctrl.get("nav_path"), e_k)
        unit_v4 = _unit_pnl(v4.get("nav_path"), e_v4)
        sh_c = c.get("sharpe")
        sh_k = ctrl.get("sharpe")
        beat = (
            sh_c is not None
            and sh_k is not None
            and (float(sh_c) > float(sh_k) + 1e-12 or (
                abs(float(sh_c) - float(sh_k)) <= 1e-12
                and float(c.get("cagr") or 0) > float(ctrl.get("cagr") or 0)
            ))
        )
        unit_up = unit_c is not None and unit_v4 is not None and float(unit_c) > float(unit_v4)
        if beat:
            label = RISK_ALLOCATION_INCREMENT
        elif not unit_up:
            label = DE_RISKING_ONLY
        else:
            label = "INCONCLUSIVE_UNIT_PNL"
        comparisons[cid] = {
            "control": k,
            "cand_sharpe": sh_c,
            "ctrl_sharpe": sh_k,
            "cand_cagr": c.get("cagr"),
            "ctrl_cagr": ctrl.get("cagr"),
            "mean_exposure_cand": e_c,
            "mean_exposure_ctrl": e_k,
            "mean_exposure_v4": e_v4,
            "unit_pnl_cand": unit_c,
            "unit_pnl_ctrl": unit_k,
            "unit_pnl_v4": unit_v4,
            "beat_control": beat,
            "unit_pnl_up_vs_v4": unit_up,
            "label": label,
        }
    payload = {
        "status": "DONE",
        "results": results,
        "comparisons": comparisons,
        "key_test": "B02 vs CTRL_EXP_B02",
        "key": comparisons.get("B02"),
        "jobs_requested": jobs,
    }
    dump_json(CTRL_JSON, payload)
    lines = [
        "# Round 14 · Exposure-matched 对照",
        "",
        "`scale=E_cand/E_v4`，`w_ctrl=w_v4*scale`。无风险信息。关键鉴别 = **B02 vs CTRL_EXP_B02**。",
        "",
        "| 候选 | 对照 | 候选 Sharpe | 对照 Sharpe | 单位暴露 PnL 候选 | v4 | 判定 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for cid in CANDIDATE_IDS:
        row = comparisons.get(cid) or {}
        lines.append(
            f"| `{cid}` | `{row.get('control')}` | {row.get('cand_sharpe')} | {row.get('ctrl_sharpe')} | "
            f"{row.get('unit_pnl_cand')} | {row.get('unit_pnl_v4')} | **{row.get('label')}** |"
        )
    lines += ["", f"B02 关键结论 = `{(comparisons.get('B02') or {}).get('label')}`。DE_RISKING_ONLY 不能升 v5。", ""]
    write_md(CTRL_MD, "\n".join(lines))
    return payload


def load_candidates() -> dict[str, Any]:
    return load_json(CAND_JSON)


def load_controls() -> dict[str, Any]:
    return load_json(CTRL_JSON)


def load_daily_map() -> dict[str, pd.Series]:
    from ..round11.common import daily_from_nav

    out: dict[str, pd.Series] = {}
    mapping = {R14_00: parity_dir(R14_00) / "nav.parquet", "v4": parity_dir(R14_00) / "nav.parquet"}
    for cid in CANDIDATE_IDS:
        mapping[cid] = experiment_dir(cid) / "nav.parquet"
    for cid in CONTROL_IDS:
        mapping[cid] = control_dir(cid) / "nav.parquet"
    for key, path in mapping.items():
        if path.is_file():
            out[key] = daily_from_nav(pd.read_parquet(path))
    return out


__all__ = [
    "load_candidates",
    "load_controls",
    "load_daily_map",
    "run_candidates",
    "run_controls",
]
