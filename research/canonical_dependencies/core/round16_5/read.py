"""读 Round11–16 主报告与 Round16 00–13。缺失标 MISSING_ARTIFACT，不伪造。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import RESULT_ROOT, ROUND16_5_ROOT, ROUND16_ROOT
from .common import dump_json, load_json, write_md

READ_JSON = ROUND16_5_ROOT / "read" / "prior_artifacts.json"
READ_MD = ROUND16_5_ROOT / "00_prior_read.md"

PRIOR_REPORTS = (
    RESULT_ROOT / "round11" / "ROUND11_ROBUSTNESS_OVERFIT_AUDIT.md",
    RESULT_ROOT / "round12" / "ROUND12_RISK_REMEDIATION_REPORT.md",
    RESULT_ROOT / "round13" / "ROUND13_BASELINE_CONSOLIDATION_REPORT.md",
    RESULT_ROOT / "round14" / "ROUND14_RISK_BUDGET_REPORT.md",
    RESULT_ROOT / "round15" / "ROUND15_RESEARCH_REPORT.md",
    RESULT_ROOT / "round16" / "ROUND16_OPTION_RESEARCH_REPORT.md",
    RESULT_ROOT / "round16" / "ROUND17_RESEARCH_HANDOFF.md",
)

R16_NOTES = (
    "00_anchor_parity.md",
    "00_option_data_root_cause.md",
    "01_option_data_coverage.md",
    "02_contract_chain_reconstruction.md",
    "03_option_quote_quality.md",
    "03_freeze.md",
    "04_implied_volatility_validation.md",
    "05_option_factor_definitions.md",
    "06_option_factor_incrementality.md",
    "07_equity_option_redundancy.md",
    "08_experiment_results.md",
    "09_matched_controls.md",
    "10_execution_cost_stress.md",
    "11_statistical_validation.md",
    "12_forward_oos.md",
    "13_baseline_decision.md",
)

R16_JSON = (
    ROUND16_ROOT / "parity" / "parity.json",
    ROUND16_ROOT / "options" / "treat_dates.json",
    ROUND16_ROOT / "options" / "options.json",
    ROUND16_ROOT / "controls" / "controls.json",
    ROUND16_ROOT / "spa" / "spa.json",
    ROUND16_ROOT / "promotion" / "promotion.json",
    ROUND16_ROOT / "data" / "processed" / "option_factors_daily.parquet",
    ROUND16_ROOT / "candidates" / "R16_O01" / "report.json",
    ROUND16_ROOT / "candidates" / "R16_O01" / "nav.parquet",
    ROUND16_ROOT / "candidates" / "R16_O01" / "trades.parquet",
    ROUND16_ROOT / "parity" / "R16_00_CORRECTED_V4" / "nav.parquet",
    ROUND16_ROOT / "parity" / "R16_00_CORRECTED_V4" / "trades.parquet",
)


def _status(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "status": "OK" if exists else "MISSING_ARTIFACT",
        "bytes": int(path.stat().st_size) if exists else 0,
    }


def run_read(**_: Any) -> dict[str, Any]:
    reports = [_status(p) for p in PRIOR_REPORTS]
    notes = [_status(ROUND16_ROOT / name) for name in R16_NOTES]
    extras = [_status(p) for p in R16_JSON]
    missing = [r for r in reports + notes + extras if r["status"] == "MISSING_ARTIFACT"]
    payload = {
        "status": "DONE",
        "n_ok": int(sum(1 for r in reports + notes + extras if r["exists"])),
        "n_missing": len(missing),
        "reports": reports,
        "round16_notes": notes,
        "round16_json": extras,
        "missing": missing,
        "note": "缺失标 MISSING_ARTIFACT。不伪造数字。",
        "round16_known": {
            "placebo_was_ledger_swap": True,
            "n_events_treat_dates": 332,
            "n_treat_days_nav_aligned": 298,
            "spa_p": 0.496,
            "incremental": "INCONCLUSIVE",
            "redundancy": "REDUNDANT",
            "iv": "last_close_only",
            "promotion": "RECOMMENDED_BASELINE_V5=R16_O01 AWAITING_USER_LOCK",
            "PRIOR_SELECTION_EXISTS": True,
        },
    }
    dump_json(READ_JSON, payload)
    lines = [
        "# Round 16.5 · 先验读档",
        "",
        f"OK=`{payload['n_ok']}` MISSING_ARTIFACT=`{payload['n_missing']}`。",
        "",
        "Round16 已知缺陷（本轮必须修）：placebo 为账本日收益置换；332 vs 298 未拆；经验 p 禁止报 0；权威增量=FULL_ENGINE_NAV_DIFFERENCE；保留 SPA p=0.496。",
        "",
        "PRIOR_SELECTION_EXISTS = TRUE。2018–2026 重复检验不是独立确认。",
        "",
    ]
    if missing:
        lines.append("缺失：")
        for m in missing:
            lines.append(f"- `{m['path']}`")
        lines.append("")
    write_md(READ_MD, "\n".join(lines))
    return payload


def load_read() -> dict[str, Any]:
    return load_json(READ_JSON)
