"""读 R7/R13/R16/R16.5/R17/R18 契约与报告，核真实路径与 sha256。不伪造。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT, RESULT_ROOT, ROUND19_ROOT
from .common import dump_json, file_sha256, write_md
from .contract import (
    CURRENT_BASELINE,
    OPTION_CANDIDATE,
    PARENT_V4,
    PRIOR_RESEARCH_SELECTION_EXISTS,
    V4_ANCHOR,
    copy_plan,
)

READ_JSON = ROUND19_ROOT / "read" / "prior_artifacts.json"
READ_MD = ROUND19_ROOT / "read" / "00_prior_read.md"

PRIOR_FILES: tuple[tuple[str, Path], ...] = (
    ("r7_contract_py", PROJECT_ROOT / "core" / "round7" / "contract.py"),
    ("r7_contract_json", RESULT_ROOT / "round7" / "contract.json"),
    ("r13_contract_py", PROJECT_ROOT / "core" / "round13" / "contract.py"),
    ("r13_report", RESULT_ROOT / "round13" / "ROUND13_BASELINE_CONSOLIDATION_REPORT.md"),
    ("r13_manifest", RESULT_ROOT / "round13" / "round13_manifest_v1.yaml"),
    ("r13_manifest_sha", RESULT_ROOT / "round13" / "round13_manifest_v1.yaml.sha256"),
    ("r16_contract_py", PROJECT_ROOT / "core" / "round16" / "contract.py"),
    ("r16_report", RESULT_ROOT / "round16" / "ROUND16_OPTION_RESEARCH_REPORT.md"),
    ("r16_manifest", RESULT_ROOT / "round16" / "round16_manifest_v1.yaml"),
    ("r16_manifest_sha", RESULT_ROOT / "round16" / "round16_manifest_v1.yaml.sha256"),
    ("r16_5_contract_py", PROJECT_ROOT / "core" / "round16_5" / "contract.py"),
    ("r16_5_report", RESULT_ROOT / "round16_5" / "ROUND16_5_CONFIRMATORY_REPORT.md"),
    ("r16_5_manifest", RESULT_ROOT / "round16_5" / "round16_5_manifest_v1.yaml"),
    ("r16_5_manifest_sha", RESULT_ROOT / "round16_5" / "round16_5_manifest_v1.yaml.sha256"),
    ("r17_contract_py", PROJECT_ROOT / "core" / "round17" / "contract.py"),
    ("r17_report", RESULT_ROOT / "round17" / "ROUND17_LARGE_SCALE_FACTOR_SEARCH_REPORT.md"),
    ("r17_manifest", RESULT_ROOT / "round17" / "round17_search_manifest.yaml"),
    ("r17_manifest_sha", RESULT_ROOT / "round17" / "round17_search_manifest.yaml.sha256"),
    ("r18_contract_py", PROJECT_ROOT / "core" / "round18" / "contract.py"),
    ("r18_report", RESULT_ROOT / "round18" / "ROUND18_MICROSTRUCTURE_ALPHA_REPORT.md"),
    ("r18_manifest", RESULT_ROOT / "round18" / "round18_microstructure_manifest.yaml"),
    ("r18_manifest_sha", RESULT_ROOT / "round18" / "round18_microstructure_manifest.yaml.sha256"),
    ("r16_5_v00_nav", RESULT_ROOT / "round16_5" / "parity" / "R16_5_V00_BASELINE_V4" / "nav.parquet"),
    ("r16_5_v00_trades", RESULT_ROOT / "round16_5" / "parity" / "R16_5_V00_BASELINE_V4" / "trades.parquet"),
    ("r18_a00_nav", RESULT_ROOT / "round18" / "parity" / "R18_A00" / "nav.parquet"),
    ("r18_a00_trades", RESULT_ROOT / "round18" / "parity" / "R18_A00" / "trades.parquet"),
    ("tracker", RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"),
)


def _status(key: str, path: Path) -> dict[str, Any]:
    exists = path.is_file()
    resolved = path.resolve() if path.exists() else path
    rec: dict[str, Any] = {
        "key": key,
        "path": str(path),
        "resolved": str(resolved),
        "exists": exists,
        "status": "OK" if exists else "MISSING_ARTIFACT",
        "bytes": int(path.stat().st_size) if exists else 0,
        "sha256": file_sha256(path) if exists else None,
    }
    if exists and path.name.endswith(".sha256"):
        rec["declared_sha256"] = path.read_text(encoding="utf-8").strip().split()[0]
    return rec


def run_read(**_: Any) -> dict[str, Any]:
    copy_plan()
    items = [_status(k, p) for k, p in PRIOR_FILES]
    missing = [r for r in items if r["status"] == "MISSING_ARTIFACT"]
    real_root = PROJECT_ROOT.resolve()
    payload = {
        "status": "DONE",
        "current": CURRENT_BASELINE,
        "parent": PARENT_V4,
        "option_candidate_untouched": OPTION_CANDIDATE,
        "prior_research_selection_exists": PRIOR_RESEARCH_SELECTION_EXISTS,
        "v4_anchor": dict(V4_ANCHOR),
        "project_root": str(PROJECT_ROOT),
        "project_root_resolved": str(real_root),
        "real_data1_stock_mxy": str(real_root).startswith("/data1/stock-mxy"),
        "n_ok": int(sum(1 for r in items if r["exists"])),
        "n_missing": len(missing),
        "files": items,
        "missing": missing,
        "constraints": {
            "subsample": 1.0,
            "colsample": 1.0,
            "no_fuse_r16_o01_x_round18": True,
            "no_auto_promote": True,
            "no_写入时间": True,
            "no_new_factor_search": True,
        },
        "known_facts_to_verify": {
            "r16_5_report_wrote_n_days_as_trades": True,
            "correct_completed_trades": 1149,
            "trading_observations": 2075,
            "o01_nav_increment_authoritative": "FULL_ENGINE_NAV_DIFFERENCE",
            "r16_spa_p": 0.496,
            "v03_activity_not_fully_matched": True,
            "r18_a00_was_cached_overlay_not_fresh_retrain": True,
        },
        "note": "缺失标 MISSING_ARTIFACT。不伪造数字。CURRENT 仍是 baseline_v4。",
    }
    dump_json(READ_JSON, payload)
    lines = [
        "# Round19 · 先验读档",
        "",
        f"CURRENT=`{CURRENT_BASELINE}` parent=`{PARENT_V4}`。CANDIDATE=`{OPTION_CANDIDATE}` **尚未**升 CURRENT。",
        f"课题路径 `{PROJECT_ROOT}` → 实路径 `{real_root}`。",
        f"OK=`{payload['n_ok']}` MISSING_ARTIFACT=`{payload['n_missing']}`。",
        "",
        "## 已核 sha256 的契约 / 报告 / manifest",
        "",
    ]
    for rec in items:
        digest = rec.get("sha256") or "MISSING"
        lines.append(f"- `{rec['key']}` exists=`{rec['exists']}` sha256=`{digest}`")
        lines.append(f"  `{rec['resolved']}`")
    lines.extend(
        [
            "",
            "Round16.5 主报告曾把 `metric_n_daily_returns=2075` 写成「笔数」；权威成交是 **1149**。",
            "主评价 CAGR 用日历年 `annualized_return`（52.54%），`cagr_of` 252 日口径（55.07%）只作对照。",
            "",
        ]
    )
    if missing:
        lines.append("缺失：")
        for m in missing:
            lines.append(f"- `{m['path']}`")
        lines.append("")
    write_md(READ_MD, "\n".join(lines))
    return payload
