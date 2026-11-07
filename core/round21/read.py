"""读 R7–R20 契约与报告，核真实路径与 sha256。不伪造。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT, RESULT_ROOT, ROUND21_ROOT
from .common import dump_json, file_sha256, write_md
from .contract import CURRENT_ALIAS, CURRENT_BASELINE, PARENT_V4, R20_PRIOR, V5_ANCHOR, copy_plan

READ_JSON = ROUND21_ROOT / "read" / "prior_artifacts.json"
READ_MD = ROUND21_ROOT / "read" / "00_prior_read.md"

PRIOR_FILES: tuple[tuple[str, Path], ...] = (
    ("r7_contract_py", PROJECT_ROOT / "core" / "round7" / "contract.py"),
    ("r7_train_py", PROJECT_ROOT / "core" / "round7" / "train.py"),
    ("r16_o01_contract", PROJECT_ROOT / "core" / "round16_5" / "contract.py"),
    ("r17_factor_registry", RESULT_ROOT / "round17" / "inventory" / "factor_registry.csv"),
    ("r17_report", RESULT_ROOT / "round17" / "ROUND17_LARGE_SCALE_FACTOR_SEARCH_REPORT.md"),
    ("r18_contract_py", PROJECT_ROOT / "core" / "round18" / "contract.py"),
    ("r18_f1_py", PROJECT_ROOT / "core" / "round18" / "f1.py"),
    ("r18_manifest", RESULT_ROOT / "round18" / "round18_microstructure_manifest.yaml"),
    ("r18_pit_json", RESULT_ROOT / "round18" / "pit" / "pit.json"),
    ("r19_report", RESULT_ROOT / "round19" / "ROUND19_RESEARCH_REPORT.md"),
    ("r19_1_dir", RESULT_ROOT / "round19_1"),
    ("r20_manifest", RESULT_ROOT / "round20" / "round20_manifest.yaml"),
    ("r20_manifest_sha", RESULT_ROOT / "round20" / "round20_manifest.yaml.sha256"),
    ("r20_report", RESULT_ROOT / "round20" / "ROUND20_FACTOR_RESEARCH_REPORT.md"),
    ("r20_parity", RESULT_ROOT / "round20" / "parity" / "parity.json"),
    ("replay_r7_caliber_py", PROJECT_ROOT / "tools" / "replay_r7_caliber.py"),
    ("replay_v5_summary", RESULT_ROOT / "replay_r7_caliber" / "v5_R16_O01" / "replay_summary.json"),
    ("replay_v5_nav", RESULT_ROOT / "replay_r7_caliber" / "v5_R16_O01" / "nav.parquet"),
    ("replay_v5_trades", RESULT_ROOT / "replay_r7_caliber" / "v5_R16_O01" / "trades.parquet"),
    ("data_inventory_r21_project", RESULT_ROOT / "data_inventory"),
    ("tracker", RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"),
)


def _status(key: str, path: Path) -> dict[str, Any]:
    exists = path.exists()
    is_file = path.is_file()
    resolved = path.resolve() if exists else path
    rec: dict[str, Any] = {
        "key": key,
        "path": str(path),
        "resolved": str(resolved),
        "exists": exists,
        "is_file": is_file,
        "is_dir": path.is_dir() if exists else False,
        "status": "OK" if exists else "MISSING_ARTIFACT",
        "bytes": int(path.stat().st_size) if is_file else 0,
        "sha256": file_sha256(path) if is_file else None,
    }
    return rec


def run_read(**_: Any) -> dict[str, Any]:
    copy_plan()
    items = [_status(k, p) for k, p in PRIOR_FILES]
    by_key = {r["key"]: r for r in items}
    r19_1 = by_key["r19_1_dir"]
    r19_1["expected_absent"] = True
    if r19_1["exists"]:
        r19_1["status"] = "UNEXPECTED_PRESENT"
    else:
        r19_1["status"] = "ABSENT_AS_EXPECTED"
    inv = by_key["data_inventory_r21_project"]
    inv["expected_absent"] = True
    if inv["exists"]:
        inv["status"] = "UNEXPECTED_PRESENT"
    else:
        inv["status"] = "ABSENT_AS_EXPECTED_SELF_BUILD_SOURCE_AUDIT"
    missing = [r for r in items if r["status"] == "MISSING_ARTIFACT"]
    payload = {
        "round": 21,
        "current": CURRENT_BASELINE,
        "alias": CURRENT_ALIAS,
        "parent": PARENT_V4,
        "v5_anchor": dict(V5_ANCHOR),
        "r20_prior": dict(R20_PRIOR),
        "files": items,
        "n_ok": sum(1 for r in items if r["status"] in {"OK", "ABSENT_AS_EXPECTED", "ABSENT_AS_EXPECTED_SELF_BUILD_SOURCE_AUDIT"}),
        "n_missing": len(missing),
        "missing_keys": [r["key"] for r in missing],
        "round19_1_exists": bool(r19_1["exists"]),
        "project_data_inventory_exists": bool(inv["exists"]),
    }
    dump_json(READ_JSON, payload)
    lines = [
        "# Round21 · prior 只读核验",
        "",
        f"CURRENT=`{CURRENT_BASELINE}` = `{CURRENT_ALIAS}` parent=`{PARENT_V4}`。",
        f"R20 PRIOR：盘点 {R20_PRIOR['inventory_n']} / PIT {R20_PRIOR['pit_n']} / KEEP {R20_PRIOR['keep_n']} / combo {R20_PRIOR['combo_n']} / best `{R20_PRIOR['best_code']}` Sharpe {R20_PRIOR['best_sharpe']} < {R20_PRIOR['v5_sharpe']}。",
        "",
        f"round19_1 存在=`{r19_1['exists']}`（计划：仍不存在）。`result/data_inventory/` 存在=`{inv['exists']}`（计划：不存在，本轮自建 source_audit）。",
        "",
        f"missing={payload['missing_keys']}",
        "",
    ]
    for rec in items:
        lines.append(f"- `{rec['key']}` {rec['status']} sha256=`{rec['sha256']}` path=`{rec['path']}`")
    lines.append("")
    write_md(READ_MD, "\n".join(lines))
    return payload
