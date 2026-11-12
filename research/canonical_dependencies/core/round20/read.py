"""读 R7–R19 契约与报告，核真实路径与 sha256。不伪造。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import PROJECT_ROOT, RESULT_ROOT, ROUND20_ROOT
from .common import dump_json, file_sha256, write_md
from .contract import (
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    PARENT_V4,
    PRIOR_CONSTRAINTS,
    PRIOR_RESEARCH_SELECTION_EXISTS,
    V5_ANCHOR,
    copy_plan,
)

READ_JSON = ROUND20_ROOT / "read" / "prior_artifacts.json"
READ_MD = ROUND20_ROOT / "read" / "00_prior_read.md"

PRIOR_FILES: tuple[tuple[str, Path], ...] = (
    ("r7_contract_py", PROJECT_ROOT / "core" / "round7" / "contract.py"),
    ("r7_train_py", PROJECT_ROOT / "core" / "round7" / "train.py"),
    ("r7_contract_json", RESULT_ROOT / "round7" / "contract.json"),
    ("r8_contract_py", PROJECT_ROOT / "core" / "round8" / "contract.py"),
    ("r9_contract_py", PROJECT_ROOT / "core" / "round9" / "contract.py"),
    ("r11_contract_py", PROJECT_ROOT / "core" / "round11" / "contract.py"),
    ("r13_contract_py", PROJECT_ROOT / "core" / "round13" / "contract.py"),
    ("r13_report", RESULT_ROOT / "round13" / "ROUND13_BASELINE_CONSOLIDATION_REPORT.md"),
    ("r16_contract_py", PROJECT_ROOT / "core" / "round16" / "contract.py"),
    ("r16_report", RESULT_ROOT / "round16" / "ROUND16_OPTION_RESEARCH_REPORT.md"),
    ("r16_manifest", RESULT_ROOT / "round16" / "round16_manifest_v1.yaml"),
    ("r16_5_contract_py", PROJECT_ROOT / "core" / "round16_5" / "contract.py"),
    ("r16_5_report", RESULT_ROOT / "round16_5" / "ROUND16_5_CONFIRMATORY_REPORT.md"),
    ("r16_5_manifest", RESULT_ROOT / "round16_5" / "round16_5_manifest_v1.yaml"),
    ("r17_contract_py", PROJECT_ROOT / "core" / "round17" / "contract.py"),
    ("r17_report", RESULT_ROOT / "round17" / "ROUND17_LARGE_SCALE_FACTOR_SEARCH_REPORT.md"),
    ("r17_manifest", RESULT_ROOT / "round17" / "round17_search_manifest.yaml"),
    ("r17_factor_registry", RESULT_ROOT / "round17" / "inventory" / "factor_registry.csv"),
    ("r17_inventory_md", RESULT_ROOT / "round17" / "00_existing_factor_inventory.md"),
    ("r18_contract_py", PROJECT_ROOT / "core" / "round18" / "contract.py"),
    ("r18_report", RESULT_ROOT / "round18" / "ROUND18_MICROSTRUCTURE_ALPHA_REPORT.md"),
    ("r18_manifest", RESULT_ROOT / "round18" / "round18_microstructure_manifest.yaml"),
    ("r18_pit_json", RESULT_ROOT / "round18" / "pit" / "pit.json"),
    ("r18_pit_md", RESULT_ROOT / "round18" / "06_pit_coverage.md"),
    ("r19_contract_py", PROJECT_ROOT / "core" / "round19" / "contract.py"),
    ("r19_report", RESULT_ROOT / "round19" / "ROUND19_RESEARCH_REPORT.md"),
    ("r19_manifest_a", RESULT_ROOT / "round19" / "round19_track_a_manifest.yaml"),
    ("r19_manifest_b", RESULT_ROOT / "round19" / "round19_track_b_manifest.yaml"),
    ("replay_r7_caliber_py", PROJECT_ROOT / "tools" / "replay_r7_caliber.py"),
    ("replay_v5_summary", RESULT_ROOT / "replay_r7_caliber" / "v5_R16_O01" / "replay_summary.json"),
    ("replay_v5_nav", RESULT_ROOT / "replay_r7_caliber" / "v5_R16_O01" / "nav.parquet"),
    ("replay_v5_trades", RESULT_ROOT / "replay_r7_caliber" / "v5_R16_O01" / "trades.parquet"),
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
    r19_1 = RESULT_ROOT / "round19_1"
    real_root = PROJECT_ROOT.resolve()
    payload = {
        "status": "DONE",
        "current": CURRENT_BASELINE,
        "current_alias": CURRENT_ALIAS,
        "parent": PARENT_V4,
        "user_lock": True,
        "auto_promote": False,
        "prior_research_selection_exists": PRIOR_RESEARCH_SELECTION_EXISTS,
        "v5_anchor": dict(V5_ANCHOR),
        "project_root": str(PROJECT_ROOT),
        "project_root_resolved": str(real_root),
        "n_ok": int(sum(1 for r in items if r["exists"])),
        "n_missing": len(missing),
        "files": items,
        "missing": missing,
        "round19_1_exists": r19_1.exists(),
        "round19_1_note": "result/round19_1/ 不存在；R19.1 数字以追踪表 + replay_r7_caliber 为准。",
        "prior_constraints": dict(PRIOR_CONSTRAINTS),
        "constraints": {
            "subsample": 1.0,
            "colsample": 1.0,
            "no_o01_b03_90_10_changes": True,
            "no_auto_promote": True,
            "no_写入时间": True,
            "no_invent_factors": True,
        },
        "note": "缺失标 MISSING_ARTIFACT。不伪造数字。CURRENT = baseline_v5 = R16_O01 USER_LOCK。",
    }
    dump_json(READ_JSON, payload)
    lines = [
        "# Round20 · 先验读档 R7–R19",
        "",
        f"CURRENT=`{CURRENT_BASELINE}` alias=`{CURRENT_ALIAS}` parent=`{PARENT_V4}` USER_LOCK。禁止自动升 CURRENT。",
        f"课题路径 `{PROJECT_ROOT}` → 实路径 `{real_root}`。",
        f"OK=`{payload['n_ok']}` MISSING_ARTIFACT=`{payload['n_missing']}`。",
        f"`result/round19_1/` exists=`{payload['round19_1_exists']}`。{payload['round19_1_note']}",
        "",
        "## 已知约束（PRIOR_*，不得伪装新发现）",
        "",
        "- 因子库：R17 `factor_registry.csv` 75 行（基线 14/13 + unused 31 + R17 新 30）+ R18 F1/F2/F3。不得捏造清单外因子。",
        "- R18 F2 `PIT_TIMESTAMP_UNVERIFIED`，不得进可晋级模型。F1/F3 公式已差，可进审计库但不得换名重开假设。",
        "- R17 5000 搜索已测；本轮禁止再做 5000 随机搜索。",
        "- R19 Track B：无 material 训练 bug；A00=locked vs open 重训原特征一致。",
        "- Native 5 seed 在 1.0/1.0 下排序退化。100 个原生 seed = 100 个相同模型。",
        "- 2018–2026 均已被 R1–19 研究过 → Discovery/Validation/Audit 不是独立 OOS。",
        "",
        "## 已核 sha256 的契约 / 报告 / manifest",
        "",
    ]
    for rec in items:
        digest = rec.get("sha256") or "MISSING"
        lines.append(f"- `{rec['key']}` exists=`{rec['exists']}` sha256=`{digest}`")
        lines.append(f"  `{rec['resolved']}`")
    lines.extend(["", "主评价 CAGR 用日历年 365.25。v5 主回测 = Round7 5bp 口径，不传 validation_hooks。", ""])
    if missing:
        lines.append("缺失：")
        for m in missing:
            lines.append(f"- `{m['path']}`")
        lines.append("")
    write_md(READ_MD, "\n".join(lines))
    return payload
