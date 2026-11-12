"""看 NAV 前冻结公式、矩阵与晋级门。生成后禁止搜 epsilon。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND18_ROOT
from ..round11.contract import source_tree_sha256
from ..round11.freeze import _dump_yaml
from .common import dump_json, load_json, sha256_json, write_md
from .contract import (
    COVERAGE_MIN,
    EXPERIMENTS,
    F1_FORMULA,
    F1_NAME,
    F2_FORMULA,
    F2_NAME,
    F3_FORMULA,
    F3_NAME,
    FORBIDDEN,
    MANIFEST_NAME,
    MAXDD_WORSEN_CAP,
    PRIOR_RESEARCH_SELECTION_EXISTS,
    PROMOTION_GATES,
    PROTOCOL_SOURCE_FILES,
    SUBSAMPLE,
)

FREEZE_JSON = ROUND18_ROOT / "configs" / "freeze.json"
FREEZE_MD = ROUND18_ROOT / "07_frozen_manifest.md"


def manifest_path():
    return ROUND18_ROOT / MANIFEST_NAME


def require_frozen() -> dict[str, Any]:
    frozen = load_json(FREEZE_JSON)
    if not frozen.get("frozen"):
        raise RuntimeError("manifest not frozen; refuse to train/backtest candidates")
    return frozen


def freeze_manifest(*, overwrite: bool = False) -> dict[str, Any]:
    if FREEZE_JSON.is_file() and not overwrite:
        existing = load_json(FREEZE_JSON)
        if existing.get("frozen"):
            return existing
    pit = load_json(ROUND18_ROOT / "pit" / "pit.json")
    build = load_json(ROUND18_ROOT / "factors" / "factor_build.json")
    body = {
        "round": 18,
        "subsample": SUBSAMPLE,
        "colsample": SUBSAMPLE,
        "formulas": {
            F1_NAME: F1_FORMULA,
            F2_NAME: F2_FORMULA,
            F3_NAME: F3_FORMULA,
        },
        "epsilon_locked": {"F1_A_day_coef": 0.01},
        "experiments": list(EXPERIMENTS),
        "promotion_gates": PROMOTION_GATES,
        "maxdd_worsen_cap": MAXDD_WORSEN_CAP,
        "coverage_min": COVERAGE_MIN,
        "prior_research_selection_exists": PRIOR_RESEARCH_SELECTION_EXISTS,
        "forbidden": list(FORBIDDEN),
        "pit_snapshot": {
            name: {k: (rec or {}).get(k) for k in ("coverage", "pit_status", "promote_ok", "n_valid")}
            for name, rec in (pit.get("factors") or {}).items()
        },
        "build_n_valid": (build.get("n_valid") or {}),
        "protocol_sha256": source_tree_sha256(PROTOCOL_SOURCE_FILES) if hasattr(source_tree_sha256, "__call__") else None,
    }
    try:
        body["protocol_sha256"] = source_tree_sha256(PROTOCOL_SOURCE_FILES)
    except Exception:
        body["protocol_sha256"] = None
    path = manifest_path()
    path.write_text(_dump_yaml(body), encoding="utf-8")
    digest = sha256_json(body)
    payload = {"frozen": True, "sha256": digest, "path": str(path), "manifest": body}
    dump_json(FREEZE_JSON, payload)
    (ROUND18_ROOT / f"{MANIFEST_NAME}.sha256").write_text(digest + "\n", encoding="utf-8")
    write_md(
        FREEZE_MD,
        "# 07 冻结清单\n\n"
        f"sha256=`{digest}`\n\n"
        f"- F1 `{F1_FORMULA}`\n- F2 `{F2_FORMULA}`\n- F3 `{F3_FORMULA}`\n"
        f"- subsample=colsample=1.0\n- 晋级门已锁，事后不放宽。\n"
        "此后禁止因看到 NAV 而改公式/矩阵/门槛。\n",
    )
    return payload
