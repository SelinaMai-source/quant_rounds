"""冻结 round13_manifest_v1.yaml + sha256。必须在任何候选绩效之前。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..paths import ROUND13_ROOT, assert_project_result_path, ensure_round13_tree
from ..round11.contract import artifact_file_hashes, source_tree_sha256
from ..round11.freeze import _dump_yaml
from ..stock_readonly import sha256_file
from .contract import (
    AUTO_PROMOTE_V5,
    B01A_T1,
    BLEND_PRED,
    CANDIDATE_FORMULAS,
    CANDIDATE_IDS,
    CAPITALS,
    CONTROL_IDS,
    CTRL_V2_TARGETS,
    CTRL_V3_TARGETS,
    KEEP_BASELINE_V4,
    MANIFEST_NAME,
    PRE_REGISTERED_IDS,
    PRE_REGISTERED_QUESTIONS,
    PROTOCOL_SOURCE_FILES,
    R11_V2_CORRECTED_DIR,
    R11_V3_CORRECTED_DIR,
    R12_01_DIR,
    R12_01_TARGETS,
    RANK_NAV,
    RISK_PRED,
    S10_UP_RATIO,
    S9_STATUS,
    SNAPSHOT_FINGERPRINT,
    SNAPSHOT_ID,
    STAGES,
    STRATEGY_SOURCE_FILES,
    TOP_K,
    pre_registered_spec,
)


def manifest_path():
    return ROUND13_ROOT / MANIFEST_NAME


def sha256_path():
    return ROUND13_ROOT / f"{MANIFEST_NAME}.sha256"


def build_manifest_payload(*, frozen_at: str | None = None) -> dict[str, Any]:
    strategy = source_tree_sha256(STRATEGY_SOURCE_FILES)
    protocol = source_tree_sha256(PROTOCOL_SOURCE_FILES)
    return {
        "manifest_id": "round13_manifest_v1",
        "frozen_at": frozen_at or datetime.now().astimezone().isoformat(),
        "mission": "Selection-Robust Baseline Consolidation. NOT a factor/threshold search.",
        "rule_zh": (
            "freeze 之后再看候选成绩。之后新增实验一律 POST_HOC_DIAGNOSTIC。"
            "不得自动升 CURRENT baseline_v5。过门则 RECOMMENDED_BASELINE_V5 + AWAITING_USER_LOCK。"
            "默认 KEEP_BASELINE_V4。正式候选仅 A01–A05。"
        ),
        "auto_promote_v5": AUTO_PROMOTE_V5,
        "default_promotion": KEEP_BASELINE_V4,
        "status": ["historical_development", "NOT_final_OOS", "NOT_production_validated"],
        "engine": {
            "formula": "target_name_budget = entry_NAV × RANK_NAV[rank] × allocation_multiplier",
            "rank_nav": list(RANK_NAV),
            "top_k": TOP_K,
            "injection": "allocation_multiplier = clip(w / RANK_NAV[rank], MIN_KEEP, 1.0)",
            "unified_single_portfolio": True,
            "never_average_two_navs": True,
        },
        "overlays": {
            "v2": {"kind": "exact_v2"},
            "v3": {"kind": "exact_e02"},
            "v4": {"kind": "preregistered", "allocator": "B03_conservative"},
            "fused_score_unchanged": True,
        },
        "candidates": {k: v for k, v in CANDIDATE_FORMULAS.items()},
        "complexity_order": ["A01", "A03", "A02", "A04", "A05"],
        "scenarios": {
            "S0": "corrected normal execution",
            "S1": "+20bp buy adverse",
            "S2": "+30bp buy adverse",
            "S3": "1M liquidity 3% Top1-first leftover",
            "S4": "3M liquidity 3% Top1-first leftover",
            "S5": "Ex2024",
            "S6": "Ex2025",
            "S7": "Ex2024+2025",
            "S8": "worst historical rolling 12M from existing daily returns",
            "S9": S9_STATUS,
            "S10": f"B01A up_ratio < {S10_UP_RATIO}",
            "S9_excluded_from_promotion_worst_median": True,
            "S9_post_hoc_diagnostic": "down_pressure_p250 upper quartile labeled POST_HOC_DIAGNOSTIC",
            "no_new_cutoffs": True,
            "b01a_t1": B01A_T1,
        },
        "statistics": {
            "block_bootstrap_days": 20,
            "n_boot": 5000,
            "n_boot_min": 2000,
            "cpcv": "round11.cpcv_oos_sharpe embargo=12 block>=60",
            "spa": "Hansen SPA family vs corrected v4",
            "placebo_n": 1000,
            "timing_n": 500,
        },
        "snapshot": {"id": SNAPSHOT_ID, "fingerprint": SNAPSHOT_FINGERPRINT},
        "pre_registered_ids": list(PRE_REGISTERED_IDS),
        "control_ids": list(CONTROL_IDS),
        "candidate_ids": list(CANDIDATE_IDS),
        "experiments": [pre_registered_spec(code) for code in PRE_REGISTERED_IDS],
        "r12_01_targets": dict(R12_01_TARGETS),
        "ctrl_v2_targets": dict(CTRL_V2_TARGETS),
        "ctrl_v3_targets": dict(CTRL_V3_TARGETS),
        "anchor_sha256": {
            "R12_01": artifact_file_hashes(R12_01_DIR) if R12_01_DIR.is_dir() else {},
            "R11_V2_CORRECTED": artifact_file_hashes(R11_V2_CORRECTED_DIR) if R11_V2_CORRECTED_DIR.is_dir() else {},
            "R11_V3_CORRECTED": artifact_file_hashes(R11_V3_CORRECTED_DIR) if R11_V3_CORRECTED_DIR.is_dir() else {},
        },
        "locked_predictions": {
            "blend": {"path": str(BLEND_PRED), "sha256": sha256_file(BLEND_PRED) if BLEND_PRED.is_file() else "MISSING"},
            "risk": {"path": str(RISK_PRED), "sha256": sha256_file(RISK_PRED) if RISK_PRED.is_file() else "MISSING"},
        },
        "source_tree_sha256": strategy,
        "protocol_sha256": protocol,
        "questions": dict(PRE_REGISTERED_QUESTIONS),
        "stages": list(STAGES),
        "liquidity": {"participation": 0.03, "priority": "Top1 first leftover Top2", "capitals": list(CAPITALS), "no_search": True},
        "forbidden": [
            "new_factors",
            "xgb_retrain",
            "90_10_search",
            "B03_threshold_search",
            "exit_search",
            "60_40_grid",
            "A04b",
            "average_two_navs",
            "invent_S9_cutoff",
            "auto_promote_CURRENT_v5",
            "write_写入时间_to_tracker",
        ],
        "git_commit": None,
        "git_commit_note": "无可用 git commit；source_tree_sha256 是文件哈希，不伪造 commit",
    }


def freeze_manifest(*, overwrite: bool = False) -> dict[str, Any]:
    ensure_round13_tree()
    path = manifest_path()
    digest_path = sha256_path()
    assert_project_result_path(path)
    assert_project_result_path(digest_path)
    if path.is_file() and not overwrite:
        digest = sha256_file(path)
        existing = digest_path.read_text(encoding="utf-8").strip() if digest_path.is_file() else ""
        if existing and existing.split()[0] != digest:
            raise RuntimeError(f"manifest sha256 mismatch: file={digest} sidecar={existing}")
        if not digest_path.is_file():
            digest_path.write_text(digest + "  " + MANIFEST_NAME + "\n", encoding="utf-8")
        return {
            "status": "already_frozen",
            "path": str(path),
            "sha256": digest,
            "sha256_path": str(digest_path),
            "source_tree_sha256": source_tree_sha256(STRATEGY_SOURCE_FILES)["tree"],
        }
    payload = build_manifest_payload()
    path.write_text(_dump_yaml(payload), encoding="utf-8")
    digest = sha256_file(path)
    digest_path.write_text(digest + "  " + MANIFEST_NAME + "\n", encoding="utf-8")
    return {
        "status": "frozen",
        "path": str(path),
        "sha256": digest,
        "sha256_path": str(digest_path),
        "frozen_at": payload["frozen_at"],
        "source_tree_sha256": payload["source_tree_sha256"]["tree"],
        "s9": S9_STATUS,
    }


def require_frozen() -> dict[str, Any]:
    path = manifest_path()
    digest_path = sha256_path()
    if not path.is_file() or not digest_path.is_file():
        raise RuntimeError("MANIFEST_NOT_FROZEN STOP — 必须先 freeze 再看候选成绩")
    digest = sha256_file(path)
    sidecar = digest_path.read_text(encoding="utf-8").strip().split()[0]
    if digest != sidecar:
        raise RuntimeError(f"manifest sha256 mismatch: file={digest} sidecar={sidecar}")
    return {"path": str(path), "sha256": digest}


__all__ = ["build_manifest_payload", "freeze_manifest", "manifest_path", "require_frozen", "sha256_path"]
