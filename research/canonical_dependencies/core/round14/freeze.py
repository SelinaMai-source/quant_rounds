"""冻结 round14_manifest_v1.yaml + sha256。必须在风险诊断之后、任何候选绩效之前。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..paths import ROUND14_ROOT, assert_project_result_path, ensure_round14_tree
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
    COMPLEXITY_ORDER,
    CONTROL_IDS,
    INCLUDE_D,
    KEEP_BASELINE_V4,
    MANIFEST_NAME,
    PRE_REGISTERED_IDS,
    PRE_REGISTERED_QUESTIONS,
    PROTOCOL_SOURCE_FILES,
    R12_01_DIR,
    R12_01_TARGETS,
    R13_00_DIR,
    RANK_NAV,
    REF_MIN_OBS,
    REF_SIGNAL_DAYS,
    RHO_MIN_OBS,
    RHO_WINDOW,
    RISK_PRED,
    S8_UP_RATIO,
    SIGMA_MIN_OBS,
    SIGMA_WINDOW,
    SNAPSHOT_FINGERPRINT,
    SNAPSHOT_ID,
    STAGES,
    STRATEGY_SOURCE_FILES,
    TOP_K,
    pre_registered_spec,
)
from .risk import DIAG_MD, RISK_PARQUET, require_risk


def manifest_path():
    return ROUND14_ROOT / MANIFEST_NAME


def sha256_path():
    return ROUND14_ROOT / f"{MANIFEST_NAME}.sha256"


def build_manifest_payload(*, frozen_at: str | None = None) -> dict[str, Any]:
    strategy = source_tree_sha256(STRATEGY_SOURCE_FILES)
    protocol = source_tree_sha256(PROTOCOL_SOURCE_FILES)
    risk_sha = sha256_file(RISK_PARQUET) if RISK_PARQUET.is_file() else "MISSING"
    return {
        "manifest_id": "round14_manifest_v1",
        "frozen_at": frozen_at or datetime.now().astimezone().isoformat(),
        "mission": "Alpha-Preserving Risk Budgeting. NOT stock-selection search. Only Top1/Top2 target capital.",
        "rule_zh": (
            "parity → 风险诊断 → freeze → 候选。freeze 之后新增一律 POST_HOC_DIAGNOSTIC。"
            "不得自动升 CURRENT baseline_v5。过门则 RECOMMENDED_BASELINE_V5 + AWAITING_USER_LOCK。"
            "默认 KEEP_BASELINE_V4。DE_RISKING_ONLY 不能升 v5。正式候选仅 A01/B01/B02/B03/C01。"
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
            "identity_lock": "daily Top1/Top2 code == v4 else FAIL_ALPHA_MUTATION",
        },
        "risk_features": {
            "include_d": INCLUDE_D,
            "sigma_window": SIGMA_WINDOW,
            "sigma_min_obs": SIGMA_MIN_OBS,
            "rho_window": RHO_WINDOW,
            "rho_min_obs": RHO_MIN_OBS,
            "ref_signal_days": REF_SIGNAL_DAYS,
            "ref_min_obs": REF_MIN_OBS,
            "missing_fill_zero": False,
            "fallback_to_v4": True,
            "rho_missing": "explicit no-corr-adjust, not silent zero",
            "mrc2_le_0": "risk cap factor = 1",
            "panel_sha256": risk_sha,
            "diagnostics_path": str(DIAG_MD),
            "diagnostics_changed_formulas": False,
        },
        "candidates": {k: v for k, v in CANDIDATE_FORMULAS.items()},
        "complexity_order": list(COMPLEXITY_ORDER),
        "controls": {
            "kind": "exposure_matched",
            "scale": "E_cand / E_v4",
            "w_ctrl": "w_v4 * scale",
            "risk_information": False,
            "ids": list(CONTROL_IDS),
            "key_test": "B02 vs CTRL_EXP_B02",
        },
        "scenarios": {
            "S0": "corrected normal execution",
            "S1": "+20bp buy adverse",
            "S2": "+30bp buy adverse",
            "S3": "1M liquidity 3% Top1-first leftover",
            "S4": "3M liquidity 3% Top1-first leftover",
            "S5": "Ex2024",
            "S6": "Ex2025",
            "S7": "Ex2024+2025",
            "S8": f"B01A up_ratio < {S8_UP_RATIO}",
            "S9": "worst historical rolling 12M from existing daily returns",
            "no_new_cutoffs": True,
            "b01a_t1": B01A_T1,
        },
        "statistics": {
            "block_bootstrap_days": 20,
            "n_boot": 5000,
            "n_boot_min": 2000,
            "matched_exposure_bootstrap": True,
            "cpcv": "round11.cpcv_oos_sharpe embargo=12 block>=60",
            "spa": "Hansen SPA family vs corrected v4",
        },
        "snapshot": {"id": SNAPSHOT_ID, "fingerprint": SNAPSHOT_FINGERPRINT},
        "pre_registered_ids": list(PRE_REGISTERED_IDS),
        "control_ids": list(CONTROL_IDS),
        "candidate_ids": list(CANDIDATE_IDS),
        "experiments": [pre_registered_spec(code) for code in PRE_REGISTERED_IDS],
        "r12_01_targets": dict(R12_01_TARGETS),
        "anchor_sha256": {
            "R12_01": artifact_file_hashes(R12_01_DIR) if R12_01_DIR.is_dir() else {},
            "R13_00": artifact_file_hashes(R13_00_DIR) if R13_00_DIR.is_dir() else {},
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
            "change_top1_top2_identity",
            "cartesian_product",
            "vol_target_grid_10_12_15",
            "auto_promote_CURRENT_v5",
            "promote_DE_RISKING_ONLY",
            "write_写入时间_to_tracker",
            "diagnostics_change_formulas",
        ],
        "git_commit": None,
        "git_commit_note": "无可用 git commit；source_tree_sha256 是文件哈希，不伪造 commit",
    }


def freeze_manifest(*, overwrite: bool = False) -> dict[str, Any]:
    require_risk()
    ensure_round14_tree()
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
    }


def require_frozen() -> dict[str, Any]:
    path = manifest_path()
    digest_path = sha256_path()
    if not path.is_file() or not digest_path.is_file():
        raise RuntimeError("MANIFEST_NOT_FROZEN STOP — 必须先风险诊断再 freeze，再看候选成绩")
    digest = sha256_file(path)
    sidecar = digest_path.read_text(encoding="utf-8").strip().split()[0]
    if digest != sidecar:
        raise RuntimeError(f"manifest sha256 mismatch: file={digest} sidecar={sidecar}")
    return {"path": str(path), "sha256": digest}


__all__ = ["build_manifest_payload", "freeze_manifest", "manifest_path", "require_frozen", "sha256_path"]
