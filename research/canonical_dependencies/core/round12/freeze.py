"""冻结 round12_manifest_v1.yaml + sha256。必须在任何新策略绩效之前。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..paths import ROUND12_ROOT, assert_project_result_path, ensure_round12_tree
from ..round11.contract import source_tree_sha256, artifact_file_hashes
from ..round11.freeze import _dump_yaml
from ..stock_readonly import sha256_file
from .contract import (
    AUTO_PROMOTE_V5,
    BASELINE_V4,
    BLEND_PRED,
    CAPITALS,
    FALLBACK_P50_BPS,
    FALLBACK_P75_BPS,
    FALLBACK_P90_BPS,
    FILL_SEMANTICS,
    KEEP_BASELINE_V4,
    MANIFEST_NAME,
    PARENT_V4,
    PARTICIPATION,
    PRE_REGISTERED_IDS,
    PRE_REGISTERED_QUESTIONS,
    PROTOCOL_SOURCE_FILES,
    R11_V4_CORRECTED,
    R11_V4_CORRECTED_DIR,
    R12_01_TARGETS,
    RISK_PRED,
    SNAPSHOT_FINGERPRINT,
    SNAPSHOT_ID,
    STAGES,
    STRATEGY_SOURCE_FILES,
    T_DECISION,
    V4_DIR,
    V4_TARGETS,
    VARIANT_IDS,
    pre_registered_spec,
)


def manifest_path():
    return ROUND12_ROOT / MANIFEST_NAME


def sha256_path():
    return ROUND12_ROOT / f"{MANIFEST_NAME}.sha256"


def build_manifest_payload(*, frozen_at: str | None = None) -> dict[str, Any]:
    strategy = source_tree_sha256(STRATEGY_SOURCE_FILES)
    protocol = source_tree_sha256(PROTOCOL_SOURCE_FILES)
    return {
        "manifest_id": "round12_manifest_v1",
        "frozen_at": frozen_at or datetime.now().astimezone().isoformat(),
        "mission": "Risk Remediation & Deployability. NOT an Alpha search.",
        "rule_zh": (
            "freeze 之后再看新策略成绩。之后新增实验一律 POST_HOC_DIAGNOSTIC。"
            "不得自动升 baseline_v5。默认 KEEP_BASELINE_V4。"
            "正式策略实验仅预注册 8 个 ID。"
        ),
        "auto_promote_v5": AUTO_PROMOTE_V5,
        "default_promotion": KEEP_BASELINE_V4,
        "status": ["historical_development", "NOT_final_OOS", "NOT_production_validated"],
        "lineage": {
            "baseline_v4": BASELINE_V4,
            "parent": PARENT_V4,
            "realism_reference": R11_V4_CORRECTED,
            "realism_reference_is_not_v5": True,
        },
        "snapshot": {
            "id": SNAPSHOT_ID,
            "fingerprint": SNAPSHOT_FINGERPRINT,
        },
        "execution": {
            "engine": "local_executor -> Stock fixed_sleeve_backtest",
            "exit_mode": "close_from_d2",
            "t_decision": T_DECISION,
            "fill_semantics": FILL_SEMANTICS,
            "legacy_sell_tax_bps": 5.0,
            "corrected_fee": "historical_stamp_tax 10bp→5bp at 2023-08-28",
            "commission_transfer": "FEE_HISTORY_UNVERIFIED",
        },
        "liquidity_rule": {
            "pre_registered": True,
            "participation": PARTICIPATION,
            "amount_field": "signal-day daily_raw.amount",
            "priority": "Top1 first, leftover to Top2",
            "capitals": list(CAPITALS),
            "diagnostic_capital": 10_000_000.0,
            "no_search": True,
        },
        "execution_fallback_bps": {
            "P50": FALLBACK_P50_BPS,
            "P75": FALLBACK_P75_BPS,
            "P90": FALLBACK_P90_BPS,
            "note": "分钟不可识别时使用；不是新搜索。",
        },
        "pre_registered_ids": list(PRE_REGISTERED_IDS),
        "variant_ids": list(VARIANT_IDS),
        "experiments": [pre_registered_spec(code) for code in PRE_REGISTERED_IDS],
        "legacy_v4_targets": dict(V4_TARGETS),
        "r12_01_targets": dict(R12_01_TARGETS),
        "legacy_v4_sha256": artifact_file_hashes(V4_DIR) if V4_DIR.is_dir() else {},
        "r11_v4_corrected_sha256": artifact_file_hashes(R11_V4_CORRECTED_DIR) if R11_V4_CORRECTED_DIR.is_dir() else {},
        "locked_predictions": {
            "blend": {"path": str(BLEND_PRED), "sha256": sha256_file(BLEND_PRED) if BLEND_PRED.is_file() else "MISSING"},
            "risk": {"path": str(RISK_PRED), "sha256": sha256_file(RISK_PRED) if RISK_PRED.is_file() else "MISSING"},
        },
        "source_tree_sha256": strategy,
        "protocol_sha256": protocol,
        "questions": dict(PRE_REGISTERED_QUESTIONS),
        "stages": list(STAGES),
        "forbidden": [
            "new_factors",
            "xgb_retrain",
            "90_10_search",
            "B03_threshold_search",
            "exit_search",
            "timing_as_alpha",
            "calibration_as_trade_rule",
            "invent_25bp_because_R12_03_looks_good",
            "auto_promote_v5",
            "write_写入时间_to_tracker",
        ],
        "git_commit": None,
        "git_commit_note": "无可用 git commit；source_tree_sha256 是文件哈希，不伪造 commit",
    }


def freeze_manifest(*, overwrite: bool = False) -> dict[str, Any]:
    ensure_round12_tree()
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
        raise RuntimeError("MANIFEST_NOT_FROZEN STOP — 必须先 freeze 再看新策略成绩")
    digest = sha256_file(path)
    sidecar = digest_path.read_text(encoding="utf-8").strip().split()[0]
    if digest != sidecar:
        raise RuntimeError(f"manifest sha256 mismatch: file={digest} sidecar={sidecar}")
    return {"path": str(path), "sha256": digest}


__all__ = ["build_manifest_payload", "freeze_manifest", "manifest_path", "require_frozen", "sha256_path"]
