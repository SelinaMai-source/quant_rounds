"""冻结 round11_validation_manifest_v1.yaml + sha256。必须在任何 parity 测试之前调用。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..paths import PROJECT_ROOT, ROUND11_ROOT, assert_project_result_path, ensure_round11_tree
from ..stock_readonly import sha256_file
from .contract import (
    BASELINE_V1,
    BASELINE_V2,
    BASELINE_V3,
    BASELINE_V4,
    BASELINES,
    BLEND_PRED,
    MANIFEST_NAME,
    PROTOCOL_SOURCE_FILES,
    RISK_PRED,
    SNAPSHOT_FINGERPRINT,
    SNAPSHOT_ID,
    STRATEGY_SOURCE_FILES,
    artifact_file_hashes,
    inspect_all,
    source_tree_sha256,
)


def manifest_path():
    return ROUND11_ROOT / MANIFEST_NAME


def sha256_path():
    return ROUND11_ROOT / f"{MANIFEST_NAME}.sha256"


def _dump_yaml(obj: Any) -> str:
    from ..round3.runner import _dump_simple_yaml

    text = _dump_simple_yaml(obj)
    if not text.endswith("\n"):
        text += "\n"
    return text


def build_manifest_payload(*, frozen_at: str | None = None, facts: dict[str, Any] | None = None) -> dict[str, Any]:
    facts = facts or inspect_all()
    strategy = source_tree_sha256(STRATEGY_SOURCE_FILES)
    protocol = source_tree_sha256(PROTOCOL_SOURCE_FILES)
    pred = {
        "blend": {
            "path": str(BLEND_PRED),
            "sha256": sha256_file(BLEND_PRED) if BLEND_PRED.is_file() else "MISSING",
        },
        "risk": {
            "path": str(RISK_PRED),
            "sha256": sha256_file(RISK_PRED) if RISK_PRED.is_file() else "MISSING",
        },
    }
    baselines = []
    for key, spec in BASELINES.items():
        item = (facts.get("baselines") or {}).get(key) or {}
        baselines.append({
            "version": key,
            "code": spec.code,
            "replay_code": spec.replay_code,
            "anchor_dir": str(spec.anchor_dir),
            "overlay_spec": dict(spec.overlay_spec),
            "predictions": spec.predictions,
            "parent": spec.parent,
            "targets": dict(spec.targets),
            "artifact_sha256": item.get("artifact_sha256") or artifact_file_hashes(spec.anchor_dir),
            "notes": list(spec.notes),
        })
    return {
        "manifest_id": "round11_validation_manifest_v1",
        "frozen_at": frozen_at or datetime.now().astimezone().isoformat(),
        "phase": 1,
        "rule_zh": (
            "freeze 之后再测。之后新增检验必须标 POST_HOC_DIAGNOSTIC。"
            "不产生 baseline_v5。不为过关改策略。不覆盖 legacy。"
        ),
        "auto_promote_v5": False,
        "status": ["historical_development", "NOT_final_OOS", "NOT_production_validated"],
        "lineage": {
            "baseline_v1": BASELINE_V1,
            "baseline_v2": BASELINE_V2,
            "baseline_v3": BASELINE_V3,
            "baseline_v4": BASELINE_V4,
            "v2_is_replay_round7": True,
            "v2_is_not_old_fs00263_round4": True,
        },
        "snapshot": {
            "id": SNAPSHOT_ID,
            "fingerprint": SNAPSHOT_FINGERPRINT,
            "shared_across_v1_v4": bool(facts.get("shared_snapshot")),
        },
        "execution": {
            "engine": "local_executor -> Stock fixed_sleeve_backtest",
            "exit_mode": "close_from_d2",
            "sell_tax_bps_current_contract": 5.0,
            "historical_fee_corrected_this_phase": False,
        },
        "locked_predictions": pred,
        "source_tree_sha256": strategy,
        "protocol_sha256": protocol,
        "baselines": baselines,
        "n_baselines": len(baselines),
        "later_stages_not_started": [
            "pit",
            "engine",
            "cost",
            "time",
            "model",
            "placebo",
            "trials",
            "alphaforge",
            "stats",
            "capacity",
            "dossier",
        ],
        "git_commit": None,
        "git_commit_note": "无可用 git commit；source_tree_sha256 是策略代码文件哈希，不伪造 commit",
    }


def freeze_manifest(*, overwrite: bool = False, facts: dict[str, Any] | None = None) -> dict[str, Any]:
    """写入 manifest + sidecar sha256。已冻结则只校验，不改内容。"""
    ensure_round11_tree()
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
            "n_baselines": len(BASELINES),
        }
    payload = build_manifest_payload(facts=facts)
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
        "n_baselines": payload["n_baselines"],
    }


def require_frozen() -> dict[str, Any]:
    path = manifest_path()
    digest_path = sha256_path()
    if not path.is_file() or not digest_path.is_file():
        raise RuntimeError("MANIFEST_NOT_FROZEN STOP — 必须先 freeze 再测")
    digest = sha256_file(path)
    sidecar = digest_path.read_text(encoding="utf-8").strip().split()[0]
    if digest != sidecar:
        raise RuntimeError(f"manifest sha256 mismatch: file={digest} sidecar={sidecar}")
    return {"path": str(path), "sha256": digest}


__all__ = ["build_manifest_payload", "freeze_manifest", "manifest_path", "require_frozen", "sha256_path"]
