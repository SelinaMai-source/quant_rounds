"""冻结 round9_manifest_v1.yaml + sha256。必须在 R9_00 PASS 之后、任何非 parity 实验之前调用。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..paths import PROJECT_ROOT, ROUND9_ROOT, assert_project_result_path, ensure_round9_tree
from ..stock_readonly import sha256_file
from .contract import (
    ANCHOR_DIR,
    BASELINE_V1,
    BASELINE_V2,
    BASELINE_V3,
    BLEND_PRED,
    DISAGREE_ALIAS,
    DISAGREE_MULT,
    E02_OVERLAY,
    MANIFEST_NAME,
    META_XGB,
    PARENT_CODE,
    R9_00,
    RANK_NAV,
    RISK_PRED,
    ROLLING_MIN_PERIODS,
    ROLLING_WINDOW,
    TARGET_CAGR,
    TARGET_LOYO_MED,
    TARGET_LOYO_WORST,
    TARGET_MAXDD,
    TARGET_N_TRADES,
    TARGET_SHARPE,
    TARGET_TOTAL_RETURN,
    TOP_K,
    V2_OVERLAY,
    V4_GATES,
    XS_BOTTOM,
    XS_LOW_MULT,
)
from .registry import SPECS

SOURCE_FILES = (
    "core/round9/contract.py",
    "core/round9/registry.py",
    "core/round9/overlays.py",
    "core/round9/runner.py",
    "core/round9/parity.py",
    "core/round9/geometry.py",
    "core/round8/overlays.py",
    "core/round8/geometry.py",
    "core/replay_round7/inject.py",
    "core/local_executor.py",
)


def manifest_path():
    return ROUND9_ROOT / MANIFEST_NAME


def sha256_path():
    return ROUND9_ROOT / f"{MANIFEST_NAME}.sha256"


def _dump_yaml(obj: Any) -> str:
    from ..round3.runner import _dump_simple_yaml

    text = _dump_simple_yaml(obj)
    if not text.endswith("\n"):
        text += "\n"
    return text


def build_manifest_payload(*, frozen_at: str | None = None) -> dict[str, Any]:
    sources = {}
    for rel in SOURCE_FILES:
        path = PROJECT_ROOT / rel
        sources[rel] = sha256_file(path) if path.is_file() else "MISSING"
    experiments = []
    for code, spec in SPECS.items():
        experiments.append({
            "code": code,
            "title_zh": spec.title_zh,
            "stage": spec.stage,
            "parent": spec.parent,
            "post_hoc": bool(spec.post_hoc),
            "diagnostic": bool(spec.diagnostic),
            "skip": bool(spec.skip),
            "skip_reason": spec.skip_reason,
            "predictions": spec.predictions,
            "overlay_kind": (spec.overlay_spec or {}).get("kind"),
            "allocator": (spec.overlay_spec or {}).get("allocator"),
        })
    return {
        "manifest_id": "round9_manifest_v1",
        "frozen_at": frozen_at or datetime.now().astimezone().isoformat(),
        "rule_zh": "freeze 之后新增组必须标 POST_HOC_EXPLORATORY。不自动升 baseline_v4。",
        "auto_promote_v4": False,
        "status": ["historical_development", "NOT_final_OOS", "NOT_production_validated"],
        "lineage": {
            "baseline_v1": BASELINE_V1,
            "baseline_v2": BASELINE_V2,
            "baseline_v3": BASELINE_V3,
            "parent_artifact": PARENT_CODE,
            "r9_00": R9_00,
        },
        "alpha": {
            "locked": "dual90_10_gap_floor",
            "blend_pred": str(BLEND_PRED),
            "risk_pred": str(RISK_PRED),
            "score_mutated": False,
            "meta_xgb": dict(META_XGB),
        },
        "v2_bottom": {
            "market": "B01a breadth_tier 45/50/60",
            "confidence": "top1_minus_median rolling250 p<=0.20 x0.5",
            "pit": f"rolling_{ROLLING_WINDOW}",
            "min_periods": ROLLING_MIN_PERIODS,
            "p_le": XS_BOTTOM,
            "low_mult": XS_LOW_MULT,
            "overlay": dict(V2_OVERLAY),
        },
        "e02": {
            "overlay": dict(E02_OVERLAY),
            "disagree_states": "A/B/C",
            "alias": dict(DISAGREE_ALIAS),
            "top2_only_mult": dict(DISAGREE_MULT),
            "top1_lock": True,
            "dir": str(ANCHOR_DIR),
        },
        "capacity": {
            "forced_top_k": int(TOP_K),
            "rank_name_budget_nav_fractions": list(RANK_NAV),
        },
        "execution": {
            "engine": "local_executor -> Stock fixed_sleeve_backtest",
            "exit_mode": "close_from_d2",
        },
        "anchor": {
            "dir": str(ANCHOR_DIR),
            "cagr": TARGET_CAGR,
            "sharpe": TARGET_SHARPE,
            "max_drawdown": TARGET_MAXDD,
            "completed_trades": TARGET_N_TRADES,
            "total_return": TARGET_TOTAL_RETURN,
            "loyo_median": TARGET_LOYO_MED,
            "loyo_worst": TARGET_LOYO_WORST,
        },
        "v4_gates": dict(V4_GATES),
        "source_sha256": sources,
        "experiments": experiments,
        "n_experiments": len(experiments),
    }


def freeze_manifest(*, overwrite: bool = False) -> dict[str, Any]:
    """写入 manifest + sidecar sha256。已冻结则只校验，不改内容。"""
    ensure_round9_tree()
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
            "n_experiments": len(SPECS),
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
        "n_experiments": payload["n_experiments"],
    }


__all__ = ["build_manifest_payload", "freeze_manifest", "manifest_path", "sha256_path"]
