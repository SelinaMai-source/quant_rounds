"""冻结 round8_manifest_v1.yaml + sha256。freeze 之后新增组标 POST_HOC_EXPLORATORY，不得参与 v3 promotion。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..paths import ROUND8_ROOT, assert_project_result_path, ensure_round8_tree
from ..stock_readonly import sha256_file
from .contract import (
    ANCHOR_DIR,
    BASELINE_V1,
    BASELINE_V2,
    BLEND_PRED,
    CLEAN_RISK_IN_V2,
    MANIFEST_NAME,
    PARENT_CODE,
    R8_00,
    RANK_NAV,
    RISK_PRED,
    ROLLING_MIN_PERIODS,
    ROLLING_WINDOW,
    TARGET_CAGR,
    TARGET_MAXDD,
    TARGET_N_TRADES,
    TARGET_SHARPE,
    TARGET_TOTAL_RETURN,
    TOP_K,
    USED_LOCKED_ROUND7_RISK,
    V2_OVERLAY,
    XS_BOTTOM,
    XS_LOW_MULT,
)
from .registry import SPECS

SOURCE_FILES = (
    "core/round8/contract.py",
    "core/round8/registry.py",
    "core/round8/overlays.py",
    "core/round8/runner.py",
    "core/round8/parity.py",
    "core/round8/geometry.py",
    "core/replay_round7/inject.py",
    "core/replay_round7/runner.py",
    "core/local_executor.py",
)


def manifest_path():
    return ROUND8_ROOT / MANIFEST_NAME


def sha256_path():
    return ROUND8_ROOT / f"{MANIFEST_NAME}.sha256"


def _dump_yaml(obj: Any) -> str:
    from ..round3.runner import _dump_simple_yaml

    text = _dump_simple_yaml(obj)
    if not text.endswith("\n"):
        text += "\n"
    return text


def build_manifest_payload(*, frozen_at: str | None = None) -> dict[str, Any]:
    from ..paths import PROJECT_ROOT

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
        })
    return {
        "manifest_id": "round8_manifest_v1",
        "frozen_at": frozen_at or datetime.now().astimezone().isoformat(),
        "rule_zh": "freeze 之后新增组必须标 POST_HOC_EXPLORATORY，不能参与 baseline_v3 promotion。",
        "auto_promote_v3": False,
        "status": ["historical_development", "NOT_final_OOS", "NOT_production_validated"],
        "lineage": {
            "baseline_v1": BASELINE_V1,
            "baseline_v2": BASELINE_V2,
            "parent_artifact": PARENT_CODE,
            "r8_00": R8_00,
            "p08_role": "secondary_challenger_not_default_v2",
            "ecology_default_in_v2": False,
        },
        "alpha": {
            "locked": "dual90_10_gap_floor",
            "blend_pred": str(BLEND_PRED),
            "risk_pred": str(RISK_PRED),
            "clean_risk": CLEAN_RISK_IN_V2,
            "used_locked_round7_risk": USED_LOCKED_ROUND7_RISK,
            "score_mutated": False,
        },
        "market": {
            "name": "B01a",
            "kind": "breadth_tier",
            "t1": 0.45,
            "t2": 0.50,
            "t3": 0.60,
            "mapping_zh": "<45%→0；[45,50)×0.50；[50,60)×0.75；≥60%→1.0",
        },
        "confidence": {
            "feature": "top1_minus_median",
            "pit": f"rolling_{ROLLING_WINDOW}",
            "min_periods": ROLLING_MIN_PERIODS,
            "p_le": XS_BOTTOM,
            "low_mult": XS_LOW_MULT,
            "nan_to": 1.0,
        },
        "combine": {
            "kind": "stack",
            "how": "market_x_confidence",
            "overlay": dict(V2_OVERLAY),
        },
        "capacity": {
            "catalog_top_k_ignored": 1,
            "forced_top_k": int(TOP_K),
            "rank_name_budget_nav_fractions": list(RANK_NAV),
        },
        "execution": {
            "allocation_multiplier": "date_level",
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
        },
        "source_sha256": sources,
        "experiments": experiments,
        "n_experiments": len(experiments),
    }


def freeze_manifest(*, overwrite: bool = False) -> dict[str, Any]:
    """写入 manifest + sidecar sha256。已冻结则只校验，不改内容。"""
    ensure_round8_tree()
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
