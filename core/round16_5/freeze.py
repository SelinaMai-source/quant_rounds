"""冻结 round16_5_manifest_v1.yaml。必须在看 V03/V04 NAV 之前。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..paths import ROUND16_5_ROOT, assert_project_result_path, ensure_round16_5_tree
from ..round11.contract import source_tree_sha256
from ..round11.freeze import _dump_yaml
from ..stock_readonly import sha256_file
from .common import dump_json, load_json, write_md
from .contract import (
    AUTO_PROMOTE_V5,
    BID_ASK_NOT_AVAILABLE,
    BLOCK_DAYS,
    BOOT_N,
    CANDIDATE_FORMULAS,
    COST_BPS,
    EMBARGO,
    ENGINE_CHECK_SEEDS,
    FORBIDDEN,
    KEEP_BASELINE_V4,
    LIVE_NOT_VALIDATED,
    LOSS_NET_LT_0,
    MANIFEST_NAME,
    MASK_CHECK_N,
    MIN_BLOCK,
    O01_ANCHOR,
    O01_ROUND16,
    PCT_MIN_OBS,
    PCT_WINDOW,
    PERM_N,
    PERM_SEED,
    PRE_REGISTERED_IDS,
    PRIOR_SELECTION_EXISTS,
    PROMOTION_GATES,
    PROMISING_BUT_UNCONFIRMED,
    PROTOCOL_SOURCE_FILES,
    RANK_NAV,
    RF_FALLBACK,
    RF_GRID,
    RIDGE_ALPHA,
    R16_SPA_P,
    SEVERE_LOSS_RET,
    SKEW_Q,
    SNAPSHOT_FINGERPRINT,
    SNAPSHOT_ID,
    SPA_BOOT,
    STAGES,
    STRATEGY_SOURCE_FILES,
    TOP_K,
    V00,
    V01,
    V02,
    V03,
    V03_RIDGE_ALPHA,
    V04,
    V04_FIT_MIN,
    V04_FIT_WINDOW,
    V4_ANCHOR,
    pre_registered_spec,
)
from .parity import load_parity_status

FREEZE_MD = ROUND16_5_ROOT / "03_freeze.md"
FREEZE_JSON = ROUND16_5_ROOT / "configs" / "freeze.json"


def manifest_path():
    return ROUND16_5_ROOT / MANIFEST_NAME


def sha256_path():
    return ROUND16_5_ROOT / f"{MANIFEST_NAME}.sha256"


def freeze_manifest(*, overwrite: bool = False, facts: dict[str, Any] | None = None) -> dict[str, Any]:
    ensure_round16_5_tree()
    path = manifest_path()
    if path.is_file() and not overwrite:
        existing = {
            "status": "ALREADY_FROZEN",
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        sha_p = sha256_path()
        if sha_p.is_file():
            existing["sha256"] = sha_p.read_text(encoding="utf-8").strip()
        return existing
    parity = load_parity_status()
    if not parity.get("passed"):
        raise RuntimeError("REPRODUCIBILITY_FAIL_OR_MISSING STOP freeze")
    payload: dict[str, Any] = {
        "manifest_id": "round16_5_manifest_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "mission": "Confirmatory validation of O01 Put Skew Top2 veto. Not a factor search.",
        "PRIOR_SELECTION_EXISTS": PRIOR_SELECTION_EXISTS,
        "auto_promote_v5": AUTO_PROMOTE_V5,
        "default_promotion": KEEP_BASELINE_V4,
        "default_candidate": O01_ROUND16,
        "default_status": PROMISING_BUT_UNCONFIRMED,
        "default_live": LIVE_NOT_VALIDATED,
        "status": ["historical_development", "NOT_final_OOS", "NOT_production_validated", "NOT_independent_confirmation"],
        "closed_family": {
            V00: CANDIDATE_FORMULAS[V00],
            V01: CANDIDATE_FORMULAS[V01],
            V02: CANDIDATE_FORMULAS[V02],
            V03: CANDIDATE_FORMULAS[V03],
            V04: CANDIDATE_FORMULAS[V04],
        },
        "v03_not_candidate": True,
        "v02_not_candidate": True,
        "v04_only_new_alpha": True,
        "engine": {
            "formula": "target_name_budget = entry_NAV × RANK_NAV[rank] × allocation_multiplier",
            "rank_nav": list(RANK_NAV),
            "top_k": TOP_K,
            "zero_top2": "MIN_KEEP",
            "no_top3_fill": True,
            "backtest": "run_r15_backtest + apply_top2_veto + HISTORICAL_FEE_CORRECTED + B03",
            "placebo": "FULL_ENGINE year-stratified, not ledger daily-return swap",
        },
        "parity": {k: parity.get(k) for k in ("status", "passed")},
        "anchors": {"v4": dict(V4_ANCHOR), "o01": dict(O01_ANCHOR)},
        "o01_signal": {
            "col": "PUT_SKEW25",
            "window": PCT_WINDOW,
            "min_obs": PCT_MIN_OBS,
            "quantile": SKEW_Q,
            "shift": 1,
            "source": "core/round16/backtest.py _signals",
        },
        "placebo": {
            "n_target": PERM_N,
            "seed": PERM_SEED,
            "mask_check_n": MASK_CHECK_N,
            "engine_check_seeds": list(ENGINE_CHECK_SEEDS),
            "eligible": "v4 has real Top2 AND target_name_budget>0 AND O01 signal observable in option coverage",
            "stratification": "year primary; year x B01A-breadth bucket secondary",
            "fallback": "MATCHING_CONSTRAINT_FAILURE → pre-registered year-only merge",
            "empirical_p": "(1 + #beat_or_tie) / (1 + N)",
            "never_p_zero": True,
            "authoritative_increment": "FULL_ENGINE_NAV_DIFFERENCE",
        },
        "v03": {
            "model": "Ridge",
            "alpha": V03_RIDGE_ALPHA,
            "features": ["market_up_ratio", "B01A", "candidate_count", "index_RV20"],
            "no_new_equity_factors": True,
            "activity_match": "threshold from past training window only; no full-sample quantile pick",
            "mature_labels": True,
            "not_candidate": True,
        },
        "v04": {
            "model": "Ridge residual PUT_SKEW25 = f(X_equity)+eps",
            "alpha": RIDGE_ALPHA,
            "fit_window": V04_FIT_WINDOW,
            "fit_min": V04_FIT_MIN,
            "standardize": "train_only",
            "residual_percentile": {"window": PCT_WINDOW, "min": PCT_MIN_OBS, "q": SKEW_Q, "shift": 1},
            "no_threshold_search": True,
        },
        "quote": {
            "no_fabricate_bid_ask": True,
            "missing_label": BID_ASK_NOT_AVAILABLE,
            "iv_limitation": "last_close_only",
            "no_secrets_in_reports": True,
            "verdicts": [
                "QUOTE_QUALITY_SUPPORTED",
                "QUOTE_QUALITY_PARTIALLY_SUPPORTED",
                "LAST_PRICE_PROXY_ONLY",
                "QUOTE_QUALITY_FAIL",
            ],
            "rf_fallback": RF_FALLBACK,
            "rf_grid_if_no_history": list(RF_GRID),
        },
        "prediction": {
            "target": "Top2 loss (not top1_net)",
            "loss_net_lt": LOSS_NET_LT_0,
            "severe_loss_ret": SEVERE_LOSS_RET,
            "metrics": ["AUC", "Brier", "logloss", "calibration"],
            "models": ["equity_only", "equity_plus_skew"],
            "oos": "chronological year holdout",
        },
        "stats": {
            "block_days": BLOCK_DAYS,
            "bootstrap_n": BOOT_N,
            "cpcv_embargo": EMBARGO,
            "cpcv_min_block": MIN_BLOCK,
            "spa_boot": SPA_BOOT,
            "round16_spa_p_retained": R16_SPA_P,
            "cpcv_is_not_forward": True,
            "pairs": ["V01-V00", "V03-V00", "V04-V00", "V01-V03", "V04-V03", "V04-V01"],
            "windows": ["FULL_RESEARCH_WINDOW", "OPTION_VALID_SIGNAL_WINDOW", "ACTUALLY_TREATED_DAYS"],
            "cost_bps": list(COST_BPS),
        },
        "promotion_gates": dict(PROMOTION_GATES),
        "snapshot": {"id": SNAPSHOT_ID, "fingerprint": SNAPSHOT_FINGERPRINT},
        "forbidden": list(FORBIDDEN),
        "stages": list(STAGES),
        "experiments": [pre_registered_spec(c) for c in PRE_REGISTERED_IDS],
        "source_tree_sha256": source_tree_sha256(STRATEGY_SOURCE_FILES),
        "protocol_sha256": source_tree_sha256(PROTOCOL_SOURCE_FILES),
        "note": "冻结后禁止按 V03/V04 NAV 放宽门。CURRENT 不得自动升 v5。新想法只能 POST_HOC_DIAGNOSTIC。",
    }
    if facts:
        payload["facts"] = facts
    text = _dump_yaml(payload)
    assert_project_result_path(path)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    digest = sha256_file(path)
    sha256_path().write_text(digest + "\n", encoding="utf-8")
    dump_json(FREEZE_JSON, {"status": "FROZEN", "sha256": digest, "path": str(path), "gates": dict(PROMOTION_GATES)})
    import csv

    reg = ROUND16_5_ROOT / "experiment_registry.csv"
    assert_project_result_path(reg)
    with reg.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["code", "role", "parent", "is_candidate", "formula"])
        w.writeheader()
        for cid in PRE_REGISTERED_IDS:
            spec = pre_registered_spec(cid)
            w.writerow({
                "code": cid,
                "role": spec.get("role"),
                "parent": spec.get("parent"),
                "is_candidate": spec.get("is_candidate"),
                "formula": spec.get("formula") or CANDIDATE_FORMULAS.get(cid, ""),
            })
    write_md(
        FREEZE_MD,
        "\n".join([
            "# Round 16.5 freeze",
            "",
            f"sha256=`{digest}`",
            f"PRIOR_SELECTION_EXISTS=`{PRIOR_SELECTION_EXISTS}`",
            f"promotion gates locked **before** V03/V04 NAV.",
            "",
            "默认 CURRENT=`baseline_v4` CANDIDATE=`R16_O01` status=`PROMISING_BUT_UNCONFIRMED` LIVE=`NOT_VALIDATED`。",
            "",
        ]),
    )
    return {"status": "FROZEN", "sha256": digest, "path": str(path)}


def require_frozen() -> dict[str, Any]:
    path = manifest_path()
    if not path.is_file():
        raise RuntimeError("MANIFEST_NOT_FROZEN STOP")
    return {"path": str(path), "sha256": sha256_file(path)}


def load_freeze() -> dict[str, Any]:
    return load_json(FREEZE_JSON)
