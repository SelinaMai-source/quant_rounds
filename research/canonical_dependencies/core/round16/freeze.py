"""冻结 round16_manifest_v1.yaml。必须在任何候选 NAV 之前。之后禁止按收益改阈值。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..paths import ROUND16_ROOT, assert_project_result_path, ensure_round16_tree
from ..round11.contract import source_tree_sha256
from ..round11.freeze import _dump_yaml
from ..stock_readonly import sha256_file
from .audit import load_audit
from .common import dump_json, load_json, write_md
from .contract import (
    AUTO_PROMOTE_V5,
    CANDIDATE_FORMULAS,
    COV_MIN_DAYS,
    COV_MIN_SIGNAL,
    COV_MIN_YEARS,
    ETF_UNDERLYINGS,
    FORBIDDEN,
    KEEP_BASELINE_V4,
    MANIFEST_NAME,
    O01,
    O02,
    O03,
    O04,
    OPTION_IDS,
    PCT_MIN_OBS,
    PCT_WINDOW,
    PRE_REGISTERED_IDS,
    PROTOCOL_SOURCE_FILES,
    RANK_NAV,
    SKIP_ADVANCED_FACTORS,
    SKIP_FACTOR_FOR_TRADING,
    SNAPSHOT_FINGERPRINT,
    SNAPSHOT_ID,
    STAGES,
    STRATEGY_SOURCE_FILES,
    TOP_K,
    V4_ANCHOR,
    pre_registered_spec,
)


def manifest_path():
    return ROUND16_ROOT / MANIFEST_NAME


def sha256_path():
    return ROUND16_ROOT / f"{MANIFEST_NAME}.sha256"


def _skips_from_coverage(cov: dict[str, Any], fac: dict[str, Any]) -> dict[str, Any]:
    gates = (fac.get("gates") or {})
    f1 = bool(gates.get("F1_pass"))
    f2 = bool(gates.get("F2_pass"))
    f3 = bool(gates.get("F3_pass"))
    f4 = bool(gates.get("F4_pass"))
    advanced = bool(gates.get("advanced_pass"))
    skip_trading = not any([f1, f2, f3, f4])
    return {
        "SKIP_IV_SURFACE": not bool(int(fac.get("n_factor_days") or 0)),
        "SKIP_PUT_SKEW25": not f2,
        "SKIP_F3": not f3,
        "SKIP_ADVANCED_FACTORS": not advanced,
        "SKIP_FACTOR_FOR_TRADING": skip_trading,
        "SKIP_O01": not f2,
        "SKIP_O02": not f1,
        "SKIP_O03": (not f2) or (not f3),
        "SKIP_O04": not (f1 or f2),
        "F1": "PASS" if f1 else SKIP_FACTOR_FOR_TRADING,
        "F2": "PASS" if f2 else SKIP_FACTOR_FOR_TRADING,
        "F3": "PASS" if f3 else ("SKIP_F3" if not gates.get("F3_has_10d") else SKIP_FACTOR_FOR_TRADING),
        "F4": "PASS" if f4 else SKIP_FACTOR_FOR_TRADING,
        "advanced": "PASS" if advanced else SKIP_ADVANCED_FACTORS,
    }


def freeze_manifest(*, overwrite: bool = False, facts: dict[str, Any] | None = None) -> dict[str, Any]:
    ensure_round16_tree()
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
    from .coverage import load_coverage
    from .factors import load_factors
    from .iv import load_iv_status
    from .quality import load_quality
    from .parity import load_parity_status

    audit = load_audit()
    cov = load_coverage()
    fac = load_factors()
    quality = load_quality()
    iv = load_iv_status()
    parity = load_parity_status()
    skips = _skips_from_coverage(cov, fac)
    runnable = {
        O01: not skips["SKIP_O01"],
        O02: not skips["SKIP_O02"],
        O03: not skips["SKIP_O03"],
        O04: not skips["SKIP_O04"],
    }
    payload: dict[str, Any] = {
        "manifest_id": "round16_manifest_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "mission": "Option data recovery & pre-registered option-implied alpha. Default KEEP_BASELINE_V4.",
        "auto_promote_v5": AUTO_PROMOTE_V5,
        "default_promotion": KEEP_BASELINE_V4,
        "status": ["historical_development", "NOT_final_OOS", "NOT_production_validated"],
        "engine": {
            "formula": "target_name_budget = entry_NAV × RANK_NAV[rank] × allocation_multiplier",
            "rank_nav": list(RANK_NAV),
            "top_k": TOP_K,
            "zero_top2": "MIN_KEEP",
            "no_top3_fill": True,
            "backtest": "run_r15_backtest + HISTORICAL_FEE_CORRECTED + B03",
        },
        "parity": {k: parity.get(k) for k in ("status", "passed")},
        "root_cause": {
            "verdict": audit.get("verdict"),
            "history_status": audit.get("history_status"),
            "history_available": audit.get("history_available"),
            "why_63_days": audit.get("why_63_days"),
        },
        "coverage": {
            "n_valid_days": cov.get("n_valid_days"),
            "n_contracts": cov.get("n_contracts"),
            "n_rows": cov.get("n_quote_rows"),
            "year_span": cov.get("year_span"),
            "underlyings": list(ETF_UNDERLYINGS),
        },
        "quality": {k: quality.get(k) for k in ("status", "valid_quote_rate", "invalid_quote_rate", "stale_quote_rate") if quality},
        "iv": {k: iv.get(k) for k in ("status", "n_iv", "unit_tests") if iv},
        "gates": {
            "min_days": COV_MIN_DAYS,
            "min_years": COV_MIN_YEARS,
            "min_signal_after_warmup": COV_MIN_SIGNAL,
            "percentile_window": PCT_WINDOW,
            "percentile_min": PCT_MIN_OBS,
            "factor_gates": (fac.get("gates") or {}),
        },
        "skips": skips,
        "runnable_at_freeze": runnable,
        "factors_locked": {
            "F1": "ATM_IV30; nearest-maturity proxy if single tenor, labeled separately",
            "F2": "PUT_SKEW25 = IV_25D_PUT - IV_25D_CALL (both valid)",
            "F3": "PUT_WING_CONVEXITY = IV_10D_PUT - 2*IV_25D_PUT + ATM_IV; else SKIP_F3",
            "F4": "IV_RV_SPREAD = ATM_IV30 - RV20 (RV uses D and before only)",
            "percentile": "past 252 valid option days, min 126; else fallback v4",
            "pit": "Asia/Shanghai; available_at <= D close; no future quotes/contracts/full-sample scalers",
        },
        "candidates": {k: CANDIDATE_FORMULAS[k] for k in OPTION_IDS},
        "experiments": [pre_registered_spec(c) for c in PRE_REGISTERED_IDS],
        "v4_anchor": dict(V4_ANCHOR),
        "snapshot": {"id": SNAPSHOT_ID, "fingerprint": SNAPSHOT_FINGERPRINT},
        "forbidden": list(FORBIDDEN),
        "stages": list(STAGES),
        "source_tree_sha256": source_tree_sha256(STRATEGY_SOURCE_FILES),
        "protocol_sha256": source_tree_sha256(PROTOCOL_SOURCE_FILES),
        "note": "冻结后禁止按候选 NAV 改阈值/公式。CURRENT 不得自动升 v5。",
    }
    if facts:
        payload["facts"] = facts
    text = _dump_yaml(payload)
    assert_project_result_path(path)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    digest = sha256_file(path)
    sha256_path().write_text(digest + "\n", encoding="utf-8")
    dump_json(ROUND16_ROOT / "configs" / "freeze.json", {"status": "FROZEN", "sha256": digest, "path": str(path), "skips": skips, "runnable": runnable})
    import csv
    reg = ROUND16_ROOT / "experiment_registry.csv"
    assert_project_result_path(reg)
    with reg.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["code", "role", "parent", "formula", "runnable"])
        w.writeheader()
        for cid in PRE_REGISTERED_IDS:
            spec = pre_registered_spec(cid)
            w.writerow({
                "code": cid,
                "role": spec.get("role"),
                "parent": spec.get("parent"),
                "formula": spec.get("formula") or CANDIDATE_FORMULAS.get(cid, ""),
                "runnable": runnable.get(cid, cid.endswith("V4")),
            })
    write_md(
        ROUND16_ROOT / "03_freeze.md",
        f"# Round16 freeze\n\nsha256=`{digest}`\nskips={skips}\nrunnable={runnable}\n",
    )
    return {"status": "FROZEN", "sha256": digest, "path": str(path), "skips": skips, "runnable": runnable}


def load_skips() -> dict[str, Any]:
    cfg = load_json(ROUND16_ROOT / "configs" / "freeze.json")
    return dict(cfg.get("skips") or {})


def load_runnable() -> dict[str, Any]:
    cfg = load_json(ROUND16_ROOT / "configs" / "freeze.json")
    return dict(cfg.get("runnable") or {})


def require_frozen() -> dict[str, Any]:
    path = manifest_path()
    if not path.is_file():
        raise RuntimeError("MANIFEST_NOT_FROZEN STOP")
    return {"path": str(path), "sha256": sha256_file(path)}
