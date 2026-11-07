"""冻结 round15_manifest_v1.yaml + SKIP/升级门。必须在任何新策略 NAV 之前。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..paths import ROUND15_ROOT, assert_project_result_path, ensure_round15_tree
from ..round11.contract import artifact_file_hashes, source_tree_sha256
from ..round11.freeze import _dump_yaml
from ..stock_readonly import sha256_file
from .contract import (
    AUTO_PROMOTE_V5,
    CANDIDATE_FORMULAS,
    CANDIDATE_IDS,
    EXEC_IDS,
    FORBIDDEN_SEARCH,
    KEEP_BASELINE_V4,
    MANIFEST_NAME,
    MINUTE_YEARS,
    OPTION_IDS,
    PRE_REGISTERED_IDS,
    PRE_REGISTERED_QUESTIONS,
    PROMOTION_MAX_CAGR_DROP_PP,
    PROMOTION_MDD_RETURN_SLACK_PP,
    PROMOTION_MDD_SLACK_PP,
    PROMOTION_SHARPE_SLACK,
    PROTOCOL_SOURCE_FILES,
    R12_01_DIR,
    R12_01_TARGETS,
    R13_00_DIR,
    R14_00_DIR,
    RANK_NAV,
    SNAPSHOT_FINGERPRINT,
    SNAPSHOT_ID,
    STAGES,
    STRATEGY_SOURCE_FILES,
    T01_MIN_N,
    T02_MIN_N,
    T02_MIN_OOS_R2,
    TOP1_IDS,
    TOP_K,
    pre_registered_spec,
)
from .inventory import load_inventory, require_inventory
from .sdk import load_sdk, require_sdk


def manifest_path():
    return ROUND15_ROOT / MANIFEST_NAME


def sha256_path():
    return ROUND15_ROOT / f"{MANIFEST_NAME}.sha256"


def _coverage_and_skips() -> dict[str, Any]:
    inv = load_inventory()
    sdk = load_sdk()
    minute = inv.get("minute") or {}
    opts_local = inv.get("options_local") or {}
    from .sdk import audit_option_coverage

    cov_opt = audit_option_coverage()
    option_ok = bool(sdk.get("option_available"))
    has_0925 = bool(minute.get("has_0925_auction"))
    common_years = list(minute.get("common_minute_years") or [])
    minute_ok = all(int(y) in {int(x) for x in common_years} for y in MINUTE_YEARS)
    sample = sdk.get("sample") or {}
    n_codes = int(sample.get("n_option_codes") or 0)
    stable_chain = bool(cov_opt.get("stable_chain"))
    bilateral_25d = bool(cov_opt.get("bilateral_25d"))
    has_10d_put = bool(cov_opt.get("has_10d_put"))
    chain_dense = bool(cov_opt.get("chain_dense"))
    skips = {
        "SKIP_IV_SURFACE": (not stable_chain),
        "SKIP_PUT_SKEW25": (not stable_chain) or (not bilateral_25d),
        "SKIP_F3": (not stable_chain) or (not has_10d_put),
        "SKIP_ADVANCED_OPTION_FACTORS": (not stable_chain) or (not chain_dense),
        "SKIP_AUCTION_SIGNAL": (not has_0925),
        "SKIP_O01": (not stable_chain) or (not bilateral_25d),
        "SKIP_O02": (not stable_chain),
        "SKIP_O03": (not stable_chain) or (not bilateral_25d) or (not has_10d_put),
        "SKIP_O04": (not stable_chain),
        "SKIP_T01": False,  # 规则：样本<T01_MIN_N 则 SKIP；归因后执行
        "SKIP_T02": False,  # 规则：OOF R2 < T02_MIN_OOS_R2 或 n<T02_MIN_N 则 SKIP
        "SKIP_X01": (not minute_ok),
        "SKIP_X02": (not minute_ok),
    }
    runnable = {
        "O01": not skips["SKIP_O01"],
        "O02": not skips["SKIP_O02"],
        "O03": not skips["SKIP_O03"],
        "O04": not skips["SKIP_O04"],
        "T01": True,
        "T02": True,
        "X01": not skips["SKIP_X01"],
        "X02": not skips["SKIP_X02"],
    }
    return {
        "option_available": option_ok,
        "option_coverage_audit": cov_opt,
        "n_option_codes": n_codes,
        "local_option_hits": opts_local.get("n_hits"),
        "stable_chain": stable_chain,
        "bilateral_25d_known_at_freeze": bilateral_25d,
        "has_10d_put_known_at_freeze": has_10d_put,
        "chain_dense_known_at_freeze": chain_dense,
        "has_0925": has_0925,
        "first_bar": minute.get("first_bar"),
        "common_minute_years": common_years,
        "minute_ok_for_x": minute_ok,
        "sdk_status": sdk.get("status"),
        "sdk_skip_reason": sdk.get("skip_reason"),
        "skips": skips,
        "runnable": runnable,
        "t01_rule": f"SKIP_T01 if n<{T01_MIN_N}",
        "t02_rule": f"SKIP_T02 if OOF_R2<{T02_MIN_OOS_R2} or n<{T02_MIN_N}",
        "note": (
            "无 IV 面时不得把 PC ratio 叫 RN kurtosis。"
            "bilateral 25D / 10D put 在 freeze 时未知则保守 SKIP，因子核验后不得改名替补。"
        ),
    }


def build_manifest_payload(*, frozen_at: str | None = None) -> dict[str, Any]:
    strategy = source_tree_sha256(STRATEGY_SOURCE_FILES)
    protocol = source_tree_sha256(PROTOCOL_SOURCE_FILES)
    cov = _coverage_and_skips()
    return {
        "manifest_id": "round15_manifest_v1",
        "frozen_at": frozen_at or datetime.now().astimezone().isoformat(),
        "mission": "Option Tail × Top1 × Execution. Limited pre-registered alpha. Default KEEP_BASELINE_V4.",
        "rule_zh": (
            "parity → inventory+SDK → freeze SKIP/门 → 才允许新策略 NAV。"
            "数据不够 SKIP 不 PASS。禁止搜 Top2 0/5/10、波动/MRC 帽、v3/v4 平均。"
            "不得自动升 CURRENT v5。过门则 RECOMMENDED_BASELINE_V5 + AWAITING_USER_LOCK。"
            "2018–2026 是 HISTORICAL DEVELOPMENT DATA。"
        ),
        "auto_promote_v5": AUTO_PROMOTE_V5,
        "default_promotion": KEEP_BASELINE_V4,
        "status": ["historical_development", "NOT_final_OOS", "NOT_production_validated"],
        "engine": {
            "formula": "target_name_budget = entry_NAV × RANK_NAV[rank] × allocation_multiplier",
            "rank_nav": list(RANK_NAV),
            "top_k": TOP_K,
            "zero_top2": "MIN_KEEP",
            "no_top3_fill": True,
            "backtest": "run_r12_backtest + HISTORICAL_FEE_CORRECTED",
        },
        "coverage": cov,
        "skips": cov["skips"],
        "runnable_at_freeze": cov["runnable"],
        "factors_locked": {
            "O_F1": "ATM_IV30; IV_RV_SPREAD=ATM_IV30-lagged RV (RV only D-known)",
            "O_F2": "PUT_SKEW25=IV_25D_PUT-IV_25D_CALL (both valid)",
            "O_F3": "PUT_WING_CONVEXITY=IV_10D_PUT-2*IV_25D_PUT+ATM_IV; else SKIP_F3",
            "O_F4": "PC volume/OI ratio and daily change; proxy, not bearish",
            "percentile": "past 252 valid option days, min 126; else fallback v4",
            "pit": "option timestamp <= D close; record available_at; fail=STOP_OPTION_BACKTEST",
        },
        "candidates": {k: v for k, v in CANDIDATE_FORMULAS.items()},
        "families": {"option": list(OPTION_IDS), "top1": list(TOP1_IDS), "execution_not_alpha": list(EXEC_IDS)},
        "controls": {
            "matched_window": "same days, same coverage, same cost vs v4",
            "O01_O03": "same count of Top2 veto days",
            "O02": "constant average risk + year/regime random timing N>=500; NOT daily exposure-match",
            "permutation_n": 500,
        },
        "promotion_gates": {
            "RETURN": {
                "pit": "PASS",
                "cagr_gt_v4": True,
                "sharpe_ge_v4": True,
                "mdd_worse_le_pp": PROMOTION_MDD_RETURN_SLACK_PP,
                "not_de_risking_only": "P(beat control)>=0.60 OR unit-exposure PnL AND timing control both support",
                "cpcv_median_sharpe_ge_v4": True,
                "family_spa_not_winner_bias": True,
                "plus20_plus30_not_collapse": True,
                "worst12m_no_unexplained_worsening": True,
            },
            "STABILITY": {
                "cagr_ge_v4_minus_pp": PROMOTION_MAX_CAGR_DROP_PP,
                "sharpe_ge_v4_minus": PROMOTION_SHARPE_SLACK,
                "mdd_worse_le_pp": PROMOTION_MDD_SLACK_PP,
                "structural_improve_n": 3,
                "structural_include": ["cpcv_p10", "bootstrap_stability", "control_test"],
            },
            "EXECUTABILITY": "不得仅因 X01/X02 好看而升。Alpha 须在共同分钟窗口仍满足 RETURN 或 STABILITY。",
            "pass_status": "RECOMMENDED_BASELINE_V5=<id> + AWAITING_USER_LOCK + HISTORICAL_DEVELOPMENT_ONLY",
            "no_true_forward": "不得标记 FORWARD_VALIDATION_SUPPORTED / LIVE_VALIDATED",
            "else": KEEP_BASELINE_V4,
        },
        "snapshot": {"id": SNAPSHOT_ID, "fingerprint": SNAPSHOT_FINGERPRINT},
        "pre_registered_ids": list(PRE_REGISTERED_IDS),
        "candidate_ids": list(CANDIDATE_IDS),
        "experiments": [pre_registered_spec(code) for code in PRE_REGISTERED_IDS],
        "r12_01_targets": dict(R12_01_TARGETS),
        "anchor_sha256": {
            "R12_01": artifact_file_hashes(R12_01_DIR) if R12_01_DIR.is_dir() else {},
            "R13_00": artifact_file_hashes(R13_00_DIR) if R13_00_DIR.is_dir() else {},
            "R14_00": artifact_file_hashes(R14_00_DIR) if R14_00_DIR.is_dir() else {},
        },
        "source_tree_sha256": strategy,
        "protocol_sha256": protocol,
        "questions": dict(PRE_REGISTERED_QUESTIONS),
        "stages": list(STAGES),
        "forbidden": list(FORBIDDEN_SEARCH),
        "git_commit": None,
        "git_commit_note": "无可用 git commit；source_tree_sha256 是文件哈希，不伪造 commit",
    }


def freeze_manifest(*, overwrite: bool = False) -> dict[str, Any]:
    require_inventory()
    require_sdk()
    ensure_round15_tree()
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
        skips = load_json_safe()
        if not skips:
            skips = _coverage_and_skips().get("skips") or {}
            from .common import dump_json

            dump_json(ROUND15_ROOT / "inventory" / "skips_frozen.json", skips)
        return {
            "status": "already_frozen",
            "path": str(path),
            "sha256": digest,
            "sha256_path": str(digest_path),
            "source_tree_sha256": source_tree_sha256(STRATEGY_SOURCE_FILES)["tree"],
            "skips": skips,
        }
    payload = build_manifest_payload()
    path.write_text(_dump_yaml(payload), encoding="utf-8")
    digest = sha256_file(path)
    digest_path.write_text(digest + "  " + MANIFEST_NAME + "\n", encoding="utf-8")
    from .common import dump_json, write_md

    dump_json(ROUND15_ROOT / "inventory" / "skips_frozen.json", payload["skips"])
    write_md(
        ROUND15_ROOT / "03_freeze.md",
        "\n".join([
            "# Round 15 · Freeze",
            "",
            f"sha256=`{digest}`。在任何新策略 NAV 之前。",
            "",
            f"skips={payload['skips']}",
            "",
            f"runnable={payload['runnable_at_freeze']}",
            "",
        ]),
    )
    return {
        "status": "frozen",
        "path": str(path),
        "sha256": digest,
        "sha256_path": str(digest_path),
        "frozen_at": payload["frozen_at"],
        "source_tree_sha256": payload["source_tree_sha256"]["tree"],
        "skips": payload["skips"],
        "runnable": payload["runnable_at_freeze"],
    }


def load_json_safe() -> dict[str, Any]:
    from .common import load_json

    return load_json(ROUND15_ROOT / "inventory" / "skips_frozen.json")


def require_frozen() -> dict[str, Any]:
    path = manifest_path()
    digest_path = sha256_path()
    if not path.is_file() or not digest_path.is_file():
        raise RuntimeError("MANIFEST_NOT_FROZEN STOP — 必须先盘点+SDK 再 freeze，再看新策略成绩")
    digest = sha256_file(path)
    sidecar = digest_path.read_text(encoding="utf-8").strip().split()[0]
    if digest != sidecar:
        raise RuntimeError(f"manifest sha256 mismatch: file={digest} sidecar={sidecar}")
    return {"path": str(path), "sha256": digest}


def load_skips() -> dict[str, bool]:
    from .common import load_json

    frozen = load_json(ROUND15_ROOT / "inventory" / "skips_frozen.json")
    if frozen:
        return {k: bool(v) for k, v in frozen.items()}
    return {}


__all__ = [
    "build_manifest_payload",
    "freeze_manifest",
    "load_skips",
    "manifest_path",
    "require_frozen",
    "sha256_path",
]
