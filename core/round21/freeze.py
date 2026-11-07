"""看任何新因子 NAV 前冻结 hypothese + manifest + sha256。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND21_ROOT, assert_project_result_path, ensure_round21_tree
from ..round11.contract import source_tree_sha256
from ..round11.freeze import _dump_yaml
from ..stock_readonly import sha256_file
from .audit import AUDIT_JSON
from .common import dump_json, load_json, write_md
from .contract import (
    AWAITING_USER_REVIEW,
    AUTO_PROMOTE_CURRENT,
    CLUSTER_SPEARMAN_ABS,
    COLSAMPLE,
    COMBO_BUDGET,
    CONTRACT_MD,
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    EXACT_GITHUB_REPRODUCTION_PASS,
    FEE_SCHEDULE,
    FORBIDDEN,
    GITHUB_EXACT,
    IC_RULES,
    INTERNAL_WINDOW_NAMES,
    KEEP_BASELINE_V5,
    LABELS,
    LATER_STAGES,
    MANIFEST_NAME,
    MATURITY_AWARE_LABEL,
    MATURITY_BY,
    MODELS,
    PARENT_V4,
    PHASE_A_STAGES,
    PRIOR_CONSTRAINTS,
    PROMOTION_GATES,
    PROTOCOL_SOURCE_FILES,
    R20_PRIOR,
    RECOMMEND_NEW_ALPHA_CANDIDATE,
    ROUND21_INTERNAL,
    SEED_LADDER,
    SELECTION_RULES,
    STAGES,
    SUBSAMPLE,
    V5_ANCHOR,
)
from .github import GH_JSON
from .hypotheses import HYP_JSON, require_hypotheses
from .parity import PARITY_JSON
from .read import READ_JSON

FREEZE_JSON = ROUND21_ROOT / "configs" / "freeze.json"


def manifest_path():
    return ROUND21_ROOT / MANIFEST_NAME


def sha256_path():
    return ROUND21_ROOT / f"{MANIFEST_NAME}.sha256"


def _candidate_nav_present() -> list[str]:
    root = ROUND21_ROOT / "candidates"
    if not root.is_dir():
        return []
    found = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".parquet", ".json"} and "nav" in path.name.lower():
            found.append(str(path))
    return found


def require_frozen() -> dict[str, Any]:
    frozen = load_json(FREEZE_JSON)
    if not frozen.get("frozen"):
        raise RuntimeError("MANIFEST_NOT_FROZEN STOP; refuse new-factor NAV")
    if not manifest_path().is_file():
        raise RuntimeError("MANIFEST_NOT_FROZEN STOP; missing yaml")
    require_hypotheses()
    return frozen


def freeze_manifest(*, overwrite: bool = False) -> dict[str, Any]:
    ensure_round21_tree()
    if FREEZE_JSON.is_file() and not overwrite:
        existing = load_json(FREEZE_JSON)
        if existing.get("frozen") and manifest_path().is_file() and sha256_path().is_file():
            return existing
    leaked = _candidate_nav_present()
    if leaked:
        raise RuntimeError(f"REFUSE_FREEZE_AFTER_CANDIDATE_NAV: {leaked[:8]}")
    from .hypotheses import freeze_hypotheses

    hyp = freeze_hypotheses(overwrite=False)
    parity = load_json(PARITY_JSON)
    audit = load_json(AUDIT_JSON)
    github = load_json(GH_JSON)
    read = load_json(READ_JSON)
    protocol = source_tree_sha256(PROTOCOL_SOURCE_FILES)
    body = {
        "manifest_id": "round21_manifest",
        "round": 21,
        "current": CURRENT_BASELINE,
        "current_alias": CURRENT_ALIAS,
        "parent": PARENT_V4,
        "user_lock": True,
        "auto_promote_current": AUTO_PROMOTE_CURRENT,
        "default_promotion": KEEP_BASELINE_V5,
        "recommend_label": RECOMMEND_NEW_ALPHA_CANDIDATE,
        "review_flag": AWAITING_USER_REVIEW,
        "do_not_lock_v6": True,
        "subsample": SUBSAMPLE,
        "colsample": COLSAMPLE,
        "fee_schedule": FEE_SCHEDULE,
        "historical_stamp_tax": False,
        "validation_hooks": None,
        "cagr_daycount": 365.25,
        "v5_anchor": dict(V5_ANCHOR),
        "parity_status": parity.get("status"),
        "parity_passed": bool(parity.get("passed")),
        "formal_promotion_blocked": bool(not parity.get("passed")),
        "github_status": github.get("GITHUB_METHOD_STATUS") or github.get("status"),
        "github_sha": github.get("commit_sha"),
        "exact_github_reproduction_pass": False,
        "forbidden_claim": EXACT_GITHUB_REPRODUCTION_PASS,
        "github_namespace": GITHUB_EXACT,
        "internal_namespace": ROUND21_INTERNAL,
        "internal_windows": list(INTERNAL_WINDOW_NAMES),
        "do_not_pick_window_ex_post": True,
        "maturity_aware_label": MATURITY_AWARE_LABEL,
        "maturity_by": MATURITY_BY,
        "hypotheses_n": hyp.get("n"),
        "hypotheses_ids": list(hyp.get("factor_ids") or []),
        "a1_is_not_r18_f1": True,
        "near_neighbors": {
            "A4": "fb_close_seal_streak_norm",
            "B1": "r17_d_amt_xs_rank",
            "C3": "h_min1_path_eff_full",
            "A1_ne_R18_F1": True,
        },
        "source_audit_counts": (audit.get("counts") or {}),
        "r20_prior": dict(R20_PRIOR),
        "ic_rules": dict(IC_RULES),
        "labels": list(LABELS),
        "cluster_spearman_abs": CLUSTER_SPEARMAN_ABS,
        "cluster_must_run": ["Original", "New", "Add", "Replace"],
        "selection_rules": dict(SELECTION_RULES),
        "combo_budget": dict(COMBO_BUDGET),
        "models": dict(MODELS),
        "seed_ladder": dict(SEED_LADDER),
        "promotion_gates": dict(PROMOTION_GATES),
        "prior_constraints": dict(PRIOR_CONSTRAINTS),
        "forbidden": list(FORBIDDEN),
        "stages_phase_a": list(PHASE_A_STAGES),
        "stages_later": list(LATER_STAGES),
        "stages": list(STAGES),
        "protocol_sha256": protocol,
        "prior_artifact_hashes": {
            rec.get("key"): rec.get("sha256") for rec in (read.get("files") or []) if rec.get("key") and rec.get("sha256")
        },
        "note": (
            "冻结发生在任何新因子 NAV 之前。事后不放宽。"
            "Parity FAIL 只停正式晋级评价，不阻止本冻结。"
            "未构造因子值。"
        ),
    }
    path = manifest_path()
    assert_project_result_path(path)
    text = _dump_yaml(body)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")
    digest = sha256_file(path)
    sha256_path().write_text(digest + "\n", encoding="utf-8")
    payload = {
        "frozen": True,
        "status": "FROZEN",
        "sha256": digest,
        "path": str(path),
        "sha256_path": str(sha256_path()),
        "parity_status": parity.get("status"),
        "formal_promotion_blocked": bool(not parity.get("passed")),
        "github_status": github.get("GITHUB_METHOD_STATUS") or github.get("status"),
        "github_sha": github.get("commit_sha"),
        "hypotheses_n": hyp.get("n"),
        "subsample": SUBSAMPLE,
        "colsample": COLSAMPLE,
        "current": CURRENT_BASELINE,
        "auto_promote_current": AUTO_PROMOTE_CURRENT,
        "values_constructed": False,
    }
    dump_json(FREEZE_JSON, payload)
    lines = [
        "# Round21 · 00 研究契约（看新因子 NAV 前冻结）",
        "",
        "本文件与 `round21_manifest.yaml` 在任何**新因子** NAV 之前冻结。事后不放宽门槛。",
        "",
        "## sha256",
        "",
        f"- `{path}`",
        f"- sha256=`{digest}`",
        f"- sidecar=`{sha256_path()}`",
        "",
        "## 锁",
        "",
        f"- CURRENT=`{CURRENT_BASELINE}` = `{CURRENT_ALIAS}` USER_LOCK。AUTO_PROMOTE=`{AUTO_PROMOTE_CURRENT}`。",
        f"- 默认 `{KEEP_BASELINE_V5}`。过门只写 `{RECOMMEND_NEW_ALPHA_CANDIDATE}` + `{AWAITING_USER_REVIEW}`。不锁 v6。",
        f"- subsample=`{SUBSAMPLE}` colsample=`{COLSAMPLE}`。",
        f"- 主回测 `{FEE_SCHEDULE}`。parity=`{parity.get('status')}` formal_promotion_blocked=`{not parity.get('passed')}`",
        f"- GitHub status=`{github.get('GITHUB_METHOD_STATUS') or github.get('status')}` SHA=`{github.get('commit_sha')}` exact_pass=False",
        f"- hypotheses N=`{hyp.get('n')}`",
        "",
        "## 预注册",
        "",
        f"- 内部滚动窗 `{list(INTERNAL_WINDOW_NAMES)}`，不事后挑窗。maturity-aware `{MATURITY_AWARE_LABEL}` by `{MATURITY_BY}`。",
        f"- Spearman≥{CLUSTER_SPEARMAN_ABS} 为簇边界，非唯一 DROP；簇内必须 Original/New/Add/Replace。",
        f"- 入主组合新因子 ≤{SELECTION_RULES['max_new_factors_in_main_combo_research']}。",
        f"- 预算 Pair≤{COMBO_BUDGET['pair']} Triple≤{COMBO_BUDGET['triple']} ortho/PCA/非线性≤{COMBO_BUDGET['ortho_pca_nonlinear']}；超预算开下一批次。",
        "- 同时报 FIXED_FACTOR_SET_REPLAY 与 NESTED_WALK_FORWARD_FEATURE_SELECTION；二者都不是独立 OOS。",
        "- 内部 Evidence 向量/分层硬过滤+Pareto，不优化人为加权总分；GitHub 无 Score 不伪造。",
        "- Seed：仅当预注册预测增量门槛过了才跑；native 5 退化则 group-date；不从 100/300 里挑最好 seed。",
        "",
        f"R20 PRIOR：{R20_PRIOR}。",
        "",
    ]
    write_md(CONTRACT_MD, "\n".join(lines))
    return payload
