"""看新候选 NAV 前冻结 IC/筛选/预算/seed/晋级门，并 sha256。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND20_ROOT, assert_project_result_path, ensure_round20_tree
from ..round11.contract import source_tree_sha256
from ..round11.freeze import _dump_yaml
from ..stock_readonly import sha256_file
from .common import dump_json, load_json, write_md
from .contract import (
    AWAITING_USER_REVIEW,
    AUTO_PROMOTE_CURRENT,
    BLOCK_DAYS,
    BOOT_N,
    CLUSTER_SPEARMAN_ABS,
    COLSAMPLE,
    COMBO_BUDGET,
    CONTRACT_MD,
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    FEE_SCHEDULE,
    FORBIDDEN,
    GROUP_DATE_SPEC,
    IC_MIN_SOURCE,
    IC_MIN_VALID_NAMES_PER_DAY,
    IC_RULES,
    KEEP_BASELINE_V5,
    LABELS,
    LATER_STAGES,
    MANIFEST_NAME,
    MODELS,
    NATIVE_SEED_DIVERSITY_AUDIT,
    NOT_TRIGGERED,
    PAIRED_POS_RATIO_MIN,
    PARENT_V4,
    PCA_EXPLAINED_VARIANCE,
    PHASE_A_STAGES,
    PRIOR_CONSTRAINTS,
    PRIOR_RESEARCH_SELECTION_EXISTS,
    PROMOTION_GATES,
    PROTOCOL_SOURCE_FILES,
    RECOMMEND_NEW_ALPHA_CANDIDATE,
    SEED_LADDER,
    SELECTION_RULES,
    SKIP_DEPENDENCY,
    SPA_BOOT,
    STAGES,
    SUBSAMPLE,
    V5_ANCHOR,
)
from .inventory import INV_JSON
from .parity import PARITY_JSON
from .read import READ_JSON

FREEZE_JSON = ROUND20_ROOT / "configs" / "freeze.json"


def manifest_path():
    return ROUND20_ROOT / MANIFEST_NAME


def sha256_path():
    return ROUND20_ROOT / f"{MANIFEST_NAME}.sha256"


def _candidate_nav_present() -> list[str]:
    root = ROUND20_ROOT / "candidates"
    if not root.is_dir():
        return []
    found = []
    for path in root.rglob("*"):
        if path.suffix in {".parquet", ".json"} and path.is_file():
            found.append(str(path))
    return found


def require_frozen() -> dict[str, Any]:
    frozen = load_json(FREEZE_JSON)
    if not frozen.get("frozen"):
        raise RuntimeError("MANIFEST_NOT_FROZEN STOP; refuse new-candidate NAV")
    if not manifest_path().is_file():
        raise RuntimeError("MANIFEST_NOT_FROZEN STOP; missing yaml")
    return frozen


def freeze_manifest(*, overwrite: bool = False) -> dict[str, Any]:
    ensure_round20_tree()
    if FREEZE_JSON.is_file() and not overwrite:
        existing = load_json(FREEZE_JSON)
        if existing.get("frozen") and manifest_path().is_file() and sha256_path().is_file():
            return existing
    leaked = _candidate_nav_present()
    if leaked:
        raise RuntimeError(f"REFUSE_FREEZE_AFTER_CANDIDATE_NAV: {leaked[:8]}")
    parity = load_json(PARITY_JSON)
    inventory = load_json(INV_JSON)
    read = load_json(READ_JSON)
    protocol = source_tree_sha256(PROTOCOL_SOURCE_FILES)
    body = {
        "manifest_id": "round20_manifest",
        "round": 20,
        "current": CURRENT_BASELINE,
        "current_alias": CURRENT_ALIAS,
        "parent": PARENT_V4,
        "user_lock": True,
        "auto_promote_current": AUTO_PROMOTE_CURRENT,
        "default_promotion": KEEP_BASELINE_V5,
        "recommend_label": RECOMMEND_NEW_ALPHA_CANDIDATE,
        "review_flag": AWAITING_USER_REVIEW,
        "do_not_lock_v6": True,
        "prior_research_selection_exists": PRIOR_RESEARCH_SELECTION_EXISTS,
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
        "inventory_n_total": inventory.get("n_total"),
        "inventory_n_r17": inventory.get("n_r17_registry"),
        "inventory_n_r18": inventory.get("n_r18_f123"),
        "f2_not_promotion_eligible": True,
        "f1_f3_not_rebranded": True,
        "ic_rules": dict(IC_RULES),
        "ic_min_valid_names_per_day": IC_MIN_VALID_NAMES_PER_DAY,
        "ic_min_source": IC_MIN_SOURCE,
        "labels": list(LABELS),
        "cluster_spearman_abs": CLUSTER_SPEARMAN_ABS,
        "selection_rules": dict(SELECTION_RULES),
        "combo_budget": dict(COMBO_BUDGET),
        "pca_explained_variance_ratio": PCA_EXPLAINED_VARIANCE,
        "models": dict(MODELS),
        "m2_unavailable_label": SKIP_DEPENDENCY,
        "seed_ladder": dict(SEED_LADDER),
        "native_seed_audit": NATIVE_SEED_DIVERSITY_AUDIT,
        "group_date": dict(GROUP_DATE_SPEC),
        "untriggered_s4_label": NOT_TRIGGERED,
        "paired_pos_ratio_min": PAIRED_POS_RATIO_MIN,
        "paired_pos_ratio_is_not_oos_p": True,
        "block_days": BLOCK_DAYS,
        "bootstrap_n": BOOT_N,
        "spa_boot": SPA_BOOT,
        "promotion_gates": dict(PROMOTION_GATES),
        "prior_constraints": dict(PRIOR_CONSTRAINTS),
        "forbidden": list(FORBIDDEN),
        "stages_phase_a": list(PHASE_A_STAGES),
        "stages_later": list(LATER_STAGES),
        "stages": list(STAGES),
        "protocol_sha256": protocol,
        "prior_artifact_hashes": {
            rec.get("key"): rec.get("sha256") for rec in (read.get("files") or []) if rec.get("key")
        },
        "note": (
            "冻结发生在任何新候选 Alpha NAV 之前。事后不放宽。"
            "Parity FAIL 只停正式晋级评价，不阻止本冻结。"
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
        "subsample": SUBSAMPLE,
        "colsample": COLSAMPLE,
        "current": CURRENT_BASELINE,
        "auto_promote_current": AUTO_PROMOTE_CURRENT,
        "inventory_n_total": inventory.get("n_total"),
    }
    dump_json(FREEZE_JSON, payload)
    lines = [
        "# Round20 · 00 研究契约（看新候选 NAV 前冻结）",
        "",
        "本文件与 `round20_manifest.yaml` 在任何**新候选** Alpha NAV 之前冻结。事后不放宽门槛。",
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
        f"- subsample=`{SUBSAMPLE}` colsample=`{COLSAMPLE}`。禁止改 O01 / B03 九格 / 90-10 / clean_risk / 退出 / 仓位 / gap。",
        f"- 主回测 `{FEE_SCHEDULE}`，不传 validation_hooks。费用校正只作 HISTORICAL_FEE_CORRECTED_SENSITIVITY。",
        f"- parity=`{parity.get('status')}` formal_promotion_blocked=`{not parity.get('passed')}`",
        "",
        "## 预注册 IC / 筛选 / 预算",
        "",
        f"- 日截面最少有效股票数=`{IC_MIN_VALID_NAMES_PER_DAY}`（来源：{IC_MIN_SOURCE}）。不足则跳过该日，**不把 IC 记 0**。",
        f"- 双标签 `{list(LABELS)}` 分开算 IC / RankIC / HAC / 块 bootstrap。",
        f"- 高相关簇初始 `abs(Spearman)≥{CLUSTER_SPEARMAN_ABS}`，不是唯一 DROP 依据。",
        f"- 入主组合研究的新因子 ≤{SELECTION_RULES['max_new_factors_in_main_combo_research']}；质量不够就少用，不凑数。",
        f"- 预算上限：单+{COMBO_BUDGET['single']} / 双+{COMBO_BUDGET['pair']} / 三+{COMBO_BUDGET['triple']} / 跨家族+{COMBO_BUDGET['cross_family']} / pruned-ortho-PCA+{COMBO_BUDGET['pruned_ortho_pca']} ≤ **{COMBO_BUDGET['max_total']}**；LOFO 另计。",
        "- 组合 registry 在看正式组合 NAV **之前**生成；禁止事后 Best×Best。",
        f"- PCA：家族内、`explained_variance_ratio={PCA_EXPLAINED_VARIANCE}`，不搜阈值。正交按 fold 拟合，禁止全历史 PCA/正交。",
        "",
        "## 模型 / seed",
        "",
        f"- M0=`{MODELS['M0']}`",
        f"- M1=`{MODELS['M1']}`",
        f"- M2=`{MODELS['M2']}`（不可用则 `{SKIP_DEPENDENCY}`，不换未冻结模型）",
        f"- Seed：S1={SEED_LADDER['S1_native']} → S2={SEED_LADDER['S2_group_date']}（最多 {SEED_LADDER['S2_max_advance']} 入围）→ S3={SEED_LADDER['S3_group_date']}（最多 {SEED_LADDER['S3_max_advance']}）→ S4={SEED_LADDER['S4_group_date']} 仅当预注册不确定区间（正增量比 75–85% 或 30→100 |Δmean Sharpe|>0.02）。未触发标 `{NOT_TRIGGERED}`，不得改成 10 seed 假装完成。",
        "- Native 5 seed 若退化：先 NATIVE_SEED_DIVERSITY_AUDIT，再 group-date 80% 不放回、按年分层、整日 query 同进同出，仍 subsample=1.0。",
        f"- 探索性门槛：Positive paired Δ ratio ≥{int(PAIRED_POS_RATIO_MIN*100)}%。不得把 80/100 写成 OOS p 值。",
        "",
        "## 晋级门",
        "",
        "- PIT PASS + v5 parity PASS + 增量可解释 + OOF 不恶化 + 同口径净收益优于 v5 + MaxDD 可接受 + 非仅 2024/2025 + 费用校正不严重反转 + 选择偏差披露。",
        "- F2 不得晋级。F1/F3 不得换名当新假设。禁止再做 5000 随机搜索。",
        "",
        f"inventory n_total=`{inventory.get('n_total')}`（R17 `{inventory.get('n_r17_registry')}` + R18 `{inventory.get('n_r18_f123')}`）。",
        "",
    ]
    write_md(CONTRACT_MD, "\n".join(lines))
    return payload
