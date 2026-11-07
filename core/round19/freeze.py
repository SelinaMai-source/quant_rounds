"""看 C3/C4/concentration/B01 新 NAV 之前冻结 Track A/B manifest 并 sha256。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND19_ROOT, assert_project_result_path, ensure_round19_tree
from ..round11.contract import source_tree_sha256
from ..round11.freeze import _dump_yaml
from ..stock_readonly import sha256_file
from .common import dump_json, load_json, write_md
from .contract import (
    AWAITING_USER_LOCK,
    AUTO_PROMOTE_V5,
    COLSAMPLE,
    CURRENT_BASELINE,
    EVENT_LAYERS,
    FORBIDDEN,
    FORMAL_PROMOTION_BLOCKED,
    KEEP_BASELINE_V4,
    LAST_PRICE_PROXY_ONLY,
    MANIFEST_A_NAME,
    MANIFEST_B_NAME,
    MANIFEST_MD,
    O01_ANCHOR,
    O01_FORMULA,
    OPTION_CANDIDATE,
    PRIOR_RESEARCH_SELECTION_EXISTS,
    PROTOCOL_SOURCE_FILES,
    R16_SPA_P,
    SIX_LAYER_PARITY,
    STAGE0,
    SUBSAMPLE,
    TRACK_A_EXPERIMENTS,
    TRACK_A_GATES,
    TRACK_A_STAGES,
    TRACK_B_EXPERIMENTS,
    TRACK_B_GATES,
    TRACK_B_STAGES,
    V03_FEATURES,
    V4_ANCHOR,
)
from .parity import load_parity_status
from .read import READ_JSON

FREEZE_JSON = ROUND19_ROOT / "configs" / "freeze.json"


def manifest_a_path():
    return ROUND19_ROOT / MANIFEST_A_NAME


def manifest_b_path():
    return ROUND19_ROOT / MANIFEST_B_NAME


def sha256_a_path():
    return ROUND19_ROOT / f"{MANIFEST_A_NAME}.sha256"


def sha256_b_path():
    return ROUND19_ROOT / f"{MANIFEST_B_NAME}.sha256"


def _write_yaml(path, body: dict[str, Any]) -> str:
    assert_project_result_path(path)
    text = _dump_yaml(body)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def freeze_manifest(*, overwrite: bool = False, facts: dict[str, Any] | None = None) -> dict[str, Any]:
    ensure_round19_tree()
    if FREEZE_JSON.is_file() and not overwrite:
        existing = load_json(FREEZE_JSON)
        if existing.get("frozen") and manifest_a_path().is_file() and manifest_b_path().is_file():
            return existing
    parity = load_parity_status()
    read = load_json(READ_JSON)
    protocol = source_tree_sha256(PROTOCOL_SOURCE_FILES)
    shared = {
        "round": 19,
        "current": CURRENT_BASELINE,
        "parent": "R9_B03",
        "auto_promote_v5": AUTO_PROMOTE_V5,
        "default_promotion": KEEP_BASELINE_V4,
        "option_candidate": OPTION_CANDIDATE,
        "prior_research_selection_exists": PRIOR_RESEARCH_SELECTION_EXISTS,
        "subsample": SUBSAMPLE,
        "colsample": COLSAMPLE,
        "v4_anchor": dict(V4_ANCHOR),
        "o01_historical_anchor": dict(O01_ANCHOR),
        "parity_status": parity.get("status"),
        "parity_passed": bool(parity.get("passed")),
        "formal_promotion_blocked": bool(not parity.get("passed")),
        "formal_promotion_block_code": None if parity.get("passed") else FORMAL_PROMOTION_BLOCKED,
        "forbidden": list(FORBIDDEN),
        "lock_flag_if_recommend": AWAITING_USER_LOCK,
        "no_fuse_option_x_round18": True,
        "protocol_sha256": protocol,
        "prior_artifact_hashes": {
            rec.get("key"): rec.get("sha256") for rec in (read.get("files") or []) if rec.get("key")
        },
        "note": "冻结发生在任何 Track A C3/C4/concentration 或 Track B B01 新 NAV 之前。事后不放宽。",
    }
    if facts:
        shared["facts"] = facts
    body_a = {
        "manifest_id": "round19_track_a_manifest",
        "track": "A",
        "mission": "Confirmatory validation of R16_O01. Not a factor search. Not independent forward.",
        **shared,
        "o01_formula": O01_FORMULA,
        "event_layers": dict(EVENT_LAYERS),
        "v03_features_frozen": list(V03_FEATURES),
        "quote_default": LAST_PRICE_PROXY_ONLY,
        "spa_p_retained": R16_SPA_P,
        "experiments": [dict(x) for x in TRACK_A_EXPERIMENTS],
        "promotion_gates": dict(TRACK_A_GATES),
        "stages": list(STAGE0) + list(TRACK_A_STAGES),
        "controls_cannot_promote": ("R19_A_C2", "R19_A_C3", "R19_A_C4"),
    }
    body_b = {
        "manifest_id": "round19_track_b_manifest",
        "track": "B",
        "mission": "Forensics: is Round18 degradation a pipeline bug or the factors themselves?",
        **shared,
        "force_retrain_required_for_b01": True,
        "round18_skips_retrain_when_extras_empty": True,
        "six_layer_parity": [dict(x) for x in SIX_LAYER_PARITY],
        "experiments": [dict(x) for x in TRACK_B_EXPERIMENTS],
        "promotion_gates": dict(TRACK_B_GATES),
        "stages": list(STAGE0) + list(TRACK_B_STAGES),
        "diagnostic_not_eligible": True,
        "do_not_compete_with_o01_for_v5": True,
    }
    digest_a = _write_yaml(manifest_a_path(), body_a)
    digest_b = _write_yaml(manifest_b_path(), body_b)
    payload = {
        "frozen": True,
        "status": "FROZEN",
        "track_a": {"path": str(manifest_a_path()), "sha256": digest_a, "sha256_path": str(sha256_a_path())},
        "track_b": {"path": str(manifest_b_path()), "sha256": digest_b, "sha256_path": str(sha256_b_path())},
        "parity_status": parity.get("status"),
        "formal_promotion_blocked": bool(not parity.get("passed")),
        "subsample": SUBSAMPLE,
        "colsample": COLSAMPLE,
        "current": CURRENT_BASELINE,
        "auto_promote_v5": AUTO_PROMOTE_V5,
    }
    dump_json(FREEZE_JSON, payload)
    lines = [
        "# Round19 · 01 冻结清单",
        "",
        "两份独立 manifest 在看 Track A C3/C4/concentration 与 Track B B01 新 NAV **之前**冻结。事后不放宽。",
        "",
        "## sha256",
        "",
        f"- Track A `{manifest_a_path()}`",
        f"  sha256=`{digest_a}`",
        f"  sidecar=`{sha256_a_path()}`",
        f"- Track B `{manifest_b_path()}`",
        f"  sha256=`{digest_b}`",
        f"  sidecar=`{sha256_b_path()}`",
        "",
        "## 共享锁",
        "",
        f"- CURRENT=`{CURRENT_BASELINE}` AUTO_PROMOTE_V5=`{AUTO_PROMOTE_V5}`",
        f"- subsample=`{SUBSAMPLE}` colsample=`{COLSAMPLE}`",
        f"- 禁止 R16_O01 × Round18 融合；禁止改 CURRENT；追踪表不写「写入时间」。",
        f"- parity=`{parity.get('status')}` formal_promotion_blocked=`{not parity.get('passed')}`",
        "",
        "## Track A 预注册晋级门",
        "",
        "- C0/C1 exact parity PASS。",
        "- 事件账 332/298/241/39 可重建；口径混用 = FAIL。",
        "- C3 必须 `ACTIVITY_MATCHING_PASS` 才可比 Sharpe；匹配对象 = 决策时已知的 eligible Top2 日 + 年频处理次数 + year×B01A 分布 + 计划 Top2 veto 次数 + 计划风险预算。",
        "- 禁止用未来盈亏/是否成交选日期。事后 fill 匹配只标 `EX_POST_DIAGNOSTIC_ONLY`。",
        f"- C3 模型 = 原 V03 四列 Ridge `{list(V03_FEATURES)}`，只修匹配规则。",
        "- C4 N=1000 全引擎，禁止 ledger shuffle。C3/C4 不得晋级。",
        "- O01 必须同时：优于公平 C3、C4 p 具支持性、leave-one-event-out 不靠少数极端日、+20/+30bp 不翻转、披露 SPA 0.496 与三次检验选择偏差。",
        "- PIT 无法证明 `available_at <= decision_timestamp` → 不得写 `PIT_PASS` / `LIVE_READY`。无 bid/ask → `LAST_PRICE_PROXY_ONLY`。",
        "- 过门只允许 `A_RECOMMEND_LOCK_O01_AS_DEVELOPMENT_V5` + `AWAITING_USER_LOCK`。默认 `A_KEEP_O01_AS_UNCONFIRMED_CANDIDATE`。",
        "",
        "## Track B 预注册判定",
        "",
        "- `R19_B00` = locked Round7 预测重放。",
        "- `R19_B01` = Round18 管线 **force_retrain** 原 14/13 列、同一面板/折/标签/1.0/1.0/seeds，不加 F1/F2/F3。",
        "- 六层 parity 分别 PASS/FAIL/NOT_IDENTIFIABLE，不能只用 CAGR。XGBoost/环境差异不得当可忽略 PASS。",
        "- material bug：隔离修复 → 先证 B01≈B00 → 原公式重跑 R18，命名 `R19_B_REPAIRED_*`。",
        "- 无 bug → `CLOSE_ROUND18_FACTOR_HYPOTHESES` 或 `INCONCLUSIVE_DUE_TO_DATA_PIT`。禁止搜 epsilon / 改 F 公式 / 改 head 权重。",
        "",
    ]
    write_md(MANIFEST_MD, "\n".join(lines))
    return payload


def require_frozen() -> dict[str, Any]:
    frozen = load_json(FREEZE_JSON)
    if not frozen.get("frozen"):
        raise RuntimeError("MANIFEST_NOT_FROZEN STOP; refuse Track A/B/C new NAV")
    if not manifest_a_path().is_file() or not manifest_b_path().is_file():
        raise RuntimeError("MANIFEST_NOT_FROZEN STOP; missing track yaml")
    return frozen


def require_parity_for_promotion() -> dict[str, Any]:
    parity = load_parity_status()
    if not parity.get("passed"):
        raise RuntimeError(f"{FORMAL_PROMOTION_BLOCKED} Stage0 parity FAIL")
    return parity
