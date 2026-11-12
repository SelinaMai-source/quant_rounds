"""经济门：在任何候选组合回测之前落盘。禁止回调。"""

from __future__ import annotations

from typing import Any

import yaml

from ..paths import ROUND22_1_ROOT
from .common import dump_json, load_json, sha256_json, write_md
from .contract import (
    AUTO_PROMOTE_CURRENT,
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    DECISION_AUDIT,
    DECISION_RULE,
    ECONOMIC_MATERIALITY,
    FACTOR_IDS,
    FEE_SCHEDULE,
    FIVE_FEATURES,
    FUSION_WEIGHTS,
    KEEP_BASELINE_V5,
    MANIFEST_PATH,
    MD,
    PIT_PASS_IDS,
    RECOMMEND_V6_RESEARCH_CANDIDATE,
    TEN_FEATURES,
    V5_ANCHOR,
)

FREEZE_JSON = ROUND22_1_ROOT / "configs" / "frozen_gates.json"


def require_frozen_gate() -> dict[str, Any]:
    payload = load_json(FREEZE_JSON)
    if not payload or payload.get("status") != "FROZEN":
        raise RuntimeError("economic gate is not frozen; refuse candidate backtest")
    if payload.get("do_not_retune") is not True or payload.get("frozen_before_formal_nav") is not True:
        raise RuntimeError("frozen gate flags missing; refuse candidate backtest")
    body = payload.get("body") or {}
    if sha256_json(body) != payload.get("body_sha256"):
        raise RuntimeError("frozen gate hash mismatch; refuse retune")
    if (body.get("economic_materiality") or {}).get("id") != "R21_3_ECONOMIC_MATERIALITY_V1":
        raise RuntimeError("economic gate id drifted; refuse retune")
    return payload


def run_freeze(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and FREEZE_JSON.is_file() and MANIFEST_PATH.is_file():
        existing = load_json(FREEZE_JSON)
        if existing.get("status") == "FROZEN" and existing.get("body_sha256"):
            return existing
    body = {
        "round": "22.1",
        "baseline": CURRENT_BASELINE,
        "alias": CURRENT_ALIAS,
        "fee_schedule": FEE_SCHEDULE,
        "fusion": dict(FUSION_WEIGHTS),
        "five_n": len(FIVE_FEATURES),
        "ten_n": len(TEN_FEATURES),
        "five_features": list(FIVE_FEATURES),
        "ten_features": list(TEN_FEATURES),
        "v5_anchor": dict(V5_ANCHOR),
        "economic_materiality": ECONOMIC_MATERIALITY,
        "decision_rule": DECISION_RULE,
        "decision_audit": DECISION_AUDIT,
        "n_factors": len(FACTOR_IDS),
        "factor_ids": list(FACTOR_IDS),
        "pit_pass_ids": list(PIT_PASS_IDS),
        "allowed_decisions": [KEEP_BASELINE_V5, RECOMMEND_V6_RESEARCH_CANDIDATE],
        "auto_promote": AUTO_PROMOTE_CURRENT,
        "current_baseline_changed": False,
        "do_not_retrain_51": True,
        "do_not_overwrite_round22": True,
        "paused_methods": ["PCA", "ElasticNet", "hierarchical_ensemble", "timing", "position", "exit", "new_formulas"],
        "testing_label": "REPEATED_HISTORICAL_TESTING",
    }
    payload = {
        "status": "FROZEN",
        "frozen_before_formal_nav": True,
        "do_not_retune": True,
        "candidate_backtest_started": False,
        "body": body,
        "body_sha256": sha256_json(body),
    }
    dump_json(FREEZE_JSON, payload)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_md(
        MD["00"],
        "\n".join(
            [
                "# Round22.1 · 00 研究契约",
                "",
                f"- CURRENT = `{CURRENT_BASELINE}` alias `{CURRENT_ALIAS}`。AUTO_PROMOTE=False。CURRENT_BASELINE_CHANGED=FALSE。",
                f"- 主口径 `{FEE_SCHEDULE}`。融合 Ten {FUSION_WEIGHTS['ten']:.2f} + Five {FUSION_WEIGHTS['five']:.2f}。",
                f"- Five {len(FIVE_FEATURES)} 列，Ten {len(TEN_FEATURES)} 列（Ten = Five 去掉 `g0_amount_log`）。",
                f"- 经济门 `{ECONOMIC_MATERIALITY['id']}` 已冻结，body_sha256=`{payload['body_sha256']}`。禁止回调。",
                f"- 决策只允许 `{KEEP_BASELINE_V5}` 或 `{RECOMMEND_V6_RESEARCH_CANDIDATE}`。无独立 Forward 不得写成 live-ready。",
                f"- 决策时点审计：{DECISION_AUDIT}。",
                "- 不覆盖 result/round22 或 result/round21*。不重训 51 模型。不自动升 v6。",
                "- 11 个 PIT_PASS 因子只补证据；14 个阻断因子只做语义核对或授权内恢复。",
                "- 新增账本字段不得改成交价、仓位、退出、费率、NAV。",
                "",
            ]
        ),
    )
    return payload
