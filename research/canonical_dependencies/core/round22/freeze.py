"""经济门与实验漏斗：在任何候选组合回测之前落盘。禁止回调。"""

from __future__ import annotations

from typing import Any

import yaml

from ..paths import ROUND22_ROOT
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
    OOF_INCREMENT_GATE,
    PROFILE_A,
    PROFILE_B,
    RECOMMEND_V6_RESEARCH_CANDIDATE,
    TEN_FEATURES,
    TRAIN_GATE,
    V5_ANCHOR,
)

FREEZE_JSON = ROUND22_ROOT / "configs" / "frozen_gates.json"


def require_frozen_gate() -> dict[str, Any]:
    payload = load_json(FREEZE_JSON)
    if not payload or payload.get("status") != "FROZEN":
        raise RuntimeError("economic gate is not frozen; refuse candidate backtest")
    if payload.get("do_not_retune") is not True or payload.get("frozen_before_formal_nav") is not True:
        raise RuntimeError("frozen gate flags missing; refuse candidate backtest")
    body = payload.get("body") or {}
    if sha256_json(body) != payload.get("body_sha256"):
        raise RuntimeError("frozen gate hash mismatch; refuse retune")
    return payload


def run_freeze(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and FREEZE_JSON.is_file() and MANIFEST_PATH.is_file():
        existing = load_json(FREEZE_JSON)
        if existing.get("status") == "FROZEN" and existing.get("body_sha256"):
            return existing
    body = {
        "round": "22",
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
        "oof_increment_gate": OOF_INCREMENT_GATE,
        "train_gate": TRAIN_GATE,
        "decision_rule": DECISION_RULE,
        "decision_audit": DECISION_AUDIT,
        "profiles": {"A": PROFILE_A, "B": PROFILE_B},
        "n_factors": len(FACTOR_IDS),
        "factor_ids": list(FACTOR_IDS),
        "allowed_decisions": [KEEP_BASELINE_V5, RECOMMEND_V6_RESEARCH_CANDIDATE],
        "auto_promote": AUTO_PROMOTE_CURRENT,
        "current_baseline_changed": False,
        "paused_methods": ["PCA", "ElasticNet", "hierarchical_ensemble", "timing", "position", "exit"],
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
                "# Round22 · 00 研究契约",
                "",
                f"- CURRENT = `{CURRENT_BASELINE}` alias `{CURRENT_ALIAS}`。AUTO_PROMOTE=False。CURRENT_BASELINE_CHANGED=FALSE。",
                f"- 主口径 `{FEE_SCHEDULE}`。融合 Ten {FUSION_WEIGHTS['ten']:.2f} + Five {FUSION_WEIGHTS['five']:.2f}。",
                f"- Five {len(FIVE_FEATURES)} 列，Ten {len(TEN_FEATURES)} 列（Ten = Five 去掉 `g0_amount_log`）。",
                f"- 经济门 `{ECONOMIC_MATERIALITY['id']}` 已冻结，body_sha256=`{payload['body_sha256']}`。禁止回调。",
                f"- 决策只允许 `{KEEP_BASELINE_V5}` 或 `{RECOMMEND_V6_RESEARCH_CANDIDATE}`。无独立 Forward 不得写成 live-ready。",
                f"- 决策时点审计：{DECISION_AUDIT}。",
                "- Profile A 与 Profile B 分表。龙虎榜只进 Profile B。",
                "- 低相关 |ρ|<0.60 只做 Add。Remove/Replace 仅当训练折 |ρ|≥0.60。E3 不得没有 E1。E5 不得没有 E4。",
                "- 明确不做：PCA、Elastic Net、Hierarchical Ensemble、择时、仓位、退出。",
                "",
            ]
        ),
    )
    return payload
