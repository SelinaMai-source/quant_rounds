"""经济门：在任何候选组合回测之前落盘。禁止回调。"""

from __future__ import annotations

from typing import Any

import yaml

from .common import dump_json, load_json, sha256_json, write_md
from .contract import (
    AUTO_PROMOTE_CURRENT,
    C1_FORMULA_FROZEN,
    C1_ID,
    C4_FORMULA_FROZEN,
    C4_ID,
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    DECISION_AUDIT,
    DECISION_RULE,
    ECONOMIC_MATERIALITY,
    EXISTING_11,
    FEE_SCHEDULE,
    FIVE_FEATURES,
    FUSION_WEIGHTS,
    KEEP_BASELINE_V5,
    MANIFEST_PATH,
    MD,
    NEW_FACTOR_SPECS,
    RECOMMEND_V6_RESEARCH_CANDIDATE,
    ROUND22_2_ROOT,
    SUBPERIOD_ALPHA_RESEARCH_CANDIDATE,
    TEN_FEATURES,
    V5_ANCHOR,
    V6_RESEARCH_CANDIDATE,
)

FREEZE_JSON = ROUND22_2_ROOT / "configs" / "frozen_gates.json"


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
        "round": "22.2",
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
        "c1_id": C1_ID,
        "c4_id": C4_ID,
        "c1_formula_frozen": C1_FORMULA_FROZEN,
        "c4_formula_frozen": C4_FORMULA_FROZEN,
        "c1_no_buy_value_xl": True,
        "do_not_invent_c_formula": True,
        "existing_11": list(EXISTING_11),
        "new_factor_ids": [s["factor_id"] for s in NEW_FACTOR_SPECS],
        "allowed_decisions": [
            KEEP_BASELINE_V5,
            RECOMMEND_V6_RESEARCH_CANDIDATE,
            V6_RESEARCH_CANDIDATE,
            SUBPERIOD_ALPHA_RESEARCH_CANDIDATE,
        ],
        "forbid_live_ready_v6": True,
        "auto_promote": AUTO_PROMOTE_CURRENT,
        "current_baseline_changed": False,
        "do_not_overwrite_round22": True,
        "do_not_overwrite_round22_1": True,
        "do_not_overwrite_round21": True,
        "paused_methods": ["PCA", "ElasticNet", "hierarchical_ensemble", "timing", "position", "exit", "new_c_formulas"],
        "testing_label": "REPEATED_HISTORICAL_TESTING",
        "replace_only_if_train_fold_abs_rho_ge": 0.60,
        "real_drop_via": ["five_drops", "ten_drops"],
        "forbid_alias_add": True,
        "top_net_formula": "sell_proceeds + cash_dividend - buy_cost",
        "reuse_round22_1_top1_top2_lineage": True,
    }
    payload = {
        "status": "FROZEN",
        "frozen_before_formal_nav": True,
        "do_not_retune": True,
        "candidate_backtest_started": False,
        "body": body,
        "body_sha256": sha256_json(body),
    }
    from .hypotheses import write_hypotheses

    write_hypotheses()
    dump_json(FREEZE_JSON, payload)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_md(
        MD["00"],
        "\n".join(
            [
                "# Round22.2 · 00 研究契约",
                "",
                f"- CURRENT = `{CURRENT_BASELINE}` alias `{CURRENT_ALIAS}`。AUTO_PROMOTE=False。CURRENT_BASELINE_CHANGED=FALSE。",
                f"- 主口径 `{FEE_SCHEDULE}`。融合 Ten {FUSION_WEIGHTS['ten']:.2f} + Five {FUSION_WEIGHTS['five']:.2f}。",
                f"- Five {len(FIVE_FEATURES)} 列，Ten {len(TEN_FEATURES)} 列（Ten = Five 去掉 `g0_amount_log`）。",
                f"- 经济门 `{ECONOMIC_MATERIALITY['id']}` 已冻结，body_sha256=`{payload['body_sha256']}`。禁止回调。",
                f"- 决策允许 `{KEEP_BASELINE_V5}` / `{V6_RESEARCH_CANDIDATE}` / `{SUBPERIOD_ALPHA_RESEARCH_CANDIDATE}`。禁止 `LIVE_READY_V6`。",
                f"- 决策时点审计：{DECISION_AUDIT}。",
                f"- C1 冻结公式 `{C1_FORMULA_FROZEN}`（表无 BUY_VALUE_XL，不得改成别的高收益式）。",
                f"- C4 冻结公式 `{C4_FORMULA_FROZEN}`。必须滞后则新版本号，禁止同名悄悄改期。",
                "- 不覆盖 result/round22 / result/round22_1 / result/round21*。不自动升 v6。",
                "- 11 个已有因子只提取 22.1 JSON，不重算同一套。HN1/BT1/MG1 等价则 DUPLICATE。",
                "- 真删走 five_drops/ten_drops，禁止 alias ADD。Replace 仅当训练折 |ρ|≥0.60。",
                "- 净收益 = sell_proceeds + cash_dividend − buy_cost。复用 22.1 Top1/Top2 lineage。",
                "- 无独立 Forward 不得 live-ready。短窗最多 SUBPERIOD_ALPHA_RESEARCH_CANDIDATE。",
                "",
            ]
        ),
    )
    return payload
