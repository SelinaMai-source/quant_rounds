"""经济门：在任何候选组合回测之前落盘。对阈值对象做 canonical JSON。禁止回调。"""

from __future__ import annotations

from typing import Any

import yaml

from .common import dump_json, load_json, sha256_json, write_md
from .contract import (
    AUTO_PROMOTE_CURRENT,
    C1_FORMULA_FROZEN,
    C1_ID,
    C1_LAG1_ID,
    C4_FORMULA_FROZEN,
    C4_ID,
    C_AVAILABLE_FROM,
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    DECISION_AUDIT,
    DECISION_RULE,
    ECONOMIC_MATERIALITY,
    FEE_SCHEDULE,
    FIRST_PREDICTION_YEAR,
    FIVE_FEATURES,
    FUSION_WEIGHTS,
    HN2_FORMULA_FROZEN,
    HN2_ID,
    KEEP_BASELINE_V5,
    LAST_PREDICTION_YEAR,
    MAIN_LEDGER,
    MANIFEST_PATH,
    MD,
    NEW_FACTOR_SPECS,
    REPLAY_GROUPS,
    ROUND22_3_ROOT,
    SHORT_LEDGER,
    SUBPERIOD_ALPHA_RESEARCH_CANDIDATE,
    TEN_FEATURES,
    V5_ANCHOR,
    V6_RESEARCH_CANDIDATE,
    canonical_threshold,
)

FREEZE_JSON = ROUND22_3_ROOT / "configs" / "frozen_gates.json"


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
    thresh = body.get("canonical_threshold") or canonical_threshold()
    if sha256_json(thresh) != payload.get("threshold_sha256"):
        raise RuntimeError("canonical threshold hash mismatch; refuse retune")
    return payload


def run_freeze(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and FREEZE_JSON.is_file() and MANIFEST_PATH.is_file():
        existing = load_json(FREEZE_JSON)
        if existing.get("status") == "FROZEN" and existing.get("threshold_sha256"):
            return existing
    thresh = canonical_threshold()
    body = {
        "round": "22.3",
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
        "canonical_threshold": thresh,
        "decision_rule": DECISION_RULE,
        "decision_audit": DECISION_AUDIT,
        "main_ledger": dict(MAIN_LEDGER),
        "short_ledger": dict(SHORT_LEDGER),
        "c1_id": C1_ID,
        "c1_lag1_id": C1_LAG1_ID,
        "c4_id": C4_ID,
        "hn2_id": HN2_ID,
        "c1_formula_frozen": C1_FORMULA_FROZEN,
        "c4_formula_frozen": C4_FORMULA_FROZEN,
        "hn2_formula_frozen": HN2_FORMULA_FROZEN,
        "c1_no_buy_value_xl": True,
        "c_available_from": C_AVAILABLE_FROM,
        "nan_before_pit": True,
        "never_fill_zero": True,
        "never_drop_names": True,
        "first_prediction_year": FIRST_PREDICTION_YEAR,
        "last_prediction_year": LAST_PREDICTION_YEAR,
        "walk_forward": "exact prior 3 calendar years train, next 1 calendar year predict from 2018",
        "compare_to": "R16_O01",
        "forbid_compare_to_r222_e0": True,
        "replay_codes": [g["code"] for g in REPLAY_GROUPS],
        "new_factor_ids": [s["factor_id"] for s in NEW_FACTOR_SPECS],
        "allowed_decisions": [
            KEEP_BASELINE_V5,
            "RECOMMEND_V6_RESEARCH_CANDIDATE",
            V6_RESEARCH_CANDIDATE,
            SUBPERIOD_ALPHA_RESEARCH_CANDIDATE,
        ],
        "forbid_live_ready_v6": True,
        "auto_promote": AUTO_PROMOTE_CURRENT,
        "current_baseline_changed": False,
        "do_not_overwrite_round22": True,
        "do_not_overwrite_round22_1": True,
        "do_not_overwrite_round22_2": True,
        "do_not_overwrite_round21": True,
        "paused_methods": ["PCA", "ElasticNet", "hierarchical_ensemble", "timing", "position", "exit", "new_c_formulas"],
        "testing_label": "REPEATED_HISTORICAL_TESTING",
        "replace_only_if_train_fold_abs_rho_ge": 0.60,
        "do_not_lower_replace_gate": True,
        "real_drop_via": ["five_drops", "ten_drops"],
        "forbid_alias_add": True,
        "top_net_formula": "sell_proceeds + cash_dividend - buy_cost",
        "forbid_paste_2023_preds_onto_2018_nav": True,
    }
    payload = {
        "status": "FROZEN",
        "frozen_before_formal_nav": True,
        "do_not_retune": True,
        "candidate_backtest_started": False,
        "body": body,
        "body_sha256": sha256_json(body),
        "threshold_sha256": sha256_json(thresh),
        "threshold_object": thresh,
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
                "# Round22.3 · 00 研究契约",
                "",
                f"- CURRENT = `{CURRENT_BASELINE}` alias `{CURRENT_ALIAS}`。AUTO_PROMOTE=False。CURRENT_BASELINE_CHANGED=FALSE。",
                f"- 主口径 `{FEE_SCHEDULE}`。融合 Ten {FUSION_WEIGHTS['ten']:.2f} + Five {FUSION_WEIGHTS['five']:.2f}。",
                f"- Five {len(FIVE_FEATURES)} 列，Ten {len(TEN_FEATURES)} 列。",
                f"- 经济门 `{ECONOMIC_MATERIALITY['id']}` 已冻结。body_sha256=`{payload['body_sha256']}`。threshold_sha256=`{payload['threshold_sha256']}`。禁止回调。",
                f"- 主账 `2018-01-02–2026-07-24` n_obs=2075 对照 `R16_O01`。短窗 `2023-01-03–2026-07-24` n_obs=860 对照 `R222_E0`，不得混表。",
                f"- C1 `{C1_FORMULA_FROZEN}`。C4 `{C4_FORMULA_FROZEN}`。HN2 `{HN2_FORMULA_FROZEN}`。",
                f"- C1/C4 在 `{C_AVAILABLE_FROM}` 前保持 NaN，不填 0，不删股票。规范 3y/1y 从 2018 起。",
                "- Δ 只对照全历史 v5，不对照 R222_E0 Sharpe 2.735。禁止把 2023+ 预测贴到 2018 NAV。",
                "- 不覆盖 result/round22* / result/round21*。不自动升 v6。禁止 LIVE_READY_V6。",
                "- Replace 仅当训练折 |ρ|≥0.60，不准因 C4 vs g1_ret_5≈0.576 降门。",
                "",
            ]
        ),
    )
    return payload
