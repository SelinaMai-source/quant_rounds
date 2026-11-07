"""Round 21.4 研究契约冻结：看结果前写 manifest。禁止回调。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND21_4_ROOT
from .common import dump_json, write_md
from .contract import (
    A_CONTRASTS,
    A_IDENTITY_ASSERTS,
    A_SPECS,
    AUTO_PROMOTE_CURRENT,
    COMBO_MATRIX,
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    ECONOMIC_MATERIALITY,
    FEE_SCHEDULE,
    FUSION_WEIGHTS,
    KEEP_BASELINE_V5,
    MANIFEST_NAME,
    MD,
    OOF_INCREMENT_GATE,
    PREREGISTERED_INTERACTIONS,
    Q_ITEMS,
    REPLICATION_ANCHORS,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    WXYZ_FACTOR_IDS,
    WXYZ_MECHANISMS,
)

FREEZE_JSON = ROUND21_4_ROOT / "configs" / "round21_4_contract.json"


def _manifest() -> str:
    lines = [
        "# Round21.4 manifest（看正式回测结果前冻结；禁止按结果回调）",
        f"round: '21.4'",
        f"current: {CURRENT_BASELINE}",
        f"alias: {CURRENT_ALIAS}",
        f"fee_schedule: {FEE_SCHEDULE}",
        f"fusion: '0.90 ten + 0.10 five'",
        "auto_promote_current: false",
        "current_baseline_changed: false",
        "decision_space: [KEEP_BASELINE_V5, V6_RESEARCH_CANDIDATE]",
        "v5_anchor:",
        f"  cagr: {V5_ANCHOR['cagr']}",
        f"  sharpe: {V5_ANCHOR['sharpe']}",
        f"  max_drawdown: {V5_ANCHOR['max_drawdown']}",
        f"  completed_trades: {V5_ANCHOR['completed_trades']}",
        f"  nav_end: {V5_NAV_END_EXACT}",
        "economic_materiality:",
        f"  id: {ECONOMIC_MATERIALITY['id']}",
        "  carried_from: round21_3",
        "  do_not_retune: true",
        f"  delta_cagr_min_abs: {ECONOMIC_MATERIALITY['delta_cagr_min_abs']}",
        f"  delta_sharpe_min: {ECONOMIC_MATERIALITY['delta_sharpe_min']}",
        f"  nav_rel_min: {ECONOMIC_MATERIALITY['nav_rel_min']}",
        f"  maxdd_worse_max_abs: {ECONOMIC_MATERIALITY['maxdd_worse_max_abs']}",
        "  predeclared_immaterial: '+0.003 sharpe / +42.90 RMB (21.2 L1 E3)'",
        f"a_specs: {len(A_SPECS)}",
        f"a_contrasts: {[c[2] for c in A_CONTRASTS]}",
        f"a_identity_asserts: {list(A_IDENTITY_ASSERTS)}",
        f"replication_anchors: {sorted(REPLICATION_ANCHORS)}",
        f"combo_matrix: {[c['code'] + ':' + c['method'] for c in COMBO_MATRIX]}",
        f"preregistered_interactions: {list(PREREGISTERED_INTERACTIONS)}",
        f"wxyz_factors: {len(WXYZ_FACTOR_IDS)}",
        f"wxyz_mechanisms: {sorted(WXYZ_MECHANISMS)}",
        f"oof_increment_gate: {OOF_INCREMENT_GATE['id']}",
        f"n_questions: {len(Q_ITEMS)}",
        "no_alias_add: true",
        "subsample: 1.0",
        "colsample_bytree: 1.0",
    ]
    return "\n".join(lines) + "\n"


def run_freeze(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    from .common import load_json

    if resume and FREEZE_JSON.is_file() and (ROUND21_4_ROOT / MANIFEST_NAME).is_file():
        existing = load_json(FREEZE_JSON)
        if existing.get("status") == "FROZEN":
            return existing
    payload = {
        "status": "FROZEN",
        "current": CURRENT_BASELINE,
        "alias": CURRENT_ALIAS,
        "fee_schedule": FEE_SCHEDULE,
        "fusion": FUSION_WEIGHTS,
        "auto_promote_current": AUTO_PROMOTE_CURRENT,
        "decision_space": [KEEP_BASELINE_V5, "V6_RESEARCH_CANDIDATE"],
        "v5_anchor": dict(V5_ANCHOR),
        "v5_nav_end": V5_NAV_END_EXACT,
        "economic_materiality": ECONOMIC_MATERIALITY,
        "a_specs": A_SPECS,
        "a_contrasts": [list(c) for c in A_CONTRASTS],
        "a_identity_asserts": [list(x) for x in A_IDENTITY_ASSERTS],
        "replication_anchors": REPLICATION_ANCHORS,
        "combo_matrix": COMBO_MATRIX,
        "preregistered_interactions": [list(x) for x in PREREGISTERED_INTERACTIONS],
        "wxyz_factor_ids": list(WXYZ_FACTOR_IDS),
        "wxyz_mechanisms": WXYZ_MECHANISMS,
        "oof_increment_gate": OOF_INCREMENT_GATE,
        "n_questions": len(Q_ITEMS),
    }
    dump_json(FREEZE_JSON, payload)
    (ROUND21_4_ROOT / MANIFEST_NAME).write_text(_manifest(), encoding="utf-8")
    write_md(
        MD["00"],
        "\n".join(
            [
                "# Round21.4 · 00 研究契约",
                "",
                f"CURRENT=`{CURRENT_BASELINE}` 不自动升 v6。主口径 `{FEE_SCHEDULE}`。融合 0.90 ten + 0.10 five。",
                f"冻结经济门沿用 21.3 `{ECONOMIC_MATERIALITY['id']}`（禁止回调）：ΔCAGR≥+1.0pct 且 ΔSharpe≥+0.05 且 |ΔNAV|/NAV≥1% 且 Top1/Top2 变化>0 且 MaxDD 恶化≤2pct 且费用校正方向不翻转。",
                "+0.003/+43 元预先 `ECONOMICALLY_IMMATERIAL`。",
                f"Track A：`{len(A_SPECS)}` 配置 A0–A9；断言 A3 复现 21.2、A6 复现 21.3；恒等 `{list(A_IDENTITY_ASSERTS)}`。",
                f"Track D：C0–C11 `{len(COMBO_MATRIX)}` 矩阵，不预设任何方法优于 Raw；全不如 Raw 则 `RAW_BASELINE`。",
                f"Track E：W/X/Y/Z `{len(WXYZ_FACTOR_IDS)}` 预注册因子，机制 `{sorted(WXYZ_MECHANISMS)}`。",
                "决策空间仅 `KEEP_BASELINE_V5` / `V6_RESEARCH_CANDIDATE`（无独立 Forward 不 live-ready）。",
                "",
            ]
        ),
    )
    return payload
