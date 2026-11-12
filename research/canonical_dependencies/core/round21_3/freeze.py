"""冻结经济意义门 + manifest。看正式回测前写入，禁止按结果回调。"""

from __future__ import annotations

from typing import Any

import yaml

from ..paths import ROUND21_3_ROOT
from .common import dump_json, sha256_json, write_md
from .contract import (
    AUTO_PROMOTE_CURRENT,
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    ECONOMIC_MATERIALITY,
    FEE_SCHEDULE,
    KEEP_BASELINE_V5,
    L1_SPECS,
    MANIFEST_NAME,
    MD,
    NEW_FACTOR_IDS,
    PREREGISTERED_INTERACTIONS,
    RECOMMEND_BASELINE_V6_CANDIDATE,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    V6_RESEARCH_CANDIDATE,
)

FREEZE_JSON = ROUND21_3_ROOT / "configs" / "freeze.json"
MANIFEST = ROUND21_3_ROOT / MANIFEST_NAME


def run_freeze(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and FREEZE_JSON.is_file() and MANIFEST.is_file() and MD["00"].is_file():
        from .common import load_json

        existing = load_json(FREEZE_JSON)
        if existing.get("status") == "FROZEN":
            return existing
    payload = {
        "status": "FROZEN",
        "round": "21.3",
        "current": CURRENT_BASELINE,
        "alias": CURRENT_ALIAS,
        "auto_promote": AUTO_PROMOTE_CURRENT,
        "current_baseline_changed": False,
        "fee_schedule": FEE_SCHEDULE,
        "v5_anchor": dict(V5_ANCHOR),
        "v5_nav_end": V5_NAV_END_EXACT,
        "economic_materiality": dict(ECONOMIC_MATERIALITY),
        "allowed_decisions": [KEEP_BASELINE_V5, RECOMMEND_BASELINE_V6_CANDIDATE, V6_RESEARCH_CANDIDATE],
        "no_independent_forward_max": V6_RESEARCH_CANDIDATE,
        "l1_specs": list(L1_SPECS),
        "new_factor_ids": list(NEW_FACTOR_IDS),
        "preregistered_interactions": [list(x) for x in PREREGISTERED_INTERACTIONS],
        "five_only_drop": "g0_amount_log",
        "no_alias_add": True,
        "e3_assert": "E3 = E0 - {g0_amount_log} + {r21_l1_known_float_chg}",
        "predeclared_21_2_immaterial": True,
    }
    payload["sha256"] = sha256_json(payload)
    dump_json(FREEZE_JSON, payload)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_md(
        MD["00"],
        "\n".join(
            [
                "# Round21.3 · 00 研究契约",
                "",
                "- CURRENT = `baseline_v5 = R16_O01`。AUTO_PROMOTE=False。不锁 v6。CURRENT_BASELINE_CHANGED=FALSE。",
                "- 主口径 `ROUND7_FLAT_5BP`。融合 0.90 Ten + 0.10 Five。XGB subsample=1.0 colsample=1.0。B03+O01。",
                "- 不改股票池、择时、仓位、成交、退出、手续费。",
                "- 相关与替换只对真实 14/13（`core/round7/contract.py`）。",
                "- E0/E1/E2/E3 必须真实重训。`five_drops`/`ten_drops` 真删。禁止 alias ADD 为 Replace。",
                "- `g0_amount_log` 仅 Five Head；E1/E3 只从 Five 删除，Ten 保持 13 列。",
                "- L1 vs g0_amount_log ρ≈−0.073 属 Feature Recomposition，必须有 E1。",
                "- 经济意义门已冻结（禁止按结果回调）：",
                f"  - ΔCAGR ≥ +{ECONOMIC_MATERIALITY['delta_cagr_min_abs']} 绝对",
                f"  - ΔSharpe ≥ +{ECONOMIC_MATERIALITY['delta_sharpe_min']}",
                f"  - |ΔNAV|/v5 NAV ≥ {ECONOMIC_MATERIALITY['nav_rel_min']}（约 ≥{ECONOMIC_MATERIALITY['nav_abs_min']:.0f} 元）",
                "  - Top1 或 Top2 变化率 > 0 且变化交易净 PnL 可解释",
                "  - MaxDD 恶化不超过 2 个百分点",
                "  - 费用校正口径下方向不翻转",
                "  - 不只依赖 2024 或 2025 单年",
                "- 21.2 +0.003 Sharpe / +43 元 **预先判定不达标**。",
                "- 决策只允许 KEEP_BASELINE_V5 / RECOMMEND_BASELINE_V6_CANDIDATE / V6_RESEARCH_CANDIDATE。",
                "- 无独立 Forward 最多 V6_RESEARCH_CANDIDATE。不得自动改 CURRENT。",
                f"- T/U/V 冻结 {len(NEW_FACTOR_IDS)} 个；交互预算 {len(PREREGISTERED_INTERACTIONS)} 条，禁止事后加乘。",
                f"- manifest `{MANIFEST}` sha256=`{payload['sha256']}`",
                "",
            ]
        ),
    )
    return payload
