"""更新 STANDARD + evaluate_factor。覆盖拒绝前必须 Data Recovery。"""

from __future__ import annotations

from pathlib import Path

from .contract import FACTORY_STEPS_EXTRA, STANDARD_MD

STANDARD_ADDENDUM = """
## Round 21.3 强制增量（L1 法医 + T/U/V + 经济意义门）

在 Round 21.2 之上，本轮额外强制：

| 步 | 含义 |
|---|---|
| `DATA_RECOVERY_BEFORE_COVERAGE_REJECT` | 覆盖不足必须走 DISCOVER→RECOVER→VALIDATE→RECOMPUTE，禁止直接放弃 |
| `FORMULA_EQUIVALENCE_CHECK` | 代数≡G/H/K/L 记 `DUPLICATE`，不新开 ID |
| `TRUE_DELETE_ASSERTION` | E3 = E0 − {dropped} + {new}；`g0_amount_log` 仅 Five-only |
| `FEE_LEDGER_COMPLETENESS` | `ROUND7_FLAT_5BP` 与 `HISTORICAL_FEE_CORRECTED` 必须有真实 Sharpe；`ran`+`None` 视为未完成 |
| `BOOTSTRAP_SPA_NUMERICAL` | Bootstrap N=5000 block=20；SPA 必须输出观测统计量 / 备择数 / 数值 p |
| `SEED_STATUS_ENUM` | 只能是 COMPLETE / NOT_TRIGGERED / BLOCKED / FAILED |
| `ECONOMIC_MATERIALITY_GATE` | ΔCAGR≥+1.0pct 且 ΔSharpe≥+0.05 且 \\|ΔNAV\\|/NAV≥1% 且 Top1/Top2 变化>0 且 MaxDD 恶化≤2pct 且费用方向不翻转。21.2 +0.003/+43 预先不达标 |
| `FORWARD_AVAILABILITY` | 无独立 Forward 不得 live-ready，最多 `V6_RESEARCH_CANDIDATE` |

决策只允许 `KEEP_BASELINE_V5` / `RECOMMEND_BASELINE_V6_CANDIDATE` / `V6_RESEARCH_CANDIDATE`。`AUTO_PROMOTE=False`。`CURRENT_BASELINE_CHANGED=FALSE`。失败进 Rejected Archive。
"""


def update_standard() -> dict:
    text = STANDARD_MD.read_text(encoding="utf-8") if STANDARD_MD.is_file() else "# Alpha Factor Research Standard\n"
    marker = "## Round 21.3 强制增量（L1 法医 + T/U/V + 经济意义门）"
    if marker in text:
        head, _ = text.split(marker, 1)
        text = head.rstrip() + "\n" + STANDARD_ADDENDUM
    else:
        text = text.rstrip() + "\n" + STANDARD_ADDENDUM
    STANDARD_MD.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return {"path": str(STANDARD_MD), "extra_steps": list(FACTORY_STEPS_EXTRA)}


def update_evaluate_factor() -> dict:
    path = Path(__file__).resolve().parents[2] / "tools" / "evaluate_factor.py"
    src = path.read_text(encoding="utf-8")
    changed = False
    if "from core.round21_3.contract import FACTORY_STEPS_EXTRA as R213_EXTRA" not in src:
        src = src.replace(
            "    from core.round21_2.contract import FACTORY_STEPS_EXTRA as R212_EXTRA\n",
            "    from core.round21_2.contract import FACTORY_STEPS_EXTRA as R212_EXTRA\n"
            "    from core.round21_3.contract import FACTORY_STEPS_EXTRA as R213_EXTRA\n",
        )
        src = src.replace(
            "        extra = list(dict.fromkeys([*FACTORY_STEPS_EXTRA, *R212_EXTRA]))\n",
            "        extra = list(dict.fromkeys([*FACTORY_STEPS_EXTRA, *R212_EXTRA, *R213_EXTRA]))\n",
        )
        changed = True
    if "DATA_RECOVERY_BEFORE_COVERAGE_REJECT" not in src:
        src = src.replace(
            "FORMULA_EQUIVALENCE_CHECK → REPLACE_HASH_INTEGRITY",
            "FORMULA_EQUIVALENCE_CHECK → REPLACE_HASH_INTEGRITY →\n"
            "DATA_RECOVERY_BEFORE_COVERAGE_REJECT → FEE_LEDGER_COMPLETENESS →\n"
            "BOOTSTRAP_SPA_NUMERICAL → SEED_STATUS_ENUM → ECONOMIC_MATERIALITY_GATE → FORWARD_AVAILABILITY",
        )
        changed = True
    if changed:
        path.write_text(src, encoding="utf-8")
    return {"path": str(path), "extra_steps": list(FACTORY_STEPS_EXTRA), "updated": changed}
