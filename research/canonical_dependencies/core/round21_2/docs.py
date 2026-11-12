"""更新 STANDARD + evaluate_factor 工厂步。保留 Round21.1 段落。"""

from __future__ import annotations

from pathlib import Path

from .contract import FACTORY_STEPS_EXTRA, STANDARD_MD


STANDARD_ADDENDUM = """
## Round 21.2 强制增量（第二波 Alpha + 替换法医修补）

在 Round 21.1 替换法医之上，本轮额外强制：

| 步 | 含义 |
|---|---|
| `FORMULA_EQUIVALENCE_CHECK` | 新公式不得是历史失败因子换名；必须写 `round21_2_prior_factor_lineage.csv` |
| `REPLACE_HASH_INTEGRITY` | E0/E1/E2/E3 的 `feature_set_hash` / pred hash 必须互异；禁止 alias ADD=Replace |
| 显式 ρ 字段 | `full_sample_signed_rho` / `full_sample_abs_rho` / `fold_signed_rho` / `fold_abs_rho` / `mean_daily_rho` / `median_daily_rho` / `mean_absolute_daily_rho` |
| Coverage masks | `applicability_mask` / `missingness_indicator` / `strict_pit_mask` / full-universe XGB 原生 NA+flag；全池 80% 不是唯一门槛 |
| NO_OP 分列 | 近恒定稀有 flag（如 `r21_a1_missing_flag`）不得与有增量候选并列 |

G–N 32 个新公式必须从原始字段计算。缺失不填 0。`BLOCKED_DATA` 不得伪造。正式晋级只接受 `STRICT_PIT`。money_flow 2016–2020 回填年不得当 STRICT。决策只允许 `KEEP_BASELINE_V5` 或 `RECOMMEND_ALPHA_CANDIDATE`。`AUTO_PROMOTE=False`。`CURRENT_BASELINE_CHANGED=FALSE`。
"""


def update_standard() -> dict:
    text = STANDARD_MD.read_text(encoding="utf-8") if STANDARD_MD.is_file() else "# Alpha Factor Research Standard\n"
    marker = "## Round 21.2 强制增量（第二波 Alpha + 替换法医修补）"
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
    if "FORMULA_EQUIVALENCE_CHECK" not in src:
        src = src.replace(
            "OOF_INCREMENTALITY → FULL_PORTFOLIO_INCREMENTALITY",
            "OOF_INCREMENTALITY → FULL_PORTFOLIO_INCREMENTALITY →\n"
            "FORMULA_EQUIVALENCE_CHECK → REPLACE_HASH_INTEGRITY",
        )
        changed = True
    if "from core.round21_2.contract import FACTORY_STEPS_EXTRA as R212_EXTRA" not in src:
        src = src.replace(
            "    from core.round21_1.contract import FACTORY_STEPS_EXTRA\n",
            "    from core.round21_1.contract import FACTORY_STEPS_EXTRA\n"
            "    from core.round21_2.contract import FACTORY_STEPS_EXTRA as R212_EXTRA\n",
        )
        src = src.replace(
            "        print(\"\\n\".join(list(FACTORY_STEPS) + list(FACTORY_STEPS_EXTRA)), flush=True)",
            "        extra = list(dict.fromkeys([*FACTORY_STEPS_EXTRA, *R212_EXTRA]))\n"
            "        print(\"\\n\".join(list(FACTORY_STEPS) + extra), flush=True)",
        )
        changed = True
    if changed:
        path.write_text(src, encoding="utf-8")
    return {"path": str(path), "extra_steps": list(FACTORY_STEPS_EXTRA), "updated": changed}
