"""更新 STANDARD + evaluate_factor 工厂步。"""

from __future__ import annotations

from pathlib import Path

from .contract import FACTORY_STEPS_EXTRA, STANDARD_MD


STANDARD_ADDENDUM = """
## Round 21.1 强制增量（替换法医）

每个新因子除 PIT / IC / Rolling / Top-tail / Conditional IC 外，正式晋级还必须完成：

| 步 | 含义 |
|---|---|
| `NEW_VS_BASELINE_CORRELATION` | 首板候选截面 vs **真实** 14/13（`core/round7/contract.py`），禁止用 KEEP-45 / ZyQuant 350 顶替 |
| `NEAREST_LEGACY_FACTOR_IDENTIFICATION` | `j* = argmax |Spearman|`，**TRAIN_FOLD** 官方；FULL_HISTORY 仅诊断 |
| `FOUR_WAY_REPLACEMENT_TEST` | E0 Control / E1 Remove / E2 Add / E3 Replace，必须真实重训，禁止 alias ADD |
| `CLUSTER_REPLACEMENT_IF_APPLICABLE` | 仅当训练窗 `|ρ|≥0.85` 且簇 ≥2 时做 E4/E5 |
| `OOF_INCREMENTALITY` | 四路 OOF NDCG / Rank IC |
| `FULL_PORTFOLIO_INCREMENTALITY` | 同口径 Round7 5bp 全引擎 |

未过 PIT 只出诊断，不得填 0 拉覆盖。E 族无公告时点仍 `BLOCKED_PIT`。

覆盖法医分类仅允许：`STRUCTURAL_NOT_APPLICABLE` / `EVENT_CONDITIONAL` / `TRUE_DATA_MISSING` / `SOURCE_ALIGNMENT_FAILURE` / `FORMULA_UNDEFINED` / `TIMESTAMP_UNVERIFIED`。必须同时报全池 Coverage、适用子样本 Conditional Coverage、真实缺失率。

训练缓存 key 必须包含 dropped + added + fold + head + `feature_set_hash`。`train_both_heads` 支持 `five_drops` / `ten_drops`，锁定列可以在实验层删除，但不得改 `FIVE_FEATURES` / `TEN_FEATURES` 常量。

Zoo 增补：`nearest_legacy_factor`、`correlation`、`correlation_stability`、`removed_factor`、`replace/add/remove_result`、`incremental_value_after_removal`、`replacement_verdict`。
"""


def update_standard() -> dict:
    text = STANDARD_MD.read_text(encoding="utf-8") if STANDARD_MD.is_file() else "# Alpha Factor Research Standard\n"
    marker = "## Round 21.1 强制增量（替换法医）"
    if marker in text:
        head, _ = text.split(marker, 1)
        # 丢掉旧增量后的内容再重写增量（保留 Round21 正文）
        text = head.rstrip() + "\n" + STANDARD_ADDENDUM
    else:
        text = text.rstrip() + "\n" + STANDARD_ADDENDUM
    STANDARD_MD.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return {"path": str(STANDARD_MD), "extra_steps": list(FACTORY_STEPS_EXTRA)}


def update_evaluate_factor() -> dict:
    path = Path(__file__).resolve().parents[2] / "tools" / "evaluate_factor.py"
    src = path.read_text(encoding="utf-8")
    if "NEW_VS_BASELINE_CORRELATION" not in src:
        raise RuntimeError("tools/evaluate_factor.py missing Round21.1 extra steps")
    return {"path": str(path), "extra_steps": list(FACTORY_STEPS_EXTRA), "already_updated": True}
