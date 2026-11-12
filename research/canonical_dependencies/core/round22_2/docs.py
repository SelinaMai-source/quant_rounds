"""更新 STANDARD + evaluate_factor CLI。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import write_md
from .contract import EVALUATE_PY, FACTORY_STEPS_EXTRA, MD, STANDARD_MD

STANDARD_MARKER = "## Round 22.2 强制增量"
STANDARD_ADDENDUM = """

## Round 22.2 强制增量（严格 PIT 时间线 × Matched 重训 × 去重新机制）

在 Round 22.1 之上，本轮额外强制：

| 步 | 含义 |
|---|---|
| `STRICT_PIT_TIMELINE` | C1/C4 必须按 `UPDATE_TIME`/`available_at` vs `D_close` 逐年审计。覆盖≥80% + 分档恒等 ≠ 全历史严格 PIT |
| `MATCHED_SAMPLE_RETRAIN` | 非全历史则建同日同池 matched v5 并重训。禁止截断全历史 NAV，禁止短窗 Sharpe 与 2018–2026 v5 混表。训练不足 3 年标 `INSUFFICIENT_HISTORY` |
| `NO_ALIAS_ADD` | 真删走 `five_drops`/`ten_drops`。Replace 仅当训练折 \\|ρ\\|≥0.60 |
| `DUPLICATE_LINEAGE_CHECK` | HN1/BT1/MG1 与 Round21 等价则 `DUPLICATE`，不当新 Alpha。必须滞后则新版本号 |
| `TOP1_TOP2_LEDGER_TRACE` | 净收益 = sell_proceeds+cash_dividend−buy_cost。缺列不得写成已算不合格 |
| `ECONOMIC_MATERIALITY_GATE` | 沿用 R21_3_ECONOMIC_MATERIALITY_V1，禁止回调 |
| `FORWARD_AVAILABILITY` | 无独立 Forward 不得 live-ready。短窗最多 `SUBPERIOD_ALPHA_RESEARCH_CANDIDATE` |

决策默认 `KEEP_BASELINE_V5`。禁止 `LIVE_READY_V6`。`AUTO_PROMOTE=False`。`CURRENT_BASELINE_CHANGED=FALSE`。
"""


def update_standard() -> dict[str, Any]:
    text = STANDARD_MD.read_text(encoding="utf-8") if STANDARD_MD.is_file() else "# Alpha Factor Research Standard\n"
    if STANDARD_MARKER in text:
        head, _ = text.split(STANDARD_MARKER, 1)
        text = head.rstrip() + STANDARD_ADDENDUM
    else:
        text = text.rstrip() + STANDARD_ADDENDUM
    STANDARD_MD.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    write_md(
        MD["21"],
        "# Round22.2 · 21 STANDARD\n\n已追加 Round 22.2 强制增量：严格 PIT 时间线、matched 重训、去重、真删、Top1/Top2、经济门、禁止 live-ready。\n",
    )
    return {"path": str(STANDARD_MD), "extra_steps": list(FACTORY_STEPS_EXTRA)}


def update_evaluate_factor() -> dict[str, Any]:
    path = Path(EVALUATE_PY)
    src = path.read_text(encoding="utf-8")
    changed = False
    if "R222_EXTRA" not in src:
        if "from core.round22_1.contract import FACTORY_STEPS_EXTRA as R221_EXTRA" in src:
            src = src.replace(
                "    from core.round22_1.contract import FACTORY_STEPS_EXTRA as R221_EXTRA\n",
                "    from core.round22_1.contract import FACTORY_STEPS_EXTRA as R221_EXTRA\n"
                "    from core.round22_2.contract import FACTORY_STEPS_EXTRA as R222_EXTRA\n",
            )
            changed = True
        if "*R221_EXTRA]))" in src and "*R222_EXTRA" not in src:
            src = src.replace(
                "        extra = list(dict.fromkeys([*FACTORY_STEPS_EXTRA, *R212_EXTRA, *R213_EXTRA, *R214_EXTRA, *R22_EXTRA, *R221_EXTRA]))",
                "        extra = list(dict.fromkeys([*FACTORY_STEPS_EXTRA, *R212_EXTRA, *R213_EXTRA, *R214_EXTRA, *R22_EXTRA, *R221_EXTRA, *R222_EXTRA]))",
            )
            changed = True
        needle = """                    from core.round22_1.evaluate import evaluate_factor as evaluate_factor_221

                    try:
                        payload = evaluate_factor_221(args.factor_id, through=args.through, persist=args.persist_gate)
                    except KeyError:
                        print(f"UNKNOWN_FACTOR:{args.factor_id}", flush=True)
                        return 2"""
        repl = """                    from core.round22_1.evaluate import evaluate_factor as evaluate_factor_221

                    try:
                        payload = evaluate_factor_221(args.factor_id, through=args.through, persist=args.persist_gate)
                    except KeyError:
                        from core.round22_2.evaluate import evaluate_factor as evaluate_factor_222

                        try:
                            payload = evaluate_factor_222(args.factor_id, through=args.through, persist=args.persist_gate)
                        except KeyError:
                            print(f"UNKNOWN_FACTOR:{args.factor_id}", flush=True)
                            return 2"""
        if "evaluate_factor_222" not in src and needle in src:
            src = src.replace(needle, repl)
            changed = True
    if changed:
        path.write_text(src, encoding="utf-8")
    return {"path": str(path), "changed": changed, "has_222": "evaluate_factor_222" in path.read_text(encoding="utf-8")}
