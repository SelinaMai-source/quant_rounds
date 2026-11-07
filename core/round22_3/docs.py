"""更新 STANDARD + evaluate_factor CLI。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import write_md
from .contract import EVALUATE_PY, FACTORY_STEPS_EXTRA, MD, STANDARD_MD

STANDARD_MARKER = "## Round 22.3 强制增量"
STANDARD_ADDENDUM = """

## Round 22.3 强制增量（两本账 × 全历史重跑 × canonical 门槛哈希）

在 Round 22.2 之上，本轮额外强制：

| 步 | 含义 |
|---|---|
| `TWO_LEDGER_ISOLATION` | 主账 2018–2026 n_obs≈2075 对照 R16_O01；22.2 短窗 2023–2026 n_obs=860 对照 R222_E0。禁止混表。短窗行标 `MATCHED_SUBPERIOD_NOT_COMPARABLE` |
| `CANONICAL_THRESHOLD_HASH` | 经济门对阈值对象做 canonical JSON。body.round 导致的 SHA 差标 `SEMANTICALLY_IDENTICAL_HASH_MISMATCH`，门槛漂移才 `ECONOMIC_GATE_SEMANTIC_DRIFT` |
| `FULL_HISTORY_REPLAY` | 22.2 全部正式实验组按规范 3y/1y 从 2018 起重训+全袖套。禁止把 2023+ 预测贴到 2018 NAV |
| `NAN_BEFORE_PIT_KEEP_POOL` | C1/C4 在 2020-09-24 前保持 NaN，不填 0，不删股票。XGB 按缺失处理 |
| `C4_DUAL_LEDGER_ATTRIBUTION` | 短窗 E1 ΔSharpe=+0.025 只作附录。主裁决用全历史 vs v5。\\|ρ\\| vs g1_ret_5≈0.576 不准降 Replace 门 |
| `HN2_IDENTITY_DECOMPOSITION` | 锁 22.2 公式并检验是否 Δlog(circ)−Δlog(holders)。H1–H3 全历史真训 |
| `FS_REPORT_ID_STATEMENT_ALIGN` | facts→reports 用 report_id；利润与现金流按 instrument+period+statement+available_at+revision。不同 statement 的 report_id 不得当同一 ID |

决策默认 `KEEP_BASELINE_V5`。禁止 `LIVE_READY_V6`。`AUTO_PROMOTE=False`。`CURRENT_BASELINE_CHANGED=FALSE`。无独立 Forward 不得 live-ready。
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
        "# Round22.3 · 21 STANDARD\n\n已追加 Round 22.3 强制增量：两本账隔离、canonical 门槛哈希、全历史重跑、NaN 保池、C4 双账、HN2 恒等、FS report_id 对齐。\n",
    )
    return {"path": str(STANDARD_MD), "extra_steps": list(FACTORY_STEPS_EXTRA)}


def update_evaluate_factor() -> dict[str, Any]:
    path = Path(EVALUATE_PY)
    src = path.read_text(encoding="utf-8")
    changed = False
    if "R223_EXTRA" not in src:
        if "from core.round22_2.contract import FACTORY_STEPS_EXTRA as R222_EXTRA" in src:
            src = src.replace(
                "    from core.round22_2.contract import FACTORY_STEPS_EXTRA as R222_EXTRA\n",
                "    from core.round22_2.contract import FACTORY_STEPS_EXTRA as R222_EXTRA\n"
                "    from core.round22_3.contract import FACTORY_STEPS_EXTRA as R223_EXTRA\n",
            )
            changed = True
        if "*R222_EXTRA]))" in src and "*R223_EXTRA" not in src:
            src = src.replace(
                "        extra = list(dict.fromkeys([*FACTORY_STEPS_EXTRA, *R212_EXTRA, *R213_EXTRA, *R214_EXTRA, *R22_EXTRA, *R221_EXTRA, *R222_EXTRA]))",
                "        extra = list(dict.fromkeys([*FACTORY_STEPS_EXTRA, *R212_EXTRA, *R213_EXTRA, *R214_EXTRA, *R22_EXTRA, *R221_EXTRA, *R222_EXTRA, *R223_EXTRA]))",
            )
            changed = True
        needle = """                        from core.round22_2.evaluate import evaluate_factor as evaluate_factor_222

                        try:
                            payload = evaluate_factor_222(args.factor_id, through=args.through, persist=args.persist_gate)
                        except KeyError:
                            print(f"UNKNOWN_FACTOR:{args.factor_id}", flush=True)
                            return 2"""
        repl = """                        from core.round22_2.evaluate import evaluate_factor as evaluate_factor_222

                        try:
                            payload = evaluate_factor_222(args.factor_id, through=args.through, persist=args.persist_gate)
                        except KeyError:
                            from core.round22_3.evaluate import evaluate_factor as evaluate_factor_223

                            try:
                                payload = evaluate_factor_223(args.factor_id, through=args.through, persist=args.persist_gate)
                            except KeyError:
                                print(f"UNKNOWN_FACTOR:{args.factor_id}", flush=True)
                                return 2"""
        if "evaluate_factor_223" not in src and needle in src:
            src = src.replace(needle, repl)
            changed = True
    if changed:
        path.write_text(src, encoding="utf-8")
    return {"path": str(path), "changed": changed, "has_223": "evaluate_factor_223" in path.read_text(encoding="utf-8")}
