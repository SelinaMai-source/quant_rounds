"""更新 STANDARD + evaluate_factor。禁止静默 skip。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import dump_json, write_md
from .contract import EVALUATE_PY, FACTORY_STEPS_EXTRA, MD, STANDARD_MD

STANDARD_MARKER = "## Round 22.1 强制增量"
STANDARD_ADDENDUM = """

## Round 22.1 强制增量（IC 宇宙 × 强制证据 × Top1/Top2 账本 × 公式身份）

在 Round 22 之上，本轮额外强制：

| 步 | 含义 |
|---|---|
| `NO_SILENT_SKIP` | mandatory 阶段 `Rolling` / `Shape` / `TopTail` / `ConditionalIC` 禁止 `SKIP_IF_MISSING`。缺文件先跑，失败留日志 |
| `IC_UNIVERSE_AUDIT` | IC 必须写 universe / n_pairs / label。主宇宙是一进二 `candidate_panel`，不是全市场。日 n<5 跳过；累计有效日<60 才 `NOT_IDENTIFIABLE`。ICIR=mean/std(ddof=1)，未年化 |
| `MANDATORY_ROLLING_CONDITIONAL_SHAPE` | 11 个 PIT_PASS 必须有 rolling/conditional/shape/top_tail。缺任一 JSON 不得 `FULL_EVIDENCE_COMPLETE` |
| `TOP1_TOP2_LEDGER_TRACE` | 回测必须可追踪 Top1/Top2。`top_net_ok=False` 若因缺列不得解释为已算不合格 |
| `DATA_RECOVERY_BEFORE_COVERAGE_REJECT` | 覆盖拒绝前先恢复。C 必须全表分档核对。无凭证不猜密码。`download_attempts=0` 不得 `RECOVERY_COMPLETE` |
| `FORMULA_IDENTITY_PROVENANCE` | Original Prompt vs Frozen vs 实际代码。只标 `PRE-FREEZE_REVISION` / `POST-FREEZE_DEVIATION`，不借机改公式 |
| `ECONOMIC_MATERIALITY_GATE` | 沿用 R21_3_ECONOMIC_MATERIALITY_V1，禁止回调 |
| `FORWARD_AVAILABILITY` | 无独立 Forward 不得 live-ready。未完成不得 `FULL_RESEARCH_COMPLETE` |

决策只允许 `KEEP_BASELINE_V5` 或 `RECOMMEND_V6_RESEARCH_CANDIDATE`。`AUTO_PROMOTE=False`。`CURRENT_BASELINE_CHANGED=FALSE`。
"""


def update_standard() -> dict[str, Any]:
    text = STANDARD_MD.read_text(encoding="utf-8") if STANDARD_MD.is_file() else "# Alpha Factor Research Standard\n"
    if STANDARD_MARKER in text:
        head, _ = text.split(STANDARD_MARKER, 1)
        text = head.rstrip() + STANDARD_ADDENDUM
    else:
        text = text.rstrip() + STANDARD_ADDENDUM
    STANDARD_MD.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    write_md(MD["18"], "# Round22.1 · 18 STANDARD\n\n已追加 Round 22.1 强制增量：禁止静默 skip；IC 必须写宇宙/n_pairs/label；回测必须可追踪 Top1/Top2；覆盖拒绝前先恢复；公式身份；未完成不得 FULL_RESEARCH_COMPLETE。\n")
    return {"path": str(STANDARD_MD), "extra_steps": list(FACTORY_STEPS_EXTRA)}


def update_evaluate_factor() -> dict[str, Any]:
    path = Path(EVALUATE_PY)
    src = path.read_text(encoding="utf-8")
    changed = False
    if "from core.round22.contract import FACTORY_STEPS_EXTRA as R22_EXTRA" in src and "R221_EXTRA" not in src:
        src = src.replace(
            "    from core.round22.contract import FACTORY_STEPS_EXTRA as R22_EXTRA\n",
            "    from core.round22.contract import FACTORY_STEPS_EXTRA as R22_EXTRA\n"
            "    from core.round22_1.contract import FACTORY_STEPS_EXTRA as R221_EXTRA\n",
        )
        changed = True
    if "R221_EXTRA" in src and "*R22_EXTRA]))" in src and "*R221_EXTRA" not in src:
        src = src.replace(
            "        extra = list(dict.fromkeys([*FACTORY_STEPS_EXTRA, *R212_EXTRA, *R213_EXTRA, *R214_EXTRA, *R22_EXTRA]))",
            "        extra = list(dict.fromkeys([*FACTORY_STEPS_EXTRA, *R212_EXTRA, *R213_EXTRA, *R214_EXTRA, *R22_EXTRA, *R221_EXTRA]))",
        )
        changed = True
    needle = """                from core.round22.evaluate import evaluate_factor as evaluate_factor_22

                try:
                    payload = evaluate_factor_22(args.factor_id, through=args.through, persist=args.persist_gate)
                except KeyError:
                    print(f"UNKNOWN_FACTOR:{args.factor_id}", flush=True)
                    return 2"""
    repl = """                from core.round22.evaluate import evaluate_factor as evaluate_factor_22

                try:
                    payload = evaluate_factor_22(args.factor_id, through=args.through, persist=args.persist_gate)
                except KeyError:
                    from core.round22_1.evaluate import evaluate_factor as evaluate_factor_221

                    try:
                        payload = evaluate_factor_221(args.factor_id, through=args.through, persist=args.persist_gate)
                    except KeyError:
                        print(f"UNKNOWN_FACTOR:{args.factor_id}", flush=True)
                        return 2"""
    if "evaluate_factor_221" not in src and needle in src:
        src = src.replace(needle, repl)
        changed = True
    if changed:
        path.write_text(src, encoding="utf-8")
    return {"path": str(path), "changed": changed, "has_221": "evaluate_factor_221" in path.read_text(encoding="utf-8")}
