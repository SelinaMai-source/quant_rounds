"""更新 STANDARD + evaluate_factor。Head-specific 四路强制。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import dump_json
from .contract import FACTORY_STEPS_EXTRA, STANDARD_MD
from ..paths import ROUND21_4_ROOT

STANDARD_ADDENDUM = """
## Round 21.4 强制增量（L1 Head-specific × Forward 修复 × 组合工厂 × W/X/Y/Z）

在 Round 21.3 之上，本轮额外强制：

| 步 | 含义 |
|---|---|
| `HEAD_SPECIFIC_FOUR_WAY` | 新因子必须跑 Five-only / Ten-only / Both-add / Both-replace，禁止只报双头均值 |
| `COMBINATION_FAILURE_ANALYSIS` | 单因子 IC 有效但 Add 变差必须按 8 类归因，给证据 |
| `DATA_RECOVERY_BEFORE_COVERAGE_REJECT` | Forward `_max_date` 必须扫全部分区；覆盖不足走 DISCOVER→RECOVER→VALIDATE→INTEGRATE→RETEST |
| `ECONOMIC_MATERIALITY_GATE` | 沿用 21.3，禁止回调。+0.003/+43 预先不达标 |

组合方法在 `research/factor_combination/`。C0–C11 不预设优于 Raw。无独立 Forward 最多 `V6_RESEARCH_CANDIDATE`。`AUTO_PROMOTE=False`。`CURRENT_BASELINE_CHANGED=FALSE`。
"""

FACTORY_JSON = ROUND21_4_ROOT / "configs" / "factory.json"


def update_standard() -> dict:
    text = STANDARD_MD.read_text(encoding="utf-8") if STANDARD_MD.is_file() else "# Alpha Factor Research Standard\n"
    marker = "## Round 21.4 强制增量（L1 Head-specific × Forward 修复 × 组合工厂 × W/X/Y/Z）"
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
    if "from core.round21_4.contract import FACTORY_STEPS_EXTRA as R214_EXTRA" not in src:
        if "from core.round21_3.contract import FACTORY_STEPS_EXTRA as R213_EXTRA" in src:
            src = src.replace(
                "    from core.round21_3.contract import FACTORY_STEPS_EXTRA as R213_EXTRA\n",
                "    from core.round21_3.contract import FACTORY_STEPS_EXTRA as R213_EXTRA\n"
                "    from core.round21_4.contract import FACTORY_STEPS_EXTRA as R214_EXTRA\n",
            )
        else:
            src = src.replace(
                "    from core.round21_2.contract import FACTORY_STEPS_EXTRA as R212_EXTRA\n",
                "    from core.round21_2.contract import FACTORY_STEPS_EXTRA as R212_EXTRA\n"
                "    from core.round21_4.contract import FACTORY_STEPS_EXTRA as R214_EXTRA\n",
            )
        src = src.replace(
            "        extra = list(dict.fromkeys([*FACTORY_STEPS_EXTRA, *R212_EXTRA, *R213_EXTRA]))\n",
            "        extra = list(dict.fromkeys([*FACTORY_STEPS_EXTRA, *R212_EXTRA, *R213_EXTRA, *R214_EXTRA]))\n",
        )
        if "R213_EXTRA]))" in src and "R214_EXTRA" not in src.split("extra = list")[-1][:200]:
            src = src.replace(
                "        extra = list(dict.fromkeys([*FACTORY_STEPS_EXTRA, *R212_EXTRA]))\n",
                "        extra = list(dict.fromkeys([*FACTORY_STEPS_EXTRA, *R212_EXTRA, *R214_EXTRA]))\n",
            )
        changed = True
    if "HEAD_SPECIFIC_FOUR_WAY" not in src:
        src = src.replace(
            "DATA_RECOVERY_BEFORE_COVERAGE_REJECT → FEE_LEDGER_COMPLETENESS →\n"
            "BOOTSTRAP_SPA_NUMERICAL → SEED_STATUS_ENUM → ECONOMIC_MATERIALITY_GATE → FORWARD_AVAILABILITY",
            "DATA_RECOVERY_BEFORE_COVERAGE_REJECT → FEE_LEDGER_COMPLETENESS →\n"
            "BOOTSTRAP_SPA_NUMERICAL → SEED_STATUS_ENUM → ECONOMIC_MATERIALITY_GATE → FORWARD_AVAILABILITY →\n"
            "HEAD_SPECIFIC_FOUR_WAY → COMBINATION_FAILURE_ANALYSIS",
        )
        changed = True
    if "from core.round21_4.evaluate import evaluate_factor as evaluate_factor_214" not in src:
        src = src.replace(
            "            from core.round21_3.evaluate import evaluate_factor as evaluate_factor_213\n\n"
            "            try:\n"
            "                payload = evaluate_factor_213(args.factor_id, through=args.through, persist=args.persist_gate)\n"
            "            except KeyError:\n"
            "                print(f\"UNKNOWN_FACTOR:{args.factor_id}\", flush=True)\n"
            "                return 2",
            "            from core.round21_3.evaluate import evaluate_factor as evaluate_factor_213\n\n"
            "            try:\n"
            "                payload = evaluate_factor_213(args.factor_id, through=args.through, persist=args.persist_gate)\n"
            "            except KeyError:\n"
            "                from core.round21_4.evaluate import evaluate_factor as evaluate_factor_214\n\n"
            "                try:\n"
            "                    payload = evaluate_factor_214(args.factor_id, through=args.through, persist=args.persist_gate)\n"
            "                except KeyError:\n"
            "                    print(f\"UNKNOWN_FACTOR:{args.factor_id}\", flush=True)\n"
            "                    return 2",
        )
        changed = True
    if changed:
        path.write_text(src, encoding="utf-8")
    return {"path": str(path), "extra_steps": list(FACTORY_STEPS_EXTRA), "updated": changed}


def run_factory(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and FACTORY_JSON.is_file() and MD_OK():
        existing = dump_json  # placeholder
        from .common import load_json

        existing = load_json(FACTORY_JSON)
        if existing.get("status") == "DONE":
            return existing
    std = update_standard()
    ev = update_evaluate_factor()
    payload = {"status": "DONE", "standard": std, "evaluate_factor": ev, "head_specific": True}
    dump_json(FACTORY_JSON, payload)
    from .common import write_md
    from .contract import MD

    write_md(
        ROUND21_4_ROOT / "21_factory.md",
        "# Round21.4 · Factory\n\nHead-specific 四路强制。STANDARD 已追加 21.4 增量。evaluate_factor 接入 R214_EXTRA。\n",
    )
    write_md(MD["00"].parent / "21_evaluate_factor.md", "# Round21.4 evaluate_factor\n\nHEAD_SPECIFIC_FOUR_WAY 强制。\n")
    return payload


def MD_OK() -> bool:
    return (ROUND21_4_ROOT / "21_factory.md").is_file()
