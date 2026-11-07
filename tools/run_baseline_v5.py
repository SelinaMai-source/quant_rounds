#!/usr/bin/env python3
"""baseline_v5 复现入口。

--dry-run : 只打印契约与权威锚点，不跑引擎
--parity  : 对照已落盘 ROUND7_FLAT_5BP 产物
--tests   : 跑不变量测试
--reproduce : 调用 tools/replay_r7_caliber.py 全量重放（需服务器数据与 Stock 引擎）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--tests", action="store_true")
    parser.add_argument("--reproduce", action="store_true")
    args = parser.parse_args()

    from core.baseline_v5.contract import (
        ALIAS,
        FEE_COMPONENTS,
        FEE_SCHEDULE,
        FIVE_FEATURES,
        FUSION_WEIGHTS,
        IDENTITY,
        INFORMATION_LANE,
        O01_FORMULA,
        TEN_FEATURES,
        V5_ANCHOR,
        frozen_feature_hashes,
    )

    if args.dry_run or not any([args.parity, args.tests, args.reproduce]):
        print(json.dumps({
            "identity": IDENTITY,
            "alias": ALIAS,
            "fee_schedule": FEE_SCHEDULE,
            "fee_components": FEE_COMPONENTS,
            "fusion": FUSION_WEIGHTS,
            "information_lane": INFORMATION_LANE,
            "o01_formula": O01_FORMULA,
            "five_n": len(FIVE_FEATURES),
            "ten_n": len(TEN_FEATURES),
            "feature_hashes": frozen_feature_hashes(),
            "anchor": V5_ANCHOR,
            "note": "正式重放需 --reproduce，并具备服务器面板与 first_board_hermes",
        }, ensure_ascii=False, indent=2))

    if args.tests:
        from core.baseline_v5.tests_invariants import run_all
        result = run_all()
        print(json.dumps({"tests": result}, ensure_ascii=False, indent=2))
        if result["failed"]:
            return 1

    if args.parity:
        from core.baseline_v5.parity import check_parity
        result = check_parity()
        print(json.dumps({"parity": result}, ensure_ascii=False, indent=2))
        if result["status"] != "MATCH":
            return 2

    if args.reproduce:
        script = ROOT / "tools" / "replay_r7_caliber.py"
        if not script.is_file():
            print("missing tools/replay_r7_caliber.py", file=sys.stderr)
            return 3
        # 默认在课题根跑，避免 publish 树缺数据
        proj = Path("/data1/stock-mxy/课题/一进二策略")
        cwd = proj if (proj / "tools" / "replay_r7_caliber.py").is_file() else ROOT
        proc = subprocess.run([sys.executable, str(cwd / "tools" / "replay_r7_caliber.py")], cwd=str(cwd))
        return int(proc.returncode)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
