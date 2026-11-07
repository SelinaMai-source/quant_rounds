#!/usr/bin/env python3
"""顺序跑 R22.6 → R23.2 → R24.2，每轮后可 --publish。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.research_integrity_v2.io_v2 import dump_json, write_text  # noqa: E402
from core.paths import RESULT_ROOT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--from-round", default="22.6", choices=["22.6", "23.2", "24.2"])
    args = parser.parse_args()
    order = ["22.6", "23.2", "24.2"]
    start = order.index(args.from_round)
    results = {}
    for name in order[start:]:
        if name == "22.6":
            from core.round22_6.pipeline import run_all as run
        elif name == "23.2":
            from core.round23_2.pipeline import run_all as run
        else:
            from core.round24_2.pipeline import run_all as run
        print(f"===== RUN ROUND {name} =====", flush=True)
        results[name] = run(resume=args.resume, jobs=args.jobs)
        if args.publish:
            from tools.publish_round_branch import publish_round

            branch = {"22.6": "round22_6", "23.2": "round23_2", "24.2": "round24_2"}[name]
            pub = publish_round(branch)
            results[f"publish_{branch}"] = pub
            print(json.dumps({"published": branch, "verified": pub.get("verified"), "remote_sha": pub.get("remote_sha")}, ensure_ascii=False), flush=True)
            if not pub.get("verified"):
                print("PUBLISH_FAILED", pub.get("push_stderr"), flush=True)
                break
    # master report
    master = RESULT_ROOT / "round22_6_24_2_master"
    master.mkdir(parents=True, exist_ok=True)
    d226 = json.loads((RESULT_ROOT / "round22_6" / "decision.json").read_text()) if (RESULT_ROOT / "round22_6" / "decision.json").is_file() else {}
    d232 = json.loads((RESULT_ROOT / "round23_2" / "decision.json").read_text()) if (RESULT_ROOT / "round23_2" / "decision.json").is_file() else {}
    d242 = json.loads((RESULT_ROOT / "round24_2" / "decision.json").read_text()) if (RESULT_ROOT / "round24_2" / "decision.json").is_file() else {}
    body = [
        "# Round22.6–24.2 总研究报告",
        "",
        "## 范围",
        "一进二锁定基线 R16_O01。二进三独立。CURRENT 未改。",
        "",
        "## 决策摘要",
        f"- Round22.6: `{d226.get('decision')}` FS2=`{d226.get('fs2_verdict')}` Challenger=`{d226.get('challenger_verdict')}`",
        f"- Round23.2: `{d232.get('decision')}` endpoint=`{d232.get('endpoint_parity')}`",
        f"- Round24.2: `{d242.get('decision')}` best=`{d242.get('best_candidate')}`",
        "",
        "CURRENT_BASELINE_CHANGED=FALSE",
        "PRODUCTION_CONFIG_CHANGED=FALSE",
        "",
    ]
    write_text(master / "MASTER_RESEARCH_REPORT.md", "\n".join(body))
    dump_json(master / "run_summary.json", {"rounds": {k: str(type(v)) for k, v in results.items()}})
    print("MASTER_WRITTEN", master / "MASTER_RESEARCH_REPORT.md")


if __name__ == "__main__":
    main()
