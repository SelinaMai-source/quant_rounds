#!/usr/bin/env python3
"""Round 23.5 入口。最多两个训练任务。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.round23_5.pipeline import run_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all")
    parser.add_argument("--paper-id", default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--max-memory-gb", type=float, default=16.0)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    jobs = max(1, min(2, int(args.jobs)))
    if args.stage not in {"all", "literature", "experiments"}:
        raise SystemExit(f"unsupported stage {args.stage}")
    payload = run_all(
        resume=True if args.resume or args.stage == "all" else False,
        jobs=jobs,
        paper_id=args.paper_id,
        experiment_id=args.experiment_id,
    )
    payload["jobs"] = jobs
    payload["max_memory_gb"] = args.max_memory_gb
    payload["publish"] = bool(args.publish)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
