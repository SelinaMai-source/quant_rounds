#!/usr/bin/env python3
"""Round 22.6 入口。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.round22_6.pipeline import STAGES, run_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", *STAGES.keys()])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--max-memory-gb", type=float, default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print({"stage": args.stage, "resume": args.resume, "jobs": args.jobs, "dry_run": True})
        return
    if args.stage == "all":
        out = run_all(resume=args.resume, jobs=args.jobs, experiment_id=args.experiment_id)
    else:
        out = {args.stage: STAGES[args.stage](resume=args.resume, jobs=args.jobs, experiment_id=args.experiment_id)}
    print({k: (v.get("status") or v.get("verdict") or v.get("decision") or "OK") for k, v in out.items()})
    if args.publish:
        from tools.publish_round_branch import publish_round

        publish_round("round22_6")


if __name__ == "__main__":
    main()
