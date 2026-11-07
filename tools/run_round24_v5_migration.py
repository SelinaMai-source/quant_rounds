#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.round24_v5_migration.pipeline import STAGES, run_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", *STAGES.keys()])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--max-memory-gb", type=float, default=None)
    parser.add_argument("--experiment-id", default=None)
    args = parser.parse_args()
    if args.dry_run:
        print({"stage": args.stage, "dry_run": True, "jobs": args.jobs})
        return
    if args.stage == "all":
        out = run_all(resume=args.resume, jobs=args.jobs, experiment_id=args.experiment_id)
    else:
        out = {args.stage: STAGES[args.stage](resume=args.resume, jobs=args.jobs, experiment_id=args.experiment_id)}
    print({k: (v.get("status") or v.get("decision") or "OK") for k, v in out.items()})


if __name__ == "__main__":
    main()
