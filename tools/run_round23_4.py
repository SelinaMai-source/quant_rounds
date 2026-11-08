#!/usr/bin/env python3
"""Round23.4 入口。--resume 只在代码与阶段指纹一致时生效。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.round23_4.pipeline import STAGES, run_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", *STAGES.keys()])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    jobs = max(1, min(int(args.jobs), 2))
    if args.stage == "all":
        out = run_all(resume=args.resume, jobs=jobs, experiment_id=args.experiment_id)
    else:
        out = {args.stage: STAGES[args.stage](resume=args.resume, jobs=jobs, experiment_id=args.experiment_id)}
    if args.publish:
        print({"publish_requested": True, "note": "publish is a separate git step; this flag does not skip gates"})
    print({k: (v.get("status") or v.get("risk_outcome") or "OK") for k, v in out.items()})
    bad = [k for k, v in out.items() if v.get("status") in {"FAIL", "BLOCKED"} and k in {"source_freeze"}]
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
