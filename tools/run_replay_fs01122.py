#!/usr/bin/env python3
"""薄封装：转发到 scripts/run_replay_fs01122.py。"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_replay_fs01122.py"


def main() -> int:
    sys.argv = [str(SCRIPT), *sys.argv[1:]]
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
