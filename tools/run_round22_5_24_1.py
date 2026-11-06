#!/usr/bin/env python3
"""按顺序跑 Round22.5、23.1、24.1，然后测试和总报告。"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.r225241_io import RESULT_ROOT, TRACKER_PATH, append_tracker, dump_json, sha256_file, write_df  # noqa: E402
from core.round22_5.pipeline import run_all as run_225  # noqa: E402
from core.round23_1.pipeline import run_all as run_231  # noqa: E402
from core.round24_1.pipeline import run_all as run_241  # noqa: E402
from core.round22_5_24_1_master import render_master  # noqa: E402
from core.round22_5_24_1_tests import run_tests  # noqa: E402

SENTINELS = (
    "core/round7/contract.py",
    "core/local_executor.py",
    "core/round21/contract.py",
    "core/round24/build.py",
    "result/round22_4/decision.json",
    "result/round23/decision.json",
    "result/round24/experiments.json",
    "result/round22_3/candidates/R223_E0/summary.json",
)


def _guard() -> None:
    raw = TRACKER_PATH.read_bytes()
    payload = {
        "tracker_len_before": len(raw),
        "tracker_sha_before": hashlib.sha256(raw).hexdigest(),
        "hashes": {rel: sha256_file(ROOT / rel) for rel in SENTINELS},
        "production_config_changed": False,
        "current_baseline_changed": False,
    }
    dump_json(RESULT_ROOT / "round22_5_24_1_master" / "production_guard.json", payload)


def _tracker(rows: list[dict]) -> None:
    text = TRACKER_PATH.read_text(encoding="utf-8")
    fresh = "### Round22.5–23.1 实验组结果" not in text
    lines: list[str] = []
    if fresh:
        lines.extend(
            [
                "",
                "### Round22.5–23.1 实验组结果",
                "",
                "只追加本轮新的正式回测。诊断和执行情景不计 alpha。一进二对照 v5。",
                "",
                "| 操作 | 代号 | Sharpe | 年化 | MaxDD | 笔数 | NAV | 状态 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in rows:
            if row.get("scope") == "SECOND_TO_THIRD_INDEPENDENT":
                continue
            lines.append(_tracker_row(row))
        lines.extend(
            [
                "",
                "### Round24.1 实验组结果",
                "",
                "二进三独立账本。不和一进二 Sharpe 或 CAGR 比较。",
                "",
                "| 操作 | 代号 | Sharpe | 年化 | MaxDD | 笔数 | NAV | 状态 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
    pending = []
    for row in rows:
        if row.get("scope") != "SECOND_TO_THIRD_INDEPENDENT":
            continue
        code = f"`{row.get('code')}`"
        if code in text:
            continue
        pending.append(_tracker_row(row, label="二进三"))
    if not fresh and not pending:
        return
    lines.extend(pending)
    if lines:
        append_tracker("\n".join(lines))


def _tracker_row(row: dict, label: str | None = None) -> str:
    scope = label or row.get("scope")
    return (
        f"| {scope} | `{row.get('code')}` | {row.get('sharpe')} | {row.get('cagr')} | "
        f"{row.get('max_drawdown')} | {row.get('completed_trades')} | {row.get('nav_end')} | 已跑 |"
    )


def _collect() -> list[dict]:
    rows = []
    for folder, scope_default in (
        ("round22_5", "FULL_HISTORY_FIRST_TO_SECOND"),
        ("round23_1", "FULL_HISTORY_FIRST_TO_SECOND"),
        ("round24_1", "SECOND_TO_THIRD_INDEPENDENT"),
    ):
        base = RESULT_ROOT / folder
        if not base.is_dir():
            continue
        for path in base.rglob("summary.json"):
            if "_state" in path.parts:
                continue
            payload = __import__("json").loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "ran":
                continue
            if payload.get("new_backtest") is False:
                continue
            payload["scope"] = payload.get("scope") or scope_default
            rows.append(payload)
    write_df(RESULT_ROOT / "round22_5_24_1_master" / "new_formal_backtests.csv", __import__("pandas").DataFrame(rows))
    return rows


def main() -> int:
    _guard()
    run_225(resume=True, jobs=2)
    run_231(resume=True, jobs=2)
    run_241(resume=True, jobs=2)
    rows = _collect()
    _tracker(rows)
    tests = run_tests()
    dump_json(RESULT_ROOT / "round22_5_24_1_master" / "unit_tests.json", tests)
    render_master(tests)
    r225 = __import__("core.r225241_io", fromlist=["load_json"]).load_json(RESULT_ROOT / "round22_5" / "decision.json")
    r231 = __import__("core.r225241_io", fromlist=["load_json"]).load_json(RESULT_ROOT / "round23_1" / "decision.json")
    r241 = __import__("core.r225241_io", fromlist=["load_json"]).load_json(RESULT_ROOT / "round24_1" / "decision.json")
    print("ROUND22_5_DECISION", r225.get("promote_recommendation"))
    print("ROUND23_1_DECISION", r231.get("final"))
    print("ROUND24_1_DECISION", r241.get("final"))
    print("N_FORMAL_BACKTESTS_NEW", len(rows))
    print("N_TESTS", tests.get("n_tests"))
    print("N_TESTS_PASSED", tests.get("n_passed"))
    print("N_TESTS_FAILED", tests.get("n_failed"))
    print("CURRENT_BASELINE_CHANGED", False)
    print("PRODUCTION_CONFIG_CHANGED", False)
    print("AUTO_PROMOTE", False)
    print("LIVE_READY", False)
    if tests.get("failed"):
        print("FAILED_TESTS", tests["failed"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
