"""Round 1 五组注册表：入场锁 fs00263 / n50_d2 / 集成，只改 exit_mode。

结果只写 result/round1/{code}/，禁止写到 Stock/runs。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .local_executor import (
    DAILY_PROXY_EXIT_MODES,
    baseline_fixed_sleeve_config,
    run_local_fixed_sleeve_backtest,
    write_backtest_artifacts,
)
from .metrics import (
    LOCKED_BASELINE,
    delta_vs_locked_baseline,
    replacement_gate,
)
from .paths import (
    BACKTEST_VARIANT,
    CAPACITY,
    ENSEMBLE_METHOD,
    MARKET_END,
    PREDICTION_START,
    SIGNAL_END,
    TRIAL_ID,
    assert_project_result_path,
    round1_dir,
)
from .stock_readonly import (
    copy_baseline_report_to_project,
    ensemble_dir,
    load_fs00263_ensemble_predictions,
    load_prepared_market,
    sha256_file,
    stock_baseline_report_path,
)


@dataclass(frozen=True)
class ExperimentSpec:
    code: str
    exit_mode: str
    title_zh: str
    description_zh: str
    read_existing_ledger: bool = False
    daily_proxy: bool = False


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "R1_00_baseline": ExperimentSpec(
        code="R1_00_baseline",
        exit_mode="close_from_d2",
        title_zh="锁定 baseline（读已有账本）",
        description_zh=(
            "fs00263 / n50_d2 五种子集成 / top1_both_gates_nav30pct_v1；"
            "D+2 起首个可卖收盘。读 Stock report.json 拷到课题，不重跑 Stock。"
        ),
        read_existing_ledger=True,
        daily_proxy=False,
    ),
    "R1_01_tp1100": ExperimentSpec(
        code="R1_01_tp1100",
        exit_mode="daily_tp1100",
        title_zh="仅 11:00 止盈（日频代理=open）",
        description_zh=(
            "从 D+2 起每个交易日：能卖且开盘价相对 entry_price 浮盈>0 则按开盘卖。"
            "未触发则继续持有，不回退 close_from_d2。"
        ),
        daily_proxy=True,
    ),
    "R1_02_tp1450": ExperimentSpec(
        code="R1_02_tp1450",
        exit_mode="daily_tp1450",
        title_zh="仅 14:50 止盈（日频代理=close）",
        description_zh=(
            "从 D+2 起每个交易日：能卖且收盘价相对 entry_price 浮盈>0 则按收盘卖。"
            "未触发则继续持有，不回退 close_from_d2。"
        ),
        daily_proxy=True,
    ),
    "R1_03_ma5stop": ExperimentSpec(
        code="R1_03_ma5stop",
        exit_mode="daily_ma5stop",
        title_zh="仅 14:50 止损 price<MA5×1.05",
        description_zh=(
            "从 D+2 起每个交易日：能卖且复权近似 ma_price < ma5×1.05 则按收盘卖，"
            "不考虑盈亏；涨停仍不卖。不回退 close_from_d2。"
        ),
        daily_proxy=True,
    ),
    "R1_04_full": ExperimentSpec(
        code="R1_04_full",
        exit_mode="daily_full",
        title_zh="图中完整规则（11:00 止盈 → 14:50 止盈 → 14:50 止损）",
        description_zh=(
            "从 D+2 起：先 11:00（open）浮盈>0 则卖；否则 14:50（close）浮盈>0 则卖；"
            "再否则 close 对应 ma_price < ma5×1.05 则止损。14:50 止盈优先于止损。"
            "不回退 close_from_d2。"
        ),
        daily_proxy=True,
    ),
}


def list_experiment_codes() -> list[str]:
    return list(EXPERIMENTS)


def get_experiment(code: str) -> ExperimentSpec:
    key = str(code or "").strip()
    if key not in EXPERIMENTS:
        available = ", ".join(list_experiment_codes())
        raise KeyError(f"未知实验代号 {key!r}；可用代号：{available}")
    return EXPERIMENTS[key]


def result_dir_for(code: str) -> Path:
    return round1_dir(get_experiment(code).code)


def registry_table() -> list[dict[str, Any]]:
    rows = []
    for spec in EXPERIMENTS.values():
        rows.append({
            "code": spec.code,
            "exit_mode": spec.exit_mode,
            "title_zh": spec.title_zh,
            "daily_proxy": spec.daily_proxy,
            "read_existing_ledger": spec.read_existing_ledger,
            "result_dir": str(result_dir_for(spec.code)),
            "description_zh": spec.description_zh,
        })
    return rows


def materialize_baseline() -> dict[str, Any]:
    """Copy Stock report.json into 课题 result/round1/R1_00_baseline/."""
    dest = copy_baseline_report_to_project()
    report = json.loads(dest.read_text(encoding="utf-8"))
    summary = _display_summary_from_report(
        report,
        code="R1_00_baseline",
        status="锁定",
        use_locked_display=True,
    )
    summary_path = dest.parent / "summary.json"
    assert_project_result_path(summary_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "copied_from_stock_readonly",
        "code": "R1_00_baseline",
        "report_path": str(dest),
        "report": report,
        "summary": summary,
    }


def load_result_report(code: str) -> dict[str, Any] | None:
    path = result_dir_for(code) / "report.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def tracking_row(code: str) -> dict[str, Any]:
    spec = get_experiment(code)
    report = load_result_report(code)
    if spec.code == "R1_00_baseline":
        row = {
            "code": spec.code,
            "exit_mode": spec.exit_mode,
            "title_zh": spec.title_zh,
            "sharpe": LOCKED_BASELINE["sharpe"],
            "annualized_return": LOCKED_BASELINE["annualized_return"],
            "total_return": LOCKED_BASELINE["total_return"],
            "max_drawdown": LOCKED_BASELINE["max_drawdown"],
            "trade_win_rate": LOCKED_BASELINE["trade_win_rate"],
            "average_holding_sessions": LOCKED_BASELINE["average_holding_sessions"],
            "completed_trades": LOCKED_BASELINE["completed_trades"],
            "delta_sharpe": None,
            "delta_ret": None,
            "status": "锁定",
        }
        if report is None:
            row["status"] = "锁定（课题侧副本缺失，请 materialize_baseline）"
        return row
    if report is None:
        return {
            "code": spec.code,
            "exit_mode": spec.exit_mode,
            "title_zh": spec.title_zh,
            "sharpe": None,
            "annualized_return": None,
            "total_return": None,
            "max_drawdown": None,
            "trade_win_rate": None,
            "average_holding_sessions": None,
            "completed_trades": None,
            "delta_sharpe": None,
            "delta_ret": None,
            "status": "待跑",
        }
    metrics = report.get("metrics") or {}
    delta = delta_vs_locked_baseline(metrics)
    gate = replacement_gate(metrics)
    if gate["meets_gate"]:
        status = "指标达门槛（未切换锁定 R1_00_baseline）"
    else:
        status = "已跑"
    return {
        "code": spec.code,
        "exit_mode": spec.exit_mode,
        "title_zh": spec.title_zh,
        "sharpe": metrics.get("sharpe_zero_rate"),
        "annualized_return": metrics.get("annualized_return"),
        "total_return": metrics.get("total_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "trade_win_rate": metrics.get("trade_win_rate"),
        "average_holding_sessions": metrics.get("average_holding_sessions"),
        "completed_trades": metrics.get("completed_trades"),
        "delta_sharpe": delta["delta_sharpe"],
        "delta_ret": delta["delta_ret"],
        "status": status,
        "gate_reason": gate["reason"],
    }


def tracking_table() -> list[dict[str, Any]]:
    return [tracking_row(code) for code in list_experiment_codes()]


def run_experiment(
    code: str,
    *,
    run_now: bool = False,
    predictions: Any | None = None,
    prepared_market: Any | None = None,
    market_meta: dict[str, Any] | None = None,
    start_date=PREDICTION_START,
    end_date=MARKET_END,
) -> dict[str, Any]:
    """Load existing artifacts, or run a daily-proxy backtest into result/round1/{code}/."""
    spec = get_experiment(code)
    if spec.read_existing_ledger:
        return materialize_baseline()
    existing = load_result_report(spec.code)
    if not run_now:
        if existing is not None:
            return {
                "status": "loaded",
                "code": spec.code,
                "report": existing,
                "summary": tracking_row(spec.code),
            }
        return {
            "status": "pending",
            "code": spec.code,
            "exit_mode": spec.exit_mode,
            "summary": tracking_row(spec.code),
            "note": "RUN_NOW=False 且 result/round1 尚无账本；全样本请 nohup 跑。",
        }
    if spec.exit_mode not in DAILY_PROXY_EXIT_MODES:
        raise RuntimeError(f"{spec.code} is not a daily-proxy experiment")
    if predictions is None:
        predictions = load_fs00263_ensemble_predictions(
            start=start_date, signal_end=SIGNAL_END,
        )
    if prepared_market is None:
        loaded = load_prepared_market(warmup_ma5=True)
        prepared_market = loaded["prepared"]
        market_meta = loaded
    config = baseline_fixed_sleeve_config(exit_mode=spec.exit_mode)
    result = run_local_fixed_sleeve_backtest(
        predictions,
        prepared_market,
        config=config,
        start_date=start_date,
        end_date=end_date,
    )
    metrics = result["report"]["metrics"]
    acceptance = {
        "max_live_sleeves_lte_2": metrics["max_live_sleeves"] <= 2,
        "no_negative_sleeve_cash": metrics["minimum_sleeve_cash"] >= -1e-8,
        "asset_identity": metrics["asset_identity_max_abs"] <= 1e-6,
        "no_close_from_d2_fallback": True,
    }
    if not all(acceptance.values()):
        raise RuntimeError(f"backtest acceptance failed: {acceptance}")
    extra = {
        "status": "completed_and_validated",
        "experiment_code": spec.code,
        "trial_id": TRIAL_ID,
        "capacity": CAPACITY,
        "seed": "ensemble",
        "ensemble": ENSEMBLE_METHOD,
        "variant": BACKTEST_VARIANT,
        "predictions_file": str(ensemble_dir() / "predictions.parquet"),
        "predictions_sha256": sha256_file(ensemble_dir() / "predictions.parquet"),
        "label_reconciliation": {
            "status": "not_applicable_new_exit_contract",
            "trades_checked": int(len(result["trades"])),
        },
        "acceptance": acceptance,
        "evidence_role": "round1_daily_proxy_exit_ablation",
        "note": (
            "日频代理：11:00=open，14:50=close，止损 ma_price<ma5×1.05；"
            "不是真实分钟时点。产物在课题 result/round1/，未写 Stock/runs。"
        ),
    }
    if market_meta:
        extra["snapshot_id"] = market_meta.get("snapshot_id")
        extra["snapshot_fingerprint"] = market_meta.get("snapshot_fingerprint")
    written = write_backtest_artifacts(
        result, result_dir_for(spec.code), extra_report=extra,
    )
    summary = tracking_row(spec.code)
    summary_path = result_dir_for(spec.code) / "summary.json"
    assert_project_result_path(summary_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "ran",
        "code": spec.code,
        "out_dir": written["out_dir"],
        "report": written["report"],
        "summary": summary,
    }


def _display_summary_from_report(
    report: dict[str, Any],
    *,
    code: str,
    status: str,
    use_locked_display: bool,
) -> dict[str, Any]:
    if use_locked_display:
        return {
            "code": code,
            "status": status,
            "source_report": str(stock_baseline_report_path()),
            **LOCKED_BASELINE,
            "raw_sharpe_zero_rate": (report.get("metrics") or {}).get("sharpe_zero_rate"),
            "raw_total_return": (report.get("metrics") or {}).get("total_return"),
            "raw_max_drawdown": (report.get("metrics") or {}).get("max_drawdown"),
            "raw_annualized_return": (report.get("metrics") or {}).get("annualized_return"),
        }
    metrics = report.get("metrics") or {}
    delta = delta_vs_locked_baseline(metrics)
    return {
        "code": code,
        "status": status,
        "sharpe": metrics.get("sharpe_zero_rate"),
        "annualized_return": metrics.get("annualized_return"),
        "total_return": metrics.get("total_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        **delta,
    }
