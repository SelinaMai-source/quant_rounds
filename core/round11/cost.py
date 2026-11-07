"""HISTORICAL_FEE_CORRECTED 重放 + 买侧压力。产物只写 result/round11/cost/。"""

from __future__ import annotations

from typing import Any

from ..local_executor import (
    HISTORICAL_POST_CUTOVER_SELL_TAX_BPS,
    HISTORICAL_PRE_CUTOVER_SELL_TAX_BPS,
    STAMP_TAX_CUTOVER,
)
from ..paths import ROUND11_ROOT
from ..replay_round7.inject import inject_round7_into_overlay_caches
from ..stock_readonly import load_prepared_market
from .common import dump_json, pct, stage_dir, write_md
from .contract import BASELINES
from .runner import run_spec_to

COST_MD = ROUND11_ROOT / "04_cost_corrected.md"
BUY_STRESS_BPS = (10, 20, 30, 50)
FEE_HISTORY_NOTE = (
    "佣金 1bp / 过户 0.1bp / 滑点 12.3bp 无独立历史费率表可核。"
    "标 FEE_HISTORY_UNVERIFIED，不编造。"
)


def _run_one(
    key: str,
    *,
    prepared_market,
    market_meta,
    resume: bool,
    dest_name: str,
    validation_hooks: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    spec = BASELINES[key]
    dest = stage_dir("cost") / dest_name
    ran = run_spec_to(
        spec,
        dest,
        prepared_market=prepared_market,
        market_meta=market_meta,
        resume=resume,
        validation_hooks=validation_hooks,
        extra_report=extra,
    )
    report = ran.get("report") or {}
    metrics = report.get("metrics") or {}
    return {
        "version": key,
        "code": spec.code,
        "out_dir": str(dest),
        "status": ran.get("status"),
        "cagr": metrics.get("annualized_return"),
        "sharpe": metrics.get("sharpe_zero_rate"),
        "max_drawdown": metrics.get("max_drawdown"),
        "completed_trades": metrics.get("completed_trades"),
        "total_return": metrics.get("total_return"),
        "loyo_median": metrics.get("loyo_median"),
        "loyo_worst": metrics.get("loyo_worst"),
        "yearly_returns": metrics.get("yearly_returns") or {},
        "fee_schedule": report.get("fee_schedule") or extra.get("fee_schedule"),
        "nav_path": str(dest / "nav.parquet"),
    }


def run_cost(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    inject_round7_into_overlay_caches()
    loaded = load_prepared_market(warmup_ma5=True)
    prepared_market = loaded["prepared"]
    market_meta = loaded
    dest = stage_dir("cost")
    corrected: dict[str, Any] = {}
    stress: dict[str, Any] = {}
    hist_hooks = {"historical_stamp_tax": True}
    for key in ("v1", "v2", "v3", "v4"):
        corrected[key] = _run_one(
            key,
            prepared_market=prepared_market,
            market_meta=market_meta,
            resume=resume,
            dest_name=f"R11_{key.upper()}_CORRECTED",
            validation_hooks=hist_hooks,
            extra={
                "fee_schedule": "HISTORICAL_FEE_CORRECTED",
                "commission_transfer_status": "FEE_HISTORY_UNVERIFIED",
                "stamp_tax_rule": (
                    f"sell_date < {STAMP_TAX_CUTOVER.date()} → "
                    f"{HISTORICAL_PRE_CUTOVER_SELL_TAX_BPS}bp else "
                    f"{HISTORICAL_POST_CUTOVER_SELL_TAX_BPS}bp"
                ),
            },
        )
    for bps in BUY_STRESS_BPS:
        for key in ("v1", "v2", "v3", "v4"):
            tag = f"R11_{key.upper()}_CORR_BUY{bps:02d}"
            stress[tag] = _run_one(
                key,
                prepared_market=prepared_market,
                market_meta=market_meta,
                resume=resume,
                dest_name=tag,
                validation_hooks={**hist_hooks, "buy_extra_slippage_bps": float(bps)},
                extra={
                    "fee_schedule": "HISTORICAL_FEE_CORRECTED",
                    "buy_extra_slippage_bps": bps,
                    "commission_transfer_status": "FEE_HISTORY_UNVERIFIED",
                },
            )
    v4c = corrected["v4"]
    payload = {
        "status": "DONE",
        "fee_schedule": "HISTORICAL_FEE_CORRECTED",
        "cutover": str(STAMP_TAX_CUTOVER.date()),
        "pre_bps": HISTORICAL_PRE_CUTOVER_SELL_TAX_BPS,
        "post_bps": HISTORICAL_POST_CUTOVER_SELL_TAX_BPS,
        "commission_transfer": "FEE_HISTORY_UNVERIFIED",
        "commission_transfer_note": FEE_HISTORY_NOTE,
        "legacy_sell_tax_bps": 5.0,
        "corrected": corrected,
        "buy_stress": stress,
        "focus_v4_corrected": {
            "cagr": v4c.get("cagr"),
            "sharpe": v4c.get("sharpe"),
            "max_drawdown": v4c.get("max_drawdown"),
        },
        "post_hoc_diagnostic": True,
    }
    dump_json(dest / "cost.json", payload)
    lines = [
        "# Round 11 · 历史印花税修正 / 买侧压力",
        "",
        "卖出日 `<2023-08-28` → **10bp**，否则 **5bp**。只在课题 `local_executor` validation hook。",
        "不改 Stock 执行器，不覆盖 legacy。佣金/过户/滑点 = **FEE_HISTORY_UNVERIFIED**。",
        "",
        "## 关键：corrected-cost v4",
        "",
        f"- CAGR = **{pct(v4c.get('cagr'))}**（精确 `{v4c.get('cagr')}`）",
        f"- Sharpe = **{v4c.get('sharpe')}**",
        f"- MaxDD = **{pct(v4c.get('max_drawdown'))}**（精确 `{v4c.get('max_drawdown')}`）",
        f"- 笔数 = {v4c.get('completed_trades')}",
        "",
        "## v1–v4 HISTORICAL_FEE_CORRECTED",
        "",
        "| 版本 | CAGR | Sharpe | MaxDD | 笔数 | 累计 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("v1", "v2", "v3", "v4"):
        row = corrected[key]
        lines.append(
            f"| {key} `{BASELINES[key].code}` | {pct(row.get('cagr'))} | "
            f"{row.get('sharpe')} | {pct(row.get('max_drawdown'))} | "
            f"{row.get('completed_trades')} | {row.get('total_return')} |"
        )
    lines += [
        "",
        "## 买侧 +10/20/30/50bp（叠在 corrected 上）",
        "",
        "| 实验 | CAGR | Sharpe | MaxDD | 笔数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for tag, row in stress.items():
        lines.append(
            f"| `{tag}` | {pct(row.get('cagr'))} | {row.get('sharpe')} | "
            f"{pct(row.get('max_drawdown'))} | {row.get('completed_trades')} |"
        )
    lines += ["", FEE_HISTORY_NOTE, ""]
    write_md(COST_MD, "\n".join(lines))
    return payload


def load_cost_status() -> dict[str, Any]:
    path = ROUND11_ROOT / "cost" / "cost.json"
    if not path.is_file():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["load_cost_status", "run_cost"]
