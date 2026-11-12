"""R2 增长选择与 R3 风险选择。默认权重不进入“离默认最近”的排序。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.round23_4.contract import (
    BOOT_BLOCK,
    BOOT_LB_Q,
    BOOT_N,
    LOG_GROWTH_TOLERANCE,
    MAXDD_WORSEN,
    MIN_INNER_DAYS,
    MIN_INNER_TRADES,
    UI_BLOCK_WORSE,
    UI_IMPROVE_MIN,
)
from core.research_integrity_v3.risk_metrics import annualized_log_growth, ulcer_index


def block_bootstrap_lb(diff: np.ndarray, *, n_boot: int = BOOT_N, width: int = BOOT_BLOCK, seed: int = 17) -> float:
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n < max(3 * width, width + 1):
        return float("nan")
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=int(np.ceil(n / width)))
        idx = np.concatenate([np.arange(int(s), int(s) + width) % n for s in starts])[:n]
        means[i] = float(diff[idx].mean())
    return float(np.quantile(means, BOOT_LB_Q))


def paired_mean_lb_keep_blocks(block_diffs: list[np.ndarray], *, seed: int = 17) -> float:
    """两块各自环形抽样后再平均，不把重置边界拼成一条价格路径。"""
    usable = [np.asarray(d, dtype=float) for d in block_diffs if d is not None and len(d) >= BOOT_BLOCK]
    if len(usable) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(BOOT_N, dtype=float)
    for i in range(BOOT_N):
        parts = []
        for series in usable:
            n = len(series)
            starts = rng.integers(0, n, size=int(np.ceil(n / BOOT_BLOCK)))
            idx = np.concatenate([np.arange(int(s), int(s) + BOOT_BLOCK) % n for s in starts])[:n]
            parts.append(float(series[idx].mean()))
        means[i] = float(np.mean(parts))
    return float(np.quantile(means, BOOT_LB_Q))


def _ui(returns: pd.Series) -> float:
    wealth = (1.0 + returns.dropna()).cumprod()
    if wealth.empty:
        return float("nan")
    dd = 1.0 - wealth / wealth.cummax()
    return ulcer_index(dd)


def _maxdd(returns: pd.Series) -> float:
    wealth = (1.0 + returns.dropna()).cumprod()
    if wealth.empty:
        return float("nan")
    return float((wealth / wealth.cummax() - 1.0).min())


def choose_r2(records: list[dict[str, Any]], *, default_key: str = "l_0.00") -> dict[str, Any]:
    default = next(r for r in records if r["key"] == default_key)
    if int(default.get("n_days") or 0) < MIN_INNER_DAYS or int(default.get("n_trades") or 0) < MIN_INNER_TRADES:
        return {"chosen": default_key, "reason": "INSUFFICIENT_INNER_SAMPLE_WEIGHT_0", "qualified": []}
    qualified = []
    for rec in records:
        if rec["key"] == default_key:
            continue
        blocks = rec.get("block_mean_deltas") or []
        both_nonneg = len(blocks) == 2 and all(d is not None and float(d) >= 0.0 for d in blocks)
        growth_better = (
            rec.get("mean_log_growth") is not None
            and default.get("mean_log_growth") is not None
            and float(rec["mean_log_growth"]) > float(default["mean_log_growth"])
        )
        boot_ok = rec.get("boot_lb") is not None and np.isfinite(rec["boot_lb"]) and float(rec["boot_lb"]) > 0.0
        dd_ok = True
        for cand_dd, base_dd in zip(rec.get("block_maxdd") or [], default.get("block_maxdd") or []):
            if cand_dd is None or base_dd is None or not np.isfinite(cand_dd) or not np.isfinite(base_dd):
                dd_ok = False
            elif float(cand_dd) < float(base_dd) - MAXDD_WORSEN:
                dd_ok = False
        sample_ok = int(rec.get("n_days") or 0) >= MIN_INNER_DAYS and int(rec.get("n_trades") or 0) >= MIN_INNER_TRADES
        rec["r2_pass"] = bool(sample_ok and both_nonneg and growth_better and boot_ok and dd_ok and not rec.get("window_fail"))
        if rec["r2_pass"]:
            qualified.append(rec)
    if not qualified:
        return {"chosen": default_key, "reason": "FALLBACK_0_NO_QUALIFIED", "qualified": []}
    qualified.sort(key=lambda r: (abs(float(r["w_lgbm"])), -float(r["mean_log_growth"])))
    return {"chosen": qualified[0]["key"], "reason": "QUALIFIED_NEAREST_THEN_GROWTH", "qualified": [r["key"] for r in qualified]}


def choose_r3(records: list[dict[str, Any]], *, default_key: str = "l_0.00") -> dict[str, Any]:
    default = next(r for r in records if r["key"] == default_key)
    if int(default.get("n_days") or 0) < MIN_INNER_DAYS or int(default.get("n_trades") or 0) < MIN_INNER_TRADES:
        return {"chosen": default_key, "reason": "INSUFFICIENT_INNER_SAMPLE_WEIGHT_0", "qualified": []}
    qualified = []
    base_g = default.get("mean_ann_log_growth")
    base_ui = default.get("mean_ui")
    for rec in records:
        if rec["key"] == default_key:
            continue
        g_ok = (
            rec.get("mean_ann_log_growth") is not None
            and base_g is not None
            and np.isfinite(rec["mean_ann_log_growth"])
            and np.isfinite(base_g)
            and float(base_g) - float(rec["mean_ann_log_growth"]) <= LOG_GROWTH_TOLERANCE + 1e-15
        )
        ui_ok = (
            rec.get("mean_ui") is not None
            and base_ui is not None
            and np.isfinite(rec["mean_ui"])
            and np.isfinite(base_ui)
            and float(base_ui) > 0
            and float(rec["mean_ui"]) <= float(base_ui) * (1.0 - UI_IMPROVE_MIN)
        )
        per_ui = True
        for cand_ui, def_ui in zip(rec.get("block_ui") or [], default.get("block_ui") or []):
            if cand_ui is None or def_ui is None or not np.isfinite(cand_ui) or not np.isfinite(def_ui):
                per_ui = False
            elif float(def_ui) == 0.0 and float(cand_ui) > 0.0:
                per_ui = False
            elif float(def_ui) > 0 and float(cand_ui) > float(def_ui) * (1.0 + UI_BLOCK_WORSE):
                per_ui = False
        dd_ok = True
        for cand_dd, base_dd in zip(rec.get("block_maxdd") or [], default.get("block_maxdd") or []):
            if cand_dd is None or base_dd is None or not np.isfinite(cand_dd) or not np.isfinite(base_dd):
                dd_ok = False
            elif float(cand_dd) < float(base_dd) - MAXDD_WORSEN:
                dd_ok = False
        leverage_ok = bool(rec.get("no_leverage", True))
        sample_ok = int(rec.get("n_days") or 0) >= MIN_INNER_DAYS and int(rec.get("n_trades") or 0) >= MIN_INNER_TRADES
        rec["r3_pass"] = bool(sample_ok and g_ok and ui_ok and per_ui and dd_ok and leverage_ok and not rec.get("window_fail"))
        if rec["r3_pass"]:
            qualified.append(rec)
    if not qualified:
        return {"chosen": default_key, "reason": "FALLBACK_0_NO_QUALIFIED", "qualified": []}
    qualified.sort(key=lambda r: (float(r["mean_ui"]), abs(float(r["w_lgbm"]))))
    return {"chosen": qualified[0]["key"], "reason": "QUALIFIED_LOWEST_UI_THEN_SMALLER_W", "qualified": [r["key"] for r in qualified]}


def block_metrics(returns: pd.Series) -> dict[str, float]:
    clean = returns.dropna()
    return {
        "n_days": int(clean.shape[0]),
        "mean_log_growth": float(np.log1p(clean).mean()) if len(clean) else float("nan"),
        "ann_log_growth": annualized_log_growth(clean),
        "ui": _ui(clean),
        "maxdd": _maxdd(clean),
    }
