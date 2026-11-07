"""Bootstrap / 成本 / 阈值邻域 / winner dependence。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND8_ROOT, assert_project_result_path, ensure_round8_tree, round8_dir
from .contract import ANCHOR_DIR, R7_V1_DIR, TARGET_CAGR, TARGET_SHARPE, TARGET_MAXDD
from .runner import load_summary


def _daily(nav: pd.DataFrame) -> pd.Series:
    n = nav.copy()
    n["trade_date"] = pd.to_datetime(n["trade_date"])
    n = n.sort_values("trade_date")
    s = n.set_index("trade_date")["total_asset"].astype(float).pct_change().dropna()
    return s


def block_bootstrap_delta(
    cand: pd.Series,
    base: pd.Series,
    *,
    block: int = 20,
    n: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    aligned = pd.concat([cand.rename("c"), base.rename("b")], axis=1).dropna()
    delta = aligned["c"] - aligned["b"]
    x = delta.to_numpy()
    if len(x) < block * 2:
        return {"status": "too_short", "n": int(len(x))}
    n_blocks = int(np.ceil(len(x) / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(x) - block + 1, size=(n, n_blocks))
    samples_ret = []
    samples_sharpe = []
    samples_dd_c = []
    samples_dd_b = []
    c = aligned["c"].to_numpy()
    b = aligned["b"].to_numpy()
    for row in starts:
        idx = np.concatenate([np.arange(s, s + block) for s in row])[: len(x)]
        d = x[idx]
        cc = c[idx]
        bb = b[idx]
        wealth = float(np.prod(1.0 + d))
        years = len(d) / 252.0
        samples_ret.append(wealth ** (1.0 / max(years, 1e-9)) - 1.0)
        samples_sharpe.append(
            (float(cc.mean()) / float(cc.std(ddof=1)) - float(bb.mean()) / float(bb.std(ddof=1))) * np.sqrt(252)
            if cc.std(ddof=1) > 0 and bb.std(ddof=1) > 0 else np.nan
        )
        def _dd(r):
            nav = np.cumprod(1.0 + r)
            peak = np.maximum.accumulate(nav)
            return float((nav / peak - 1.0).min())
        samples_dd_c.append(_dd(cc))
        samples_dd_b.append(_dd(bb))
    sr = np.array(samples_ret, dtype=float)
    ss = np.array(samples_sharpe, dtype=float)
    return {
        "n": n,
        "block": block,
        "median_ann_delta": float(np.nanmedian(sr)),
        "p05_ann_delta": float(np.nanpercentile(sr, 5)),
        "p95_ann_delta": float(np.nanpercentile(sr, 95)),
        "p_delta_ret_gt0": float(np.mean(sr > 0)),
        "p_delta_sharpe_gt0": float(np.nanmean(ss > 0)),
        "median_maxdd_candidate": float(np.nanmedian(samples_dd_c)),
        "median_maxdd_baseline": float(np.nanmedian(samples_dd_b)),
    }


def winner_dependence(code: str) -> dict[str, Any]:
    path = round8_dir(code) / "trades.parquet"
    if not path.is_file():
        path = ANCHOR_DIR / "trades.parquet" if code in {"R8_00_baseline_v2", "baseline_v2"} else path
    if not path.is_file():
        return {"status": "missing"}
    t = pd.read_parquet(path)
    dollar = pd.to_numeric(t.get("sell_proceeds"), errors="coerce") - pd.to_numeric(t.get("buy_cost"), errors="coerce")
    if dollar.isna().all():
        dollar = pd.to_numeric(t["ret"], errors="coerce")
    total = float(dollar.sum())
    ordered = dollar.sort_values(ascending=False)
    n = len(ordered)

    def share(k: float, tail: str) -> float:
        m = max(1, int(np.ceil(n * k)))
        if tail == "top":
            return float(ordered.head(m).sum() / total) if total else float("nan")
        return float(ordered.tail(m).sum() / total) if total else float("nan")

    removed = {
        "remove_best": float(total - float(ordered.iloc[0])) if n else None,
        "remove_top3": float(total - float(ordered.head(3).sum())) if n >= 3 else None,
        "remove_top1pct": float(total - float(ordered.head(max(1, int(np.ceil(n * 0.01)))).sum())) if n else None,
    }
    return {
        "code": code,
        "role": "DIAGNOSTIC_NOT_REALISTIC",
        "top1pct": share(0.01, "top"),
        "top5pct": share(0.05, "top"),
        "top10pct": share(0.10, "top"),
        "bottom1pct": share(0.01, "bot"),
        "bottom5pct": share(0.05, "bot"),
        "bottom10pct": share(0.10, "bot"),
        "removed_dollar_pnl": removed,
    }


def run_robustness(finalists: list[str], *, resume: bool = True) -> dict[str, Any]:
    ensure_round8_tree()
    base_nav = pd.read_parquet(ANCHOR_DIR / "nav.parquet")
    base = _daily(base_nav)
    boot = {}
    depend = {"R8_00_baseline_v2": winner_dependence("R8_00_baseline_v2")}
    for code in finalists:
        nav_path = round8_dir(code) / "nav.parquet"
        if not nav_path.is_file():
            continue
        boot[code] = block_bootstrap_delta(_daily(pd.read_parquet(nav_path)), base)
        depend[code] = winner_dependence(code)
        path = ROUND8_ROOT / "bootstrap" / f"{code}.json"
        assert_project_result_path(path)
        path.write_text(json.dumps(boot[code], ensure_ascii=False, indent=2) + "\n")
    payload = {
        "bootstrap": boot,
        "winner_dependence": depend,
        "note": "winner dependence is DIAGNOSTIC / NOT REALISTIC",
    }
    out = ROUND8_ROOT / "bootstrap" / "summary.json"
    assert_project_result_path(out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    return payload


def v2_anchor_metrics() -> dict[str, float]:
    s = load_summary("R8_00_baseline_v2") or {}
    return {
        "cagr": float(s.get("cagr") or TARGET_CAGR),
        "sharpe": float(s.get("sharpe") or TARGET_SHARPE),
        "max_drawdown": float(s.get("max_drawdown") or TARGET_MAXDD),
        "loyo_median": float(s["loyo_median"]) if s.get("loyo_median") is not None else None,
        "loyo_worst": float(s["loyo_worst"]) if s.get("loyo_worst") is not None else None,
        "cagr_ex2024": float(s["cagr_ex2024"]) if s.get("cagr_ex2024") is not None else None,
    }


__all__ = ["run_robustness", "v2_anchor_metrics", "winner_dependence"]
