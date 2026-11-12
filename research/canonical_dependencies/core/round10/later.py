"""I shadow NOT_AVAILABLE、J 回撤、K≤5。pair 只能 veto，禁止双重乘仓。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..paths import MARKET_END, ROUND10_ROOT, assert_project_result_path, ensure_round10_tree, round10_dir
from .allocator import b03_nav, load_alloc, write_alloc
from .contract import PARENT_CODE, R10_00
from .daily import load_daily
from .registry import get_spec, specs_for_stage
from .runner import load_summary, run_codes


def run_shadow() -> dict[str, Any]:
    ensure_round10_tree()
    payload = {
        "status": "NOT_AVAILABLE",
        "reason": "快照 hermes-cn-a-2010-20260724-v7-all-features 行情最大日 2026-07-24。无 2026-07-25 之后完整可用数据。",
        "market_end": str(MARKET_END),
        "genuine_oos": False,
        "note": "Round1–9 的 metric_nav_end 均止于该日，不是 NOT_GENUINE_OOS。禁止伪造 07-25+，禁止按 OOS 调参。",
    }
    path = ROUND10_ROOT / "shadow" / "status.json"
    assert_project_result_path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    md = ROUND10_ROOT / "09_shadow_oos.md"
    assert_project_result_path(md)
    md.write_text(
        f"""# 09 — Shadow OOS

**NOT_AVAILABLE**

{payload['reason']}

manifest 已在 Phase 1 freeze。本检查发生在 freeze 之后。没有 post-2026-07-24 绩效可写。
禁止根据 OOS 调参。短期样本即使以后有了也只能 veto，不能单独证明 winner。
""",
        encoding="utf-8",
    )
    return payload


def run_drawdown() -> dict[str, Any]:
    ensure_round10_tree()
    nav = pd.read_parquet(round10_dir(R10_00) / "nav.parquet")
    nav["trade_date"] = pd.to_datetime(nav["trade_date"]).dt.normalize()
    nav = nav.sort_values("trade_date")
    asset = nav.set_index("trade_date")["total_asset"].astype(float)
    ret = asset.pct_change().fillna(0.0)
    peak = asset.cummax()
    dd = asset / peak - 1.0
    episodes = []
    in_dd = False
    start = None
    trough = 0.0
    trough_day = None
    for day, val in dd.items():
        if val < 0 and not in_dd:
            in_dd = True
            start = day
            trough = val
            trough_day = day
        elif in_dd:
            if val < trough:
                trough = val
                trough_day = day
            if val == 0:
                episodes.append({"start": str(pd.Timestamp(start).date()), "trough_date": str(pd.Timestamp(trough_day).date()), "end": str(pd.Timestamp(day).date()), "depth": float(trough)})
                in_dd = False
    if in_dd and start is not None:
        episodes.append({"start": str(pd.Timestamp(start).date()), "trough_date": str(pd.Timestamp(trough_day).date()), "end": str(pd.Timestamp(dd.index[-1]).date()), "depth": float(trough)})
    episodes = sorted(episodes, key=lambda x: x["depth"])[:5]
    trades = pd.read_parquet(round10_dir(R10_00) / "trades.parquet")
    trades["signal_date"] = pd.to_datetime(trades["signal_date"]).dt.normalize()
    trades["_rank"] = trades.groupby("signal_date")["score"].rank(method="first", ascending=False)
    trades["dollar"] = pd.to_numeric(trades.get("sell_proceeds"), errors="coerce") - pd.to_numeric(trades.get("buy_cost"), errors="coerce")
    daily = load_daily()
    for ep in episodes:
        t0, t1 = pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"])
        sl = trades.loc[(trades["signal_date"] >= t0) & (trades["signal_date"] <= t1)]
        ep["top1_pnl"] = float(sl.loc[sl["_rank"].eq(1), "dollar"].sum()) if len(sl) else 0.0
        ep["top2_pnl"] = float(sl.loc[sl["_rank"].eq(2), "dollar"].sum()) if len(sl) else 0.0
        sub = daily.loc[(daily.index >= t0) & (daily.index <= t1)]
        ep["agree_counts"] = sub["agreement"].value_counts().to_dict() if len(sub) else {}
        ep["sep_counts"] = sub["sep_bucket"].value_counts().to_dict() if len(sub) else {}
    ret_t = pd.to_numeric(trades.get("ret"), errors="coerce").dropna()
    q = {f"P{p}": float(np.nanpercentile(ret_t, p)) for p in (1, 5, 10, 50, 90, 95, 99)} if len(ret_t) else {}
    dollar = trades["dollar"].dropna().sort_values(ascending=False)
    total = float(dollar.sum()) if len(dollar) else 0.0
    n = len(dollar)

    def share(k, tail):
        m = max(1, int(np.ceil(n * k)))
        if not total:
            return None
        return float(dollar.head(m).sum() / total) if tail == "top" else float(dollar.tail(m).sum() / total)

    payload = {
        "episodes": episodes,
        "quantiles": q,
        "right_tail": {"top1pct": share(0.01, "top"), "top5pct": share(0.05, "top"), "top10pct": share(0.10, "top")},
        "left_tail": {"bot1pct": share(0.01, "bot"), "bot5pct": share(0.05, "bot"), "bot10pct": share(0.10, "bot")},
        "top2_is_main_dd": all(abs(e.get("top2_pnl") or 0) > abs(e.get("top1_pnl") or 0) for e in episodes[:1]) if episodes else False,
    }
    path = ROUND10_ROOT / "metrics" / "drawdown_top5.json"
    assert_project_result_path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    md = ROUND10_ROOT / "10_tail_drawdown.md"
    assert_project_result_path(md)
    md.write_text(
        f"""# 10 — 回撤与尾部

Top5 回撤 episode：

```json
{json.dumps(episodes, ensure_ascii=False, indent=2)}
```

交易收益分位：{q}

右尾贡献：{payload['right_tail']}
左尾贡献：{payload['left_tail']}

Top2 是否第一大回撤主因：{payload['top2_is_main_dd']}。不做新的全市场择时。
""",
        encoding="utf-8",
    )
    return payload


def _best_of(codes: list[str]) -> str | None:
    best, key = None, (-9.0, -9.0)
    for code in codes:
        try:
            spec = get_spec(code)
        except Exception:
            continue
        if spec.skip or spec.diagnostic:
            continue
        s = load_summary(code)
        if not s or s.get("sharpe") is None:
            continue
        k = (float(s["sharpe"]), float(s["cagr"]))
        if k > key:
            key = k
            best = code
    return best


def run_combine(*, resume: bool = True, jobs: int = 1) -> dict[str, Any]:
    ensure_round10_tree()
    daily = load_daily()
    try:
        base = b03_nav(daily["agreement"], daily["sep_bucket"])
    except Exception:
        base = pd.Series(0.0, index=daily.index)
    shrink_codes = [c for c in ("R10_C01", "R10_C02", "R10_C03") if (load_summary(c) or {}).get("cagr") is not None]
    pair_codes = []
    for c in ("R10_B01", "R10_B02", "R10_B03"):
        s = load_summary(c)
        if s and s.get("cagr") is not None:
            get_spec(c).skip = False
            get_spec(c).skip_reason = ""
            pair_codes.append(c)
    best_shrink = _best_of(shrink_codes) or "R10_C01"
    best_pair = _best_of(pair_codes)
    best_stress = "R10_C03" if (load_summary("R10_C03") or {}).get("cagr") is not None else best_shrink

    def _load(code: str) -> pd.Series:
        try:
            return load_alloc(code).reindex(daily.index).fillna(0.0)
        except Exception:
            return base.copy()

    slots = {
        "R10_K01": _load(best_shrink) if best_shrink else base,
        "R10_K02": _load(best_pair) if best_pair else base,
        "R10_K03": _load(best_shrink) if best_shrink else base,
        "R10_K04": _load(best_stress),
        "R10_K05": _load(best_shrink) if best_shrink else base,
    }
    # K03: shrinkage primary + pair veto（只把 pair=0 的日子打成 0，不双重乘）
    if best_pair:
        pair_nav = _load(best_pair)
        veto = pair_nav <= 1e-12
        slots["R10_K03"] = slots["R10_K03"].where(~veto, 0.0)
        get_spec("R10_K03").skip = False
        get_spec("R10_K03").skip_reason = ""
    else:
        get_spec("R10_K03").skip = True
        get_spec("R10_K03").skip_reason = "NO_PAIR_VETO"
    # K05 only if pair-relative family had increment
    if best_pair and (load_summary(best_pair) or {}).get("cagr") is not None:
        v4 = load_summary(R10_00) or {}
        if float((load_summary(best_pair) or {}).get("cagr") or -9) > float(v4.get("cagr") or 0):
            get_spec("R10_K05").skip = False
            get_spec("R10_K05").skip_reason = ""
        else:
            get_spec("R10_K05").skip = True
            get_spec("R10_K05").skip_reason = "NO_PAIR_RELATIVE_INCREMENT"
    else:
        get_spec("R10_K05").skip = True
        get_spec("R10_K05").skip_reason = "NO_PAIR_RELATIVE_INCREMENT"

    get_spec("R10_K01").skip = False
    get_spec("R10_K01").skip_reason = ""
    if best_pair:
        get_spec("R10_K02").skip = False
        get_spec("R10_K02").skip_reason = ""
    else:
        get_spec("R10_K02").skip = True
        get_spec("R10_K02").skip_reason = "SKIP_B_NO_STABLE_PAIR"
    get_spec("R10_K04").skip = False
    get_spec("R10_K04").skip_reason = ""

    for code, series in slots.items():
        if not get_spec(code).skip:
            write_alloc(code, daily.index, series)
    codes = [s.code for s in specs_for_stage("combine")]
    bt = run_codes(codes, resume=resume, jobs=jobs)
    payload = {
        "best_shrink": best_shrink,
        "best_pair": best_pair,
        "best_stress": best_stress,
        "rule": "pair veto only, no double multiply",
        "failed": bt.get("failed") or [],
        "jobs": {k: v.get("status") for k, v in (bt.get("jobs") or {}).items()},
    }
    path = ROUND10_ROOT / "metrics" / "combine_slots.json"
    assert_project_result_path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    from .docs import write_named_md

    write_named_md("11_final_candidates.md", "combine", codes, f"slots={payload}\n")
    return payload


def run_later(*, stage: str, resume: bool = True, jobs: int = 1, experiment: str = "") -> dict[str, Any]:
    if stage == "shadow":
        return run_shadow()
    if stage in {"drawdown", "tail"}:
        return run_drawdown()
    if stage == "combine":
        return run_combine(resume=resume, jobs=jobs)
    raise RuntimeError(stage)


__all__ = ["run_later"]
