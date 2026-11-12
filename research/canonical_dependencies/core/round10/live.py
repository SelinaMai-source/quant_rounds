"""D/E/F/G 实跑回测 + H 文档/fail-safe。不造随机删行情的假绩效。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..paths import PROJECT_ROOT, ROUND10_ROOT, assert_project_result_path, ensure_round10_tree
from .contract import BASE_CASH, HERMES_DAILY_RAW, PARENT_CODE, R10_00
from .daily import load_daily
from .registry import specs_for_stage
from .runner import load_summary, run_codes


def _build_amount_cache() -> dict[str, Any]:
    ensure_round10_tree()
    dest = ROUND10_ROOT / "capacity" / "daily_amount.parquet"
    if dest.is_file():
        return {"status": "exists", "path": str(dest)}
    daily = load_daily()
    ids = sorted(set(daily["top1_id"].astype(str)) | set(daily["top2_id"].astype(str)))
    if not HERMES_DAILY_RAW.exists():
        return {"status": "NO_DAILY_RAW"}
    import pyarrow.dataset as ds

    table = ds.dataset(str(HERMES_DAILY_RAW), format="parquet").to_table(
        columns=["trade_date", "instrument_id", "amount"],
        filter=ds.field("instrument_id").isin(ids),
    )
    frame = table.to_pandas()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    assert_project_result_path(dest)
    frame.to_parquet(dest, index=False)
    return {"status": "wrote", "n": int(len(frame)), "path": str(dest)}


def _elasticity(codes: list[str]) -> dict[str, Any]:
    rows = []
    for code in codes:
        s = load_summary(code) or {}
        bp = None
        if "bp" in code:
            try:
                bp = int(code.split("_")[2].replace("bp", ""))
            except Exception:
                try:
                    bp = int("".join(ch for ch in code.split("_")[-1] if ch.isdigit()))
                except Exception:
                    bp = None
        rows.append({"code": code, "bp": bp, "cagr": s.get("cagr"), "sharpe": s.get("sharpe"), "max_drawdown": s.get("max_drawdown")})
    frame = pd.DataFrame(rows).dropna(subset=["bp", "cagr"])
    out = {"rows": rows}
    if len(frame) >= 2:
        x = frame["bp"].astype(float).to_numpy()
        y = frame["cagr"].astype(float).to_numpy()
        coef = np.polyfit(x, y, 1)
        out["elasticity_cagr_per_bp"] = float(coef[0])
        zero_cagr = None
        if coef[0] != 0:
            zero_cagr = float(-coef[1] / coef[0])
        out["break_even_extra_bp_cagr0"] = zero_cagr
    return out


def _perturb_report() -> dict[str, Any]:
    from .contract import PERTURB_SEEDS

    out = {}
    daily = load_daily()
    v4_top1 = daily["top1_id"].astype(str)
    v4_top2 = daily["top2_id"].astype(str)
    for tag, scale in (("E01", 0.005), ("E02", 0.01)):
        cagrs, sharpes, dds, t1, t2 = [], [], [], [], []
        for seed in PERTURB_SEEDS:
            code = f"R10_{tag}_s{seed}"
            s = load_summary(code) or {}
            if s.get("cagr") is not None:
                cagrs.append(float(s["cagr"]))
                sharpes.append(float(s["sharpe"]))
                dds.append(float(s["max_drawdown"]))
            path = ROUND10_ROOT / code / "trades.parquet"
            if path.is_file():
                t = pd.read_parquet(path)
                t["signal_date"] = pd.to_datetime(t["signal_date"]).dt.normalize()
                t["_rank"] = t.groupby("signal_date")["score"].rank(method="first", ascending=False)
                p1 = t.loc[t["_rank"].eq(1)].drop_duplicates("signal_date").set_index("signal_date")["instrument_id"].astype(str)
                p2 = t.loc[t["_rank"].eq(2)].drop_duplicates("signal_date").set_index("signal_date")["instrument_id"].astype(str)
                idx = v4_top1.index.intersection(p1.index)
                t1.append(float((p1.reindex(idx) == v4_top1.reindex(idx)).mean()) if len(idx) else None)
                idx2 = v4_top2.index.intersection(p2.index)
                t2.append(float((p2.reindex(idx2) == v4_top2.reindex(idx2)).mean()) if len(idx2) else None)
        fragile = False
        if t1 and (float(np.nanmean(t1)) < 0.85 or (cagrs and float(np.nanstd(cagrs)) > 0.03)):
            fragile = True
        out[tag] = {
            "scale": scale,
            "cagr_mean": float(np.nanmean(cagrs)) if cagrs else None,
            "cagr_std": float(np.nanstd(cagrs)) if cagrs else None,
            "sharpe_mean": float(np.nanmean(sharpes)) if sharpes else None,
            "sharpe_std": float(np.nanstd(sharpes)) if sharpes else None,
            "maxdd_mean": float(np.nanmean(dds)) if dds else None,
            "top1_stability": float(np.nanmean(t1)) if t1 else None,
            "top2_stability": float(np.nanmean(t2)) if t2 else None,
            "flag": "RANKING_FRAGILE" if fragile else "OK",
        }
    return out


def _neighborhood_verdict() -> dict[str, Any]:
    v4 = load_summary(R10_00) or {}
    codes = [s.code for s in specs_for_stage("neighborhood")]
    rows = []
    for code in codes:
        s = load_summary(code) or {}
        rows.append({"code": code, "cagr": s.get("cagr"), "sharpe": s.get("sharpe")})
    cagrs = [r["cagr"] for r in rows if r["cagr"] is not None]
    flag = "STABLE_PLATEAU"
    if cagrs and v4.get("cagr") is not None:
        if min(cagrs) < float(v4["cagr"]) - 0.03:
            flag = "PARAMETER_FRAGILE"
    return {"rows": rows, "flag": flag, "min_cagr": min(cagrs) if cagrs else None}


def _capacity_label() -> dict[str, Any]:
    v4 = load_summary(R10_00) or {}
    rows = []
    for code in [s.code for s in specs_for_stage("capacity")]:
        s = load_summary(code) or {}
        rows.append({
            "code": code, "cagr": s.get("cagr"), "sharpe": s.get("sharpe"),
            "max_drawdown": s.get("max_drawdown"), "n": s.get("completed_trades"),
        })
    label = "CAPACITY_LIMITED"
    g1 = load_summary("R10_G01_1m_p3") or {}
    g2 = load_summary("R10_G02_3m_p3") or {}
    g3 = load_summary("R10_G03_10m_p3") or {}
    def _ok(s):
        return s.get("cagr") is not None and float(s["cagr"]) >= float(v4.get("cagr") or 0) - 0.03
    if _ok(g3):
        label = "ROBUST_UP_TO_10M"
    elif _ok(g2):
        label = "ROBUST_UP_TO_3M"
    elif _ok(g1):
        label = "ROBUST_UP_TO_1M"
    if g1.get("cagr") is not None and v4.get("cagr") is not None:
        if float(g1["cagr"]) < float(v4["cagr"]) - 0.05:
            label = "CAPACITY_FRAGILE"
    return {"label": label, "rows": rows}


def write_fault_docs() -> dict[str, Any]:
    ensure_round10_tree()
    dep = """# 08 — 运行依赖与 fail-safe

不是 Alpha 优化。缺数时禁止悄悄填 0。

## 依赖图

| 输入 | 来源 | 截止 | 决策时点 | 缺失处理 | 过期处理 | 关键度 |
| --- | --- | --- | --- | --- | --- | --- |
| 日行情 OHLC/量额 | Hermes `daily_raw` | D 收盘后 | D 收盘打分 / D+1 开盘买 | **NO TRADE** | 不用过期日线 | CRITICAL |
| 后复权 | Hermes `daily_post_adjusted` | D 收盘后 | 收益/相关 | 相关诊断 SKIP | 不用过期 | IMPORTANT |
| 涨跌停事件 | Hermes `daily_limit_events` | D 收盘后 | 候选宇宙 | **NO TRADE** | **NO TRADE** | CRITICAL |
| five-head 预测 | 锁定 dual90/10 five | D 收盘后、D+1 开盘前 | 融合 | **NO TRADE** | **NO TRADE** | CRITICAL |
| ten-head 预测 | 锁定 dual90/10 ten | 同上 | 融合 | **NO TRADE** | **NO TRADE** | CRITICAL |
| `d1_open_gap` | 候选面板 / D+1 开盘 | D+1 开盘 | 预滤 ≥ -1% | **NO TRADE** | **NO TRADE** | CRITICAL |
| 候选宇宙 | first_board 面板 | D 收盘 | 选股 | **NO TRADE** | **NO TRADE** | CRITICAL |
| B01a 广度 / top1−median | 当日横截面 | D 收盘 | Top1 LOCK 仓位 | 弱信号按规则降仓，不算填 0 | 滚动窗不足则用默认 1.0 | IMPORTANT |
| agreement A/B/C | five/ten Top1 | D 收盘 | Top2 allocator | 两头齐备才能分态；否则 Top2=0 | 同左 | IMPORTANT |
| Top2 separation | fused 截面分位 | D 收盘 | Top2 allocator | 分位不足则 MID | 同左 | IMPORTANT |
| 行业 L1 PIT | Hermes `industry_membership` known_at | D 已知 | pair veto | fallback **baseline_v4 allocator** | 不用当前申万回填 | OPTIONAL |
| pair 相关 | 过去 20/60 日收益 | D 之前 | pair veto | fallback **baseline_v4 allocator** | 绝不用未来 | OPTIONAL |

## Fail-safe

- five / ten / d1_open_gap / 日行情 / 候选宇宙缺失 → **NO TRADE**
- pair 行业或相关缺失 → 回退 v4 B03，不发明仓位
- 禁止随机删除主行情后声称绩效

## 确定性 replay

同一 snapshot + 同一代码 + 同一 config + 同一 seed → 同一信号 / 同一委托 / 同一仓位。
Phase 1 `R10_00` 已与 `R9_B03` 逐笔对齐，作为 replay 锚点。
"""
    path = ROUND10_ROOT / "operational_dependency_map.md"
    assert_project_result_path(path)
    path.write_text(dep, encoding="utf-8")
    md = ROUND10_ROOT / "08_operational_fault_tolerance.md"
    assert_project_result_path(md)
    md.write_text(dep + "\n\nH 组不另造随机删行情假账本。\n", encoding="utf-8")
    return {"status": "wrote", "path": str(path)}


def run_live(*, stage: str = "live", resume: bool = True, jobs: int = 1, experiment: str = "") -> dict[str, Any]:
    ensure_round10_tree()
    from .docs import write_named_md

    wanted = {
        "execution": ["execution"],
        "perturbation": ["perturbation"],
        "neighborhood": ["neighborhood"],
        "capacity": ["capacity"],
        "fault": [],
        "live": ["execution", "perturbation", "neighborhood", "capacity"],
    }.get(stage, [stage])
    out: dict[str, Any] = {"stage": stage}
    if "capacity" in wanted or stage in {"capacity", "live"}:
        out["amount_cache"] = _build_amount_cache()
    failed = []
    for st in wanted:
        codes = [c.strip() for c in experiment.split(",") if c.strip()] if experiment and stage == st else [s.code for s in specs_for_stage(st)]
        if not codes:
            continue
        bt = run_codes(codes, resume=resume, jobs=jobs)
        failed.extend(bt.get("failed") or [])
        out[st] = {k: v.get("status") for k, v in (bt.get("jobs") or {}).items()}
        extra = ""
        if st == "execution":
            elas = _elasticity(codes)
            out["elasticity"] = elas
            extra = (
                f"买侧不利不是费用。弹性 ΔCAGR/Δbp = {elas.get('elasticity_cagr_per_bp')}。"
                f"粗算 CAGR=0 的额外 bp ≈ {elas.get('break_even_extra_bp_cagr0')}。\n"
            )
            write_named_md("04_execution_robustness.md", st, codes, extra)
        elif st == "perturbation":
            rep = _perturb_report()
            out["perturbation_report"] = rep
            extra = f"五种子 NAV 均已落盘。dispersion={rep}\n"
            write_named_md("05_signal_perturbation.md", st, codes, extra)
        elif st == "neighborhood":
            nv = _neighborhood_verdict()
            out["neighborhood"] = nv
            extra = f"邻域判定 **{nv.get('flag')}**。center=v4 的 20/50 与 0/5/10。\n"
            write_named_md("06_parameter_neighborhood.md", st, codes, extra)
        elif st == "capacity":
            cap = _capacity_label()
            out["capacity"] = cap
            extra = (
                f"研究标签 **{cap.get('label')}**。无 ADV 引擎，用日成交额 1%/3%/5% diagnostic，不伪造盘口。"
                f"300k 无帽 = v4 正式账本。\n"
            )
            write_named_md("07_capacity_liquidity.md", st, codes, extra)
    if stage in {"fault", "live"}:
        out["fault"] = write_fault_docs()
    out["failed"] = failed
    path = ROUND10_ROOT / "metrics" / f"live_{stage}.json"
    assert_project_result_path(path)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n")
    return out


__all__ = ["run_live"]
