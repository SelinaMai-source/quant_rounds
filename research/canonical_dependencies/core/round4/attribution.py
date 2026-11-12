"""门禁 A：年份集中拆解 + Random Top1 + Event Beta vs Selection Alpha。没写完不准搜策略。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND3_ROOT, ROUND4_ROOT, assert_project_result_path, ensure_round4_tree
from ..round2.data import load_candidate_features
from ..round2.metrics import right_tail_contribution
from .registry import BASELINE_V2_ALIAS, RANDOM_SEEDS, SPECS, get_spec
from .runner import load_report, run_spec, tracking_row

COMPLETE_YEARS = tuple(range(2018, 2026))


def _write_json(rel: str, payload: Any) -> None:
    path = ensure_round4_tree() / rel
    assert_project_result_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _pct(v: Any, d: int = 2) -> str:
    if v is None:
        return "—"
    return f"{float(v) * 100:+.{d}f}%"


def _num(v: Any, d: int = 3) -> str:
    if v is None:
        return "—"
    return f"{float(v):.{d}f}"


def _load_trades(code: str) -> pd.DataFrame:
    for root in (ROUND4_ROOT / code, ROUND3_ROOT / code):
        path = root / "trades.parquet"
        if path.is_file():
            frame = pd.read_parquet(path)
            frame["signal_date"] = pd.to_datetime(frame["signal_date"])
            if "actual_exit_date" in frame:
                frame["actual_exit_date"] = pd.to_datetime(frame["actual_exit_date"])
            return frame
    return pd.DataFrame()


def _load_nav(code: str) -> pd.DataFrame:
    for root in (ROUND4_ROOT / code, ROUND3_ROOT / code):
        path = root / "nav.parquet"
        if path.is_file():
            frame = pd.read_parquet(path)
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
            return frame
    return pd.DataFrame()


def year_trade_breakdown(trades: pd.DataFrame, nav: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {}
    frame = trades.copy()
    frame["year"] = pd.to_datetime(frame["signal_date"]).dt.year
    if "ret" not in frame:
        return {}
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    panel = load_candidate_features(["trade_date", "instrument_id", "label_d1_touch"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["instrument_id"] = panel["instrument_id"].astype(str)
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    merged = frame.merge(
        panel.rename(columns={"trade_date": "signal_date"}),
        on=["signal_date", "instrument_id"],
        how="left",
    )
    rows = {}
    yearly_nav = {}
    if nav is not None and not nav.empty:
        ordered = nav.sort_values("trade_date").copy()
        ordered["year"] = pd.to_datetime(ordered["trade_date"]).dt.year
        prior = 300_000.0
        for year, block in ordered.groupby("year", sort=True):
            last = float(block["total_asset"].iloc[-1])
            yearly_nav[str(int(year))] = last / prior - 1.0
            prior = last
    for year, block in merged.groupby("year", sort=True):
        ret = block["ret"].dropna()
        tail = right_tail_contribution(block)
        live = None
        if nav is not None and not nav.empty:
            sl = nav.loc[pd.to_datetime(nav["trade_date"]).dt.year == int(year)]
            if not sl.empty and "live_sleeves" in sl:
                live = float(pd.to_numeric(sl["live_sleeves"], errors="coerce").mean())
        touch = pd.to_numeric(block.get("label_d1_touch"), errors="coerce") if "label_d1_touch" in block else pd.Series(dtype=float)
        rows[str(int(year))] = {
            "n_trades": int(len(block)),
            "win_rate": float((ret > 0).mean()) if len(ret) else None,
            "mean_ret": float(ret.mean()) if len(ret) else None,
            "sum_ret": float(ret.sum()) if len(ret) else None,
            "mean_alloc": float(pd.to_numeric(block.get("allocation_multiplier"), errors="coerce").mean())
            if "allocation_multiplier" in block else None,
            "mean_live_sleeves": live,
            "continue_rate": float(touch.mean()) if touch.notna().any() else None,
            "top1pct_share": tail.get("top1pct_share"),
            "top5pct_share": tail.get("top5pct_share"),
            "top10pct_share": tail.get("top10pct_share"),
            "nav_return": yearly_nav.get(str(int(year))),
            "ytd": int(year) == 2026,
        }
    return rows


def diagnostic_equal_weight_index(predictions: pd.DataFrame) -> dict[str, Any]:
    """DIAGNOSTIC INDEX：候选等权，不是可执行组合。"""
    panel = load_candidate_features(["trade_date", "instrument_id", "label_e2_net", "is_first_board_hard"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    hard = panel["is_first_board_hard"]
    if hard.dtype == object:
        fb = panel.loc[hard.astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    else:
        fb = panel.loc[hard.fillna(False).astype(bool)].copy()
    fb["label_e2_net"] = pd.to_numeric(fb["label_e2_net"], errors="coerce")
    daily = fb.groupby("trade_date")["label_e2_net"].mean().sort_index()
    daily = daily.loc[daily.index >= pd.Timestamp("2018-01-01")]
    yearly = {}
    for year, block in daily.groupby(daily.index.year):
        yearly[str(int(year))] = float((1.0 + block.fillna(0)).prod() - 1.0)
    wealth = float((1.0 + daily.fillna(0)).prod())
    first, last = daily.index.min(), daily.index.max()
    years = max((last - first).days / 365.25, 1 / 365.25)
    return {
        "tag": "DIAGNOSTIC INDEX",
        "executable": False,
        "note": "候选/首板等权 label_e2_net，不是双袖套可执行组合。",
        "cagr": wealth ** (1.0 / years) - 1.0,
        "yearly": yearly,
        "n_days": int(daily.notna().sum()),
        "ret_2024": yearly.get("2024"),
    }


def _year_nav_return(nav: pd.DataFrame, year: int) -> float | None:
    if nav is None or nav.empty:
        return None
    ordered = nav.sort_values("trade_date").copy()
    ordered["trade_date"] = pd.to_datetime(ordered["trade_date"])
    prior = ordered.loc[ordered["trade_date"].dt.year < year]
    sl = ordered.loc[ordered["trade_date"].dt.year == year]
    if sl.empty:
        return None
    start = float(prior["total_asset"].iloc[-1]) if not prior.empty else 300_000.0
    end = float(sl["total_asset"].iloc[-1])
    return end / start - 1.0


def write_attribution(
    *,
    predictions,
    prepared,
    meta,
    resume: bool = True,
    run_now: bool = True,
) -> dict[str, Any]:
    ensure_round4_tree()
    specs = [get_spec(BASELINE_V2_ALIAS)]
    specs += [get_spec(f"R4_RAND_{s}") for s in RANDOM_SEEDS]
    specs += [get_spec("R4_TOP2_diag"), get_spec("R4_TOP3_diag")]
    results = {}
    for spec in specs:
        results[spec.code] = run_spec(
            spec, predictions=predictions, prepared_market=prepared, market_meta=meta,
            resume=resume, run_now=run_now,
        )

    v2_trades = _load_trades(BASELINE_V2_ALIAS)
    v2_nav = _load_nav(BASELINE_V2_ALIAS)
    if v2_trades.empty:
        v2_trades = _load_trades("R3_03")
        v2_nav = _load_nav("R3_03")
    yearly = year_trade_breakdown(v2_trades, v2_nav)
    index = diagnostic_equal_weight_index(predictions)

    rand_rows = []
    rand_2024 = []
    rand_full = []
    for seed in RANDOM_SEEDS:
        code = f"R4_RAND_{seed}"
        report = load_report(code) or {}
        m = report.get("metrics") or {}
        nav = _load_nav(code)
        y2024 = _year_nav_return(nav, 2024)
        row = {
            "code": code,
            "seed": seed,
            "cagr": m.get("annualized_return"),
            "sharpe": m.get("sharpe_zero_rate"),
            "max_drawdown": m.get("max_drawdown"),
            "completed_trades": m.get("completed_trades"),
            "ret_2024": y2024,
            "cagr_ex2024": m.get("cagr_ex2024"),
        }
        rand_rows.append(row)
        if y2024 is not None:
            rand_2024.append(y2024)
        if m.get("annualized_return") is not None:
            rand_full.append(float(m["annualized_return"]))

    v2m = (load_report(BASELINE_V2_ALIAS) or {}).get("metrics") or {}
    v2_2024 = (v2m.get("yearly_returns") or {}).get("2024")
    if v2_2024 is None:
        v2_2024 = yearly.get("2024", {}).get("nav_return")

    rand_2024_med = float(np.median(rand_2024)) if rand_2024 else None
    rand_2024_p10 = float(np.percentile(rand_2024, 10)) if rand_2024 else None
    rand_2024_p90 = float(np.percentile(rand_2024, 90)) if rand_2024 else None
    event_beta = rand_2024_med
    selection_alpha = None
    if v2_2024 is not None and rand_2024_med is not None:
        selection_alpha = float(v2_2024) - float(rand_2024_med)
    index_2024 = index.get("ret_2024")

    topk = {}
    for code in (BASELINE_V2_ALIAS, "R4_TOP2_diag", "R4_TOP3_diag"):
        report = load_report(code) or {}
        m = report.get("metrics") or {}
        nav = _load_nav(code)
        topk[code] = {
            **tracking_row(SPECS[code], report),
            "ret_2024": _year_nav_return(nav, 2024),
            "ret_2024_ex_share": None,
        }

    # F gate 预计算：低分差日 Top2 是否有非负 edge，且不是只靠 2024
    f_gate = _f_gate(v2_trades, _load_trades("R4_TOP2_diag"), v2m, (load_report("R4_TOP2_diag") or {}).get("metrics") or {})

    payload = {
        "written_at": datetime.now().astimezone().isoformat(),
        "yearly": yearly,
        "random_top1": {
            "n_seeds": len(RANDOM_SEEDS),
            "seeds": list(RANDOM_SEEDS),
            "rows": rand_rows,
            "cagr_median": float(np.median(rand_full)) if rand_full else None,
            "cagr_p10": float(np.percentile(rand_full, 10)) if rand_full else None,
            "cagr_p90": float(np.percentile(rand_full, 90)) if rand_full else None,
            "ret_2024_median": rand_2024_med,
            "ret_2024_p10": rand_2024_p10,
            "ret_2024_p90": rand_2024_p90,
        },
        "diagnostic_index": index,
        "topk": topk,
        "event_beta_vs_alpha": {
            "v2_2024": v2_2024,
            "random_top1_2024_median": rand_2024_med,
            "equal_weight_index_2024": index_2024,
            "event_beta": event_beta,
            "selection_alpha": selection_alpha,
            "alpha_share_of_2024": (
                float(selection_alpha) / float(v2_2024) if v2_2024 and selection_alpha is not None else None
            ),
            "note": (
                "Event Beta = 随机 Top1（同契约、不用 score）2024 收益中位；"
                "Selection Alpha = v2 2024 − 随机中位。等权指数标 DIAGNOSTIC INDEX。"
            ),
        },
        "f_gate": f_gate,
        "v2_metrics": {
            "cagr": v2m.get("annualized_return"),
            "sharpe": v2m.get("sharpe_zero_rate"),
            "max_drawdown": v2m.get("max_drawdown"),
            "cagr_ex2024": v2m.get("cagr_ex2024"),
            "completed_trades": v2m.get("completed_trades"),
        },
    }
    _write_json("01_attr/attribution.json", payload)
    _write_attr_md(payload)
    return payload


def _f_gate(top1_trades: pd.DataFrame, top2_trades: pd.DataFrame, top1_m: dict, top2_m: dict) -> dict[str, Any]:
    if top1_trades.empty or top2_trades.empty:
        return {"passed": False, "status": "INCOMPLETE", "reason": "缺少 Top1/Top2 诊断账本"}
    from .overlays import agreement_frame
    from ..stock_readonly import load_fs00263_ensemble_predictions

    ens = load_fs00263_ensemble_predictions()
    agree = agreement_frame(ens)
    t1 = top1_trades.copy()
    t2 = top2_trades.copy()
    t1["signal_date"] = pd.to_datetime(t1["signal_date"])
    t2["signal_date"] = pd.to_datetime(t2["signal_date"])
    t1 = t1.merge(agree.reset_index().rename(columns={"index": "signal_date"}), on="signal_date", how="left")
    t2 = t2.merge(agree.reset_index().rename(columns={"index": "signal_date"}), on="signal_date", how="left")
    low1 = t1.loc[pd.to_numeric(t1.get("margin_p250"), errors="coerce") <= 0.30]
    low2 = t2.loc[pd.to_numeric(t2.get("margin_p250"), errors="coerce") <= 0.30]
    e1 = float(pd.to_numeric(low1["ret"], errors="coerce").mean()) if len(low1) else None
    e2 = float(pd.to_numeric(low2["ret"], errors="coerce").mean()) if len(low2) else None
    edge = None if e1 is None or e2 is None else e2 - e1
    t1_ex = t1.loc[t1["signal_date"].dt.year != 2024]
    t2_ex = t2.loc[t2["signal_date"].dt.year != 2024]
    low1x = t1_ex.loc[pd.to_numeric(t1_ex.get("margin_p250"), errors="coerce") <= 0.30]
    low2x = t2_ex.loc[pd.to_numeric(t2_ex.get("margin_p250"), errors="coerce") <= 0.30]
    e1x = float(pd.to_numeric(low1x["ret"], errors="coerce").mean()) if len(low1x) else None
    e2x = float(pd.to_numeric(low2x["ret"], errors="coerce").mean()) if len(low2x) else None
    edge_ex = None if e1x is None or e2x is None else e2x - e1x
    only_2024 = bool(edge is not None and edge >= 0 and (edge_ex is None or edge_ex < 0))
    passed = bool(edge is not None and edge >= 0 and not only_2024)
    return {
        "passed": passed,
        "status": "PASS" if passed else "ABORTED_AFTER_DIAGNOSTIC",
        "low_margin_mean_ret_top1": e1,
        "low_margin_mean_ret_top2": e2,
        "edge": edge,
        "low_margin_mean_ret_top1_ex2024": e1x,
        "low_margin_mean_ret_top2_ex2024": e2x,
        "edge_ex2024": edge_ex,
        "only_2024": only_2024,
        "n_low_top1": int(len(low1)),
        "n_low_top2": int(len(low2)),
        "note": "低分差时 Top2 要有非负 edge，且不是只靠 2024。不过 → 不上 Top2。",
    }


def _write_attr_md(payload: dict[str, Any]) -> None:
    yearly = payload.get("yearly") or {}
    ev = payload.get("event_beta_vs_alpha") or {}
    rnd = payload.get("random_top1") or {}
    idx = payload.get("diagnostic_index") or {}
    fg = payload.get("f_gate") or {}
    lines = [
        "# 01 年份集中归因",
        "",
        f"- 生成时间：{payload.get('written_at')}",
        "- 对照：`baseline_v2` = `R4_00_baseline_v2` = `R3_03`",
        "- 性质：诊断，不调参。",
        "",
        "## 逐年拆解（v2）",
        "",
        "| 年 | 交易数 | 胜率 | 均收益 | 连板(D+1触) | 均乘子 | 均占用袖套 | Top1%贡献 | Top5% | Top10% | NAV收益 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for year in [str(y) for y in range(2018, 2027)]:
        row = yearly.get(year) or {}
        if not row:
            continue
        tag = " YTD" if row.get("ytd") else ""
        lines.append(
            f"| {year}{tag} | {row.get('n_trades')} | {_pct(row.get('win_rate'))} | {_pct(row.get('mean_ret'))} "
            f"| {_pct(row.get('continue_rate'))} | {_num(row.get('mean_alloc'))} | {_num(row.get('mean_live_sleeves'))} "
            f"| {_pct(row.get('top1pct_share'))} | {_pct(row.get('top5pct_share'))} | {_pct(row.get('top10pct_share'))} "
            f"| {_pct(row.get('nav_return'))} |"
        )
    lines += [
        "",
        "## Random Top1",
        "",
        f"同契约（入场/退出/费用/缺口/双袖套 + B01a×C03），不用 score。种子 {rnd.get('n_seeds')} 个。",
        "",
        f"| | CAGR | 2024 收益 |",
        f"|---|---:|---:|",
        f"| 中位 | {_pct(rnd.get('cagr_median'))} | {_pct(rnd.get('ret_2024_median'))} |",
        f"| p10 | {_pct(rnd.get('cagr_p10'))} | {_pct(rnd.get('ret_2024_p10'))} |",
        f"| p90 | {_pct(rnd.get('cagr_p90'))} | {_pct(rnd.get('ret_2024_p90'))} |",
        "",
        "## 候选等权指数（DIAGNOSTIC INDEX）",
        "",
        f"不可执行。CAGR {_pct(idx.get('cagr'))}，2024 {_pct(idx.get('ret_2024'))}。{idx.get('note')}",
        "",
        "## Event Beta vs Selection Alpha（2024）",
        "",
        f"| 项 | 值 |",
        f"|---|---:|",
        f"| v2 2024 | {_pct(ev.get('v2_2024'))} |",
        f"| Event Beta（随机 Top1 中位） | {_pct(ev.get('event_beta'))} |",
        f"| 等权指数 2024 | {_pct(ev.get('equal_weight_index_2024'))} |",
        f"| Selection Alpha（v2 − 随机中位） | {_pct(ev.get('selection_alpha'))} |",
        f"| Alpha 占 2024 收益 | {_pct(ev.get('alpha_share_of_2024'))} |",
        "",
        ev.get("note") or "",
        "",
        "## Top1 / Top2 / Top3 同契约",
        "",
        "| 代号 | 操作 | CAGR | Sharpe | MaxDD | 2024 | 笔数 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        BASELINE_V2_ALIAS: "Top1 30%",
        "R4_TOP2_diag": "Top2 15/15",
        "R4_TOP3_diag": "Top3 10/10/10",
    }
    for code, lab in labels.items():
        row = (payload.get("topk") or {}).get(code) or {}
        lines.append(
            f"| `{code}` | {lab} | {_pct(row.get('cagr'))} | {_num(row.get('sharpe'))} "
            f"| {_pct(row.get('max_drawdown'))} | {_pct(row.get('ret_2024'))} | {row.get('completed_trades')} |"
        )
    lines += [
        "",
        "## F 分散 Gate",
        "",
        f"状态：**{fg.get('status')}**。低分差 edge={_pct(fg.get('edge'))}，去2024 edge={_pct(fg.get('edge_ex2024'))}。",
        f"{fg.get('note')}",
        "",
        "归因写完才允许 `--stage all` 搜新策略。",
        "",
    ]
    path = ROUND4_ROOT / "01_year_concentration_attribution.md"
    assert_project_result_path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_attr_status() -> dict[str, Any]:
    path = ROUND4_ROOT / "01_year_concentration_attribution.md"
    js = ROUND4_ROOT / "01_attr" / "attribution.json"
    if not path.is_file() or not js.is_file():
        return {"passed": False, "reason": "归因未写完"}
    payload = json.loads(js.read_text(encoding="utf-8"))
    ok = bool(payload.get("yearly") and payload.get("random_top1", {}).get("rows") and payload.get("event_beta_vs_alpha"))
    return {"passed": ok, "f_gate": payload.get("f_gate"), "event_beta_vs_alpha": payload.get("event_beta_vs_alpha")}
