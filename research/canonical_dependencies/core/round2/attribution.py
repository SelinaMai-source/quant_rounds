"""P0：01_loss_attribution。重建锁定账本（因 Stock 无 nav/trades），再做亏损归因。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..local_executor import baseline_fixed_sleeve_config, run_local_fixed_sleeve_backtest, write_backtest_artifacts
from ..metrics import LOCKED_BASELINE
from ..paths import FS00263_FEATURES, FS00263_FEATURE_NAMES_ZH, assert_project_result_path, ensure_round2_tree, round2_dir
from ..stock_readonly import load_fs00263_ensemble_predictions, load_prepared_market
from .data import (
    load_all_seed_predictions,
    load_candidate_features,
    load_regime_frame,
    seed_rank_pct_std,
    seed_top1_agreement,
)
from .metrics import attach_round2_metrics, right_tail_contribution


def reconstruct_locked_ledger(
    *,
    predictions=None,
    prepared_market=None,
    market_meta=None,
) -> dict[str, Any]:
    dest = round2_dir("00_audit") / "locked_reconstruct"
    dest.mkdir(parents=True, exist_ok=True)
    report_path = dest / "report.json"
    if report_path.is_file() and (dest / "nav.parquet").is_file() and (dest / "trades.parquet").is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return {
            "status": "loaded",
            "out_dir": str(dest),
            "report": report,
        }
    if predictions is None:
        predictions = load_fs00263_ensemble_predictions()
    if prepared_market is None:
        loaded = load_prepared_market(warmup_ma5=True)
        prepared_market = loaded["prepared"]
        market_meta = loaded
    config = baseline_fixed_sleeve_config(exit_mode="close_from_d2")
    from ..paths import MARKET_END, PREDICTION_START

    result = run_local_fixed_sleeve_backtest(
        predictions, prepared_market, config=config,
        start_date=PREDICTION_START,
        end_date=MARKET_END,
    )
    extra = {
        "status": "reconstructed_locked_baseline",
        "experiment_code": "R2_B00",
        "note": "课题本地执行器按锁定契约重建；Stock 该变体只有 report.json。未写 Stock。",
        "snapshot_id": (market_meta or {}).get("snapshot_id"),
        "locked_display": LOCKED_BASELINE,
    }
    written = write_backtest_artifacts(result, dest, extra_report=extra)
    return {"status": "reconstructed", "out_dir": str(dest), "report": written["report"]}


def _drawdown_episodes(nav: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    ordered = nav.sort_values("trade_date").copy()
    ordered["trade_date"] = pd.to_datetime(ordered["trade_date"])
    values = ordered["total_asset"].astype(float)
    peak = values.cummax()
    dd = values / peak - 1.0
    ordered["dd"] = dd
    episodes = []
    in_dd = False
    start = None
    peak_val = None
    trough_val = None
    trough_date = None
    for row in ordered.itertuples(index=False):
        if row.dd < 0 and not in_dd:
            in_dd = True
            start = row.trade_date
            peak_val = float(row.total_asset) / (1.0 + float(row.dd)) if row.dd != -1 else float(row.total_asset)
            trough_val = float(row.total_asset)
            trough_date = row.trade_date
        elif in_dd:
            if float(row.total_asset) < trough_val:
                trough_val = float(row.total_asset)
                trough_date = row.trade_date
            if row.dd >= -1e-12:
                episodes.append({
                    "start": start,
                    "trough": trough_date,
                    "end": row.trade_date,
                    "depth": trough_val / peak_val - 1.0 if peak_val else None,
                })
                in_dd = False
    if in_dd:
        episodes.append({
            "start": start,
            "trough": trough_date,
            "end": ordered["trade_date"].iloc[-1],
            "depth": trough_val / peak_val - 1.0 if peak_val else None,
        })
    frame = pd.DataFrame(episodes)
    if frame.empty:
        return frame
    return frame.sort_values("depth").head(top_n)


def build_trade_attribution(trades: pd.DataFrame) -> pd.DataFrame:
    work = trades.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"])
    work["instrument_id"] = work["instrument_id"].astype(str)
    regime = load_regime_frame().reset_index()
    work = work.merge(regime, left_on="signal_date", right_on="trade_date", how="left", suffixes=("", "_reg"))
    ens = load_fs00263_ensemble_predictions()
    agree = seed_top1_agreement(ens, load_all_seed_predictions()).reset_index()
    work = work.merge(agree, on="signal_date", how="left")
    seeds = load_all_seed_predictions()
    std = seed_rank_pct_std(seeds)
    work = work.merge(std, on=["signal_date", "instrument_id"], how="left")
    panel = load_candidate_features()
    feat_cols = ["trade_date", "instrument_id", *list(FS00263_FEATURES), "label_d1_touch", "label_e2_net"]
    feat_cols = list(dict.fromkeys(feat_cols))
    work = work.merge(
        panel[feat_cols],
        left_on=["signal_date", "instrument_id"],
        right_on=["trade_date", "instrument_id"],
        how="left",
        suffixes=("", "_panel"),
    )
    # same-day feature percentiles among candidates
    for col in FS00263_FEATURES:
        if col in panel:
            pct = panel.groupby("trade_date")[col].rank(pct=True).rename(f"{col}_pct")
            tmp = panel[["trade_date", "instrument_id"]].copy()
            tmp[f"{col}_pct"] = pct.to_numpy()
            work = work.merge(
                tmp, left_on=["signal_date", "instrument_id"],
                right_on=["trade_date", "instrument_id"], how="left", suffixes=("", f"_{col}"),
            )
    if "open_gap" not in work.columns:
        # executor does not persist gap; leave missing
        work["open_gap"] = np.nan
    work["large_loss"] = pd.to_numeric(work["ret"], errors="coerce") <= -0.05
    work["seed_disagree"] = pd.to_numeric(work["agree_n"], errors="coerce") <= 2
    work["fail_high"] = pd.to_numeric(work.get("fail_ratio_p250"), errors="coerce") >= 0.80
    work["down_high"] = pd.to_numeric(work.get("down_count_p250"), errors="coerce") >= 0.80
    work["con_weak"] = pd.to_numeric(work.get("con_ratio_p250"), errors="coerce") <= 0.30
    work = work.loc[:, ~work.columns.duplicated()].copy()
    return work


def write_attribution_markdown(
    *,
    predictions=None,
    prepared_market=None,
    market_meta=None,
) -> Path:
    ensure_round2_tree()
    reconstructed = reconstruct_locked_ledger(
        predictions=predictions, prepared_market=prepared_market, market_meta=market_meta,
    )
    dest_dir = Path(reconstructed["out_dir"])
    nav = pd.read_parquet(dest_dir / "nav.parquet")
    trades = pd.read_parquet(dest_dir / "trades.parquet")
    report = reconstructed["report"]
    metrics = report.setdefault("metrics", {})
    attach_round2_metrics(nav, trades, metrics)

    attr = build_trade_attribution(trades)
    attr_path = ensure_round2_tree() / "attribution" / "locked_trades_attributed.parquet"
    assert_project_result_path(attr_path)
    attr.to_parquet(attr_path, index=False)

    episodes = _drawdown_episodes(nav, top_n=10)
    ep_path = ensure_round2_tree() / "01_attr" / "top10_drawdown_episodes.csv"
    assert_project_result_path(ep_path)
    episodes.to_csv(ep_path, index=False)

    tail = right_tail_contribution(trades)
    yearly = metrics.get("yearly_returns") or {}

    # episode environment
    ep_notes = []
    for row in episodes.itertuples(index=False):
        sl = attr.loc[
            (pd.to_datetime(attr["signal_date"]) >= pd.Timestamp(row.start))
            & (pd.to_datetime(attr["signal_date"]) <= pd.Timestamp(row.trough))
        ]
        ep_notes.append({
            "start": str(pd.Timestamp(row.start).date()),
            "trough": str(pd.Timestamp(row.trough).date()),
            "end": str(pd.Timestamp(row.end).date()),
            "depth": float(row.depth) if row.depth is not None else None,
            "n_trades": int(len(sl)),
            "mean_ret": float(sl["ret"].mean()) if len(sl) else None,
            "fail_high_share": float(sl["fail_high"].mean()) if len(sl) else None,
            "down_high_share": float(sl["down_high"].mean()) if len(sl) else None,
            "con_weak_share": float(sl["con_weak"].mean()) if len(sl) else None,
            "seed_disagree_share": float(sl["seed_disagree"].mean()) if len(sl) else None,
            "mean_agree_n": float(sl["agree_n"].mean()) if len(sl) and sl["agree_n"].notna().any() else None,
        })

    loss = attr.loc[attr["large_loss"]]
    all_n = max(len(attr), 1)
    loss_n = len(loss)
    explained = loss.loc[loss["fail_high"] | loss["down_high"] | loss["con_weak"] | loss["seed_disagree"]]
    hypothesis = (
        f"大亏（ret≤−5%）{loss_n}/{all_n} 笔。"
        f"其中炸板高/跌停高/CON 弱/种子≤2 一致 至少一条命中 {len(explained)} 笔"
        f"（{len(explained)/max(loss_n,1):.1%}）。"
    )
    if loss_n and len(explained) / loss_n >= 0.5:
        verdict = "用户判断得到部分支持：回撤簇和大亏里，FAIL/CON/跌停环境 + 5-seed disagreement 覆盖过半。"
    else:
        verdict = "用户判断未充分成立：大亏不能主要由 FAIL/CON/跌停 + seed disagreement 解释，还需看个股微观结构与右尾缺失。"

    # continuation / cost / exposure by year
    year_rows = []
    attr["year"] = pd.to_datetime(attr["signal_date"]).dt.year
    for year, block in attr.groupby("year"):
        year_rows.append({
            "year": int(year),
            "n": int(len(block)),
            "mean_ret": float(block["ret"].mean()),
            "win_rate": float((block["ret"] > 0).mean()),
            "con_rate": float(pd.to_numeric(block.get("label_d1_touch"), errors="coerce").mean())
            if "label_d1_touch" in block else None,
            "mean_cost_share": None,
            "mean_agree_n": float(block["agree_n"].mean()) if block["agree_n"].notna().any() else None,
        })

    lines = [
        "# 01 Loss attribution",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 锁定账本重建：`{dest_dir}` status={reconstructed['status']}",
        "- Stock 锁定变体目录只有 `report.json`，没有 nav/trades；本文件用课题 `local_executor` + 锁定契约重建，不写 Stock、不覆盖 Round 1。",
        "",
        "## 重建 vs 锁定展示",
        "",
        f"- 锁定展示：Sharpe {LOCKED_BASELINE['sharpe']} / CAGR {LOCKED_BASELINE['annualized_return']:.2%} / 累计 {LOCKED_BASELINE['total_return']:.2%} / MaxDD {LOCKED_BASELINE['max_drawdown']:.2%} / 2022 +{LOCKED_BASELINE['yearly_returns_pct']['2022']:.2f}%",
        f"- 重建 raw：Sharpe {metrics.get('sharpe_zero_rate')} / CAGR {metrics.get('annualized_return')} / 累计 {metrics.get('total_return')} / MaxDD {metrics.get('max_drawdown')}",
        f"- 重建分年：{ {k: round(v*100, 2) for k, v in yearly.items()} }",
        "",
        "## UP_NUM / FAIL_RATIO / CON 构造（Hermes 无现成字段名）",
        "",
        "- 主源：只读 cache `first_board_single_model_combined_2013_20260724_v1/market_context.parquet` + `first_board_market_limit_down_2013_20260724_v1/market_limit_down_context.parquet`。",
        "- `UP_NUM` = `market_limit_count`（收盘涨停家数）。合法构造：`daily_limit_events` 中 `limit_type=up` 且 `close_limit_value` 非空的当日计数。D 日收盘可知。",
        "- `FAIL_RATIO` = `market_failed_limit_rate` = 1 − 封成率；封成率 = 收盘仍涨停 / 当日曾触涨停。",
        "- `SUC` = `market_seal_success_rate`；`CON` = `prev_limit_relimit_rate`（已滞后一日，晋级率）。",
        "- 跌停 count = `market_close_limit_down_count`。",
        "- 对不上的研报字段（龙虎榜、完整 L2、新闻、最高连板高度精确梯队）标 **missing**，禁止假数据。",
        "- `open_gap`：锁定 trades 未落盘，本表记 missing。",
        "",
        "## Top 10 drawdown episodes",
        "",
        "| start | trough | end | depth | n | mean ret | fail高 | 跌停高 | CON弱 | seed≤2 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ep_notes:
        lines.append(
            "| {start} | {trough} | {end} | {depth:.2%} | {n_trades} | {mean_ret} | {fail_high_share} | {down_high_share} | {con_weak_share} | {seed_disagree_share} |".format(
                start=row["start"], trough=row["trough"], end=row["end"],
                depth=row["depth"] or 0,
                n_trades=row["n_trades"],
                mean_ret="—" if row["mean_ret"] is None else f"{row['mean_ret']:.2%}",
                fail_high_share="—" if row["fail_high_share"] is None else f"{row['fail_high_share']:.0%}",
                down_high_share="—" if row["down_high_share"] is None else f"{row['down_high_share']:.0%}",
                con_weak_share="—" if row["con_weak_share"] is None else f"{row['con_weak_share']:.0%}",
                seed_disagree_share="—" if row["seed_disagree_share"] is None else f"{row['seed_disagree_share']:.0%}",
            )
        )
    lines += [
        "",
        "## 右尾贡献",
        "",
        f"- 完成交易 {tail['n_trades']}，收益加总 {tail['sum_ret']}",
        f"- Top1% 贡献份额 {tail['top1pct_share']}；去掉后剩余加总 {tail['without_top1pct_sum']}",
        f"- Top5% 贡献份额 {tail['top5pct_share']}；去掉后剩余加总 {tail['without_top5pct_sum']}",
        f"- Top10% 贡献份额 {tail['top10pct_share']}；去掉后剩余加总 {tail['without_top10pct_sum']}",
        "",
        "## 2018–2026 分年",
        "",
        "| year | n | mean ret | win | CON(label_d1) | mean agree | NAV year |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in year_rows:
        yret = yearly.get(str(row["year"]))
        lines.append(
            f"| {row['year']} | {row['n']} | {row['mean_ret']:.2%} | {row['win_rate']:.1%} | "
            f"{'—' if row['con_rate'] is None else f'{row['con_rate']:.1%}'} | "
            f"{'—' if row['mean_agree_n'] is None else f'{row['mean_agree_n']:.2f}'} | "
            f"{'—' if yret is None else f'{yret:.2%}'} |"
        )
    lines += [
        "",
        "## 假设检验：炸板 / 晋级失败 / 跌停 + seed disagreement",
        "",
        hypothesis,
        "",
        verdict,
        "",
        "fs00263 六因子原始值与当日分位已写入 `result/round2/attribution/locked_trades_attributed.parquet`。",
        f"中文名：{FS00263_FEATURE_NAMES_ZH}",
        "",
        "## C1 诊断（不锁参数）",
        "",
        f"- 全样本 seed_rank_pct_std 均值：{float(attr['seed_rank_pct_std'].mean()) if attr['seed_rank_pct_std'].notna().any() else 'missing'}",
        f"- 大亏单 seed_rank_pct_std 均值：{float(loss['seed_rank_pct_std'].mean()) if loss_n and loss['seed_rank_pct_std'].notna().any() else 'missing'}",
        f"- 大亏 agree_n 均值：{float(loss['agree_n'].mean()) if loss_n and loss['agree_n'].notna().any() else 'missing'} vs 全样本 {float(attr['agree_n'].mean()) if attr['agree_n'].notna().any() else 'missing'}",
        "",
    ]
    dest = ensure_round2_tree() / "01_loss_attribution.md"
    assert_project_result_path(dest)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "reconstructed": reconstructed["status"],
        "episodes": ep_notes,
        "right_tail": tail,
        "yearly_nav": yearly,
        "hypothesis": hypothesis,
        "verdict": verdict,
        "attr_path": str(attr_path),
    }
    json_path = ensure_round2_tree() / "01_attr" / "summary.json"
    assert_project_result_path(json_path)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return dest
