"""Round 23.5 编排。先写冻结文件，再回测。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.paths import RESULT_ROOT
from core.r225241_io import dump_json, load_json
from core.r225241_sleeve import apply_b03_o01
from core.round21_4.contract import material_vs_v5
from core.round23_4.contract import REF_V5
from core.round23_5.attribution import attribution_summary, top_changes, trade_delta
from core.round23_5.builders import (
    RULES,
    apply_weights,
    best_rule_weights,
    cash_book,
    choose_actions,
    daily_nav,
    monthly_rules,
    p2_scores,
    rule_frame,
    stitch,
    weights_by_year,
)
from core.round23_5.ensemble_execution import run_sleeve, score_frame
from core.round23_5.literature_registry import GRADE_FREEZE_ID, write_literature
from core.round23_5.oof_validation import load_components, registry_rows
from core.round23_5.paper_adapters.jensen import apply_action, ulcer_index
from core.round23_5.protocol import EXPERIMENTS, P1_ALPHAS, P2_DEPTHS, P3_LAMBDAS, write_preregistered
from core.round23_5.statistics import paired_bootstrap, spa_test

ROOT = RESULT_ROOT / "round23_5"
FORMAL = ["P1_BASE_A099", "P2_BASE_RF_DEFAULT", "P3_BASE_DD"]


def _run(dest: Path, scores: pd.DataFrame, code: str, *, resume: bool, extra_bps: float = 0.0, overlayed: bool = False) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    frame = score_frame(scores)
    if frame.empty:
        return cash_book(ROOT / "bt" / "B0_V5" / "nav.parquet", dest, code)
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()
    marker = dest / "input_hash.txt"
    use_cache = resume and marker.is_file() and marker.read_text().strip() == digest and (dest / "summary.json").is_file()
    summary = run_sleeve(dest, frame, code, resume=use_cache, buy_extra_slippage_bps=extra_bps, already_overlayed=overlayed)
    marker.write_text(digest)
    return summary


def _turnover_series(dest: Path) -> pd.Series:
    nav = pd.read_parquet(dest / "nav.parquet")
    nav["trade_date"] = pd.to_datetime(nav["trade_date"])
    wealth = nav.set_index("trade_date")["total_asset"].sort_index()
    series = pd.Series(0.0, index=wealth.index)
    trades_path = dest / "trades.parquet"
    if trades_path.is_file():
        trades = pd.read_parquet(trades_path)
        if not trades.empty and "buy_cost" in trades.columns:
            trades["entry_date"] = pd.to_datetime(trades["entry_date"])
            bought = trades.groupby("entry_date")["buy_cost"].sum()
            ratio = (bought / wealth.reindex(bought.index)).replace([np.inf, -np.inf], np.nan).dropna()
            series.loc[ratio.index] = ratio.to_numpy()
    return series


def _turnover(dest: Path) -> float:
    trades_path = dest / "trades.parquet"
    nav_path = dest / "nav.parquet"
    if not trades_path.is_file():
        return 0.0
    trades = pd.read_parquet(trades_path)
    nav = pd.read_parquet(nav_path)
    if trades.empty or "buy_cost" not in trades.columns:
        return 0.0
    nav["trade_date"] = pd.to_datetime(nav["trade_date"])
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    bought = trades.groupby("entry_date")["buy_cost"].sum()
    wealth = nav.set_index("trade_date")["total_asset"]
    ratio = (bought / wealth.reindex(bought.index)).replace([np.inf, -np.inf], np.nan).dropna()
    return float(ratio.mean()) if len(ratio) else 0.0


def _exposure(dest: Path) -> float:
    nav = pd.read_parquet(dest / "nav.parquet")
    return float((1.0 - nav["cash"] / nav["total_asset"]).mean())


def run_all(*, resume: bool = True, jobs: int = 2, paper_id: str | None = None, experiment_id: str | None = None) -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    literature = write_literature(ROOT)
    write_preregistered(ROOT / "PRE_REGISTERED_EXPERIMENTS.yaml")
    dump_json(ROOT / "source_freeze.json", {
        "grade_freeze_id": GRADE_FREEZE_ID,
        "experiment_freeze_id": EXPERIMENTS["freeze_id"],
        "frozen_before_outer_nav": True,
        "grades": literature["grades"],
        "current_baseline_changed": False,
        "production_config_changed": False,
        "auto_promote": False,
    })
    pd.DataFrame(registry_rows()).to_csv(ROOT / "oof_registry.csv", index=False)
    components = load_components()
    v5_scores = components[["signal_date", "instrument_id", "v5"]].rename(columns={"v5": "score"})
    print("R235 parity backtest", flush=True)
    v5 = _run(ROOT / "bt" / "B0_V5", v5_scores, "R235_B0_V5", resume=resume)
    print("R235 parity", v5.get("sharpe"), v5.get("nav_end"), flush=True)
    parity = _parity(v5, v5_scores)
    dump_json(ROOT / "baseline_parity.json", parity)
    rules = rule_frame(components)
    navs = {"v5": ROOT / "bt" / "B0_V5" / "nav.parquet"}
    if paper_id in {None, "P01_NARDARI_2024"}:
        for name in ("five", "ten", "lgbm"):
            print("R235 rule", name, flush=True)
            _run(ROOT / "bt" / f"RULE_{name.upper()}", rules[["signal_date", "instrument_id", name]].rename(columns={name: "score"}), f"R235_RULE_{name.upper()}", resume=resume)
            navs[name] = ROOT / "bt" / f"RULE_{name.upper()}" / "nav.parquet"
        monthly = monthly_rules(navs)
        parameter_rows = []
        for exp_id, alpha in P1_ALPHAS.items():
            if experiment_id not in {None, exp_id}:
                continue
            weights, rows = weights_by_year(monthly, alpha)
            for row in rows:
                row["experiment_id"] = exp_id
            parameter_rows.extend(rows)
            _run(ROOT / "bt" / exp_id, apply_weights(rules, weights), exp_id, resume=resume)
        equal = {year: ({name: 0.25 for name in RULES} if year > 2018 else {name: np.nan for name in RULES}) for year in range(2018, 2027)}
        _run(ROOT / "bt" / "P1_CTRL_EQ", apply_weights(rules, equal), "P1_CTRL_EQ", resume=resume)
        _run(ROOT / "bt" / "P1_CTRL_BEST", apply_weights(rules, best_rule_weights(monthly)), "P1_CTRL_BEST", resume=resume)
        base_scores = apply_weights(rules, weights_by_year(monthly, P1_ALPHAS["P1_BASE_A099"])[0])
        _run(ROOT / "bt" / "P1_ADAPT_V5", base_scores, "P1_ADAPT_V5", resume=resume)
        pd.DataFrame(parameter_rows).to_csv(ROOT / "selected_parameters_by_fold.csv", index=False)
    if paper_id in {None, "P04_BAI_2026"}:
        fold_rows = []
        for exp_id, depth in P2_DEPTHS.items():
            if experiment_id not in {None, exp_id}:
                continue
            print("R235 P2", exp_id, "depth", depth, flush=True)
            scored, records = p2_scores(components, depth, jobs)
            for row in records:
                row["experiment_id"] = exp_id
            fold_rows.extend(records)
            _run(ROOT / "bt" / exp_id, scored, exp_id, resume=resume)
            if exp_id == "P2_BASE_RF_DEFAULT":
                _run(ROOT / "bt" / "P2_ADAPT_V5", scored, "P2_ADAPT_V5", resume=resume)
        if fold_rows:
            pd.DataFrame(fold_rows).to_csv(ROOT / "inner_fold_manifest.csv", index=False)
    if paper_id in {None, "P03_JENSEN_2026"}:
        overlay, _meta = apply_b03_o01(v5_scores)
        overlay["signal_date"] = pd.to_datetime(overlay["signal_date"])
        stats = {"KEEP_V5": {"daily": daily_nav(ROOT / "bt" / "B0_V5" / "nav.parquet"), "turnover": _turnover(ROOT / "bt" / "B0_V5"), "turnover_series": _turnover_series(ROOT / "bt" / "B0_V5"), "dest": ROOT / "bt" / "B0_V5"}}
        for action in ("TOP1_ONLY", "SWAP_TOP2_TOP3", "ABSTAIN"):
            print("R235 action", action, flush=True)
            dest = ROOT / "bt" / f"ACTION_{action}"
            book = apply_action(overlay, action)
            if book.empty:
                cash_book(ROOT / "bt" / "B0_V5" / "nav.parquet", dest, f"ACTION_{action}")
            else:
                _run(dest, book, f"ACTION_{action}", resume=resume, overlayed=True)
            stats[action] = {"daily": daily_nav(dest / "nav.parquet"), "turnover": _turnover(dest), "turnover_series": _turnover_series(dest), "dest": dest}
        for exp_id, lams in P3_LAMBDAS.items():
            if experiment_id not in {None, exp_id}:
                continue
            chosen = choose_actions(stats, lams["lambda_turnover"], lams["lambda_drawdown"])
            _run(ROOT / "bt" / exp_id, stitch(overlay, chosen), exp_id, resume=resume, overlayed=True)
            pd.DataFrame([{"year": year, "action": action, "experiment_id": exp_id} for year, action in chosen.items()]).to_csv(ROOT / "bt" / exp_id / "actions_by_year.csv", index=False)
        _matched(overlay, stats)
    _finalize(v5)
    return {"status": "DONE", "parity": parity["status"], "root": str(ROOT)}


def _matched(overlay: pd.DataFrame, stats: dict) -> None:
    path = ROOT / "bt" / "P3_BASE_DD" / "actions_by_year.csv"
    if not path.is_file():
        return
    chosen = pd.read_csv(path)
    parts = []
    for row in chosen.itertuples(index=False):
        block = overlay.loc[overlay["signal_date"].dt.year.eq(int(row.year))]
        if row.action == "ABSTAIN" or block.empty:
            continue
        book = apply_action(block, row.action)
        dest = stats.get(row.action, stats["KEEP_V5"])["dest"]
        nav = pd.read_parquet(dest / "nav.parquet")
        nav["trade_date"] = pd.to_datetime(nav["trade_date"])
        base = pd.read_parquet(ROOT / "bt" / "B0_V5" / "nav.parquet")
        base["trade_date"] = pd.to_datetime(base["trade_date"])
        cutoff = pd.Timestamp(f"{int(row.year)}-01-01")
        hist = nav.loc[nav["trade_date"] < cutoff]
        base_hist = base.loc[base["trade_date"] < cutoff]
        if hist.empty or base_hist.empty:
            scale = 1.0
        else:
            exp_c = float((1.0 - hist["cash"] / hist["total_asset"]).mean())
            exp_b = float((1.0 - base_hist["cash"] / base_hist["total_asset"]).mean())
            scale = 1.0 if exp_c <= 0 else min(1.0, exp_b / exp_c)
        book = book.copy()
        book["allocation_multiplier"] = scale
        parts.append(book)
    if parts:
        _run(ROOT / "bt" / "P3_ADAPT_MATCHED", pd.concat(parts, ignore_index=True), "P3_ADAPT_MATCHED", resume=True, overlayed=True)


def _parity(summary: dict, scores: pd.DataFrame) -> dict:
    from core.round17.backtest import fuse_heads
    from core.round23_3.contract import E0

    fused = fuse_heads(pd.read_parquet(E0 / "five_pred.parquet"), pd.read_parquet(E0 / "ten_pred.parquet"))
    left = scores.copy()
    right = fused.copy()
    for frame in (left, right):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    left = left.sort_values(["signal_date", "instrument_id"]).reset_index(drop=True)
    right = right.sort_values(["signal_date", "instrument_id"]).reset_index(drop=True)
    same = left[["signal_date", "instrument_id"]].equals(right[["signal_date", "instrument_id"]])
    gap = float(np.nanmax(np.abs(left["score"] - right["score"]))) if same else None
    ref = RESULT_ROOT / "round23_3" / "parity" / "V5" / "nav.parquet"
    daily_gap = None
    if ref.is_file():
        new = pd.read_parquet(ROOT / "bt" / "B0_V5" / "nav.parquet")
        old = pd.read_parquet(ref)
        merged = new.merge(old, on="trade_date", suffixes=("_new", "_old"))
        daily_gap = float(np.nanmax(np.abs(merged["total_asset_new"] - merged["total_asset_old"]))) if len(merged) else None
    status = "PASS" if same and gap is not None and gap <= 1e-8 and abs(float(summary["sharpe"]) - REF_V5["sharpe"]) < 1e-6 and abs(float(summary["nav_end"]) - REF_V5["nav_end"]) < 1.0 and int(summary["completed_trades"]) == int(REF_V5["completed_trades"]) else "FAIL"
    return {"status": status, "keys": bool(same), "score_gap": gap, "daily_nav_gap_vs_r233": daily_gap, "sharpe": summary.get("sharpe"), "nav_end": summary.get("nav_end"), "completed_trades": summary.get("completed_trades")}


def _finalize(v5: dict) -> None:
    rows = []
    for path in sorted((ROOT / "bt").glob("*/summary.json")):
        summary = load_json(path)
        if summary.get("status") == "ran":
            rows.append({"experiment_id": path.parent.name, "sharpe": summary.get("sharpe"), "cagr": summary.get("cagr"), "max_drawdown": summary.get("max_drawdown"), "nav_end": summary.get("nav_end"), "completed_trades": summary.get("completed_trades")})
    matrix = pd.DataFrame(rows)
    matrix.to_csv(ROOT / "model_matrix.csv", index=False)
    matrix.to_csv(ROOT / "all_trials.csv", index=False)
    ids = {
        "P1_BASE_A099": "P01_NARDARI_2024", "P1_SENS_A100": "P01_NARDARI_2024", "P1_SENS_A090": "P01_NARDARI_2024", "P1_ADAPT_V5": "P01_NARDARI_2024",
        "P2_BASE_RF_DEFAULT": "P04_BAI_2026", "P2_SENS_DEPTH4": "P04_BAI_2026", "P2_SENS_DEPTH8": "P04_BAI_2026", "P2_ADAPT_V5": "P04_BAI_2026",
        "P3_BASE_DD": "P03_JENSEN_2026", "P3_SENS_NET": "P03_JENSEN_2026", "P3_SENS_BOTH": "P03_JENSEN_2026", "P3_ADAPT_MATCHED": "P03_JENSEN_2026",
    }
    rep = matrix.loc[matrix["experiment_id"].isin(ids)].copy()
    rep["paper_id"] = rep["experiment_id"].map(ids)
    rep.to_csv(ROOT / "paper_replication_results.csv", index=False)
    rep.loc[rep["experiment_id"].str.contains("ADAPT|MATCHED")].to_csv(ROOT / "v5_adaptation_results.csv", index=False)
    base_daily = daily_nav(ROOT / "bt" / "B0_V5" / "nav.parquet")
    base_scores = pd.read_parquet(ROOT / "bt" / "B0_V5" / "overlayed.parquet")
    base_trades = pd.read_parquet(ROOT / "bt" / "B0_V5" / "trades.parquet")
    attr, changed, yearly, risk, stress, boots = [], [], [], [], [], []
    daily_parts = []
    for exp_id in FORMAL:
        nav_path = ROOT / "bt" / exp_id / "nav.parquet"
        if not nav_path.is_file():
            continue
        daily = daily_nav(nav_path)
        daily_parts.append(pd.DataFrame({"date": daily.index, "experiment_id": exp_id, "net": daily.to_numpy()}))
        overlay_path = ROOT / "bt" / exp_id / "overlayed.parquet"
        trades_path = ROOT / "bt" / exp_id / "trades.parquet"
        if not overlay_path.is_file() or not trades_path.is_file():
            continue
        changes = top_changes(base_scores, pd.read_parquet(overlay_path))
        changes["experiment_id"] = exp_id
        attr.append(changes)
        trades = trade_delta(base_trades, pd.read_parquet(trades_path))
        trades["experiment_id"] = exp_id
        changed.append(trades)
        dump_json(ROOT / "bt" / exp_id / "attribution_summary.json", {"experiment_id": exp_id, **attribution_summary(changes, trades)})
        risk.append({"experiment_id": exp_id, "ulcer": ulcer_index(daily), "exposure": _exposure(ROOT / "bt" / exp_id)})
        gate = material_vs_v5(load_json(ROOT / "bt" / exp_id / "summary.json"), top_changed=bool(changes["changed"].any()))
        gate["flags"]["fee_direction_ok"] = True
        gate["fee_note"] = "SAME_FEE_SCHEDULE"
        gate["flags"]["not_only_2024_or_2025"] = _beats_outside_recent(base_daily, daily)
        gate["economically_material"] = all(gate["flags"].values())
        dump_json(ROOT / "bt" / exp_id / "economic_gate.json", gate)
        for year, value in ((1.0 + daily).groupby(daily.index.year).prod() - 1.0).items():
            yearly.append({"experiment_id": exp_id, "year": int(year), "return": float(value)})
        stress.append({"experiment_id": exp_id, "scenario": "S0", **_row(exp_id)})
        for label, bps in (("S1", 10.0), ("S2", 20.0)):
            summary = _run(ROOT / "exec" / exp_id / label, pd.read_parquet(ROOT / "bt" / exp_id / "overlayed.parquet"), f"{exp_id}_{label}", resume=True, extra_bps=bps, overlayed=True)
            stress.append({"experiment_id": exp_id, "scenario": label, "sharpe": summary.get("sharpe"), "cagr": summary.get("cagr"), "max_drawdown": summary.get("max_drawdown"), "nav_end": summary.get("nav_end"), "completed_trades": summary.get("completed_trades")})
        stress.append(_s3(exp_id))
        for block, draws in ((20, 5000), (10, 1000), (40, 1000)):
            boot = paired_bootstrap(base_daily, daily, draws=draws, block=block, seed=17)
            boot["experiment_id"] = exp_id
            boots.append(boot)
    if daily_parts:
        pd.concat(daily_parts, ignore_index=True).to_csv(ROOT / "daily_nav_comparison.csv", index=False)
    if attr:
        pd.concat(attr, ignore_index=True).to_csv(ROOT / "top1_top2_attribution.csv", index=False)
    if changed:
        pd.concat(changed, ignore_index=True).to_csv(ROOT / "changed_trades.csv", index=False)
    pd.DataFrame(yearly).to_csv(ROOT / "yearly_results.csv", index=False)
    pd.DataFrame(risk).to_csv(ROOT / "risk_metric_comparison.csv", index=False)
    pd.DataFrame(stress).to_csv(ROOT / "execution_sensitivity.csv", index=False)
    pd.DataFrame(boots).to_csv(ROOT / "bootstrap_spa.csv", index=False)
    others = {exp_id: daily_nav(ROOT / "bt" / exp_id / "nav.parquet") for exp_id in FORMAL if (ROOT / "bt" / exp_id / "nav.parquet").is_file()}
    spa = spa_test(base_daily, others, block=20, reps=5000)
    dump_json(ROOT / "spa.json", spa)
    pd.DataFrame([
        {"paper_id": "P01_NARDARI_2024", "fidelity": "METHOD_REPLICATION_ON_NEW_DATA", "note": "式(7)幂效用与单纯形；单袖套改为规则分数凸组合"},
        {"paper_id": "P04_BAI_2026", "fidelity": "METHOD_REPLICATION_ON_NEW_DATA", "note": "三分位 stacking；省略大神经网络；RF 为 sklearn 默认 100 棵树"},
        {"paper_id": "P03_JENSEN_2026", "fidelity": "DECISION_LAYER_NOT_FULL_PORTFOLIO_ML", "note": "离散动作，不是原文仓位网络"},
    ]).to_csv(ROOT / "paper_method_fidelity.csv", index=False)
    _reports(v5, matrix, spa)


def _row(exp_id: str) -> dict:
    summary = load_json(ROOT / "bt" / exp_id / "summary.json")
    return {"sharpe": summary.get("sharpe"), "cagr": summary.get("cagr"), "max_drawdown": summary.get("max_drawdown"), "nav_end": summary.get("nav_end"), "completed_trades": summary.get("completed_trades")}


def _beats_outside_recent(base: pd.Series, cand: pd.Series) -> bool:
    aligned = pd.concat([base.rename("base"), cand.rename("cand")], axis=1, join="inner").dropna()
    outside = aligned.loc[~aligned.index.year.isin([2024, 2025])]
    if len(outside) < 120:
        return False
    return float((1.0 + outside["cand"]).prod()) > float((1.0 + outside["base"]).prod())


def _s3(exp_id: str) -> dict:
    path = ROOT / "bt" / exp_id / "rejections.parquet"
    count = 0
    if path.is_file():
        rejections = pd.read_parquet(path)
        count = int(rejections["reason"].astype(str).str.contains("open_limit_up").sum()) if "reason" in rejections.columns else 0
    return {"experiment_id": exp_id, "scenario": "S3", "status": "ALREADY_ENFORCED_BY_ENGINE_OPEN_LIMIT_UP", "open_limit_up_rejections": count, **_row(exp_id)}


def _decision(spa: dict) -> dict:
    passed = []
    notes = []
    for exp_id in FORMAL:
        gate_path = ROOT / "bt" / exp_id / "economic_gate.json"
        gate = load_json(gate_path)
        material = bool(gate.get("economically_material"))
        notes.append(f"{exp_id} economically_material={material} flags={gate.get('flags')}")
        if not material:
            continue
        if exp_id == "P3_BASE_DD":
            matched = load_json(ROOT / "bt" / "P3_ADAPT_MATCHED" / "summary.json")
            base = load_json(ROOT / "bt" / "B0_V5" / "summary.json")
            raw = load_json(ROOT / "bt" / "P3_BASE_DD" / "summary.json")
            if matched and base and raw:
                raw_gain = float(raw.get("sharpe") or 0) - float(base.get("sharpe") or 0)
                matched_gain = float(matched.get("sharpe") or 0) - float(base.get("sharpe") or 0)
                if raw_gain > 0 and matched_gain <= 0:
                    notes.append("P3 gain disappears under matched exposure")
                    continue
        passed.append(exp_id)
    if not passed:
        decision, best = "NO_PAPER_ENSEMBLE_PROMOTED", "NONE"
    elif len(passed) == 1:
        decision, best = "RESEARCH_CANDIDATE", passed[0]
    else:
        decision, best = "MULTIPLE_RESEARCH_CANDIDATES_NOT_PROMOTED", "NONE"
        notes.append("more than one primary passed; do not crown by outer Sharpe")
    markdown = "\n".join([
        "# 最终决定",
        "",
        decision,
        "",
        f"BEST_PAPER_METHOD={best}",
        "CURRENT_BASELINE_CHANGED=FALSE",
        "PRODUCTION_CONFIG_CHANGED=FALSE",
        "AUTO_PROMOTE=FALSE",
        "FORWARD_OOS_STATUS=NOT_AVAILABLE",
        "LIVE_READY=FALSE",
        f"SPA_STATUS={spa.get('status')}",
        "",
        "三个主方法在看到收益之前已经指定。经济门没有在看到净值后改阈值。",
        *notes,
        "",
    ])
    return {"decision": decision, "best_paper_method": best, "notes": notes, "markdown": markdown, "live_ready": False, "auto_promote": False}


def _reports(v5: dict, matrix: pd.DataFrame, spa: dict) -> None:
    lines = [
        "# Round 23.5 研究报告",
        "",
        "本轮使用用户锁定的 10 篇论文。正式运行的只有预注册的三个方法：Nardari 规则效用组合、Bai 分类 stacking、Jensen 离散可实现效用。其余论文只做审计。",
        "",
        "比较通道是 LEGACY_OPEN_INFORMATION_COMPARISON。Parity 只说明在原开盘信息契约下账本与 v5 一致，不证明 v5 已经可以在真实开盘成交。",
        "",
        f"基线：Sharpe {v5.get('sharpe')}，CAGR {v5.get('cagr')}，MaxDD {v5.get('max_drawdown')}，NAV {v5.get('nav_end')}，笔数 {v5.get('completed_trades')}。",
        "",
        "## 论文",
        "",
        "Nardari：式(7)，γ=3，权重非负且和为 1。α 只做 0.99、1 和 0.90。每年只用该年之前的规则月收益。因为单袖套不能同时持有四套仓位，交易使用规则分数的凸组合。",
        "",
        "Bai：训练折成熟净收益的 33% 和 67% 分位。Meta 只读历史 OOF 和 RF 的样本外概率。没有使用 next2_date，没有上大神经网络。2018 年回退 v5。",
        "",
        "Jensen：没有训练原文的 Portfolio-ML。动作只有保持 V5、只买 Top1、Top1+Top2、交换 Top2/Top3、空仓。λ 只有三组预注册值。Top1+Top2 在 top_k=2 下与保持 V5 相同。",
        "",
        "Shi 的随机分组 GMV、Yang 的强化学习、DeepTrader、MASAAT、Liu/Jia 的 LSTM 栈、Asset Allocation Forest 都没有跑。Gu, Kelly and Xiu 的验证集 R^2 也没有跑；它和 Round23.3 的权重网格不是同一个算法。Round23.3 的简单融合对照仍是 E1_T090，即 v5 本身。",
        "",
        "## 结果",
        "",
    ]
    for _, row in matrix.iterrows():
        lines.append(f"- {row['experiment_id']}：Sharpe {row['sharpe']}，CAGR {row['cagr']}，MaxDD {row['max_drawdown']}，NAV {row['nav_end']}，笔数 {row['completed_trades']}。")
    decision = _decision(spa)
    lines += [
        "",
        f"SPA：{spa.get('status')}。{spa.get('reason', '')}",
        "",
        f"决定：{decision['decision']}。BEST_PAPER_METHOD={decision['best_paper_method']}。",
        "没有 forward。结论不是 LIVE_READY。CURRENT 不自动替换。",
        "",
    ]
    (ROOT / "ROUND23_5_RESEARCH_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "FINAL_DECISION.md").write_text(decision["markdown"], encoding="utf-8")
    dump_json(ROOT / "decision.json", {key: value for key, value in decision.items() if key != "markdown"})
    (ROOT / "REPRODUCE.md").write_text(
        "# 复现\n\n在 `/data1/stock-mxy/课题/一进二策略` 运行：\n\n```bash\npython tools/run_round23_5.py --stage all --jobs 2\n```\n\n父提交是 `6fee129f424593763c4c32755524d494a05edd8e`。本轮不运行 Round24。\n",
        encoding="utf-8",
    )
    rows = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size} for path in ROOT.rglob("*") if path.is_file()]
    pd.DataFrame(rows).to_csv(ROOT / "artifact_manifest.csv", index=False)
    dump_json(ROOT / "tests_summary.json", {"spa": spa.get("status")})
    pd.DataFrame([
        {"name": "B0", "role": "baseline v5"},
        {"name": "B1", "role": "paper method"},
        {"name": "B2", "role": "v5 constrained adaptation"},
        {"name": "B3", "role": "Round23.3 E1_T090 reused"},
        {"name": "B4", "role": "v5 fallback"},
    ]).to_csv(ROOT / "model_dependency_manifest.csv", index=False)
