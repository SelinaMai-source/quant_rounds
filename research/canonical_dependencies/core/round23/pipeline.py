"""Round 23。模型与规则拆开。M0 仍是 v5。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.fold_ranker import train_ranker_head
from core.limit_rules import board_of, limit_pct
from core.paths import RESULT_ROOT
from core.research_io import dump_json, load_json, write_df, write_text
from core.research_sleeve import run_heads_sleeve, run_overlay_sleeve, run_scored_sleeve
from core.round17.backtest import fuse_heads
from core.round22_3.contract import (
    ECONOMIC_MATERIALITY,
    FEE_SCHEDULE,
    FIVE_FEATURES,
    MAIN_WINDOW_END,
    MAIN_WINDOW_START,
    TEN_FEATURES,
    V5_ANCHOR,
)

ROOT = RESULT_ROOT / "round23"
R22_3 = RESULT_ROOT / "round22_3"
R22_4 = RESULT_ROOT / "round22_4"
E0_FIVE = R22_3 / "candidates" / "R223_E0" / "five_pred.parquet"
E0_TEN = R22_3 / "candidates" / "R223_E0" / "ten_pred.parquet"
E0_NAV = R22_3 / "candidates" / "R223_E0" / "nav.parquet"
E0_OVERLAY = R22_3 / "candidates" / "R223_E0" / "overlayed.parquet"
PANEL = R22_3 / "data" / "five_candidate.parquet"
TRAIN = R22_3 / "data" / "five_training.parquet"
TEN_TRAIN = R22_3 / "data" / "ten_training.parquet"
TEN_CAND = R22_3 / "data" / "ten_candidate.parquet"
BREADTH = RESULT_ROOT / "round22" / "data" / "market_daily.parquet"
KEEP = "KEEP_BASELINE_V5"
RECOMMEND = "RECOMMEND_MODEL_RULE_RESEARCH"
SEED012 = (0, 1, 2)
STACK_RHO = 0.85
WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
RULES = (
    {"code": "R23_R1_BREADTH", "kind": "environment", "quantile": 0.20, "scenario": False},
    {"code": "R23_R2_LIQUIDITY", "kind": "liquidity", "quantile": 0.20, "scenario": False},
    {"code": "R23_R3_OPEN_SEAL", "kind": "same_open_optimism", "scenario": True},
)


def _save(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    dump_json(ROOT / "_state" / f"{name}.json", payload)
    return payload


def _cached(name: str) -> dict[str, Any] | None:
    payload = load_json(ROOT / "_state" / f"{name}.json")
    if payload.get("status") in {"DONE", "PARTIAL", "BLOCKED"}:
        return payload
    return None


def run_freeze(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and (ROOT / "frozen_manifest.json").is_file():
        return load_json(ROOT / "frozen_manifest.json")
    r224 = load_json(R22_4 / "decision.json")
    body = {
        "round": "23",
        "frozen_before_new_nav": True,
        "m0": "baseline_v5=R16_O01",
        "r0": "O01 + B03 + 90/10 rank fusion",
        "round22_4_decision": r224.get("final_decision"),
        "m22_4_replaces_m0": False,
        "fee_schedule": FEE_SCHEDULE,
        "window": [MAIN_WINDOW_START, MAIN_WINDOW_END],
        "economic_gate_id": ECONOMIC_MATERIALITY["id"],
        "seeds": list(SEED012),
        "forbid_pick_best_seed": True,
        "primary_seed_aggregate": "mean_of_rank_pct",
        "secondary_seed_aggregate": "median_of_rank_pct",
        "weight_grid": list(WEIGHT_GRID),
        "weight_objective": "inner-fold mean daily spearman vs label_e2_net; not portfolio Sharpe",
        "stacking_stop_if_abs_rank_corr_ge": STACK_RHO,
        "rules": list(RULES),
        "rule_thresholds_fit_on": "inner years only",
        "joint_requires": "model-only and rule-only both pass the economic gate",
        "current_baseline_changed": False,
        "auto_promote": False,
        "no_fake_minute_after_2022": True,
    }
    dump_json(ROOT / "frozen_manifest.json", body)
    write_text(ROOT / "00_frozen_manifest.md", "# Round23 冻结\n\nM0=v5。规则阈值只在内折估计。权重不用组合 Sharpe。seed 0/1/2 同时报告均值和中位数。\n")
    body["status"] = "FROZEN"
    return _save("freeze", body)


def _daily_spearman(frame: pd.DataFrame, a: str, b: str) -> float | None:
    rhos = []
    for _, block in frame.groupby("signal_date"):
        if len(block) < 5:
            continue
        rho = block[a].corr(block[b], method="spearman")
        if pd.notna(rho):
            rhos.append(float(rho))
    if not rhos:
        return None
    return float(np.mean(rhos))


def run_diagnose(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and (cached := _cached("diagnose")):
        return cached
    five = pd.read_parquet(E0_FIVE)
    ten = pd.read_parquet(E0_TEN)
    five["signal_date"] = pd.to_datetime(five["signal_date"])
    ten["signal_date"] = pd.to_datetime(ten["signal_date"])
    five["instrument_id"] = five["instrument_id"].astype(str)
    ten["instrument_id"] = ten["instrument_id"].astype(str)
    work = five.rename(columns={"score": "five"}).merge(ten.rename(columns={"score": "ten"}), on=["signal_date", "instrument_id"])
    labels = pd.read_parquet(PANEL, columns=["trade_date", "instrument_id", "label_e2_net"])
    labels["signal_date"] = pd.to_datetime(labels["trade_date"])
    labels["instrument_id"] = labels["instrument_id"].astype(str)
    work = work.merge(labels[["signal_date", "instrument_id", "label_e2_net"]], on=["signal_date", "instrument_id"], how="left")
    rho = _daily_spearman(work, "five", "ten")

    def tops(col: str) -> pd.Series:
        ordered = work.sort_values(["signal_date", col, "instrument_id"], ascending=[True, False, True], kind="mergesort")
        return ordered.groupby("signal_date", sort=False).head(1).set_index("signal_date")

    t5 = tops("five")
    t10 = tops("ten")
    both = t5.index.intersection(t10.index)
    agree = t5.loc[both, "instrument_id"].astype(str).eq(t10.loc[both, "instrument_id"].astype(str))
    disagree = both[~agree.to_numpy()]
    d5 = t5.loc[disagree, "label_e2_net"]
    d10 = t10.loc[disagree, "label_e2_net"]
    libs = {}
    for name in ("lightgbm", "catboost"):
        try:
            __import__(name)
            libs[name] = "AVAILABLE_NOT_USED_AS_STACK"
        except Exception:
            libs[name] = "NOT_AVAILABLE"
    stop = rho is not None and abs(rho) >= STACK_RHO
    payload = {
        "status": "DONE",
        "mean_daily_spearman_five_ten": rho,
        "top1_agreement": float(agree.mean()) if len(agree) else None,
        "n_disagree_days": int(len(disagree)),
        "disagree_top1_mean_label_five": float(d5.mean()) if len(d5) else None,
        "disagree_top1_mean_label_ten": float(d10.mean()) if len(d10) else None,
        "stacking": "STOP_STACKING_NOT_COMPLEMENTARY" if stop else "NESTED_WEIGHTS_ONLY",
        "libraries": libs,
        "counts_as_new_model": False,
    }
    dump_json(ROOT / "diagnose.json", payload)
    write_text(
        ROOT / "01_diagnose.md",
        f"# Five/Ten\n\n日均 Spearman={rho}。Top1 一致率={payload['top1_agreement']}。分歧日五头 Top1 标签均值={payload['disagree_top1_mean_label_five']}，十头={payload['disagree_top1_mean_label_ten']}。\n\nStacking：`{payload['stacking']}`。LightGBM/CatBoost：{libs}。相关够高就不再叠元模型。\n",
    )
    return _save("diagnose", payload)


def _blend(five: pd.DataFrame, ten: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    a = five.copy()
    b = ten.copy()
    a["signal_date"] = pd.to_datetime(a["signal_date"])
    b["signal_date"] = pd.to_datetime(b["signal_date"])
    a["instrument_id"] = a["instrument_id"].astype(str)
    b["instrument_id"] = b["instrument_id"].astype(str)
    work = a.rename(columns={"score": "five"}).merge(b.rename(columns={"score": "ten"}), on=["signal_date", "instrument_id"])
    work["pct_five"] = work.groupby("signal_date")["five"].rank(method="average", pct=True)
    work["pct_ten"] = work.groupby("signal_date")["ten"].rank(method="average", pct=True)
    work["year"] = work["signal_date"].dt.year
    work["w_ten"] = work["year"].map(weights.to_dict()).astype(float)
    work["score"] = work["w_ten"] * work["pct_ten"] + (1.0 - work["w_ten"]) * work["pct_five"]
    gaps = pd.read_parquet(PANEL, columns=["trade_date", "instrument_id", "d1_open_gap", "label_e2_net"])
    gaps["signal_date"] = pd.to_datetime(gaps["trade_date"])
    gaps["instrument_id"] = gaps["instrument_id"].astype(str)
    work = work.merge(gaps[["signal_date", "instrument_id", "d1_open_gap", "label_e2_net"]], on=["signal_date", "instrument_id"], how="left")
    work = work.loc[pd.to_numeric(work["d1_open_gap"], errors="coerce").ge(-0.01)].copy()
    return work


def _inner_weights(work: pd.DataFrame) -> pd.Series:
    chosen = {}
    for year in range(2018, 2027):
        inner = work.loc[work["signal_date"].dt.year.between(year - 3, year - 1)]
        if inner["signal_date"].dt.year.nunique() < 2:
            chosen[year] = 0.90
            continue
        best_w, best_ic = 0.90, -np.inf
        for w in WEIGHT_GRID:
            score = w * inner["pct_ten"] + (1.0 - w) * inner["pct_five"]
            trial = inner.assign(score=score)
            ic = _daily_spearman(trial.dropna(subset=["label_e2_net"]), "score", "label_e2_net")
            if ic is not None and ic > best_ic:
                best_w, best_ic = w, ic
        chosen[year] = best_w
    return pd.Series(chosen)


def run_ensemble(*, resume: bool = True, jobs: int = 2, **_: Any) -> dict[str, Any]:
    if resume and (cached := _cached("ensemble")):
        return cached
    if not (ROOT / "frozen_manifest.json").is_file():
        raise RuntimeError("freeze first")
    five = pd.read_parquet(E0_FIVE)
    ten = pd.read_parquet(E0_TEN)
    fused = fuse_heads(five, ten)
    rows = []
    rows.append(
        {
            "code": "R23_M_RANK_90_10",
            "status": "IDENTICAL_TO_M0",
            "reason": "现有 fuse_heads 已经是截面 rank 后再 90/10。不把同一条 NAV 再记成一次改善。",
            "sharpe": V5_ANCHOR["sharpe"],
            "cagr": V5_ANCHOR["cagr"],
            "delta_sharpe": 0.0,
            "delta_cagr": 0.0,
            "economically_material": False,
            "n_fused": int(len(fused)),
        }
    )
    probe = five.copy()
    probe["signal_date"] = pd.to_datetime(probe["signal_date"])
    ten_p = ten.copy()
    ten_p["signal_date"] = pd.to_datetime(ten_p["signal_date"])
    joined = probe.rename(columns={"score": "five"}).merge(ten_p.rename(columns={"score": "ten"}), on=["signal_date", "instrument_id"])
    joined["pct_five"] = joined.groupby("signal_date")["five"].rank(pct=True)
    joined["pct_ten"] = joined.groupby("signal_date")["ten"].rank(pct=True)
    labels = pd.read_parquet(PANEL, columns=["trade_date", "instrument_id", "label_e2_net", "d1_open_gap"])
    labels["signal_date"] = pd.to_datetime(labels["trade_date"])
    labels["instrument_id"] = labels["instrument_id"].astype(str)
    joined = joined.merge(labels, on=["signal_date", "instrument_id"], how="left")
    weights = _inner_weights(joined)
    write_df(ROOT / "inner_weights.csv", weights.rename("w_ten").reset_index().rename(columns={"index": "year"}))
    blended = _blend(five, ten, weights)
    weight_sleeve = run_scored_sleeve(
        ROOT / "candidates" / "R23_M_NN_WEIGHT",
        blended[["signal_date", "instrument_id", "score"]],
        {"code": "R23_M_NN_WEIGHT", "round": "23", "top_changed": True, "objective": "inner rank IC"},
        resume=resume,
        base_nav=E0_NAV,
    )
    rows.append(weight_sleeve)
    five_train = pd.read_parquet(TRAIN)
    five_cand = pd.read_parquet(PANEL)
    ten_train = pd.read_parquet(TEN_TRAIN)
    ten_cand = pd.read_parquet(TEN_CAND)
    five_dir = ROOT / "seeds012" / "five"
    ten_dir = ROOT / "seeds012" / "ten"
    train_ranker_head(
        training=five_train,
        candidates=five_cand,
        features=list(FIVE_FEATURES),
        extras=[],
        head_name="five",
        output_dir=five_dir,
        seeds=SEED012,
        n_jobs=jobs,
        resume=resume,
    )
    train_ranker_head(
        training=ten_train,
        candidates=ten_cand,
        features=list(TEN_FEATURES),
        extras=[],
        head_name="ten",
        output_dir=ten_dir,
        seeds=SEED012,
        n_jobs=jobs,
        resume=resume,
    )
    for code, five_name, ten_name in (
        ("R23_M_SEED_MEAN", "predictions.parquet", "predictions.parquet"),
        ("R23_M_SEED_MEDIAN", "predictions_median.parquet", "predictions_median.parquet"),
    ):
        sleeve = run_heads_sleeve(
            ROOT / "candidates" / code,
            pd.read_parquet(five_dir / five_name),
            pd.read_parquet(ten_dir / ten_name),
            {"code": code, "round": "23", "top_changed": True, "seeds": list(SEED012), "picked_best_seed": False},
            resume=resume,
            base_nav=E0_NAV,
        )
        rows.append(sleeve)
    diag = load_json(ROOT / "diagnose.json")
    payload = {"status": "DONE", "rows": rows, "stacking": diag.get("stacking"), "picked_best_seed": False}
    dump_json(ROOT / "ensemble.json", payload)
    write_df(ROOT / "ensemble.csv", pd.DataFrame(rows))
    return _save("ensemble", payload)


def _rules_frames() -> dict[str, pd.DataFrame]:
    overlay = pd.read_parquet(E0_OVERLAY)
    overlay["signal_date"] = pd.to_datetime(overlay["signal_date"])
    overlay["instrument_id"] = overlay["instrument_id"].astype(str)
    breadth = pd.read_parquet(BREADTH)
    breadth["trade_date"] = pd.to_datetime(breadth.index if "trade_date" not in breadth.columns else breadth["trade_date"])
    if "trade_date" not in breadth.columns:
        breadth = breadth.reset_index().rename(columns={"index": "trade_date"})
    breadth["trade_date"] = pd.to_datetime(breadth["trade_date"])
    panel = pd.read_parquet(PANEL, columns=["trade_date", "instrument_id", "g0_amount_log", "d1_open_gap"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["instrument_id"] = panel["instrument_id"].astype(str)
    inst = pd.read_parquet(
        "/data1/stock-mxy/data/datasets/hermes-cn-a-2010-20260724-v7-all-features/instruments",
        columns=["instrument_id", "symbol", "exchange", "list_date"],
    )
    inst["instrument_id"] = inst["instrument_id"].astype(str)
    return {"overlay": overlay, "breadth": breadth, "panel": panel, "inst": inst}


def _apply_year_cutoff(frame: pd.DataFrame, value: pd.Series, quantile: float, higher_is_keep: bool) -> pd.DataFrame:
    work = frame.copy()
    work["year"] = work["signal_date"].dt.year
    work["metric"] = work["signal_date"].map(value) if value.index.inferred_type != "mixed" and work.index is not None else np.nan
    # value indexed by signal date for breadth; for names, metric already on rows
    return work


def run_rules(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and (cached := _cached("rules")):
        return cached
    if not (ROOT / "frozen_manifest.json").is_file():
        raise RuntimeError("freeze first")
    data = _rules_frames()
    overlay = data["overlay"]
    breadth = data["breadth"].drop_duplicates("trade_date").set_index("trade_date")["close_br"]
    panel = data["panel"]
    base_days = overlay["signal_date"].nunique()
    rows = []
    # R1 breadth
    cutoffs = {}
    for year in range(2018, 2027):
        hist = breadth.loc[(breadth.index.year >= year - 3) & (breadth.index.year <= year - 1)]
        cutoffs[year] = float(hist.quantile(0.20)) if len(hist) else None
    r1 = overlay.copy()
    r1["year"] = r1["signal_date"].dt.year
    r1["close_br"] = r1["signal_date"].map(breadth)
    r1["cutoff"] = r1["year"].map(cutoffs)
    keep1 = r1["close_br"].ge(r1["cutoff"]) | r1["cutoff"].isna()
    abstain_days = int(r1.loc[~keep1, "signal_date"].nunique())
    sleeve = run_overlay_sleeve(
        ROOT / "candidates" / "R23_R1_BREADTH",
        r1.loc[keep1].drop(columns=["year", "close_br", "cutoff"]),
        {
            "code": "R23_R1_BREADTH",
            "round": "23",
            "top_changed": True,
            "scenario": False,
            "abstain_day_share": abstain_days / base_days if base_days else None,
        },
        resume=resume,
        base_nav=E0_NAV,
    )
    sleeve["abstain_day_share"] = abstain_days / base_days if base_days else None
    sleeve["cagr_drag"] = None if sleeve.get("cagr") is None else float(sleeve["cagr"]) - float(V5_ANCHOR["cagr"])
    rows.append(sleeve)
    # R2 liquidity
    joined = overlay.merge(panel.rename(columns={"trade_date": "signal_date"}), on=["signal_date", "instrument_id"], how="left")
    masks = []
    for year, block in joined.groupby(joined["signal_date"].dt.year):
        hist = joined.loc[joined["signal_date"].dt.year.between(int(year) - 3, int(year) - 1), "g0_amount_log"]
        cutoff = float(hist.quantile(0.20)) if hist.notna().sum() else None
        if cutoff is None:
            masks.append(block)
        else:
            masks.append(block.loc[block["g0_amount_log"].isna() | block["g0_amount_log"].ge(cutoff)])
    r2 = pd.concat(masks, ignore_index=True)
    dropped = 1.0 - len(r2) / max(len(joined), 1)
    sleeve = run_overlay_sleeve(
        ROOT / "candidates" / "R23_R2_LIQUIDITY",
        r2[["signal_date", "instrument_id", "score", "allocation_multiplier"]],
        {"code": "R23_R2_LIQUIDITY", "round": "23", "top_changed": True, "scenario": False, "name_drop_share": dropped},
        resume=resume,
        base_nav=E0_NAV,
    )
    sleeve["name_drop_share"] = dropped
    sleeve["cagr_drag"] = None if sleeve.get("cagr") is None else float(sleeve["cagr"]) - float(V5_ANCHOR["cagr"])
    rows.append(sleeve)
    # R3 same-open scenario. No minute fabrication.
    inst = data["inst"]
    r3 = joined.merge(inst, on="instrument_id", how="left")
    amps = []
    for symbol, exch, day in zip(r3["symbol"], r3["exchange"], r3["signal_date"]):
        amps.append(limit_pct(board_of(symbol, exch), pd.Timestamp(day), 999, False))
    r3["amp"] = amps
    gap = pd.to_numeric(r3["d1_open_gap"], errors="coerce")
    sealed_open = r3["amp"].notna() & gap.ge(r3["amp"] - 0.003)
    kept = r3.loc[~sealed_open.fillna(False), ["signal_date", "instrument_id", "score", "allocation_multiplier"]]
    sleeve = run_overlay_sleeve(
        ROOT / "candidates" / "R23_R3_OPEN_SEAL",
        kept,
        {
            "code": "R23_R3_OPEN_SEAL",
            "round": "23",
            "top_changed": True,
            "scenario": True,
            "note": "D+1 开盘已到法定涨停则视为买不到。不用 2022 年后缺失的 09:31 分钟。不能晋级。",
        },
        resume=resume,
        base_nav=E0_NAV,
    )
    sleeve["scenario"] = True
    sleeve["economically_material"] = False
    sleeve["open_seal_drop_share"] = float(sealed_open.fillna(False).mean())
    sleeve["cagr_drag"] = None if sleeve.get("cagr") is None else float(sleeve["cagr"]) - float(V5_ANCHOR["cagr"])
    rows.append(sleeve)
    payload = {"status": "DONE", "rows": rows, "n_rules": 3}
    dump_json(ROOT / "rules.json", payload)
    write_df(ROOT / "rules.csv", pd.DataFrame(rows))
    return _save("rules", payload)


def run_joint(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and (cached := _cached("joint")):
        return cached
    ensemble = load_json(ROOT / "ensemble.json").get("rows") or []
    rules = load_json(ROOT / "rules.json").get("rows") or []
    model_ok = [r for r in ensemble if r.get("economically_material") and not str(r.get("code", "")).endswith("90_10")]
    rule_ok = [r for r in rules if r.get("economically_material") and not r.get("scenario")]
    if not model_ok or not rule_ok:
        payload = {
            "status": "NOT_TRIGGERED",
            "reason": "模型侧和规则侧没有同时过经济门，不做四格联合。",
            "n_model_pass": len(model_ok),
            "n_rule_pass": len(rule_ok),
        }
    else:
        payload = {"status": "BLOCKED", "reason": "前置同时通过，但本轮没有再开第四套联合搜索。", "n_model_pass": len(model_ok), "n_rule_pass": len(rule_ok)}
    dump_json(ROOT / "joint.json", payload)
    return _save("joint", payload)


def run_decision(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    del resume
    ensemble = load_json(ROOT / "ensemble.json").get("rows") or []
    rules = load_json(ROOT / "rules.json").get("rows") or []
    passed = [r.get("code") for r in ensemble + rules if r.get("economically_material") and not r.get("scenario")]
    decision = RECOMMEND if passed else KEEP
    best = None
    for row in ensemble + rules:
        ds = row.get("delta_sharpe")
        if ds is None:
            continue
        if best is None or abs(float(ds)) > abs(float(best["delta_sharpe"])):
            best = {"code": row.get("code"), "delta_sharpe": ds, "delta_cagr": row.get("delta_cagr"), "scenario": row.get("scenario")}
    payload = {
        "status": "DONE",
        "final_decision": decision,
        "passed": passed,
        "best_abs_move": best,
        "live_ready": False,
        "current_baseline_changed": False,
        "auto_promote": False,
        "next_round_baseline": "baseline_v5=R16_O01",
        "reason": "没有模型或规则同时过经济门。" if not passed else "有候选过门，但仍无独立 forward，不能当已验证最优。",
    }
    dump_json(ROOT / "decision.json", payload)
    return payload


def run_report(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    del resume
    decision = load_json(ROOT / "decision.json") or run_decision()
    diagnose = load_json(ROOT / "diagnose.json")
    lines = [
        "# Round23 研究报告",
        "",
        f"决策：**{decision.get('final_decision')}**。{decision.get('reason')}",
        "",
        f"M0 仍是 v5。22.4 没有替换 M0。Five/Ten 日均 Spearman={diagnose.get('mean_daily_spearman_five_ten')}，Top1 一致率={diagnose.get('top1_agreement')}。",
        f"Stacking：{diagnose.get('stacking')}。",
        "",
        "经济门没有放宽。情景审计不能当 alpha。空仓会在 cagr_drag 里体现。",
        "",
    ]
    for name in ("ensemble.csv", "rules.csv"):
        path = ROOT / name
        if path.is_file():
            frame = pd.read_csv(path)
            cols = [c for c in ("code", "sharpe", "cagr", "delta_sharpe", "delta_cagr", "economically_material", "scenario", "cagr_drag", "status") if c in frame.columns]
            lines.append(f"## {name}")
            lines.append("")
            lines.append("```")
            lines.append(frame[cols].to_string(index=False))
            lines.append("```")
            lines.append("")
    write_text(ROOT / "ROUND23_RESEARCH_REPORT.md", "\n".join(lines))
    return {"status": "DONE", "decision": decision.get("final_decision"), "final_report": str(ROOT / "ROUND23_RESEARCH_REPORT.md")}


def expand_stages(raw: str) -> list[str]:
    stages = ["freeze", "diagnose", "ensemble", "rules", "joint", "decision", "report"]
    if raw in {"all", "sequence"}:
        return stages
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    bad = [p for p in parts if p not in stages]
    if bad:
        raise RuntimeError(f"unknown stage {bad}")
    return parts


def run_stage(stage: str, *, resume: bool = True, jobs: int = 2, **_: Any) -> dict[str, Any]:
    table = {
        "freeze": run_freeze,
        "diagnose": run_diagnose,
        "ensemble": run_ensemble,
        "rules": run_rules,
        "joint": run_joint,
        "decision": run_decision,
        "report": run_report,
    }
    return table[stage](resume=resume, jobs=jobs)
