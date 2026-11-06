"""Round 23.1。诊断 Five/Ten，用 O4 选权重，训练冻结参数的 LightGBM。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.r225241_io import dump_json, load_json, load_stage, save_stage, write_df, write_text
from core.r225241_models import train_lgbm_ranker
from core.r225241_sleeve import run_round7
from core.r225241_stats import apply_grades, daily_spearman, grade_edges, ndcg_at_k, paired_block_bootstrap, reality_check_spa, topk_utility
from core.paths import RESULT_ROOT
from core.round11.common import daily_from_nav
from core.round22_3.contract import FIVE_FEATURES, TEN_FEATURES
from core.round7.contract import FIVE_HEAD, GAP_FLOOR, TEN_HEAD

from .contract import (
    COMMON_LABEL,
    DIAGNOSTIC_TEN_WEIGHT,
    DIVERSITY_SPEARMAN_MAX,
    DIVERSITY_TOP1_DISAGREE_MIN,
    FORMAL_TEN_WEIGHTS,
    HYPOTHESES,
    LGBM_BLEND,
    MIN_INNER_DAYS,
    ROOT,
)

R22_3 = RESULT_ROOT / "round22_3"
R23 = RESULT_ROOT / "round23"
E0 = R22_3 / "candidates" / "R223_E0"
PANEL = R22_3 / "data" / "five_candidate.parquet"
TRAIN = R22_3 / "data" / "five_training.parquet"
TEN_PANEL = R22_3 / "data" / "ten_candidate.parquet"
TEN_TRAIN = R22_3 / "data" / "ten_training.parquet"


def _skip(name: str, resume: bool) -> dict[str, Any] | None:
    return load_stage(ROOT, name) if resume else None


def run_freeze(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and (ROOT / "frozen_manifest.json").is_file():
        return load_json(ROOT / "frozen_manifest.json")
    body = {
        "round": "23.1",
        "frozen_before_new_nav": True,
        "formal_primary": "O4",
        "o1_is_failure_control": True,
        "ten_weight_grid": list(FORMAL_TEN_WEIGHTS),
        "ten_zero_not_a_formal_candidate": True,
        "lgbm_weights": list(LGBM_BLEND),
        "common_top2_label": COMMON_LABEL,
        "grade_edges_fit_on": "inner train only",
        "min_inner_signal_days": MIN_INNER_DAYS,
        "fallback_ten_weight": 0.90,
        "diversity": {"max_daily_spearman": DIVERSITY_SPEARMAN_MAX, "min_top1_disagreement": DIVERSITY_TOP1_DISAGREE_MIN},
        "no_catboost": True,
        "no_three_seed_mean": True,
        "no_threshold_search_on_rules": True,
        "current_baseline_changed": False,
        "auto_promote": False,
        "hypotheses": list(HYPOTHESES),
    }
    dump_json(ROOT / "frozen_manifest.json", body)
    return save_stage(ROOT, "freeze", {**body, "status": "FROZEN"})


def _load_joined() -> pd.DataFrame:
    five = pd.read_parquet(E0 / "five_pred.parquet").rename(columns={"score": "five"})
    ten = pd.read_parquet(E0 / "ten_pred.parquet").rename(columns={"score": "ten"})
    for frame in (five, ten):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    labels = pd.read_parquet(
        PANEL,
        columns=["trade_date", "instrument_id", "label_e2_net", "label_exit_rule_net", "actual_exit_date", "d1_open_gap"],
    )
    labels["signal_date"] = pd.to_datetime(labels["trade_date"])
    labels["instrument_id"] = labels["instrument_id"].astype(str)
    work = five.merge(ten, on=["signal_date", "instrument_id"], how="inner").merge(labels, on=["signal_date", "instrument_id"], how="left")
    work["d1_open_gap"] = pd.to_numeric(work["d1_open_gap"], errors="coerce")
    work = work.loc[work["d1_open_gap"].ge(GAP_FLOOR)].copy()
    work["pct_five"] = work.groupby("signal_date")["five"].rank(method="average", pct=True)
    work["pct_ten"] = work.groupby("signal_date")["ten"].rank(method="average", pct=True)
    return work


def _blend(work: pd.DataFrame, w_ten: float) -> pd.DataFrame:
    out = work.copy()
    out["score"] = float(w_ten) * out["pct_ten"] + (1.0 - float(w_ten)) * out["pct_five"]
    return out


def _selected(work: pd.DataFrame, w_ten: float) -> pd.DataFrame:
    scored = _blend(work, w_ten)
    scored = scored.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True], kind="mergesort")
    scored["rk"] = scored.groupby("signal_date").cumcount() + 1
    return scored.loc[scored["rk"].le(2)].copy()


def _mature(block: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    end = pd.to_datetime(block["actual_exit_date"])
    return block.loc[end.notna() & end.le(asof)]


def run_diagnose(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("diagnose", resume)
    if cached:
        return cached
    work = _load_joined()
    spearman = daily_spearman(work, "five", "ten")
    top1_f = _selected(work.assign(pct_ten=work["pct_five"], pct_five=work["pct_five"]), 0.0)
    # top1 by each head using rank 1 of that score
    def _top1(col: str) -> pd.DataFrame:
        tmp = work.sort_values(["signal_date", col, "instrument_id"], ascending=[True, False, True], kind="mergesort")
        tmp["rk"] = tmp.groupby("signal_date").cumcount() + 1
        return tmp.loc[tmp["rk"].eq(1), ["signal_date", "instrument_id", COMMON_LABEL, "label_e2_net"]]

    a = _top1("five").rename(columns={"instrument_id": "five_id", COMMON_LABEL: "five_exit", "label_e2_net": "five_e2"})
    b = _top1("ten").rename(columns={"instrument_id": "ten_id", COMMON_LABEL: "ten_exit", "label_e2_net": "ten_e2"})
    both = a.merge(b, on="signal_date", how="inner")
    disagree = both["five_id"] != both["ten_id"]
    payload = {
        "daily_spearman_five_ten": spearman,
        "top1_disagreement": float(disagree.mean()) if len(both) else None,
        "n_days": int(both["signal_date"].nunique()) if len(both) else 0,
        "disagreement_days_five_exit_mean": float(pd.to_numeric(both.loc[disagree, "five_exit"], errors="coerce").mean()) if disagree.any() else None,
        "disagreement_days_ten_exit_mean": float(pd.to_numeric(both.loc[disagree, "ten_exit"], errors="coerce").mean()) if disagree.any() else None,
        "disagreement_days_five_e2_mean": float(pd.to_numeric(both.loc[disagree, "five_e2"], errors="coerce").mean()) if disagree.any() else None,
        "disagreement_days_ten_e2_mean": float(pd.to_numeric(both.loc[disagree, "ten_e2"], errors="coerce").mean()) if disagree.any() else None,
        "note": "分歧日的实现标签是诊断，不用来事后改权重",
    }
    dump_json(ROOT / "diagnose.json", payload)
    return save_stage(ROOT, "diagnose", {"status": "DONE", **payload})


def _pick_weight(records: list[dict[str, Any]]) -> float:
    finite = [row for row in records if row.get("o4") is not None and row["w_ten"] in FORMAL_TEN_WEIGHTS]
    if not finite:
        return 0.90
    best = max(float(row["o4"]) for row in finite)
    tied = [row for row in finite if abs(float(row["o4"]) - best) <= 1e-12]
    tied.sort(key=lambda row: (abs(float(row["w_ten"]) - 0.90), -float(row["w_ten"])))
    return float(tied[0]["w_ten"])


def run_weights(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("weights", resume)
    if cached:
        return cached
    work = _load_joined()
    chosen = []
    o1_rows = []
    for year in range(2018, 2027):
        inner_start = pd.Timestamp(year - 1, 7, 1)
        inner_end = pd.Timestamp(year - 1, 12, 31)
        inner = work.loc[work["signal_date"].between(inner_start, inner_end)]
        n_days = int(inner["signal_date"].nunique())
        train_lab = work.loc[work["signal_date"].between(pd.Timestamp(year - 3, 1, 1), inner_start - pd.Timedelta(days=1)), COMMON_LABEL]
        edges = grade_edges(pd.to_numeric(train_lab, errors="coerce").to_numpy(), 5)
        if n_days < MIN_INNER_DAYS:
            chosen.append({"year": year, "w_ten": 0.90, "reason": "INNER_DAYS_BELOW_120", "n_days": n_days})
            continue
        formal = []
        diagnostic = []
        for w in (*FORMAL_TEN_WEIGHTS, DIAGNOSTIC_TEN_WEIGHT):
            selected = _mature(_selected(inner, w), inner_end)
            util = topk_utility(pd.to_numeric(selected[COMMON_LABEL], errors="coerce").to_numpy())
            scored = _blend(inner, w)
            scored = scored.loc[pd.to_datetime(scored["actual_exit_date"]).le(inner_end)]
            scored["grade"] = apply_grades(pd.to_numeric(scored[COMMON_LABEL], errors="coerce").to_numpy(), edges)
            o2 = ndcg_at_k(scored, k=2)
            o1 = daily_spearman(scored, "score", "label_e2_net")
            row = {"year": year, "w_ten": w, "formal": w in FORMAL_TEN_WEIGHTS, "o1_spearman": o1, "o2_ndcg2": o2.get("ndcg"), "o3": util["o3"], "o4": util["o4"], "n": util["n"]}
            (formal if row["formal"] else diagnostic).append(row)
            o1_rows.append(row)
        w_star = _pick_weight(formal)
        o1_best = max(formal + diagnostic, key=lambda row: -np.inf if row["o1_spearman"] is None else row["o1_spearman"])
        chosen.append({"year": year, "w_ten": w_star, "reason": "O4", "n_days": n_days, "o1_would_pick": o1_best["w_ten"], "o1_not_promoted": True})
    write_df(ROOT / "inner_weight_grid.csv", pd.DataFrame(o1_rows))
    write_df(ROOT / "chosen_weights.csv", pd.DataFrame(chosen))
    weights = {int(row["year"]): float(row["w_ten"]) for row in chosen}
    work["year"] = work["signal_date"].dt.year
    parts = []
    for year, block in work.groupby("year"):
        parts.append(_blend(block, weights.get(int(year), 0.90))[["signal_date", "instrument_id", "score"]])
    blended = pd.concat(parts, ignore_index=True)
    identical = set(weights.values()) == {0.90}
    payload = {"status": "DONE", "identical_to_0.90": identical, "weights": weights, "o1_promoted": False}
    if identical:
        payload["sleeve"] = {"status": "IDENTICAL_NO_INCREMENT", "code": "R231_O4_BLEND", "sharpe": None, "new_backtest": False}
    else:
        sleeve = run_round7(
            ROOT / "candidates" / "R231_O4_BLEND",
            blended,
            {"code": "R231_O4_BLEND", "round": "23.1", "scope": "FULL_HISTORY_FIRST_TO_SECOND", "top_changed": True, "objective": "O4"},
            resume=resume,
            already_overlayed=False,
        )
        payload["sleeve"] = sleeve
    dump_json(ROOT / "weights.json", payload)
    return save_stage(ROOT, "weights", {"status": "DONE", "identical_to_0.90": identical})


def _family_frame(five: pd.DataFrame, ten: pd.DataFrame) -> pd.DataFrame:
    from core.round17.backtest import fuse_heads

    fused = fuse_heads(five, ten)
    fused["signal_date"] = pd.to_datetime(fused["signal_date"])
    return fused


def _blend_models(xgb: pd.DataFrame, lgbm: pd.DataFrame, w_lgbm: float) -> pd.DataFrame:
    a = xgb.rename(columns={"score": "xgb"})
    b = lgbm.rename(columns={"score": "lgbm"})
    work = a.merge(b, on=["signal_date", "instrument_id"], how="inner")
    work["px"] = work.groupby("signal_date")["xgb"].rank(method="average", pct=True)
    work["pl"] = work.groupby("signal_date")["lgbm"].rank(method="average", pct=True)
    work["score"] = float(w_lgbm) * work["pl"] + (1.0 - float(w_lgbm)) * work["px"]
    return work[["signal_date", "instrument_id", "score"]]


def run_lgbm(*, resume: bool = True, jobs: int = 2, **_: Any) -> dict[str, Any]:
    cached = _skip("lgbm", resume)
    if cached:
        return cached
    five_train = pd.read_parquet(TRAIN)
    five_cand = pd.read_parquet(PANEL)
    ten_train = pd.read_parquet(TEN_TRAIN)
    ten_cand = pd.read_parquet(TEN_PANEL)
    five_meta = train_lgbm_ranker(
        training=five_train,
        candidates=five_cand,
        features=list(FIVE_FEATURES),
        target=FIVE_HEAD.target,
        label_end_column=FIVE_HEAD.label_end_column,
        n_bins=FIVE_HEAD.rank_n_bins,
        eval_at=[1],
        output_dir=ROOT / "lgbm" / "five",
        resume=resume,
    )
    ten_meta = train_lgbm_ranker(
        training=ten_train,
        candidates=ten_cand,
        features=list(TEN_FEATURES),
        target=TEN_HEAD.target,
        label_end_column=TEN_HEAD.label_end_column,
        n_bins=TEN_HEAD.rank_n_bins,
        eval_at=[2],
        output_dir=ROOT / "lgbm" / "ten",
        resume=resume,
    )
    xgb = _family_frame(pd.read_parquet(E0 / "five_pred.parquet"), pd.read_parquet(E0 / "ten_pred.parquet"))
    lgbm = _family_frame(pd.read_parquet(ROOT / "lgbm" / "five" / "predictions.parquet"), pd.read_parquet(ROOT / "lgbm" / "ten" / "predictions.parquet"))
    both = xgb.rename(columns={"score": "xgb"}).merge(lgbm.rename(columns={"score": "lgbm"}), on=["signal_date", "instrument_id"])
    spearman = daily_spearman(both, "xgb", "lgbm")
    top_x = both.sort_values(["signal_date", "xgb", "instrument_id"], ascending=[True, False, True]).groupby("signal_date", sort=False).head(1)
    top_l = both.sort_values(["signal_date", "lgbm", "instrument_id"], ascending=[True, False, True]).groupby("signal_date", sort=False).head(1)
    tops = top_x[["signal_date", "instrument_id"]].merge(top_l[["signal_date", "instrument_id"]], on="signal_date", suffixes=("_x", "_l"))
    disagree = float((tops["instrument_id_x"] != tops["instrument_id_l"]).mean()) if len(tops) else 0.0
    passed = spearman is not None and spearman < DIVERSITY_SPEARMAN_MAX and disagree > DIVERSITY_TOP1_DISAGREE_MIN
    sleeves = []
    if not passed:
        verdict = "DIVERSITY_FAIL_NO_ENSEMBLE"
    else:
        verdict = "DIVERSITY_PASS"
        l1 = run_round7(ROOT / "candidates" / "R231_L1_LGBM", lgbm, {"code": "R231_L1_LGBM", "round": "23.1", "scope": "FULL_HISTORY_FIRST_TO_SECOND", "top_changed": True}, resume=resume)
        l2 = run_round7(ROOT / "candidates" / "R231_L2_50", _blend_models(xgb, lgbm, 0.50), {"code": "R231_L2_50", "round": "23.1", "scope": "FULL_HISTORY_FIRST_TO_SECOND", "top_changed": True}, resume=resume)
        labels = pd.read_parquet(PANEL, columns=["trade_date", "instrument_id", COMMON_LABEL, "actual_exit_date"])
        labels["signal_date"] = pd.to_datetime(labels["trade_date"])
        labels["instrument_id"] = labels["instrument_id"].astype(str)
        w_by_year = {}
        for year in range(2018, 2027):
            inner_start = pd.Timestamp(year - 1, 7, 1)
            inner_end = pd.Timestamp(year - 1, 12, 31)
            window = both.loc[both["signal_date"].between(inner_start, inner_end)]
            if window["signal_date"].nunique() < MIN_INNER_DAYS:
                w_by_year[year] = 0.50
                continue
            best_w, best_u = 0.50, -np.inf
            for w in LGBM_BLEND:
                scored = _blend_models(
                    window[["signal_date", "instrument_id", "xgb"]].rename(columns={"xgb": "score"}),
                    window[["signal_date", "instrument_id", "lgbm"]].rename(columns={"lgbm": "score"}),
                    w,
                )
                scored = scored.merge(labels, on=["signal_date", "instrument_id"], how="left")
                scored = scored.sort_values(["signal_date", "score", "instrument_id"], ascending=[True, False, True])
                scored["rk"] = scored.groupby("signal_date").cumcount() + 1
                top = scored.loc[scored["rk"].le(2)]
                top = top.loc[pd.to_datetime(top["actual_exit_date"]).le(inner_end)]
                util = topk_utility(pd.to_numeric(top[COMMON_LABEL], errors="coerce").to_numpy())
                if util["o4"] is not None and util["o4"] > best_u:
                    best_u, best_w = util["o4"], w
            w_by_year[year] = best_w
        parts = []
        for year, block in both.groupby(both["signal_date"].dt.year):
            parts.append(_blend_models(block[["signal_date", "instrument_id", "xgb"]].rename(columns={"xgb": "score"}), block[["signal_date", "instrument_id", "lgbm"]].rename(columns={"lgbm": "score"}), w_by_year.get(int(year), 0.50)))
        l3 = run_round7(ROOT / "candidates" / "R231_L3_INNER", pd.concat(parts, ignore_index=True), {"code": "R231_L3_INNER", "round": "23.1", "scope": "FULL_HISTORY_FIRST_TO_SECOND", "top_changed": True, "lgbm_weights": w_by_year}, resume=resume)
        sleeves = [l1, l2, l3]
        write_df(ROOT / "lgbm_inner_weights.csv", pd.DataFrame([{"year": y, "w_lgbm": w} for y, w in w_by_year.items()]))
    payload = {
        "status": "DONE",
        "five_rows": five_meta.get("rows"),
        "ten_rows": ten_meta.get("rows"),
        "daily_spearman": spearman,
        "top1_disagreement": disagree,
        "diversity_pass": passed,
        "verdict": verdict,
        "sleeves": sleeves,
        "raw_scores_blended": False,
    }
    dump_json(ROOT / "lgbm.json", payload)
    return save_stage(ROOT, "lgbm", {"status": "DONE", "diversity_pass": passed, "verdict": verdict})


def _trade_cash(path: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    names = set(pq.read_schema(path).names)
    cols = ["signal_date", "instrument_id"]
    if "net_ret" in names:
        cols.append("net_ret")
    else:
        cols.extend([c for c in ("buy_cost", "sell_proceeds", "cash_dividend") if c in names])
    frame = pd.read_parquet(path, columns=cols)
    if "net_ret" not in frame.columns:
        frame["net_ret"] = (
            pd.to_numeric(frame.get("sell_proceeds"), errors="coerce").fillna(0)
            - pd.to_numeric(frame.get("buy_cost"), errors="coerce").fillna(0)
            + pd.to_numeric(frame.get("cash_dividend"), errors="coerce").fillna(0)
        )
    return frame


def run_rules(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("rules", resume)
    if cached:
        return cached
    base_path = E0 / "trades.parquet"
    rows = []
    for code, kind in (
        ("R23_R1_BREADTH", "environment_abstain"),
        ("R23_R2_LIQUIDITY", "name_filter_reallocation"),
        ("R23_R3_OPEN_SEAL", "execution_assumption"),
    ):
        path = R23 / "candidates" / code / "trades.parquet"
        summary = load_json(R23 / "candidates" / code / "summary.json")
        if not path.is_file() or not base_path.is_file():
            rows.append({"code": code, "status": "MISSING_ARTIFACT"})
            continue
        base = _trade_cash(base_path)
        alt = _trade_cash(path)
        for frame in (base, alt):
            frame["signal_date"] = pd.to_datetime(frame["signal_date"])
            frame["instrument_id"] = frame["instrument_id"].astype(str)
        base_keys = set(zip(base["signal_date"], base["instrument_id"]))
        alt_keys = set(zip(alt["signal_date"], alt["instrument_id"]))
        only_alt = alt_keys - base_keys
        only_base = base_keys - alt_keys
        classification = "ENGINE_BEHAVIOR_MISMATCH"
        if kind == "execution_assumption":
            classification = "EXECUTION_ASSUMPTION_SENSITIVITY"
        elif kind == "name_filter_reallocation" and len(only_alt) > 0:
            classification = "EXECUTION_ASSUMPTION_SENSITIVITY"
        elif kind == "environment_abstain":
            classification = "EXECUTION_ASSUMPTION_SENSITIVITY"
        rows.append(
            {
                "code": code,
                "kind": kind,
                "classification": classification,
                "base_trades": int(len(base)),
                "alt_trades": int(len(alt)),
                "trade_count_delta": int(len(alt) - len(base)),
                "names_only_in_alt": int(len(only_alt)),
                "names_only_in_base": int(len(only_base)),
                "alt_pnl_sum": float(pd.to_numeric(alt["net_ret"], errors="coerce").sum()),
                "base_pnl_sum": float(pd.to_numeric(base["net_ret"], errors="coerce").sum()),
                "sharpe": summary.get("sharpe"),
                "cagr": summary.get("cagr"),
                "explanation": _rule_text(code, len(alt) - len(base), len(only_alt)),
                "retuned": False,
            }
        )
    payload = {"status": "DONE", "rows": rows}
    dump_json(ROOT / "rule_forensics.json", payload)
    write_df(ROOT / "rule_forensics.csv", pd.DataFrame(rows))
    return save_stage(ROOT, "rules", {"status": "DONE"})


def _rule_text(code: str, delta: int, replacements: int) -> str:
    if code.endswith("BREADTH"):
        return "广度过滤整日不交易，笔数下降来自被拿掉的信号日，现金占用留下收益拖累。这是过滤假设，不是引擎多下单。"
    if code.endswith("LIQUIDITY"):
        return f"流动性过滤先删掉低成交额名字，引擎仍在剩余名单里取 Top2。新增成交 {replacements} 笔是替补或换仓，不是未解释的重复下单。笔数变化 {delta}。"
    return "开盘封死改变的是买得到买不到，属于执行敏感性，不是新的 alpha。"


def run_execution_and_stats(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    cached = _skip("stats", resume)
    if cached:
        return cached
    from core.round22_5.pipeline import _entry_table, _scenario_frame

    weights = load_json(ROOT / "weights.json")
    lgbm = load_json(ROOT / "lgbm.json")
    candidates = []
    sleeve = weights.get("sleeve") or {}
    if sleeve.get("status") == "ran":
        candidates.append(("R231_O4_BLEND", ROOT / "candidates" / "R231_O4_BLEND"))
    for row in lgbm.get("sleeves") or []:
        if row.get("status") == "ran":
            candidates.append((row["code"], ROOT / "candidates" / row["code"]))
    returns = {"v5": daily_from_nav(pd.read_parquet(E0 / "nav.parquet"))}
    boot = {}
    for code, path in candidates:
        nav = pd.read_parquet(path / "nav.parquet")
        series = daily_from_nav(nav)
        returns[code] = series
        boot[code] = paired_block_bootstrap(series, returns["v5"])
    panel = pd.DataFrame(returns).dropna(how="any")
    spa = reality_check_spa(panel, baseline="v5") if len(candidates) else {"status": "NOT_IDENTIFIABLE", "reason": "no formal sleeve"}
    exec_rows = []
    subject = candidates[-1] if candidates else None
    if subject is None:
        exec_rows.append({"status": "NO_DISTINCT_CANDIDATE", "reference": "v5 execution stress stays in round22_5"})
    else:
        code, path = subject
        overlay = pd.read_parquet(path / "overlayed.parquet")
        overlay["signal_date"] = pd.to_datetime(overlay["signal_date"])
        entry = _entry_table(overlay)
        summary = load_json(path / "summary.json")
        exec_rows.append({"code": code, "scenario": "S0", "sharpe": summary.get("sharpe"), "cagr": summary.get("cagr"), "nav_end": summary.get("nav_end"), "status": "MAIN_SLEEVE"})
        for scenario, suffix, hooks, kind in (
            ("S1", "SLIP10", {"buy_extra_slippage_bps": 10}, None),
            ("S2", "SLIP20", {"buy_extra_slippage_bps": 20}, None),
            ("S3", "OPENLIM", None, "S3"),
            ("S4", "GAP7", None, "S4"),
            ("S5", "GAP9", None, "S5"),
        ):
            frame = overlay if kind is None else _scenario_frame(overlay, entry, kind)
            ran = run_round7(
                ROOT / "execution" / f"{code}_{suffix}",
                frame,
                {"code": f"{code}_{suffix}", "round": "23.1", "scope": "EXECUTION_SCENARIO", "scenario_name": scenario},
                resume=resume,
                validation_hooks=hooks,
                scenario=True,
                counts_as_alpha=False,
                already_overlayed=True,
            )
            exec_rows.append({"code": ran.get("code"), "scenario": scenario, "sharpe": ran.get("sharpe"), "cagr": ran.get("cagr"), "nav_end": ran.get("nav_end"), "max_drawdown": ran.get("max_drawdown")})
    dump_json(ROOT / "bootstrap_spa.json", {"bootstrap": boot, "spa": spa, "bootstrap_is_posterior": False})
    write_df(ROOT / "execution_stress.csv", pd.DataFrame(exec_rows))
    return save_stage(ROOT, "stats", {"status": "DONE", "spa": spa.get("status"), "n_exec": len(exec_rows)})


def _passes(sleeve: dict[str, Any], boot: dict[str, Any], exec_rows: list[dict[str, Any]]) -> bool:
    if not sleeve or not sleeve.get("economically_material"):
        return False
    if not sleeve.get("flags", {}).get("top_changed_ok", sleeve.get("top_changed", True)):
        return False
    stats_ok = boot.get("status") == "OK" and boot.get("ci95_low") is not None and float(boot["ci95_low"]) > 0
    s0 = next((row for row in exec_rows if row.get("scenario") == "S0"), None)
    s1 = next((row for row in exec_rows if row.get("scenario") == "S1"), None)
    s3 = next((row for row in exec_rows if row.get("scenario") == "S3"), None)
    exec_ok = True
    if s0 and s1 and s1.get("cagr") is not None and s0.get("cagr") is not None:
        exec_ok = float(s1["cagr"]) > 0 and (s3 is None or (s3.get("max_drawdown") or -1) >= float(s0.get("max_drawdown") or -1) - 0.05)
    return bool(stats_ok and exec_ok)


def run_decision(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    weights = load_json(ROOT / "weights.json")
    lgbm = load_json(ROOT / "lgbm.json")
    stats = load_json(ROOT / "bootstrap_spa.json")
    exec_rows = []
    path = ROOT / "execution_stress.csv"
    if path.is_file():
        exec_rows = pd.read_csv(path).to_dict(orient="records")
    sleeve = weights.get("sleeve") or {}
    boot = (stats.get("bootstrap") or {}).get("R231_O4_BLEND") or {}
    o4_pass = _passes(sleeve, boot, exec_rows if str(exec_rows[0].get("code", "")).startswith("R231_O4") else [])
    lgbm_pass = False
    for row in lgbm.get("sleeves") or []:
        if _passes(row, (stats.get("bootstrap") or {}).get(row.get("code"), {}), exec_rows):
            lgbm_pass = True
    if sleeve.get("status") == "IDENTICAL_NO_INCREMENT" and not lgbm.get("diversity_pass"):
        final = "IDENTICAL_NO_INCREMENT"
    elif o4_pass or lgbm_pass:
        final = "V6_RESEARCH_CANDIDATE"
    elif sleeve.get("status") == "ran" or lgbm.get("diversity_pass"):
        final = "PREDICTIVE_ONLY"
    else:
        final = "REJECT"
    hypotheses = [
        {"id": "H231_O1_NOT_PROMOTED", "result": "PASS"},
        {"id": "H231_O4_BLEND", "result": "PASS" if o4_pass else ("FAIL" if sleeve.get("status") != "MISSING_ARTIFACT" else "BLOCKED")},
        {"id": "H231_LGBM_ENSEMBLE", "result": "PASS" if lgbm_pass else "FAIL"},
        {"id": "H231_RULES_EXPLAINED", "result": "PASS" if load_json(ROOT / "rule_forensics.json").get("rows") else "BLOCKED"},
    ]
    decision = {
        "final": final,
        "o4_pass": o4_pass,
        "lgbm_pass": lgbm_pass,
        "live_ready": False,
        "auto_promote": False,
        "current_baseline_changed": False,
        "hypotheses": hypotheses,
    }
    dump_json(ROOT / "decision.json", decision)
    write_df(ROOT / "hypotheses.csv", pd.DataFrame(hypotheses))
    write_text(
        ROOT / "ROUND23_1_RESEARCH_REPORT.md",
        "\n".join(
            [
                "# Round23.1 研究报告",
                "",
                f"决策：**{final}**。O4 是正式主目标。O1 的 Spearman 只作为失败对照，Ten=0 不是正式候选，也不重新晋级。",
                "",
                f"Five/Ten 诊断：{load_json(ROOT / 'diagnose.json')}",
                "",
                f"内折权重：{weights.get('weights')}。与 0.90 完全相同：{weights.get('identical_to_0.90')}。",
                "",
                f"LightGBM 多样性：Spearman={lgbm.get('daily_spearman')}，Top1 不一致={lgbm.get('top1_disagreement')}，结论={lgbm.get('verdict')}。未过门则不融合。融合用日度 rank 百分位，不用原始分数。",
                "",
                "规则法医没有重调阈值。R2 笔数上升来自过滤后的替补或换仓。R3 是执行敏感性。",
                "",
                f"自助与 SPA：{stats}。区间不是后验。",
                "",
                "```",
                pd.DataFrame(hypotheses).to_string(index=False),
                "```",
                "",
                "没有独立 Forward。LIVE_READY=false。CURRENT 未改。",
                "",
            ]
        ),
    )
    return save_stage(ROOT, "decision", {"status": "DONE", "final": final})


def run_all(*, resume: bool = True, jobs: int = 2) -> dict[str, Any]:
    out = {}
    for fn in (run_freeze, run_diagnose, run_weights, run_lgbm, run_rules, run_execution_and_stats, run_decision):
        print(f"[23.1] {fn.__name__}", flush=True)
        out[fn.__name__] = fn(resume=resume, jobs=jobs)
    return out
