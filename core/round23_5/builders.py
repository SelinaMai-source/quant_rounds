"""三个预注册方法的分数与动作。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.r225241_io import dump_json
from core.round11.common import daily_from_nav
from core.round23_5.nested_validation import outer_years, sufficient_history, year_cutoff
from core.round23_5.oof_validation import mature_mask
from core.round23_5.paper_adapters.bai import expected_class, fit_meta, fit_rf, tercile_codes
from core.round23_5.paper_adapters.jensen import apply_action, ulcer_index, utility
from core.round23_5.paper_adapters.nardari import combination_weights, monthly_gross
from core.round7.contract import FIVE_FEATURES

RULES = ["v5", "five", "ten", "lgbm"]


def daily_nav(path: Path) -> pd.Series:
    nav = pd.read_parquet(path)
    nav["trade_date"] = pd.to_datetime(nav["trade_date"])
    series = daily_from_nav(nav)
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def rule_frame(components: pd.DataFrame) -> pd.DataFrame:
    work = components.copy()
    work["five"] = work["pct_five"]
    work["ten"] = work["pct_ten"]
    lgbm_five = work.groupby("signal_date")["lgbm_five"].rank(method="average", pct=True)
    lgbm_ten = work.groupby("signal_date")["lgbm_ten"].rank(method="average", pct=True)
    blended = 0.90 * lgbm_ten + 0.10 * lgbm_five
    work["lgbm"] = blended.where(blended.notna(), work["v5"])
    work["lgbm_fallback"] = blended.isna()
    return work


def monthly_rules(navs: dict[str, Path]) -> pd.DataFrame:
    frame = pd.DataFrame({name: monthly_gross(daily_nav(path)) for name, path in navs.items()})
    return frame.sort_index().dropna()


def single_utility(gross: np.ndarray, alpha: float) -> float:
    age = np.arange(len(gross) - 1, -1, -1, dtype=float)
    discount = np.power(float(alpha), age)
    values = (np.power(np.maximum(gross, 1e-8), -2.0) - 1.0) / -2.0
    return float(np.dot(discount, values))


def weights_by_year(monthly: pd.DataFrame, alpha: float) -> tuple[dict[int, dict[str, float]], list[dict]]:
    weights: dict[int, dict[str, float]] = {}
    rows: list[dict] = []
    for year in outer_years():
        hist = monthly.loc[monthly.index < year_cutoff(year), RULES].dropna()
        if not sufficient_history(year, monthly.index.min()) or len(hist) < 12:
            weights[year] = {name: np.nan for name in RULES}
            rows.append({"year": year, "alpha": alpha, "status": "INSUFFICIENT_META_HISTORY", "n_months": int(len(hist))})
            continue
        solved, ok = combination_weights(hist.to_numpy(dtype=float), alpha)
        weights[year] = {name: float(value) for name, value in zip(RULES, solved)}
        rows.append({"year": year, "alpha": alpha, "status": "OK" if ok else "OPTIMIZER_FALLBACK_EQUAL", "n_months": int(len(hist)), **weights[year]})
    return weights, rows


def best_rule_weights(monthly: pd.DataFrame, alpha: float = 0.99) -> dict[int, dict[str, float]]:
    weights = {}
    for year in outer_years():
        hist = monthly.loc[monthly.index < year_cutoff(year), RULES].dropna()
        if len(hist) < 12:
            weights[year] = {name: np.nan for name in RULES}
            continue
        scores = {name: single_utility(hist[name].to_numpy(dtype=float), alpha) for name in RULES}
        best = max(scores, key=scores.get)
        weights[year] = {name: 1.0 if name == best else 0.0 for name in RULES}
    return weights


def apply_weights(rules: pd.DataFrame, weights: dict[int, dict[str, float]]) -> pd.DataFrame:
    parts = []
    for year, block in rules.groupby(rules["signal_date"].dt.year, sort=True):
        chosen = weights.get(int(year), {})
        part = block[["signal_date", "instrument_id"]].copy()
        if not chosen or not np.isfinite(list(chosen.values())[0]):
            part["score"] = block["v5"].to_numpy()
            part["fallback_v5"] = True
        else:
            acc = np.zeros(len(block))
            for name in RULES:
                acc += float(chosen[name]) * block[name].to_numpy(dtype=float)
            part["score"] = acc
            part["fallback_v5"] = False
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _align_proba(model, features: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(features.to_numpy(dtype=float))
    out = np.zeros((len(features), 3), dtype=float)
    for column, label in enumerate(model.classes_):
        out[:, int(label)] = raw[:, column]
    return out


def p2_scores(components: pd.DataFrame, depth: int | None, jobs: int) -> tuple[pd.DataFrame, list[dict]]:
    from core.round23_3.contract import PANEL, TRAIN

    features = list(FIVE_FEATURES)
    train = pd.read_parquet(TRAIN, columns=["trade_date", "instrument_id", "label_exit_rule_net", "actual_exit_date", *features])
    cand = pd.read_parquet(PANEL, columns=["trade_date", "instrument_id", *features])
    for frame in (train, cand):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
    base = components[["signal_date", "instrument_id", "v5", "pct_five", "pct_ten", "lgbm_five", "lgbm_ten"]].copy()
    base["pct_lgbm_five"] = base.groupby("signal_date")["lgbm_five"].rank(method="average", pct=True).fillna(base["pct_five"])
    base["pct_lgbm_ten"] = base.groupby("signal_date")["lgbm_ten"].rank(method="average", pct=True).fillna(base["pct_ten"])
    pool = base.merge(cand.rename(columns={"trade_date": "signal_date"}), on=["signal_date", "instrument_id"], how="left")
    oof_by_year: dict[int, pd.DataFrame] = {}
    for past in range(2014, 2027):
        past_cut = year_cutoff(past)
        fit_rows = train.loc[mature_mask(train, past_cut - pd.Timedelta(days=1))].dropna(subset=["label_exit_rule_net", *features])
        predict_rows = train.loc[train["trade_date"].dt.year.eq(past)].dropna(subset=features)
        if len(fit_rows) < 300 or predict_rows.empty:
            continue
        low, high = fit_rows["label_exit_rule_net"].quantile([0.33, 0.67])
        labels = tercile_codes(fit_rows["label_exit_rule_net"], float(low), float(high))
        if labels.nunique() < 3:
            continue
        print(f"R235 RF OOF depth={depth} year={past} rows={len(fit_rows)}", flush=True)
        model = fit_rf(fit_rows[features], labels, max_depth=depth, seed=17, n_jobs=min(2, jobs))
        part = predict_rows[["trade_date", "instrument_id", "label_exit_rule_net", "actual_exit_date"]].copy()
        part[["p0", "p1", "p2"]] = _align_proba(model, predict_rows[features])
        oof_by_year[past] = part
    pieces, records = [], []
    for year in outer_years():
        block = pool.loc[pool["signal_date"].dt.year.eq(year)].copy()
        cutoff = year_cutoff(year)
        if year <= 2018:
            pieces.append(_fallback_block(block))
            records.append({"year": year, "status": "INSUFFICIENT_META_HISTORY", "max_depth": depth, "fit_cutoff": None})
            continue
        oof_parts = [oof_by_year[past] for past in range(2014, year) if past in oof_by_year]
        if not oof_parts:
            pieces.append(_fallback_block(block))
            records.append({"year": year, "status": "INSUFFICIENT_META_HISTORY", "max_depth": depth, "fit_cutoff": str(cutoff.date())})
            continue
        oof = pd.concat(oof_parts, ignore_index=True)
        oof = oof.loc[mature_mask(oof, cutoff - pd.Timedelta(days=1))]
        oof = oof.merge(base.rename(columns={"signal_date": "trade_date"})[["trade_date", "instrument_id", "pct_five", "pct_ten", "pct_lgbm_five", "pct_lgbm_ten"]], on=["trade_date", "instrument_id"], how="inner")
        if len(oof) < 200:
            pieces.append(_fallback_block(block))
            records.append({"year": year, "status": "INSUFFICIENT_META_HISTORY", "max_depth": depth, "fit_cutoff": str(cutoff.date()), "meta_rows": int(len(oof))})
            continue
        low, high = oof["label_exit_rule_net"].quantile([0.33, 0.67])
        oof["label"] = tercile_codes(oof["label_exit_rule_net"], float(low), float(high))
        if oof["label"].nunique() < 3:
            pieces.append(_fallback_block(block))
            records.append({"year": year, "status": "INSUFFICIENT_META_HISTORY", "max_depth": depth, "fit_cutoff": str(cutoff.date()), "meta_rows": int(len(oof))})
            continue
        meta_cols = ["pct_five", "pct_ten", "pct_lgbm_five", "pct_lgbm_ten", "p0", "p1", "p2"]
        meta = fit_meta(oof[meta_cols], oof["label"])
        final_fit = train.loc[mature_mask(train, cutoff - pd.Timedelta(days=1))].dropna(subset=["label_exit_rule_net", *features])
        flow, fhigh = final_fit["label_exit_rule_net"].quantile([0.33, 0.67])
        rf = fit_rf(final_fit[features], tercile_codes(final_fit["label_exit_rule_net"], float(flow), float(fhigh)), max_depth=depth, seed=17, n_jobs=min(2, jobs))
        ready = block.dropna(subset=features).copy()
        missing = block.loc[block[features].isna().any(axis=1), ["signal_date", "instrument_id", "v5"]].copy()
        if ready.empty:
            pieces.append(_fallback_block(block))
            records.append({"year": year, "status": "NO_FEATURES_FALLBACK_V5", "max_depth": depth, "fit_cutoff": str(cutoff.date())})
            continue
        ready[["p0", "p1", "p2"]] = _align_proba(rf, ready[features])
        proba = meta.predict_proba(ready[meta_cols].to_numpy(dtype=float))
        aligned = np.zeros((len(ready), 3), dtype=float)
        for column, label in enumerate(meta.classes_):
            aligned[:, int(label)] = proba[:, column]
        ready["score"] = expected_class(aligned)
        ready["fallback_v5"] = False
        piece = ready[["signal_date", "instrument_id", "score", "fallback_v5"]]
        if not missing.empty:
            missing["score"] = missing["v5"]
            missing["fallback_v5"] = True
            piece = pd.concat([piece, missing[["signal_date", "instrument_id", "score", "fallback_v5"]]], ignore_index=True)
        pieces.append(piece)
        records.append({"year": year, "status": "OK", "max_depth": depth, "fit_cutoff": str(cutoff.date()), "meta_rows": int(len(oof)), "train_rows": int(len(final_fit)), "uses_prediction_year_labels": False})
    return pd.concat(pieces, ignore_index=True), records


def _fallback_block(block: pd.DataFrame) -> pd.DataFrame:
    out = block[["signal_date", "instrument_id"]].copy()
    out["score"] = block["v5"].to_numpy()
    out["fallback_v5"] = True
    return out


def choose_actions(stats: dict[str, dict], lam_to: float, lam_dd: float) -> dict[int, str]:
    chosen = {}
    for year in outer_years():
        if year <= 2018:
            chosen[year] = "KEEP_V5"
            continue
        best_name, best_value = "KEEP_V5", -np.inf
        cutoff = year_cutoff(year)
        for name, item in stats.items():
            net = item["daily"].loc[item["daily"].index < cutoff]
            if len(net) < 120:
                continue
            series = item.get("turnover_series")
            if series is not None:
                hist_to = series.loc[series.index < cutoff]
                turnover = float(hist_to.mean()) if len(hist_to) else 0.0
            else:
                turnover = float(item["turnover"])
            value = utility(net, turnover, ulcer_index(net), lam_to, lam_dd)
            if value > best_value:
                best_name, best_value = name, value
        chosen[year] = best_name
    return chosen


def stitch(overlay: pd.DataFrame, chosen: dict[int, str]) -> pd.DataFrame:
    parts = []
    for year, action in chosen.items():
        block = overlay.loc[pd.to_datetime(overlay["signal_date"]).dt.year.eq(year)]
        if action == "ABSTAIN" or block.empty:
            continue
        parts.append(apply_action(block, action))
    if not parts:
        return overlay.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def cash_book(template: Path, dest: Path, code: str) -> dict:
    nav = pd.read_parquet(template).copy()
    nav["cash"] = 300000.0
    nav["market_value"] = 0.0
    nav["total_asset"] = 300000.0
    nav["live_sleeves"] = 0
    dest.mkdir(parents=True, exist_ok=True)
    nav.to_parquet(dest / "nav.parquet", index=False)
    pd.DataFrame(columns=["signal_date", "instrument_id", "ret", "entry_date", "buy_cost"]).to_parquet(dest / "trades.parquet", index=False)
    summary = {"code": code, "status": "ran", "sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "nav_end": 300000.0, "completed_trades": 0, "cash_only": True}
    dump_json(dest / "summary.json", summary)
    return summary
