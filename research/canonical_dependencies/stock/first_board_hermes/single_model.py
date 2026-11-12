"""Single-model first-board touch prediction and factor contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


KEYS = ["trade_date", "instrument_id"]
MINUTE_EXTENSION_FEATURES = [
    "h_min1_path_eff_full",
    "fb_touch_count",
    "fb_last_touch_time_norm",
    "fb_limit_close_minute_ratio",
    "fb_close_seal_streak_norm",
    "fb_post_touch_drawdown",
    "fb_pre_touch_amount_ratio",
    "fb_last30_amount_ratio",
]
MICROSTRUCTURE_FEATURES = [
    "h1_auction_volume_ratio",
    "h1_consistent_buy_ratio",
    "h1_consistent_sell_ratio",
    "h1_last15m_volume_ratio",
    "h4_smart_money_price",
    "h4_volume_order_imbalance",
    "h4_informed_buy_close_ratio",
]
INDUSTRY_FEATURES = [
    "ind_first_board_count_log",
    "ind_first_board_share",
    "ind_seal_rate_excess",
    "ind_stock_seal_quality_rank",
]
SINGLE_MODEL_FEATURES = [
    # First-board price path and seal quality.
    "g_prim_ttl_norm",
    "g_prim_px_vol_corr",
    "g_prim_prelimit_accel",
    *MINUTE_EXTENSION_FEATURES,
    "h5_first_limit_value_to_amount",
    "h5_close_limit_value_to_amount",
    "h5_close_limit_retention",
    "h5_last_first_limit_value_change_sym",
    # Hermes intraday/microstructure factors.
    "h6_volume_weighted_entropy",
    "h6_average_trade_net_inflow_ratio",
    "h6_weighted_close_skew",
    "h6_weighted_close_ratio",
    *MICROSTRUCTURE_FEATURES,
    # Money flow.
    "h4_main_net_to_turnover",
    "h4_main_net_5_to_turnover",
    "h4_large_net_to_turnover",
    "h4_xl_net_concentration",
    "h4_retail_net_to_turnover",
    "fb_close_net_to_turnover",
    # Stock state.
    "g0_turnover",
    "g0_amount_log",
    "g0_amt_vs_5",
    "g0_float_mv_log",
    "g0_close_open",
    "g1_ret_5",
    "g3_hist_limit_60",
    "g3_days_since_limit",
    # Industry context, never market-wide constants.
    *INDUSTRY_FEATURES,
]

# These six vendor fields are retained for reproducibility of the earlier
# reduced contracts.  The current full-44 contract starts training in 2015
# and handles their small number of missing rows by complete-case deletion
# inside each train/predict fold; source values are never imputed.
HISTORICALLY_SPARSE_MICROSTRUCTURE_FEATURES = [
    "h1_auction_volume_ratio",
    "h1_consistent_buy_ratio",
    "h1_consistent_sell_ratio",
    "h1_last15m_volume_ratio",
    "h4_volume_order_imbalance",
    "h4_informed_buy_close_ratio",
]
SINGLE_MODEL_HISTORICAL38_FEATURES = [
    feature
    for feature in SINGLE_MODEL_FEATURES
    if feature not in HISTORICALLY_SPARSE_MICROSTRUCTURE_FEATURES
]
HISTORICALLY_INCOMPLETE_ADDITIONAL_FEATURES = [
    "fb_close_net_to_turnover",
    "h4_smart_money_price",
    "h5_last_first_limit_value_change_sym",
    "ind_stock_seal_quality_rank",
]
SINGLE_MODEL_HISTORICAL34_FEATURES = [
    feature
    for feature in SINGLE_MODEL_HISTORICAL38_FEATURES
    if feature not in HISTORICALLY_INCOMPLETE_ADDITIONAL_FEATURES
]
PRE2014_UNAVAILABLE_MONEY_FLOW_FEATURES = [
    "h4_main_net_to_turnover",
    "h4_main_net_5_to_turnover",
    "h4_large_net_to_turnover",
    "h4_xl_net_concentration",
    "h4_retail_net_to_turnover",
]
SINGLE_MODEL_HISTORICAL29_FEATURES = [
    feature
    for feature in SINGLE_MODEL_HISTORICAL34_FEATURES
    if feature not in PRE2014_UNAVAILABLE_MONEY_FLOW_FEATURES
]

# Weak-signal removal is a model-input contract. The immutable source panel
# retains every column so prior experiments remain reproducible.
SINGLE_MODEL_WEAK_FEATURES = [
    "fb_pre_touch_amount_ratio",
    "fb_last30_amount_ratio",
    "h5_last_first_limit_value_change_sym",
    "h1_consistent_sell_ratio",
    "h1_last15m_volume_ratio",
    "h4_informed_buy_close_ratio",
    "fb_close_net_to_turnover",
    "g0_turnover",
    "g0_close_open",
    "g3_hist_limit_60",
    "g3_days_since_limit",
    "ind_first_board_count_log",
    "ind_first_board_share",
    "ind_seal_rate_excess",
]
SINGLE_MODEL_REDUCED30_FEATURES = [
    feature
    for feature in SINGLE_MODEL_FEATURES
    if feature not in SINGLE_MODEL_WEAK_FEATURES
]


@dataclass(frozen=True)
class TouchClassifierConfig:
    n_estimators: int = 100
    max_depth: int = 3
    min_child_weight: float = 1.0
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    random_state: int = 42
    n_jobs: int = 2
    tree_method: str = "hist"
    device: str = "cpu"


def symmetric_change(current: pd.Series, baseline: pd.Series) -> pd.Series:
    """Return a scale-free, bounded change in [-1, 1]."""
    current_num = pd.to_numeric(current, errors="coerce")
    baseline_num = pd.to_numeric(baseline, errors="coerce")
    denominator = current_num.abs() + baseline_num.abs()
    result = (current_num - baseline_num) / denominator.where(denominator.gt(0))
    return result.where(denominator.ne(0), 0.0)


def compute_minute_extension_factors(
    bars: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    tolerance: float = 1e-4,
) -> pd.DataFrame:
    """Compute seal-path factors for complete AmazingData minute days."""
    required_bars = {*KEYS, "bar_time", "open", "high", "low", "close", "amount"}
    required_selection = {*KEYS, "limit_up"}
    missing_bars = sorted(required_bars - set(bars.columns))
    missing_selection = sorted(required_selection - set(selection.columns))
    if missing_bars or missing_selection:
        raise ValueError(
            f"missing minute inputs: bars={missing_bars}, selection={missing_selection}"
        )
    if selection.duplicated(KEYS).any():
        raise ValueError("minute selection contains duplicate keys")
    limits = selection.copy()
    limits["trade_date"] = pd.to_datetime(limits["trade_date"]).dt.date
    limits["instrument_id"] = limits["instrument_id"].astype(str)
    limit_lookup = limits.set_index(KEYS)["limit_up"]

    work = bars.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"]).dt.date
    work["instrument_id"] = work["instrument_id"].astype(str)
    rows: list[dict[str, Any]] = []
    for key, group in work.groupby(KEYS, sort=True):
        if key not in limit_lookup.index:
            raise ValueError(f"minute bars absent from selection: {key}")
        group = group.sort_values("bar_time")
        limit_up = float(limit_lookup.loc[key])
        high = pd.to_numeric(group["high"], errors="coerce").to_numpy(float)
        low = pd.to_numeric(group["low"], errors="coerce").to_numpy(float)
        open_ = pd.to_numeric(group["open"], errors="coerce").to_numpy(float)
        close = pd.to_numeric(group["close"], errors="coerce").to_numpy(float)
        amount = pd.to_numeric(group["amount"], errors="coerce").to_numpy(float)
        touch_positions = np.flatnonzero(high >= limit_up - tolerance)
        touched = high >= limit_up - tolerance
        touch_event_positions = np.flatnonzero(
            touched & ~np.r_[False, touched[:-1]]
        )
        close_limit = close >= limit_up - tolerance
        total_amount = float(np.nansum(amount))
        first = int(touch_positions[0]) if len(touch_positions) else None
        trailing = 0
        for value in close_limit[::-1]:
            if not bool(value):
                break
            trailing += 1
        post_drawdown = np.nan
        pre_amount_ratio = np.nan
        if first is not None:
            post_low = np.nanmin(low[first:])
            post_drawdown = post_low / limit_up - 1.0 if np.isfinite(post_low) else np.nan
            if total_amount > 0:
                pre_amount_ratio = float(np.nansum(amount[:first])) / total_amount
        path_steps = np.r_[close[0] - open_[0], np.diff(close)]
        path_length = float(np.sum(np.abs(path_steps)))
        displacement = float(abs(close[-1] - open_[0]))
        path_efficiency = (
            displacement / path_length
            if np.isfinite(path_length) and path_length > 0
            else (1.0 if path_length == 0 and displacement == 0 else np.nan)
        )
        if np.isfinite(path_efficiency) and not 0.0 <= path_efficiency <= 1.0 + 1e-12:
            raise RuntimeError(f"minute path efficiency out of range for {key}: {path_efficiency}")
        rows.append({
            "trade_date": key[0],
            "instrument_id": key[1],
            "h_min1_path_eff_full": min(path_efficiency, 1.0),
            # Count independent below-limit -> limit-touch episodes.  Counting
            # every quote/trade at the limit would make this depend on vendor
            # tick density rather than on the price path.
            "fb_touch_count": int(len(touch_event_positions)),
            # Use the last touch-episode start in the complete minute session.
            # The last touched bar would be the closing bar for every close-
            # limit stock and therefore carry no cross-sectional information.
            "fb_last_touch_time_norm": (
                float(touch_event_positions[-1]) / float(len(group) - 1)
                if len(touch_event_positions) and len(group) > 1 else
                (0.0 if len(touch_event_positions) else np.nan)
            ),
            "fb_limit_close_minute_ratio": float(np.mean(close_limit)),
            "fb_close_seal_streak_norm": trailing / float(len(group)),
            "fb_post_touch_drawdown": post_drawdown,
            "fb_pre_touch_amount_ratio": pre_amount_ratio,
            "fb_last30_amount_ratio": (
                float(np.nansum(amount[-30:])) / total_amount
                if total_amount > 0 else np.nan
            ),
        })
    result = pd.DataFrame(rows, columns=[*KEYS, *MINUTE_EXTENSION_FEATURES])
    if result.duplicated(KEYS).any():
        raise RuntimeError("minute extension factors contain duplicate keys")
    return result


def attach_d1_touch_label(
    panel: pd.DataFrame,
    market: pd.DataFrame,
    *,
    tolerance: float = 1e-4,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach the causal D+1 tradable intraday-limit-touch label."""
    required_panel = {*KEYS, "next1_date"}
    required_market = {*KEYS, "high", "limit_up", "paused", "volume"}
    if missing := sorted(required_panel - set(panel.columns)):
        raise ValueError(f"panel missing touch-label columns: {missing}")
    if missing := sorted(required_market - set(market.columns)):
        raise ValueError(f"market missing touch-label columns: {missing}")
    if panel.duplicated(KEYS).any() or market.duplicated(KEYS).any():
        raise ValueError("duplicate keys in touch-label inputs")
    result = panel.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    result["next1_date"] = pd.to_datetime(result["next1_date"])
    prices = market.copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices["instrument_id"] = prices["instrument_id"].astype(str)
    entry = prices[[*KEYS, "high", "limit_up", "paused", "volume"]].rename(
        columns={"trade_date": "next1_date"}
    )
    result = result.merge(
        entry,
        on=["next1_date", "instrument_id"],
        how="left",
        validate="many_to_one",
    )
    row_present = result["high"].notna() & result["limit_up"].notna()
    tradable = (
        row_present
        & ~result["paused"].fillna(True).astype(bool)
        & pd.to_numeric(result["volume"], errors="coerce").gt(0)
    )
    touched = tradable & pd.to_numeric(result["high"], errors="coerce").ge(
        pd.to_numeric(result["limit_up"], errors="coerce") - tolerance
    )
    result["label_d1_touch"] = touched.astype(float).where(row_present)
    result["label_d1_touch_available_at"] = result["next1_date"].where(row_present)
    result = result.drop(columns=["high", "limit_up", "paused", "volume"])
    audit = {
        "rows": len(result),
        "valid_labels": int(result["label_d1_touch"].notna().sum()),
        "positive_labels": int(result["label_d1_touch"].eq(1).sum()),
        "base_rate": float(result["label_d1_touch"].mean()),
        "missing_market_rows": int((~row_present).sum()),
        "untradable_negative_rows": int((row_present & ~tradable).sum()),
        "definition": "D+1 tradable and high >= limit_up - 1e-4",
    }
    return result, audit


def apply_open_gate_with_fallback(
    predictions: pd.DataFrame,
    candidate_panel: pd.DataFrame,
    market: pd.DataFrame,
    *,
    top_k: int = 3,
    gap_min: float = -0.02,
    gap_max: float = 0.07,
    tolerance: float = 1e-4,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply D+1 observable gates in score order and refill up to ``top_k``."""
    required_predictions = {"signal_date", "instrument_id", "score", "eligible_probability"}
    required_panel = {*KEYS, "next1_date"}
    required_market = {*KEYS, "open", "close", "limit_up", "paused", "volume"}
    for name, frame, required in (
        ("predictions", predictions, required_predictions),
        ("candidate_panel", candidate_panel, required_panel),
        ("market", market, required_market),
    ):
        if missing := sorted(required - set(frame.columns)):
            raise ValueError(f"{name} missing open-gate columns: {missing}")
    pred = predictions.copy()
    pred["signal_date"] = pd.to_datetime(pred["signal_date"])
    pred["instrument_id"] = pred["instrument_id"].astype(str)
    panel = candidate_panel[[*KEYS, "next1_date"]].copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["next1_date"] = pd.to_datetime(panel["next1_date"])
    panel["instrument_id"] = panel["instrument_id"].astype(str)
    panel = panel.rename(columns={"trade_date": "signal_date"})
    prices = market.copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices["instrument_id"] = prices["instrument_id"].astype(str)
    signal_close = prices[[*KEYS, "close"]].rename(
        columns={"trade_date": "signal_date", "close": "signal_close"}
    )
    entry = prices[[*KEYS, "open", "limit_up", "paused", "volume"]].rename(
        columns={
            "trade_date": "next1_date",
            "open": "entry_open",
            "limit_up": "entry_limit_up",
            "paused": "entry_paused",
            "volume": "entry_volume",
        }
    )
    work = pred.merge(panel, on=["signal_date", "instrument_id"], validate="one_to_one")
    work = work.merge(
        signal_close, on=["signal_date", "instrument_id"], how="left", validate="many_to_one"
    ).merge(entry, on=["next1_date", "instrument_id"], how="left", validate="many_to_one")
    work["open_gap"] = (
        pd.to_numeric(work["entry_open"], errors="coerce")
        / pd.to_numeric(work["signal_close"], errors="coerce")
        - 1.0
    )
    selected_indices: list[int] = []
    audits: list[dict[str, Any]] = []
    for signal_date, block in work.groupby("signal_date", sort=True):
        accepted = 0
        ranked = block.sort_values(["score", "instrument_id"], ascending=[False, True])
        for index, row in ranked.iterrows():
            if not bool(row["eligible_probability"]):
                reason = "below_training_base_rate"
            elif (
                pd.isna(row["signal_close"])
                or pd.isna(row["entry_open"])
                or pd.isna(row["entry_limit_up"])
            ):
                reason = "missing_execution_price"
            elif bool(row["entry_paused"]) or not float(row["entry_volume"]) > 0:
                reason = "entry_not_tradable"
            elif float(row["entry_open"]) >= float(row["entry_limit_up"]) - tolerance:
                reason = "entry_open_limit_up"
            elif not gap_min <= float(row["open_gap"]) <= gap_max:
                reason = "entry_gap_outside_range"
            elif accepted >= top_k:
                reason = "rank_after_filled_top_k"
            else:
                reason = "selected"
                selected_indices.append(index)
                accepted += 1
            audits.append({
                "signal_date": signal_date,
                "instrument_id": row["instrument_id"],
                "score": row["score"],
                "open_gap": row["open_gap"],
                "decision": reason,
            })
    selected = work.loc[selected_indices, pred.columns].sort_values(
        ["signal_date", "score", "instrument_id"], ascending=[True, False, True]
    ).reset_index(drop=True)
    audit = pd.DataFrame(audits)
    report = {
        "input_predictions": len(pred),
        "selected_predictions": len(selected),
        "selected_signal_days": int(selected["signal_date"].nunique()),
        "top_k": top_k,
        "gap_range": [gap_min, gap_max],
        "decisions": audit["decision"].value_counts().sort_index().to_dict(),
    }
    return selected, audit, report


def rolling_touch_train_predict(
    training_panel: pd.DataFrame,
    candidate_panel: pd.DataFrame,
    market_calendar: Sequence[Any],
    output_dir: Path,
    *,
    features: Sequence[str] = SINGLE_MODEL_FEATURES,
    train_sessions: int = 240,
    first_prediction_year: int = 2022,
    last_prediction_year: int = 2026,
    config: TouchClassifierConfig = TouchClassifierConfig(),
    min_train_samples: int = 30,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Daily walk-forward XGBClassifier with day and capped class weights."""
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
    from xgboost import XGBClassifier

    features = list(features)
    unknown = sorted(set(features) - set(SINGLE_MODEL_FEATURES))
    if unknown or not features:
        raise ValueError(f"invalid single-model feature subset: {unknown}")
    required = {*KEYS, *features}
    for name, frame in (("training", training_panel), ("candidate", candidate_panel)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} panel missing columns: {missing}")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{name} panel contains duplicate keys")
    needed_labels = {"label_d1_touch", "label_d1_touch_available_at"}
    if missing := sorted(needed_labels - set(training_panel.columns)):
        raise ValueError(f"training panel missing touch labels: {missing}")

    train = training_panel.copy()
    candidates = candidate_panel.copy()
    for frame in (train, candidates):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    train["label_d1_touch_available_at"] = pd.to_datetime(
        train["label_d1_touch_available_at"]
    )
    train_finite = np.isfinite(train[features].apply(pd.to_numeric, errors="coerce")).all(axis=1)
    train_finite &= train["label_d1_touch"].notna()
    candidate_finite = np.isfinite(
        candidates[features].apply(pd.to_numeric, errors="coerce")
    ).all(axis=1)
    train = train.loc[train_finite].copy()
    candidates = candidates.loc[candidate_finite].copy()

    calendar = pd.DatetimeIndex(pd.to_datetime(list(market_calendar))).drop_duplicates().sort_values()
    positions = {day: i for i, day in enumerate(calendar)}
    first_training_position = positions.get(pd.Timestamp(train["trade_date"].min()))
    if first_training_position is None:
        raise RuntimeError("first training date is absent from the market calendar")
    prediction_blocks: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for signal_date in pd.DatetimeIndex(candidates["trade_date"].unique()).sort_values():
        if not first_prediction_year <= signal_date.year <= last_prediction_year:
            continue
        position = positions.get(signal_date)
        if position is None or position - first_training_position < train_sessions:
            continue
        window = calendar[position - train_sessions:position]
        fold = train.loc[
            train["trade_date"].between(window[0], window[-1])
            & train["label_d1_touch_available_at"].le(signal_date)
        ].copy()
        test = candidates.loc[candidates["trade_date"].eq(signal_date)].copy()
        if len(fold) < min_train_samples or test.empty or fold["label_d1_touch"].nunique() < 2:
            audit_rows.append({
                "signal_date": signal_date,
                "train_rows": len(fold),
                "candidate_rows": len(test),
                "status": "insufficient_training_data",
            })
            continue
        day_count = fold["trade_date"].value_counts()
        day_weight = fold["trade_date"].map(1.0 / day_count).to_numpy(float)
        y = fold["label_d1_touch"].to_numpy(int)
        positives = max(int(y.sum()), 1)
        negatives = max(int(len(y) - positives), 1)
        positive_multiplier = min(negatives / positives, 5.0)
        weights = day_weight * np.where(y == 1, positive_multiplier, 1.0)
        # Class balancing must not give high-positive-rate days more aggregate weight.
        weight_frame = pd.DataFrame({"day": fold["trade_date"].to_numpy(), "weight": weights})
        weights = weights / weight_frame.groupby("day")["weight"].transform("sum").to_numpy()
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **asdict(config),
        )
        model.fit(fold[features].to_numpy(np.float32), y, sample_weight=weights)
        model_sha256 = hashlib.sha256(model.get_booster().save_raw()).hexdigest()
        probability = model.predict_proba(test[features].to_numpy(np.float32))[:, 1]
        block = test[["trade_date", "instrument_id"]].rename(
            columns={"trade_date": "signal_date"}
        )
        block["score"] = probability
        block["predicted_class"] = model.predict(
            test[features].to_numpy(np.float32)
        ).astype(np.int8)
        block["training_base_rate"] = float(np.mean(y))
        # Retained only for backward audit of the superseded ranking experiment.
        block["eligible_probability"] = block["score"].ge(block["training_base_rate"])
        block["train_start"] = window[0]
        block["train_end"] = window[-1]
        prediction_blocks.append(block)
        audit_rows.append({
            "signal_date": signal_date,
            "train_start": window[0],
            "train_end": window[-1],
            "train_rows": len(fold),
            "candidate_rows": len(test),
            "training_base_rate": float(np.mean(y)),
            "positive_weight_multiplier": positive_multiplier,
            "model_sha256": model_sha256,
            "status": "trained",
        })
    if not prediction_blocks:
        raise RuntimeError("no touch predictions were produced")
    predictions = pd.concat(prediction_blocks, ignore_index=True).sort_values(
        ["signal_date", "score", "instrument_id"], ascending=[True, False, True]
    ).reset_index(drop=True)
    if predictions.duplicated(["signal_date", "instrument_id"]).any():
        raise RuntimeError("duplicate touch predictions")

    labels = training_panel[["trade_date", "instrument_id", "label_d1_touch"]].rename(
        columns={"trade_date": "signal_date"}
    )
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    evaluation = predictions.merge(
        labels, on=["signal_date", "instrument_id"], how="left", validate="one_to_one"
    ).dropna(subset=["label_d1_touch"])
    y_true = evaluation["label_d1_touch"].to_numpy(int)
    y_score = evaluation["score"].to_numpy(float)
    ranked = evaluation.sort_values(
        ["signal_date", "score", "instrument_id"], ascending=[True, False, True]
    )
    top1 = ranked.groupby("signal_date", sort=False).head(1)
    top3 = ranked.groupby("signal_date", sort=False).head(3)
    baseline = float(evaluation["label_d1_touch"].mean())
    yearly: dict[str, Any] = {}
    for year, block in evaluation.groupby(evaluation["signal_date"].dt.year):
        ordered = block.sort_values(
            ["signal_date", "score", "instrument_id"], ascending=[True, False, True]
        )
        year_top1 = ordered.groupby("signal_date", sort=False).head(1)
        year_top3 = ordered.groupby("signal_date", sort=False).head(3)
        rate = float(block["label_d1_touch"].mean())
        yearly[str(year)] = {
            "rows": len(block),
            "pr_auc": float(average_precision_score(block["label_d1_touch"], block["score"])),
            "brier_score": float(brier_score_loss(block["label_d1_touch"], block["score"])),
            "candidate_touch_rate": rate,
            "top1_touch_rate": float(year_top1["label_d1_touch"].mean()),
            "top3_touch_rate": float(year_top3["label_d1_touch"].mean()),
            "top3_lift": float(year_top3["label_d1_touch"].mean() / rate),
        }
    evaluation["probability_group"] = pd.qcut(
        evaluation["score"].rank(method="first"), 10, labels=False, duplicates="drop"
    )
    probability_groups = (
        evaluation.groupby("probability_group", sort=True)
        .agg(rows=("label_d1_touch", "size"), mean_probability=("score", "mean"),
             touch_rate=("label_d1_touch", "mean"))
        .reset_index().to_dict("records")
    )
    report = {
        "status": "trained_oos",
        "model": "XGBClassifier",
        "target": "label_d1_touch",
        "selection_contract": "buy every row with predicted_class == 1; no TopK truncation",
        "train_sessions": train_sessions,
        "config": asdict(config),
        "features": features,
        "feature_contract_sha256": hashlib.sha256(
            json.dumps(features, separators=(",", ":")).encode()
        ).hexdigest(),
        "complete_case_policy": {
            "training_rows_before": len(training_panel),
            "training_rows_after": len(train),
            "candidate_rows_before": len(candidate_panel),
            "candidate_rows_after": len(candidates),
        },
        "metrics": {
            "rows": len(evaluation),
            "log_loss": float(log_loss(y_true, y_score, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y_true, y_score)),
            "pr_auc": float(average_precision_score(y_true, y_score)),
            "candidate_touch_rate": baseline,
            "top1_touch_rate": float(top1["label_d1_touch"].mean()),
            "top3_touch_rate": float(top3["label_d1_touch"].mean()),
            "top3_lift": float(top3["label_d1_touch"].mean() / baseline),
        },
        "yearly_metrics": yearly,
        "probability_groups": probability_groups,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_path = output_dir / "predictions.parquet"
    audit_path = output_dir / "daily_training_audit.parquet"
    predictions.to_parquet(prediction_path, index=False)
    pd.DataFrame(audit_rows).to_parquet(audit_path, index=False)
    report["predictions_sha256"] = _sha256(prediction_path)
    report["daily_training_audit_sha256"] = _sha256(audit_path)
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return predictions, report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
