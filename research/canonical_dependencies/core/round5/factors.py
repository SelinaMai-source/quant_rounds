"""从 Hermes 候选面板 + DataSnapshot 构造 Family L/M/I/R/V。缓存 result/round5/factors/。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..paths import (
    CANDIDATE_PANEL_RELATIVE,
    FS00263_FEATURES,
    MARKET_END,
    PREDICTION_START,
    SIGNAL_END,
    SNAPSHOT_RELATIVE,
    assert_project_result_path,
    ensure_round5_tree,
    stock_path,
)
from ..round2.data import load_money_flow_daily_agg
from ..stock_readonly import ensure_stock_src_on_path
from .registry import (
    FAMILY_INTRA,
    FAMILY_LIMIT,
    FAMILY_LIQ,
    FAMILY_MONEY,
    FAMILY_REL,
    LOCKED_FEATURES,
)

EXCLUDE = {"fb_touch_count"}
NO_NEW_DEPENDENCY = {"g_prim_ttl_norm", "g_prim_px_vol_corr", "g_prim_prelimit_accel"}

FAMILY_COLUMNS: dict[str, tuple[str, ...]] = {
    FAMILY_LIMIT: (
        "h5_first_limit_value_to_amount",
        "h5_close_limit_value_to_amount",
        "h5_last_first_limit_value_change_sym",
        "h5_last_first_limit_value_ratio",
        "fb_close_seal_streak_norm",
        "fb_post_touch_drawdown",
        "fb_last_touch_time_norm",
        "fb_pre_touch_amount_ratio",
        "fb_last30_amount_ratio",
        "r5_l_first_minus_close_seal",
        "r5_l_open_limit",
    ),
    FAMILY_MONEY: (
        "h4_main_net_5_to_turnover",
        "h4_large_net_to_turnover",
        "h4_xl_net_concentration",
        "h4_retail_net_to_turnover",
        "fb_close_net_to_turnover",
        "h4_smart_money_price",
        "r5_m_main_minus_retail",
        "r5_m_main_vs_mkt",
    ),
    FAMILY_INTRA: (
        "h1_auction_volume_ratio",
        "h1_consistent_buy_ratio",
        "h1_consistent_sell_ratio",
        "h6_volume_weighted_entropy",
        "h6_average_trade_net_inflow_ratio",
        "h6_weighted_close_skew",
        "h6_weighted_close_ratio",
        "r5_i_buy_minus_sell",
    ),
    FAMILY_REL: (
        "ind_first_board_share",
        "ind_seal_rate_excess",
        "h8_bp",
        "h8_market_heat_score",
        "h8_pullback_from_20d_high",
        "h8_ret_3",
        "r5_r_ret_vs_ind",
    ),
    FAMILY_LIQ: (
        "g0_turnover",
        "g0_amount_log",
        "g0_float_mv_log",
        "h8_amount_zscore",
        "h8_turnover_rate_zscore",
        "h8_volume_price_sync",
        "r5_v_amt_vs_5_x_turn",
    ),
}

LABELS = (
    "label_e2_net",
    "label_exit_rule_net",
    "label_d1_touch",
    "label_d1_touch_available_at",
    "actual_exit_date",
    "actual_entry_date",
    "next1_date",
    "next2_date",
    "is_first_board_hard",
    "label_reject_reason",
)


def _cache_dir():
    path = ensure_round5_tree() / "factors"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_candidate() -> pd.DataFrame:
    path = stock_path(CANDIDATE_PANEL_RELATIVE)
    frame = pd.read_parquet(path)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    return frame


def _add_constructed(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if {"h5_first_limit_value_to_amount", "h5_close_limit_value_to_amount"} <= set(out.columns):
        out["r5_l_first_minus_close_seal"] = (
            pd.to_numeric(out["h5_first_limit_value_to_amount"], errors="coerce")
            - pd.to_numeric(out["h5_close_limit_value_to_amount"], errors="coerce")
        )
    if {"h4_main_net_to_turnover", "h4_retail_net_to_turnover"} <= set(out.columns):
        out["r5_m_main_minus_retail"] = (
            pd.to_numeric(out["h4_main_net_to_turnover"], errors="coerce")
            - pd.to_numeric(out["h4_retail_net_to_turnover"], errors="coerce")
        )
    if {"h1_consistent_buy_ratio", "h1_consistent_sell_ratio"} <= set(out.columns):
        out["r5_i_buy_minus_sell"] = (
            pd.to_numeric(out["h1_consistent_buy_ratio"], errors="coerce")
            - pd.to_numeric(out["h1_consistent_sell_ratio"], errors="coerce")
        )
    if {"g0_amt_vs_5", "g0_turnover"} <= set(out.columns):
        out["r5_v_amt_vs_5_x_turn"] = (
            pd.to_numeric(out["g0_amt_vs_5"], errors="coerce")
            * pd.to_numeric(out["g0_turnover"], errors="coerce")
        )
    if {"g0_ret_1d", "ind_first_board_count_log"} <= set(out.columns):
        ret = pd.to_numeric(out["g0_ret_1d"], errors="coerce")
        day_ind = ret.groupby(out["trade_date"]).transform("mean")
        out["r5_r_ret_vs_ind"] = ret - day_ind
    try:
        mkt = load_money_flow_daily_agg(start=PREDICTION_START.replace(year=2014), end=SIGNAL_END)
        mkt = mkt.reset_index()
        mkt["trade_date"] = pd.to_datetime(mkt["trade_date"])
        out = out.merge(
            mkt[["trade_date", "main_net_to_turnover"]].rename(
                columns={"main_net_to_turnover": "r5_mkt_main_net"}
            ),
            on="trade_date",
            how="left",
        )
        if "h4_main_net_to_turnover" in out.columns:
            out["r5_m_main_vs_mkt"] = (
                pd.to_numeric(out["h4_main_net_to_turnover"], errors="coerce")
                - pd.to_numeric(out["r5_mkt_main_net"], errors="coerce")
            )
    except Exception:
        out["r5_m_main_vs_mkt"] = np.nan
    # daily_raw open-limit（D 收盘可见）
    try:
        ensure_stock_src_on_path()
        from zyquant.data import DataSnapshot

        snap = DataSnapshot(stock_path(SNAPSHOT_RELATIVE), verify_hashes=False)
        raw = snap.table(
            "daily_raw",
            start=PREDICTION_START.replace(year=2014),
            end=MARKET_END,
            cutoff=snap.metadata.as_of_date,
            fields=["trade_date", "instrument_id", "open", "limit_up", "paused", "volume"],
        )
        raw["trade_date"] = pd.to_datetime(raw["trade_date"])
        raw["instrument_id"] = raw["instrument_id"].astype(str)
        keys = out[["trade_date", "instrument_id"]].drop_duplicates()
        raw = raw.merge(keys, on=["trade_date", "instrument_id"], how="inner")
        tradable = (~raw["paused"].fillna(True).astype(bool)) & pd.to_numeric(raw["volume"], errors="coerce").gt(0)
        raw["r5_l_open_limit"] = (
            tradable
            & pd.to_numeric(raw["open"], errors="coerce").ge(
                pd.to_numeric(raw["limit_up"], errors="coerce") - 1e-4
            )
        ).astype(float)
        out = out.merge(
            raw[["trade_date", "instrument_id", "r5_l_open_limit"]],
            on=["trade_date", "instrument_id"],
            how="left",
        )
    except Exception:
        out["r5_l_open_limit"] = np.nan
    if "label_exit_rule_net" in out.columns:
        out["label_bigloss_5"] = (
            pd.to_numeric(out["label_exit_rule_net"], errors="coerce") <= -0.05
        ).astype(float)
        out.loc[out["label_exit_rule_net"].isna(), "label_bigloss_5"] = np.nan
    return out


def build_factor_panel(*, resume: bool = True) -> pd.DataFrame:
    dest = _cache_dir() / "factor_panel.parquet"
    if resume and dest.is_file():
        frame = pd.read_parquet(dest)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        return frame
    panel = _add_constructed(_load_candidate())
    assert_project_result_path(dest)
    panel.to_parquet(dest, index=False)
    meta = {
        "written_at": datetime.now().astimezone().isoformat(),
        "n_rows": int(len(panel)),
        "n_cols": int(panel.shape[1]),
        "exclude": sorted(EXCLUDE),
        "no_new_dependency": sorted(NO_NEW_DEPENDENCY),
        "families": {k: list(v) for k, v in FAMILY_COLUMNS.items()},
        "note": "fb_touch_count EXCLUDE；g_prim_* NO_NEW_DEPENDENCY。未从 AmazingData 分钟重建。",
    }
    (dest.with_suffix(".json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return panel


def load_training_panel(*, features: list[str], target: str) -> pd.DataFrame:
    panel = build_factor_panel(resume=True)
    extra = [
        "trade_date", "instrument_id", target, "actual_exit_date", "next2_date",
        "label_d1_touch_available_at", "label_e2_net", "label_exit_rule_net",
        "label_bigloss_5", "is_first_board_hard",
    ]
    cols = []
    for c in list(features) + extra:
        if c in panel.columns and c not in cols:
            cols.append(c)
    return panel[cols].copy()


def family_of(name: str) -> str | None:
    for fam, cols in FAMILY_COLUMNS.items():
        if name in cols:
            return fam
    if name in LOCKED_FEATURES or name in FS00263_FEATURES:
        return "LOCKED"
    return None


def candidate_factor_names() -> list[str]:
    names = []
    for cols in FAMILY_COLUMNS.values():
        for c in cols:
            if c not in EXCLUDE and c not in NO_NEW_DEPENDENCY:
                names.append(c)
    return list(dict.fromkeys(names))


def coverage_table(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = panel if panel is not None else build_factor_panel(resume=True)
    rows = []
    all_names = list(LOCKED_FEATURES) + candidate_factor_names()
    for name in all_names:
        if name not in frame.columns:
            rows.append({"factor": name, "family": family_of(name), "coverage": 0.0, "first_year": None, "present": False})
            continue
        s = pd.to_numeric(frame[name], errors="coerce")
        years = frame.loc[s.notna(), "trade_date"].dt.year if "trade_date" in frame else pd.Series(dtype=int)
        rows.append({
            "factor": name,
            "family": family_of(name),
            "coverage": float(s.notna().mean()),
            "first_year": int(years.min()) if len(years) else None,
            "last_year": int(years.max()) if len(years) else None,
            "present": True,
            "exclude": name in EXCLUDE,
            "no_new_dependency": name in NO_NEW_DEPENDENCY,
        })
    return pd.DataFrame(rows)
