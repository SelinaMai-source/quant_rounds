"""Round17 新因子。PIT：只用决策前已成熟标签。缺失不填 0。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND17_ROOT, assert_project_result_path, ensure_round17_tree
from ..round7.contract import FIVE_HEAD, FIVE_PANEL, MARKET_CONTEXT, TEN_HEAD
from ..round7.contract import LIMIT_DOWN_CONTEXT
from .common import dump_json, file_sha256, write_md
from .contract import EB_PRIOR_N

FACTOR_PARQUET = ROUND17_ROOT / "factors" / "new_factors.parquet"
FACTOR_JSON = ROUND17_ROOT / "factors" / "factor_build.json"
FACTOR_MD = ROUND17_ROOT / "08_new_families.md"
REGISTRY_CSV = ROUND17_ROOT / "inventory" / "factor_registry.csv"

KEYS = ["trade_date", "instrument_id"]

# 经济上可区分的新因子。不是历史 44 子集重放。
FACTOR_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "r17_a_mkt_relimit_lag1", "family": "A_continuation", "src": "market_context.prev_limit_relimit_rate"},
    {"name": "r17_a_mkt_red_lag1", "family": "A_continuation", "src": "market_context.prev_limit_red_rate"},
    {"name": "r17_a_eb_mkt_success", "family": "A_continuation", "src": "EB market first-board success, matured labels, prior=20"},
    {"name": "r17_a_eb_size_success", "family": "A_continuation", "src": "EB float-mv quintile success, matured labels, prior=20"},
    {"name": "r17_a_stock_prior_success", "family": "A_continuation", "src": "EB own prior first-board success, matured only"},
    {"name": "r17_a_days_since_prior_fb", "family": "A_continuation", "src": "days since this stock last first-board in panel"},
    {"name": "r17_b_mkt_limit_z20", "family": "B_contagion", "src": "market_limit_count z vs trailing 20"},
    {"name": "r17_b_ind_vs_mkt_count", "family": "B_contagion", "src": "ind_first_board_count_log vs market_limit_count_log"},
    {"name": "r17_b_mkt_down_pressure", "family": "B_contagion", "src": "market_limit_down_context.market_down_pressure"},
    {"name": "r17_b_failed_limit_rate", "family": "B_contagion", "src": "market_failed_limit_rate"},
    {"name": "r17_b_peer_amount_share", "family": "B_contagion", "src": "stock amount share of same-day first-board universe"},
    {"name": "r17_c_buy_minus_sell", "family": "C_microstructure", "src": "h1_consistent_buy_ratio - h1_consistent_sell_ratio"},
    {"name": "r17_c_auction_vs_last15", "family": "C_microstructure", "src": "h1_auction_volume_ratio / h1_last15m_volume_ratio"},
    {"name": "r17_c_large_minus_retail", "family": "C_microstructure", "src": "h4_large_net_to_turnover - h4_retail_net_to_turnover"},
    {"name": "r17_c_close_seal_quality", "family": "C_microstructure", "src": "fb_close_seal_streak_norm * h5_close_limit_retention"},
    {"name": "r17_c_entropy_x_skew", "family": "C_microstructure", "src": "h6_volume_weighted_entropy * h6_weighted_close_skew"},
    {"name": "r17_d_amt_xs_rank", "family": "D_relative_liquidity", "src": "same-day rank pct g0_amount_log"},
    {"name": "r17_d_turn_vs_peer", "family": "D_relative_liquidity", "src": "g0_turnover / same-day median"},
    {"name": "r17_d_mv_xs_rank", "family": "D_relative_liquidity", "src": "same-day rank pct g0_float_mv_log"},
    {"name": "r17_d_amt_vs5_xs_z", "family": "D_relative_liquidity", "src": "same-day z of g0_amt_vs_5"},
    {"name": "r17_d_turnover_xs_z", "family": "D_relative_liquidity", "src": "same-day z of g0_turnover"},
    {"name": "r17_e_prior_fb_count_log", "family": "E_event_history", "src": "log1p count of prior matured first-boards, same stock"},
    {"name": "r17_e_prior_mean_label", "family": "E_event_history", "src": "EB mean of matured prior label_e2_net, same stock"},
    {"name": "r17_e_hist_limit_intensity", "family": "E_event_history", "src": "g3_hist_limit_60 / (1+g3_days_since_limit)"},
    {"name": "r17_e_listing_age", "family": "E_event_history", "src": "g7_listing_days_log"},
    {"name": "r17_f_gap_xs_rank", "family": "F_xs_strength", "src": "same-day rank pct d1_open_gap"},
    {"name": "r17_f_ret5_xs_rank", "family": "F_xs_strength", "src": "same-day rank pct g1_ret_5"},
    {"name": "r17_f_seal_xs_rank", "family": "F_xs_strength", "src": "same-day rank pct fb_close_seal_streak_norm"},
    {"name": "r17_f_pxvol_xs_rank", "family": "F_xs_strength", "src": "same-day rank pct g_prim_px_vol_corr"},
    {"name": "r17_f_ind_seal_xs_rank", "family": "F_xs_strength", "src": "same-day rank pct ind_seal_rate_excess"},
)


def _xs_rank(s: pd.Series, dates: pd.Series) -> pd.Series:
    return s.groupby(dates).rank(method="average", pct=True)


def _xs_z(s: pd.Series, dates: pd.Series) -> pd.Series:
    g = s.groupby(dates)
    mu = g.transform("mean")
    sd = g.transform("std")
    out = (s - mu) / sd.replace(0.0, np.nan)
    return out


def _xs_median_ratio(s: pd.Series, dates: pd.Series) -> pd.Series:
    med = s.groupby(dates).transform("median")
    return s / med.replace(0.0, np.nan)


def _merge_asof_stats(left: pd.DataFrame, hist: pd.DataFrame, by: list[str], stats: list[str]) -> pd.DataFrame:
    pieces = []
    left = left.sort_values("trade_date")
    hist = hist.sort_values("mature_date")
    groups = left.groupby(by, sort=False) if by else [(None, left)]
    hist_groups = {k: v for k, v in (hist.groupby(by, sort=False) if by else [(None, hist)])}
    for key, block in groups:
        right = hist_groups.get(key)
        if right is None or right.empty:
            add = block.copy()
            for col in stats:
                add[col] = np.nan
            pieces.append(add)
            continue
        merged = pd.merge_asof(
            block.sort_values("trade_date"),
            right.sort_values("mature_date")[["mature_date", *stats]],
            left_on="trade_date",
            right_on="mature_date",
            direction="backward",
            allow_exact_matches=False,
        )
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True)


def _eb(k: pd.Series, n: pd.Series, prior_mean: float, prior_n: float = EB_PRIOR_N) -> pd.Series:
    n = n.fillna(0.0)
    k = k.fillna(0.0)
    return (prior_n * prior_mean + k) / (prior_n + n)


def build_new_factors(*, resume: bool = True) -> dict[str, Any]:
    ensure_round17_tree()
    if resume and FACTOR_PARQUET.is_file() and FACTOR_JSON.is_file():
        return dump_json(FACTOR_JSON, {**__import__("json").loads(FACTOR_JSON.read_text(encoding="utf-8")), "resumed": True})

    cand = pd.read_parquet(FIVE_HEAD.panel_dir / "candidate_panel.parquet")
    cand["trade_date"] = pd.to_datetime(cand["trade_date"])
    cand["instrument_id"] = cand["instrument_id"].astype(str)
    dates = cand["trade_date"]

    mkt = pd.read_parquet(MARKET_CONTEXT)
    mkt["trade_date"] = pd.to_datetime(mkt["trade_date"])
    down = pd.read_parquet(LIMIT_DOWN_CONTEXT)
    down["trade_date"] = pd.to_datetime(down["trade_date"])

    out = cand[KEYS].copy()
    out["r17_c_buy_minus_sell"] = pd.to_numeric(cand["h1_consistent_buy_ratio"], errors="coerce") - pd.to_numeric(cand["h1_consistent_sell_ratio"], errors="coerce")
    last15 = pd.to_numeric(cand["h1_last15m_volume_ratio"], errors="coerce").replace(0.0, np.nan)
    out["r17_c_auction_vs_last15"] = pd.to_numeric(cand["h1_auction_volume_ratio"], errors="coerce") / last15
    out["r17_c_large_minus_retail"] = pd.to_numeric(cand["h4_large_net_to_turnover"], errors="coerce") - pd.to_numeric(cand["h4_retail_net_to_turnover"], errors="coerce")
    out["r17_c_close_seal_quality"] = pd.to_numeric(cand["fb_close_seal_streak_norm"], errors="coerce") * pd.to_numeric(cand["h5_close_limit_retention"], errors="coerce")
    out["r17_c_entropy_x_skew"] = pd.to_numeric(cand["h6_volume_weighted_entropy"], errors="coerce") * pd.to_numeric(cand["h6_weighted_close_skew"], errors="coerce")

    amt = pd.to_numeric(cand["g0_amount_log"], errors="coerce")
    out["r17_d_amt_xs_rank"] = _xs_rank(amt, dates)
    out["r17_d_turn_vs_peer"] = _xs_median_ratio(pd.to_numeric(cand["g0_turnover"], errors="coerce"), dates)
    out["r17_d_mv_xs_rank"] = _xs_rank(pd.to_numeric(cand["g0_float_mv_log"], errors="coerce"), dates)
    out["r17_d_amt_vs5_xs_z"] = _xs_z(pd.to_numeric(cand["g0_amt_vs_5"], errors="coerce"), dates)
    out["r17_d_turnover_xs_z"] = _xs_z(pd.to_numeric(cand["g0_turnover"], errors="coerce"), dates)

    out["r17_e_hist_limit_intensity"] = pd.to_numeric(cand["g3_hist_limit_60"], errors="coerce") / (
        1.0 + pd.to_numeric(cand["g3_days_since_limit"], errors="coerce").clip(lower=0)
    )
    out["r17_e_listing_age"] = pd.to_numeric(cand["g7_listing_days_log"], errors="coerce")

    out["r17_f_gap_xs_rank"] = _xs_rank(pd.to_numeric(cand["d1_open_gap"], errors="coerce"), dates)
    out["r17_f_ret5_xs_rank"] = _xs_rank(pd.to_numeric(cand["g1_ret_5"], errors="coerce"), dates)
    out["r17_f_seal_xs_rank"] = _xs_rank(pd.to_numeric(cand["fb_close_seal_streak_norm"], errors="coerce"), dates)
    out["r17_f_pxvol_xs_rank"] = _xs_rank(pd.to_numeric(cand["g_prim_px_vol_corr"], errors="coerce"), dates)
    out["r17_f_ind_seal_xs_rank"] = _xs_rank(pd.to_numeric(cand["ind_seal_rate_excess"], errors="coerce"), dates)

    day_amt = np.exp(amt)
    out["r17_b_peer_amount_share"] = day_amt / day_amt.groupby(dates).transform("sum")

    mkt_use = mkt.copy()
    mkt_use["r17_a_mkt_relimit_lag1"] = pd.to_numeric(mkt_use["prev_limit_relimit_rate"], errors="coerce")
    mkt_use["r17_a_mkt_red_lag1"] = pd.to_numeric(mkt_use["prev_limit_red_rate"], errors="coerce")
    mkt_use["r17_b_failed_limit_rate"] = pd.to_numeric(mkt_use["market_failed_limit_rate"], errors="coerce")
    lim = pd.to_numeric(mkt_use["market_limit_count"], errors="coerce")
    mkt_use["r17_b_mkt_limit_z20"] = (lim - lim.rolling(20, min_periods=10).mean()) / lim.rolling(20, min_periods=10).std()
    mkt_keep = mkt_use[["trade_date", "r17_a_mkt_relimit_lag1", "r17_a_mkt_red_lag1", "r17_b_failed_limit_rate", "r17_b_mkt_limit_z20", "market_limit_count_log"]]
    out = out.merge(mkt_keep, on="trade_date", how="left")
    down_keep = down[["trade_date", "market_down_pressure"]].rename(columns={"market_down_pressure": "r17_b_mkt_down_pressure"})
    out = out.merge(down_keep, on="trade_date", how="left")
    out["r17_b_ind_vs_mkt_count"] = pd.to_numeric(cand["ind_first_board_count_log"], errors="coerce") - pd.to_numeric(out["market_limit_count_log"], errors="coerce")
    out = out.drop(columns=["market_limit_count_log"])

    # Event history: only matured D+1/D+2 labels strictly before decision date.
    hist = cand[KEYS + ["next2_date", "label_e2_net", "g0_float_mv_log"]].copy()
    hist["next2_date"] = pd.to_datetime(hist["next2_date"])
    hist["label_e2_net"] = pd.to_numeric(hist["label_e2_net"], errors="coerce")
    hist = hist.dropna(subset=["next2_date", "label_e2_net"])
    hist["success"] = (hist["label_e2_net"] > 0).astype(float)
    hist = hist.sort_values("next2_date")
    hist["mature_date"] = hist["next2_date"]
    hist["n"] = 1.0
    hist["k"] = hist["success"]
    hist["sum_label"] = hist["label_e2_net"]
    glob = hist[["mature_date", "n", "k", "sum_label"]].copy()
    glob = glob.groupby("mature_date", as_index=False)[["n", "k", "sum_label"]].sum().sort_values("mature_date")
    glob["n"] = glob["n"].cumsum()
    glob["k"] = glob["k"].cumsum()
    glob["sum_label"] = glob["sum_label"].cumsum()
    glob_prior = float(hist["success"].mean()) if len(hist) else 0.5
    left = out[KEYS].copy()
    gmerge = pd.merge_asof(
        left.sort_values("trade_date"),
        glob.rename(columns={"n": "g_n", "k": "g_k", "sum_label": "g_sum"}),
        left_on="trade_date",
        right_on="mature_date",
        direction="backward",
        allow_exact_matches=False,
    )
    out["r17_a_eb_mkt_success"] = _eb(gmerge["g_k"], gmerge["g_n"], glob_prior).to_numpy()

    # size-bucket EB
    def _size_bucket(s: pd.Series) -> pd.Series:
        r = s.rank(method="average", pct=True)
        return np.floor(r.clip(0, 0.999) * 5).astype("float64")

    hist["size_q"] = hist.groupby("trade_date")["g0_float_mv_log"].transform(_size_bucket)
    cand_size = cand[KEYS + ["g0_float_mv_log"]].copy()
    cand_size["size_q"] = cand_size.groupby("trade_date")["g0_float_mv_log"].transform(_size_bucket)
    size_hist = hist.dropna(subset=["size_q"]).copy()
    size_hist["size_q"] = size_hist["size_q"].astype(int)
    size_daily = size_hist.groupby(["size_q", "mature_date"], as_index=False)[["n", "k"]].sum()
    size_daily = size_daily.sort_values(["size_q", "mature_date"])
    size_daily["n"] = size_daily.groupby("size_q")["n"].cumsum()
    size_daily["k"] = size_daily.groupby("size_q")["k"].cumsum()
    left_s = cand_size[KEYS + ["size_q"]].copy()
    left_s["size_q"] = pd.to_numeric(left_s["size_q"], errors="coerce")
    sm = _merge_asof_stats(
        left_s.dropna(subset=["size_q"]).assign(size_q=lambda d: d["size_q"].astype(int)),
        size_daily.rename(columns={"n": "s_n", "k": "s_k"}),
        by=["size_q"],
        stats=["s_n", "s_k"],
    )
    sm = sm[KEYS + ["s_n", "s_k"]]
    out = out.merge(sm, on=KEYS, how="left")
    out["r17_a_eb_size_success"] = _eb(out.pop("s_k"), out.pop("s_n"), glob_prior)

    stock_hist = hist.groupby(["instrument_id", "mature_date"], as_index=False).agg(n=("n", "sum"), k=("k", "sum"), sum_label=("sum_label", "sum"), last_fb=("trade_date", "max"))
    stock_hist = stock_hist.sort_values(["instrument_id", "mature_date"])
    stock_hist["n"] = stock_hist.groupby("instrument_id")["n"].cumsum()
    stock_hist["k"] = stock_hist.groupby("instrument_id")["k"].cumsum()
    stock_hist["sum_label"] = stock_hist.groupby("instrument_id")["sum_label"].cumsum()
    stock_left = out[KEYS].copy()
    st = _merge_asof_stats(
        stock_left,
        stock_hist.rename(columns={"n": "st_n", "k": "st_k", "sum_label": "st_sum", "last_fb": "st_last"}),
        by=["instrument_id"],
        stats=["st_n", "st_k", "st_sum", "st_last"],
    )
    st = st[KEYS + ["st_n", "st_k", "st_sum", "st_last"]]
    out = out.merge(st, on=KEYS, how="left")
    out["r17_a_stock_prior_success"] = _eb(out["st_k"], out["st_n"], glob_prior)
    out["r17_e_prior_fb_count_log"] = np.log1p(out["st_n"].fillna(0.0))
    n = out["st_n"].fillna(0.0)
    out["r17_e_prior_mean_label"] = (EB_PRIOR_N * float(hist["label_e2_net"].mean()) + out["st_sum"].fillna(0.0)) / (EB_PRIOR_N + n)
    last_fb = pd.to_datetime(out.pop("st_last"))
    out["r17_a_days_since_prior_fb"] = (pd.to_datetime(out["trade_date"]) - last_fb).dt.days.astype("float64")
    out = out.drop(columns=["st_n", "st_k", "st_sum"], errors="ignore")

    names = [spec["name"] for spec in FACTOR_SPECS]
    keep = KEYS + names
    missing = [n for n in names if n not in out.columns]
    if missing:
        raise RuntimeError(f"factor build missing columns: {missing}")
    frame = out[keep].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    if frame.duplicated(KEYS).any():
        raise RuntimeError("duplicate factor keys")

    # coverage on 2018-2021, never fill 0
    dev = frame.loc[frame["trade_date"].between("2018-01-01", "2021-12-31")]
    coverage = {}
    for name in names:
        s = pd.to_numeric(dev[name], errors="coerce")
        coverage[name] = {
            "n": int(len(s)),
            "finite": int(np.isfinite(s).sum()),
            "finite_share": float(np.isfinite(s).mean()) if len(s) else 0.0,
            "mean": float(s[np.isfinite(s)].mean()) if np.isfinite(s).any() else None,
            "std": float(s[np.isfinite(s)].std()) if np.isfinite(s).sum() > 1 else None,
        }
    FACTOR_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(FACTOR_PARQUET)
    frame.to_parquet(FACTOR_PARQUET, index=False)
    payload = {
        "status": "built",
        "n_factors_specified": len(names),
        "n_rows": int(len(frame)),
        "date_min": str(frame["trade_date"].min().date()),
        "date_max": str(frame["trade_date"].max().date()),
        "filled_with_zero": False,
        "eb_prior_n": EB_PRIOR_N,
        "minute_new_features": "none_added; existing panel minute fields only used as already covering 2013-2026",
        "coverage_dev_2018_2021": coverage,
        "sha256": file_sha256(FACTOR_PARQUET),
        "specs": list(FACTOR_SPECS),
    }
    dump_json(FACTOR_JSON, payload)
    lines = [
        "# 08 新因子家族 A–F",
        "",
        "在看策略收益之前冻结。不是历史 44 子集重放。缺失不填 0。",
        f"n_factors=`{len(names)}` rows=`{len(frame)}` sha256=`{payload['sha256']}`",
        "",
        "| name | family | src | finite_share_2018_2021 |",
        "|---|---|---|---|",
    ]
    for spec in FACTOR_SPECS:
        cov = coverage[spec["name"]]
        lines.append(f"| `{spec['name']}` | {spec['family']} | {spec['src']} | {cov['finite_share']:.4f} |")
    write_md(FACTOR_MD, "\n".join(lines))
    return payload


def load_new_factors() -> pd.DataFrame:
    if not FACTOR_PARQUET.is_file():
        build_new_factors(resume=False)
    frame = pd.read_parquet(FACTOR_PARQUET)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    return frame


def factor_names() -> list[str]:
    return [spec["name"] for spec in FACTOR_SPECS]
