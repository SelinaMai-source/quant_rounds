"""每日一行 Top2 特征 + 标签。PIT；标签只在成熟后进训练折。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND9_ROOT, assert_project_result_path, ensure_round9_tree, round9_dir
from ..round8.geometry import load_gaps
from .allocator import sep_bucket
from .contract import AGREE_SCORE, DISAGREE_LABEL, DISAGREE_MULT, META_FEATURES, R9_00
from .geometry import build_geometry


def _daily_nav(path) -> pd.Series:
    nav = pd.read_parquet(path)
    nav["trade_date"] = pd.to_datetime(nav["trade_date"]).dt.normalize()
    return nav.sort_values("trade_date").drop_duplicates("trade_date").set_index("trade_date")["total_asset"].astype(float)


def _top2_trades(code: str) -> pd.DataFrame:
    path = round9_dir(code) / "trades.parquet"
    if not path.is_file():
        return pd.DataFrame()
    t = pd.read_parquet(path)
    t["signal_date"] = pd.to_datetime(t["signal_date"]).dt.normalize()
    t["instrument_id"] = t["instrument_id"].astype(str)
    t["_rank"] = t.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    t["ret"] = pd.to_numeric(t.get("ret"), errors="coerce")
    t["dollar"] = pd.to_numeric(t.get("sell_proceeds"), errors="coerce") - pd.to_numeric(
        t.get("buy_cost"), errors="coerce"
    )
    if "actual_exit_date" in t.columns:
        t["actual_exit_date"] = pd.to_datetime(t["actual_exit_date"]).dt.normalize()
    return t.loc[t["_rank"].eq(2)].copy()


def build_daily_frame(*, v3_code: str = R9_00, a01_code: str = "R9_A01") -> pd.DataFrame:
    geo = build_geometry().copy()
    geo = geo.reset_index()
    geo["signal_date"] = pd.to_datetime(geo["signal_date"]).dt.normalize()
    geo = geo.set_index("signal_date").sort_index()
    out = pd.DataFrame(index=geo.index)
    out["top1_id"] = geo["top1_id"].astype(str)
    out["top2_id"] = geo["top2_id"].astype(str)
    out["head_state"] = geo["head_state"].astype(str)
    out["agreement"] = out["head_state"]
    out["agreement_alias"] = out["agreement"].map(DISAGREE_LABEL)
    out["agreement_score"] = out["agreement"].map(AGREE_SCORE).astype(float)
    out["e02_mult"] = out["agreement"].map(DISAGREE_MULT).astype(float)
    out["rank_corr"] = pd.to_numeric(geo.get("rank_corr"), errors="coerce")
    out["candidate_count"] = pd.to_numeric(geo.get("candidate_count"), errors="coerce")
    for col in (
        "top1_minus_median_p250",
        "top2_minus_median_p250",
        "top1_minus_top2_p250",
        "top1_minus_median",
        "top2_minus_median",
    ):
        out[col] = pd.to_numeric(geo.get(col), errors="coerce")
    out["sep_bucket"] = sep_bucket(out["top2_minus_median_p250"])

    gaps = load_gaps().copy()
    gaps["signal_date"] = pd.to_datetime(gaps["signal_date"]).dt.normalize()
    gaps["instrument_id"] = gaps["instrument_id"].astype(str)
    top2_keys = out.reset_index()[["signal_date", "top2_id"]].rename(columns={"top2_id": "instrument_id"})
    g2 = top2_keys.merge(gaps, on=["signal_date", "instrument_id"], how="left")
    out["top2_d1_open_gap"] = pd.to_numeric(g2["d1_open_gap"], errors="coerce").to_numpy()

    t2 = _top2_trades(v3_code)
    if not t2.empty:
        one = t2.drop_duplicates("signal_date").set_index("signal_date")
        out["top2_ret"] = one.reindex(out.index)["ret"]
        out["top2_dollar"] = one.reindex(out.index)["dollar"]
        out["label_exit_date"] = one.reindex(out.index)["actual_exit_date"] if "actual_exit_date" in one else pd.NaT
        out["top2_budget"] = pd.to_numeric(one.reindex(out.index).get("target_name_budget"), errors="coerce")
    else:
        out["top2_ret"] = np.nan
        out["top2_dollar"] = np.nan
        out["label_exit_date"] = pd.NaT
        out["top2_budget"] = np.nan

    v3_path = round9_dir(v3_code) / "nav.parquet"
    a01_path = round9_dir(a01_code) / "nav.parquet"
    if v3_path.is_file() and a01_path.is_file():
        v3 = _daily_nav(v3_path)
        a01 = _daily_nav(a01_path)
        aligned = pd.concat([v3.rename("v3"), a01.rename("a01")], axis=1).dropna()
        inc = aligned["v3"].pct_change() - aligned["a01"].pct_change()
        # 信号日 ≈ 交易日前一信号；增量记在交易日，再 shift 对齐 signal_date 不硬凑。日度增量按交易日存。
        out["v3_nav"] = v3.reindex(out.index)
        out["a01_nav"] = a01.reindex(out.index)
        out["incremental_ret"] = inc.reindex(out.index)
    return out


def attach_panel_columns(daily: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """把 Top2 票的真实 panel 列挂到日表。缺列跳过，不发明字段。"""
    from ..paths import CANDIDATE_PANEL_RELATIVE, stock_path
    from .contract import FIVE_FEATURES, TEN_PANEL

    candidates = [
        TEN_PANEL / "candidate_panel.parquet",
        stock_path(
            "data/first_board_hermes/aligned_original_candidate_full44_plus_d1_open_gap_v1/candidate_panel.parquet"
        ),
        stock_path(str(CANDIDATE_PANEL_RELATIVE)),
    ]
    panel_path = next((p for p in candidates if p.is_file()), None)
    if panel_path is None:
        return daily, []
    try:
        import pyarrow.parquet as pq

        have = set(pq.ParquetFile(str(panel_path)).schema.names)
    except Exception:
        have = set(pd.read_parquet(panel_path).columns)
    date_col = "trade_date" if "trade_date" in have else ("signal_date" if "signal_date" in have else None)
    if date_col is None or "instrument_id" not in have:
        return daily, []
    keep = [c for c in columns if c in have and c not in FIVE_FEATURES]
    if not keep:
        return daily, []
    work = pd.read_parquet(panel_path, columns=[date_col, "instrument_id", *keep])
    work["signal_date"] = pd.to_datetime(work[date_col]).dt.normalize()
    work["instrument_id"] = work["instrument_id"].astype(str)
    keys = daily.reset_index()[["signal_date", "top2_id"]].rename(columns={"top2_id": "instrument_id"})
    merged = keys.merge(work, on=["signal_date", "instrument_id"], how="left")
    out = daily.copy()
    for col in keep:
        out[col] = pd.to_numeric(merged[col], errors="coerce").to_numpy()
    return out, keep


def persist_daily(daily: pd.DataFrame, name: str = "top2_daily_features.parquet") -> Any:
    ensure_round9_tree()
    path = ROUND9_ROOT / "ledger" / name
    assert_project_result_path(path)
    daily.reset_index().to_parquet(path, index=False)
    return path


def feature_matrix(daily: pd.DataFrame, cols: tuple[str, ...] = META_FEATURES) -> pd.DataFrame:
    frame = daily.copy()
    for col in cols:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.loc[:, list(cols)]


__all__ = ["attach_panel_columns", "build_daily_frame", "feature_matrix", "persist_daily"]
