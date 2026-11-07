"""二板特征：CORE3 / TARGET7 / V5_MIGRATED。"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.research_integrity_v2.io_v2 import dump_json, write_df, write_parquet
from core.round24_v5_migration.contract import (
    BASELINE_AUDIT,
    CAND_START,
    CORE3,
    PERF_END,
    R24_2,
    ROOT,
    TARGET7,
    V5_MIGRATED_PLANNED,
)


def sync_baseline_audit() -> list[str]:
    src = Path("/data1/stock-mxy/research_publish/wt_baseline_v5/baseline_v5_audit")
    BASELINE_AUDIT.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in (
        "factor_registry.csv",
        "factor_formulas.md",
        "feature_order.json",
        "factor_source_lineage.json",
        "factor_pit_contract.md",
        "SECOND_TO_THIRD_FACTOR_MIGRATION_PLAN.md",
        "second_board_coverage_matrix.csv",
        "second_to_third_factor_mapping.csv",
    ):
        p = src / name
        if not p.is_file():
            p = src / "data_catalog" / name
        if p.is_file():
            dst = BASELINE_AUDIT / p.name
            shutil.copy2(p, dst)
            copied.append(str(dst))
    cat = src / "data_catalog" / "DATA_SOURCE_CATALOG.md"
    if cat.is_file():
        dst = BASELINE_AUDIT / "DATA_SOURCE_CATALOG.md"
        shutil.copy2(cat, dst)
        copied.append(str(dst))
    return copied


def build_universe(*, resume: bool = True) -> dict[str, Any]:
    out = ROOT / "data" / "universe.parquet"
    if resume and out.is_file():
        u = pd.read_parquet(out, columns=["trade_date"])
        return {"status": "REUSED", "n": int(len(u)), "path": str(out)}
    src = R24_2 / "data" / "universe_v2.parquet"
    if not src.is_file():
        src = R24 / "data" / "second_board.parquet"
    frame = pd.read_parquet(src)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame = frame.loc[frame["trade_date"].between(CAND_START, PERF_END)].copy()
    if "is_st" in frame.columns:
        frame["eligible_st_ok"] = ~frame["is_st"].fillna(False).astype(bool)
    else:
        frame["eligible_st_ok"] = True
    write_parquet(out, frame)
    write_df(
        ROOT / "universe_manifest.csv",
        pd.DataFrame(
            [
                {
                    "n": len(frame),
                    "min_date": str(frame["trade_date"].min().date()),
                    "max_date": str(frame["trade_date"].max().date()),
                    "source": str(src),
                    "no_d1_success_filter": True,
                }
            ]
        ),
    )
    return {"status": "DONE", "n": int(len(frame)), "path": str(out)}


def _map_factors() -> pd.DataFrame:
    """按 prompt 表 + baseline 审计写迁移映射。"""
    audit_map = BASELINE_AUDIT / "second_to_third_factor_mapping.csv"
    base = pd.read_csv(audit_map) if audit_map.is_file() else pd.DataFrame()
    rows = []
    plan = {
        "g_prim_px_vol_corr": ("RECOMPUTE_ON_SECOND_BOARD", "g_prim_px_vol_corr", "分钟价量相关；无本地公式则 BLOCKED_DATA/UNVERIFIED"),
        "fb_close_seal_streak_norm": ("RECOMPUTE_ON_SECOND_BOARD", "fb_close_seal_streak_norm", "二板 D 日收盘前最终连续封板长度"),
        "fb_last30_amount_ratio": ("RECOMPUTE_ON_SECOND_BOARD", "fb_last30_amount_ratio", "二板 D 日最后30分钟成交额比"),
        "h5_last_first_limit_value_change_sym": ("RECOMPUTE_ON_SECOND_BOARD", "h5_last_first_limit_value_change_sym", "二板涨停事件；缺表则 BLOCKED_DATA"),
        "h1_consistent_buy_ratio": ("RECOMPUTE_ON_SECOND_BOARD", "h1_consistent_buy_ratio", "需主动买分钟；可能 BLOCKED_DATA"),
        "h1_last15m_volume_ratio": ("RECOMPUTE_ON_SECOND_BOARD", "h1_last15m_volume_ratio", "最后15分钟量比"),
        "h4_large_net_to_turnover": ("RECOMPUTE_ON_SECOND_BOARD", "h4_large_net_to_turnover", "大单净流入/成交额"),
        "h4_retail_net_to_turnover": ("RECOMPUTE_ON_SECOND_BOARD", "h4_retail_net_to_turnover", "小单净流入/成交额"),
        "g0_amount_log": ("RECOMPUTE_ON_SECOND_BOARD", "g0_amount_log", "log(D 成交额)"),
        "g0_amt_vs_5": ("RECOMPUTE_ON_SECOND_BOARD", "g0_amt_vs_5", "D 成交额/前5日均额"),
        "g1_ret_5": ("RECOMPUTE_ON_SECOND_BOARD", "g1_ret_5", "五日收益"),
        "g3_days_since_limit": ("SEMANTIC_REDEFINITION", "g3_days_since_prev_limit_before_streak", "新 Factor ID：连板序列开始前最近合法涨停距离"),
        "ind_seal_rate_excess": ("RECOMPUTE_ON_SECOND_BOARD", "ind_seal_rate_excess", "行业封板率超额"),
        "d1_open_gap": ("EXECUTION_ONLY", "d1_open_gap", "不得进入 D-close 模型"),
    }
    for src_f, (cls, tgt, note) in plan.items():
        formula_status = "SOURCE_VERIFIED_PENDING_BUILD"
        blocked = cls in {"EXECUTION_ONLY"}
        if len(base):
            hit = base.loc[base["source_feature_id"] == src_f]
            if len(hit):
                formula_status = str(hit.iloc[0].get("formula_status") or formula_status)
                if str(hit.iloc[0].get("blocked")).lower() in {"true", "1"}:
                    blocked = True
        rows.append(
            {
                "source_factor": src_f,
                "migration_class": cls,
                "target_feature_id": tgt,
                "note": note,
                "formula_status": formula_status,
                "blocked": blocked,
                "in_V5_MIGRATED": tgt in V5_MIGRATED_PLANNED,
            }
        )
    out = pd.DataFrame(rows)
    write_df(ROOT / "second_to_third_factor_mapping.csv", out)
    return out


def build_features(*, resume: bool = True) -> dict[str, Any]:
    out = ROOT / "data" / "features.parquet"
    if resume and out.is_file() and (ROOT / "feature_set_manifest.json").is_file():
        from core.research_integrity_v2.io_v2 import load_json

        return load_json(ROOT / "feature_set_manifest.json")
    mapping = _map_factors()
    src = R24_2 / "data" / "second_board_features.parquet"
    frame = pd.read_parquet(src)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    # CORE3 命名对齐 v5
    frame["g0_amount_log"] = pd.to_numeric(frame.get("amount_log"), errors="coerce")
    frame["g0_amt_vs_5"] = pd.to_numeric(frame.get("amount_vs_5"), errors="coerce")
    frame["g1_ret_5"] = pd.to_numeric(frame.get("return_5"), errors="coerce")
    # 新 g3：用 longest legacy 不可用；若无连板前涨停距，先标 NaN + BLOCKED 构建状态
    frame["g3_days_since_prev_limit_before_streak"] = np.nan
    # 分钟迁移列：从 TARGET7 代理 + 显式 NaN 占位（不得伪造盘口）
    for col in V5_MIGRATED_PLANNED:
        if col not in frame.columns:
            frame[col] = np.nan
    # fb_close_seal_streak_norm ≈ tail seal（最终连续封板）；不得用 longest_run 替代正式 TARGET7 名
    if "second_board_tail_seal_fraction" in frame.columns:
        frame["fb_close_seal_streak_norm"] = pd.to_numeric(frame["second_board_tail_seal_fraction"], errors="coerce")
    # 覆盖审计：仅二板候选
    cov_rows = []
    for year, block in frame.groupby(frame["trade_date"].dt.year):
        for feat in list(CORE3) + list(TARGET7) + list(V5_MIGRATED_PLANNED):
            s = pd.to_numeric(block.get(feat), errors="coerce") if feat in block.columns else pd.Series(dtype=float)
            cov_rows.append(
                {
                    "year": int(year),
                    "factor": feat,
                    "n_second_board_candidates": int(len(block)),
                    "n_formula_computable": int(s.notna().sum()) if len(s) else 0,
                    "coverage_ratio": float(s.notna().mean()) if len(s) else 0.0,
                    "missing_reason": None if (len(s) and s.notna().any()) else "NOT_BUILT_OR_NO_SOURCE",
                }
            )
    write_df(ROOT / "second_board_coverage_matrix.csv", pd.DataFrame(cov_rows))
    # 冻结特征集
    v5_present = [c for c in V5_MIGRATED_PLANNED if c in frame.columns]
    # 实际可训：至少 CORE3 全有限值比例 > 0 的列进入；分钟全 NaN 仍保留在集合中但不删
    manifest = {
        "CORE3": list(CORE3),
        "TARGET7": list(TARGET7),
        "V5_MIGRATED": v5_present,
        "excluded_from_d_close_model": ["d1_open_gap"],
        "g3_new_id": "g3_days_since_prev_limit_before_streak",
        "tail_seal_definition": "final continuous seal before close (second_board_tail_seal_fraction)",
        "frozen_before_backtest": True,
        "n_rows": int(len(frame)),
        "filled_with_zero": False,
        "mapping_rows": int(len(mapping)),
    }
    dump_json(ROOT / "feature_set_manifest.json", manifest)
    write_parquet(out, frame)
    # factor registry excerpt
    reg = []
    for feat in list(CORE3) + list(TARGET7) + list(V5_MIGRATED_PLANNED):
        reg.append(
            {
                "factor_id": feat,
                "set": "CORE3" if feat in CORE3 else ("TARGET7" if feat in TARGET7 else "V5_MIGRATED"),
                "present": feat in frame.columns,
                "finite_ratio": float(pd.to_numeric(frame[feat], errors="coerce").notna().mean()) if feat in frame.columns else 0.0,
            }
        )
    write_df(ROOT / "factor_registry.csv", pd.DataFrame(reg))
    # lineage
    if (BASELINE_AUDIT / "factor_source_lineage.json").is_file():
        shutil.copy2(BASELINE_AUDIT / "factor_source_lineage.json", ROOT / "factor_source_lineage.json")
    else:
        dump_json(ROOT / "factor_source_lineage.json", {"status": "COPIED_FROM_MAPPING", "rows": mapping.to_dict(orient="records")})
    # also csv lineage summary
    write_df(ROOT / "factor_source_lineage.csv", mapping)
    return {"status": "DONE", **manifest}
