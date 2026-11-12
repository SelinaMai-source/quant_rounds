"""建立 round21_2_prior_factor_lineage.csv。不得把历史失败因子换名。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND21_2_ROOT
from .common import dump_json, write_md
from .contract import FIVE_FEATURES, NEW_FACTOR_IDS, TEN_FEATURES

LINEAGE_CSV = ROUND21_2_ROOT / "round21_2_prior_factor_lineage.csv"
LINEAGE_JSON = ROUND21_2_ROOT / "lineage" / "lineage.json"

# 预登记近邻：公式/数据关系，不是回测后挑选。
PRIOR = {
    "r21_g1_avg_ticket_value": {
        "near": ["g0_amount_log", "g0_amt_vs_5"],
        "relation": "amount / DEAL_AMOUNT；与成交额水平相关但单位是笔均金额",
        "not_rename_of": [],
    },
    "r21_g2_large_ticket_asym": {
        "near": ["h4_large_net_to_turnover"],
        "relation": "大单买卖笔均量不对称；不是净流入/成交额",
        "not_rename_of": ["h4_large_net_to_turnover"],
    },
    "r21_g3_order_count_imbalance": {
        "near": ["h1_consistent_buy_ratio", "h4_large_net_to_turnover"],
        "relation": "买卖笔数差/和；语义依赖供应商笔数口径",
        "not_rename_of": ["h1_consistent_buy_ratio"],
    },
    "r21_g4_large_ticket_conc_gap": {
        "near": ["h4_large_net_to_turnover"],
        "relation": "大买量份额 − 大买笔数份额；R21 无此式",
        "not_rename_of": ["h4_xl_net_concentration"],
    },
    "r21_h1_late_flow_dominance": {
        "near": ["h4_large_net_to_turnover", "h1_last15m_volume_ratio"],
        "relation": "(net_in_close-net_in_open)/amount",
        "not_rename_of": ["h1_last15m_volume_ratio"],
    },
    "r21_h2_large_small_flow_div": {
        "near": ["h4_large_net_to_turnover", "h4_retail_net_to_turnover"],
        "relation": "(l+xl-s-m)/amount；若代数等于 main_net 则 DUPLICATE",
        "not_rename_of": ["h4_large_net_to_turnover"],
    },
    "r21_h3_flow_dir_disagree": {
        "near": ["h4_large_net_to_turnover", "h4_retail_net_to_turnover"],
        "relation": "sign(xl)*sign(s) 方向分歧",
        "not_rename_of": [],
    },
    "r21_h4_large_flow_persist": {
        "near": ["h1_consistent_buy_ratio", "g1_ret_5"],
        "relation": "过去 3–5 日大单净流入方向一致性 × D 日强度",
        "not_rename_of": ["h1_consistent_buy_ratio"],
    },
    "r21_i1_failed_board_ratio_20": {
        "near": ["g3_days_since_limit"],
        "relation": "过去 20 日炸板次数/触板次数；不是距上次涨停天数",
        "not_rename_of": ["g3_days_since_limit"],
    },
    "r21_i2_days_since_failed": {
        "near": ["g3_days_since_limit"],
        "relation": "距上次炸板交易日；无历史 → NA 不是大常数",
        "not_rename_of": ["g3_days_since_limit"],
    },
    "r21_i3_failed_recurrence_w20": {
        "near": ["g3_days_since_limit"],
        "relation": "冻结递减权 0.9^k 的炸板加权",
        "not_rename_of": ["g3_days_since_limit"],
    },
    "r21_i4_hit_to_close_conversion": {
        "near": ["g3_days_since_limit", "fb_close_seal_streak_norm"],
        "relation": "历史收盘封住/触板",
        "not_rename_of": ["fb_close_seal_streak_norm"],
    },
    "r21_j1_preboard_turnover_5": {
        "near": ["g0_amt_vs_5", "g1_ret_5"],
        "relation": "D-5..D-1 换手累计，不含 D 日",
        "not_rename_of": ["g0_amt_vs_5"],
    },
    "r21_j2_preboard_px_vwap20": {
        "near": ["g1_ret_5"],
        "relation": "Close_{D-1}/VWAP_20-1，成交量加权均价代理",
        "not_rename_of": [],
    },
    "r21_j3_preboard_high_prox": {
        "near": ["g1_ret_5"],
        "relation": "Close_{D-1}/max(High_{D-20..D-1})-1",
        "not_rename_of": [],
    },
    "r21_j4_prior_ret_per_turnover": {
        "near": ["g1_ret_5", "g0_amt_vs_5"],
        "relation": "D-5..D-1 收益 / 换手累计",
        "not_rename_of": ["g1_ret_5"],
    },
    "r21_k1_pre_hit_vol_frac": {
        "near": ["fb_limit_close_minute_ratio", "h1_last15m_volume_ratio"],
        "relation": "首触前成交量/全日；不是封住分钟占比",
        "not_rename_of": ["fb_limit_close_minute_ratio", "r21_a4_tail_seal_share"],
    },
    "r21_k2_vol_time_dispersion": {
        "near": ["h1_last15m_volume_ratio"],
        "relation": "成交量时间二阶矩",
        "not_rename_of": [],
    },
    "r21_k3_minute_vol_entropy": {
        "near": ["h6_volume_weighted_entropy"],
        "relation": "必须与供应商 volume_weighted_entropy 做等价检验",
        "not_rename_of": ["h6_volume_weighted_entropy"],
    },
    "r21_k4_signed_path_vol_pressure": {
        "near": ["g_prim_px_vol_corr"],
        "relation": "分钟收盘方向×量 / 总量；不是真实主动买卖",
        "not_rename_of": ["g_prim_px_vol_corr", "r18_f1"],
    },
    "r21_l1_known_float_chg": {
        "near": ["r21_d3_holder_per_circ", "r21_d4_holder_x_float"],
        "relation": "log 已知流通股本比；用 available_at",
        "not_rename_of": ["r21_d1_log_holder_ratio"],
    },
    "r21_l2_float_shock_vs_liq": {
        "near": ["g0_amount_log"],
        "relation": "已知流通变动 / 20 日均成交股数",
        "not_rename_of": [],
    },
    "r21_l3_days_since_float_up_ann": {
        "near": ["g3_days_since_limit"],
        "relation": "距最近已知流通增加公告交易日",
        "not_rename_of": ["g3_days_since_limit"],
    },
    "r21_l4_turnover_x_float_chg": {
        "near": ["g0_amt_vs_5"],
        "relation": "D 日换手 × 已知流通变动比",
        "not_rename_of": ["r21_d4_holder_x_float"],
    },
    "r21_m1_rev_yoy_change": {
        "near": [],
        "relation": "最新 − 上一期 营收同比；财务 PIT",
        "not_rename_of": [],
    },
    "r21_m2_ocf_over_abs_np": {
        "near": [],
        "relation": "经营现金流 / |净利润|；零/负利润单独 NA",
        "not_rename_of": [],
    },
    "r21_m3_earnings_info_age": {
        "near": ["g3_days_since_limit"],
        "relation": "距最近财务披露交易日",
        "not_rename_of": [],
    },
    "r21_m4_growth_quality_div": {
        "near": [],
        "relation": "营收同比 − 净利润同比；不是分析师 surprise",
        "not_rename_of": [],
    },
    "r21_n1_lagged_peer_limit_up": {
        "near": ["ind_seal_rate_excess"],
        "relation": "同行 D-1..D-5 涨停指示加权；排除自身",
        "not_rename_of": ["ind_seal_rate_excess", "r21_b1_fb_amount_share"],
    },
    "r21_n2_peer_return_div": {
        "near": ["ind_seal_rate_excess", "g1_ret_5"],
        "relation": "自身近收益 − 同行加权近收益",
        "not_rename_of": ["ind_seal_rate_excess"],
    },
    "r21_n3_peer_failed_spillover": {
        "near": ["ind_seal_rate_excess"],
        "relation": "同行 D-1..D-10 炸板指示",
        "not_rename_of": ["r21_b3_amt_surprise_vs_peer"],
    },
    "r21_n4_peer_adj_fb_intensity": {
        "near": ["r21_b1_fb_amount_share", "r21_b2_seal_rel_strength"],
        "relation": "自身 D 日首板强度 − 同行当日强度；与 R21 B 族去重",
        "not_rename_of": ["r21_b1_fb_amount_share", "r21_b2_seal_rel_strength", "r21_b3_amt_surprise_vs_peer"],
    },
}


def run_lineage(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and LINEAGE_CSV.is_file() and LINEAGE_JSON.is_file():
        from .common import load_json

        existing = load_json(LINEAGE_JSON)
        if existing.get("status") == "DONE":
            return existing
    rows = []
    for fid in NEW_FACTOR_IDS:
        card = PRIOR.get(fid) or {}
        rows.append(
            {
                "new_factor": fid,
                "family": fid.split("_")[1][0].upper(),
                "historical_near_neighbors": "|".join(card.get("near") or []),
                "formula_data_relation": card.get("relation"),
                "explicitly_not_rename_of": "|".join(card.get("not_rename_of") or []),
                "legacy_14_13": "|".join([*FIVE_FEATURES] if False else []),
                "do_not_rename_failed_historical": True,
            }
        )
    LINEAGE_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(LINEAGE_CSV, index=False)
    also = ROUND21_2_ROOT / "factor_lineage.csv"
    pd.DataFrame(rows).to_csv(also, index=False)
    payload = {
        "status": "DONE",
        "n": len(rows),
        "legacy_five": list(FIVE_FEATURES),
        "legacy_ten": list(TEN_FEATURES),
        "path": str(LINEAGE_CSV),
        "do_not_rename_failed_historical": True,
    }
    dump_json(LINEAGE_JSON, payload)
    return payload
