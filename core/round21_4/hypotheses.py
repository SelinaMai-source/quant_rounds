"""W/X/Y/Z 预注册假设卡。看收益前冻结。等价记 DUPLICATE。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND21_4_ROOT
from .common import dump_json, write_md
from .contract import MD, WXYZ_FACTOR_IDS, WXYZ_MECHANISMS

HYP_JSON = ROUND21_4_ROOT / "hypotheses" / "hypotheses.json"

HYPOTHESES: tuple[dict[str, Any], ...] = (
    {
        "factor_id": "r21_w1_updown_vol_ratio",
        "family": "W",
        "short": "涨日/跌日量比",
        "formula": "mean(V|ret>0, 20d lag1) / mean(V|ret<0, 20d lag1) - 1",
        "inputs": ["daily_raw.volume", "daily_raw.close"],
        "hypothesis": "首板前上涨日放量 vs 下跌日缩量，反映资金响应弹性，≠资金流分档",
        "pit": "仅用 trade_date 之前 20 日；当日不计入",
        "unit": "ratio_minus_1",
        "near": ["g0_amt_vs_5", "fb_last30_amount_ratio"],
    },
    {
        "factor_id": "r21_w2_impact_instability",
        "family": "W",
        "short": "冲击不稳定",
        "formula": "std(|ret| / (turnover+eps), 20d lag1)",
        "inputs": ["daily_raw.close", "daily_raw.turnover"],
        "hypothesis": "单位换手冲击的不稳定度",
        "pit": "lag1 rolling",
        "unit": "std",
        "near": ["g_prim_px_vol_corr"],
    },
    {
        "factor_id": "r21_w3_volume_innov_residual",
        "family": "W",
        "short": "成交创新残差",
        "formula": "amount_{t-1} / max(amount, 20d lag2) - 1",
        "inputs": ["daily_raw.amount"],
        "hypothesis": "成交额相对近窗高点的创新，残差化避免水平量纲",
        "pit": "max 窗口不含当日",
        "unit": "ratio_minus_1",
        "near": ["g0_amount_log", "g0_amt_vs_5"],
    },
    {
        "factor_id": "r21_w4_response_asymmetry",
        "family": "W",
        "short": "响应不对称",
        "formula": "corr(ret+, V) - corr(ret-, V) over 20d lag1",
        "inputs": ["daily_raw.volume", "daily_raw.close"],
        "hypothesis": "涨跌方向上量价响应不对称",
        "pit": "lag1",
        "unit": "corr_diff",
        "near": ["g_prim_px_vol_corr"],
    },
    {
        "factor_id": "r21_x1_near_limit_unsealed_freq20",
        "family": "X",
        "short": "近板未封频率",
        "formula": "mean(high>=0.98*limit_up & close<limit_up, 20d lag1); 分 10%/20%/ST",
        "inputs": ["daily_raw.high", "daily_raw.close", "daily_raw.limit_up"],
        "hypothesis": "经常摸板但不封，结构与 I 族炸板不同",
        "pit": "lag1；按 limit/pre_close 分档，不混 10/20/ST",
        "unit": "freq",
        "near": ["r21_i1_failed_board_ratio_20", "g3_days_since_limit"],
    },
    {
        "factor_id": "r21_x2_close_to_limit_gap",
        "family": "X",
        "short": "收盘缺口",
        "formula": "(limit_up - close) / limit_up 当 high>=0.98*limit 否则 NA，20d lag1 均值",
        "inputs": ["daily_raw.close", "daily_raw.limit_up", "daily_raw.high"],
        "hypothesis": "近板日收盘离板距离",
        "pit": "lag1，无近板 → NA 不填 0",
        "unit": "gap",
        "near": ["d1_open_gap"],
    },
    {
        "factor_id": "r21_x3_near_limit_persist",
        "family": "X",
        "short": "近板持续",
        "formula": "consecutive near-limit days ending t-1",
        "inputs": ["daily_raw.high", "daily_raw.limit_up"],
        "hypothesis": "连续逼近涨停的持续天数",
        "pit": "lag1 streak",
        "unit": "days",
        "near": ["fb_close_seal_streak_norm"],
    },
    {
        "factor_id": "r21_x4_touch_day_vol_share",
        "family": "X",
        "short": "触板日量占比",
        "formula": "sum(V|near_limit, 20d lag1) / sum(V, 20d lag1)",
        "inputs": ["daily_raw.volume", "daily_raw.high", "daily_raw.limit_up"],
        "hypothesis": "近板日成交占窗口比重",
        "pit": "lag1",
        "unit": "share",
        "near": ["fb_last30_amount_ratio"],
    },
    {
        "factor_id": "r21_y1_turnover_innov_minus_peer",
        "family": "Y",
        "short": "换手创新减 Peer",
        "formula": "(TO/max20 - 1) - peer_median(same)",
        "inputs": ["daily_raw.turnover"],
        "hypothesis": "相对同业的换手创新",
        "pit": "peer=同交易所×代码前3位，exclude-self，用 t-1 截面",
        "unit": "diff",
        "near": ["g0_amt_vs_5"],
    },
    {
        "factor_id": "r21_y2_turnover_accel_diff",
        "family": "Y",
        "short": "换手加速度差",
        "formula": "dTO_5 - peer_median(dTO_5)",
        "inputs": ["daily_raw.turnover"],
        "hypothesis": "换手加速度相对 peer",
        "pit": "lag1 exclude-self",
        "unit": "diff",
        "near": ["g0_amt_vs_5"],
    },
    {
        "factor_id": "r21_y3_peer_quantile_exposure",
        "family": "Y",
        "short": "分位暴露",
        "formula": "cs_rank(turnover)_{t-1} within peer",
        "inputs": ["daily_raw.turnover"],
        "hypothesis": "在 peer 内的换手分位",
        "pit": "t-1 截面，peer>=5 否则 NA",
        "unit": "rank_0_1",
        "near": ["ind_seal_rate_excess"],
    },
    {
        "factor_id": "r21_y4_impact_diff_vs_peer",
        "family": "Y",
        "short": "冲击差",
        "formula": "|ret|/TO - peer_median",
        "inputs": ["daily_raw.close", "daily_raw.turnover"],
        "hypothesis": "单位换手冲击相对 peer",
        "pit": "lag1 exclude-self",
        "unit": "diff",
        "near": ["g_prim_px_vol_corr"],
    },
    {
        "factor_id": "r21_z1_vol5_vol20_ratio",
        "family": "Z",
        "short": "5日/20日波动比",
        "formula": "std(ret,5)/std(ret,20) lag1",
        "inputs": ["daily_raw.close"],
        "hypothesis": "波动压缩/扩张",
        "pit": "lag1；零区间 NA",
        "unit": "ratio",
        "near": ["g1_ret_5"],
    },
    {
        "factor_id": "r21_z2_downside_semivar_ratio",
        "family": "Z",
        "short": "下行半方差比",
        "formula": "semivar_down20 / var20 lag1",
        "inputs": ["daily_raw.close"],
        "hypothesis": "下行风险占比",
        "pit": "lag1；零方差 NA",
        "unit": "ratio",
        "near": ["g1_ret_5"],
    },
    {
        "factor_id": "r21_z3_vol_per_turnover",
        "family": "Z",
        "short": "波动/换手",
        "formula": "std(ret,20) / mean(TO,20) lag1",
        "inputs": ["daily_raw.close", "daily_raw.turnover"],
        "hypothesis": "单位换手的波动",
        "pit": "lag1；零换手 NA",
        "unit": "ratio",
        "near": ["g0_amt_vs_5"],
    },
    {
        "factor_id": "r21_z4_range_position",
        "family": "Z",
        "short": "区间位置/突破",
        "formula": "(close_{t-1} - min20) / (max20 - min20)",
        "inputs": ["daily_raw.close", "daily_raw.high", "daily_raw.low"],
        "hypothesis": "价格在 20 日区间中的位置；零区间 NA",
        "pit": "lag1",
        "unit": "pos_0_1",
        "near": ["g1_ret_5"],
    },
)


def require_hypotheses() -> tuple[dict[str, Any], ...]:
    ids = [h["factor_id"] for h in HYPOTHESES]
    if list(ids) != list(WXYZ_FACTOR_IDS):
        raise RuntimeError(f"hypothesis ids != frozen WXYZ_FACTOR_IDS: {ids}")
    return HYPOTHESES


def run_hypotheses(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_hypotheses()
    payload = {
        "status": "FROZEN",
        "n": len(HYPOTHESES),
        "ids": list(WXYZ_FACTOR_IDS),
        "mechanisms": dict(WXYZ_MECHANISMS),
        "cards": list(HYPOTHESES),
    }
    dump_json(HYP_JSON, payload)
    write_md(
        MD["20"],
        "# Round21.4 · 20 W/X/Y/Z 预注册\n\n"
        f"N=`{len(HYPOTHESES)}` 机制=`{list(WXYZ_MECHANISMS)}`。看收益前冻结。等价记 DUPLICATE。缺失不填 0。\n",
    )
    return payload
