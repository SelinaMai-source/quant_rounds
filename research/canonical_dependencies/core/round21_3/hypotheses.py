"""冻结 T/U/V 12–18 个假设卡。看收益前冻结。代数≡G/H/K/L 记 DUPLICATE。"""

from __future__ import annotations

from typing import Any

import yaml

from ..paths import ROUND21_3_ROOT
from .common import dump_json, sha256_json, write_md
from .contract import DUPLICATE, MD, NEW_FACTOR_IDS

HYP_YAML = ROUND21_3_ROOT / "hypotheses" / "hypotheses.yaml"
HYP_JSON = ROUND21_3_ROOT / "hypotheses" / "hypotheses.json"

HYPOTHESES: list[dict[str, Any]] = [
    {
        "factor_id": "r21_t1_ticket_vs_med20",
        "family": "T",
        "short": "单笔金额相对 20 日中位异常",
        "formula": "(amount/DEAL_AMOUNT) / median_20d(amount/DEAL_AMOUNT) - 1",
        "inputs": ["daily_raw.amount", "mkt_equd.DEAL_AMOUNT"],
        "hypothesis": "相对历史中位的票面异常，不是 G1 水平值",
        "expected_direction": "极端正异常削弱接力",
        "decision_time": "D close known_at",
        "redundancy": "≠ r21_g1_avg_ticket_value (level)",
        "falsify": "若与 G1 Spearman |ρ|≥0.999 → DUPLICATE",
        "near": ["r21_g1_avg_ticket_value", "g0_amount_log"],
        "fail": "DEAL_AMOUNT 缺失则 NA，不填 0",
    },
    {
        "factor_id": "r21_t2_bucket_js_div",
        "family": "T",
        "short": "档位分布 vs 历史 JS 散度",
        "formula": "JS(p_today(S,M,L,XL), mean_20d p) on BUY_ORD_* shares",
        "inputs": ["BUY_ORD_S/M/L/XL"],
        "hypothesis": "当日档位结构相对自身历史偏移提供订单结构信息",
        "expected_direction": "高散度 = 结构突变",
        "decision_time": "D close",
        "redundancy": "≠ G4 (level share gap). 档位定义重叠则禁",
        "falsify": "字段缺失或档位镜像 → BLOCKED_SEMANTICS",
        "near": ["r21_g4_large_ticket_conc_gap"],
        "fail": "无订单档位则 NA",
    },
    {
        "factor_id": "r21_t3_large_ticket_asym_chg",
        "family": "T",
        "short": "大单买卖笔均不对称的历史相对变化",
        "formula": "G2_t - G2_{t-1}  where G2=log((BVolXL/BOrdXL)/(SVolXL/SOrdXL))",
        "inputs": ["BUY_VOL_XL", "BUY_ORD_XL", "SELL_VOL_XL", "SELL_ORD_XL"],
        "hypothesis": "不对称的一阶差分，不是 G2 水平",
        "expected_direction": "差分转正 = 大买单加码",
        "decision_time": "D close",
        "redundancy": "代数=G2 则 DUPLICATE；本定义是 ΔG2",
        "falsify": "若序列恒等于 G2 → DUPLICATE",
        "near": ["r21_g2_large_ticket_asym"],
        "fail": "G2 不足两日则 NA",
    },
    {
        "factor_id": "r21_t4_dlog_count_minus_dlog_amt",
        "family": "T",
        "short": "Δlog(笔数)−Δlog(额)",
        "formula": "Δlog(DEAL_AMOUNT) - Δlog(amount)",
        "inputs": ["mkt_equd.DEAL_AMOUNT", "daily_raw.amount"],
        "hypothesis": "笔数增速相对额增速：碎单化 vs 大单化",
        "expected_direction": "正值 = 笔数快于额 = 碎单",
        "decision_time": "D close",
        "redundancy": "≠ G1/G3",
        "falsify": "DEAL_AMOUNT 与 amount 同步缺失",
        "near": ["r21_g1_avg_ticket_value", "r21_g3_order_count_imbalance"],
        "fail": "非正值取 NA",
    },
    {
        "factor_id": "r21_t5_deal_count_vs_med20",
        "family": "T",
        "short": "成交笔数相对 20 日中位",
        "formula": "DEAL_AMOUNT / median_20d(DEAL_AMOUNT) - 1",
        "inputs": ["mkt_equd.DEAL_AMOUNT"],
        "hypothesis": "笔数自身异常，不是票面金额",
        "expected_direction": "极端放量碎单",
        "decision_time": "D close",
        "redundancy": "≠ T1（T1 用 amount/deal）",
        "falsify": "字段不支持则不计算",
        "near": ["g0_amt_vs_5"],
        "fail": "DEAL_AMOUNT 缺失则 NA",
    },
    {
        "factor_id": "r21_u1_pre_post_hit_impact",
        "family": "U",
        "short": "触板前后冲击强度差",
        "formula": "mean(|Δclose|)_{post-hit} - mean(|Δclose|)_{pre-hit}",
        "inputs": ["min1.close", "first_limit_time"],
        "hypothesis": "触板后冲击相对触板前的变化刻画封板质量",
        "expected_direction": "触板后冲击下降 = 封住更稳",
        "decision_time": "D 15:00",
        "redundancy": "≠ K4 signed path（K4 是全日符号量）",
        "falsify": "无触板时刻则 NA",
        "near": ["r21_k4_signed_path_vol_pressure"],
        "fail": "分钟缺失 NA",
    },
    {
        "factor_id": "r21_u2_limit_vs_pullback_vol",
        "family": "U",
        "short": "触板后涨停价量 vs 回落分钟量",
        "formula": "vol(close>=limit*0.999, post-hit) / vol(close<limit*0.999, post-hit) - 1",
        "inputs": ["min1.volume", "min1.close", "limit_price", "touch"],
        "hypothesis": "触板后仍在板上的量相对回落量",
        "expected_direction": "板上量占优更好",
        "decision_time": "D 15:00",
        "redundancy": "≠ K1 pre-hit vol frac",
        "falsify": "未触板 NA",
        "near": ["r21_k1_pre_hit_vol_frac"],
        "fail": "无回落分钟则 NA 不填 0",
    },
    {
        "factor_id": "r21_u3_state_persist_switch",
        "family": "U",
        "short": "板上/回落有限状态持续与切换",
        "formula": "n_switches / n_post_bars ; persist = longest onboard run / n_post",
        "inputs": ["min1.close", "limit_price", "touch"],
        "hypothesis": "非 tick 盘口的状态机：切换越多越不稳",
        "expected_direction": "高切换差",
        "decision_time": "D 15:00",
        "redundancy": "≠ 盘口 tick",
        "falsify": "无触板 NA",
        "near": ["fb_close_seal_streak_norm"],
        "fail": "分钟不足 NA",
    },
    {
        "factor_id": "r21_u4_pre_post_vol_conc",
        "family": "U",
        "short": "触板前后量集中度差",
        "formula": "HHI(vol)_post - HHI(vol)_pre",
        "inputs": ["min1.volume", "touch"],
        "hypothesis": "触板后量更集中或更分散；控 K2/K3（K2=时间二阶矩，K3=全日熵）",
        "expected_direction": "触板后更集中更好",
        "decision_time": "D 15:00",
        "redundancy": "≠ K2/K3 全日统计",
        "falsify": "|ρ| vs K2/K3 ≥0.999 → DUPLICATE",
        "near": ["r21_k2_vol_time_dispersion", "r21_k3_minute_vol_entropy"],
        "fail": "无触板 NA",
    },
    {
        "factor_id": "r21_u5_open_recover_eff",
        "family": "U",
        "short": "开板恢复效率 + Event Flag",
        "formula": "若曾开板：回板分钟数倒数；未开板保持 NA，不填 0",
        "inputs": ["min1.close", "limit_price", "touch"],
        "hypothesis": "开板后快速回封优于久不回封；未开板是另一状态，禁止填 0",
        "expected_direction": "恢复越快越好",
        "decision_time": "D 15:00",
        "redundancy": "Event flag 与连续值分列",
        "falsify": "未开板填 0 → 违规",
        "near": ["r21_i4_hit_to_close_conversion"],
        "fail": "未开板 NA + opened_flag=0",
    },
    {
        "factor_id": "r21_v1_ann_to_effective_days",
        "family": "V",
        "short": "公告已知日→生效日交易日数",
        "formula": "trading_days(available_at → effective_from) of latest known float event",
        "inputs": ["share_capital.available_at", "effective_from"],
        "hypothesis": "公告到生效的前瞻窗口，不是 L3（距上次增发公告已过天数）",
        "expected_direction": "窗口越近供给冲击越近",
        "decision_time": "asof available_at ≤ D 15:00",
        "redundancy": "≠ r21_l3_days_since_float_up_ann",
        "falsify": "无 effective_from → NA",
        "near": ["r21_l3_days_since_float_up_ann"],
        "fail": "生效日已过仍报剩余负数？→ clip NA if effective already passed without new event",
    },
    {
        "factor_id": "r21_v2_classified_float_shock",
        "family": "V",
        "short": "分类股本冲击",
        "formula": "sign(float_chg) * log1p(|Δtradable|) 分增发/解禁桶",
        "inputs": ["share_capital.tradable_shares", "float_chg"],
        "hypothesis": "冲击方向×规模，不是连续 L1 对数变化本身",
        "expected_direction": "正向供给冲击不利",
        "decision_time": "asof available_at",
        "redundancy": "≠ L1 raw log_chg",
        "falsify": "无 prev_float NA",
        "near": ["r21_l1_known_float_chg"],
        "fail": "分类字段缺失则仅用符号桶",
    },
    {
        "factor_id": "r21_v3_float_shock_vs_liq",
        "family": "V",
        "short": "股本冲击/20日量（对照 L2）",
        "formula": "(tradable - prev) / vol20   【若代数=L2 → DUPLICATE，沿用 L2 不新开训练 ID】",
        "inputs": ["share_capital", "daily_raw.volume"],
        "hypothesis": "与 L2 等价则 DUPLICATE",
        "expected_direction": "同 L2",
        "decision_time": "asof",
        "redundancy": "≡ r21_l2_float_shock_vs_liq → DUPLICATE",
        "falsify": "逐元素相等或 |ρ|≥0.999",
        "near": ["r21_l2_float_shock_vs_liq"],
        "fail": "DUPLICATE 不进新四路",
        "duplicate_of": "r21_l2_float_shock_vs_liq",
    },
    {
        "factor_id": "r21_v4_future_event_overlap",
        "family": "V",
        "short": "已公告未来窗内多事件重叠",
        "formula": "count(events with available_at≤D and effective_from in (D, D+20])",
        "inputs": ["share_capital.available_at", "effective_from"],
        "hypothesis": "未来 20 日已公告供给事件重叠",
        "expected_direction": "重叠越多供给压力越大",
        "decision_time": "asof available_at ≤ D",
        "redundancy": "≠ L3 回顾天数",
        "falsify": "无 effective_from → NA",
        "near": ["r21_l3_days_since_float_up_ann"],
        "fail": "未来窗不可观测则 NA",
    },
]


def require_hypotheses() -> list[dict[str, Any]]:
    if len(HYPOTHESES) < 12 or len(HYPOTHESES) > 18:
        raise RuntimeError(f"T/U/V must freeze 12–18, got {len(HYPOTHESES)}")
    ids = [h["factor_id"] for h in HYPOTHESES]
    if tuple(ids) != NEW_FACTOR_IDS:
        raise RuntimeError(f"hypothesis ids != NEW_FACTOR_IDS: {ids}")
    return list(HYPOTHESES)


def run_hypotheses(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and HYP_JSON.is_file() and MD["03"].is_file():
        from .common import load_json

        existing = load_json(HYP_JSON)
        if existing.get("status") == "FROZEN":
            return existing
    cards = require_hypotheses()
    payload = {
        "status": "FROZEN",
        "n": len(cards),
        "ids": [c["factor_id"] for c in cards],
        "cards": cards,
        "duplicate_predeclared": ["r21_v3_float_shock_vs_liq"],
        "v5_v6_preregister": ["r21_v5", "r21_v6"],
        "v5_v6_note": "仅 V1–V4 PIT 通过后预注册交互，本轮不先算",
        "sha256": sha256_json(cards),
    }
    dump_json(HYP_JSON, payload)
    HYP_YAML.parent.mkdir(parents=True, exist_ok=True)
    HYP_YAML.write_text(yaml.safe_dump(cards, allow_unicode=True, sort_keys=False), encoding="utf-8")
    lines = ["# Round21.3 · 03 T/U/V 假设（看收益前冻结）", "", f"N=`{len(cards)}`。代数≡G/H/K/L 记 `{DUPLICATE}`。", ""]
    for c in cards:
        lines.append(f"## `{c['factor_id']}`")
        lines.append(f"- 假设：{c['hypothesis']}")
        lines.append(f"- 公式：`{c['formula']}`")
        lines.append(f"- 字段：{c['inputs']}")
        lines.append(f"- 决策时点：{c['decision_time']}")
        lines.append(f"- 方向：{c['expected_direction']}")
        lines.append(f"- 冗余：{c['redundancy']}")
        lines.append(f"- 证伪：{c['falsify']}")
        lines.append(f"- 失败：{c['fail']}")
        lines.append("")
    write_md(MD["03"], "\n".join(lines))
    return payload
