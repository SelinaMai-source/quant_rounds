"""Round21 六大家族 ≥18 假设卡。公式在代码/yaml，不只写 markdown。看 NAV 前冻结。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND21_ROOT, assert_project_result_path, ensure_round21_tree
from .common import dump_json, load_json, write_md
from .contract import (
    BLOCKED_DATA,
    BLOCKED_PIT,
    DIAGNOSTIC_ONLY,
    F1_FORMULA,
    F1_NAME,
    HYP_MD,
    NEAR_NEIGHBORS,
    PIT_ELIGIBLE_MODEL,
)

HYP_JSON = ROUND21_ROOT / "hypotheses" / "hypotheses.json"
HYP_YAML = ROUND21_ROOT / "hypotheses" / "hypotheses.yaml"
CARDS_DIR = ROUND21_ROOT / "hypotheses" / "cards"
FACTORS_DIR = ROUND21_ROOT / "factors"

# 公式必须新。A1 ≠ R18 F1（F1 = V_seal / (A_below + 0.01*A_day)）。
# 分钟只允许 close-limit 子集上的个股自身 bar，禁止全市场分钟均值。
# 缺失不填 0。非两融标的融资 NULL 不填 0。
HYPOTHESES: tuple[dict[str, Any], ...] = (
    {
        "factor_id": "r21_a1_vw_limit_gap",
        "family": "A",
        "short": "量加权利差",
        "formula": (
            "sum((limit_px - close_m) * amount_m | bar_sec >= touch_sec, close_m < limit_px) "
            "/ sum(amount_m | bar_sec >= touch_sec, close_m < limit_px, amount_m > 0)"
        ),
        "formula_note": (
            "首次有效触板后、收盘价仍低于涨停价的分钟，用成交额加权的 (涨停价-分钟收盘)。"
            f"不是 R18 {F1_NAME}={F1_FORMULA}。"
        ),
        "inputs": {
            "min1": ["bar_time", "close", "amount"],
            "daily_limit_events": ["limit_price", "first_limit_time_seconds"],
            "daily_raw": ["limit_up"],
        },
        "pit_rule": "分钟与涨停事件均属当日 D 收盘前可知；决策日=D",
        "fillna_zero": False,
        "near_neighbors": list(NEAR_NEIGHBORS["r21_a1_vw_limit_gap"]),
        "near_neighbor_note": "A1 ≠ R18 F1；F1 是 V_seal/A_below 吸收比，本式是量加权利差",
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
    {
        "factor_id": "r21_a2_minutes_to_reseal",
        "family": "A",
        "short": "恢复分钟数",
        "formula": (
            "(min(bar_sec | bar_sec >= touch_sec, close_m >= limit_px, bar_sec > first_leave_sec) "
            "- touch_sec) / 60"
        ),
        "formula_note": (
            "first_leave_sec = 首次触板后第一次 close_m < limit_px 的 bar_sec。"
            "若全天未再封住则 NaN，不填 0。"
        ),
        "inputs": {
            "min1": ["bar_time", "close"],
            "daily_limit_events": ["limit_price", "first_limit_time_seconds"],
        },
        "pit_rule": "当日分钟路径，决策日=D",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
    {
        "factor_id": "r21_a3_retouch_count",
        "family": "A",
        "short": "再触板次数",
        "formula": (
            "count(upcross_i) where upcross = (high_m[t] >= limit_px) & (close_m[t-1] < limit_px) "
            "& (bar_sec > touch_sec)"
        ),
        "formula_note": "首次触板后，从打开到再次 high>=涨停价的次数。无分钟则 BLOCKED_DATA。",
        "inputs": {
            "min1": ["bar_time", "high", "close"],
            "daily_limit_events": ["limit_price", "first_limit_time_seconds"],
        },
        "pit_rule": "当日分钟路径，决策日=D",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
    {
        "factor_id": "r21_a4_tail_seal_share",
        "family": "A",
        "short": "末端封板占比",
        "formula": "mean(I(close_m >= limit_px) | bar_sec >= 14:30 Asia/Shanghai)",
        "formula_note": (
            "末 30 分钟封住分钟占比。近邻 fb_close_seal_streak_norm 是连续封住长度归一化，"
            "本式是末端窗口份额，不是 streak。"
        ),
        "inputs": {
            "min1": ["bar_time", "close"],
            "daily_limit_events": ["limit_price"],
        },
        "pit_rule": "当日分钟路径，决策日=D",
        "fillna_zero": False,
        "near_neighbors": list(NEAR_NEIGHBORS["r21_a4_tail_seal_share"]),
        "near_neighbor_note": "A4 ≈ fb_close_seal_streak_norm 风险；公式不同，簇内必须 Original/New/Add/Replace",
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
    {
        "factor_id": "r21_b1_fb_amount_share",
        "family": "B",
        "short": "金额份额",
        "formula": "amount_i / sum(amount_j | j in same-day first-board candidate pool)",
        "formula_note": (
            "候选池内金额份额，不是全市场、也不是横截面 rank。"
            "近邻 r17_d_amt_xs_rank 是金额 xs rank。分钟不得构造全市场均值。"
            "行业竞争仅当 industry_membership.known_at 可 PIT 核验时另开卡片，本卡用候选池。"
        ),
        "inputs": {"daily_raw": ["amount"], "candidate_panel": ["trade_date", "instrument_id"]},
        "pit_rule": "D 日成交额，决策日=D 收盘后",
        "fillna_zero": False,
        "near_neighbors": list(NEAR_NEIGHBORS["r21_b1_fb_amount_share"]),
        "near_neighbor_note": "B1 ≈ r17_d_amt_xs_rank 风险；份额 ≠ rank",
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
    {
        "factor_id": "r21_b2_seal_rel_strength",
        "family": "B",
        "short": "封板相对强度",
        "formula": "close_limit_value_i / median(close_limit_value_j | j in same-day first-board pool)",
        "formula_note": "相对同日首板候选中位数。分母为 NaN 或 median 不可得则本行 NaN。",
        "inputs": {
            "daily_limit_events": ["close_limit_value"],
            "candidate_panel": ["trade_date", "instrument_id"],
        },
        "pit_rule": "D 日封单额，决策日=D",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
    {
        "factor_id": "r21_b3_amt_surprise_vs_peer",
        "family": "B",
        "short": "成交惊喜−peer",
        "formula": (
            "(amount_i / mean(amount_i, 5d, min_periods=3)) "
            "- median(amount_j / mean(amount_j, 5d, min_periods=3) | j in same-day first-board pool)"
        ),
        "formula_note": "个股 5 日均额惊喜减去同日候选中位数。日频 daily_raw，不是分钟全市场。",
        "inputs": {"daily_raw": ["amount"], "candidate_panel": ["trade_date", "instrument_id"]},
        "pit_rule": "D 及此前日额，决策日=D",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
    {
        "factor_id": "r21_b4_activity_x_resilience",
        "family": "B",
        "short": "活跃×韧性交互",
        "formula": "r21_b1_fb_amount_share * (1 / (1 + r21_a2_minutes_to_reseal))",
        "formula_note": (
            "预注册交互，不是事后搜出来的乘积。A2 为 NaN 则本交互 NaN。"
            "构造阶段先算 A2/B1 再乘，禁止用 0 顶 A2。"
        ),
        "inputs": {"depends_on": ["r21_b1_fb_amount_share", "r21_a2_minutes_to_reseal"]},
        "pit_rule": "依赖因子 PIT 的较严者",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
    {
        "factor_id": "r21_c1_min1_amihud",
        "family": "C",
        "short": "分钟 Amihud",
        "formula": (
            "mean(|close_m/open_m - 1| / amount_m | bar_sec >= touch_sec, amount_m > 0, open_m > 0)"
        ),
        "formula_note": "触板后分钟冲击。amount=0 的 bar 剔除，不把分母改成 1。",
        "inputs": {
            "min1": ["bar_time", "open", "close", "amount"],
            "daily_limit_events": ["first_limit_time_seconds"],
        },
        "pit_rule": "当日分钟，决策日=D",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
    {
        "factor_id": "r21_c2_post_touch_amt_vs_impact",
        "family": "C",
        "short": "触板后量vs冲击",
        "formula": (
            "sum(amount_m | bar_sec >= touch_sec) "
            "/ max(max(high_m)-min(low_m), tick) | bar_sec >= touch_sec"
        ),
        "formula_note": "触板后成交额 / 触板后价格振幅。振幅 0 则 NaN。",
        "inputs": {
            "min1": ["bar_time", "high", "low", "amount"],
            "daily_limit_events": ["first_limit_time_seconds"],
        },
        "pit_rule": "当日分钟，决策日=D",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
    {
        "factor_id": "r21_c3_post_touch_path_eff",
        "family": "C",
        "short": "路径效率",
        "formula": (
            "abs(close_last_after_touch - close_at_touch) "
            "/ sum(|close_m[t]-close_m[t-1]| | bar_sec >= touch_sec)"
        ),
        "formula_note": (
            "仅触板后路径。近邻 h_min1_path_eff_full 是全日路径效率（R20 入选列）。"
            "本式窗口不同，簇内必须 Original/New/Add/Replace。"
        ),
        "inputs": {
            "min1": ["bar_time", "close"],
            "daily_limit_events": ["first_limit_time_seconds"],
        },
        "pit_rule": "当日分钟，决策日=D",
        "fillna_zero": False,
        "near_neighbors": list(NEAR_NEIGHBORS["r21_c3_post_touch_path_eff"]),
        "near_neighbor_note": "C3 ≈ h_min1_path_eff_full（R20 入选列）",
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
    {
        "factor_id": "r21_c4_downside_quad_var",
        "family": "C",
        "short": "下行二次变差",
        "formula": (
            "sum(min(close_m[t]-close_m[t-1], 0)^2 | bar_sec >= touch_sec) "
            "/ sum((close_m[t]-close_m[t-1])^2 | bar_sec >= touch_sec)"
        ),
        "formula_note": "触板后下行二次变差占比。分母 0 则 NaN。",
        "inputs": {
            "min1": ["bar_time", "close"],
            "daily_limit_events": ["first_limit_time_seconds"],
        },
        "pit_rule": "当日分钟，决策日=D",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
    {
        "factor_id": "r21_d1_log_holder_ratio",
        "family": "D",
        "short": "log 户数比",
        "formula": "log(HOLDER_NUM_t / HOLDER_NUM_{t-1})  where t ordered by ANN_DT per MARKET_CODE",
        "formula_note": (
            "只用 ANN_DT。默认下一交易日起用：known_at = next_trading_day(ANN_DT)。"
            "ANN_DT 空或 HOLDER_NUM<=0 则 NaN。稀疏不填 0。"
        ),
        "inputs": {"holder_num": ["MARKET_CODE", "ANN_DT", "HOLDER_NUM"]},
        "pit_rule": "ANN_DT 的下一交易日；HOLDER_ENDDATE 不作为可用时点",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
    {
        "factor_id": "r21_d2_holder_daily_chg",
        "family": "D",
        "short": "日均变化",
        "formula": "(HOLDER_NUM_t - HOLDER_NUM_{t-1}) / max(calendar_days(ANN_DT_{t-1}, ANN_DT_t), 1)",
        "formula_note": "只用 ANN_DT。下一交易日起用。间隔缺失则 NaN。",
        "inputs": {"holder_num": ["MARKET_CODE", "ANN_DT", "HOLDER_NUM"]},
        "pit_rule": "ANN_DT 的下一交易日",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
    {
        "factor_id": "r21_d3_holder_per_circ",
        "family": "D",
        "short": "户数/流通",
        "formula": "HOLDER_NUM / circulating_shares  as-of asof_date=next_trading_day(ANN_DT)",
        "formula_note": "流通股本取 daily_valuation.circulating_shares，按 valuation.available_at ≤ asof_date。禁止总股本顶替。",
        "inputs": {
            "holder_num": ["MARKET_CODE", "ANN_DT", "HOLDER_NUM"],
            "daily_valuation": ["circulating_shares", "available_at"],
        },
        "pit_rule": "max(next_trading_day(ANN_DT), valuation.available_at)",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
    {
        "factor_id": "r21_d4_holder_x_float",
        "family": "D",
        "short": "户数×流通交互",
        "formula": "log1p(HOLDER_NUM) * log1p(free_float_shares)",
        "formula_note": (
            "预注册交互。free_float_shares 来自 daily_valuation，按 available_at。"
            "任一侧缺失则 NaN。与 D3（户数/流通股本）分母口径不同。"
        ),
        "inputs": {
            "holder_num": ["MARKET_CODE", "ANN_DT", "HOLDER_NUM"],
            "daily_valuation": ["free_float_shares", "available_at"],
        },
        "pit_rule": "max(next_trading_day(ANN_DT), valuation.available_at)",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
    {
        "factor_id": "r21_e1_block_discount",
        "family": "E",
        "short": "折价",
        "formula": "1 - B_SHARE_PRICE / close_raw  on TRADE_DATE",
        "formula_note": (
            "block_trading 只有 TRADE_DATE，无公告时点。可诊断，禁止 PIT_ELIGIBLE_MODEL。"
            "close_raw 取 daily_raw.close。多笔同日取金额加权价后再算折价。"
        ),
        "inputs": {
            "block_trading": ["TRADE_DATE", "MARKET_CODE", "B_SHARE_PRICE", "B_SHARE_AMOUNT"],
            "daily_raw": ["close"],
        },
        "pit_rule": "无 ANN/available_at → 禁模型；仅 DIAGNOSTIC_ONLY",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": DIAGNOSTIC_ONLY,
        "blocked": BLOCKED_PIT,
    },
    {
        "factor_id": "r21_e2_block_vol20_to_circ",
        "family": "E",
        "short": "20 日量/流通",
        "formula": "sum(B_SHARE_VOLUME, 20 trading days ending TRADE_DATE) / circulating_shares",
        "formula_note": "无公告时点，禁模型。流通股按 valuation.available_at。",
        "inputs": {
            "block_trading": ["TRADE_DATE", "MARKET_CODE", "B_SHARE_VOLUME"],
            "daily_valuation": ["circulating_shares", "available_at"],
        },
        "pit_rule": "无 ANN → 禁模型",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": DIAGNOSTIC_ONLY,
        "blocked": BLOCKED_PIT,
    },
    {
        "factor_id": "r21_e3_discount_wtd_pressure",
        "family": "E",
        "short": "折价加权压力",
        "formula": (
            "sum((1 - B_SHARE_PRICE/close_raw) * B_SHARE_AMOUNT, 20 trading days) "
            "/ circulating_market_cap"
        ),
        "formula_note": "无公告时点，禁模型。",
        "inputs": {
            "block_trading": ["TRADE_DATE", "MARKET_CODE", "B_SHARE_PRICE", "B_SHARE_AMOUNT"],
            "daily_raw": ["close"],
            "daily_valuation": ["circulating_market_cap", "available_at"],
        },
        "pit_rule": "无 ANN → 禁模型",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": DIAGNOSTIC_ONLY,
        "blocked": BLOCKED_PIT,
    },
    {
        "factor_id": "r21_e4_block_event_intensity_60",
        "family": "E",
        "short": "60 日事件强度",
        "formula": "sum(B_FREQUENCY, 60 trading days ending TRADE_DATE)",
        "formula_note": "事件次数强度。无公告时点，禁模型。无事件日保持 NaN，不填 0。",
        "inputs": {"block_trading": ["TRADE_DATE", "MARKET_CODE", "B_FREQUENCY"]},
        "pit_rule": "无 ANN → 禁模型",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": DIAGNOSTIC_ONLY,
        "blocked": BLOCKED_PIT,
    },
    {
        "factor_id": "r21_f1_fin_bal_chg",
        "family": "F",
        "short": "余额变化",
        "formula": "financing_balance[available_at] / financing_balance[prev available_at] - 1",
        "formula_note": (
            "按 daily_margin.available_at 对齐。现场 2018-01 样本显示表内只有两融标的行，"
            "非两融标的在表中缺失——join 后保持 NULL，禁止填 0。"
            "抽样 available_at − trade_date ∈ {1,3}，D 收盘默认不可用则滞后到 available_at。"
        ),
        "inputs": {"daily_margin": ["financing_balance", "available_at", "trade_date"]},
        "pit_rule": "known_at = available_at；不得用 trade_date 当天当已发布",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
    {
        "factor_id": "r21_f2_fin_repay_pressure",
        "family": "F",
        "short": "还款压力",
        "formula": "financing_repayment_value / financing_balance.shift(1)  aligned on available_at",
        "formula_note": "分母上一期余额 ≤0 或缺失则 NaN。非两融 NULL。",
        "inputs": {
            "daily_margin": [
                "financing_repayment_value",
                "financing_balance",
                "available_at",
                "trade_date",
            ]
        },
        "pit_rule": "known_at = available_at",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
    {
        "factor_id": "r21_f3_fin_buy_intensity",
        "family": "F",
        "short": "买入强度",
        "formula": "financing_buy_value / amount  with margin.available_at ≤ decision_date",
        "formula_note": "amount 用 daily_raw 同 trade_date。非两融或 amount≤0 则 NaN。",
        "inputs": {
            "daily_margin": ["financing_buy_value", "available_at", "trade_date"],
            "daily_raw": ["amount"],
        },
        "pit_rule": "known_at = available_at",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
    {
        "factor_id": "r21_f4_fin_bal_accel",
        "family": "F",
        "short": "加速度",
        "formula": "r21_f1_fin_bal_chg - r21_f1_fin_bal_chg.shift(1)  aligned on available_at",
        "formula_note": "余额变化的一阶差分。任一侧 NaN 则 NaN。",
        "inputs": {"depends_on": ["r21_f1_fin_bal_chg"]},
        "pit_rule": "known_at = available_at",
        "fillna_zero": False,
        "near_neighbors": [],
        "model_eligible_default": PIT_ELIGIBLE_MODEL,
    },
)


def hypothesis_cards() -> list[dict[str, Any]]:
    return [dict(card) for card in HYPOTHESES]


def _dump_yaml(obj: Any) -> str:
    from ..round11.freeze import _dump_yaml as _y

    return _y(obj)


def require_hypotheses() -> dict[str, Any]:
    payload = load_json(HYP_JSON)
    if not payload.get("frozen"):
        raise RuntimeError("HYPOTHESES_NOT_FROZEN STOP; refuse new-factor NAV")
    if int(payload.get("n") or 0) < 18:
        raise RuntimeError("HYPOTHESES_LT_18")
    return payload


def freeze_hypotheses(*, overwrite: bool = False) -> dict[str, Any]:
    ensure_round21_tree()
    if HYP_JSON.is_file() and not overwrite:
        existing = load_json(HYP_JSON)
        if existing.get("frozen") and existing.get("n", 0) >= 18 and HYP_YAML.is_file():
            return existing
    leaked = []
    cand = ROUND21_ROOT / "candidates"
    if cand.is_dir():
        for path in cand.rglob("*"):
            if path.suffix in {".parquet", ".json"} and path.is_file() and "nav" in path.name.lower():
                leaked.append(str(path))
    if leaked:
        raise RuntimeError(f"REFUSE_HYPOTHESIS_AFTER_CANDIDATE_NAV: {leaked[:8]}")

    cards = hypothesis_cards()
    families = sorted({c["family"] for c in cards})
    missing_f = [f for f in "ABCDEF" if f not in families]
    if missing_f:
        raise RuntimeError(f"families missing: {missing_f}")
    if len(cards) < 18:
        raise RuntimeError("need >=18 hypothesis cards")

    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for card in cards:
        fid = card["factor_id"]
        yml = _dump_yaml(card)
        if not yml.endswith("\n"):
            yml += "\n"
        card_path = CARDS_DIR / f"{fid}.yaml"
        assert_project_result_path(card_path)
        card_path.write_text(yml, encoding="utf-8")
        dest = FACTORS_DIR / fid
        dest.mkdir(parents=True, exist_ok=True)
        hyp_path = dest / "hypothesis.yaml"
        assert_project_result_path(hyp_path)
        hyp_path.write_text(yml, encoding="utf-8")
        written.append({"factor_id": fid, "card": str(card_path), "factor_dir": str(hyp_path)})

    body = {
        "round": 21,
        "n": len(cards),
        "families": families,
        "factor_ids": [c["factor_id"] for c in cards],
        "cards": cards,
        "near_neighbors": dict(NEAR_NEIGHBORS),
        "a1_is_not_r18_f1": True,
        "r18_f1": {"name": F1_NAME, "formula": F1_FORMULA},
        "blocked_codes": [BLOCKED_DATA, BLOCKED_PIT],
        "note": "公式冻结在任何新因子 NAV 之前。本文件不是因子值。",
    }
    text = _dump_yaml(body)
    if not text.endswith("\n"):
        text += "\n"
    assert_project_result_path(HYP_YAML)
    HYP_YAML.write_text(text, encoding="utf-8")
    payload = {
        "frozen": True,
        "status": "FROZEN",
        "n": len(cards),
        "families": families,
        "factor_ids": [c["factor_id"] for c in cards],
        "yaml": str(HYP_YAML),
        "written": written,
        "a1_is_not_r18_f1": True,
        "e_family_default": DIAGNOSTIC_ONLY,
        "values_constructed": False,
    }
    dump_json(HYP_JSON, payload)
    lines = [
        "# Round21 · 假设卡（看新因子 NAV 前冻结）",
        "",
        f"N=`{len(cards)}` 家族=`{families}`。公式在 `core/round21/hypotheses.py` 与 yaml，不只写 markdown。",
        "",
        "## 近邻风险（先验，执行时测）",
        "",
        "- A4 `r21_a4_tail_seal_share` vs `fb_close_seal_streak_norm`",
        "- B1 `r21_b1_fb_amount_share` vs `r17_d_amt_xs_rank`",
        "- C3 `r21_c3_post_touch_path_eff` vs `h_min1_path_eff_full`（R20 入选列）",
        f"- A1 `r21_a1_vw_limit_gap` ≠ R18 F1 `{F1_NAME}` = `{F1_FORMULA}`",
        "",
        "## 卡片",
        "",
    ]
    for card in cards:
        lines.append(
            f"- `{card['factor_id']}` [{card['family']}] {card['short']}: `{card['formula']}` "
            f"eligible=`{card.get('model_eligible_default')}` blocked=`{card.get('blocked')}`"
        )
    lines += ["", "未构造因子值。E 家族无公告时点 → 可诊断、禁 `PIT_ELIGIBLE_MODEL`。", ""]
    write_md(HYP_MD, "\n".join(lines))
    return payload
