"""盘点 R17 factor_registry.csv + R18 F1/F2/F3。不发明。家族只来自已有表/前缀。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND20_ROOT
from .common import dump_json, load_json, write_md
from .contract import (
    F1_FORMULA,
    F1_NAME,
    F2_FORMULA,
    F2_NAME,
    F3_FORMULA,
    F3_NAME,
    FIVE_FEATURES,
    INVENTORY_MD,
    PIT_TIMESTAMP_UNVERIFIED,
    R17_REGISTRY,
    R18_PIT_JSON,
    TEN_FEATURES,
)

INV_JSON = ROUND20_ROOT / "inventory" / "inventory.json"
REG_CSV = ROUND20_ROOT / "inventory" / "factor_registry.csv"

# Copied from core/round5/factors.py FAMILY_COLUMNS / NO_NEW_DEPENDENCY. Not a new taxonomy.
R5_FAMILY_COLUMNS: dict[str, tuple[str, ...]] = {
    "L_limit_seal": (
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
    "M_money_flow": (
        "h4_main_net_5_to_turnover",
        "h4_large_net_to_turnover",
        "h4_xl_net_concentration",
        "h4_retail_net_to_turnover",
        "fb_close_net_to_turnover",
        "h4_smart_money_price",
        "r5_m_main_minus_retail",
        "r5_m_main_vs_mkt",
    ),
    "I_intraday": (
        "h1_auction_volume_ratio",
        "h1_consistent_buy_ratio",
        "h1_consistent_sell_ratio",
        "h6_volume_weighted_entropy",
        "h6_average_trade_net_inflow_ratio",
        "h6_weighted_close_skew",
        "h6_weighted_close_ratio",
        "r5_i_buy_minus_sell",
    ),
    "R_relative_industry": (
        "ind_first_board_share",
        "ind_seal_rate_excess",
        "h8_bp",
        "h8_market_heat_score",
        "h8_pullback_from_20d_high",
        "h8_ret_3",
        "r5_r_ret_vs_ind",
    ),
    "V_liquidity_size": (
        "g0_turnover",
        "g0_amount_log",
        "g0_float_mv_log",
        "h8_amount_zscore",
        "h8_turnover_rate_zscore",
        "h8_volume_price_sync",
        "r5_v_amt_vs_5_x_turn",
    ),
}
NO_NEW_DEPENDENCY = {"g_prim_ttl_norm", "g_prim_px_vol_corr", "g_prim_prelimit_accel"}
R5_BY_NAME = {name: fam for fam, cols in R5_FAMILY_COLUMNS.items() for name in cols}
R17_FAMILIES = {
    "A_continuation",
    "B_contagion",
    "C_microstructure",
    "D_relative_liquidity",
    "E_event_history",
    "F_xs_strength",
}

# Prefix families already used as column-name groups in R17 inventory. Not new hypotheses.
PREFIX_FAMILIES: tuple[tuple[str, str], ...] = (
    ("r17_a_", "A_continuation"),
    ("r17_b_", "B_contagion"),
    ("r17_c_", "C_microstructure"),
    ("r17_d_", "D_relative_liquidity"),
    ("r17_e_", "E_event_history"),
    ("r17_f_", "F_xs_strength"),
    ("g_prim_", "g_prim_path"),
    ("h_min1_", "h_min1_minute"),
    ("h5_", "h5_limit_value"),
    ("h6_", "h6_trade_dist"),
    ("h1_", "h1_intraday_volume"),
    ("h4_", "h4_money_flow"),
    ("fb_", "fb_first_board"),
    ("g0_", "g0_daily_size"),
    ("g1_", "g1_return"),
    ("g3_", "g3_limit_history"),
    ("ind_", "ind_industry"),
    ("d1_", "d1_open_gap"),
)

R18_ROWS: tuple[dict[str, Any], ...] = (
    {
        "name": F1_NAME,
        "source": "round18_F1",
        "baseline": False,
        "head": "r18_audit",
        "available_at": "D close minutes + limit events",
        "pit": "D_close_minutes_plus_limit_events",
        "old_trial_id": "R18_F1",
        "formula": F1_FORMULA,
        "hypothesis_status": "prior_r18_not_rebranded",
        "promotion_eligible": "pending_r20_gates",
        "note": "公式已差；可进审计库，不得换名重开假设。",
    },
    {
        "name": F2_NAME,
        "source": "round18_F2",
        "baseline": False,
        "head": "r18_audit",
        "available_at": "mkt_equ_mf_new UPDATE_TIME",
        "pit": PIT_TIMESTAMP_UNVERIFIED,
        "old_trial_id": "R18_F2",
        "formula": F2_FORMULA,
        "hypothesis_status": "prior_r18_not_promotion_eligible",
        "promotion_eligible": False,
        "note": "PIT_TIMESTAMP_UNVERIFIED。不得进可晋级模型。",
    },
    {
        "name": F3_NAME,
        "source": "round18_F3",
        "baseline": False,
        "head": "r18_audit",
        "available_at": "as-of-D free_float_shares * D unadj close",
        "pit": "asof_D_shares_times_unadj_close",
        "old_trial_id": "R18_F3",
        "formula": F3_FORMULA,
        "hypothesis_status": "prior_r18_not_rebranded",
        "promotion_eligible": "pending_r20_gates",
        "note": "公式已差；可进审计库，不得换名重开假设。",
    },
)


def _family_of(name: str, source: str = "", *, r18: bool = False) -> tuple[str, str]:
    if r18:
        return "R18_microstructure", "r18_contract"
    if source in R17_FAMILIES:
        return source, "r17_FACTOR_SPECS"
    if name in R5_BY_NAME:
        return R5_BY_NAME[name], "r5_FAMILY_COLUMNS"
    for prefix, fam in PREFIX_FAMILIES:
        if name.startswith(prefix):
            return fam, "name_prefix"
    return "unclassified", "none"


def _r18_pit_override(name: str, default_pit: str) -> str:
    pit = load_json(R18_PIT_JSON)
    rec = (pit.get("factors") or {}).get(name) or {}
    return str(rec.get("pit_status") or default_pit)


def run_inventory(**_: Any) -> dict[str, Any]:
    if not R17_REGISTRY.is_file():
        raise RuntimeError(f"MISSING_ARTIFACT R17 factor_registry: {R17_REGISTRY}")
    r17 = pd.read_csv(R17_REGISTRY)
    expected_cols = {"name", "source", "baseline", "head", "available_at", "pit", "old_trial_id"}
    missing_cols = expected_cols - set(r17.columns)
    if missing_cols:
        raise RuntimeError(f"R17 registry missing columns: {sorted(missing_cols)}")
    rows: list[dict[str, Any]] = []
    for rec in r17.to_dict(orient="records"):
        name = str(rec["name"])
        source = str(rec.get("source") or "")
        fam, fam_src = _family_of(name, source)
        if source == "baseline_five":
            hypo = "locked_baseline"
            eligible: Any = False
            note = "R7 锁定 14/13。不是新假设。"
        elif source == "single_model44_unused":
            hypo = "existing_unused"
            eligible = "pending_r20_gates"
            extra = " R5 NO_NEW_DEPENDENCY。" if name in NO_NEW_DEPENDENCY else ""
            note = "R17 unused 31 之一。本轮不发明，不当作新因子。" + extra
        elif source in R17_FAMILIES:
            hypo = "prior_r17"
            eligible = "pending_r20_gates"
            note = "R17 已规格化新因子。禁止再当新假设换名。"
        else:
            hypo = "prior_r17"
            eligible = "pending_r20_gates"
            note = "来自 R17 factor_registry.csv。不发明。"
        rows.append(
            {
                **{k: rec.get(k) for k in r17.columns},
                "family": fam,
                "family_source": fam_src,
                "formula": "",
                "hypothesis_status": hypo,
                "promotion_eligible": eligible,
                "note": note,
                "in_five": name in FIVE_FEATURES,
                "in_ten": name in TEN_FEATURES,
            }
        )
    r17_names = {str(r["name"]) for r in rows}
    invented = [r["name"] for r in R18_ROWS if r["name"] in r17_names]
    if invented:
        raise RuntimeError(f"R18 names collide with R17 registry: {invented}")
    for rec in R18_ROWS:
        fam, fam_src = _family_of(rec["name"], r18=True)
        pit = _r18_pit_override(rec["name"], rec["pit"])
        rows.append(
            {
                "name": rec["name"],
                "source": rec["source"],
                "baseline": rec["baseline"],
                "head": rec["head"],
                "available_at": rec["available_at"],
                "pit": pit,
                "old_trial_id": rec["old_trial_id"],
                "family": fam,
                "family_source": fam_src,
                "formula": rec["formula"],
                "hypothesis_status": rec["hypothesis_status"],
                "promotion_eligible": rec["promotion_eligible"],
                "note": rec["note"],
                "in_five": False,
                "in_ten": False,
            }
        )
    frame = pd.DataFrame(rows)
    if int(frame["name"].duplicated().sum()) != 0:
        dups = frame.loc[frame["name"].duplicated(), "name"].tolist()
        raise RuntimeError(f"duplicate factor names: {dups}")
    REG_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REG_CSV, index=False)

    n_r17 = int(len(r17))
    n_r18 = len(R18_ROWS)
    n_total = int(len(frame))
    n_baseline = int((frame["source"] == "baseline_five").sum())
    n_unused = int((frame["source"] == "single_model44_unused").sum())
    n_r17_new = int(frame["source"].isin(R17_FAMILIES).sum())
    n_f2 = int((frame["name"] == F2_NAME).sum())
    family_counts = (
        frame.groupby(["family", "family_source"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["family_source", "family"])
        .to_dict(orient="records")
    )
    hypo_counts = frame.groupby("hypothesis_status").size().to_dict()
    payload = {
        "status": "DONE",
        "n_r17_registry": n_r17,
        "n_r18_f123": n_r18,
        "n_total": n_total,
        "n_baseline_five": n_baseline,
        "n_ten": len(TEN_FEATURES),
        "n_unused44": n_unused,
        "n_r17_new": n_r17_new,
        "n_expected": 75 + 3,
        "counts_match_plan": bool(n_r17 == 75 and n_r18 == 3 and n_total == 78),
        "invented": False,
        "f2_promotion_eligible": False,
        "f2_pit": PIT_TIMESTAMP_UNVERIFIED,
        "f1_f3_rebranded": False,
        "family_counts": family_counts,
        "hypothesis_counts": {str(k): int(v) for k, v in hypo_counts.items()},
        "registry_csv": str(REG_CSV),
        "r17_registry_src": str(R17_REGISTRY),
        "note": "只合并已有清单。F1/F3 不换名。F2 不得晋级。",
    }
    dump_json(INV_JSON, payload)

    fam_lines = ["| family | source | n |", "|---|---|---|"]
    for rec in family_counts:
        fam_lines.append(f"| `{rec['family']}` | `{rec['family_source']}` | {rec['n']} |")
    hypo_lines = ["| hypothesis_status | n |", "|---|---|"]
    for k, v in sorted(hypo_counts.items()):
        hypo_lines.append(f"| `{k}` | {int(v)} |")
    lines = [
        "# 01 因子盘点",
        "",
        "不发明清单外因子。F1/F3 不换名重开假设。F2 不得进可晋级模型。",
        "",
        "## 计数",
        "",
        f"- R17 `factor_registry.csv` 行=`{n_r17}`（计划 75 = 基线 14 + unused 31 + R17 新 30）",
        f"- 基线 five=`{n_baseline}` ten=`{len(TEN_FEATURES)}` unused44=`{n_unused}` r17_new=`{n_r17_new}`",
        f"- R18 F1/F2/F3=`{n_r18}`",
        f"- 合计=`{n_total}` counts_match_plan=`{payload['counts_match_plan']}` invented=`False`",
        "",
        "## R18",
        "",
        f"- F1 `{F1_NAME}` `{F1_FORMULA}` → 审计库，**不是**新假设。",
        f"- F2 `{F2_NAME}` `{F2_FORMULA}` pit=`{PIT_TIMESTAMP_UNVERIFIED}` **promotion_eligible=False**。",
        f"- F3 `{F3_NAME}` `{F3_FORMULA}` → 审计库，**不是**新假设。",
        "",
        "## 家族（只引用已有表）",
        "",
        "- `r5_FAMILY_COLUMNS`：Round5 L/M/I/R/V。",
        "- `r17_FACTOR_SPECS`：A_continuation / B_contagion / C_microstructure / D_relative_liquidity / E_event_history / F_xs_strength。",
        "- `r18_contract`：F1/F2/F3 → `R18_microstructure`。",
        "- `name_prefix`：R5 表未收录时，用因子名已有前缀（g_prim_/h1_/h4_/fb_/…），不当作新经济故事。",
        "",
        *fam_lines,
        "",
        "## 假设状态",
        "",
        *hypo_lines,
        "",
        f"registry：`{REG_CSV}`",
        "",
    ]
    write_md(INVENTORY_MD, "\n".join(lines))
    return payload
