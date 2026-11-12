"""Track E：D 族增量归因 + 公式 provenance。不借机改公式。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND22_1_ROOT, ROUND22_ROOT
from .common import dump_json, finite_or_none, load_json, write_csv, write_md
from .contract import (
    COMPOSITE_OF_KNOWN_FACTORS,
    E0_CODE,
    HYPOTHESIS_NOT_CONFIRMED,
    MD,
    MONOTONICALLY_EQUIVALENT_FOR_RANKING,
    POST_FREEZE_DEVIATION,
    PRE_FREEZE_REVISION,
    V5_ANCHOR,
    spec_by_id,
)
from .freeze import require_frozen_gate
from .ic_audit import run_ic_audit
from .ledger import run_ledger

DF_JSON = ROUND22_1_ROOT / "diagnostics" / "dfamily.json"


def _spearman_daily(a: pd.Series, b: pd.Series, dates: pd.Series) -> dict[str, Any]:
    rows = []
    for day, idx in dates.groupby(dates):
        xa = pd.to_numeric(a.loc[idx.index], errors="coerce")
        xb = pd.to_numeric(b.loc[idx.index], errors="coerce")
        mask = xa.notna() & xb.notna()
        if int(mask.sum()) < 5:
            continue
        rho = xa[mask].corr(xb[mask], method="spearman")
        if rho is None or not np.isfinite(rho):
            continue
        # 排序一致：rank 完全相同
        ra = xa[mask].rank(method="first")
        rb = xb[mask].rank(method="first")
        same = float((ra.to_numpy() == rb.to_numpy()).mean())
        mono = float(((ra.to_numpy() - rb.to_numpy()) == 0).mean())
        # 单调：一个是另一个的严格单调变换（秩相关 = ±1）
        rows.append({"trade_date": str(pd.Timestamp(day).date()), "rho": float(rho), "rank_agree": same, "n": int(mask.sum())})
    if not rows:
        return {"status": "NOT_IDENTIFIABLE", "n_days": 0}
    frame = pd.DataFrame(rows)
    mean_rho = float(frame["rho"].mean())
    mean_agree = float(frame["rank_agree"].mean())
    share_abs1 = float((frame["rho"].abs() >= 0.999).mean())
    equiv = bool(share_abs1 >= 0.98)
    return {
        "status": MONOTONICALLY_EQUIVALENT_FOR_RANKING if equiv else "NOT_EQUIVALENT",
        "n_days": int(len(frame)),
        "mean_spearman": mean_rho,
        "mean_rank_agree": mean_agree,
        "share_abs_rho_ge_0.999": share_abs1,
        "equivalent_for_ranking": equiv,
        "equivalent_means_same_model": False,
        "equivalent_means_two_alphas": False,
        "note": "排序等价 ≠ 模型预测相同，也 ≠ 两条独立 Alpha",
    }


def _q_gaps(ledger: dict[str, Any], stats22: dict[str, Any]) -> list[dict[str, Any]]:
    sleeve = {r["code"]: r for r in (load_json(ROUND22_ROOT / "backtest" / "sleeve.json").get("rows") or [])}
    material = {r["code"]: r for r in (stats22.get("material") or [])}
    pairs = [
        ("Q(E4)-Q(E0)", "R22_E4_r22_d2_minute_high_lead_min", E0_CODE),
        ("Q(E5)-Q(E4)", "R22_E5_r22_d2_minute_high_lead_min", "R22_E4_r22_d2_minute_high_lead_min"),
        ("Q(E5)-Q(E0)", "R22_E5_r22_d2_minute_high_lead_min", E0_CODE),
        ("Q(E1)-Q(E0)", "R22_E1_r22_d2_minute_high_lead_min", E0_CODE),
        ("Q(E4_d1)-Q(E0)", "R22_E4_r22_d1_minute_high_touch_rank", E0_CODE),
        ("Q(E5_d1)-Q(E4_d1)", "R22_E5_r22_d1_minute_high_touch_rank", "R22_E4_r22_d1_minute_high_touch_rank"),
        ("Q(E4_d3)-Q(E0)", "R22_E4_r22_d3_lead_resid_past_only", E0_CODE),
        ("Q(E5_d3)-Q(E4_d3)", "R22_E5_r22_d3_lead_resid_past_only", "R22_E4_r22_d3_lead_resid_past_only"),
        ("Q(E1_d4)-Q(E0)", "R22_E1_r22_d4_lead_x_seal_streak", E0_CODE),
    ]
    out = []
    for name, a, b in pairs:
        ra = sleeve.get(a) or {}
        rb = sleeve.get(b) or {}
        if b == E0_CODE and not rb:
            rb = {"cagr": V5_ANCHOR["cagr"], "sharpe": V5_ANCHOR["sharpe"], "max_drawdown": V5_ANCHOR["max_drawdown"], "nav_end": V5_ANCHOR["nav_end"]}
        ds = None if ra.get("sharpe") is None or rb.get("sharpe") is None else float(ra["sharpe"]) - float(rb["sharpe"])
        dc = None if ra.get("cagr") is None or rb.get("cagr") is None else float(ra["cagr"]) - float(rb["cagr"])
        dn = None if ra.get("nav_end") is None or rb.get("nav_end") is None else float(ra["nav_end"]) - float(rb["nav_end"])
        out.append(
            {
                "contrast": name,
                "code_a": a,
                "code_b": b,
                "delta_sharpe": ds,
                "delta_cagr": dc,
                "delta_nav": dn,
                "source": "round22 sleeve.json (replay, no retrain)",
            }
        )
    return out


def _trade_attr(code_a: str, code_b: str) -> dict[str, Any]:
    pa = ROUND22_ROOT / "candidates" / code_a / "trades.parquet"
    pb = ROUND22_ROOT / "candidates" / code_b / "trades.parquet"
    if not pa.is_file() or not pb.is_file():
        return {"status": "MISSING", "code_a": code_a, "code_b": code_b}
    a = pd.read_parquet(pa)
    b = pd.read_parquet(pb)
    ka = set(zip(pd.to_datetime(a["signal_date"]).astype(str), a["instrument_id"].astype(str)))
    kb = set(zip(pd.to_datetime(b["signal_date"]).astype(str), b["instrument_id"].astype(str)))
    add = ka - kb
    rem = kb - ka
    def _cash(frame: pd.DataFrame, keys: set) -> float:
        if not keys:
            return 0.0
        idx = list(zip(pd.to_datetime(frame["signal_date"]).astype(str), frame["instrument_id"].astype(str)))
        mask = [k in keys for k in idx]
        sell = pd.to_numeric(frame.loc[mask, "sell_proceeds"], errors="coerce").fillna(0)
        div = pd.to_numeric(frame.loc[mask, "cash_dividend"], errors="coerce").fillna(0)
        buy = pd.to_numeric(frame.loc[mask, "buy_cost"], errors="coerce").fillna(0)
        return float((sell + div - buy).sum())
    years = {}
    for year in (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026):
        ya = {(d, i) for d, i in ka if d.startswith(str(year))}
        yb = {(d, i) for d, i in kb if d.startswith(str(year))}
        years[str(year)] = {"add": len(ya - yb), "rem": len(yb - ya), "sym": len(ya ^ yb)}
    return {
        "status": "COMPUTED",
        "code_a": code_a,
        "code_b": code_b,
        "n_add": len(add),
        "n_rem": len(rem),
        "n_sym": len(ka ^ kb),
        "cash_add": _cash(a, add),
        "cash_rem": _cash(b, rem),
        "cash_net": _cash(a, add) - _cash(b, rem),
        "by_year": years,
        "focus_2022": years.get("2022"),
        "ex2024": {k: years[k] for k in years if k != "2024"},
        "ex2025": {k: years[k] for k in years if k != "2025"},
    }


def _provenance() -> list[dict[str, Any]]:
    rows = []
    cards = {
        "r22_a2_own_touch_limit_excess_20": {
            "prompt": "mean(I_high_touch_limit)_{D-20..D-1} - mean(mkt_touch_breadth)_{D-20..D-1}",
            "code": "own_t.sub(touch_br.shift(1).rolling(20).mean())",
            "verdict": "MATCH_FROZEN",
        },
        "r22_a4_touch5_over_mkt_breadth": {
            "prompt": "mean(I_touch)_{D-5..D-1} / (eps + mean(mkt_touch_breadth)_{D-20..D-1})  [用户原始口径 touch breadth]",
            "frozen": spec_by_id("r22_a4_touch5_over_mkt_breadth")["formula"],
            "code": "own_t5.div(close_br.shift(1).rolling(20).mean() + 1e-6)  # 分母是 close_br 不是 touch_br",
            "verdict": PRE_FREEZE_REVISION,
            "note": "冻结契约已写 mkt_close_breadth；实现用 close_br。原始 prompt 的 touch 分母在冻结前被改成收盘广度。不是 POST-FREEZE 改公式。",
        },
        "r22_b1_dyn_peer_ret_d1": {
            "prompt": "mean(ret_{p,D-1}) top_k=5 positive corr overlap>=40 exclude self",
            "code": "dynamic peer D-1 return mean; peer from candidate history codes",
            "verdict": "MATCH_FROZEN",
        },
        "r22_c1_xl_net_imbalance": {
            "prompt": "(BUY_VALUE_XL - SELL_VALUE_XL) / amount；曾被理解成条件成交笔数",
            "code": "未构造。Round22 标 BLOCKED_SEMANTICS。实际语义是 XL 净流入而非条件成交笔数。",
            "verdict": PRE_FREEZE_REVISION,
            "note": "本轮只审计，不换高收益公式。",
        },
        "r22_d4_lead_x_seal_streak": {
            "prompt": "r22_d2 * fb_close_seal_streak_norm",
            "code": "r22_d2 * fb_close_seal_streak_norm",
            "verdict": COMPOSITE_OF_KNOWN_FACTORS,
            "note": "已知列交互，不是新原始信息。",
        },
    }
    for fid, rec in cards.items():
        spec = spec_by_id(fid)
        rows.append(
            {
                "factor_id": fid,
                "original_prompt": rec.get("prompt"),
                "frozen_formula": spec.get("formula"),
                "actual_code": rec.get("code"),
                "verdict": rec.get("verdict"),
                "note": rec.get("note"),
                "formula_changed_this_round": False,
            }
        )
    return rows


def run_dfamily(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen_gate()
    if resume and DF_JSON.is_file() and MD["06"].is_file():
        existing = load_json(DF_JSON)
        if existing.get("status") in {"DONE", "COMPLETE"}:
            return existing
    ic = run_ic_audit(resume=True)
    ledger = run_ledger(resume=True)
    panel = pd.read_parquet(ROUND22_ROOT / "data" / "factor_panel.parquet")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    d1, d2 = "r22_d1_minute_high_touch_rank", "r22_d2_minute_high_lead_min"
    equiv = {"status": "MISSING_COLUMNS"}
    if d1 in panel.columns and d2 in panel.columns:
        equiv = _spearman_daily(panel[d1], panel[d2], panel["trade_date"])
        # 全样本 IC 是否相同
        ic_rows = {r.get("factor_id"): r for r in (ic.get("rows") or [])}
        e2_d1 = ((ic_rows.get(d1) or {}).get("label_e2_net") or {}).get("mean")
        e2_d2 = ((ic_rows.get(d2) or {}).get("label_e2_net") or {}).get("mean")
        equiv["ic_e2_d1"] = e2_d1
        equiv["ic_e2_d2"] = e2_d2
        equiv["ic_e2_identical"] = e2_d1 is not None and e2_d2 is not None and abs(float(e2_d1) - float(e2_d2)) < 1e-15
    stats22 = load_json(ROUND22_ROOT / "stats" / "stats.json")
    gaps = _q_gaps(ledger, stats22)
    attrs = [
        _trade_attr("R22_E5_r22_d2_minute_high_lead_min", "R22_E4_r22_d2_minute_high_lead_min"),
        _trade_attr("R22_E4_r22_d2_minute_high_lead_min", E0_CODE),
        _trade_attr("R22_E5_r22_d2_minute_high_lead_min", E0_CODE),
        _trade_attr("R22_E1_r22_d2_minute_high_lead_min", E0_CODE),
    ]
    # hash 核验 E0/E4/E5/E1
    hashes = {}
    for code in (E0_CODE, "R22_E4_r22_d2_minute_high_lead_min", "R22_E5_r22_d2_minute_high_lead_min", "R22_E1_r22_d2_minute_high_lead_min"):
        hashes[code] = {
            "feature": load_json(ROUND22_ROOT / "candidates" / code / "feature_list.json"),
            "model": load_json(ROUND22_ROOT / "candidates" / code / "model_hash.json"),
            "pred": load_json(ROUND22_ROOT / "candidates" / code / "prediction_hash.json"),
        }
    e0_five = (hashes.get(E0_CODE) or {}).get("feature", {}).get("five") or []
    e4_five = (hashes.get("R22_E4_r22_d2_minute_high_lead_min") or {}).get("feature", {}).get("five") or []
    e5_five = (hashes.get("R22_E5_r22_d2_minute_high_lead_min") or {}).get("feature", {}).get("five") or []
    e1_five = (hashes.get("R22_E1_r22_d2_minute_high_lead_min") or {}).get("feature", {}).get("five") or []
    e0_ten = (hashes.get(E0_CODE) or {}).get("feature", {}).get("ten") or []
    true_delete = "fb_close_seal_streak_norm" in e0_five and "fb_close_seal_streak_norm" not in e4_five
    true_add_d2 = d2 in e5_five and "fb_close_seal_streak_norm" not in e5_five
    ten_unchanged_e4 = list(e0_ten) == list((hashes.get("R22_E4_r22_d2_minute_high_lead_min") or {}).get("feature", {}).get("ten") or [])
    rmr = next((r for r in (stats22.get("replace_minus_remove") or []) if r.get("factor_id") == d2), {})
    replace_minus_remove = finite_or_none(rmr.get("delta_sharpe_replace_minus_remove"))
    new_alpha_alone = False
    explanation = {
        "status": HYPOTHESIS_NOT_CONFIRMED,
        "note": "解释诊断不是方法搜索。D2 Replace 相对 Remove 的 +0.140 Sharpe 说明新列不是完全无用，但对 v5 仍为负，不能单独当新 Alpha。",
        "group_lofo": HYPOTHESIS_NOT_CONFIRMED,
        "gain": HYPOTHESIS_NOT_CONFIRMED,
        "pred_rank_change": "见换笔环境",
    }
    prov = _provenance()
    write_csv(ROUND22_1_ROOT / "diagnostics" / "d_family_contrasts.csv", gaps)
    write_csv(ROUND22_1_ROOT / "diagnostics" / "formula_provenance.csv", prov)
    dump_json(ROUND22_1_ROOT / "diagnostics" / "formula_provenance.json", {"rows": prov})
    payload = {
        "status": "DONE",
        "d1_d2": equiv,
        "hash_check": {
            "true_delete_seal": true_delete,
            "true_add_d2_on_e5": true_add_d2,
            "ten_unchanged_e4": ten_unchanged_e4,
            "e1_has_d2": d2 in e1_five,
            "e0_five_n": len(e0_five),
            "e4_five_n": len(e4_five),
            "e5_five_n": len(e5_five),
        },
        "contrasts": gaps,
        "replace_minus_remove_d2": replace_minus_remove,
        "replace_minus_remove_is_new_alpha_alone": new_alpha_alone,
        "trade_attr": attrs,
        "explanation": explanation,
        "provenance": prov,
        "d4_label": COMPOSITE_OF_KNOWN_FACTORS,
    }
    dump_json(DF_JSON, payload)
    write_md(
        MD["06"],
        "\n".join(
            [
                "# Round22.1 · 06 D 族归因",
                "",
                f"D1 vs D2 日 Spearman 均值=`{(equiv or {}).get('mean_spearman')}` 排序一致=`{(equiv or {}).get('mean_rank_agree')}` 状态=`{(equiv or {}).get('status')}`。",
                "排序等价 ≠ 模型预测相同，也 ≠ 两条独立 Alpha。",
                f"真删 `fb_close_seal_streak_norm`=`{true_delete}`。真加 D2=`{true_add_d2}`。Ten 在 E4 不变=`{ten_unchanged_e4}`。",
                f"D2 Replace−Remove Sharpe=`{replace_minus_remove}`。不得单独当新 Alpha。",
                f"解释诊断 `{HYPOTHESIS_NOT_CONFIRMED}`。",
                "",
            ]
        ),
    )
    write_md(
        MD["07"],
        "# Round22.1 · 07 公式 provenance\n\n只标 PRE-FREEZE_REVISION / POST-FREEZE_DEVIATION / MATCH_FROZEN / COMPOSITE_OF_KNOWN_FACTORS。本轮不改公式。\n\n"
        + "\n".join(f"- `{r['factor_id']}` `{r['verdict']}`：{r.get('note') or r.get('actual_code')}" for r in prov)
        + "\n",
    )
    write_md(
        MD["17"],
        "# Round22.1 · 17 Replace−Remove\n\n"
        + "\n".join(f"- {r['contrast']}: ΔSharpe=`{r.get('delta_sharpe')}` ΔCAGR=`{r.get('delta_cagr')}` ΔNAV=`{r.get('delta_nav')}`" for r in gaps)
        + f"\n\nD2 Replace−Remove=`{replace_minus_remove}`。不是新 Alpha 单独成立的证据。\n",
    )
    return payload
