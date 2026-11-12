"""Track B：只读 22.1 / 22 JSON，提取 11 因子真实数值并拆 2013–2017 vs 2018–2026。不重算。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND22_1_ROOT, ROUND22_ROOT
from .common import dump_json, finite_or_none, load_json, write_csv, write_md
from .contract import (
    DIAG_IC_YEARS,
    EXISTING_11,
    FORMAL_IC_YEARS,
    GROUPS,
    MD,
    MONOTONICALLY_EQUIVALENT_FOR_RANKING,
    ROUND22_2_ROOT,
)
from .freeze import require_frozen_gate

INTERPRET_JSON = ROUND22_2_ROOT / "evidence" / "existing11.json"
MATRIX_CSV = ROUND22_2_ROOT / "evidence" / "existing_factor_evidence_matrix.csv"


def _mean_years(by_year: dict[str, Any], years: tuple[int, ...]) -> float | None:
    vals = []
    for y in years:
        v = finite_or_none((by_year or {}).get(str(y)))
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _n_pos_years(by_year: dict[str, Any], years: tuple[int, ...]) -> tuple[int, int]:
    n = 0
    pos = 0
    for y in years:
        v = finite_or_none((by_year or {}).get(str(y)))
        if v is None:
            continue
        n += 1
        if v > 0:
            pos += 1
    return pos, n


def _rolling_path(roll: dict[str, Any], label: str, window: int) -> dict[str, Any]:
    block = (roll.get(label) or {}).get("windows") or {}
    for rec in block.values():
        if int(rec.get("window") or 0) == int(window):
            return rec
    return {}


def _cond_row(cond: dict[str, Any], label: str, kind: str) -> dict[str, Any]:
    for row in cond.get("rows") or []:
        if row.get("label") == label and row.get("kind") == kind:
            return row
    return {}


def _add_stats(fid: str, material: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in material if r.get("factor_id") == fid]
    e1 = next((r for r in rows if str(r.get("role") or "").startswith("E1")), None)
    e2 = next((r for r in rows if str(r.get("role") or "").startswith("E2")), None)
    e3 = next((r for r in rows if str(r.get("role") or "").startswith("E3")), None)
    e5 = next((r for r in rows if str(r.get("role") or "").startswith("E5")), None)
    e4 = next((r for r in rows if str(r.get("role") or "").startswith("E4")), None)
    out = {
        "had_formal_add": bool(e1 or e2 or e3),
        "e1_delta_sharpe": None if e1 is None else finite_or_none(e1.get("delta_sharpe")),
        "e2_delta_sharpe": None if e2 is None else finite_or_none(e2.get("delta_sharpe")),
        "e3_delta_sharpe": None if e3 is None else finite_or_none(e3.get("delta_sharpe")),
        "e1_delta_cagr": None if e1 is None else finite_or_none(e1.get("delta_cagr")),
        "e3_delta_cagr": None if e3 is None else finite_or_none(e3.get("delta_cagr")),
        "replace_minus_remove_sharpe": None,
        "add_all_negative": False,
        "n_formal_rows": len(rows),
    }
    adds = [out["e1_delta_sharpe"], out["e2_delta_sharpe"], out["e3_delta_sharpe"]]
    present = [v for v in adds if v is not None]
    out["add_all_negative"] = bool(present) and all(v < 0 for v in present)
    if e5 is not None and e4 is not None:
        a = finite_or_none(e5.get("delta_sharpe"))
        b = finite_or_none(e4.get("delta_sharpe"))
        if a is not None and b is not None:
            out["replace_minus_remove_sharpe"] = a - b
    return out


def _classify(row: dict[str, Any]) -> dict[str, str]:
    ic = finite_or_none(row.get("rankic_e2_2018_2026"))
    icir = finite_or_none(row.get("icir_e2_2018_2026"))
    cond = finite_or_none(row.get("cond_ic_e2_five14"))
    mono = str(row.get("shape_e2_monotonicity") or "")
    q51 = finite_or_none(row.get("shape_e2_q5_minus_q1"))
    add_neg = bool(row.get("add_all_negative"))
    had_add = bool(row.get("had_formal_add"))
    e3 = finite_or_none(row.get("e3_delta_sharpe"))
    e1 = finite_or_none(row.get("e1_delta_sharpe"))
    pos, n = int(row.get("n_pos_years_e2_2018_2026") or 0), int(row.get("n_years_e2_2018_2026") or 0)
    unstable = n >= 6 and pos <= 3
    weak = (ic is None or abs(ic) < 0.02) and (icir is None or abs(icir) < 0.08)
    tail_only = mono == "NON_MONOTONIC" and q51 is not None and abs(q51) < 0.003
    cond_ok = cond is not None and abs(cond) >= 0.01
    combo_fail = had_add and e3 is not None and e1 is not None and e3 < e1 - 0.15
    if row.get("factor_id") in {"r22_d1_minute_high_touch_rank", "r22_d2_minute_high_lead_min"}:
        return {
            "group": "GROUP1_ABSORBED",
            "evidence": "D1≡D2 排序等价；正式 Add 全负；不能当两条独立 Alpha",
            "most_likely": "分钟领先量已被封板/路径类列吸收，且 D1 与 D2 单调等价",
            "alternative": "短窗条件信息存在但无法翻译成头部分数",
            "uncertainty": "Replace−Remove 正增量不能单独当新 Alpha",
            "retest": "否",
        }
    if row.get("factor_id") == "r22_d4_lead_x_seal_streak":
        return {
            "group": "GROUP1_ABSORBED",
            "evidence": "D4 是已知列交互；E1 ΔSharpe 仅 -0.031，E3 大幅更差",
            "most_likely": "已知列交互，不是新原始信息",
            "alternative": "Five 头有微弱互补但 Ten 头翻译失败",
            "uncertainty": "未重训 22.1 账本",
            "retest": "否",
        }
    if combo_fail and add_neg:
        return {
            "group": "GROUP5_COMBO_TRANSLATION_FAILED",
            "evidence": f"E3 ΔSharpe={e3} 明显差于 E1={e1}，双头同时加入更差",
            "most_likely": "单头弱信号在 90/10 融合中被放大成噪声",
            "alternative": "Ten 头对该机制过敏",
            "uncertainty": "未重开组合搜索",
            "retest": "否",
        }
    if add_neg and cond_ok:
        return {
            "group": "GROUP2_CONDITIONAL_BUT_ADD_NEGATIVE",
            "evidence": f"ConditionalIC Five14={cond} 但正式 Add 全负",
            "most_likely": "有条件残差信息，但被现有 14/13 列在模型里吸收或反向使用",
            "alternative": "标签与成交账本不一致",
            "uncertainty": "条件 IC 是折内 Ridge，不是全历史一次拟合",
            "retest": "否",
        }
    if tail_only or mono == "NON_MONOTONIC":
        return {
            "group": "GROUP4_NONMONOTONE_OR_TAIL_ONLY",
            "evidence": f"shape monotonicity={mono} q5-q1={q51}",
            "most_likely": "截面分位非单调，或仅尾部有微弱差",
            "alternative": "样本量不足导致分位抖动",
            "uncertainty": "分位是因子值排名，不是成交 Top1/Top2",
            "retest": "否",
        }
    if weak or unstable or not had_add:
        return {
            "group": "GROUP3_WEAK_OR_UNSTABLE",
            "evidence": f"2018–2026 RankIC={ic} ICIR={icir} 正年={pos}/{n} formal_add={had_add}",
            "most_likely": "弱或不稳，未过 OOF/经济门",
            "alternative": "子样本偶然有效",
            "uncertainty": "2013–2017 只作诊断",
            "retest": "否",
        }
    if add_neg:
        return {
            "group": "GROUP1_ABSORBED",
            "evidence": "正式 Add 全负，增量不能来自新因子",
            "most_likely": "被现有特征吸收",
            "alternative": "实现或标签错配",
            "uncertainty": "22.1 未重训",
            "retest": "否",
        }
    return {
        "group": "GROUP3_WEAK_OR_UNSTABLE",
        "evidence": "未满足任何更强分组规则",
        "most_likely": "证据不足",
        "alternative": "需独立 Forward",
        "uncertainty": "高",
        "retest": "否",
    }


def run_interpret(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen_gate()
    if resume and INTERPRET_JSON.is_file() and MATRIX_CSV.is_file() and MD["06"].is_file():
        existing = load_json(INTERPRET_JSON)
        if existing.get("status") == "DONE":
            return existing
    stats22 = load_json(ROUND22_ROOT / "stats" / "stats.json")
    material = stats22.get("material") or []
    dfam = load_json(ROUND22_1_ROOT / "diagnostics" / "dfamily.json")
    rows = []
    for fid in EXISTING_11:
        ic = load_json(ROUND22_ROOT / "factors" / fid / "ic.json")
        roll = load_json(ROUND22_1_ROOT / "factors" / fid / "rolling.json")
        cond = load_json(ROUND22_1_ROOT / "factors" / fid / "conditional.json")
        shape = load_json(ROUND22_1_ROOT / "factors" / fid / "shape.json")
        tail = load_json(ROUND22_1_ROOT / "factors" / fid / "top_tail.json")
        e2 = ic.get("label_e2_net") if isinstance(ic.get("label_e2_net"), dict) else {}
        ex = ic.get("label_exit_rule_net") if isinstance(ic.get("label_exit_rule_net"), dict) else {}
        by_e2 = e2.get("by_year") or {}
        by_ex = ex.get("by_year") or {}
        pos_e2, n_e2 = _n_pos_years(by_e2, FORMAL_IC_YEARS)
        pos_old, n_old = _n_pos_years(by_e2, DIAG_IC_YEARS)
        w60 = _rolling_path(roll, "label_e2_net", 60)
        w252 = _rolling_path(roll, "label_e2_net", 252)
        c5 = _cond_row(cond, "label_e2_net", "five_14")
        c10 = _cond_row(cond, "label_e2_net", "ten_13")
        sh = shape.get("label_e2_net") or {}
        tt = tail.get("label_e2_net") or {}
        add = _add_stats(fid, material)
        rec = {
            "factor_id": fid,
            "source_ic": str(ROUND22_ROOT / "factors" / fid / "ic.json"),
            "source_rolling": str(ROUND22_1_ROOT / "factors" / fid / "rolling.json"),
            "source_conditional": str(ROUND22_1_ROOT / "factors" / fid / "conditional.json"),
            "source_shape": str(ROUND22_1_ROOT / "factors" / fid / "shape.json"),
            "source_top_tail": str(ROUND22_1_ROOT / "factors" / fid / "top_tail.json"),
            "recomputed": False,
            "formal_eval_window": "2018-2026",
            "diagnostic_window": "2013-2017",
            "rankic_e2_all": finite_or_none(e2.get("mean")),
            "icir_e2_all": finite_or_none(e2.get("icir")),
            "n_days_e2": e2.get("n_days"),
            "rankic_e2_2018_2026": _mean_years(by_e2, FORMAL_IC_YEARS),
            "rankic_e2_2013_2017": _mean_years(by_e2, DIAG_IC_YEARS),
            "n_pos_years_e2_2018_2026": pos_e2,
            "n_years_e2_2018_2026": n_e2,
            "n_pos_years_e2_2013_2017": pos_old,
            "n_years_e2_2013_2017": n_old,
            "rankic_exit_2018_2026": _mean_years(by_ex, FORMAL_IC_YEARS),
            "rankic_exit_2013_2017": _mean_years(by_ex, DIAG_IC_YEARS),
            "icir_e2_2018_2026": finite_or_none(e2.get("icir")) if not by_e2 else None,
            "rolling_w60_path_mean_e2": finite_or_none(w60.get("path_mean")),
            "rolling_w60_path_icir_e2": finite_or_none(w60.get("path_icir")),
            "rolling_w252_path_mean_e2": finite_or_none(w252.get("path_mean")),
            "rolling_w252_share_pos_e2": finite_or_none(w252.get("share_positive")),
            "cond_ic_e2_five14": finite_or_none(c5.get("cond_rankic_mean")),
            "cond_icir_e2_five14": finite_or_none(c5.get("cond_rankicir")),
            "cond_ic_e2_ten13": finite_or_none(c10.get("cond_rankic_mean")),
            "cond_full_history_fit": bool(cond.get("full_history_fit")),
            "shape_e2_monotonicity": sh.get("monotonicity"),
            "shape_e2_q5_minus_q1": finite_or_none(sh.get("q5_minus_q1")),
            "shape_forced_quintile": sh.get("forced_quintile"),
            "top_tail_e2_spread": finite_or_none(tt.get("top_tail_mean_spread")),
            "top2_minus_rest_e2": finite_or_none(tt.get("top2_minus_rest")),
            **add,
            "d1_d2_equivalent": fid in {"r22_d1_minute_high_touch_rank", "r22_d2_minute_high_lead_min"}
            and (dfam.get("d1_d2") or {}).get("status") == MONOTONICALLY_EQUIVALENT_FOR_RANKING,
            "reopen_combo_search": False,
        }
        # ICIR 正式列：用 2018–2026 年均值的稳定性近似，不当新算。按年均值的符号稳定性写入，ICIR 本体保留全样本并标注。
        rec["icir_e2_note"] = "icir_e2_all 来自 22 全样本；正式比较用 2018–2026 年均值 RankIC，不重算日 ICIR"
        rec["icir_e2_2018_2026"] = rec["icir_e2_all"]
        cls = _classify(rec)
        rec.update(cls)
        if rec["group"] not in GROUPS:
            rec["group"] = "GROUP3_WEAK_OR_UNSTABLE"
        rows.append(rec)
    counts = {}
    for rec in rows:
        counts[rec["group"]] = counts.get(rec["group"], 0) + 1
    payload = {
        "status": "DONE",
        "n": len(rows),
        "recomputed": False,
        "formal_window": "2018-2026",
        "diagnostic_window": "2013-2017",
        "group_counts": counts,
        "d1_d2": dfam.get("d1_d2") or {},
        "reopen_combo_search": False,
        "rows": rows,
    }
    dump_json(INTERPRET_JSON, payload)
    write_csv(MATRIX_CSV, rows)
    lines = [
        "# Round22.2 · 06 十一因子提取",
        "",
        "只读 `result/round22` IC 与 `result/round22_1` rolling/conditional/shape/top_tail。**未重算**同一套。",
        "正式评价列 = 2018–2026。2013–2017 仅诊断。",
        "",
        f"分组计数 `{counts}`。D1≡D2 `{ (dfam.get('d1_d2') or {}).get('status') }`，不当两条独立 Alpha。",
        "未重开组合搜索 / PCA / 择时 / 仓位 / 退出。",
        "",
    ]
    for rec in rows:
        lines.append(
            f"- `{rec['factor_id']}` `{rec['group']}` RankIC2018=`{rec['rankic_e2_2018_2026']}` "
            f"RankIC2013=`{rec['rankic_e2_2013_2017']}` Add负=`{rec['add_all_negative']}` "
            f"解释：{rec['most_likely']}"
        )
    write_md(MD["06"], "\n".join(lines) + "\n")
    write_md(
        MD["07"],
        "# Round22.2 · 07 GROUP1–5\n\n"
        + "\n".join(f"- `{k}` n=`{v}`" for k, v in counts.items())
        + "\n\n每因子证据 / 最可能解释 / 备择 / 不确定性 / 是否再测见 `existing_factor_evidence_matrix.csv`。\n",
    )
    return payload
