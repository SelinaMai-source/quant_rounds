"""三轮收口：测试、因子库、模型登记、追踪表、总报告。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.paths import RESULT_ROOT
from core.research_io import (
    MODEL_REGISTRY_ROOT,
    ZOO_ROOT,
    append_tracker,
    append_zoo_note,
    dump_json,
    load_json,
    write_df,
    write_text,
)
from core.round22_24_tests import run_unit_tests

ROOT = RESULT_ROOT / "round22_24_master"


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def close() -> dict[str, Any]:
    ROOT.mkdir(parents=True, exist_ok=True)
    tests = run_unit_tests()
    dump_json(ROOT / "unit_tests.json", tests)
    d224 = load_json(RESULT_ROOT / "round22_4" / "decision.json")
    d23 = load_json(RESULT_ROOT / "round23" / "decision.json")
    exp24 = load_json(RESULT_ROOT / "round24" / "experiments.json")
    base24 = exp24.get("baseline_2to3_v0") or {}
    nulls = pd.read_csv(RESULT_ROOT / "round22_4" / "null_controls.csv") if (RESULT_ROOT / "round22_4" / "null_controls.csv").is_file() else pd.DataFrame()
    ens = pd.read_csv(RESULT_ROOT / "round23" / "ensemble.csv") if (RESULT_ROOT / "round23" / "ensemble.csv").is_file() else pd.DataFrame()
    rules = pd.read_csv(RESULT_ROOT / "round23" / "rules.csv") if (RESULT_ROOT / "round23" / "rules.csv").is_file() else pd.DataFrame()

    def best_delta(frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty or "delta_sharpe" not in frame.columns:
            return {}
        work = frame.dropna(subset=["delta_sharpe"]).copy()
        if "scenario" in work.columns:
            work = work.loc[~work["scenario"].fillna(False).astype(bool)]
        if work.empty:
            return {}
        row = work.iloc[work["delta_sharpe"].astype(float).abs().idxmax()]
        return {"code": row.get("code"), "delta_sharpe": float(row["delta_sharpe"]), "delta_cagr": None if pd.isna(row.get("delta_cagr")) else float(row.get("delta_cagr"))}

    b224 = best_delta(nulls)
    b23 = best_delta(pd.concat([ens, rules], ignore_index=True) if not ens.empty or not rules.empty else pd.DataFrame())
    blocked = []
    fs = load_json(RESULT_ROOT / "round22_4" / "fs_evidence.json")
    for row in fs.get("decisions") or []:
        if not row.get("train"):
            blocked.append(f"{row.get('factor_id')}: 证据门未过，未训练")
    red = load_json(RESULT_ROOT / "round22_4" / "redundancy.json")
    if not red.get("remove_replace_trained"):
        blocked.append("Remove/Replace: 折内 |ρ| 未到 0.60")
    joint = load_json(RESULT_ROOT / "round23" / "joint.json")
    if joint.get("status") == "NOT_TRIGGERED":
        blocked.append("Round23 四格联合: 模型和规则没有同时过前置")
    diag = load_json(RESULT_ROOT / "round23" / "diagnose.json")
    if diag.get("stacking") == "STOP_STACKING_NOT_COMPLEMENTARY":
        blocked.append("Stacking: Five/Ten 相关过高")
    for key in ("T1", "T3", "T4"):
        item = exp24.get(key) or {}
        status = str(item.get("status", ""))
        if "BLOCK" in status or status == "NOT_TRIGGERED":
            blocked.append(f"Round24 {key}: {item.get('status')} {item.get('reason')}")
    t2 = exp24.get("T2") or {}
    if str(t2.get("status", "")).startswith("BLOCK"):
        blocked.append(f"Round24 T2: {t2.get('status')} {t2.get('reason')}")
    build = load_json(RESULT_ROOT / "round24" / "data" / "build.json")
    if not build and not (RESULT_ROOT / "round24" / "data" / "second_board.parquet").is_file():
        blocked.append("Round24 候选: 二板面板未建成")

    zoo_rows = [
        {"factor_id": "r22_4_fs2_cash_conversion_change", "round": "22.4", "note": "公告事件差，缺失不填 0"},
        {"factor_id": "r22_4_c4_allnan", "round": "22.4", "note": "空值对照，不计 alpha"},
        {"factor_id": "r22_4_c4_shuffle_source", "round": "22.4", "note": "训练折内打乱，不计 alpha"},
    ]
    write_df(ROOT / "zoo_rows.csv", pd.DataFrame(zoo_rows))
    side = ZOO_ROOT / "round22_24_evidence.csv"
    side.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(zoo_rows).to_csv(side, index=False)
    append_zoo_note("# round 22.4-24 sidecar round22_24_evidence.csv decision follows each round report")
    models = [
        {"model_id": "baseline_v5", "round": "locked", "role": "M0", "decision": "KEEP"},
        {"model_id": "R224_CTRL_ALLNAN", "round": "22.4", "role": "null", "decision": d224.get("final_decision")},
        {"model_id": "R224_CTRL_SHUFFLE", "round": "22.4", "role": "null", "decision": d224.get("final_decision")},
        {"model_id": "R23_SEED_012", "round": "23", "role": "robustness", "decision": d23.get("final_decision")},
        {"model_id": "baseline_2to3_v0", "round": "24", "role": "independent", "decision": "NOT_COMPARED_TO_V5"},
    ]
    MODEL_REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(models).to_csv(MODEL_REGISTRY_ROOT / "round22_24_models.csv", index=False)

    def _delta(frame: pd.DataFrame, code: str, col: str) -> Any:
        if frame.empty or code not in set(frame.get("code", [])):
            return None
        series = frame.loc[frame["code"].eq(code), col]
        return None if series.empty or pd.isna(series.iloc[0]) else float(series.iloc[0])

    tracker = [
        "",
        "### Round22.4 至 Round24（只追加）",
        "",
        f"- Round22.4 决策 `{d224.get('final_decision')}`。全 NaN 对照 ΔSharpe={_fmt(_delta(nulls, 'R224_CTRL_ALLNAN', 'delta_sharpe'))}、ΔCAGR={_fmt(_delta(nulls, 'R224_CTRL_ALLNAN', 'delta_cagr'))}；打乱对照 ΔSharpe={_fmt(_delta(nulls, 'R224_CTRL_SHUFFLE', 'delta_sharpe'))}、ΔCAGR={_fmt(_delta(nulls, 'R224_CTRL_SHUFFLE', 'delta_cagr'))}。空值对照不计 alpha。",
        f"- Round23 决策 `{d23.get('final_decision')}`。M0 仍是 v5。非负权重 ΔSharpe={_fmt(_delta(ens, 'R23_M_NN_WEIGHT', 'delta_sharpe'))}、ΔCAGR={_fmt(_delta(ens, 'R23_M_NN_WEIGHT', 'delta_cagr'))}。",
        f"- Round24 是独立二进三线。baseline_2to3_v0 Sharpe={_fmt(base24.get('sharpe'))} CAGR={_fmt(base24.get('cagr'))}。不与一进二 CAGR 比高低。",
        "- CURRENT 未改。经济门未放宽。",
        "",
    ]
    marker = "### Round22.4 至 Round24（只追加）"
    tracker_path = RESULT_ROOT / "一进二_Round7基线_实验组追踪.md"
    if marker not in tracker_path.read_text(encoding="utf-8"):
        append_tracker("\n".join(tracker))

    t2 = exp24.get("T2") or {}
    lines = [
        "# Round22.4–24 总报告",
        "",
        f"Round22.4：{d224.get('final_decision')}。{d224.get('reason')}",
        f"相对 v5，全 NaN 新列 ΔSharpe={_fmt(_delta(nulls, 'R224_CTRL_ALLNAN', 'delta_sharpe'))}，ΔCAGR={_fmt(_delta(nulls, 'R224_CTRL_ALLNAN', 'delta_cagr'))}。训练折内打乱 ΔSharpe={_fmt(_delta(nulls, 'R224_CTRL_SHUFFLE', 'delta_sharpe'))}，ΔCAGR={_fmt(_delta(nulls, 'R224_CTRL_SHUFFLE', 'delta_cagr'))}。打乱对照不计 alpha。",
        f"Round23：{d23.get('final_decision')}。{d23.get('reason')}",
        f"相对 v5，截面 rank 后 90/10 ΔSharpe={_fmt(_delta(ens, 'R23_M_RANK_90_10', 'delta_sharpe'))}，ΔCAGR={_fmt(_delta(ens, 'R23_M_RANK_90_10', 'delta_cagr'))}（与现有融合相同）。内折非负权重 ΔSharpe={_fmt(_delta(ens, 'R23_M_NN_WEIGHT', 'delta_sharpe'))}，ΔCAGR={_fmt(_delta(ens, 'R23_M_NN_WEIGHT', 'delta_cagr'))}。seed 均值 ΔSharpe={_fmt(_delta(ens, 'R23_M_SEED_MEAN', 'delta_sharpe'))}，中位数 ΔSharpe={_fmt(_delta(ens, 'R23_M_SEED_MEDIAN', 'delta_sharpe'))}，ΔCAGR={_fmt(_delta(ens, 'R23_M_SEED_MEDIAN', 'delta_cagr'))}。",
        f"规则相对 v5：广度过滤 ΔSharpe={_fmt(_delta(rules, 'R23_R1_BREADTH', 'delta_sharpe'))}，ΔCAGR={_fmt(_delta(rules, 'R23_R1_BREADTH', 'delta_cagr'))}；流动性 ΔSharpe={_fmt(_delta(rules, 'R23_R2_LIQUIDITY', 'delta_sharpe'))}，ΔCAGR={_fmt(_delta(rules, 'R23_R2_LIQUIDITY', 'delta_cagr'))}。开盘封死是情景审计，ΔSharpe={_fmt(_delta(rules, 'R23_R3_OPEN_SEAL', 'delta_sharpe'))}，不计 alpha。",
        f"Round24：独立基线 baseline_2to3_v0 Sharpe={_fmt(base24.get('sharpe'))}，CAGR={_fmt(base24.get('cagr'))}。T2 重训 Sharpe={_fmt(t2.get('sharpe'))}，CAGR={_fmt(t2.get('cagr'))}。这两条都不与一进二 CAGR 比高低，T2 也没有替换预注册基线。",
        "",
        "## 阻断",
        "",
    ]
    if blocked:
        lines.extend(f"- {item}" for item in blocked)
    else:
        lines.append("- 没有额外阻断。")
    lines.extend(
        [
            "",
            f"单元测试 {tests['n']} 条，失败 {tests['n_failed']}。",
            "",
            "报告：",
            "- `result/round22_4/ROUND22_4_RESEARCH_REPORT.md`",
            "- `result/round23/ROUND23_RESEARCH_REPORT.md`",
            "- `result/round24/ROUND24_SECOND_TO_THIRD_RESEARCH_REPORT.md`",
            "",
        ]
    )
    write_text(ROOT / "MASTER_RESEARCH_REPORT.md", "\n".join(lines))
    payload = {
        "status": "DONE" if tests["n_failed"] == 0 else "TESTS_FAILED",
        "tests_n": tests["n"],
        "tests_failed": tests["n_failed"],
        "round22_4": d224.get("final_decision"),
        "round23": d23.get("final_decision"),
        "round24_baseline_sharpe": base24.get("sharpe"),
        "blocked": blocked,
    }
    dump_json(ROOT / "status.json", payload)
    return payload
