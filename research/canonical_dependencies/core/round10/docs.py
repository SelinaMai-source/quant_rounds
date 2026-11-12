"""Round10 追踪表锁 v4。不改写 R1–R9 实验数字。"""

from __future__ import annotations

import json
from typing import Any

from ..paths import assert_project_result_path
from .contract import (
    ANCHOR_DIR,
    B03_CELL_ROWS,
    BOOTSTRAP_P_DELTA_CAGR,
    PARENT_CODE,
    TARGET_AVG_EXP,
    TARGET_CAGR,
    TARGET_LOYO_MED,
    TARGET_LOYO_WORST,
    TARGET_MAXDD,
    TARGET_N_TRADES,
    TARGET_SHARPE,
    TARGET_TOTAL_RETURN,
    TRACKER_PATH,
)


def _pct(x, digits=2, signed=True) -> str:
    if x is None:
        return "—"
    v = float(x)
    txt = f"{v:.{digits}%}"
    if signed and v > 0:
        txt = "+" + txt
    return txt.replace("-", "−")


def _cell_table() -> str:
    grid = {f"{r['agree']}_{r['sep']}": r["nav"] for r in B03_CELL_ROWS}

    def _n(agree: str, sep: str) -> str:
        return f"{grid[f'{agree}_{sep}']*100:.0f}"

    return (
        "| agreement \\ sep | LOW（p≤20） | MID（20–50） | HIGH（>50） |\n"
        "| --- | ---: | ---: | ---: |\n"
        f"| A | {_n('A', 'LOW')} | {_n('A', 'MID')} | {_n('A', 'HIGH')} |\n"
        f"| B | {_n('B', 'LOW')} | {_n('B', 'MID')} | {_n('B', 'HIGH')} |\n"
        f"| C | {_n('C', 'LOW')} | {_n('C', 'MID')} | {_n('C', 'HIGH')} |"
    )


def lock_v4_tracker() -> dict[str, Any]:
    """写入 baseline_v4=R9_B03 为 CURRENT DEVELOPMENT BASELINE。不改 R1–R9 实验数字。"""
    text = TRACKER_PATH.read_text(encoding="utf-8")
    changed = []

    old_lineage = (
        "baseline_v1 = r7_v1 = R7_FS01122_DUAL90_10_CLEAN\n"
        "baseline_v2 = R4_E_top1_minus_median\n"
        "baseline_v3 = R8_E02\n"
        "```"
    )
    new_lineage = (
        "baseline_v1 = r7_v1 = R7_FS01122_DUAL90_10_CLEAN\n"
        "baseline_v2 = R4_E_top1_minus_median\n"
        "baseline_v3 = R8_E02\n"
        "baseline_v4 = R9_B03\n"
        "```"
    )
    if "baseline_v4 = R9_B03" not in text and old_lineage in text:
        text = text.replace(old_lineage, new_lineage, 1)
        changed.append("lineage")

    old_lock = (
        "| 当前锁定（开发父策略） | **`baseline_v3` = `R8_E02`**"
        "（historical；v1=`R7_FS01122_DUAL90_10_CLEAN` / v2=`R4_E_top1_minus_median` 作历史行保留）"
    )
    new_lock = (
        "| 当前锁定（开发父策略） | **`baseline_v4` = `R9_B03`**"
        "（CURRENT DEVELOPMENT BASELINE；historical，非 final OOS / production；"
        "v1/v2/v3 作历史行保留；此后主比较 Δ 只对 v4）"
    )
    if old_lock in text:
        text = text.replace(old_lock, new_lock, 1)
        changed.append("current_lock")

    old_v3_intro = (
        "**v3 = `R8_E02`**：v2 底上双头分歧只乘 Top2（parent=`baseline_v2`）。"
        "用户锁定为当前开发父策略；不是新 Alpha。"
    )
    new_v3_intro = (
        "**v3 = `R8_E02`**：v2 底上双头分歧只乘 Top2（parent=`baseline_v2`）。"
        "历史开发父策略；不是新 Alpha。\n"
        "**v4 = `R9_B03`**：v3 底上 Top2 conservative replacement"
        "（B03_CELL：A 0/5/10、B 0/0/5、C 0/0/0；replacement，不再乘 E02）。"
        "用户锁定为 **CURRENT DEVELOPMENT BASELINE**；historical，非 final OOS。不是新 Alpha。"
        "此后主比较 Δ 只对 v4。"
    )
    if "**v4 = `R9_B03`**" not in text and old_v3_intro in text:
        text = text.replace(old_v3_intro, new_v3_intro, 1)
        changed.append("v4_intro")

    old_v3_note = "当前开发baseline策略（用户锁定）；historical，非 final OOS"
    new_v3_note = "历史开发父策略；historical，非 final OOS"
    if old_v3_note in text:
        text = text.replace(old_v3_note, new_v3_note, 1)
        changed.append("v3_remark")

    v4_row = (
        "| `baseline_v4`（`R9_B03`）                                                      | "
        "v3 底 + Top2 conservative replacement（B03_CELL A 0/5/10、B 0/0/5、C 0/0/0；不再乘 E02） |  "
        "1.843 | +53.74% | +3863.21% | −24.75% |  −7.47%（2022） | 1149 | "
        "CURRENT DEVELOPMENT BASELINE；historical，非 final OOS |"
    )
    marker_v3_row = "| `baseline_v3`（`R8_E02`）"
    if "`baseline_v4`（`R9_B03`）" not in text and marker_v3_row in text:
        idx = text.find(marker_v3_row)
        eol = text.find("\n", idx)
        if eol > 0:
            text = text[:eol + 1] + v4_row + "\n" + text[eol + 1:]
            changed.append("v4_row")

    v4_year = (
        "| `baseline_v4` | +37.36% | +10.27% | +35.10% | +58.82% |  −7.47% | +50.77% | +233.54% | +144.99% | +6.97% |"
    )
    marker_v3_year = "| `baseline_v3` |"
    # 当前最优分年表里的 v3 行（Round 9 分年表也有 R9 的 v3 对照，不要误改那里）
    start_opt = text.find("## 当前最优 baseline")
    end_r1 = text.find("## Round 1")
    if start_opt >= 0 and end_r1 > start_opt and "| `baseline_v4` |" not in text[start_opt:end_r1]:
        chunk = text[start_opt:end_r1]
        pos = chunk.rfind(marker_v3_year)
        if pos >= 0:
            abs_pos = start_opt + pos
            eol = text.find("\n", abs_pos)
            text = text[:eol + 1] + v4_year + "\n" + text[eol + 1:]
            changed.append("v4_yearly")

    decision = f"""
### Round9 Baseline Decision

用户确认升锁：**`baseline_v4` = `R9_B03`** 现为 **CURRENT DEVELOPMENT BASELINE**。
状态：historical development；**不是** final OOS / production。

- Parent = `R8_E02`；Top1 **LOCK** = v2 的 B01a × top1−median 底 20%×0.5
- HIGH=A / MEDIUM=B / LOW=C
- Top2 **replacement** 0/5/10（**不再乘** E02 的 1.00/0.75/0.50）
- 九格 B03_CELL（LOW/MID/HIGH = p≤20 / 20–50 / >50），来自 `core/round9/contract.py`：

{_cell_table()}

- 账本（`{ANCHOR_DIR}/summary.json`）：CAGR **53.7375%**、Sharpe **1.8434**、MaxDD **−24.7510%**、LOYO med/worst **47.3360%/33.5472%**、均暴露 **13.7110%**、**1149** 笔、累计 **+3863.21%**
- 精确值：CAGR=`{TARGET_CAGR}` Sharpe=`{TARGET_SHARPE}` MaxDD=`{TARGET_MAXDD}` LOYO=`{TARGET_LOYO_MED}`/`{TARGET_LOYO_WORST}` trades=`{TARGET_N_TRADES}` exp=`{TARGET_AVG_EXP}` total_return=`{TARGET_TOTAL_RETURN}`
- 限制：相对 v3 的 paired bootstrap P(ΔCAGR>0)≈{BOOTSTRAP_P_DELTA_CAGR}。点估计升版**不代表**统计证明。
- 此后主比较 Δ **只对 v4**。不自动升 `baseline_v5`。
"""
    if "### Round9 Baseline Decision" not in text:
        if text.endswith("\n"):
            text = text + decision
        else:
            text = text + "\n" + decision
        changed.append("baseline_decision")

    assert_project_result_path(TRACKER_PATH)
    TRACKER_PATH.write_text(text, encoding="utf-8")
    return {
        "path": str(TRACKER_PATH),
        "changed": changed,
        "parent": PARENT_CODE,
        "already": changed == [],
    }


def _pct(x, digits=2, signed=True) -> str:
    if x is None:
        return "—"
    v = float(x)
    txt = f"{v:.{digits}%}"
    if signed and v > 0:
        txt = "+" + txt
    return txt.replace("-", "−")


def _sh(x) -> str:
    return "—" if x is None else f"{float(x):.3f}"


def v4_anchor_metrics() -> dict[str, Any]:
    from .contract import (
        TARGET_CAGR,
        TARGET_LOYO_MED,
        TARGET_LOYO_WORST,
        TARGET_MAXDD,
        TARGET_SHARPE,
    )
    from .runner import load_summary
    from .contract import R10_00

    s = load_summary(R10_00) or {}
    return {
        "cagr": float(s.get("cagr") or TARGET_CAGR),
        "sharpe": float(s.get("sharpe") or TARGET_SHARPE),
        "max_drawdown": float(s.get("max_drawdown") or TARGET_MAXDD),
        "loyo_median": float(s["loyo_median"]) if s.get("loyo_median") is not None else TARGET_LOYO_MED,
        "loyo_worst": float(s["loyo_worst"]) if s.get("loyo_worst") is not None else TARGET_LOYO_WORST,
        "completed_trades": s.get("completed_trades"),
        "cagr_ex2024": s.get("cagr_ex2024"),
        "avg_gross_exposure": s.get("avg_gross_exposure"),
    }


def verdict(code: str, s: dict[str, Any] | None = None) -> str:
    from .registry import SPECS
    from .runner import load_summary

    spec = SPECS.get(code)
    s = s or load_summary(code)
    if spec and spec.skip and (not s or s.get("cagr") is None):
        return spec.skip_reason or "SKIP"
    if not s or s.get("cagr") is None:
        return (s or {}).get("status") or "未跑"
    if spec and spec.post_hoc:
        return "POST_HOC_EXPLORATORY"
    if spec and spec.diagnostic:
        return "DIAGNOSTIC"
    if code.endswith("baseline_v4") or code == "R10_00_baseline_v4":
        return "CONTROL"
    v4 = v4_anchor_metrics()
    d_cg = float(s["cagr"]) - v4["cagr"]
    d_sh = float(s["sharpe"]) - v4["sharpe"]
    if d_cg > 0 and d_sh > 0:
        return "AHEAD_V4"
    if d_cg > 0 or d_sh > 0:
        return "MIXED"
    return "NO_INCREMENT"


def metrics_table(codes: list[str]) -> str:
    from .registry import SPECS
    from .runner import load_summary
    from .contract import PARENT_CODE

    v4 = v4_anchor_metrics()
    lines = [
        "| ID | Parent | Change | CAGR | ΔCAGR | Sharpe | ΔSharpe | MaxDD | ΔMDD | LOYO Med | LOYO Worst | Avg Exp | Top2 PnL | Verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for code in codes:
        spec = SPECS.get(code)
        s = load_summary(code)
        title = spec.title_zh if spec else code
        parent = (spec.parent if spec else PARENT_CODE) or PARENT_CODE
        if not s or s.get("cagr") is None:
            lines.append(
                f"| `{code}` | `{parent}` | {title} | — | — | — | — | — | — | — | — | — | — | {verdict(code, s)} |"
            )
            continue
        d_sh = (s["sharpe"] - v4["sharpe"]) if s.get("sharpe") is not None else None
        d_cg = (s["cagr"] - v4["cagr"]) if s.get("cagr") is not None else None
        d_dd = (s["max_drawdown"] - v4["max_drawdown"]) if s.get("max_drawdown") is not None else None
        exp = s.get("avg_gross_exposure")
        top2 = s.get("top2_pnl")
        top2_txt = f"{float(top2):,.0f}" if top2 is not None else "—"
        exp_txt = _pct(exp, signed=False) if exp is not None else "—"
        lines.append(
            f"| `{code}` | `{parent}` | {title} | {_pct(s.get('cagr'))} | {_pct(d_cg)} | "
            f"{_sh(s.get('sharpe'))} | {_sh(d_sh)} | {_pct(s.get('max_drawdown'))} | {_pct(d_dd)} | "
            f"{_pct(s.get('loyo_median'))} | {_pct(s.get('loyo_worst'))} | {exp_txt} | {top2_txt} | "
            f"{verdict(code, s)} |"
        )
    return "\n".join(lines)


def yearly_table(codes: list[str]) -> str:
    from .runner import load_summary

    years = [str(y) for y in range(2018, 2027)]
    rows = ["| 实验组 | " + " | ".join(years) + " |", "| --- | " + " | ".join(["---:"] * len(years)) + " |"]
    for code in codes:
        s = load_summary(code) or {}
        yr = s.get("yearly_returns") or {}
        cells = [_pct(yr.get(y)) if yr.get(y) is not None else "—" for y in years]
        rows.append(f"| `{code}` | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def write_named_md(filename: str, stage: str, codes: list[str], extra: str = "") -> None:
    from ..paths import ROUND10_ROOT, assert_project_result_path
    from .contract import PARENT_CODE

    v4 = v4_anchor_metrics()
    body = f"""# {filename.replace('.md', '')}

对照父策略：`{PARENT_CODE}`（CAGR {_pct(v4['cagr'])} / Sharpe {_sh(v4['sharpe'])} / MaxDD {_pct(v4['max_drawdown'])}）。
Δ 只对 v4。正式账本 `local_executor` → Stock `fixed_sleeve`。

## 正式结果

{metrics_table(codes)}

## 分年

{yearly_table(codes)}

{extra}
"""
    path = ROUND10_ROOT / filename
    assert_project_result_path(path)
    path.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")


def v5_gate_row(code: str, s: dict[str, Any] | None = None) -> dict[str, Any]:
    from .contract import V5_GATES
    from .registry import SPECS
    from .runner import load_summary

    s = s or load_summary(code) or {}
    v4 = v4_anchor_metrics()
    noise = float(V5_GATES["noise_pct"])
    if not s or s.get("cagr") is None:
        return {"meets": False, "reason": "no_metrics"}
    spec = SPECS.get(code)
    if spec and (spec.diagnostic or spec.post_hoc or spec.skip or spec.stage in {"parity", "attribution", "neighborhood", "execution", "perturbation", "capacity"}):
        return {"meets": False, "reason": "not_eligible"}
    cagr_ok = float(s["cagr"]) > v4["cagr"] - 1e-15
    sh_ok = float(s["sharpe"]) > v4["sharpe"] - 1e-15
    dd_ok = float(s["max_drawdown"]) >= float(V5_GATES["maxdd_ge"])
    loyo_m = s.get("loyo_median")
    loyo_w = s.get("loyo_worst")
    loyo_m_ok = loyo_m is None or float(loyo_m) >= v4["loyo_median"] - noise
    loyo_w_ok = loyo_w is None or float(loyo_w) >= v4["loyo_worst"] - noise
    meets = cagr_ok and sh_ok and dd_ok and loyo_m_ok and loyo_w_ok
    return {
        "meets": meets,
        "cagr_ok": cagr_ok,
        "sharpe_ok": sh_ok,
        "dd_ok": dd_ok,
        "loyo_med_ok": loyo_m_ok,
        "loyo_worst_ok": loyo_w_ok,
        "reason": "ok" if meets else "failed_gates",
    }


def _j(name: str) -> dict[str, Any]:
    from ..paths import ROUND10_ROOT
    import json as _json

    p = ROUND10_ROOT / "metrics" / name
    if p.is_file():
        return _json.loads(p.read_text(encoding="utf-8"))
    p2 = ROUND10_ROOT / name
    if p2.is_file():
        return _json.loads(p2.read_text(encoding="utf-8"))
    return {}


def _brief(code: str) -> str:
    from .runner import load_summary

    s = load_summary(code) or {}
    if s.get("cagr") is None:
        from .registry import SPECS

        spec = SPECS.get(code)
        return (spec.skip_reason if spec and spec.skip else None) or "未跑"
    return (
        f"CAGR {_pct(s.get('cagr'))} / Sharpe {_sh(s.get('sharpe'))} / "
        f"MaxDD {_pct(s.get('max_drawdown'))} / n={s.get('completed_trades')} / {verdict(code, s)}"
    )


def evaluate_promotion() -> dict[str, Any]:
    from ..paths import ROUND10_ROOT, assert_project_result_path
    from .contract import COMPLEXITY_CAGR, COMPLEXITY_SHARPE, STABILITY_CAGR_BAND, V5_GATES
    from .registry import SPECS
    from .runner import load_summary

    v4 = v4_anchor_metrics()
    boot = {}
    bpath = ROUND10_ROOT / "bootstrap" / "summary.json"
    if bpath.is_file():
        boot = json.loads(bpath.read_text(encoding="utf-8"))
    rows = []
    for code, spec in SPECS.items():
        s = load_summary(code)
        rows.append({"code": code, "verdict": verdict(code, s), "gate": v5_gate_row(code, s), "summary": s})
    cands = [r for r in rows if (r.get("gate") or {}).get("meets")]
    cands.sort(key=lambda r: (float((r["summary"] or {}).get("sharpe") or -9), float((r["summary"] or {}).get("cagr") or -9)), reverse=True)
    decision = "KEEP_BASELINE_V4"
    candidate = None
    upgrade = "NONE"
    stability = None
    for r in cands:
        code = r["code"]
        s = r["summary"] or {}
        p = ((boot.get("bootstrap") or {}).get(code) or {}).get("p_delta_ret_gt0")
        d_cg = float(s["cagr"]) - v4["cagr"]
        d_sh = float(s["sharpe"]) - v4["sharpe"]
        if p is not None and p < float(V5_GATES["prefer_p_delta_cagr"]):
            continue
        if d_cg < COMPLEXITY_CAGR and d_sh < COMPLEXITY_SHARPE and "C0" in code:
            continue
        candidate = code
        decision = f"CANDIDATE_BASELINE_V5={code}"
        upgrade = "RETURN" if d_cg > 0 and d_sh > 0 else "STABILITY"
        break
    if candidate is None:
        for r in rows:
            s = r["summary"] or {}
            if not s or s.get("cagr") is None:
                continue
            spec = SPECS.get(r["code"])
            if spec and (spec.diagnostic or spec.post_hoc or spec.skip or spec.stage == "parity"):
                continue
            d_cg = float(s["cagr"]) - v4["cagr"]
            if abs(d_cg) <= STABILITY_CAGR_BAND and float(s["sharpe"]) > v4["sharpe"] + 0.02:
                stability = r["code"]
                break
    payload = {
        "decision": "KEEP_BASELINE_V4" if candidate is None else decision,
        "auto_promote_v5": False,
        "candidate_baseline_v5": candidate,
        "stability_upgrade_candidate": stability,
        "upgrade_type": upgrade if candidate else ("STABILITY" if stability else "NONE"),
        "recommendation": "KEEP_BASELINE_V4" if candidate is None else "CANDIDATE_BASELINE_V5",
        "v4": v4,
        "prefer_p": V5_GATES["prefer_p_delta_cagr"],
        "note": "不自动改锁。须你确认才升 baseline_v5。",
    }
    path = ROUND10_ROOT / "metrics" / "promotion.json"
    assert_project_result_path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    return payload


def build_answers(decision: dict[str, Any]) -> dict[str, str]:
    from ..paths import ROUND10_ROOT
    from .contract import R10_00

    parity = {}
    ppath = ROUND10_ROOT / R10_00 / "parity.json"
    if ppath.is_file():
        parity = json.loads(ppath.read_text(encoding="utf-8"))
    attr = _j("attribution.json")
    pair = _j("pair_diagnostics.json")
    shrink = _j("shrinkage_walkforward.json")
    live_e = _j("live_execution.json") or _j("live_live.json")
    pert = ((live_e.get("perturbation_report") if live_e else None) or _j("live_perturbation.json") or {})
    if "perturbation_report" in pert:
        pert = pert["perturbation_report"]
    neigh = _j("live_neighborhood.json")
    cap = _j("live_capacity.json") or _j("live_live.json")
    shadow = {}
    sp = ROUND10_ROOT / "shadow" / "status.json"
    if sp.is_file():
        shadow = json.loads(sp.read_text(encoding="utf-8"))
    dd = _j("drawdown_top5.json")
    boot = {}
    bp = ROUND10_ROOT / "bootstrap" / "summary.json"
    if bp.is_file():
        boot = json.loads(bp.read_text(encoding="utf-8"))
    pvals = {k: (v or {}).get("p_delta_ret_gt0") for k, v in (boot.get("bootstrap") or {}).items()}
    return {
        "Q1": f"parity={parity.get('status')} extra_ok={parity.get('extra_ok')} NAV max abs={(parity.get('ledger') or {}).get('nav_max_abs')}。",
        "Q2": f"归因 verdict={attr.get('verdict')}。A01={_brief('R10_A01')}；A03={_brief('R10_A03')}；A04={_brief('R10_A04')}。",
        "Q3": f"A01={_brief('R10_A01')}；A02={_brief('R10_A02')}。两者都弱于 v4 则交互不可拆。",
        "Q4": f"PIT L1 覆盖={(pair.get('pit_l1_coverage'))}；L2={pair.get('l2')}；industry={ {k: pair.get('industry', {}).get(k) for k in ('stable_same_worse','stable_same_better','n_a','n_b')} }。",
        "Q5": f"corr={pair.get('corr')}。",
        "Q6": f"B 决策={pair.get('decisions')}；B01={_brief('R10_B01')} B02={_brief('R10_B02')} B03={_brief('R10_B03')}。",
        "Q7": f"C01={_brief('R10_C01')}；C02={_brief('R10_C02')}；C03={_brief('R10_C03')}。k=50，3Y WF。",
        "Q8": "复杂 C 组相对静态 B03：ΔCAGR<0.50pct 且 ΔSharpe<0.03 则 KEEP_SIMPLE。",
        "Q9": f"D 组：{_brief('R10_D_000bp')} / {_brief('R10_D_010bp')} / {_brief('R10_D_020bp')} / {_brief('R10_D_030bp')} / {_brief('R10_D_050bp')}。",
        "Q10": f"弹性={(live_e.get('elasticity') if live_e else None)}。",
        "Q11": f"扰动={pert}。",
        "Q12": f"E01 top1/top2 稳定={(pert.get('E01') or {}).get('top1_stability')} / {(pert.get('E01') or {}).get('top2_stability')}。",
        "Q13": f"分离邻域 F01={_brief('R10_F01_sep_low')} F02={_brief('R10_F02_sep_high')}。",
        "Q14": f"分配邻域 F03={_brief('R10_F03_alloc_048')} F04={_brief('R10_F04_alloc_0612')}；判定={(neigh.get('neighborhood') or neigh).get('flag') if neigh else None}。",
        "Q15": f"容量={(cap.get('capacity') or cap).get('label') if cap else None}；G01={_brief('R10_G01_1m_p3')} G02={_brief('R10_G02_3m_p3')} G03={_brief('R10_G03_10m_p3')}。300k=v4。",
        "Q16": "CRITICAL 缺失 → NO TRADE：five/ten/d1_open_gap/日行情/候选宇宙。pair 缺失 → fallback v4。见 operational_dependency_map.md。",
        "Q17": f"Shadow={shadow.get('status')}。{shadow.get('reason')}",
        "Q18": "无 post-2026-07-24 数据，无 shadow 绩效。",
        "Q19": f"Top5 回撤={dd.get('episodes')}；Top2 是否主因={dd.get('top2_is_main_dd')}。",
        "Q20": f"bootstrap P(ΔCAGR>0)={pvals}。偏好 ≥0.70。",
        "Q21": (
            f"{decision.get('recommendation')}。candidate={decision.get('candidate_baseline_v5')}。"
            f"STABILITY_UPGRADE_CANDIDATE={decision.get('stability_upgrade_candidate')}。"
            f"upgrade_type={decision.get('upgrade_type')}。auto_promote_v5=false。"
        ),
    }


def write_conclusion(decision: dict[str, Any]) -> None:
    from ..paths import ROUND10_ROOT, assert_project_result_path
    from .contract import PARENT_CODE
    from .registry import SPECS

    v4 = decision.get("v4") or v4_anchor_metrics()
    q = build_answers(decision)
    formal = [c for c, s in SPECS.items() if s.stage not in {"robustness", "perturbation"}]
    text = f"""# 12 — Round 10 结论

父策略始终是 `baseline_v4 = {PARENT_CODE}`。  
状态：historical development；**不是** final OOS。**不自动升 baseline_v5**。

正式对照：CAGR {_pct(v4.get('cagr'))} / Sharpe {_sh(v4.get('sharpe'))} / MaxDD {_pct(v4.get('max_drawdown'))}。

## 主表

{metrics_table(formal)}

## 升版

**{decision.get('decision')}**

推荐只写 `candidate_baseline_v5` 或 `KEEP_BASELINE_V4`。偏好 P(ΔCAGR>0)≥0.70，否则默认 KEEP。
允许 `STABILITY_UPGRADE_CANDIDATE={decision.get('stability_upgrade_candidate')}`，不能自动升。

---

### Q1 — baseline_v4 是否 exact parity？
{q['Q1']}

### Q2 — B03 增量主要来自 avoid-bad / preserve-good / 交互？
{q['Q2']}

### Q3 — Agreement 与 Separation 是否真正互补？
{q['Q3']}

### Q4 — Top1–Top2 同行业是主题确认、冗余，还是无稳定效应？
{q['Q4']}

### Q5 — pair 收益相关与 Top2 边际效用是否有稳定关系？
{q['Q5']}

### Q6 — 有没有 pair-relative feature 能稳定改善 v4？
{q['Q6']}

### Q7 — shrinkage allocator 是否优于 static B03？
{q['Q7']}

### Q8 — 复杂度增加值不值得？
{q['Q8']}

### Q9 — +10/+20/+30/+50bp 下策略如何衰减？
{q['Q9']}

### Q10 — baseline_v4 的 break-even execution cost 大概在哪？
{q['Q10']}

### Q11 — score 小扰动是否造成大量 rank flip？
{q['Q11']}

### Q12 — Top1 / Top2 哪一个更 fragile？
{q['Q12']}

### Q13 — 15/20/25、45/50/55 threshold neighborhood 是否形成 plateau？
{q['Q13']}

### Q14 — 0/4/8、0/5/10、0/6/12 allocation neighborhood 是否稳定？
{q['Q14']}

### Q15 — 300k / 1m / 3m / 10m 下容量如何？
{q['Q15']}

### Q16 — 实盘需要哪些 critical input？哪些 missing 时必须 NO TRADE？
{q['Q16']}

### Q17 — 是否存在真正 untouched post-2026-07-24 shadow OOS？
{q['Q17']}

### Q18 — 如果有，v4 与 finalists 在 shadow OOS 表现如何？
{q['Q18']}

### Q19 — Top5 回撤中 Top2 是否是主要风险来源？
{q['Q19']}

### Q20 — paired bootstrap P(ΔCAGR>0) 是否 ≥ 0.70？
{q['Q20']}

### Q21 — 有没有 RECOMMENDED_BASELINE_V5？
{q['Q21']}
"""
    path = ROUND10_ROOT / "12_round10_conclusion.md"
    assert_project_result_path(path)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def write_checklist(decision: dict[str, Any]) -> None:
    from ..paths import PROJECT_ROOT, ROUND10_ROOT, assert_project_result_path
    import subprocess

    commit = ""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = "NOT_A_GIT_REPO"
    body = f"""# Live readiness checklist

状态：historical development。**不是** production go-live。

## Data Dependencies

见 `operational_dependency_map.md`。CRITICAL：日行情、涨跌停、five/ten 预测、d1_open_gap、候选宇宙。

## Data Arrival Times

- D 收盘后：候选、双头预测、agreement / separation
- D+1 开盘前：`d1_open_gap`、委托生成
- 行情最大研究日：2026-07-24

## Critical Missing-Data Rules

- five / ten / d1_open_gap / 日行情 / 候选宇宙缺失 → **NO TRADE**
- pair 行业或相关缺失 → fallback baseline_v4 B03
- 禁止填 0

## Signal Generation Deadline

D 收盘后、D+1 开盘集合竞价前完成 Top1/Top2 与 B03 状态。

## Order Generation Deadline

D+1 开盘。开盘买。

## Required Logs / Daily Reconciliation

`trade_date, data_snapshot_id, dataset_hash, model_version, config_hash, git_commit, candidate_universe_hash, five_prediction_hash, ten_prediction_hash, Top1, Top2, agreement_state, separation_state, Top1 budget, Top2 budget, expected order, actual fill, slippage, position, NAV, exceptions`

## Execution Assumptions

双袖套；预算 = 开仓净值 × (30% Top1 + 10%×allocator Top2)；滑点 12.3bp + 佣金 1bp + 过户 0.1bp + 卖印花税 5bp。

## Liquidity / Capacity Limits

无盘口引擎。研究标签用日成交额 1%/3%/5%。见 07_capacity_liquidity.md。

## Kill Switch Conditions

- 数据 snapshot 对不上
- five/ten hash 对不上
- 开盘缺口缺失
- 单日异常回撤超研究 MaxDD 再恶化 3pct 且无法 replay

## Fallback Rules

pair 缺失 → v4；主预测缺失 → NO TRADE。

## Versions

- Model / Config：`round10_manifest_v1.yaml`
- Data：`hermes-cn-a-2010-20260724-v7-all-features`
- Git commit：`{commit}`
- Decision：{decision.get('decision')}
- Daily Signal / Order / Position Hash：replay 时对 `R10_00` 与当日 ledger 做 sha256
"""
    path = ROUND10_ROOT / "live_readiness_checklist.md"
    assert_project_result_path(path)
    path.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")


def patch_tracker(decision: dict[str, Any] | None = None) -> None:
    from .contract import PARENT_CODE, TRACKER_PATH
    from .registry import SPECS
    from ..paths import assert_project_result_path

    text = TRACKER_PATH.read_text(encoding="utf-8")
    decision = decision or {}
    formal = [c for c, s in SPECS.items() if s.stage not in {"robustness"}]
    block = f"""## Round 10

baseline parent = `baseline_v4` = `{PARENT_CODE}`。Delta 只对 v4。不自动升 `baseline_v5`。
正式账本 `result/round10/`。manifest `round10_manifest_v1.yaml`（Phase 1 freeze）。
结论：`{decision.get('decision')}`。upgrade_type=`{decision.get('upgrade_type')}`。
STABILITY_UPGRADE_CANDIDATE=`{decision.get('stability_upgrade_candidate')}`。

{metrics_table(formal)}

### 分年

{yearly_table(formal)}
"""
    if "## Round 10" in text:
        s10 = text.find("## Round 10")
        s_next = text.find("## 评选说明")
        if s_next > s10:
            text = text[:s10] + block + "\n" + text[s_next:]
        else:
            text = text[:s10] + block
    else:
        if text.endswith("\n"):
            text = text + block
        else:
            text = text + "\n" + block
    assert_project_result_path(TRACKER_PATH)
    TRACKER_PATH.write_text(text, encoding="utf-8")


def write_registry_csv() -> None:
    import csv
    from ..paths import ROUND10_ROOT, assert_project_result_path
    from .registry import SPECS
    from .runner import load_summary

    v4 = v4_anchor_metrics()
    path = ROUND10_ROOT / "experiment_registry.csv"
    assert_project_result_path(path)
    fields = [
        "experiment_id", "stage", "parent", "change", "pre_registered", "post_hoc",
        "CAGR", "DeltaCAGR", "Sharpe", "DeltaSharpe", "MaxDD", "DeltaMaxDD",
        "LOYO_Median", "LOYO_Worst", "Worst12M", "AvgExposure", "Verdict", "Reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for code, spec in SPECS.items():
            s = load_summary(code) or {}
            w.writerow({
                "experiment_id": code,
                "stage": spec.stage,
                "parent": spec.parent,
                "change": spec.title_zh,
                "pre_registered": (not spec.post_hoc),
                "post_hoc": spec.post_hoc,
                "CAGR": s.get("cagr"),
                "DeltaCAGR": (None if s.get("cagr") is None else float(s["cagr"]) - v4["cagr"]),
                "Sharpe": s.get("sharpe"),
                "DeltaSharpe": (None if s.get("sharpe") is None else float(s["sharpe"]) - v4["sharpe"]),
                "MaxDD": s.get("max_drawdown"),
                "DeltaMaxDD": (None if s.get("max_drawdown") is None else float(s["max_drawdown"]) - v4["max_drawdown"]),
                "LOYO_Median": s.get("loyo_median"),
                "LOYO_Worst": s.get("loyo_worst"),
                "Worst12M": s.get("worst_rolling_12m"),
                "AvgExposure": s.get("avg_gross_exposure"),
                "Verdict": verdict(code, s),
                "Reason": spec.skip_reason,
            })


def print_complete(decision: dict[str, Any]) -> None:
    from ..paths import RESULT_ROOT, ROUND10_ROOT
    from .contract import PARENT_CODE

    write_registry_csv()
    pair = _j("pair_diagnostics.json")
    comb = _j("combine_slots.json")
    shadow = "NOT_AVAILABLE"
    sp = ROUND10_ROOT / "shadow" / "status.json"
    if sp.is_file():
        shadow = json.loads(sp.read_text(encoding="utf-8")).get("status") or shadow
    print("", flush=True)
    print("ROUND10 COMPLETE", flush=True)
    print("", flush=True)
    print("Current baseline:", flush=True)
    print(f"baseline_v4 = {PARENT_CODE}", flush=True)
    print("", flush=True)
    print("Baseline tracking:", flush=True)
    print(f" {RESULT_ROOT / '一进二_Round7基线_实验组追踪.md'}", flush=True)
    print("", flush=True)
    print("Best pair candidate:", flush=True)
    print(pair.get("decisions") and comb.get("best_pair") or "NONE", flush=True)
    print("", flush=True)
    print("Best shrinkage candidate:", flush=True)
    print(comb.get("best_shrink") or "NONE", flush=True)
    print("", flush=True)
    print("Best live-robust candidate:", flush=True)
    print(decision.get("stability_upgrade_candidate") or decision.get("candidate_baseline_v5") or "NONE", flush=True)
    print("", flush=True)
    print("Shadow OOS:", flush=True)
    print(shadow, flush=True)
    print("", flush=True)
    print("Recommendation:", flush=True)
    print(decision.get("recommendation") or "KEEP_BASELINE_V4", flush=True)
    print("", flush=True)
    print("Upgrade type:", flush=True)
    print(decision.get("upgrade_type") or "NONE", flush=True)
    print("", flush=True)
    print("Conclusion:", flush=True)
    print(f" {ROUND10_ROOT / '12_round10_conclusion.md'}", flush=True)
    print("", flush=True)
    print("Live readiness:", flush=True)
    print(f" {ROUND10_ROOT / 'live_readiness_checklist.md'}", flush=True)


__all__ = [
    "build_answers",
    "evaluate_promotion",
    "lock_v4_tracker",
    "metrics_table",
    "patch_tracker",
    "print_complete",
    "v4_anchor_metrics",
    "v5_gate_row",
    "verdict",
    "write_checklist",
    "write_conclusion",
    "write_named_md",
    "yearly_table",
]
