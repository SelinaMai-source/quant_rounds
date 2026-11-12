"""Round9 markdown / 升版判定 / 追踪表。不改写 R1–R8 实验数字。不自动升 v4。"""

from __future__ import annotations

import json
from typing import Any

from ..paths import ROUND9_ROOT, assert_project_result_path
from .allocator import activity_stats, load_alloc
from .contract import (
    COMPLEXITY_CAGR,
    COMPLEXITY_SHARPE,
    PARENT_CODE,
    R9_00,
    TARGET_CAGR,
    TARGET_LOYO_MED,
    TARGET_LOYO_WORST,
    TARGET_MAXDD,
    TARGET_SHARPE,
    TRACKER_PATH,
    V4_GATES,
)
from .registry import SPECS
from .runner import load_summary


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


def _n(x) -> str:
    return "—" if x is None else str(int(x))


def v3_anchor_metrics() -> dict[str, Any]:
    s = load_summary(R9_00) or {}
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


def v4_gate_row(code: str, s: dict[str, Any] | None = None) -> dict[str, Any]:
    s = s or load_summary(code) or {}
    v3 = v3_anchor_metrics()
    noise = float(V4_GATES["noise_pct"])
    if not s or s.get("cagr") is None:
        return {"meets": False, "reason": "no_metrics"}
    spec = SPECS.get(code)
    if spec and (spec.diagnostic or spec.post_hoc or spec.skip or spec.stage == "parity"):
        return {"meets": False, "reason": "not_eligible"}
    cagr_ok = float(s["cagr"]) > v3["cagr"] - 1e-15
    sh_ok = float(s["sharpe"]) > v3["sharpe"] - 1e-15
    dd_ok = float(s["max_drawdown"]) >= float(V4_GATES["maxdd_ge"])
    loyo_m = s.get("loyo_median")
    loyo_w = s.get("loyo_worst")
    loyo_m_ok = loyo_m is None or float(loyo_m) >= v3["loyo_median"] - noise
    loyo_w_ok = loyo_w is None or float(loyo_w) >= v3["loyo_worst"] - noise
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


def verdict(code: str, s: dict[str, Any] | None = None) -> str:
    spec = SPECS.get(code)
    s = s or load_summary(code)
    if spec and spec.skip:
        return spec.skip_reason or "SKIP"
    if not s or s.get("cagr") is None:
        return (s or {}).get("status") or "未跑"
    if spec and spec.post_hoc:
        return "POST_HOC_EXPLORATORY"
    if spec and spec.diagnostic:
        return "DIAGNOSTIC"
    if code in {R9_00, "R9_B04"}:
        return "CONTROL"
    gate = v4_gate_row(code, s)
    v3 = v3_anchor_metrics()
    d_cg = float(s["cagr"]) - v3["cagr"]
    d_sh = float(s["sharpe"]) - v3["sharpe"]
    alloc_path = None
    try:
        st = activity_stats(load_alloc(code))
        if st.get("flag") == "LOW_ACTIVITY":
            return "LOW_ACTIVITY"
    except Exception:
        pass
    exp = s.get("avg_gross_exposure")
    if exp is not None and v3.get("avg_gross_exposure") and d_cg > 0:
        if float(exp) > float(v3["avg_gross_exposure"]) * 1.15 and d_sh < 0.03:
            return "EXPOSURE_DRIVEN"
    if gate["meets"] and d_sh >= 0.01:
        return "MEETS_GATES"
    if gate["meets"]:
        return "FAIL_COMPLEXITY_PENALTY"
    if d_cg > 0 or d_sh > 0:
        return "MIXED"
    return "NO_INCREMENT"


def metrics_table(codes: list[str]) -> str:
    return _metrics_table_clean(codes, v3_anchor_metrics())


def _metrics_table_clean(codes: list[str], v3: dict[str, Any]) -> str:
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
        d_sh = (s["sharpe"] - v3["sharpe"]) if s.get("sharpe") is not None else None
        d_cg = (s["cagr"] - v3["cagr"]) if s.get("cagr") is not None else None
        d_dd = (s["max_drawdown"] - v3["max_drawdown"]) if s.get("max_drawdown") is not None else None
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
    years = [str(y) for y in range(2018, 2027)]
    rows = ["| 实验组 | " + " | ".join(years) + " |", "| --- | " + " | ".join(["---:"] * len(years)) + " |"]
    for code in codes:
        s = load_summary(code) or {}
        yr = s.get("yearly_returns") or {}
        cells = [_pct(yr.get(y)) if yr.get(y) is not None else "—" for y in years]
        rows.append(f"| `{code}` | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def write_named_md(filename: str, stage: str, codes: list[str], extra: str = "") -> None:
    v3 = v3_anchor_metrics()
    body = f"""# {filename.replace('.md', '')}

对照父策略：`{PARENT_CODE}`（CAGR {_pct(v3['cagr'])} / Sharpe {_sh(v3['sharpe'])} / MaxDD {_pct(v3['max_drawdown'])}）。
B01–B03 是 replacement allocator（0/5/10），**不再乘** E02 的 1.00/0.75/0.50。

## 正式结果

{metrics_table(codes)}

## 分年

{yearly_table(codes)}

{extra}
"""
    path = ROUND9_ROOT / filename
    assert_project_result_path(path)
    path.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")


def prefer_simple(b01: dict[str, Any] | None, c01: dict[str, Any] | None) -> str:
    if not c01 or c01.get("cagr") is None:
        return "simple"
    if not b01 or b01.get("cagr") is None:
        return "meta"
    d_cg = float(c01["cagr"]) - float(b01["cagr"])
    d_sh = float(c01["sharpe"]) - float(b01["sharpe"])
    if d_cg < COMPLEXITY_CAGR and d_sh < COMPLEXITY_SHARPE:
        return "simple"
    return "meta"


def evaluate_promotion() -> dict[str, Any]:
    v3 = v3_anchor_metrics()
    rows = []
    for code, spec in SPECS.items():
        if spec.stage == "robustness":
            continue
        s = load_summary(code)
        rows.append({
            "code": code,
            "verdict": verdict(code, s),
            "gate": v4_gate_row(code, s),
            "summary": s,
        })
    cands = [r for r in rows if (r.get("gate") or {}).get("meets") and r["verdict"] == "MEETS_GATES"]
    cands.sort(key=lambda r: (float((r["summary"] or {}).get("sharpe") or -9), float((r["summary"] or {}).get("cagr") or -9)), reverse=True)
    b01 = load_summary("R9_B01")
    c01 = load_summary("R9_C01")
    pref = prefer_simple(b01, c01)
    decision = "KEEP_BASELINE_V3"
    candidate = None
    if cands:
        candidate = cands[0]["code"]
        if pref == "simple" and any(c["code"] == "R9_B01" for c in cands):
            candidate = "R9_B01"
        decision = f"candidate_baseline_v4={candidate}"
    payload = {
        "decision": decision,
        "auto_promote_v4": False,
        "candidate_baseline_v4": candidate,
        "prefer": pref,
        "v3": v3,
        "candidates": cands,
        "note": "不自动改锁。须你确认才升 baseline_v4。",
    }
    path = ROUND9_ROOT / "metrics" / "promotion.json"
    assert_project_result_path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    return payload


def _brief(code: str) -> str:
    s = load_summary(code) or {}
    if s.get("cagr") is None:
        spec = SPECS.get(code)
        return (spec.skip_reason if spec and spec.skip else None) or "未跑"
    return (
        f"CAGR {_pct(s.get('cagr'))} / Sharpe {_sh(s.get('sharpe'))} / "
        f"MaxDD {_pct(s.get('max_drawdown'))} / n={_n(s.get('completed_trades'))} / {verdict(code, s)}"
    )


def build_answers(decision: dict[str, Any]) -> dict[str, str]:
    import json as _json
    from pathlib import Path

    def _j(name):
        p = ROUND9_ROOT / "metrics" / name
        return _json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    parity = {}
    ppath = ROUND9_ROOT / "R9_00_baseline_v3" / "parity.json"
    if ppath.is_file():
        parity = json.loads(ppath.read_text(encoding="utf-8"))
    forensic = _j("forensics_meta.json")
    meta = _j("meta_walkforward.json")
    direction = _j("direction_d00.json")
    residual = _j("residual_meta.json")
    dd = _j("drawdown_top5.json")
    combine = _j("combine_slots.json")
    boot = {}
    bpath = ROUND9_ROOT / "bootstrap" / "summary.json"
    if bpath.is_file():
        boot = json.loads(bpath.read_text(encoding="utf-8"))
    q1 = f"parity={parity.get('status')}。目标 CAGR={TARGET_CAGR} Sharpe={TARGET_SHARPE} MaxDD={TARGET_MAXDD} n=1783。观察 {parity.get('observed')}。"
    q2 = f"A01 vs v3 组合增量：{forensic.get('marginal')}。2D 表见 01_top2_marginal_forensics.md。"
    q3 = f"B01={_brief('R9_B01')}。预注册 9 格 replacement，不再乘 E02。"
    q4 = f"B02={_brief('R9_B02')}（C2=0.5A+0.5S）。"
    q5 = f"B03={_brief('R9_B03')} conservative。"
    q6 = (
        f"C01={_brief('R9_C01')}；B01={_brief('R9_B01')}。prefer={decision.get('prefer')}。"
        f"C01 activity={(meta.get('activity') or {}).get('C01')}。"
    )
    q7 = f"C02={_brief('R9_C02')}；C03={_brief('R9_C03')}；C04={_brief('R9_C04')}。单调={meta.get('monotone_c01')}。"
    q8 = f"漏泄 {meta.get('leakage')}。"
    q9 = f"D00={direction.get('decision')}。stable_oof={direction.get('stable_oof')}。"
    q10 = f"E01={_brief('R9_E01')}；E02={_brief('R9_E02')}；E03={_brief('R9_E03')}。"
    q11 = f"residual families={residual.get('families')} beaten={residual.get('beaten')} F04={_brief('R9_F04')}。"
    q12 = f"Top5 episodes={dd.get('episodes')}；DD01={_brief('R9_DD01')}。无全组合市场门。"
    q13 = f"K slots={combine.get('slots')} runnable={combine.get('runnable')} 规则=primary vs veto。"
    q14 = f"bootstrap candidate={boot.get('candidate')}；{ {k: (v or {}).get('p_delta_ret_gt0') for k, v in (boot.get('bootstrap') or {}).items()} }。"
    q15 = (
        f"v3 fee15={_brief('R9_RB_v3_fee15')} fee20={_brief('R9_RB_v3_fee20')}；"
        f"+20bp={_brief('R9_ADV_v3_20bp')}。"
    )
    q16 = (
        f"{decision.get('decision')}。auto_promote_v4=false。"
        f"candidate_baseline_v4={decision.get('candidate_baseline_v4')}。"
        "须你确认才改锁。"
    )
    return {
        "Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5, "Q6": q6, "Q7": q7,
        "Q8": q8, "Q9": q9, "Q10": q10, "Q11": q11, "Q12": q12, "Q13": q13,
        "Q14": q14, "Q15": q15, "Q16": q16,
    }


def write_conclusion(decision: dict[str, Any]) -> None:
    v3 = decision.get("v3") or v3_anchor_metrics()
    q = build_answers(decision)
    formal = [c for c, s in SPECS.items() if s.stage not in {"robustness"} and not s.diagnostic]
    text = f"""# 10 — Round 9 结论

父策略始终是 `baseline_v3 = {PARENT_CODE}`。  
状态：historical development；**不是** final OOS。**不自动升 baseline_v4**。

正式对照：CAGR {_pct(v3.get('cagr'))} / Sharpe {_sh(v3.get('sharpe'))} / MaxDD {_pct(v3.get('max_drawdown'))}。

## 主表

{metrics_table(formal)}

## 升版

**{decision.get('decision')}**

推荐只写 `candidate_baseline_v4`。复杂度：相对更简单模型 ΔCAGR<{COMPLEXITY_CAGR:.2%} 且 ΔSharpe<{COMPLEXITY_SHARPE} → 默认留简单。
B vs C 是本轮关键科学问题。prefer={decision.get('prefer')}。

---

### Q1 — R9_00 能否逐笔复现 R8_E02？
{q['Q1']}

### Q2 — Top2 边际效用 2D（A/B/C × LOW/MID/HIGH）？
{q['Q2']}

### Q3 — B01 九格 replacement 相对 v3？
{q['Q3']}

### Q4 — B02 C2=0.5A+0.5S？
{q['Q4']}

### Q5 — B03 conservative？
{q['Q5']}

### Q6 — C01 walk-forward 是否打赢 B01？是否 LOW_ACTIVITY？
{q['Q6']}

### Q7 — C02 / C03 / C04？
{q['Q7']}

### Q8 — 漏泄测试？
{q['Q8']}

### Q9 — 方向 D00 / 是否 SKIP D01–D03？
{q['Q9']}

### Q10 — execution +10/+20bp buffer？
{q['Q10']}

### Q11 — residual 真实列 family？
{q['Q11']}

### Q12 — Top5 回撤只调 Top2？
{q['Q12']}

### Q13 — K01–K05 primary vs veto？
{q['Q13']}

### Q14 — 20 日块 bootstrap 2000 paired？
{q['Q14']}

### Q15 — 费用 ×1.5/×2 与买侧不利？
{q['Q15']}

### Q16 — KEEP_BASELINE_V3 还是 candidate_baseline_v4？
{q['Q16']}
"""
    path = ROUND9_ROOT / "10_round9_conclusion.md"
    assert_project_result_path(path)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def patch_tracker() -> None:
    """只改「当前锁定」为开发父策略=v3，并追加 ## Round 9。不改 R1–R8 数字。"""
    text = TRACKER_PATH.read_text(encoding="utf-8")
    old = "| 当前锁定（本线开发基线） | **`R7_FS01122_DUAL90_10_CLEAN`** = `R1_00_baseline`（clean_risk × Top2 30/10）"
    new = "| 当前锁定（开发父策略） | **`baseline_v3` = `R8_E02`**（historical；v1=`R7_FS01122_DUAL90_10_CLEAN` / v2=`R4_E_top1_minus_median` 作历史行保留）"
    if old in text:
        text = text.replace(old, new, 1)
    elif "当前锁定（开发父策略）" not in text and "当前锁定（本线开发基线）" in text:
        import re

        text = re.sub(
            r"\| 当前锁定（本线开发基线） \|.*\|",
            new + "                                                |",
            text,
            count=1,
        )
    formal = [c for c, s in SPECS.items() if s.stage not in {"robustness"}]
    v3 = v3_anchor_metrics()
    block = f"""## Round 9

开发父策略 = `baseline_v3` = `R8_E02`。Delta 相对 v3，不是 v2 / r7_v1。
Top2 做成 0/5/10% replacement allocator + 严格 OOF meta。不自动升 `baseline_v4`。
正式账本 `result/round9/`。manifest `round9_manifest_v1.yaml`。

{metrics_table(formal)}

### 分年

{yearly_table(formal)}
"""
    if "## Round 9" in text:
        s9 = text.find("## Round 9")
        s_next = text.find("## 评选说明")
        if s_next > s9:
            text = text[:s9] + block + "\n" + text[s_next:]
        else:
            text = text[:s9] + block
    else:
        mark = "## 评选说明"
        if mark in text:
            text = text.replace(mark, block + "\n" + mark, 1)
        else:
            text = text + "\n" + block
    assert_project_result_path(TRACKER_PATH)
    TRACKER_PATH.write_text(text, encoding="utf-8")


def print_complete(decision: dict[str, Any]) -> None:
    v3 = decision.get("v3") or v3_anchor_metrics()
    sha = ""
    digest = ROUND9_ROOT / "round9_manifest_v1.yaml.sha256"
    if digest.is_file():
        sha = digest.read_text(encoding="utf-8").strip().split()[0]
    print("", flush=True)
    print("========== ROUND9 COMPLETE ==========", flush=True)
    print(f"parent: baseline_v3 = {PARENT_CODE}", flush=True)
    print(f"R9_00: CAGR={v3.get('cagr')} Sharpe={v3.get('sharpe')} MaxDD={v3.get('max_drawdown')}", flush=True)
    print(f"manifest_sha256: {sha}", flush=True)
    print(f"decision: {decision.get('decision')}", flush=True)
    print(f"candidate_baseline_v4: {decision.get('candidate_baseline_v4')}", flush=True)
    print(f"auto_promote_v4: false", flush=True)
    print(f"B vs C prefer: {decision.get('prefer')}", flush=True)
    print(f"B01: {_brief('R9_B01')}", flush=True)
    print(f"C01: {_brief('R9_C01')}", flush=True)
    print(f"B04 control: {_brief('R9_B04')}", flush=True)
    print("=====================================", flush=True)


__all__ = [
    "build_answers",
    "evaluate_promotion",
    "metrics_table",
    "patch_tracker",
    "print_complete",
    "v3_anchor_metrics",
    "v4_gate_row",
    "verdict",
    "write_conclusion",
    "write_named_md",
    "yearly_table",
]
