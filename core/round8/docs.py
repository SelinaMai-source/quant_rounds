"""写 Round8 markdown / manifest / 追踪表。不改写旧 R1–R7 实验表。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..paths import ROUND8_ROOT, assert_project_result_path, ensure_round8_tree
from .contract import (
    ANCHOR_DIR,
    BASELINE_V1,
    BASELINE_V2,
    MANIFEST_NAME,
    PARENT_CODE,
    PROMOTION_RULES,
    R8_00,
    TARGET_CAGR,
    TARGET_MAXDD,
    TARGET_N_TRADES,
    TARGET_SHARPE,
    TRACKER_PATH,
    V2_OVERLAY,
)
from .diagnostics import dump_contract_facts
from .registry import SPECS
from .robustness import v2_anchor_metrics
from .runner import load_summary

STAGES_DOC = {
    "parity": "01_baseline_v2_parity.md",
    "geometry": "02_geometry_diagnostics.md",
    "confidence": "03_smooth_confidence.md",
    "rank-specific": "04_rank_specific_confidence.md",
    "top2": "05_conditional_top2.md",
    "disagreement": "06_dual_head_agreement.md",
    "shape": "07_distribution_shape.md",
    "ecology": "08_ecology_marginal.md",
    "left-tail": "09_left_tail_attribution.md",
    "execution": "10_execution_audit.md",
    "combine": "11_final_combinations.md",
    "robustness": "12_robustness.md",
    "conclusion": "13_round8_conclusion.md",
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
    if x is None:
        return "—"
    return f"{float(x):.3f}"


def _n(x) -> str:
    if x is None:
        return "—"
    return str(int(x))


def write_tree() -> Path:
    ensure_round8_tree()
    return ROUND8_ROOT


def freeze_manifest() -> dict[str, str]:
    """委托 freeze.py，禁止用另一套 yaml 覆盖已锁定 sha256。"""
    from .freeze import freeze_manifest as _freeze

    payload = _freeze(overwrite=False)
    return {"manifest": payload.get("path", ""), "sha256": payload.get("sha256", "")}


def write_contract_md() -> Path:
    facts = dump_contract_facts()
    md = f"""# 00 — baseline_v2 契约

父策略：`{PARENT_CODE}`。  
状态：historical development baseline；**不是** final OOS；**不是** production validated。  
Alpha 仍是 Round7 双头 90/10，本文件只锁 **仓位/置信度** 定义。真值来自

`result/replay_round7/round4/R4_E_top1_minus_median/`

以及 `core/round4/overlays.py` 的 `cross_section_daily` / `_cross_section_mult`。

## 锁定量

| 项 | 值 |
| --- | --- |
| 市场层 | **B01a** 上涨家数分档 45/50/60 → 仓位 0 / 0.50 / 0.75 / 1.00 |
| 是否含 B01a | **是** |
| clean_risk | **否**（`used_locked_round7_risk=false`） |
| Top1 / Top2 预算 | 开仓净值 **30% + 10%**；同一日乘子同时作用两个 rank |
| 融合分 | 同日 percentile 后 **0.90 ten + 0.10 five** |
| 几何量 | 对融合分做 `top1 − median`（**原始差**，不再做日内分位） |
| 滚动窗 | **250** 日 `rank(pct=True)`，`min_periods=50` |
| 规则 | 分位 ≤20% → confidence ×0.5，否则 ×1.0；暖窗 NaN → 1.0 |
| 合成 | `market × confidence`（乘积） |
| 小候选日 | n=1 ⇒ 差=0；空日跳过；分位未满窗不降仓 |

## 实测（当前融合分面板）

- 信号日数：{facts.get('n_signal_days')}
- n≤3 日：{facts['small_candidate_days']['n_days']}
- n=1 / n=2：{facts['small_candidate_days']['n1']} / {facts['small_candidate_days']['n2']}
- 市场层 skip 日（广度 <45%）：{facts['market_skip_days']}
- overlay meta：`skip_days={facts['overlay_meta'].get('skip_days')} half={facts['overlay_meta'].get('half_days')} full={facts['overlay_meta'].get('full_days')}`

## 继承问题

`d1_open_gap` 仍在 ranker 与融合预滤里。D+1 开盘信息、回测也按 D+1 开盘成交。数字不能写成纯 D-close 可部署结果。

## 不是

- 不是新 Alpha
- 不自动升 `baseline_v3`
- `R4_P_08` 只作 ecology 对照，不进默认 baseline
"""
    path = ROUND8_ROOT / "00_baseline_v2_contract.md"
    assert_project_result_path(path)
    path.write_text(md, encoding="utf-8")
    return path


def _verdict(code: str, s: dict[str, Any] | None, v2: dict[str, Any]) -> str:
    spec = SPECS.get(code)
    if spec and spec.skip:
        return spec.skip_reason or "SKIP"
    if not s or s.get("cagr") is None:
        return (s or {}).get("status") or "未跑"
    if code in {R8_00, "R8_B04"}:
        return "CONTROL"
    if spec and spec.overlay_spec.get("kind") == "exact_p08":
        return "CONTROL_P08"
    if spec and spec.diagnostic:
        return "DIAGNOSTIC"
    cagr_ok = float(s["cagr"]) >= float(v2["cagr"]) - 1e-12
    sh_ok = float(s["sharpe"]) >= float(v2["sharpe"]) - 1e-12
    dd_ok = float(s["max_drawdown"]) >= float(v2["max_drawdown"]) - 0.01
    loyo_ok = s.get("loyo_median") is None or float(s["loyo_median"]) >= (v2.get("loyo_median") or -9)
    loyo_w = True
    if s.get("loyo_worst") is not None and v2.get("loyo_worst") is not None:
        loyo_w = float(s["loyo_worst"]) >= float(v2["loyo_worst"]) - 0.01
    not_2024 = True
    if s.get("cagr_ex2024") is not None and v2.get("cagr_ex2024") is not None:
        if cagr_ok and float(s["cagr_ex2024"]) + 1e-12 < float(v2["cagr_ex2024"]):
            not_2024 = False
    gates = cagr_ok and sh_ok and dd_ok and loyo_ok and loyo_w and not_2024
    dsh = float(s["sharpe"]) - float(v2["sharpe"])
    if spec and spec.post_hoc:
        return "POST_HOC_EXPLORATORY"
    if gates and dsh >= 0.01 + 1e-12:
        return "MEETS_GATES"
    if gates:
        return "FAIL_COMPLEXITY_PENALTY"
    if cagr_ok or sh_ok:
        return "MIXED"
    return "NO_INCREMENT"


def _metrics_table(codes: list[str], v2: dict[str, Any] | None = None) -> str:
    v2 = v2 or v2_anchor_metrics()
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
            status = _verdict(code, s, v2)
            lines.append(
                f"| `{code}` | `{parent}` | {title} | — | — | — | — | — | — | — | — | — | — | {status} |"
            )
            continue
        d_sh = (s["sharpe"] - v2["sharpe"]) if s.get("sharpe") is not None else None
        d_cg = (s["cagr"] - v2["cagr"]) if s.get("cagr") is not None else None
        d_dd = (s["max_drawdown"] - v2["max_drawdown"]) if s.get("max_drawdown") is not None else None
        exp = s.get("avg_gross_exposure")
        exp_txt = _pct(exp, signed=False) if exp is not None else "—"
        top2 = s.get("top2_pnl")
        top2_txt = f"{float(top2):,.0f}" if top2 is not None else "—"
        lines.append(
            f"| `{code}` | `{parent}` | {title} | {_pct(s.get('cagr'))} | {_pct(d_cg)} | "
            f"{_sh(s.get('sharpe'))} | {_sh(d_sh)} | {_pct(s.get('max_drawdown'))} | {_pct(d_dd)} | "
            f"{_pct(s.get('loyo_median'))} | {_pct(s.get('loyo_worst'))} | {exp_txt} | {top2_txt} | "
            f"{_verdict(code, s, v2)} |"
        )
    return "\n".join(lines)


def _yearly_table(codes: list[str]) -> str:
    years = [str(y) for y in range(2018, 2027)]
    header = "| 实验组 | " + " | ".join(years) + " |"
    sep = "| --- | " + " | ".join(["---:" ] * len(years)) + " |"
    rows = [header, sep]
    for code in codes:
        s = load_summary(code) or {}
        yr = s.get("yearly_returns") or {}
        cells = [_pct(yr.get(y) or yr.get(int(y)), signed=True) if yr.get(y) is not None or yr.get(int(y)) is not None else "—" for y in years]
        # yearly keys are strings in summaries
        cells = []
        for y in years:
            val = yr.get(y)
            cells.append(_pct(val) if val is not None else "—")
        rows.append(f"| `{code}` | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def write_stage_md(stage: str, extra: str = "") -> Path:
    name = STAGES_DOC[stage]
    codes = [c for c, s in SPECS.items() if s.stage == stage]
    v2 = v2_anchor_metrics()
    body = f"""# {name.split('_', 1)[-1].replace('.md', '')}

对照父策略：`{PARENT_CODE}`（CAGR {_pct(v2['cagr'])} / Sharpe {_sh(v2['sharpe'])} / MaxDD {_pct(v2['max_drawdown'])}）。

## 正式结果

{_metrics_table(codes, v2) if codes else "（本阶段无回测组）"}

## 分年

{_yearly_table(codes) if codes else "—"}

{extra}
"""
    path = ROUND8_ROOT / name
    assert_project_result_path(path)
    path.write_text(body, encoding="utf-8")
    return path


def write_parity_md(parity: dict[str, Any]) -> Path:
    extra = f"""
## Parity

- 状态：**{parity.get('status')}**
- 锚点：`{parity.get('anchor')}`
- 交易错配：{parity.get('ledger', {}).get('trade_mismatches')}
- NAV max abs：{parity.get('ledger', {}).get('nav_max_abs')}
- 指标对齐：{parity.get('metric_ok')}

观测：CAGR={parity.get('observed', {}).get('cagr')} Sharpe={parity.get('observed', {}).get('sharpe')} MaxDD={parity.get('observed', {}).get('max_drawdown')} n={parity.get('observed', {}).get('trades')}
"""
    path = ROUND8_ROOT / "01_baseline_v2_parity.md"
    v2 = v2_anchor_metrics()
    text = f"""# 01 — baseline_v2 parity

`R8_00_baseline_v2` 必须复现 `{PARENT_CODE}`。

目标约 50.75% / 1.717 / −24.37% / 1787。

{_metrics_table([R8_00], v2)}

{extra}

逐笔字段见 `{R8_00}/parity_fields.parquet`（含 five/ten/fused、几何、乘子、预算、成交）。
"""
    assert_project_result_path(path)
    path.write_text(text, encoding="utf-8")
    return path


def write_geometry_md(payload: dict[str, Any]) -> Path:
    interp = payload.get("high_sep_means") or {}
    corr_t = interp.get("corr_sep_vs_top1")
    corr_m = interp.get("corr_sep_vs_median")
    if corr_t is not None and corr_m is not None and abs(float(corr_m)) > abs(float(corr_t)):
        ab = "更接近 **B. 其他候选特别弱**（分离度与 median 的负相关更强）。"
    else:
        ab = "更接近 **A. Top1 特别强**。"

    def _decile_md(name: str, block: dict) -> str:
        lines = [
            f"### {name}",
            "",
            "| decile | n | Top1 mean ret | Top2 mean ret | win | large-loss(≤−5%) | right-tail(≥10%) | mean hold |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for k in sorted(block, key=lambda x: int(x)):
            r = block[k]
            lines.append(
                f"| {k} | {r.get('n')} | {_pct(r.get('top1_mean'))} | {_pct(r.get('top2_mean'))} | "
                f"{_pct(r.get('win_rate'), signed=False)} | {_pct(r.get('large_loss_rate'), signed=False)} | "
                f"{_pct(r.get('right_tail_rate'), signed=False)} | "
                f"{r.get('mean_holding'):.2f}" if r.get("mean_holding") is not None else "— "
            )
        # fix hold cell
        out = []
        out.append(f"### {name}")
        out.append("")
        out.append("| decile | n | Top1 mean ret | Top2 mean ret | win | large-loss(≤−5%) | right-tail(≥10%) | mean hold |")
        out.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for k in sorted(block, key=lambda x: int(x)):
            r = block[k]
            hold = f"{float(r['mean_holding']):.2f}" if r.get("mean_holding") is not None else "—"
            out.append(
                f"| {k} | {r.get('n')} | {_pct(r.get('top1_mean'))} | {_pct(r.get('top2_mean'))} | "
                f"{_pct(r.get('win_rate'), signed=False)} | {_pct(r.get('large_loss_rate'), signed=False)} | "
                f"{_pct(r.get('right_tail_rate'), signed=False)} | {hold} |"
            )
        return "\n".join(out)

    deciles = payload.get("deciles") or {}
    parts = [_decile_md(name, block) for name, block in deciles.items() if isinstance(block, dict)]
    extra = f"""# 02 — 横截面几何诊断

父策略：`{PARENT_CODE}`。分数 = Round7 融合分（同日 percentile 后再 90/10）。几何在 **fused raw** 上计算，PIT = rolling250。

## 高 Top1−Median 更像什么

- 高分位日 Top1 分：{interp.get('high_p80_mean_top1_score')}
- 高分位日 median：{interp.get('high_p80_mean_median')}
- 低分位日 Top1 分：{interp.get('low_p20_mean_top1_score')}
- 低分位日 median：{interp.get('low_p20_mean_median')}
- corr(sep, top1)={corr_t}；corr(sep, median)={corr_m}

{ab}

日频几何：`metrics/geometry_daily.parquet`。

## Decile（实现收益，不是单票故事）

{chr(10).join(parts)}
"""
    path = ROUND8_ROOT / "02_geometry_diagnostics.md"
    assert_project_result_path(path)
    path.write_text(extra, encoding="utf-8")
    return path


def pick_winners() -> dict[str, Any]:
    v2 = v2_anchor_metrics()
    groups = {
        "smooth": ["R8_B01", "R8_B02", "R8_B03"],
        "top2": ["R8_D01", "R8_D02", "R8_D03", "R8_D04"],
        "disagree": ["R8_E01", "R8_E02", "R8_E03"],
        "rank": ["R8_C01", "R8_C02", "R8_C03", "R8_C04"],
        "shape": ["R8_F01", "R8_F02", "R8_F03"],
        "ecology": ["R8_G02", "R8_G03", "R8_G04"],
        "left": ["R8_H01", "R8_H02"],
    }

    def _score(code: str):
        s = load_summary(code)
        if not s or s.get("sharpe") is None:
            return None
        stable = (
            float(s["cagr"]) >= v2["cagr"]
            and float(s["sharpe"]) >= v2["sharpe"]
            and float(s["max_drawdown"]) >= float(v2["max_drawdown"]) - 0.01
        )
        return (1 if stable else 0, float(s["sharpe"]), float(s["cagr"]), code, s)

    out = {}
    for name, codes in groups.items():
        ranked = [x for x in (_score(c) for c in codes) if x]
        ranked.sort(reverse=True)
        if not ranked:
            out[name] = None
            continue
        best = ranked[0]
        out[name] = {"code": best[3], "stable": bool(best[0]), "summary": best[4]}
    return out


def materialize_combines() -> list[str]:
    from .registry import get_spec

    winners = pick_winners()
    b = winners.get("smooth") if (winners.get("smooth") or {}).get("stable") else None
    d = winners.get("top2") if (winners.get("top2") or {}).get("stable") else None
    e = winners.get("disagree") if (winners.get("disagree") or {}).get("stable") else None

    def _kind(code_or_none, prefix):
        if not code_or_none:
            return None
        return str(code_or_none["code"]).replace("R8_", "")

    def _disagree_overlay(item):
        if not item:
            return {}
        tag = _kind(item, "E")
        if tag == "E02":
            return {"e02": True}
        return {"disagree": tag}

    slots = {
        "R8_J01": {"kind": "custom", "conf_kind": _kind(b, "B")} if b else None,
        "R8_J02": {"kind": "custom", "d_rule": _kind(d, "D")} if d else None,
        "R8_J03": {"kind": "custom", **_disagree_overlay(e)} if e else None,
        "R8_J04": {"kind": "custom", "conf_kind": _kind(b, "B"), "d_rule": _kind(d, "D")} if (b and d) else None,
        "R8_J05": {"kind": "custom", "conf_kind": _kind(b, "B"), **_disagree_overlay(e)} if (b and e) else None,
        "R8_J06": {"kind": "custom", "d_rule": _kind(d, "D"), **_disagree_overlay(e)} if (d and e) else None,
    }
    # 单模块组合 ≡ 该模块自己，不重跑
    skip_identical = {
        "R8_J01": b["code"] if b else None,
        "R8_J02": d["code"] if d else None,
        "R8_J03": e["code"] if e else None,
    }
    runnable = []
    for code, ov in slots.items():
        spec = get_spec(code)
        if ov is None:
            spec.skip = True
            spec.skip_reason = "no_stable_module_increment"
            spec.overlay_spec = {"kind": "combine_slot", "slot": int(code[-2:])}
            continue
        twin = skip_identical.get(code)
        if twin:
            spec.skip = True
            spec.skip_reason = f"identical_to_{twin}"
            spec.overlay_spec = ov
            continue
        spec.skip = False
        spec.skip_reason = ""
        spec.overlay_spec = ov
        runnable.append(code)
    path = ROUND8_ROOT / "metrics" / "combine_slots.json"
    assert_project_result_path(path)
    path.write_text(json.dumps(
        {
            "winners": {k: (v or {}) for k, v in winners.items()},
            "slots": slots,
            "runnable": runnable,
            "note": "三模块以上叠加未启用；无稳定增量的槽位 skip。J01-J03 若等于单模块则 skip。",
        },
        ensure_ascii=False, indent=2, default=str,
    ) + "\n")
    return runnable


def evaluate_promotion(codes: list[str]) -> dict[str, Any]:
    v2 = v2_anchor_metrics()
    rows = []
    for code in codes:
        s = load_summary(code)
        if not s or s.get("cagr") is None:
            continue
        ok = (
            float(s["cagr"]) >= v2["cagr"]
            and float(s["sharpe"]) >= v2["sharpe"]
            and float(s["max_drawdown"]) >= float(v2["max_drawdown"]) - 0.01
            and (s.get("loyo_median") is None or float(s["loyo_median"]) >= (v2["loyo_median"] or -9))
        )
        # 2024-only: if ex2024 worse and 2024 dominates
        yr = s.get("yearly_returns") or {}
        y2024 = yr.get("2024")
        not_2024_only = True
        if y2024 is not None and s.get("cagr_ex2024") is not None and v2.get("cagr_ex2024") is not None:
            if float(s["cagr"]) >= v2["cagr"] and float(s["cagr_ex2024"]) + 1e-12 < float(v2["cagr_ex2024"]):
                not_2024_only = False
        loyo_worst_ok = True
        if s.get("loyo_worst") is not None and v2.get("loyo_worst") is not None:
            loyo_worst_ok = float(s["loyo_worst"]) >= float(v2["loyo_worst"]) - 0.01
        promote = bool(ok and not_2024_only and loyo_worst_ok)
        spec = SPECS.get(code)
        if spec and (spec.diagnostic or spec.post_hoc or spec.skip):
            promote = False
            reason = "diagnostic_or_post_hoc" if not spec.skip else spec.skip_reason
        elif code in {R8_00, "R8_B04"}:
            promote = False
            reason = "control"
        elif promote and (float(s["sharpe"]) - v2["sharpe"]) < 0.01 + 1e-12:
            promote = False
            reason = "complexity_penalty"
        else:
            reason = "ok" if promote else "failed_gates"
        rows.append({"code": code, "promote": promote, "reason": reason, "verdict": _verdict(code, s, v2), "summary": s})
    winners = [r for r in rows if r["promote"] and r["code"] != R8_00]
    winners.sort(key=lambda r: (float(r["summary"]["sharpe"]), float(r["summary"]["cagr"])), reverse=True)
    decision = {
        "decision": "RECOMMENDED_BASELINE_V3=" + winners[0]["code"] if winners else "KEEP_BASELINE_V2",
        "keep": PARENT_CODE if not winners else None,
        "candidates": rows,
        "v2": v2,
    }
    path = ROUND8_ROOT / "metrics" / "promotion.json"
    assert_project_result_path(path)
    path.write_text(json.dumps(decision, ensure_ascii=False, indent=2, default=str) + "\n")
    return decision


def write_conclusion(decision: dict[str, Any], extras: dict[str, Any]) -> Path:
    v2 = decision.get("v2") or v2_anchor_metrics()
    q = extras.get("answers") or {}
    text = f"""# 13 — Round 8 结论

父策略始终是 `baseline_v2 = {PARENT_CODE}`。  
状态：historical development；**不是** final OOS。

正式对照：CAGR {_pct(v2.get('cagr'))} / Sharpe {_sh(v2.get('sharpe'))} / MaxDD {_pct(v2.get('max_drawdown'))}。

## 主表

{_metrics_table([c for c, s in SPECS.items() if s.stage not in {"robustness"} and not s.diagnostic], v2)}

## 升版

**{decision.get('decision')}**

复杂度惩罚：+0.01 Sharpe 不够开新层。没有同时满足 CAGR/Sharpe/MaxDD/LOYO/非2024-only 的，就保持 v2。

---

### Q1
{q.get('Q1', '')}

### Q2
{q.get('Q2', '')}

### Q3
{q.get('Q3', '')}

### Q4
{q.get('Q4', '')}

### Q5
{q.get('Q5', '')}

### Q6
{q.get('Q6', '')}

### Q7
{q.get('Q7', '')}

### Q8
{q.get('Q8', '')}

### Q9
{q.get('Q9', '')}

### Q10
{q.get('Q10', '')}

### Q11
{q.get('Q11', '')}

### Q12
{q.get('Q12', '')}

### Q13
{q.get('Q13', '')}

### Q14
{q.get('Q14', '')}

### Q15
{q.get('Q15', decision.get('decision'))}
"""
    path = ROUND8_ROOT / "13_round8_conclusion.md"
    assert_project_result_path(path)
    path.write_text(text, encoding="utf-8")
    return path


def build_answers(decision: dict[str, Any], extras: dict[str, Any]) -> dict[str, str]:
    v2 = decision.get("v2") or v2_anchor_metrics()
    geom = extras.get("geometry") or {}
    interp = (geom.get("high_sep_means") or {})
    left = extras.get("left_tail") or {}
    dis = extras.get("disagreement") or {}
    winners = extras.get("winners") or {}
    parity = extras.get("parity") or {}

    def brief(code):
        s = load_summary(code) or {}
        if s.get("cagr") is None:
            return "未跑"
        return f"CAGR {_pct(s.get('cagr'))} / Sharpe {_sh(s.get('sharpe'))} / MaxDD {_pct(s.get('max_drawdown'))} / n={_n(s.get('completed_trades'))}"

    q1 = f"parity={parity.get('status')}。锚点 {PARENT_CODE}。交易错配 {parity.get('ledger', {}).get('trade_mismatches')}，NAV max abs {parity.get('ledger', {}).get('nav_max_abs')}。"
    q2 = (
        f"Top1−Median 看的是 Top1 相对整云的脱离；Top1−Top2 只看前两名。高分离日 median={interp.get('high_p80_mean_median')}，"
        f"低分离日 median={interp.get('low_p20_mean_median')}；corr(sep,top1)={interp.get('corr_sep_vs_top1')}，corr(sep,median)={interp.get('corr_sep_vs_median')}。"
    )
    q3 = "见几何 decile：`02_geometry_diagnostics.md`。以 Top1 实现净收益是否随 top1_minus_median_p250 上升判断单调。"
    q4 = f"硬规则对照 B04={brief('R8_B04')}；B01={brief('R8_B01')}；B02={brief('R8_B02')}；B03={brief('R8_B03')}。取 Sharpe/CAGR 相对 v2 的稳定增量者。"
    q5 = f"C01={brief('R8_C01')}；C02={brief('R8_C02')}；C03={brief('R8_C03')}；C04={brief('R8_C04')}。"
    q6 = f"D 组：D01={brief('R8_D01')}；D02={brief('R8_D02')}；D03={brief('R8_D03')}；D04={brief('R8_D04')}；Top1-only={brief('R8_D_TOP1_ONLY')}。"
    q7 = "用 Top2−Median 分位对照 Top2 实现收益与边际效用（Top1+Top2 相对 Top1-only）。"
    q8 = f"状态分布见诊断。E01={brief('R8_E01')}；E02={brief('R8_E02')}；E03={brief('R8_E03')}。融合权重未改。"
    q9 = f"F01={brief('R8_F01')}；F02={brief('R8_F02')}；F03={brief('R8_F03')}。"
    g_inc = extras.get("ecology_redundant")
    q10 = f"G01={brief('R8_G01')}；G02={brief('R8_G02')}；G03={brief('R8_G03')}；G04={brief('R8_G04')}。结论：{g_inc or '见 08 节'}。"
    rp = (left.get("rank_pnl") or {})
    q11 = f"回撤期 Top1/Top2 贡献：{json.dumps(rp, ensure_ascii=False)}。"
    q12 = f"H01={brief('R8_H01')}；H02={brief('R8_H02')}；H03={(left.get('h03') or {}).get('decision', 'SKIP_H03')}。"
    q13 = f"Track B `R8_I01_D_CLOSE_ONLY`={brief('R8_I01_D_CLOSE_ONLY')}。与 v2 的缺口即 execution-time `d1_open_gap` 贡献上界。"
    q14 = (
        f"v2 +20bp={brief('R8_ADV_v2_20bp')}；+30bp={brief('R8_ADV_v2_30bp')}；+50bp={brief('R8_ADV_v2_50bp')}。"
        f"v1 对照 +20bp={brief('R8_ADV_v1_20bp')}。"
    )
    q15 = decision.get("decision") or "KEEP_BASELINE_V2"
    return {
        "Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5, "Q6": q6, "Q7": q7,
        "Q8": q8, "Q9": q9, "Q10": q10, "Q11": q11, "Q12": q12, "Q13": q13, "Q14": q14, "Q15": q15,
    }


def patch_tracker(round8_codes: list[str]) -> Path:
    """只改 lineage / 当前最优表，并追加 ## Round 8。不重写 R1–R7 实验表。"""
    text = TRACKER_PATH.read_text(encoding="utf-8")
    v2s = load_summary(R8_00) or {}
    # fallback to known replay numbers
    if v2s.get("cagr") is None:
        v2s = {
            "cagr": TARGET_CAGR,
            "sharpe": TARGET_SHARPE,
            "max_drawdown": TARGET_MAXDD,
            "total_return": 32.51487361549896,
            "completed_trades": TARGET_N_TRADES,
            "yearly_returns": {
                "2018": 0.2985916362498864, "2019": 0.0933504121460298,
                "2020": 0.35536579716348493, "2021": 0.6426447293281494,
                "2022": -0.11600321916125822, "2023": 0.6835540527161061,
                "2024": 2.201371827179363, "2025": 1.1839011955670329,
                "2026": 0.01896341730927098,
            },
            "worst_year_return": -0.11600321916125822,
        }

    lineage = """## Baseline lineage

```text
baseline_v1 = r7_v1 = R7_FS01122_DUAL90_10_CLEAN
baseline_v2 = R4_E_top1_minus_median
```

状态：**historical development**；**不是** final OOS；**不是** production validated。  
v2 仍继承 `d1_open_gap` execution-time feature。`R4_P_08` 只作 secondary challenger，生态层不进默认 baseline。

"""

    new_opt = f"""## 当前最优 baseline

本线入场都是 Round7 双头 90/10。**v1 = Round7 候选本身**（clean_risk 已含在锁里）。  
**v2 = `R4_E_top1_minus_median`**：同一套 Alpha + B01a × 截面 top1−median 底 20%×0.5。这是策略开发基线，不是新 Alpha。

| 名称 | 操作/构成 | Sharpe | 年化 | 累计 | MaxDD | 最差年 | 笔数 | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_v1` / `r7_v1`（`R1_00_baseline` / `R7_FS01122_DUAL90_10_CLEAN`） | 双头 90/10 + clean_risk + Top2 30/10，`close_from_d2` | 1.668 | 46.98% | +2597.94% | −25.42% | −9.55%（2022） | 2158 | 历史开发基线；parity 通过 |
| `baseline_v2`（`R4_E_top1_minus_median`） | 双头 90/10 + B01a × top1−median 底20%×0.5 + Top2 30/10 | {_sh(v2s.get('sharpe'))} | {_pct(v2s.get('cagr'))} | {_pct(v2s.get('total_return'))} | {_pct(v2s.get('max_drawdown'))} | {_pct(v2s.get('worst_year_return'))}（2022） | {_n(v2s.get('completed_trades'))} | 当前开发父策略；historical，非 final OOS |

### 分年

2026 为 YTD，不得当完整年。

| | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `r7_v1` | +22.06% | +8.35% | +31.23% | +78.27% | −9.55% | +70.27% | +207.72% | +76.46% | +4.27% |
| `baseline_v2` | {_yr(v2s, '2018')} | {_yr(v2s, '2019')} | {_yr(v2s, '2020')} | {_yr(v2s, '2021')} | {_yr(v2s, '2022')} | {_yr(v2s, '2023')} | {_yr(v2s, '2024')} | {_yr(v2s, '2025')} | {_yr(v2s, '2026')} |

"""

    # insert lineage after intro block if missing
    if "## Baseline lineage" not in text:
        needle = "## 共享回测契约"
        if needle in text:
            text = text.replace(needle, lineage + needle, 1)
        else:
            text = lineage + text

    # replace 当前最优 baseline section only
    start = text.find("## 当前最优 baseline")
    end = text.find("## Round 1")
    if start >= 0 and end > start:
        text = text[:start] + new_opt + text[end:]

    r8_block = _round8_tracker_section(round8_codes)
    if "## Round 8" in text:
        s8 = text.find("## Round 8")
        s_next = text.find("## 评选说明")
        if s_next < 0:
            text = text[:s8] + r8_block
        else:
            text = text[:s8] + r8_block + text[s_next:]
    else:
        mark = "## 评选说明"
        if mark in text:
            text = text.replace(mark, r8_block + mark, 1)
        else:
            text = text + "\n" + r8_block

    new_note = """## 评选说明

- 本线 **strategy development parent** = **`baseline_v2` = `R4_E_top1_minus_median`**（同一套 Round7 双头 Alpha + B01a × 截面 top1−median 底20%×0.5）。这是仓位/置信 overlay，**不是**新 Alpha。
- `r7_v1` = `baseline_v1` 的数字全部保留，作为历史对照，不再当默认开发底。
- `R4_P_08` 只作 secondary challenger；生态层默认不进 v2。+0.01 Sharpe 不够加一层。
- **不自动**升 `baseline_v3`。须同时满足 CAGR≥v2、Sharpe≥v2、MaxDD 恶化不超过 1pct、LOYO median≥v2、LOYO worst 不明显恶化、不是只由 2024 驱动。
- 状态：historical development；**不是** final OOS；**不是** production validated。`d1_open_gap` 仍是 execution-time feature。
- freeze 之后新增组标 `POST_HOC_EXPLORATORY`，不能参与 v3 promotion。

产物根：`result/replay_round7/`（R1–R7）与 `result/round8/`（Round 8）。旧两张追踪表未删。Stock `result/round7/` 只读、未改写。
"""
    nstart = text.find("## 评选说明")
    if nstart >= 0:
        text = text[:nstart] + new_note
    else:
        text = text + "\n" + new_note

    assert_project_result_path(TRACKER_PATH)
    TRACKER_PATH.write_text(text, encoding="utf-8")
    return TRACKER_PATH


def _yr(summary: dict[str, Any], year: str) -> str:
    yr = summary.get("yearly_returns") or {}
    val = yr.get(year)
    return _pct(val) if val is not None else "—"


def _round8_tracker_section(codes: list[str]) -> str:
    v2 = v2_anchor_metrics()
    formal = [c for c in codes if not SPECS[c].diagnostic]
    return f"""## Round 8

父策略 = `baseline_v2` = `R4_E_top1_minus_median`。Delta 相对 v2，不是相对 r7_v1。  
Alpha 锁死。正式账本 `result/round8/`。

{_metrics_table(formal, v2)}

### 分年

{_yearly_table(formal)}

"""


__all__ = [
    "build_answers",
    "evaluate_promotion",
    "freeze_manifest",
    "materialize_combines",
    "patch_tracker",
    "pick_winners",
    "write_conclusion",
    "write_contract_md",
    "write_geometry_md",
    "write_parity_md",
    "write_stage_md",
    "write_tree",
]
