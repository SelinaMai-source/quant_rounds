"""主报告 Q1–Q28、追踪表追加、ROUND12 COMPLETE。不写「写入时间」。不升 v5。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND12_ROOT
from .alphaforge import load_alphaforge
from .common import dump_json, pct, write_md
from .contract import (
    KEEP_BASELINE_V4,
    PRE_REGISTERED_QUESTIONS,
    PROMOTION_LOYO_SLACK,
    PROMOTION_MAX_CAGR_DROP_PP,
    PROMOTION_MDD_SLACK,
    PROMOTION_SHARPE_SLACK,
    R12_01_TARGETS,
    R12_06,
    TRACKER_PATH,
    V4_TARGETS,
    VERDICTS,
)
from .deploy import load_capacity, load_deploy
from .dsr import load_dsr
from .execution import load_execution
from .freeze import require_frozen
from .parity import load_parity_status
from .placebo import load_placebo
from .shadow import load_shadow
from .stats import load_pbo, load_staircase
from .timing_placebo import load_timing
from .trials import load_trials
from .universe import load_universe

REPORT_MD = ROUND12_ROOT / "ROUND12_RISK_REMEDIATION_REPORT.md"
CONTROLS_MD = ROUND12_ROOT / "15_controls.md"
CONTROLS_JSON = ROUND12_ROOT / "controls" / "controls.json"
DOSSIER_JSON = ROUND12_ROOT / "stats" / "dossier.json"


def _promotion(r01: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    if not cand or cand.get("cagr") is None:
        return {
            "decision": KEEP_BASELINE_V4,
            "reason": "无合格 DEPLOYMENT_CANDIDATE 数字",
            "passed_gate": False,
            "final": KEEP_BASELINE_V4,
        }
    cagr_drop_pp = (float(r01.get("cagr") or 0) - float(cand.get("cagr") or 0)) * 100.0
    sharpe_drop = float(r01.get("sharpe") or 0) - float(cand.get("sharpe") or 0)
    mdd_worse = abs(float(cand.get("max_drawdown") or 0)) - abs(float(r01.get("max_drawdown") or 0))
    loyo_drop = float(r01.get("loyo_worst") or 0) - float(cand.get("loyo_worst") or 0)
    checks = {
        "sharpe_ok": sharpe_drop <= PROMOTION_SHARPE_SLACK,
        "mdd_ok": mdd_worse <= PROMOTION_MDD_SLACK,
        "loyo_ok": loyo_drop <= PROMOTION_LOYO_SLACK,
        "cagr_ok": cagr_drop_pp <= PROMOTION_MAX_CAGR_DROP_PP,
    }
    deploy_improvements = int(bool(cand.get("liquidity") or cand.get("family"))) + int(
        bool(cand.get("buy_extra_slippage_bps"))
    )
    passed = all(checks.values()) and deploy_improvements >= 2
    return {
        "decision": KEEP_BASELINE_V4,
        "passed_gate": passed,
        "cagr_drop_pp": cagr_drop_pp,
        "sharpe_drop": sharpe_drop,
        "mdd_worse": mdd_worse,
        "loyo_drop": loyo_drop,
        "deploy_improvements": deploy_improvements,
        "checks": checks,
        "reason": (
            "门槛数字或可通过，但本轮强制默认 KEEP_BASELINE_V4，不得自动升 v5。"
            if passed
            else "未同时满足 Sharpe/MaxDD/LOYO 不差 + CAGR 掉幅≤1.0pp + 至少两项部署性改善。"
        ),
        "final": KEEP_BASELINE_V4,
    }


def _verdict(ctx: dict[str, Any]) -> str:
    surv = ((ctx.get("universe") or {}).get("audit") or {}).get("verdict")
    exe = (ctx.get("execution") or {}).get("classification")
    pbo = ctx.get("pbo") or {}
    parity = ctx.get("parity") or {}
    if not parity.get("passed"):
        return "INVALID_BACKTEST"
    if surv == "MATERIAL_DIFFERENCE":
        return "SURVIVORSHIP_RISK_TOO_HIGH"
    if exe == "HIGH_RISK":
        return "EXECUTION_RISK_TOO_HIGH"
    uw = ((pbo.get("by_universe") or {}).get("U_WINDOW") or {}).get("pbo")
    if uw is not None and float(uw) >= 0.80:
        return "BACKTEST_FRAGILE"
    if uw is not None and float(uw) >= 0.50:
        return "PROMISING_BUT_SELECTION_RISK"
    return "ROBUST_BUT_FORWARD_OOS_REQUIRED"


def _risk_dims(ctx: dict[str, Any]) -> list[dict[str, str]]:
    surv = ((ctx.get("universe") or {}).get("audit") or {}).get("verdict") or "UNQUANTIFIED"
    exe = (ctx.get("execution") or {}).get("classification") or "UNQUANTIFIED"
    pbo_g = (ctx.get("pbo") or {}).get("TRUE_GLOBAL_PBO_STATUS")
    dsr = (ctx.get("dsr") or {}).get("CANONICAL_DSR")
    return [
        {"dim": "survivorship", "level": "UNQUANTIFIED" if surv == "UNQUANTIFIED" else ("HIGH" if surv == "MATERIAL_DIFFERENCE" else "MODERATE"), "note": str(surv)},
        {"dim": "execution", "level": "HIGH" if exe in {"HIGH_RISK", "NOT_IDENTIFIABLE"} else "MODERATE", "note": str(exe)},
        {"dim": "selection_pbo", "level": "HIGH" if pbo_g == "TRUE_GLOBAL_PBO_NOT_IDENTIFIABLE" else "MODERATE", "note": str(pbo_g)},
        {"dim": "cost", "level": "MODERATE", "note": "R12_01 固化；佣金/过户 FEE_HISTORY_UNVERIFIED"},
        {"dim": "forward_oos", "level": "CRITICAL", "note": "FORWARD_OOS_NOT_AVAILABLE"},
        {"dim": "dsr", "level": "MODERATE" if dsr is not None else "UNQUANTIFIED", "note": str(dsr)},
        {"dim": "placebo", "level": "MODERATE", "note": "full-engine matched + state-timing"},
        {"dim": "promotion", "level": "CRITICAL", "note": "FORBIDDEN auto v5; KEEP_BASELINE_V4"},
    ]


def _ladder(results: dict[str, Any], r00: dict[str, Any], r01: dict[str, Any]) -> list[dict[str, Any]]:
    order = [
        ("LEGACY", r00),
        ("COST_CORRECTED", r01),
        ("EXEC_P50", results.get("R12_03_EXEC_MEDIAN") or {}),
        ("EXEC_P75", results.get("R12_04_EXEC_CONSERVATIVE") or {}),
        ("LIQ_EXEC_P50", results.get("R12_06_LIQ_EXEC") or {}),
        ("LIQ_EXEC_P75", results.get("R12_07_LIQ_EXEC_CONSERVATIVE") or {}),
    ]
    r02 = results.get("R12_02_PIT_UNIVERSE") or {}
    if r02.get("cagr") is not None:
        order.append(("PIT_CORRECTED", r02))
    return [{"rung": n, "cagr": r.get("cagr"), "sharpe": r.get("sharpe"), "max_drawdown": r.get("max_drawdown")} for n, r in order]


def gather_ctx() -> dict[str, Any]:
    frozen = require_frozen()
    parity = load_parity_status()
    deploy = load_deploy()
    results = deploy.get("results") or {}
    r00 = ((parity.get("anchors") or {}).get("R12_00") or {}).get("observed") or {}
    r01 = ((parity.get("anchors") or {}).get("R12_01") or {}).get("observed") or dict(R12_01_TARGETS)
    promo = _promotion(r01, results.get(R12_06) or {})
    ctx = {
        "frozen": frozen,
        "parity": parity,
        "universe": load_universe(),
        "execution": load_execution(),
        "deploy": deploy,
        "capacity": load_capacity(),
        "trials": load_trials(),
        "pbo": load_pbo(),
        "dsr": load_dsr(),
        "alphaforge": load_alphaforge(),
        "placebo": load_placebo(),
        "timing": load_timing(),
        "staircase": load_staircase(),
        "shadow": load_shadow(),
        "promotion": promo,
        "ladder": _ladder(results, r00, r01),
    }
    ctx["verdict"] = _verdict(ctx)
    ctx["risks"] = _risk_dims(ctx)
    ctx["questions"] = _questions(ctx)
    return ctx


def _questions(ctx: dict[str, Any]) -> dict[str, str]:
    frozen = ctx.get("frozen") or {}
    parity = ctx.get("parity") or {}
    anchors = parity.get("anchors") or {}
    univ = ctx.get("universe") or {}
    audit = univ.get("audit") or {}
    exe = ctx.get("execution") or {}
    results = (ctx.get("deploy") or {}).get("results") or {}
    r00 = (anchors.get("R12_00") or {}).get("observed") or {}
    r01 = (anchors.get("R12_01") or {}).get("observed") or {}
    r02 = results.get("R12_02_PIT_UNIVERSE") or {}
    r03 = results.get("R12_03_EXEC_MEDIAN") or {}
    r04 = results.get("R12_04_EXEC_CONSERVATIVE") or {}
    r05 = results.get("R12_05_LIQUIDITY_PRIORITY") or {}
    r06 = results.get("R12_06_LIQ_EXEC") or {}
    r07 = results.get("R12_07_LIQ_EXEC_CONSERVATIVE") or {}
    cap = ctx.get("capacity") or {}
    trials = ctx.get("trials") or {}
    pbo = ctx.get("pbo") or {}
    dsr = ctx.get("dsr") or {}
    af = ctx.get("alphaforge") or {}
    plc = ctx.get("placebo") or {}
    tim = ctx.get("timing") or {}
    stair = ctx.get("staircase") or {}
    shadow = ctx.get("shadow") or {}
    promo = ctx.get("promotion") or {}
    execp = exe.get("exec") or {}
    uw = (pbo.get("by_universe") or {}).get("U_WINDOW") or {}
    up = (pbo.get("by_universe") or {}).get("U_PROMO") or {}
    perc = (plc.get("percentiles") or {}).get("R12_01") or {}
    layers = tim.get("layers") or {}
    return {
        "Q1": f"是。manifest sha256=`{frozen.get('sha256')}`。intake 先于任何新策略绩效。",
        "Q2": f"{(anchors.get('R12_00') or {}).get('status')}。观察 CAGR={r00.get('cagr')} Sharpe={r00.get('sharpe')} MaxDD={r00.get('max_drawdown')} 笔数={r00.get('completed_trades')}。目标 CAGR={V4_TARGETS['cagr']}。",
        "Q3": f"{(anchors.get('R12_01') or {}).get('status')}。对齐 R11_V4_CORRECTED。观察 CAGR={r01.get('cagr')} Sharpe={r01.get('sharpe')} MaxDD={r01.get('max_drawdown')}。R12_01 是 REALISM_REFERENCE，**不是 v5**。",
        "Q4": "没有。正式策略仅预注册 8 个 ID。资本后缀与 10M 是同一规则诊断。未搜 cap，未发明 25bp。",
        "Q5": f"是。spine=daily_raw；ST=known_at≤day；used_current_security_list={audit.get('used_current_security_list')}；used_membership_effective_to={audit.get('used_membership_effective_to')}。",
        "Q6": f"独立退市公告 PIT={audit.get('independent_delist_announcement_pit')}。缺证据不得 SURVIVORSHIP_PASS。",
        "Q7": f"候选重叠={audit.get('n_candidate_in_pit')}/{audit.get('n_candidate_rows')}；Top 排除={audit.get('n_top_excluded')}（Top1 {audit.get('n_top1_excluded')} / Top2 {audit.get('n_top2_excluded')}）。终审=**{audit.get('verdict')}**。",
        "Q8": f"R12_02 status=`{r02.get('status')}`。{r02.get('reason') or univ.get('R12_02_will_run')}。质量门槛=SURVIVORSHIP_PASS。",
        "Q9": f"因果链 `{exe.get('causal_chain')}`。T_DECISION=D+1 09:25，成交用同一 official open → **SAME_OPEN_EXECUTION_OPTIMISM**。",
        "Q10": f"2017–2021 全量；首 bar={((exe.get('inventory') or {}).get('first_bar'))}；09:25={((exe.get('inventory') or {}).get('has_0925_auction'))}；2022+ 几乎不可用。",
        "Q11": f"P50/P75/P90={execp.get('P50')}/{execp.get('P75')}/{execp.get('P90')} bp；label=`{execp.get('label')}`；fallback={exe.get('fallback')}；校准 2018–2019 / 验证 2020–2021。",
        "Q12": f"**{exe.get('classification')}**。",
        "Q13": f"R12_03 CAGR={r03.get('cagr')} Sharpe={r03.get('sharpe')} MaxDD={r03.get('max_drawdown')}；R12_04 CAGR={r04.get('cagr')} Sharpe={r04.get('sharpe')} MaxDD={r04.get('max_drawdown')}。",
        "Q14": "是。participation=3% of signal-day amount，Top1 先分完，Top2 拿剩余。未搜 cap。",
        "Q15": f"评级=**{cap.get('rating')}**。10M 仅 diagnostic。",
        "Q16": f"R12_06 CAGR={r06.get('cagr')} Sharpe={r06.get('sharpe')} MaxDD={r06.get('max_drawdown')} 笔数={r06.get('completed_trades')}。fill={r06.get('fill')}。角色=DEPLOYMENT_CANDIDATE，不是自动 v5。",
        "Q17": f"R12_07 CAGR={r07.get('cagr')} Sharpe={r07.get('sharpe')} MaxDD={r07.get('max_drawdown')}。可作为更保守部署情景，不是新 Alpha。",
        "Q18": f"legacy v4 CAGR={V4_TARGETS['cagr']} vs R12_01 {r01.get('cagr')} vs R12_06 {r06.get('cagr')}。Sharpe {V4_TARGETS['sharpe']} / {r01.get('sharpe')} / {r06.get('sharpe')}。",
        "Q19": f"会计分解（非因果）：legacy {pct(V4_TARGETS['cagr'])} → 印花税 {pct(r01.get('cagr'))} → EXEC_P50 {pct(r03.get('cagr'))} → EXEC_P75 {pct(r04.get('cagr'))} → LIQ {pct(r05.get('cagr'))} → LIQ_EXEC_P50 {pct(r06.get('cagr'))} → LIQ_EXEC_P75 {pct(r07.get('cagr'))}。",
        "Q20": f"矩阵 n={pbo.get('n_matrix')}（已取消 max_cols=80）。U_WINDOW PBO={uw.get('pbo')} n={uw.get('n_strats')}；U_PROMO PBO={up.get('pbo')} n={up.get('n_strats')}。documented/discovered/upper={trials.get('n_documented')}/{trials.get('n_full_discovered')}/{trials.get('n_conservative_upper')}。",
        "Q21": f"**{pbo.get('TRUE_GLOBAL_PBO_STATUS')}**。禁止把可恢复子集外推到 1339/2678。",
        "Q22": f"CANONICAL_DSR={dsr.get('CANONICAL_DSR')}（N={dsr.get('canonical_n_choice')}={((dsr.get('n_trials_set') or {}).get('discovered'))}）。单测={(dsr.get('unit_tests') or {}).get('status')}。日频 SR + 日 N，未混年化 Sharpe。",
        "Q23": f"status=`{af.get('status')}` commit=`{af.get('commit')}`。只重试一次。失败不阻断。",
        "Q24": f"n={plc.get('n_placebo')} 方法={plc.get('method')}。CAGR 百分位={perc.get('cagr')} Sharpe={perc.get('sharpe')} MaxDD={perc.get('max_drawdown')} Worst12M={perc.get('worst12m')}。",
        "Q25": ("；".join(f"{k} n={v.get('n')} CAGR百分位={(v.get('percentiles') or {}).get('cagr')} p={v.get('empirical_p_cagr')}" for k, v in layers.items()) or "NOT_AVAILABLE") + f"。n_target={tim.get('n_target')}（{tim.get('n_reason')}）。",
        "Q26": str(stair.get("classification") or "NOT_AVAILABLE"),
        "Q27": f"`{shadow.get('status')}`。协议 `{shadow.get('protocol_dir')}`。工具 tools/run_shadow_daily.py。未把 CPCV 写成 OOS。",
        "Q28": f"升 v5 门槛 passed={promo.get('passed_gate')}；正式裁决=**{promo.get('final') or KEEP_BASELINE_V4}**。六选一 verdict=**{ctx.get('verdict')}**。原因：{promo.get('reason')}",
    }


def run_controls(**_: Any) -> dict[str, Any]:
    ctx = {
        "kill_switch": "WARN/NO_TRADE monitor only, no return optimization",
        "no_extra_strategy_ids": True,
        "auto_promote_v5": False,
        "default": KEEP_BASELINE_V4,
        "tracker_writes_timestamp": False,
        "post_hoc_must_be_labeled": True,
    }
    dump_json(CONTROLS_JSON, ctx)
    write_md(
        CONTROLS_MD,
        "\n".join([
            "# Round 12 · 操作开关",
            "",
            "- Kill switch：只监控 WARN/NO_TRADE，不优化收益。",
            "- 禁止额外预注册策略 ID。",
            f"- 默认 **{KEEP_BASELINE_V4}**，auto_promote_v5=false。",
            "- 追踪表不写「写入时间」。",
            "",
        ]),
    )
    return ctx


def _append_tracker(ctx: dict[str, Any]) -> None:
    text = TRACKER_PATH.read_text(encoding="utf-8")
    if "\n## Round 12\n" in text or text.endswith("## Round 12\n"):
        return
    dsr = ctx.get("dsr") or {}
    pbo = ctx.get("pbo") or {}
    plc = ctx.get("placebo") or {}
    uw = (pbo.get("by_universe") or {}).get("U_WINDOW") or {}
    r01 = ((ctx.get("parity") or {}).get("anchors") or {}).get("R12_01") or {}
    obs = r01.get("observed") or {}
    r06 = ((ctx.get("deploy") or {}).get("results") or {}).get(R12_06) or {}
    block = [
        "",
        "## Round 12",
        "",
        "Risk Remediation & Deployability。不是 Alpha 搜索。不升 `baseline_v5`。正式产物只写 `result/round12/`。",
        f"verdict = `{ctx.get('verdict')}`。promotion = `{KEEP_BASELINE_V4}`。",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| R12_01 CAGR/Sharpe/MaxDD | {pct(obs.get('cagr'))} / {obs.get('sharpe')} / {pct(obs.get('max_drawdown'))} |",
        f"| R12_06 CAGR/Sharpe/MaxDD | {pct(r06.get('cagr'))} / {r06.get('sharpe')} / {pct(r06.get('max_drawdown'))} |",
        f"| survivorship | {((ctx.get('universe') or {}).get('audit') or {}).get('verdict')} |",
        f"| execution class | {(ctx.get('execution') or {}).get('classification')} |",
        f"| EXEC_P50/P75 | {((ctx.get('execution') or {}).get('exec') or {}).get('P50')} / {((ctx.get('execution') or {}).get('exec') or {}).get('P75')} |",
        f"| capacity | {(ctx.get('capacity') or {}).get('rating')} |",
        f"| CANONICAL_DSR | {dsr.get('CANONICAL_DSR')} (N={dsr.get('canonical_n_choice')}) |",
        f"| TRUE PBO U_WINDOW / U_PROMO | {uw.get('pbo')} / {((pbo.get('by_universe') or {}).get('U_PROMO') or {}).get('pbo')} |",
        f"| global PBO | {pbo.get('TRUE_GLOBAL_PBO_STATUS')} |",
        f"| placebo n / CAGR pct | {plc.get('n_placebo')} / {((plc.get('percentiles') or {}).get('R12_01') or {}).get('cagr')} |",
        f"| forward OOS | {(ctx.get('shadow') or {}).get('status')} |",
        f"| manifest sha256 | `{((ctx.get('frozen') or {}).get('sha256'))}` |",
        "",
        "主报告：`result/round12/ROUND12_RISK_REMEDIATION_REPORT.md`。",
        "",
    ]
    TRACKER_PATH.write_text(text.rstrip() + "\n" + "\n".join(block), encoding="utf-8")


def run_report(**_: Any) -> dict[str, Any]:
    ctx = gather_ctx()
    qs = ctx["questions"]
    ladder = ctx["ladder"]
    body = [
        "# ROUND12_RISK_REMEDIATION_REPORT",
        "",
        "Round 12 = Risk Remediation & Deployability。**不是** Alpha 搜索。默认 **KEEP_BASELINE_V4**。",
        f"manifest sha256=`{(ctx.get('frozen') or {}).get('sha256')}`。",
        "",
        "## 01 · 使命与冻结",
        "",
        "先 freeze 再看新策略成绩。正式策略仅 8 个预注册 ID。不得自动升 v5。",
        "",
        "## 02 · Round11 风险 intake",
        "",
        "见 `00_round11_risk_intake.md`。R1 PBO 截断可修；R2 forward 不可用历史数据修；R6 DSR 口径可修。",
        "",
        "## 03 · 锚点 parity",
        "",
        f"R12_00/R12_01 = `{(ctx.get('parity') or {}).get('status')}`。R12_01 不是 v5。",
        "",
        "## 04 · PIT 宇宙",
        "",
        f"rows={(ctx.get('universe') or {}).get('n_rows')} eligible={(ctx.get('universe') or {}).get('n_eligible')}。"
        "未用当前证券列表 / membership.effective_to。",
        "",
        "## 05 · 幸存者偏差审计",
        "",
        f"终审=**{((ctx.get('universe') or {}).get('audit') or {}).get('verdict')}**。"
        f"{((ctx.get('universe') or {}).get('audit') or {}).get('reason')}",
        "",
        "## 06 · 成交盘点与因果链",
        "",
        f"`information_available_at <= T_DECISION < order_submission <= fill`。"
        "基线 = SAME_OPEN_EXECUTION_OPTIMISM。见 `execution/data_inventory.md`。",
        "",
        "## 07 · 分钟校准",
        "",
        f"分类=**{(ctx.get('execution') or {}).get('classification')}**；"
        f"EXEC={((ctx.get('execution') or {}).get('exec'))}。",
        "",
        "## 08 · 部署变体",
        "",
        "见 `04_deploy_variants.md`。",
        "",
        "## 09 · 流动性与容量",
        "",
        f"3% Top1 优先。评级=**{(ctx.get('capacity') or {}).get('rating')}**。",
        "",
        "## 10 · Alpha 留存与会计分解",
        "",
        "见 Q18/Q19。分解是会计而非因果。",
        "",
        "## 11 · Trial registry v2",
        "",
        f"documented/discovered/matrix = {(ctx.get('trials') or {}).get('n_documented')} / {(ctx.get('trials') or {}).get('n_full_discovered')} / {(ctx.get('trials') or {}).get('n_matrix')}。",
        "",
        "## 12 · TRUE PBO 2.0",
        "",
        f"U_WINDOW={(((ctx.get('pbo') or {}).get('by_universe') or {}).get('U_WINDOW') or {}).get('pbo')}；"
        f"全宇宙=`{(ctx.get('pbo') or {}).get('TRUE_GLOBAL_PBO_STATUS')}`。",
        "",
        "## 13 · Canonical DSR",
        "",
        f"CANONICAL_DSR=**{(ctx.get('dsr') or {}).get('CANONICAL_DSR')}**；单测={( (ctx.get('dsr') or {}).get('unit_tests') or {}).get('status')}。",
        "",
        "## 14 · AlphaForge",
        "",
        f"`{(ctx.get('alphaforge') or {}).get('status')}`。失败不阻断。",
        "",
        "## 15 · 全引擎 matched placebo",
        "",
        f"n={(ctx.get('placebo') or {}).get('n_placebo')}；R12_01 CAGR 百分位={(((ctx.get('placebo') or {}).get('percentiles') or {}).get('R12_01') or {}).get('cagr')}。",
        "",
        "## 16 · State-timing placebo",
        "",
        f"n_target={(ctx.get('timing') or {}).get('n_target')}。见 `12_timing_placebo.md`。",
        "",
        "## 17 · Staircase / SPA",
        "",
        f"{(ctx.get('staircase') or {}).get('classification')}",
        "",
        "## 18 · Forward / Shadow",
        "",
        f"**{(ctx.get('shadow') or {}).get('status')}**。协议已落地。",
        "",
        "## 19 · Performance ladder",
        "",
        "| 档 | CAGR | Sharpe | MaxDD |",
        "|---|---:|---:|---:|",
        *[f"| {r['rung']} | {pct(r.get('cagr'))} | {r.get('sharpe')} | {pct(r.get('max_drawdown'))} |" for r in ladder],
        "",
        "研究区间情景，不是预测。",
        "",
        "## 20 · 升 v5 裁决",
        "",
        f"**{(ctx.get('promotion') or {}).get('final')}**。passed_gate={(ctx.get('promotion') or {}).get('passed_gate')}。"
        f"{(ctx.get('promotion') or {}).get('reason')}",
        "",
        "## 21 · 六选一 verdict",
        "",
        f"**{ctx.get('verdict')}**",
        "",
        "候选：" + " / ".join(VERDICTS) + "。",
        "",
        "## 22 · 风险维度",
        "",
        "| 维度 | 级别 | 说明 |",
        "|---|---|---|",
        *[f"| {r['dim']} | {r['level']} | {r['note']} |" for r in ctx.get("risks") or []],
        "",
        "不做假综合分。",
        "",
        "## 23 · Q1–Q28 与限制",
        "",
    ]
    for i in range(1, 29):
        key = f"Q{i}"
        body += [f"### {key} — {PRE_REGISTERED_QUESTIONS[key]}", "", qs.get(key, "NOT_IDENTIFIABLE"), ""]
    body += [
        "### 明确没做",
        "",
        "- 不搜新 Alpha、不升 v5、不因结果改预注册规则",
        "- 不覆盖 legacy `result/roundN/`",
        "- 不把 proxy PBO / CPCV 写成 true PBO / forward OOS",
        "- 不把可恢复子集外推到 1339/2678",
        "- 追踪表不写「写入时间」",
        "",
    ]
    write_md(REPORT_MD, "\n".join(body))
    _append_tracker(ctx)
    r01 = ((ctx.get("parity") or {}).get("anchors") or {}).get("R12_01") or {}
    complete = {
        "verdict": ctx.get("verdict"),
        "promotion": KEEP_BASELINE_V4,
        "ladder": ladder,
        "R12_01": r01,
        "R12_06": ((ctx.get("deploy") or {}).get("results") or {}).get(R12_06),
        "survivorship": ((ctx.get("universe") or {}).get("audit") or {}).get("verdict"),
        "execution": (ctx.get("execution") or {}).get("classification"),
        "canonical_dsr": (ctx.get("dsr") or {}).get("CANONICAL_DSR"),
        "true_pbo_window": (((ctx.get("pbo") or {}).get("by_universe") or {}).get("U_WINDOW") or {}).get("pbo"),
        "global_pbo": (ctx.get("pbo") or {}).get("TRUE_GLOBAL_PBO_STATUS"),
        "placebo_percentile_cagr": (((ctx.get("placebo") or {}).get("percentiles") or {}).get("R12_01") or {}).get("cagr"),
        "forward_oos": (ctx.get("shadow") or {}).get("status"),
        "manifest_sha256": (ctx.get("frozen") or {}).get("sha256"),
        "dossier": str(REPORT_MD),
        "questions": qs,
    }
    dump_json(DOSSIER_JSON, complete)
    return complete


def print_complete_block(payload: dict[str, Any]) -> None:
    r01 = ((payload.get("R12_01") or {}).get("observed") or payload.get("R12_01") or {})
    r06 = payload.get("R12_06") or {}
    print("========== ROUND12 COMPLETE ==========", flush=True)
    print(f"VERDICT={payload.get('verdict')}", flush=True)
    print(f"PROMOTION={payload.get('promotion')}", flush=True)
    print(
        f"R12_01 CAGR={r01.get('cagr')} Sharpe={r01.get('sharpe')} MaxDD={r01.get('max_drawdown')}",
        flush=True,
    )
    print(
        f"R12_06 CAGR={r06.get('cagr')} Sharpe={r06.get('sharpe')} MaxDD={r06.get('max_drawdown')}",
        flush=True,
    )
    print(
        f"SURVIVORSHIP={payload.get('survivorship')} EXECUTION={payload.get('execution')} "
        f"CANONICAL_DSR={payload.get('canonical_dsr')} "
        f"TRUE_PBO_WINDOW={payload.get('true_pbo_window')} GLOBAL_PBO={payload.get('global_pbo')}",
        flush=True,
    )
    print(
        f"PLACEBO_CAGR_PCT={payload.get('placebo_percentile_cagr')} "
        f"FORWARD_OOS={payload.get('forward_oos')}",
        flush=True,
    )
    print(f"MANIFEST={payload.get('manifest_sha256')}", flush=True)
    print(f"DOSSIER={payload.get('dossier')}", flush=True)
    print("======================================", flush=True)


__all__ = ["gather_ctx", "print_complete_block", "run_controls", "run_report"]
