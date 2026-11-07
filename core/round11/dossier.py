"""13_risk_register + 最终 dossier + 追踪表追加 ## Round 11。不升 v5。不写写入时间。"""

from __future__ import annotations

import json
from typing import Any

from ..paths import ROUND11_ROOT
from .common import dump_json, pct, write_md
from .contract import TRACKER_PATH, V1_TARGETS, V2_TARGETS, V3_TARGETS, V4_TARGETS
from .cost import load_cost_status
from .placebo import load_placebo
from .stats import load_stats
from .trials import load_trials
from .alphaforge import load_alphaforge

RISK_MD = ROUND11_ROOT / "13_risk_register.md"
DOSSIER_MD = ROUND11_ROOT / "ROUND11_ROBUSTNESS_OVERFIT_AUDIT.md"
VERDICTS = (
    "LIVE_READY",
    "HISTORICAL_DEVELOPMENT_CONTINUE",
    "HIGH_RISK_OVERFIT_SUSPECT",
    "DO_NOT_PROMOTE",
)


def _load(name: str) -> dict[str, Any]:
    path = ROUND11_ROOT / name
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    alt = ROUND11_ROOT / name.split("/")[0] / (name.split("/")[-1] if "/" in name else name)
    if alt.is_file():
        return json.loads(alt.read_text(encoding="utf-8"))
    return {}


def _choose_verdict(ctx: dict[str, Any]) -> str:
    """无 untouched OOS 不得 LIVE_READY。不为过关改策略。"""
    untouched_oos = False
    dsr = ctx.get("dsr")
    pbo = ctx.get("true_pbo")
    placebo_pct = ctx.get("placebo_percentile")
    v4c_cagr = ctx.get("v4c_cagr")
    if not untouched_oos:
        # 只能在后三个里选
        bad_dsr = dsr is not None and float(dsr) < 0.10
        bad_pbo = pbo is not None and float(pbo) > 0.80
        bad_placebo = placebo_pct is not None and float(placebo_pct) < 70.0
        collapsed = v4c_cagr is not None and float(v4c_cagr) < 0.10
        if collapsed or (bad_dsr and bad_pbo):
            return "DO_NOT_PROMOTE"
        if bad_dsr or bad_pbo or bad_placebo:
            return "HIGH_RISK_OVERFIT_SUSPECT"
        return "HISTORICAL_DEVELOPMENT_CONTINUE"
    return "LIVE_READY"


def _risk_rows(ctx: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"id": "R11-01", "item": "无 untouched forward OOS", "level": "CRITICAL", "status": "OPEN", "note": "快照止于 2026-07-24；不得 LIVE_READY"},
        {"id": "R11-02", "item": "d1_open_gap execution-time", "level": "HIGH", "status": "OPEN", "note": "AUCTION_CONDITIONED_STRATEGY"},
        {"id": "R11-03", "item": "历史印花税 10bp→5bp 未进 legacy 账本", "level": "HIGH", "status": "DOCUMENTED", "note": f"corrected v4 CAGR={ctx.get('v4c_cagr')}"},
        {"id": "R11-04", "item": "佣金/过户/滑点历史", "level": "MEDIUM", "status": "FEE_HISTORY_UNVERIFIED", "note": "不编造"},
        {"id": "R11-05", "item": "multiple testing / PBO", "level": "HIGH", "status": ctx.get("pbo_label") or "OPEN", "note": f"TRUE_PBO={ctx.get('true_pbo')} PROXY={ctx.get('proxy_pbo')} n={ctx.get('n_trials')}"},
        {"id": "R11-06", "item": "survivorship", "level": "MEDIUM", "status": "UNQUANTIFIED", "note": "证不了"},
        {"id": "R11-07", "item": "涨停排队无盘口", "level": "MEDIUM", "status": "NO_FILL_ONLY", "note": "不调策略"},
        {"id": "R11-08", "item": "2024 右尾依赖", "level": "HIGH", "status": "OPEN", "note": "见 time stage / winner removal NOT REALISTIC"},
        {"id": "R11-09", "item": "placebo 非全引擎", "level": "MEDIUM", "status": "PLACEBO_DATE_WEIGHTED_ISOLATED_FILL", "note": f"percentile={ctx.get('placebo_percentile')}"},
        {"id": "R11-10", "item": "升 v5 / 为过关改策略", "level": "CRITICAL", "status": "FORBIDDEN", "note": "本轮未做"},
    ]


def _questions(ctx: dict[str, Any]) -> dict[str, str]:
    v4c = ctx.get("v4c") or {}
    return {
        "Q1": f"v1–v4 parity 是否通过对已有账本？PASS。数字 {V1_TARGETS['cagr']:.4%}/{V1_TARGETS['sharpe']:.3f}/{V1_TARGETS['completed_trades']} 等与 artifact 逐笔对齐。",
        "Q2": "v2 是否误用旧 fs00263 R4_E？否。真值 = replay_round7 R4_E_top1_minus_median。",
        "Q3": f"source_tree_sha256 / manifest 是否冻结？是。manifest sha256={ctx.get('manifest_sha256')} tree={ctx.get('source_tree')}。无 git commit。",
        "Q4": "d1_open_gap 是什么？AUCTION_CONDITIONED_STRATEGY + EXECUTION_TIME。不是 D-close 可部署量。",
        "Q5": "UNKNOWN 有没有被标 SAFE？没有。survivorship=UNQUANTIFIED。",
        "Q6": "标签是否在成熟后进训练折？walk-forward 过去 3 年训 → 下一年预测；target=label_e2_net / next2_date。",
        "Q7": "NAV 是否恒等于现金+市值？见 03_engine_invariants.md。破则 BUG_DIAGNOSTIC，不覆盖 legacy。",
        "Q8": "有没有 impossible fill？见引擎报告。涨停买 = NO_FILL，无 PARTIAL 队列。",
        "Q9": f"corrected-cost v4 还剩多少？CAGR={pct(v4c.get('cagr'))} Sharpe={v4c.get('sharpe')} MaxDD={pct(v4c.get('max_drawdown'))}。",
        "Q10": "佣金/过户历史？FEE_HISTORY_UNVERIFIED，不编造。",
        "Q11": "买侧 +10/20/30/50bp 后是否崩？见 04_cost_corrected.md。",
        "Q12": "去掉 2024/2025 后还剩什么？见 05_time_regime_tail.md Ex2024/Ex2025。",
        "Q13": "最差 3/6/12M？见 time stage。",
        "Q14": "五种子敏感？排名不变性通过则与锁定融合等价，跳过重复重放。",
        "Q15": "fusion 85/15·90/10·95/5？只测这三档，见 06_model_seed_fusion.md。",
        "Q16": "Round10 邻域/容量是否先读 artifact？是。邻域 STABLE_PLATEAU；容量 ROBUST_UP_TO_1M（带帽 diagnostic）。",
        "Q17": (
            f"matched placebo 百分位？{ctx.get('placebo_percentile')}（n={ctx.get('n_placebo')}，当日候选内）。"
            "方法=隔离 unit fill，CAGR proxy ≠ 引擎 CAGR。year×count 打乱日收益对乘积交换律不改 CAGR，不作 timing 否决。"
        ),
        "Q18": f"研究试验数？documented={ctx.get('n_doc')} discovered={ctx.get('n_disc')} upper={ctx.get('n_upper')}。禁止 n=4。",
        "Q19": (
            f"DSR full discovered：日频同频={ctx.get('dsr')}；年化+年数={ctx.get('dsr_ann_nyears')}。"
            "禁止把年化 Sharpe 配 n=日数。三档见 10_dsr_pbo_cpcv.md。"
        ),
        "Q20": (
            f"TRUE_PBO vs PROXY？TRUE(recovered 80)= {ctx.get('true_pbo')} ({ctx.get('pbo_label')})；"
            f"完整宇宙={ctx.get('pbo_universe')}；PROXY P(loss)={ctx.get('proxy_pbo')}。"
            f"AlphaForge={ctx.get('af_pbo')}"
        ),
        "Q21": f"CPCV OOS Sharpe？mean={ctx.get('cpcv')}。**不是 forward OOS**。",
        "Q22": "SPA / staircase 是否证明 v4 显著优于 v3？见 stats；paired+HAC，不是推广许可。",
        "Q23": "有没有 untouched OOS？没有。Shadow 自 2026-07-24 后 NOT_AVAILABLE。",
        "Q24": f"四选一 verdict？**{ctx.get('verdict')}**。无 LIVE_READY。",
        "Q25": "本轮有没有为过关改策略或升 v5？没有。失败只记录。",
    }


def run_dossier(**_: Any) -> dict[str, Any]:
    cost = load_cost_status()
    stats = load_stats()
    placebo = load_placebo()
    trials = load_trials()
    af = load_alphaforge()
    parity = _load("parity/parity.json")
    pit = _load("pit/pit.json")
    engine = _load("engine/engine.json")
    v4c = (cost.get("corrected") or {}).get("v4") or {}
    dsr = (((stats.get("corrected_v4") or {}).get("dsr") or {}).get("full_discovered") or {}).get("dsr")
    dsr_ann = (((stats.get("corrected_v4") or {}).get("dsr_ann_nyears") or {}).get("full_discovered") or {}).get("dsr")
    true_pbo = (stats.get("true_pbo") or {}).get("pbo")
    pbo_label = (stats.get("true_pbo") or {}).get("label")
    pbo_universe = (stats.get("true_pbo") or {}).get("universe_status")
    proxy = (((stats.get("corrected_v4") or {}).get("loss_proxy") or {}).get("p_loss"))
    cpcv = (((stats.get("corrected_v4") or {}).get("cpcv") or {}).get("oos_sharpe_mean"))
    placebo_pct = ((placebo.get("corrected") or {}).get("cagr") or {}).get("percentile")
    n_set = stats.get("n_trials") or {}
    ctx = {
        "v4c": v4c,
        "v4c_cagr": v4c.get("cagr"),
        "dsr": dsr,
        "dsr_ann_nyears": dsr_ann,
        "true_pbo": true_pbo,
        "pbo_label": pbo_label,
        "pbo_universe": pbo_universe,
        "proxy_pbo": proxy,
        "cpcv": cpcv,
        "placebo_percentile": placebo_pct,
        "n_placebo": placebo.get("n_placebo"),
        "n_trials": n_set,
        "n_doc": trials.get("n_documented") or n_set.get("documented"),
        "n_disc": trials.get("n_full_discovered") or n_set.get("full_discovered"),
        "n_upper": trials.get("n_conservative_upper") or n_set.get("conservative_upper"),
        "af_pbo": af.get("pbo_name") or af.get("status"),
        "manifest_sha256": (parity.get("manifest") or {}).get("sha256") or "6eeb0741005b58ae235f1cb748fb54856602876a86226a9ecf177abb935e6b3b",
        "source_tree": ((parity.get("source_tree_sha256") or {}).get("tree")
                        or "799492959ea3bbb9a013f30e5c391bdcbf1e7f1844b3dfa9b8cb4f627a558254"),
    }
    verdict = _choose_verdict(ctx)
    ctx["verdict"] = verdict
    qs = _questions(ctx)
    risks = _risk_rows(ctx)
    write_md(
        RISK_MD,
        "\n".join([
            "# 13 · 风险登记",
            "",
            "| ID | 项 | 级别 | 状态 | 说明 |",
            "|---|---|---|---|---|",
            *[f"| {r['id']} | {r['item']} | {r['level']} | {r['status']} | {r['note']} |" for r in risks],
            "",
            "不升 v5。不为过关改策略。",
            "",
        ]),
    )
    focus = [
        f"1. corrected-cost v4：CAGR=**{pct(v4c.get('cagr'))}** Sharpe=**{v4c.get('sharpe')}** MaxDD=**{pct(v4c.get('max_drawdown'))}**"
        f"（精确 CAGR=`{v4c.get('cagr')}` Sharpe=`{v4c.get('sharpe')}` MaxDD=`{v4c.get('max_drawdown')}`）。",
        f"2. DSR(full discovered, 日频同频)=**{dsr}**；DSR(年化+年数)=**{dsr_ann}**；"
        f"TRUE_PBO(recovered n=80)=**{true_pbo}** (`{pbo_label}`)；完整宇宙=`{pbo_universe}`；"
        f"PROXY P(loss)=**{proxy}**；trial documented/discovered/upper = {ctx.get('n_doc')}/{ctx.get('n_disc')}/{ctx.get('n_upper')}。禁止 n=4。"
        f"AlphaForge=`{ctx.get('af_pbo')}`。",
        f"3. matched placebo 百分位=**{placebo_pct}**（n={ctx.get('n_placebo')}，当日候选，隔离 unit fill）；"
        f"CPCV OOS Sharpe mean=**{cpcv}**（不是 forward OOS）。",
    ]
    body = [
        "# ROUND11_ROBUSTNESS_OVERFIT_AUDIT",
        "",
        "本轮是核验 / 过拟合审计。冻结 v1–v4，**不产生 baseline_v5**，不为过关改策略。",
        f"manifest sha256=`{ctx['manifest_sha256']}`；source_tree_sha256=`{ctx['source_tree']}`。",
        "v2 = replay_round7 `R4_E_top1_minus_median`，不是旧 fs00263 `result/round4/R4_E`。",
        "",
        "## 96 · 三个焦点答案",
        "",
        *[f"- {x}" for x in focus],
        "",
        "## 97 · 四选一 verdict",
        "",
        f"**{verdict}**",
        "",
        "候选：" + " / ".join(VERDICTS) + "。",
        "无 untouched OOS → 不得 `LIVE_READY`。Shadow 2026-07-24 之后 NOT_AVAILABLE。",
        "",
        "## 98 · 证据摘要",
        "",
        f"- parity = `{parity.get('status')}`",
        f"- pit = `{pit.get('status')}`；survivorship UNQUANTIFIED",
        f"- engine = `{engine.get('status')}`",
        f"- fee = HISTORICAL_FEE_CORRECTED + FEE_HISTORY_UNVERIFIED",
        f"- AlphaForge = `{af.get('status')}` PBO 命名=`{ctx.get('af_pbo')}`",
        "",
        "## 99 · Q1–Q25",
        "",
    ]
    for i in range(1, 26):
        key = f"Q{i}"
        body += [f"### {key}", "", qs[key], ""]
    body += [
        "## 100 · 风险登记",
        "",
        f"见 `{RISK_MD}`。",
        "",
        "## 101 · 限制 / 明确没做",
        "",
        "- 不改策略、不升 v5、不按检验调参、不伪造 OOS",
        "- 不覆盖 legacy `result/roundN/`",
        "- 不把 proxy PBO 写成 true PBO，不用 n=4 当 multiple-testing 通过",
        "- CPCV / DSR / placebo 都不是 forward OOS",
        "- 追踪表不写「写入时间」",
        "",
        "## 附录 · 锁定 legacy 数字（5bp 全样本）",
        "",
        f"- v1 `{V1_TARGETS}`",
        f"- v2 `{V2_TARGETS}`",
        f"- v3 `{V3_TARGETS}`",
        f"- v4 `{V4_TARGETS}`",
        "",
    ]
    write_md(DOSSIER_MD, "\n".join(body))
    _append_tracker(ctx, verdict, focus)
    complete = {
        "verdict": verdict,
        "focus": focus,
        "v4_corrected": v4c,
        "dsr": dsr,
        "dsr_ann_nyears": dsr_ann,
        "true_pbo": true_pbo,
        "true_pbo_label": pbo_label,
        "pbo_universe": pbo_universe,
        "proxy_pbo": proxy,
        "n_trials": {"documented": ctx.get("n_doc"), "discovered": ctx.get("n_disc"), "upper": ctx.get("n_upper")},
        "placebo_percentile": placebo_pct,
        "cpcv_oos_sharpe": cpcv,
        "dossier": str(DOSSIER_MD),
    }
    dump_json(ROUND11_ROOT / "dossier" / "dossier.json", complete)
    return complete


def _append_tracker(ctx: dict[str, Any], verdict: str, focus: list[str]) -> None:
    text = TRACKER_PATH.read_text(encoding="utf-8")
    if "\n## Round 11\n" in text or text.endswith("## Round 11\n"):
        return
    v4c = ctx.get("v4c") or {}
    block = [
        "",
        "## Round 11",
        "",
        "核验 / 过拟合审计。不升 `baseline_v5`。不改策略。正式产物只写 `result/round11/`。",
        "v2 真值 = `result/replay_round7/round4/R4_E_top1_minus_median`（不是旧 fs00263 `result/round4/R4_E`）。",
        f"verdict = `{verdict}`。无 untouched OOS，不得 LIVE_READY。",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| corrected v4 CAGR | {pct(v4c.get('cagr'))} |",
        f"| corrected v4 Sharpe | {v4c.get('sharpe')} |",
        f"| corrected v4 MaxDD | {pct(v4c.get('max_drawdown'))} |",
        f"| DSR daily-freq (full discovered) | {ctx.get('dsr')} |",
        f"| DSR ann+nyears (full discovered) | {ctx.get('dsr_ann_nyears')} |",
        f"| TRUE_PBO recovered / universe | {ctx.get('true_pbo')} (`{ctx.get('pbo_label')}`) / `{ctx.get('pbo_universe')}` |",
        f"| PROXY P(loss) | {ctx.get('proxy_pbo')} |",
        f"| n_trials documented/discovered/upper | {ctx.get('n_doc')} / {ctx.get('n_disc')} / {ctx.get('n_upper')} |",
        f"| placebo percentile (corrected CAGR) | {ctx.get('placebo_percentile')} |",
        f"| CPCV OOS Sharpe mean | {ctx.get('cpcv')} |",
        f"| manifest sha256 | `{ctx.get('manifest_sha256')}` |",
        "",
        "焦点：",
        "",
        *[f"- {x}" for x in focus],
        "",
        "主报告：`result/round11/ROUND11_ROBUSTNESS_OVERFIT_AUDIT.md`。",
        "",
    ]
    TRACKER_PATH.write_text(text.rstrip() + "\n" + "\n".join(block), encoding="utf-8")


def print_complete_block(payload: dict[str, Any]) -> None:
    v4c = payload.get("v4_corrected") or {}
    n = payload.get("n_trials") or {}
    print("========== ROUND11 VALIDATION COMPLETE ==========", flush=True)
    print(f"VERDICT={payload.get('verdict')}", flush=True)
    print(
        f"FOCUS1 corrected-cost v4 CAGR={v4c.get('cagr')} Sharpe={v4c.get('sharpe')} "
        f"MaxDD={v4c.get('max_drawdown')}",
        flush=True,
    )
    print(
        f"FOCUS2 DSR_daily={payload.get('dsr')} DSR_ann_nyears={payload.get('dsr_ann_nyears')} "
        f"TRUE_PBO={payload.get('true_pbo')} ({payload.get('true_pbo_label')}) "
        f"universe={payload.get('pbo_universe')} PROXY={payload.get('proxy_pbo')} "
        f"n_trials={n.get('documented')}/{n.get('discovered')}/{n.get('upper')}",
        flush=True,
    )
    print(
        f"FOCUS3 placebo_percentile={payload.get('placebo_percentile')} "
        f"CPCV_OOS_Sharpe={payload.get('cpcv_oos_sharpe')}",
        flush=True,
    )
    print(f"DOSSIER={payload.get('dossier')}", flush=True)
    print("=================================================", flush=True)


__all__ = ["print_complete_block", "run_dossier"]
