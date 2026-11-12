"""主报告 1–29 + Q1–Q34、追踪表追加、PART 49。不写「写入时间」。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND17_ROOT
from .common import dump_json, load_json, pct, write_md
from .contract import (
    AWAITING_USER_LOCK,
    CURRENT_BASELINE,
    FINAL_REPORT,
    KEEP_BASELINE_V4,
    N_TRIALS,
    OPTION_CANDIDATE,
    PARENT_V4,
    TRACKER_PATH,
    V4_ANCHOR,
)
from .decision import run_decision
from .figures import make_figures
from .freeze import require_frozen

DOSSIER_JSON = ROUND17_ROOT / "stats" / "dossier.json"


def gather_ctx() -> dict[str, Any]:
    return {
        "parity": load_json(ROUND17_ROOT / "parity" / "parity.json"),
        "preflight": load_json(ROUND17_ROOT / "preflight" / "preflight.json"),
        "inventory": load_json(ROUND17_ROOT / "inventory" / "existing.json"),
        "factors": load_json(ROUND17_ROOT / "factors" / "factor_build.json"),
        "pit": load_json(ROUND17_ROOT / "pit" / "pit.json"),
        "frozen": load_json(ROUND17_ROOT / "configs" / "freeze.json"),
        "search": load_json(ROUND17_ROOT / "search" / "search_status.json"),
        "top100": load_json(ROUND17_ROOT / "select" / "top100.json"),
        "top12": load_json(ROUND17_ROOT / "select" / "top12.json"),
        "four": load_json(ROUND17_ROOT / "fourway" / "fourway.json"),
        "attr": load_json(ROUND17_ROOT / "attribution" / "increments.json"),
        "stress": load_json(ROUND17_ROOT / "stress" / "stress.json"),
        "capacity": load_json(ROUND17_ROOT / "capacity" / "capacity.json"),
        "bootstrap": load_json(ROUND17_ROOT / "bootstrap" / "bootstrap.json"),
        "spa": load_json(ROUND17_ROOT / "spa" / "spa.json"),
        "pbo": load_json(ROUND17_ROOT / "pbo" / "pbo.json"),
        "fwd": load_json(ROUND17_ROOT / "forward" / "forward.json"),
        "promo": load_json(ROUND17_ROOT / "promotion" / "promotion.json"),
        "budget": load_json(ROUND17_ROOT / "configs" / "compute_budget.json"),
    }


def _q(ctx: dict[str, Any]) -> list[str]:
    par = ctx.get("parity") or {}
    pre = ctx.get("preflight") or {}
    pit = ctx.get("pit") or {}
    search = ctx.get("search") or {}
    frozen = ctx.get("frozen") or {}
    four = ctx.get("four") or {}
    attr = ctx.get("attr") or {}
    spa = ctx.get("spa") or {}
    pbo = ctx.get("pbo") or {}
    fwd = ctx.get("fwd") or {}
    promo = ctx.get("promo") or {}
    top12 = ctx.get("top12") or {}
    a00 = four.get("A00") or {}
    a01 = four.get("A01") or {}
    best = promo.get("best_b02") or {}
    b02 = best.get("B02") or {}
    rows = attr.get("rows") or []
    best_id = best.get("B02_id")
    best_inc = next((r for r in rows if r.get("B02_id") == best_id), rows[0] if rows else {})
    qs = [
        ("Q1", "CURRENT 是什么？", f"{CURRENT_BASELINE} = {PARENT_V4}"),
        ("Q2", "精确 v4 数字？", f"CAGR={V4_ANCHOR['cagr']} Sharpe={V4_ANCHOR['sharpe']} MaxDD={V4_ANCHOR['max_drawdown']}"),
        ("Q3", "A00 parity？", str(par.get("status"))),
        ("Q4", "孤立 0.8/0.8 契约？", "R17_STOCHASTIC_BAGGING_08_08 仅 core/round17 + result/round17"),
        ("Q5", "是否改 AGENTS.md / round7 / Stock？", "否"),
        ("Q6", "Seed diversity？", str(pre.get("status"))),
        ("Q7", "若退化，技术原因？", str(pre.get("technical_reason"))),
        ("Q8", "PIT 有效新因子数？", str(pit.get("n_valid"))),
        ("Q9", "分钟新特征是否进 PRIMARY？", "否；未新采集分钟。已有面板分钟列覆盖 2013–2026 仅作变换输入。"),
        ("Q10", "缺失是否填 0？", "否"),
        ("Q11", "是否重放 44 子集搜索？", "否，锁定 baseline + 1–4 新因子"),
        ("Q12", "预算？", "TEN_ONLY=3500 FIVE_ONLY=1000 BOTH=500"),
        ("Q13", "发现窗？", "2018–2021 only"),
        ("Q14", "manifest sha256？", str(frozen.get("sha256"))),
        ("Q15", "选择公式？", "0.40*rankpct(dev_sharpe)+0.25*rankpct(dev_median_year)+0.20*rankpct(dev_worst_year)+0.15*rankpct(dev_OOS_NDCG)"),
        ("Q16", "trials planned/completed/failed？", f"{N_TRIALS}/{search.get('n_completed')}/{search.get('n_failed')}"),
        ("Q17", "Top100 规则？", "仅 valid discovery + MaxDD 门 + 冻结公式；并列 trial_id 升序"),
        ("Q18", "2022–2023 是否独立 OOS？", "否，HISTORICAL_VALIDATION_SLICE"),
        ("Q19", "Top12 IDs？", str(top12.get("ids"))),
        ("Q20", "全历史标签？", "HISTORICAL_RETROSPECTIVE_AUDIT 不是 UNTOUCHED_OOS"),
        ("Q21", "A00 CAGR/Sharpe/MaxDD？", f"{a00.get('cagr')} / {a00.get('sharpe')} / {a00.get('max_drawdown')}"),
        ("Q22", "A01 CAGR/Sharpe/MaxDD？", f"{a01.get('cagr')} / {a01.get('sharpe')} / {a01.get('max_drawdown')}"),
        ("Q23", "best B02 id？", str(best.get("B02_id"))),
        ("Q24", "best B02 CAGR/Sharpe/MaxDD？", f"{b02.get('cagr')} / {b02.get('sharpe')} / {b02.get('max_drawdown')}"),
        ("Q25", "Factor increment Sharpe？", str(best_inc.get("factor_sharpe"))),
        ("Q26", "Ensemble increment Sharpe？", str(best_inc.get("ensemble_sharpe"))),
        ("Q27", "Interaction Sharpe？", str(best_inc.get("interaction_sharpe"))),
        ("Q28", "选择偏差 / SPA？", f"n_series={spa.get('n_series')} spa={spa.get('spa')} pbo={pbo.get('pbo')}"),
        ("Q29", "全部 5000 进入会计？", "是，不只 Top12"),
        ("Q30", "+20/+30bp 翻符号？", str((ctx.get("stress") or {}).get("sign_flip"))),
        ("Q31", "Forward OOS？", str(fwd.get("status"))),
        ("Q32", "期权 overlay 是否融合？", f"否。{OPTION_CANDIDATE} 未改。"),
        ("Q33", "四选一？", str(promo.get("four_way"))),
        ("Q34", "CURRENT 是否自动升 v5？", f"否。final_current={promo.get('final_current')} lock={promo.get('lock')}"),
    ]
    lines = ["## 29. Q1–Q34", ""]
    for code, q, a in qs:
        lines += [f"**{code}. {q}**", "", str(a), ""]
    return lines


def run_report(**_: Any) -> dict[str, Any]:
    require_frozen()
    ctx = gather_ctx()
    promo = ctx.get("promo") or run_decision()
    ctx["promo"] = promo
    figs = make_figures()
    ctx["figures"] = figs
    par = ctx.get("parity") or {}
    pre = ctx.get("preflight") or {}
    search = ctx.get("search") or {}
    four = ctx.get("four") or {}
    pit = ctx.get("pit") or {}
    frozen = ctx.get("frozen") or {}
    a00 = four.get("A00") or {}
    a01 = four.get("A01") or {}
    lines = [
        "# ROUND17_LARGE_SCALE_FACTOR_SEARCH_REPORT",
        "",
        "Round17 = 增量因子搜索 × 真 0.8/0.8 五 Seed。孤立命名空间。不自动升 CURRENT。不融合 R16_O01。",
        f"manifest sha256=`{frozen.get('sha256')}` parity=`{par.get('status')}` diversity=`{pre.get('status')}`",
        f"四选一=**{promo.get('four_way')}** CURRENT=**{promo.get('final_current')}**",
        "",
        "## 1. Executive Verdict",
        "",
        f"**{promo.get('four_way')}**。CURRENT 仍是 `baseline_v4 = R9_B03`。未自动晋升。未融合 `{OPTION_CANDIDATE}`。",
        "",
        f"- Parity A00 精确对齐 v4：CAGR=`{a00.get('cagr')}` Sharpe=`{a00.get('sharpe')}` MaxDD=`{a00.get('max_drawdown')}`。",
        f"- 孤立 0.8/0.8 多样性 `{pre.get('status')}`（五 Seed 预测 hash 互异；不得把 A01 运气当 alpha）。",
        f"- A01（原特征 bagging only）Sharpe=`{a01.get('sharpe')}` MaxDD=`{a01.get('max_drawdown')}`，全历史弱于 A00。",
        f"- 搜索 `{search.get('n_completed')}/{N_TRIALS}` SUCCESS=`{search.get('n_success')}` FAILED=`{search.get('n_failed')}`，未静默缩样本。",
        f"- 晋升失败门：`{promo.get('failed_gates')}`。best B02=`{(promo.get('best_b02') or {}).get('B02_id')}` Sharpe=`{((promo.get('best_b02') or {}).get('B02') or {}).get('sharpe')}`，不竞争过 A00 且 MaxDD 更差。",
        "- 仅 R17_F12_B01（新因子 + 1.0/1.0）Sharpe 略高于 A00，但联合 0.8/0.8（B02）全面落后；这是 5000 搜索后代，不是独立 OOS。",
        "",
        "## 2. Research Question",
        "",
        "在锁定 baseline_v4 特征与 90/10+B03 的前提下，加入 1–4 个 PIT 新因子，配合真 0.8/0.8 五 Seed，能否产生可晋升的 candidate_baseline_v5？",
        "",
        "## 3. Isolated bagging exception and R6_A04",
        "",
        "仅 core/round17 + result/round17 使用 R17_STOCHASTIC_BAGGING_08_08。",
        "Round6 forensic R6_A04 已证明「仅改回 0.8/0.8」可从随机性抬 CAGR。A01 运气不是晋升理由。",
        "未修改 AGENTS.md、no-xgb-subsample 规则、core/round7、Stock 已训产物。",
        "",
        "## 4. Prior contracts",
        "",
        f"CURRENT=`{CURRENT_BASELINE}` parent=`{PARENT_V4}` option=`{OPTION_CANDIDATE}`（本轮不改）。",
        "",
        "## 5. V4 parity",
        "",
        f"status=`{par.get('status')}` observed=`{(par.get('observed') or {})}`",
        "",
        "## 6. Seed diversity preflight",
        "",
        f"status=`{pre.get('status')}` reason=`{pre.get('technical_reason')}`",
        "",
        "## 7. Existing factor inventory",
        "",
        f"`{(ctx.get('inventory') or {}).get('n_five')}` five / `{(ctx.get('inventory') or {}).get('n_ten')}` ten locked. unused44=`{(ctx.get('inventory') or {}).get('n_unused44')}` 不作为子集搜索。",
        "",
        "## 8. New families A–F",
        "",
        f"specified=`{(ctx.get('factors') or {}).get('n_factors_specified')}` filled_with_zero=`{(ctx.get('factors') or {}).get('filled_with_zero')}`",
        "",
        "## 9. PIT gates",
        "",
        f"valid=`{pit.get('n_valid')}` skipped=`{pit.get('n_skipped')}`",
        "",
        "## 10. Frozen 5000 design",
        "",
        "TEN_ONLY=3500 FIVE_ONLY=1000 BOTH=500。每 trial = 锁定 baseline + 1–4 新因子。unique feature_set_hash。",
        "",
        "## 11. Selection formula (frozen before results)",
        "",
        "selection_score = 0.40*rankpct(dev_sharpe)+0.25*rankpct(dev_median_year)+0.20*rankpct(dev_worst_year)+0.15*rankpct(dev_OOS_NDCG)；并列 trial_id 升序。",
        "",
        "## 12. Promotion gates (frozen)",
        "",
        str(promo.get("failed_gates")),
        "",
        "## 13. Compute budget",
        "",
        str(ctx.get("budget")),
        "",
        "## 14. Search execution",
        "",
        f"`{search}`",
        "",
        "## 15. Discovery 2018–2021",
        "",
        "排序只用该窗。失败指标为 null。",
        "",
        "## 16. Top100",
        "",
        f"n=`{(ctx.get('top100') or {}).get('n_top100')}` scored=`{(ctx.get('top100') or {}).get('n_scored')}`。仅 2018–2021。",
        f"A00 discovery Sharpe=`{((ctx.get('top100') or {}).get('a00_discovery') or {}).get('dev_sharpe')}` MaxDD=`{((ctx.get('top100') or {}).get('a00_discovery') or {}).get('dev_maxdd')}`。",
        "发现窗 A00 MaxDD 未能从 clipped NAV 算出（`None`），故 MaxDD 恶化门在排序时未生效；晋升阶段仍用全历史 A00 MaxDD=`-0.25455988069271596` 检验。前 20 见 `07_top100.md`。",
        "",
        "## 17. 2022–2023 historical validation",
        "",
        "HISTORICAL_VALIDATION_SLICE，**不是**独立 OOS。Top100 全部再训再评后冻结 Top12。",
        "",
        "## 18. Top12 freeze",
        "",
        f"ids=`{(ctx.get('top12') or {}).get('ids')}`。锁死于查看 2024–2026 之前。之后不改 ID。12 个 finalist 均为 FIVE_ONLY。",
        "",
        "## 19. Full-history retrospective audit",
        "",
        "2018–2026-07-24 = `HISTORICAL_RETROSPECTIVE_AUDIT`，不是 UNTOUCHED_OOS。数字来自四路全历史账本。看完后未改 Top12。",
        "",
        "## 20. Four-way",
        "",
        "| code | role | CAGR | Sharpe | MaxDD |",
        "|---|---|---|---|---|",
        f"| R17_A00 | original 1.0/1.0 = v4 | {a00.get('cagr')} | {a00.get('sharpe')} | {a00.get('max_drawdown')} |",
        f"| R17_A01 | original 0.8/0.8 bagging | {a01.get('cagr')} | {a01.get('sharpe')} | {a01.get('max_drawdown')} |",
        *[
            f"| {item.get('B01_id')} / {item.get('B02_id')} | `{item.get('source_trial')}` {item.get('sleeve')} | "
            f"{(item.get('B01') or {}).get('cagr')} / {(item.get('B02') or {}).get('cagr')} | "
            f"{(item.get('B01') or {}).get('sharpe')} / {(item.get('B02') or {}).get('sharpe')} | "
            f"{(item.get('B01') or {}).get('max_drawdown')} / {(item.get('B02') or {}).get('max_drawdown')} |"
            for item in (four.get("finalists") or [])
        ],
        "",
        "这些是 5000 搜索的后代，不是 24 个独立 OOS 假设。不重调 B03。",
        "",
        "## 21. Attribution",
        "",
        "Factor=B01−A00；Ensemble=A01−A00；Joint=B02−A00；Interaction=B02−B01−A01+A00。",
        "A01 Ensemble Sharpe 增量 = `-0.3657`（bagging 伤害全历史）。12 个 Joint Sharpe 增量全部为负。",
        "唯一正 Factor Sharpe：R17_F12_B01 − A00 = `+0.0145`（R17_T4033）。best B02=R17_F06_B02 Joint Sharpe = `-0.2277`。",
        "详见 `11_attribution.md`。",
        "",
        "## 22. Finalist diversity",
        "",
        str(load_json(ROUND17_ROOT / "diversity" / "finalist_diversity.json")),
        "",
        "## 23. Cost / capacity / year robustness",
        "",
        f"stress sign_flip=`{(ctx.get('stress') or {}).get('sign_flip')}` capacity=`{ctx.get('capacity')}`",
        "",
        "## 24. Bootstrap / CPCV / SPA / PBO",
        "",
        f"bootstrap=`{ctx.get('bootstrap')}` spa=`{ctx.get('spa')}` pbo=`{ctx.get('pbo')}`",
        "CPCV 不是 forward OOS。全部 SUCCESS 进入选择偏差会计。",
        "",
        "## 25. Forward OOS",
        "",
        str((ctx.get("fwd") or {}).get("status")),
        "",
        "## 26. Option overlay not combined",
        "",
        f"{OPTION_CANDIDATE} 本轮不改、不并入新因子模型。",
        "",
        "## 27. Baseline decision",
        "",
        f"四选一=`{promo.get('four_way')}` CURRENT=`{promo.get('final_current')}` lock=`{promo.get('lock')}` auto_promote=False",
        "",
        "## 28. Limitations",
        "",
        "见 16_limitations.md。不得把 A01 运气当 alpha。不得把 2018–2026 回顾写成独立确认。",
        "",
        "## 29. Figures",
        "",
        f"written=`{(figs or {}).get('n_written')}` skipped=`{(figs or {}).get('n_skipped')}` faked=False。缺数据不造假。",
        "",
    ]
    lines += _q(ctx)
    write_md(FINAL_REPORT, "\n".join(lines))
    _append_tracker(ctx, promo)
    complete = {
        "four_way": promo.get("four_way"),
        "final_current": promo.get("final_current"),
        "lock": promo.get("lock"),
        "parity": par.get("status"),
        "diversity": pre.get("status"),
        "n_valid_factors": pit.get("n_valid"),
        "search": search,
        "top12": (ctx.get("top12") or {}).get("ids"),
        "a00": a00,
        "a01": a01,
        "best_b02": promo.get("best_b02"),
        "attr": ctx.get("attr"),
        "spa": ctx.get("spa"),
        "forward": (ctx.get("fwd") or {}).get("status"),
        "manifest_sha256": frozen.get("sha256"),
        "dossier": str(FINAL_REPORT),
        "parent": PARENT_V4,
        "option": OPTION_CANDIDATE,
        "figures": figs,
        "promotion": promo,
    }
    dump_json(DOSSIER_JSON, complete)
    return complete


def _append_tracker(ctx: dict[str, Any], promo: dict[str, Any]) -> None:
    search = ctx.get("search") or {}
    four = ctx.get("four") or {}
    a00 = four.get("A00") or {}
    a01 = four.get("A01") or {}
    best = promo.get("best_b02") or {}
    b02 = best.get("B02") or {}
    frozen = ctx.get("frozen") or {}
    block = [
        "",
        "## Round 17 — Large-Scale Factor Discovery × True Multi-Seed Ensemble",
        "",
        "孤立 0.8/0.8 仅 core/round17 + result/round17。不自动升 CURRENT。不融合 R16_O01。",
        f"四选一 = `{promo.get('four_way')}`。CURRENT = `{promo.get('final_current')}`。lock = `{promo.get('lock')}`。",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| parity | { (ctx.get('parity') or {}).get('status') } |",
        f"| seed diversity | {(ctx.get('preflight') or {}).get('status')} |",
        f"| PIT valid new factors | {(ctx.get('pit') or {}).get('n_valid')} |",
        f"| trials planned/completed/failed | {N_TRIALS}/{(search.get('n_completed'))}/{(search.get('n_failed'))} |",
        f"| A00 CAGR/Sharpe/MaxDD | {a00.get('cagr')} / {a00.get('sharpe')} / {a00.get('max_drawdown')} |",
        f"| A01 CAGR/Sharpe/MaxDD | {a01.get('cagr')} / {a01.get('sharpe')} / {a01.get('max_drawdown')} |",
        f"| best B02 | {best.get('B02_id')} |",
        f"| best B02 CAGR/Sharpe/MaxDD | {b02.get('cagr')} / {b02.get('sharpe')} / {b02.get('max_drawdown')} |",
        f"| CURRENT | {promo.get('final_current')} |",
        f"| four_way | {promo.get('four_way')} |",
        f"| option candidate | {OPTION_CANDIDATE} untouched |",
        f"| forward OOS | {(ctx.get('fwd') or {}).get('status')} |",
        f"| lock | {promo.get('lock')} |",
        f"| manifest sha256 | `{frozen.get('sha256')}` |",
        "",
        f"主报告：`result/round17/ROUND17_LARGE_SCALE_FACTOR_SEARCH_REPORT.md`。",
        "",
    ]
    text = TRACKER_PATH.read_text(encoding="utf-8") if TRACKER_PATH.is_file() else ""
    if "## Round 17 — Large-Scale Factor Discovery" in text:
        # replace existing round17 section at end if re-run
        idx = text.rfind("## Round 17 — Large-Scale Factor Discovery")
        text = text[:idx].rstrip() + "\n" + "\n".join(block)
    else:
        text = text.rstrip() + "\n" + "\n".join(block)
    TRACKER_PATH.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def print_complete_block(payload: dict[str, Any]) -> None:
    search = payload.get("search") or {}
    print("ROUND17 COMPLETE" if int(search.get("n_completed") or 0) >= N_TRIALS else "ROUND17 PARTIAL", flush=True)
    print("", flush=True)
    print("Current baseline:", flush=True)
    print("baseline_v4 = R9_B03", flush=True)
    print("", flush=True)
    print("Parity:", flush=True)
    print(payload.get("parity") or "FAIL", flush=True)
    print("", flush=True)
    print("Seed diversity:", flush=True)
    print(payload.get("diversity"), flush=True)
    print("", flush=True)
    print("Valid new factors:", flush=True)
    print(payload.get("n_valid_factors"), flush=True)
    print("", flush=True)
    print("Trials planned/completed/failed:", flush=True)
    print(f"{N_TRIALS} / {search.get('n_completed')} / {search.get('n_failed')}", flush=True)
    print("", flush=True)
    print("A00:", flush=True)
    a00 = payload.get("a00") or {}
    print(f"CAGR={a00.get('cagr')} Sharpe={a00.get('sharpe')} MaxDD={a00.get('max_drawdown')}", flush=True)
    print("", flush=True)
    print("A01:", flush=True)
    a01 = payload.get("a01") or {}
    print(f"CAGR={a01.get('cagr')} Sharpe={a01.get('sharpe')} MaxDD={a01.get('max_drawdown')}", flush=True)
    print("", flush=True)
    print("best B02:", flush=True)
    best = payload.get("best_b02") or {}
    b02 = best.get("B02") or {}
    print(f"{best.get('B02_id')} CAGR={b02.get('cagr')} Sharpe={b02.get('sharpe')} MaxDD={b02.get('max_drawdown')}", flush=True)
    print("", flush=True)
    print("Four-way / CURRENT / lock:", flush=True)
    print(f"{payload.get('four_way')} / {payload.get('final_current')} / {payload.get('lock')}", flush=True)
    print("", flush=True)
    print("Forward OOS:", flush=True)
    print(payload.get("forward"), flush=True)
    print("", flush=True)
    print("Option candidate untouched:", flush=True)
    print(OPTION_CANDIDATE, flush=True)
    print("", flush=True)
    print("Final Report:", flush=True)
    print(str(FINAL_REPORT), flush=True)
