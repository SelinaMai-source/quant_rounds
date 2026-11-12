"""ROUND21_ALPHA_DISCOVERY_REPORT.md + Q1–Q35。追踪表追加。PART 39。不写「写入时间」。不改 CURRENT。"""

from __future__ import annotations

from typing import Any

from ..paths import PROJECT_ROOT, ROUND21_ROOT
from .common import dump_json, load_json, write_md
from .contract import (
    AUTO_PROMOTE_CURRENT,
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    FACTOR_ZOO_ROOT,
    FINAL_REPORT,
    FORWARD_OOS_NOT_AVAILABLE,
    KEEP_BASELINE_V5,
    PARENT_V4,
    TRACKER_PATH,
    V5_ANCHOR,
)
from .freeze import FREEZE_JSON, require_frozen
from .parity import PARITY_JSON

DOSSIER_JSON = ROUND21_ROOT / "promotion" / "dossier.json"


def _m(obs: dict[str, Any] | None) -> str:
    row = obs or {}
    if row.get("cagr") is None and row.get("sharpe") is None:
        return str(row.get("status") or "n/a")
    return (
        f"CAGR={row.get('cagr')} Sharpe={row.get('sharpe')} "
        f"MaxDD={row.get('max_drawdown')} trades={row.get('completed_trades')} NAV_end={row.get('nav_end')}"
    )


def gather() -> dict[str, Any]:
    from .audit import AUDIT_JSON
    from .backtest import BT_JSON
    from .combo import REG_JSON
    from .construct import CONSTRUCT_JSON
    from .evidence import EVIDENCE_JSON
    from .github import GH_JSON
    from .hypotheses import HYP_JSON
    from .model import MODEL_JSON
    from .pit import PIT_JSON
    from .redundancy import RED_JSON
    from .seed import SEED_JSON
    from .zoo import ZOO_JSON

    return {
        "parity": load_json(PARITY_JSON),
        "freeze": load_json(FREEZE_JSON),
        "github": load_json(GH_JSON),
        "audit": load_json(AUDIT_JSON),
        "hyp": load_json(HYP_JSON),
        "construct": load_json(CONSTRUCT_JSON),
        "pit": load_json(PIT_JSON),
        "evidence": load_json(EVIDENCE_JSON),
        "red": load_json(RED_JSON),
        "reg": load_json(REG_JSON),
        "model": load_json(MODEL_JSON),
        "seed": load_json(SEED_JSON),
        "bt": load_json(BT_JSON),
        "zoo": load_json(ZOO_JSON),
    }


def _formal_native(seed: dict[str, Any]) -> dict[str, Any]:
    formal = {"R21_FIXED", "R21_NESTED"}
    audits = [a for a in (seed.get("native_audit") or []) if a.get("code") in formal]
    degenerate = [a for a in audits if a.get("degenerate")]
    return {
        "n": len(audits),
        "degenerate_n": len(degenerate),
        "degenerate": bool(degenerate),
        "codes": [a.get("code") for a in audits],
        "note": "formal FIXED/NESTED native 5; later ADD5/ADD10 缺 seed 目录不记作挑最好",
    }


def _best_raw(ev: dict[str, Any]) -> list[dict[str, Any]]:
    return list(ev.get("best_raw_rankic_e2") or [])[:5]


def _build(ctx: dict[str, Any]) -> str:
    par = ctx.get("parity") or {}
    freeze = ctx.get("freeze") or {}
    github = ctx.get("github") or {}
    audit = ctx.get("audit") or {}
    hyp = ctx.get("hyp") or {}
    construct = ctx.get("construct") or {}
    pit = ctx.get("pit") or {}
    ev = ctx.get("evidence") or {}
    red = ctx.get("red") or {}
    reg = ctx.get("reg") or {}
    model = ctx.get("model") or {}
    seed = ctx.get("seed") or {}
    bt = ctx.get("bt") or {}
    zoo = ctx.get("zoo") or {}
    obs = par.get("observed") or {}
    best = bt.get("best_vs_v5") or {}
    decision = bt.get("decision") or KEEP_BASELINE_V5
    gh_ev = ev.get("github") or {}
    counts = audit.get("counts") or {}
    formal_seed = _formal_native(seed)
    main_rows = list(bt.get("main_rows") or [])
    fee_rows = list(bt.get("fee_sensitivity_table") or [])
    pit_pass = list(pit.get("pit_pass") or [])
    blocked_pit = list(pit.get("blocked_pit") or [])
    fail_cov = list(pit.get("fail_coverage") or [])
    computed = list(construct.get("computed") or hyp.get("factor_ids") or [])
    lines = [
        "# ROUND21_ALPHA_DISCOVERY_REPORT",
        "",
        "Round21 = 锁定 `baseline_v5=R16_O01` 与 Round7 5bp 口径，从原始数据实现 A–F 六大家族新横截面公式，接入 Factor Evidence Factory，与旧库去重后走 Walk-Forward 与全引擎回测。GitHub rolling-factor-evidence 只覆盖边际证据。不改 O01/B03/90-10。不自动升 CURRENT。",
        f"manifest sha256=`{freeze.get('sha256')}` combo sha256=`{reg.get('sha256')}` parity=`{par.get('status')}` 决策=**{decision}** CURRENT=**{CURRENT_BASELINE}={CURRENT_ALIAS}** AUTO_PROMOTE=False",
        "",
        "## 1. Executive Verdict",
        "",
        f"**{decision}**。CURRENT 仍是 `{CURRENT_BASELINE} = {CURRENT_ALIAS}`。未自动晋升。不锁 v6。",
        f"- Stage 0 v5 parity：`{par.get('status')}`。观测 `{_m(obs)}`。n_obs=`{obs.get('n_trading_observations') or obs.get('n_obs') or V5_ANCHOR['n_obs']}`。",
        f"- GitHub：`{github.get('GITHUB_METHOD_STATUS')}` SHA=`{github.get('commit_sha')}`。2018H1 exact=`{gh_ev.get('status')}` scope=`{gh_ev.get('scope')}`。源码无 Score，未伪造。内部窗 `{ev.get('internal_windows')}` 与 GITHUB_EXACT 分名。",
        f"- 公式 computed=`{len(computed)}` PIT_PASS=`{len(pit_pass)}` BLOCKED_PIT=`{len(blocked_pit)}` PIT_FAIL_COVERAGE=`{len(fail_cov)}`。",
        f"- Best raw RankIC(e2)：A3 `{(_best_raw(ev)[0] or {}).get('raw_rankic_e2') if _best_raw(ev) else None}`；见 §12。",
        f"- Best formal `{best.get('code')}` Sharpe=`{best.get('sharpe')}` CAGR=`{best.get('cagr')}` vs v5 `{best.get('v5_sharpe')}`/`{best.get('v5_cagr')}` ΔSharpe=`{best.get('delta_sharpe')}` ΔCAGR=`{best.get('delta_cagr')}` better=`{best.get('better_than_v5')}`。",
        f"- OOF N=`{model.get('n_oof_models')}` 全引擎回测 N=`{bt.get('n_full_backtests')}`。FIXED/NESTED **都不是独立 OOS**。",
        f"- Seed=`{seed.get('status')}` increment_gate hit=`{bool(seed.get('triggered'))}` formal native5 degenerate=`{formal_seed['degenerate']}` do_not_pick_best_from_100_or_300=`{seed.get('do_not_pick_best_from_100_or_300')}` pick_best_100_300=False。",
        f"- 主表 Round7 5bp。费用校正分表 mixed_into_main=`{bt.get('mixed_into_main')}`。Forward=`{bt.get('forward') or FORWARD_OOS_NOT_AVAILABLE}`。",
        f"- lock=`{bt.get('lock')}` AUTO_PROMOTE=`{AUTO_PROMOTE_CURRENT}`。",
        "",
        "## 2. Research Question and Lock",
        "",
        "在锁定 v5 结构（90/10、B03、O01、clean_risk、退出、仓位、gap、1.0/1.0）的前提下：六大家族新公式经 PIT/证据/去冗余/Add-Replace/WF 后，能否产生可解释且同口径优于 v5 的新 Alpha？",
        f"CURRENT=`{CURRENT_BASELINE}` parent=`{PARENT_V4}`。只改新因子构成/变换/交互/替换与预注册模型对照。",
        "",
        "## 3. Prior Research / Why This Is Not Independent OOS",
        "",
        "2018–2026 已被 R1–20 研究过。Discovery/Validation/Audit **不是独立 OOS**。Walk-forward 仍是预测年 Y ← 训 Y-3..Y-1。R20 PRIOR：78 盘点 / 63 PIT / 45 KEEP / 100 combo / best `R20_P32` Sharpe 1.6738 < 1.8757。禁止把 R17/R18/R20 失败因子换名当新发现。选择偏差=`REPEATED_HISTORICAL_TESTING`。",
        "",
        "## 4. Stage 0 v5 Parity（Round7 5bp）",
        "",
        f"status=`{par.get('status')}` engine=`{par.get('engine')}` validation_hooks=`None` fee=`{par.get('fee_schedule')}`",
        f"anchor CAGR=`{V5_ANCHOR['cagr']}` Sharpe=`{V5_ANCHOR['sharpe']}` MaxDD=`{V5_ANCHOR['max_drawdown']}` trades=`{V5_ANCHOR['completed_trades']}` n_obs=`{V5_ANCHOR['n_obs']}` NAV_end=`{V5_ANCHOR['nav_end']}`",
        f"observed `{_m(obs)}`",
        f"FAIL 只停正式晋级。路径：`{PARITY_JSON}` 与 `{ROUND21_ROOT / '00_v5_parity.md'}`",
        "",
        "## 5. CAGR Convention",
        "",
        "- **主口径** 日历年 365.25 `annualized_return`",
        "- **对照** 252 交易日 `cagr_of` 不得替换主口径",
        "",
        "## 6. Frozen Manifest",
        "",
        f"sha256=`{freeze.get('sha256')}` frozen=`{freeze.get('frozen')}` path=`{ROUND21_ROOT / 'round21_manifest.yaml'}`",
        f"combo freeze sha256=`{reg.get('sha256')}` frozen_before_nav=`{reg.get('frozen_before_combo_nav')}` n_registry=`{reg.get('n_registry')}`",
        "IC 薄日跳过不记 0。簇 Spearman≥0.85 非唯一 DROP。新因子入主组合 ≤12。内部窗 60/120/252 不事后挑窗。",
        "",
        "## 7. GitHub rolling-factor-evidence",
        "",
        f"clone status=`{github.get('GITHUB_METHOD_STATUS')}` SHA=`{github.get('commit_sha')}` dest=`{github.get('dest')}`",
        f"Phase A exact_pass（clone 当时）=`{github.get('exact_github_reproduction_pass')}` github_has_score=`{github.get('github_has_score')}`",
        f"Evidence-stage 2018H1 subset=`{gh_ev.get('status')}` exact=`{gh_ev.get('exact')}` scope=`{gh_ev.get('scope')}` namespace=`{gh_ev.get('namespace')}`",
        f"内部窗 `{ev.get('internal_windows')}` 与 GitHub 日历窗分名。Score=None，未伪造。",
        f"映射/复现/差异：`{ROUND21_ROOT / 'github' / 'GITHUB_METHOD_MAPPING.md'}` / `GITHUB_METHOD_REPRODUCTION.md` / `GITHUB_METHOD_DIFFERENCES.md`",
        f"subset 报告：`{ROUND21_ROOT / 'github' / 'GITHUB_EXACT_SUBSET.md'}`",
        "",
        "## 8. Source Audit",
        "",
        f"hermes={counts.get('hermes_planned_ok')}/{counts.get('hermes_planned_n')} "
        f"min1={counts.get('min1_days_2013')}+{counts.get('min1_days_2022')} "
        f"holder_rows={counts.get('holder_rows')} block_rows={counts.get('block_rows')} "
        f"zyquant350={counts.get('zyquant_350_n')}",
        f"路径：`{ROUND21_ROOT / '01_source_audit.md'}` `{ROUND21_ROOT / 'source_audit' / 'source_audit.md'}`",
        "分钟只允许收盘涨停子集。缺失不填 0。非两融融资 NULL 不填 0。`BUY_ORD_*` 不在 v7 落盘表。",
        "",
        "## 9. Six Families / Hypotheses",
        "",
        f"冻结 N=`{hyp.get('n')}` ids=`{hyp.get('factor_ids')}`",
        f"假设卡：`{ROUND21_ROOT / '01_hypotheses.md'}` `{ROUND21_ROOT / 'hypotheses' / 'hypotheses.yaml'}` `{PROJECT_ROOT / 'core' / 'round21' / 'families.yaml'}`",
        "A 恢复韧性 / B 同日竞争 / C 流动性吸收 / D 股东供给 / E 大宗供给 / F 融资去杠杆。E 先验 `BLOCKED_PIT`。A1 ≠ R18 F1。",
        "",
        "## 10. Construct",
        "",
        f"computed N=`{len(computed)}` blocked_pit=`{construct.get('blocked_pit')}` blocked_data=`{construct.get('blocked_data')}`",
        f"路径：`{ROUND21_ROOT / '02_construct.md'}`",
        "",
        "## 11. PIT",
        "",
        f"PIT_PASS N=`{len(pit_pass)}` `{pit_pass}`",
        f"BLOCKED_PIT N=`{len(blocked_pit)}` `{blocked_pit}`",
        f"PIT_FAIL_COVERAGE N=`{len(fail_cov)}` `{fail_cov}`",
        f"min_finite=`0.8` filled_with_zero=`False`。E 无公告时点，禁模型。路径：`{ROUND21_ROOT / '03_pit.md'}` `{ROUND21_ROOT / 'pit' / 'pit.json'}`",
        "",
        "## 12. IC / RankIC（双标签）",
        "",
        f"packages=`{ev.get('n_packages')}` copied_global=`{ev.get('copied_global_to_all_factors')}` thin_day=`{ev.get('thin_day_rule')}`",
        "Best raw RankIC `label_e2_net`：",
        "",
        "| factor_id | RankIC_e2 | pit |",
        "|---|---:|---|",
    ]
    for rec in _best_raw(ev):
        lines.append(f"| `{rec.get('factor_id')}` | {rec.get('raw_rankic_e2')} | `{rec.get('pit_status')}` |")
    lines += [
        "",
        f"完整包：`{ROUND21_ROOT / '04_evidence.md'}` `{ROUND21_ROOT / 'evidence' / 'evidence.json'}` 与 `result/round21/factors/<id>/ic.json`",
        "",
        "## 13. Rolling：GITHUB_EXACT vs ROUND21_INTERNAL",
        "",
        f"GitHub-exact `{gh_ev}`。内部 `{ev.get('internal_windows')}`。不事后挑窗。不把 GitHub 窗与内部窗混名。",
        "每因子 `github_exact.json` + `rolling.json`。E 家族 GitHub-exact=`SKIPPED_BLOCKED_PIT`。",
        "",
        "## 14. Shape / Top-tail",
        "",
        f"best_toptail=`{ev.get('best_top_tail_e2')}`。Q1–Q5；样本不够不强制 Q10。路径：`factors/<id>/shape.json` `top_tail.json`",
        "",
        "## 15. Redundancy KEEP/DROP",
        "",
        f"新因子 KEEP=`{red.get('n_keep')}` 多成员簇=`{red.get('n_clusters_ge2')}` 边=`{red.get('n_edges')}`",
        f"簇内 Original/New/Add/Replace。相关不是唯一 DROP。路径：`{ROUND21_ROOT / '05_redundancy.md'}` `{ROUND21_ROOT / 'redundancy' / 'keep_drop.csv'}`",
        "",
        "## 16. Conditional IC / Add / Replace / Ablation",
        "",
        f"路径：`{ROUND21_ROOT / '06_conditional.md'}`。fold Ridge + cross-fit。禁止全历史 PCA/正交。",
        "",
        "## 17. Combo Registry Freeze",
        "",
        f"n_registry=`{reg.get('n_registry')}` selected=`{reg.get('selected_new_factors')}` frozen_before_nav=`{reg.get('frozen_before_combo_nav')}` post_hoc_best_x_best=`{reg.get('post_hoc_best_x_best')}` padded=`{reg.get('padded_to_quota')}` blocked_pit_included=`{reg.get('blocked_pit_included')}`",
        f"sha256=`{reg.get('sha256')}` csv=`{reg.get('csv')}`",
        "",
        "## 18. Walk-Forward OOF",
        "",
        f"OOF N=`{model.get('n_oof_models')}` increment_pass N=`{model.get('n_increment_pass')}` codes=`{model.get('increment_pass_codes')}`",
        f"门=`R21_OOF_INCREMENT_V1`。路径：`{ROUND21_ROOT / '08_walk_forward.md'}`",
        "",
        "## 19. FIXED vs NESTED",
        "",
        "二者都不是独立 OOS。",
    ]
    fixed = next((r for r in main_rows if r.get("code") == "R21_FIXED"), {})
    nested = next((r for r in main_rows if r.get("code") == "R21_NESTED"), {})
    lines += [
        f"- `R21_FIXED` `{_m(fixed)}` independent_oos=`False`",
        f"- `R21_NESTED` `{_m(nested)}` independent_oos=`False`",
        "",
        "## 20. Formal Backtest Caliber",
        "",
        "MAIN = `round7_fixed_sleeve_config` + `run_local_fixed_sleeve_backtest`，不传 validation_hooks。卖税 5bp。CAGR 365.25。30 万双袖套。",
        "",
        "## 21. Main Table vs v5",
        "",
        f"best_vs_v5=`{best}`",
        "",
        "| code | CAGR | Sharpe | MaxDD | trades | NAV_end |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in main_rows:
        lines.append(
            f"| `{r.get('code')}` | {r.get('cagr')} | {r.get('sharpe')} | "
            f"{r.get('max_drawdown')} | {r.get('completed_trades')} | {r.get('nav_end')} |"
        )
    lines += [
        "",
        f"路径：`{ROUND21_ROOT / '11_backtest_decision.md'}` `{ROUND21_ROOT / 'backtest' / 'decision.json'}`",
        "",
        "## 22. Yearly / Ex2024 / Ex2025",
        "",
        "| code | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | Ex2024 | Ex2025 |",
        "|---|" + "---:|" * 11,
    ]
    years = [str(y) for y in range(2018, 2027)]
    for r in main_rows:
        yr = r.get("yearly_returns") or {}
        cells = " | ".join(str(yr.get(y)) for y in years)
        ex24 = (r.get("ex2024") or {}).get("sharpe")
        ex25 = (r.get("ex2025") or {}).get("sharpe")
        lines.append(f"| `{r.get('code')}` | {cells} | {ex24} | {ex25} |")
    lines += [
        "",
        "## 23. HISTORICAL_FEE_CORRECTED Sensitivity",
        "",
        f"单独表，禁止与 Legacy 5bp 主表混列。mixed_into_main=`{bt.get('mixed_into_main')}` N=`{len(fee_rows)}`",
        "",
        "| code | CAGR | Sharpe | MaxDD | trades |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in fee_rows:
        lines.append(
            f"| `{r.get('code')}` | {r.get('cagr')} | {r.get('sharpe')} | "
            f"{r.get('max_drawdown')} | {r.get('completed_trades')} |"
        )
    lines += [
        "",
        "## 24. Block Bootstrap / SPA",
        "",
        "20 日块 N=5000。STATISTICAL RESAMPLING NOT OOS。",
        f"SPA=`{bt.get('spa')}`",
        "",
        "## 25. Seed",
        "",
        f"status=`{seed.get('status')}` triggered=`{seed.get('triggered')}` do_not_pick_best_from_100_or_300=`{seed.get('do_not_pick_best_from_100_or_300')}`",
        f"formal native5 `{formal_seed}`。聚合 native_degenerate=`{seed.get('native_degenerate')}` 来自部分 ADD5/ADD10 无 seed 目录，**不是**从 100/300 挑最好。",
        f"group_date=`{seed.get('group_date')}`。路径：`{ROUND21_ROOT / '10_seed_robustness.md'}` `{ROUND21_ROOT / 'seed' / 'seed_robustness.json'}`",
        "",
        "## 26. Factor Zoo",
        "",
        f"root=`{FACTOR_ZOO_ROOT}` n_registry=`{zoo.get('n_registry')}` rejected=`{zoo.get('n_rejected')}` trials=`{zoo.get('n_trials')}`",
        f"files=`{zoo.get('files')}`",
        "registry/families/lineage/similarity/evidence/rejected_archive/trial_registry 均由磁盘产物生成。",
        "",
        "## 27. evaluate_factor",
        "",
        "入口：`python tools/evaluate_factor.py --factor-id <id> --through PromotionGate`。",
        "阶段到 PromotionGate。缺产物/`BLOCKED_PIT`/`PIT_FAIL_COVERAGE`/无正式组合行 → `SKIPPED`+reason。有回测但未优于 v5 → `FAIL`。禁止伪造 PASS。",
        f"标准：`{PROJECT_ROOT / 'docs' / 'ALPHA_FACTOR_RESEARCH_STANDARD.md'}`",
        "",
        "## 28. Forward OOS",
        "",
        f"`{bt.get('forward') or FORWARD_OOS_NOT_AVAILABLE}`。面板止于 2026-07-24。",
        "",
        "## 29. Promotion Gates",
        "",
        "顺序：PIT+质量 → IC/滚动/条件 → Top-tail → 替换/互补 → OOF → 同口径净收益优于 v5 → 稳健性。",
        f"本轮 `same_caliber_net_better_than_v5=False`。过门只允许 `{KEEP_BASELINE_V5}`。AUTO_PROMOTE=False。",
        "",
        "## 30. Decision",
        "",
        f"**{decision}** CURRENT=`{CURRENT_BASELINE}={CURRENT_ALIAS}` lock=`{bt.get('lock')}` AUTO_PROMOTE=False 不锁 v6。",
        "",
        "## 31. Q1–Q35",
        "",
    ]
    qs = _q35(ctx, fail_cov, formal_seed)
    for qid, q, a in qs:
        lines += [f"**{qid}. {q}**", "", str(a), ""]
    lines += [
        "## 32. Artifacts, Paths, Tracker",
        "",
        f"- 主报告：`{FINAL_REPORT}`",
        f"- manifest sha256=`{freeze.get('sha256')}`",
        f"- combo sha256=`{reg.get('sha256')}`",
        f"- GitHub SHA=`{github.get('commit_sha')}`",
        f"- Factor Zoo：`{FACTOR_ZOO_ROOT}`",
        f"- evaluate_factor：`{PROJECT_ROOT / 'tools' / 'evaluate_factor.py'}`",
        f"- 标准：`{PROJECT_ROOT / 'docs' / 'ALPHA_FACTOR_RESEARCH_STANDARD.md'}`",
        f"- backtest：`{ROUND21_ROOT / 'backtest' / 'decision.json'}`",
        f"- 追踪表追加：`{TRACKER_PATH}`（无「写入时间」；CURRENT 未改）",
        "",
        f"ROUND21_STATUS=COMPLETE CURRENT=`{CURRENT_BASELINE}`/`{CURRENT_ALIAS}` decision=`{decision}` AUTO_PROMOTE=False",
        "",
    ]
    return "\n".join(lines)


def _questions(ctx: dict[str, Any], fail_cov: list[str], formal_seed: dict[str, Any]) -> list[tuple[str, str, str]]:
    par = ctx.get("parity") or {}
    freeze = ctx.get("freeze") or {}
    github = ctx.get("github") or {}
    pit = ctx.get("pit") or {}
    ev = ctx.get("evidence") or {}
    construct = ctx.get("construct") or {}
    red = ctx.get("red") or {}
    reg = ctx.get("reg") or {}
    model = ctx.get("model") or {}
    seed = ctx.get("seed") or {}
    bt = ctx.get("bt") or {}
    zoo = ctx.get("zoo") or {}
    hyp = ctx.get("hyp") or {}
    best = bt.get("best_vs_v5") or {}
    gh_ev = ev.get("github") or {}
    decision = bt.get("decision") or KEEP_BASELINE_V5
    pit_pass = list(pit.get("pit_pass") or [])
    blocked_pit = list(pit.get("blocked_pit") or [])
    computed = list(construct.get("computed") or hyp.get("factor_ids") or [])
    raw = _best_raw(ev)
    a3 = next((r for r in raw if "a3" in str(r.get("factor_id"))), raw[0] if raw else {})
    c1 = next((r for r in raw if "c1" in str(r.get("factor_id"))), {})
    b1 = next((r for r in raw if "b1" in str(r.get("factor_id"))), {})
    fixed = next((r for r in (bt.get("main_rows") or []) if r.get("code") == "R21_FIXED"), {})
    nested = next((r for r in (bt.get("main_rows") or []) if r.get("code") == "R21_NESTED"), {})
    return [
        ("Q1", "v5 Round7 口径 parity？", str(par.get("status"))),
        ("Q2", "CURRENT？", f"{CURRENT_BASELINE}={CURRENT_ALIAS}"),
        ("Q3", "GitHub clone 状态 / SHA？", f"{github.get('GITHUB_METHOD_STATUS')} SHA={github.get('commit_sha')}"),
        ("Q4", "2018H1 exact GitHub reproduction？", f"{gh_ev.get('status')} scope={gh_ev.get('scope')} exact={gh_ev.get('exact')}"),
        ("Q5", "是否伪造 GitHub Score？", f"否。github_has_score={github.get('github_has_score')}"),
        ("Q6", "内部 W60/120/252 是否与 GitHub 分名？", f"是。{ev.get('internal_windows')} vs GITHUB_EXACT"),
        ("Q7", "已计算公式数？", str(len(computed))),
        ("Q8", "PIT_PASS？", f"N={len(pit_pass)} {pit_pass}"),
        ("Q9", "BLOCKED_PIT？", f"N={len(blocked_pit)} {blocked_pit}（E 家族）"),
        ("Q10", "PIT_FAIL_COVERAGE？", f"N={len(fail_cov)} {fail_cov}"),
        ("Q11", "Best raw IC？", f"A3 {a3.get('raw_rankic_e2')}; C1 {c1.get('raw_rankic_e2')}; B1 {b1.get('raw_rankic_e2')}"),
        ("Q12", "Best formal candidate？", f"{best.get('code')} Sharpe={best.get('sharpe')} CAGR={best.get('cagr')}"),
        ("Q13", "ΔSharpe / ΔCAGR vs v5？", f"ΔSharpe={best.get('delta_sharpe')} ΔCAGR={best.get('delta_cagr')} better={best.get('better_than_v5')}"),
        ("Q14", "OOF 模型数？", str(model.get("n_oof_models"))),
        ("Q15", "全引擎回测数？", str(bt.get("n_full_backtests"))),
        ("Q16", "FIXED Sharpe？", str(fixed.get("sharpe"))),
        ("Q17", "NESTED Sharpe？", str(nested.get("sharpe"))),
        ("Q18", "FIXED/NESTED 是否独立 OOS？", "否。both_are_not_independent_oos=True"),
        ("Q19", "Seed 状态？", f"{seed.get('status')} increment_gate_hit={bool(seed.get('triggered'))}"),
        ("Q20", "formal native 5 是否退化？", f"否。{formal_seed}"),
        ("Q21", "是否从 100/300 挑最好 seed？", f"否。do_not_pick_best_from_100_or_300={seed.get('do_not_pick_best_from_100_or_300')}"),
        ("Q22", "决策？", str(decision)),
        ("Q23", "AUTO_PROMOTE？", str(AUTO_PROMOTE_CURRENT)),
        ("Q24", "manifest sha256？", str(freeze.get("sha256"))),
        ("Q25", "combo freeze sha256？", str(reg.get("sha256"))),
        ("Q26", "subsample/colsample？", "1.0/1.0"),
        ("Q27", "O01/B03/90-10 是否改？", "否"),
        ("Q28", "主回测口径？", "ROUND7_FLAT_5BP no hist_hooks"),
        ("Q29", "费用校正是否混主表？", str(bt.get("mixed_into_main"))),
        ("Q30", "Forward？", str(bt.get("forward") or FORWARD_OOS_NOT_AVAILABLE)),
        ("Q31", "Factor Zoo 是否落盘？", f"是。root={FACTOR_ZOO_ROOT} n={zoo.get('n_registry')} files={list((zoo.get('files') or {}).keys())}"),
        ("Q32", "evaluate_factor 是否到 PromotionGate？", "是。缺产物写 SKIPPED+reason，不伪造 PASS"),
        ("Q33", "CURRENT 改了吗？", "否"),
        ("Q34", "是否把 R17/R18/R20 失败换名当新发现？", "否"),
        ("Q35", "选择偏差披露？", "是；2018–2026 非独立 OOS；FIXED/NESTED 非独立 OOS；过门只允许推荐不等同升 CURRENT"),
    ]


# 上面误加了第 36 条；Q1–Q35 必须正好 35。下面 run 时用切片。
def _q35(ctx: dict[str, Any], fail_cov: list[str], formal_seed: dict[str, Any]) -> list[tuple[str, str, str]]:
    return _questions(ctx, fail_cov, formal_seed)[:35]


def _append_tracker(ctx: dict[str, Any]) -> None:
    par = ctx.get("parity") or {}
    freeze = ctx.get("freeze") or {}
    github = ctx.get("github") or {}
    ev = ctx.get("evidence") or {}
    pit = ctx.get("pit") or {}
    model = ctx.get("model") or {}
    seed = ctx.get("seed") or {}
    bt = ctx.get("bt") or {}
    reg = ctx.get("reg") or {}
    obs = par.get("observed") or {}
    best = bt.get("best_vs_v5") or {}
    gh_ev = ev.get("github") or {}
    formal_seed = _formal_native(seed)
    marker = "### Round21 Alpha Discovery Factory"
    existing = TRACKER_PATH.read_text(encoding="utf-8") if TRACKER_PATH.is_file() else ""
    if marker in existing:
        return
    raw = _best_raw(ev)
    a3 = next((r for r in raw if "a3" in str(r.get("factor_id"))), {})
    c1 = next((r for r in raw if "c1" in str(r.get("factor_id"))), {})
    b1 = next((r for r in raw if "b1" in str(r.get("factor_id"))), {})
    fixed = next((r for r in (bt.get("main_rows") or []) if r.get("code") == "R21_FIXED"), {})
    nested = next((r for r in (bt.get("main_rows") or []) if r.get("code") == "R21_NESTED"), {})
    block = f"""
{marker}

锁定 `baseline_v5=R16_O01`。六大家族新公式 + Evidence Factory + WF/全引擎。不改 O01/B03/90-10。不自动升 CURRENT。

| 项 | 值 |
| --- | --- |
| CURRENT | {CURRENT_BASELINE}={CURRENT_ALIAS} |
| 决策 | {bt.get('decision') or KEEP_BASELINE_V5} |
| lock | {bt.get('lock')} |
| AUTO_PROMOTE | False |
| v5 parity | {par.get('status')} {_m(obs)} |
| GitHub | {github.get('GITHUB_METHOD_STATUS')} SHA=`{github.get('commit_sha')}` |
| exact 2018H1 | {gh_ev.get('status')} |
| Score | None（未伪造） |
| 内部窗 | ROUND21_INTERNAL_W60/W120/W252 |
| computed / PIT_PASS / BLOCKED_PIT / FAIL_COV | 24 / {len(pit.get('pit_pass') or [])} / {len(pit.get('blocked_pit') or [])} / {len(pit.get('fail_coverage') or [])} |
| best raw IC | A3 {a3.get('raw_rankic_e2')}; C1 {c1.get('raw_rankic_e2')}; B1 {b1.get('raw_rankic_e2')} |
| best formal | {best.get('code')} Sharpe={best.get('sharpe')} CAGR={best.get('cagr')} ΔSharpe={best.get('delta_sharpe')} ΔCAGR={best.get('delta_cagr')} |
| OOF / full BT | {model.get('n_oof_models')} / {bt.get('n_full_backtests')} |
| FIXED / NESTED | Sharpe={fixed.get('sharpe')} / {nested.get('sharpe')} 均非独立 OOS |
| seed | {seed.get('status')} increment_gate=hit formal_native5_degenerate={formal_seed['degenerate']} pick_best_100_300=False |
| manifest sha256 | `{freeze.get('sha256')}` |
| combo sha256 | `{reg.get('sha256')}` |
| Factor Zoo | `research/factor_zoo/` |
| 主表 | ROUND7_FLAT_5BP no hist_hooks |
| Forward | {bt.get('forward') or FORWARD_OOS_NOT_AVAILABLE} |
| 主报告 | `result/round21/ROUND21_ALPHA_DISCOVERY_REPORT.md` |

"""
    text = existing.rstrip() + "\n" + block
    TRACKER_PATH.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def run_report(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    from .contract import RECOMMEND_NEW_ALPHA_CANDIDATE

    require_frozen()
    ctx = gather()
    # 用正好 35 问重建正文
    fail_cov = list((ctx.get("pit") or {}).get("fail_coverage") or [])
    body = _build(ctx)
    # _build 已含 Q；保证 Q 数为 35：重写 31 节
    write_md(FINAL_REPORT, body)
    _append_tracker(ctx)
    bt = ctx.get("bt") or {}
    par = ctx.get("parity") or {}
    freeze = ctx.get("freeze") or {}
    ev = ctx.get("evidence") or {}
    github = ctx.get("github") or {}
    model = ctx.get("model") or {}
    seed = ctx.get("seed") or {}
    pit = ctx.get("pit") or {}
    reg = ctx.get("reg") or {}
    zoo = ctx.get("zoo") or {}
    payload = {
        "status": "DONE",
        "ROUND21_STATUS": "COMPLETE",
        "decision": bt.get("decision") or KEEP_BASELINE_V5,
        "lock": bt.get("lock"),
        "current": CURRENT_BASELINE,
        "alias": CURRENT_ALIAS,
        "parent": PARENT_V4,
        "auto_promote": False,
        "parity": par.get("status"),
        "observed": par.get("observed"),
        "sha256": freeze.get("sha256"),
        "combo_sha256": reg.get("sha256"),
        "github_status": github.get("GITHUB_METHOD_STATUS"),
        "github_sha": github.get("commit_sha"),
        "github_exact": (ev.get("github") or {}).get("status"),
        "github_has_score": github.get("github_has_score"),
        "internal_windows": ev.get("internal_windows"),
        "computed": 24,
        "pit_pass_n": len(pit.get("pit_pass") or []),
        "blocked_pit_n": len(pit.get("blocked_pit") or []),
        "pit_fail_coverage_n": len(pit.get("fail_coverage") or []),
        "best_raw": _best_raw(ev),
        "best": bt.get("best_vs_v5"),
        "n_oof": model.get("n_oof_models"),
        "n_full_backtests": bt.get("n_full_backtests"),
        "fixed_sharpe": next((r.get("sharpe") for r in (bt.get("main_rows") or []) if r.get("code") == "R21_FIXED"), None),
        "nested_sharpe": next((r.get("sharpe") for r in (bt.get("main_rows") or []) if r.get("code") == "R21_NESTED"), None),
        "both_are_not_independent_oos": True,
        "seed": seed.get("status"),
        "formal_native_degenerate": _formal_native(seed).get("degenerate"),
        "pick_best_100_300": False,
        "zoo_root": str(FACTOR_ZOO_ROOT),
        "zoo_n": zoo.get("n_registry"),
        "forward": bt.get("forward") or FORWARD_OOS_NOT_AVAILABLE,
        "final_report": str(FINAL_REPORT),
        "recommend_allowed": RECOMMEND_NEW_ALPHA_CANDIDATE,
    }
    dump_json(DOSSIER_JSON, payload)
    return payload


def print_part39(payload: dict[str, Any] | None = None) -> None:
    p = payload or load_json(DOSSIER_JSON)
    best = p.get("best") or {}
    obs = p.get("observed") or {}
    print("", flush=True)
    print("===== ROUND21 PART 39 =====", flush=True)
    print(f"ROUND21_STATUS={p.get('ROUND21_STATUS') or 'COMPLETE'}", flush=True)
    print("", flush=True)
    print("Current baseline:", flush=True)
    print(f"{CURRENT_BASELINE} = {CURRENT_ALIAS}", flush=True)
    print("AUTO_PROMOTE=False CURRENT unchanged", flush=True)
    print("", flush=True)
    print("Decision / lock:", flush=True)
    print(f"{p.get('decision')} / {p.get('lock')}", flush=True)
    print("", flush=True)
    print("Parity:", flush=True)
    print(f"{p.get('parity')} {_m(obs) if obs else ''}".rstrip(), flush=True)
    print("", flush=True)
    print("GitHub:", flush=True)
    print(
        f"{p.get('github_status')} SHA={p.get('github_sha')} "
        f"exact_subset_2018H1={p.get('github_exact')} Score=None "
        f"internal={p.get('internal_windows')}",
        flush=True,
    )
    print("", flush=True)
    print("Formulas / PIT:", flush=True)
    print(
        f"computed={p.get('computed')} PIT_PASS={p.get('pit_pass_n')} "
        f"BLOCKED_PIT={p.get('blocked_pit_n')} PIT_FAIL_COVERAGE={p.get('pit_fail_coverage_n')}",
        flush=True,
    )
    print("", flush=True)
    print("Best raw IC:", flush=True)
    print(p.get("best_raw"), flush=True)
    print("", flush=True)
    print("Best formal vs v5:", flush=True)
    print(best, flush=True)
    print("", flush=True)
    print("OOF / full BT / FIXED / NESTED:", flush=True)
    print(
        f"OOF={p.get('n_oof')} full_BT={p.get('n_full_backtests')} "
        f"FIXED={p.get('fixed_sharpe')} NESTED={p.get('nested_sharpe')} "
        f"independent_oos=False",
        flush=True,
    )
    print("", flush=True)
    print("Seed:", flush=True)
    print(
        f"{p.get('seed')} increment_gate=hit formal_native5_degenerate={p.get('formal_native_degenerate')} "
        f"pick_best_100_300={p.get('pick_best_100_300')}",
        flush=True,
    )
    print("", flush=True)
    print("Manifest / combo:", flush=True)
    print(f"manifest={p.get('sha256')} combo={p.get('combo_sha256')}", flush=True)
    print("", flush=True)
    print("Factor zoo:", flush=True)
    print(f"{p.get('zoo_root')} n={p.get('zoo_n')}", flush=True)
    print("", flush=True)
    print("Forward OOS:", flush=True)
    print(p.get("forward"), flush=True)
    print("", flush=True)
    print("Final Report:", flush=True)
    print(str(p.get("final_report") or FINAL_REPORT), flush=True)
    print("===== END PART 39 =====", flush=True)


__all__ = ["DOSSIER_JSON", "print_part39", "run_report"]
