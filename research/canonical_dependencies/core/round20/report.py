"""ROUND20_FACTOR_RESEARCH_REPORT.md 第1–30节 + Q1–Q28。追踪表追加。PART 38。不写「写入时间」。不改 CURRENT。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND20_ROOT
from .common import dump_json, load_json, pct, write_md
from .contract import (
    AWAITING_USER_REVIEW,
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    FINAL_REPORT,
    FORWARD_OOS_NOT_AVAILABLE,
    KEEP_BASELINE_V5,
    PARENT_V4,
    TRACKER_PATH,
    V5_ANCHOR,
)
from .freeze import FREEZE_JSON, require_frozen
from .parity import PARITY_JSON

DOSSIER_JSON = ROUND20_ROOT / "promotion" / "dossier.json"


def _m(obs: dict[str, Any] | None) -> str:
    row = obs or {}
    if row.get("cagr") is None and row.get("sharpe") is None:
        return str(row.get("status") or "n/a")
    return (
        f"CAGR={row.get('cagr')} Sharpe={row.get('sharpe')} "
        f"MaxDD={row.get('max_drawdown')} trades={row.get('completed_trades')} NAV_end={row.get('nav_end')}"
    )


def gather() -> dict[str, Any]:
    from .backtest import BT_JSON
    from .combo import REG_JSON
    from .conditional import COND_JSON
    from .ic import IC_JSON
    from .inventory import INV_JSON
    from .model import MODEL_JSON
    from .pit import PIT_JSON
    from .redundancy import RED_JSON
    from .runner import COMBO_TRAIN_JSON
    from .seed import SEED_JSON
    from .shape import SHAPE_JSON

    return {
        "parity": load_json(PARITY_JSON),
        "freeze": load_json(FREEZE_JSON),
        "inv": load_json(INV_JSON),
        "pit": load_json(PIT_JSON),
        "ic": load_json(IC_JSON),
        "shape": load_json(SHAPE_JSON),
        "red": load_json(RED_JSON),
        "cond": load_json(COND_JSON),
        "reg": load_json(REG_JSON),
        "combo": load_json(COMBO_TRAIN_JSON),
        "model": load_json(MODEL_JSON),
        "seed": load_json(SEED_JSON),
        "bt": load_json(BT_JSON),
    }


def _build(ctx: dict[str, Any]) -> str:
    par = ctx.get("parity") or {}
    freeze = ctx.get("freeze") or {}
    inv = ctx.get("inv") or {}
    pit = ctx.get("pit") or {}
    ic = ctx.get("ic") or {}
    shape = ctx.get("shape") or {}
    red = ctx.get("red") or {}
    cond = ctx.get("cond") or {}
    reg = ctx.get("reg") or {}
    combo = ctx.get("combo") or {}
    model = ctx.get("model") or {}
    seed = ctx.get("seed") or {}
    bt = ctx.get("bt") or {}
    obs = par.get("observed") or {}
    best = combo.get("best") or {}
    decision = bt.get("decision") or KEEP_BASELINE_V5
    lines = [
        "# ROUND20_FACTOR_RESEARCH_REPORT",
        "",
        "Round20 = 锁定 `baseline_v5=R16_O01` 与 Round7 5bp 口径，对已有股票因子做审计、去冗余、有限组合、模型对照与 group-date 扰动。不搜择时/O01/B03。不自动升 CURRENT。",
        f"manifest sha256=`{freeze.get('sha256')}` parity=`{par.get('status')}` 决策=**{decision}** CURRENT=**{CURRENT_BASELINE}={CURRENT_ALIAS}** AUTO_PROMOTE=False",
        "",
        "## 1. Executive Verdict",
        "",
        f"**{decision}**。CURRENT 仍是 `{CURRENT_BASELINE} = {CURRENT_ALIAS}`。未自动晋升。不锁 v6。",
        f"- Stage 0 v5 parity：`{par.get('status')}`。观测 `{_m(obs)}`。n_obs=`{obs.get('n_trading_observations') or obs.get('n_obs')}`。",
        f"- Combo tested N=`{combo.get('n_tested')}` with_metrics=`{combo.get('n_with_metrics')}` skip=`{combo.get('n_skip')}`。best=`{best.get('code')}` `{_m(best)}`（非独立 OOS）。",
        f"- A00 vs A01 agree=`{(model.get('a00_a01_parity') or {}).get('agree')}` M2=`{model.get('m2_status')}` finalists=`{[f.get('code') for f in (model.get('finalists') or [])]}`。",
        f"- Native seed=`{(seed.get('native_audit') or {}).get('status')}`。S2=`{(seed.get('s2') or {}).get('status')}` S3=`{(seed.get('s3') or {}).get('status')}` S4=`{(seed.get('s4') or {}).get('status')}`。未用 100 个相同 native seed。",
        f"- 主表 Round7 5bp。费用校正敏感性分表。Forward=`{FORWARD_OOS_NOT_AVAILABLE}`。",
        f"- lock=`{bt.get('lock')}` gates_passed=`{(bt.get('gates') or {}).get('passed')}`。",
        "",
        "## 2. Research Question and Lock",
        "",
        "在锁定 v5 结构（90/10、B03、O01、clean_risk、退出、仓位、gap、1.0/1.0）的前提下：已有 75+F1/F2/F3 因子经审计/去冗余/有限组合/模型对照后，能否产生可解释且同口径优于 v5 的新 Alpha？",
        f"CURRENT=`{CURRENT_BASELINE}` parent=`{PARENT_V4}`。只改 Alpha 特征/变换/预注册模型/group-date 扰动。",
        "",
        "## 3. Prior Research / Why This Is Not Independent OOS",
        "",
        "2018–2026 已被 R1–19 研究过。Discovery/Validation/Audit **不是独立 OOS**。Walk-forward 仍是预测年 Y ← 训 Y-3..Y-1。R17 已跑 5000 搜索；本轮禁止再做。F2=`PIT_TIMESTAMP_UNVERIFIED` 不得晋级。选择偏差=`REPEATED_HISTORICAL_TESTING`。",
        "",
        "## 4. Stage 0 v5 Parity（Round7 5bp）",
        "",
        f"status=`{par.get('status')}` engine=`{par.get('engine')}` validation_hooks=`None` fee=`{par.get('fee_schedule')}`",
        f"anchor CAGR=`{V5_ANCHOR['cagr']}` Sharpe=`{V5_ANCHOR['sharpe']}` MaxDD=`{V5_ANCHOR['max_drawdown']}` trades=`{V5_ANCHOR['completed_trades']}` n_obs=`{V5_ANCHOR['n_obs']}` NAV_end=`{V5_ANCHOR['nav_end']}`",
        f"observed `{_m(obs)}`",
        f"FAIL 只停正式晋级。路径：`{PARITY_JSON}`",
        "",
        "## 5. CAGR Convention",
        "",
        "- **主口径** 日历年 365.25 `annualized_return`",
        "- **对照** 252 交易日 `cagr_of` 不得替换主口径",
        "",
        "## 6. Frozen Manifest",
        "",
        f"sha256=`{freeze.get('sha256')}` frozen=`{freeze.get('frozen')}` path=`{freeze.get('path')}`",
        "IC 薄日跳过不记 0。簇 Spearman≥0.85 非唯一 DROP。新因子入主组合 ≤12。combo registry 看 NAV 前冻结。PCA 0.90 不搜阈值。",
        "",
        "## 7. Factor Inventory",
        "",
        f"n_total=`{inv.get('n_total')}` r17=`{inv.get('n_r17_registry')}` r18=`{inv.get('n_r18_f123')}` match_plan=`{inv.get('counts_match_plan')}` F2 eligible=`{inv.get('f2_promotion_eligible')}`",
        "不得捏造清单外因子。F1/F3 不换名重开假设。",
        "",
        "## 8. PIT",
        "",
        f"n_total=`{pit.get('n_total')}` eligible=`{pit.get('n_promotion_eligible')}` f2_eligible=`{pit.get('f2_promotion_eligible')}`",
        "",
        "## 9. IC / RankIC / HAC / bootstrap",
        "",
        f"daily_rows=`{ic.get('n_daily_rows')}` skip_thin=`{(ic.get('skip_counts') or {}).get('thin')}` skip_degen=`{(ic.get('skip_counts') or {}).get('degenerate')}`",
        "双标签分开。不足最少有效股票的日跳过，IC 不记 0。",
        "",
        "## 10. Shape / Top-tail",
        "",
        f"non_monotonic=`{shape.get('n_non_monotonic')}` forced_q10=`{shape.get('forced_q10')}`。Q1–Q5；样本不够不强制 Q10。",
        "",
        "## 11. Redundancy KEEP/DROP",
        "",
        f"KEEP=`{red.get('n_keep')}` DROP=`{red.get('n_drop')}` clusters=`{red.get('n_clusters_ge2')}` f1_f3_rebranded=`{red.get('f1_f3_rebranded')}`",
        "",
        "## 12. Conditional IC / ortho / PCA / LOFO",
        "",
        f"leaders=`{len(cond.get('leaders') or [])}` ortho_winners=`{cond.get('ortho_winner_counts')}` lofo_delete_from_v5=`{cond.get('lofo_delete_from_v5')}` PCA explained=0.90 fold-fit only。禁止全历史 PCA/正交。LOFO 不得直接从 v5 删列。",
        "",
        "## 13. Combo Registry Freeze",
        "",
        f"n_registry=`{reg.get('n_registry')}` selected=`{reg.get('selected_new_factors')}` trained_at_freeze=`{reg.get('trained')}` F2 included=`{reg.get('f2_included')}` padded=`{reg.get('padded_to_quota')}`",
        "禁止事后 Best×Best。",
        "",
        "## 14. Combo Training",
        "",
        f"tested N=`{combo.get('n_tested')}` with_metrics=`{combo.get('n_with_metrics')}` skip=`{combo.get('n_skip')}` post_hoc_best_x_best=`{combo.get('post_hoc_best_x_best')}`",
        f"best=`{best.get('code')}` `{_m(best)}` extras=`{best.get('five_extras') or best.get('extras')}`",
        "引擎：fuse 90/10 + B03 + apply_top2_veto O01 + Round7 5bp。SKIP 已记原因。",
        "",
        "## 15. Model Selection A/B/C/D",
        "",
        f"A00 vs A01 agree=`{(model.get('a00_a01_parity') or {}).get('agree')}`",
        "B01–B04 = 冻结 pruned/cond-IC/ortho/PCA 代表。C01–C04 = registry 筛选代表，非独立 OOS。",
        "D01=XGB+12 selected。D02=ElasticNet 排序代理（损失≠NDCG）。D03=LightGBM LambdaRank 或 SKIP_DEPENDENCY，不换模型。",
        "正式候选 90/10。five/ten-only 仅预注册归因。",
        "",
        "## 16. A/B/C/D Table",
        "",
        "| code | role | status | CAGR | Sharpe | MaxDD | alias |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in model.get("abcd") or model.get("rows") or []:
        if str(r.get("code") or "").startswith("R20_ATTR"):
            continue
        if r.get("code") in {"R20_A00", "R20_A01", "R20_B01", "R20_B02", "R20_B03", "R20_B04", "R20_C01", "R20_C02", "R20_C03", "R20_C04", "R20_D01", "R20_D02", "R20_D03"} or r.get("code"):
            if str(r.get("code") or "").startswith("R20_") and not str(r.get("code")).startswith("R20_S") and not str(r.get("code")).startswith("R20_P"):
                lines.append(
                    f"| `{r.get('code')}` | {r.get('role')} | {r.get('status')} | {r.get('cagr')} | {r.get('sharpe')} | {r.get('max_drawdown')} | `{r.get('alias_of')}` |"
                )
    lines += [
        "",
        "## 17. Model Finalists",
        "",
        f"finalists=`{model.get('finalists')}`。最多少数几个。选择偏差已披露。",
        "",
        "## 18. Native Seed Audit",
        "",
        f"`{(seed.get('native_audit') or {})}`",
        "1.0/1.0 下 rank 退化。禁止 100 个相同 native 模型。",
        "",
        "## 19. Group-date Perturbation",
        "",
        f"spec=`{seed.get('group_date_spec')}` S1=`{seed.get('s1')}`",
        f"S2=`{(seed.get('s2') or {}).get('status')}` S3=`{(seed.get('s3') or {}).get('status')}` S4=`{(seed.get('s4') or {}).get('status')}` `{(seed.get('s4') or {}).get('reason')}`",
        "ensemble=`mean(seed_rank_pct)` 再 90/10。投票门槛 DIAGNOSTIC_ONLY。配对 Δ 不是 OOS p。",
        "",
        "## 20. Formal Backtest Caliber",
        "",
        "MAIN = `round7_fixed_sleeve_config` + `run_local_fixed_sleeve_backtest`，不传 validation_hooks。卖税 5bp。CAGR 365.25。30 万双袖套。",
        "",
        "## 21. Main Table vs v5",
        "",
        f"best_vs_v5=`{bt.get('best_vs_v5')}`",
        "MAIN rows：",
    ]
    for r in bt.get("main_rows") or []:
        lines.append(
            f"- `{r.get('code')}` CAGR=`{r.get('cagr')}` Sharpe=`{r.get('sharpe')}` "
            f"MaxDD=`{r.get('max_drawdown')}` trades=`{r.get('completed_trades')}` NAV_end=`{r.get('nav_end')}`"
        )
    lines += [
        "",
        "## 22. Yearly / Ex2024 / Ex2025",
        "",
    ]
    for r in bt.get("main_rows") or []:
        lines.append(
            f"- `{r.get('code')}` yearly=`{r.get('yearly_returns')}` "
            f"Ex2024=`{r.get('ex2024')}` Ex2025=`{r.get('ex2025')}` Ex2024+2025=`{r.get('ex2024_2025')}`"
        )
    if not bt.get("main_rows"):
        lines.append("见 `result/round20/11_backtest_decision.md`。")
    lines += [
        "",
        "## 23. HISTORICAL_FEE_CORRECTED Sensitivity",
        "",
        "单独表，禁止与 Legacy 5bp 主表混列。用户 53.62%/1.840 是对照不是本轮主表复制源。",
        f"mixed_into_main=`{bt.get('mixed_into_main')}` rows=`{bt.get('fee_sensitivity_table')}`",
        "",
        "## 24. Block Bootstrap N=5000",
        "",
        "20 日块。STATISTICAL RESAMPLING NOT OOS。",
    ]
    for r in bt.get("main_rows") or []:
        if r.get("block_bootstrap_vs_v5") is not None:
            lines.append(f"- `{r.get('code')}` `{r.get('block_bootstrap_vs_v5')}`")
    lines += [
        "",
        "## 25. SPA / PBO",
        "",
        f"`{bt.get('spa')}` 缺矩阵则 NOT_IDENTIFIABLE。",
        "",
        "## 26. Forward OOS",
        "",
        f"`{FORWARD_OOS_NOT_AVAILABLE}`。面板止于 2026-07-24。",
        "",
        "## 27. Promotion Gates",
        "",
        f"`{bt.get('gates')}` 过门只允许 `{bt.get('decision')}` + `{bt.get('lock')}`。默认 KEEP_BASELINE_V5。",
        "",
        "## 28. Decision",
        "",
        f"**{decision}** CURRENT=`{CURRENT_BASELINE}={CURRENT_ALIAS}` lock=`{bt.get('lock')}` AUTO_PROMOTE=False 不锁 v6。",
        "",
        "## 29. Q1–Q28",
        "",
    ]
    qs = [
        ("Q1", "v5 Round7 口径 parity？", str(par.get("status"))),
        ("Q2", "CURRENT？", f"{CURRENT_BASELINE}={CURRENT_ALIAS}"),
        ("Q3", "F2 可否晋级？", str(pit.get("f2_promotion_eligible"))),
        ("Q4", "因子盘点行数？", str(inv.get("n_total"))),
        ("Q5", "manifest sha256？", str(freeze.get("sha256"))),
        ("Q6", "薄日 IC 是否记 0？", "否，跳过该日"),
        ("Q7", "KEEP/DROP？", f"KEEP={red.get('n_keep')} DROP={red.get('n_drop')}"),
        ("Q8", "正交/PCA 是否全历史？", "否，fold 内"),
        ("Q9", "combo registry 是否看 NAV 前冻结？", str(reg.get("frozen_before_combo_nav"))),
        ("Q10", "是否事后 Best×Best？", str(combo.get("post_hoc_best_x_best"))),
        ("Q11", "A00 vs A01 是否一致？", str((model.get("a00_a01_parity") or {}).get("agree"))),
        ("Q12", "C01–C04 是否独立 OOS Best？", "否，冻结筛选代表"),
        ("Q13", "D02 损失是否 NDCG？", "否，MSE on rank-pct"),
        ("Q14", "M2/D03？", str(model.get("m2_status"))),
        ("Q15", "native 5 seed？", str((seed.get("native_audit") or {}).get("status"))),
        ("Q16", "是否用 100 native 相同模型？", str(seed.get("fake_100_native"))),
        ("Q17", "S4 300？", f"{(seed.get('s4') or {}).get('status')} {(seed.get('s4') or {}).get('reason')}"),
        ("Q18", "subsample/colsample？", "1.0/1.0"),
        ("Q19", "O01/B03/90-10 是否改？", "否"),
        ("Q20", "主回测口径？", "ROUND7_FLAT_5BP no hist_hooks"),
        ("Q21", "费用校正是否混主表？", str(bt.get("mixed_into_main"))),
        ("Q22", "是否仅 2024/2025？", str(((bt.get("gates") or {}).get("checks") or {}).get("not_only_2024_2025"))),
        ("Q23", "bootstrap N？", "5000 block=20 NOT OOS"),
        ("Q24", "SPA？", str((bt.get("spa") or {}).get("status"))),
        ("Q25", "Forward？", FORWARD_OOS_NOT_AVAILABLE),
        ("Q26", "决策？", str(decision)),
        ("Q27", "CURRENT 改了吗？", "否"),
        ("Q28", "选择偏差披露？", "是；2018–2026 非独立 OOS"),
    ]
    for qid, q, a in qs:
        lines += [f"**{qid}. {q}**", "", str(a), ""]
    lines += [
        "## 30. Artifacts, Paths, Tracker",
        "",
        f"- 主报告：`{FINAL_REPORT}`",
        f"- manifest sha256=`{freeze.get('sha256')}`",
        f"- combo results：`{ROUND20_ROOT / 'combo' / 'combo_results.csv'}`",
        f"- model csv：`{ROUND20_ROOT / 'model' / 'abcd_matrix.csv'}`",
        f"- seed：`{ROUND20_ROOT / 'seed' / 'seed_robustness.json'}`",
        f"- backtest：`{ROUND20_ROOT / 'backtest' / 'decision.json'}`",
        f"- 追踪表追加：`{TRACKER_PATH}`（无「写入时间」；CURRENT 未改）",
        "",
        f"ROUND20_STATUS=COMPLETE CURRENT=`{CURRENT_BASELINE}`/`{CURRENT_ALIAS}` decision=`{decision}`",
        "",
    ]
    return "\n".join(lines)


def _append_tracker(ctx: dict[str, Any]) -> None:
    par = ctx.get("parity") or {}
    freeze = ctx.get("freeze") or {}
    combo = ctx.get("combo") or {}
    model = ctx.get("model") or {}
    seed = ctx.get("seed") or {}
    bt = ctx.get("bt") or {}
    obs = par.get("observed") or {}
    best = combo.get("best") or {}
    marker = "### Round20 Factor Diagnostics"
    existing = TRACKER_PATH.read_text(encoding="utf-8") if TRACKER_PATH.is_file() else ""
    if marker in existing:
        return
    block = f"""
{marker}

锁定 `baseline_v5=R16_O01`。因子审计/去冗余/≤100 组合/模型对照/group-date。不改 O01/B03/90-10。不自动升 CURRENT。

| 项 | 值 |
| --- | --- |
| CURRENT | {CURRENT_BASELINE}={CURRENT_ALIAS} |
| 决策 | {bt.get('decision') or KEEP_BASELINE_V5} |
| lock | {bt.get('lock')} |
| AUTO_PROMOTE | False |
| v5 parity | {par.get('status')} {_m(obs)} |
| manifest sha256 | `{freeze.get('sha256')}` |
| combo N / skip | {combo.get('n_tested')} / {combo.get('n_skip')} |
| best combo | {best.get('code')} {_m(best)} |
| A00 vs A01 | {(model.get('a00_a01_parity') or {}).get('agree')} |
| M2 | {model.get('m2_status')} |
| finalists | {[f.get('code') for f in (model.get('finalists') or [])]} |
| native seed | {(seed.get('native_audit') or {}).get('status')} |
| S2/S3/S4 | {(seed.get('s2') or {}).get('status')} / {(seed.get('s3') or {}).get('status')} / {(seed.get('s4') or {}).get('status')} |
| 主表 | ROUND7_FLAT_5BP no hist_hooks |
| Forward | {FORWARD_OOS_NOT_AVAILABLE} |
| 主报告 | `result/round20/ROUND20_FACTOR_RESEARCH_REPORT.md` |

"""
    text = existing.rstrip() + "\n" + block
    TRACKER_PATH.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def run_report(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    ctx = gather()
    write_md(FINAL_REPORT, _build(ctx))
    _append_tracker(ctx)
    bt = ctx.get("bt") or {}
    par = ctx.get("parity") or {}
    freeze = ctx.get("freeze") or {}
    combo = ctx.get("combo") or {}
    model = ctx.get("model") or {}
    seed = ctx.get("seed") or {}
    payload = {
        "status": "DONE",
        "ROUND20_STATUS": "COMPLETE",
        "decision": bt.get("decision") or KEEP_BASELINE_V5,
        "lock": bt.get("lock"),
        "current": CURRENT_BASELINE,
        "alias": CURRENT_ALIAS,
        "parent": PARENT_V4,
        "auto_promote": False,
        "parity": par.get("status"),
        "sha256": freeze.get("sha256"),
        "combo_n": combo.get("n_tested"),
        "best": combo.get("best"),
        "m2_status": model.get("m2_status"),
        "finalists": model.get("finalists"),
        "a00_a01": (model.get("a00_a01_parity") or {}).get("agree"),
        "native_seed": (seed.get("native_audit") or {}).get("status"),
        "s4": (seed.get("s4") or {}).get("status"),
        "forward": FORWARD_OOS_NOT_AVAILABLE,
        "final_report": str(FINAL_REPORT),
    }
    dump_json(DOSSIER_JSON, payload)
    return payload


def print_part38(payload: dict[str, Any] | None = None) -> None:
    p = payload or load_json(DOSSIER_JSON)
    print("", flush=True)
    print("===== ROUND20 PART 38 =====", flush=True)
    print(f"ROUND20_STATUS={p.get('ROUND20_STATUS') or 'COMPLETE'}", flush=True)
    print("", flush=True)
    print("Current baseline:", flush=True)
    print(f"{CURRENT_BASELINE} = {CURRENT_ALIAS}", flush=True)
    print("AUTO_PROMOTE=False CURRENT unchanged", flush=True)
    print("", flush=True)
    print("Decision / lock:", flush=True)
    print(f"{p.get('decision')} / {p.get('lock')}", flush=True)
    print("", flush=True)
    print("Parity:", flush=True)
    print(p.get("parity"), flush=True)
    print("", flush=True)
    print("Best combo:", flush=True)
    print(p.get("best"), flush=True)
    print("", flush=True)
    print("M2 / finalists / A00-A01:", flush=True)
    print(f"{p.get('m2_status')} / {p.get('finalists')} / {p.get('a00_a01')}", flush=True)
    print("", flush=True)
    print("Seed native / S4:", flush=True)
    print(f"{p.get('native_seed')} / {p.get('s4')}", flush=True)
    print("", flush=True)
    print("Forward OOS:", flush=True)
    print(p.get("forward"), flush=True)
    print("", flush=True)
    print("Final Report:", flush=True)
    print(str(p.get("final_report") or FINAL_REPORT), flush=True)
    print("===== END PART 38 =====", flush=True)


__all__ = ["DOSSIER_JSON", "print_part38", "run_report"]
