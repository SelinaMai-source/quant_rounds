"""主报告 00/09 + 综合表 + 追踪表追加。不写写入时间。不升 CURRENT。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .common import dump_json, finite_or_none, load_json, write_md
from .contract import (
    CLUSTER_MD,
    CLUSTER_SPEARMAN_ABS,
    CONTRACT_MD,
    E0_CODE,
    FINAL_REPORT,
    KEEP_BASELINE_V5,
    RECOMMEND_NEW_ALPHA_CANDIDATE,
    TRACKER_PATH,
    V5_ANCHOR,
)
from .docs import update_evaluate_factor, update_standard

from ..paths import ROUND21_1_ROOT

DOSSIER = ROUND21_1_ROOT / "promotion" / "dossier.json"
SUMMARY_CSV = ROUND21_1_ROOT / "metrics" / "four_way_summary.csv"


def _decision(bt: dict[str, Any], attr: dict[str, Any]) -> str:
    best = bt.get("best_vs_v5") or {}
    if not best.get("better_than_v5"):
        return KEEP_BASELINE_V5
    # E3>E0 且新因子有贡献，才推荐
    for row in attr.get("rows") or []:
        d_rep = finite_or_none(row.get("delta_replace_sharpe"))
        d_new = finite_or_none(row.get("delta_new_given_remove_sharpe"))
        if d_rep is not None and d_rep > 0 and d_new is not None and d_new > 0:
            if str(best.get("code") or "").find(str(row.get("new_factor") or "___")) >= 0:
                return RECOMMEND_NEW_ALPHA_CANDIDATE
    return KEEP_BASELINE_V5


def _write_contract() -> None:
    write_md(
        CONTRACT_MD,
        "\n".join(
            [
                "# Round21.1 · 00 研究契约",
                "",
                "- CURRENT = `baseline_v5 = R16_O01`。AUTO_PROMOTE=False。不锁 v6。",
                "- 主口径 `ROUND7_FLAT_5BP`。融合 0.90 Ten + 0.10 Five。XGB 原始参数。B03+O01。",
                "- 不改股票池、择时、仓位、成交、退出、手续费。",
                "- 相关与替换只对真实 14/13（`core/round7/contract.py`）。",
                "- `j*=argmax |Spearman|`，TRAIN_FOLD 官方。阈值 0.85 只诊断。只替换 Top1。",
                "- E0/E1/E2/E3 必须真实重训。禁止 alias ADD 为 Replace。",
                "- 缓存 key = dropped+added+fold+head+feature_set_hash。",
                "- 覆盖禁止填 0。E 族仍 BLOCKED_PIT。",
                "- 决策只允许 KEEP_BASELINE_V5 或 RECOMMEND_NEW_ALPHA_CANDIDATE。",
                "",
            ]
        ),
    )


def _write_cluster(matrix: dict[str, Any]) -> None:
    e45 = [s for s in (matrix.get("specs") or []) if str(s.get("role") or "").startswith("E4") or str(s.get("role") or "").startswith("E5")]
    lines = [
        "# Round21.1 · 09 簇替换（条件触发）",
        "",
        f"仅当训练窗 `|Spearman|≥{CLUSTER_SPEARMAN_ABS}` 且簇 ≥2。禁止按回测收益选删除名单。",
        f"触发 N=`{len(e45)}`。",
        "",
    ]
    if not e45:
        lines.append("本轮 **未触发** E4/E5。")
    for s in e45:
        lines.append(f"- `{s['code']}` role={s.get('role')} cluster=`{s.get('cluster')}`")
    write_md(CLUSTER_MD, "\n".join(lines))


def _append_tracker(decision: str, bt: dict[str, Any], audit: dict[str, Any], cov: dict[str, Any]) -> None:
    best = bt.get("best_vs_v5") or {}
    block = [
        "",
        "### Round21.1 Coverage Forensics × Nearest Replace",
        "",
        "锁定 `baseline_v5=R16_O01`。覆盖法医 + 真实 14/13 最近邻四路 + 全引擎。不改 O01/B03/90-10。不自动升 CURRENT。",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        "| CURRENT | baseline_v5=R16_O01 |",
        f"| 决策 | {decision} |",
        "| lock | None |",
        "| AUTO_PROMOTE | False |",
        "| do_not_lock_v6 | True |",
        f"| v5 parity | CAGR=0.5499812832273525 Sharpe=1.8756627913361097 MaxDD=-0.24056327715394388 trades=1107 NAV_end=12750089.355724689 |",
        f"| ADD/REP 裁决 | {' + '.join(audit.get('verdicts') or [])} |",
        f"| 覆盖 FAIL 分类 N | {len(cov.get('rows') or [])} |",
        f"| 同伴 PIT_PASS | {cov.get('companion_pit_pass')} |",
        f"| 全引擎 N | {bt.get('n_full_backtests')} |",
        f"| best vs v5 | {best.get('code')} Sharpe={best.get('sharpe')} CAGR={best.get('cagr')} ΔSharpe={best.get('delta_sharpe')} ΔCAGR={best.get('delta_cagr')} |",
        "| 主表 | ROUND7_FLAT_5BP no hist_hooks |",
        f"| 主报告 | `result/round21_1/ROUND21_1_RESEARCH_REPORT.md` |",
        "",
    ]
    text = TRACKER_PATH.read_text(encoding="utf-8") if TRACKER_PATH.is_file() else ""
    if "### Round21.1 Coverage Forensics × Nearest Replace" in text:
        head, rest = text.split("### Round21.1 Coverage Forensics × Nearest Replace", 1)
        # 丢掉旧块到文末（本轮覆盖重写）
        text = head.rstrip() + "\n" + "\n".join(block)
    else:
        text = text.rstrip() + "\n" + "\n".join(block)
    if "写入时间" in "\n".join(block):
        raise RuntimeError("tracker block must not contain 写入时间")
    TRACKER_PATH.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def run_report(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    from .attribution import ATTR_JSON
    from .audit import AUDIT_JSON
    from .backtest import BT_JSON
    from .conditional import COND_JSON
    from .correlate import CORR_JSON
    from .coverage import COV_JSON
    from .parity import PARITY_JSON
    from .seed import SEED_JSON
    from .stats import STATS_JSON
    from .train import MATRIX_JSON, MODEL_JSON
    from .zoo import ZOO_JSON

    from .train import backfill_artifacts

    update_standard()
    update_evaluate_factor()
    backfill_artifacts()
    _write_contract()
    matrix = load_json(MATRIX_JSON)
    _write_cluster(matrix)
    bt = load_json(BT_JSON)
    attr = load_json(ATTR_JSON)
    audit = load_json(AUDIT_JSON)
    cov = load_json(COV_JSON)
    corr = load_json(CORR_JSON)
    cond = load_json(COND_JSON)
    model = load_json(MODEL_JSON)
    parity = load_json(PARITY_JSON)
    seed = load_json(SEED_JSON)
    stats = load_json(STATS_JSON)
    zoo = load_json(ZOO_JSON)
    decision = _decision(bt, attr)
    best = bt.get("best_vs_v5") or {}
    v5_sh = float(V5_ANCHOR["sharpe"])
    v5_cagr = float(V5_ANCHOR["cagr"])
    distinct = []
    for r in bt.get("main_rows") or []:
        if r.get("code") == E0_CODE:
            continue
        sh, cg = finite_or_none(r.get("sharpe")), finite_or_none(r.get("cagr"))
        if sh is None:
            continue
        if abs(sh - v5_sh) < 1e-12 and (cg is None or abs(cg - v5_cagr) < 1e-12):
            continue
        distinct.append(r)
    closest = max(distinct, key=lambda r: float(r["sharpe"])) if distinct else None
    closest_payload = None
    if closest:
        closest_payload = {
            "code": closest.get("code"),
            "sharpe": closest.get("sharpe"),
            "cagr": closest.get("cagr"),
            "max_drawdown": closest.get("max_drawdown"),
            "nav_end": closest.get("nav_end"),
            "completed_trades": closest.get("completed_trades"),
            "delta_sharpe": float(closest["sharpe"]) - v5_sh,
            "delta_cagr": None if closest.get("cagr") is None else float(closest["cagr"]) - v5_cagr,
            "note": "highest Sharpe among formal sleeves not numerically identical to v5",
        }

    summary_rows = []
    nearest = corr.get("nearest") or {}
    fold_signed: dict[str, list[float]] = {}
    fold_csv = ROUND21_1_ROOT / "correlation" / "fold_nearest_legacy_mapping.csv"
    if fold_csv.is_file():
        fmap = pd.read_csv(fold_csv)
        for _, fr in fmap.iterrows():
            feat = str(fr.get("new_factor") or "")
            maj = (nearest.get(feat) or {}).get("majority_legacy")
            if maj and str(fr.get("nearest_legacy")) == str(maj) and pd.notna(fr.get("spearman")):
                fold_signed.setdefault(feat, []).append(float(fr["spearman"]))
    cond_map = {}
    for r in cond.get("rows") or []:
        cond_map.setdefault(r["factor_id"], r if r.get("head") == "five" else cond_map.get(r["factor_id"], r))
    cov_map = {r["factor_id"]: r for r in (cov.get("rows") or [])}
    for row in attr.get("rows") or []:
        feat = row["new_factor"]
        nb = nearest.get(feat) or {}
        fold = nb.get("fold_nearest") or {}
        rhos = [v.get("abs_spearman") for v in fold.values() if isinstance(v, dict) and v.get("abs_spearman") is not None]
        signed_vals = fold_signed.get(feat) or []
        signed = float(sum(signed_vals) / len(signed_vals)) if signed_vals else None
        summary_rows.append(
            {
                "new_factor": feat,
                "nearest_legacy": nb.get("majority_legacy"),
                "signed_spearman": signed,
                "abs_spearman": float(sum(rhos) / len(rhos)) if rhos else None,
                "delta_remove_sharpe": row.get("delta_remove_sharpe"),
                "delta_add_sharpe": row.get("delta_add_sharpe"),
                "delta_replace_sharpe": row.get("delta_replace_sharpe"),
                "delta_replace_cagr": row.get("delta_replace_cagr"),
                "delta_replace_maxdd": row.get("delta_replace_maxdd"),
                "conditional_ic": (cond_map.get(feat) or {}).get("conditional_ic"),
                "pit": "PIT_PASS",
                "coverage_class": (cov_map.get(feat) or {}).get("classification"),
                "verdict": row.get("interpretation") or decision,
            }
        )
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

    reports = [
        str(CONTRACT_MD),
        str(ROUND21_1_ROOT / "00_v5_parity.md"),
        str(ROUND21_1_ROOT / "01_coverage_forensics.md"),
        str(ROUND21_1_ROOT / "02_round21_experiment_parity.md"),
        str(ROUND21_1_ROOT / "03_correlation.md"),
        str(ROUND21_1_ROOT / "04_fold_nearest.md"),
        str(ROUND21_1_ROOT / "05_four_way_train.md"),
        str(ROUND21_1_ROOT / "06_conditional_ic.md"),
        str(ROUND21_1_ROOT / "07_attribution.md"),
        str(ROUND21_1_ROOT / "08_walk_forward.md"),
        str(ROUND21_1_ROOT / "09_cluster.md"),
        str(ROUND21_1_ROOT / "10_seed_robustness.md"),
        str(ROUND21_1_ROOT / "11_backtest_decision.md"),
        str(ROUND21_1_ROOT / "12_stats.md"),
        str(ROUND21_1_ROOT / "13_factor_zoo.md"),
        str(FINAL_REPORT),
    ]
    lines = [
        "# Round 21.1 Research Report",
        "",
        f"决策：**{decision}**。CURRENT 保持 `baseline_v5=R16_O01`。AUTO_PROMOTE=False。不锁 v6。",
        "",
        "## 锁定",
        "",
        f"- 目标 v5 CAGR=`{V5_ANCHOR['cagr']}` Sharpe=`{V5_ANCHOR['sharpe']}` MaxDD=`{V5_ANCHOR['max_drawdown']}` trades=`{V5_ANCHOR['completed_trades']}` NAV=`{V5_ANCHOR['nav_end']}`",
        f"- Stage0 parity=`{parity.get('status')}`",
        "- 融合 0.90 Ten + 0.10 Five。XGB 原始参数。B03+O01。不改择时/仓位/成交。",
        "",
        "## ADD/REP 审计",
        "",
        f"- 裁决：`{' + '.join(audit.get('verdicts') or [])}`",
        f"- CACHE_COLLISION 排除：`{audit.get('cache_collision_excluded')}`",
        "- 根因：`core/round21/model.py` `_alias_head` + `cannot_delete_locked`。本轮修实现，不覆盖 R21。",
        "",
        "## 覆盖法医",
        "",
        f"- 原 FAIL 10：`{cov.get('fail_original')}`",
        f"- 同伴 PIT_PASS：`{cov.get('companion_pit_pass')}`",
        "- E 族仍 BLOCKED_PIT。禁止填 0。",
        "",
    ]
    for r in cov.get("rows") or []:
        lines.append(
            f"- `{r.get('factor_id')}` → **{r.get('classification')}** universe=`{r.get('universe_coverage')}` "
            f"cond=`{r.get('conditional_coverage')}` true_miss=`{r.get('true_missing_rate')}`"
        )
    lines += [
        "",
        "## 最近邻",
        "",
        f"- 折间变化：`{corr.get('fold_changed_factors')}`",
        "- 官方 TRAIN_FOLD。FULL_HISTORY 仅诊断。只替换 Top1。",
        "",
        "## 四路训练",
        "",
        f"- OOF 模型 N=`{model.get('n_oof_models')}` identifiability=`{(model.get('identifiability') or {}).get('all_ok')}`",
        f"- 增量门通过 N=`{model.get('n_increment_pass')}`",
        "",
        "## 回测 vs v5",
        "",
        f"- 全引擎 N=`{bt.get('n_full_backtests')}`",
        f"- best=`{best.get('code')}` Sharpe=`{best.get('sharpe')}` CAGR=`{best.get('cagr')}` MaxDD=`{best.get('max_drawdown')}`",
        f"- ΔSharpe=`{best.get('delta_sharpe')}` ΔCAGR=`{best.get('delta_cagr')}` better_than_v5=`{best.get('better_than_v5')}`（并列 v5 不算升级）",
        f"- closest_distinct=`{(closest_payload or {}).get('code')}` Sharpe=`{(closest_payload or {}).get('sharpe')}` "
        f"ΔSharpe=`{(closest_payload or {}).get('delta_sharpe')}` ΔCAGR=`{(closest_payload or {}).get('delta_cagr')}`",
        f"- seed=`{seed.get('seed_status')}` SPA=`{(stats.get('spa') or {}).get('status')}`",
        "",
        "## 综合表",
        "",
        "| New Factor | Nearest Legacy | Signed ρ | |ρ| | ΔRem Sh | ΔAdd Sh | ΔRep Sh | ΔRep CAGR | ΔRep MaxDD | Cond IC | PIT | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| `{r['new_factor']}` | `{r['nearest_legacy']}` | {r['signed_spearman']} | {r['abs_spearman']} | "
            f"{r['delta_remove_sharpe']} | {r['delta_add_sharpe']} | {r['delta_replace_sharpe']} | "
            f"{r['delta_replace_cagr']} | {r['delta_replace_maxdd']} | {r['conditional_ic']} | {r['pit']} | {r['verdict']} |"
        )
    lines += [
        "",
        "## 分册",
        "",
    ]
    for p in reports[:-1]:
        lines.append(f"- `{p}`")
    lines += [
        "",
        f"Final Report: `{FINAL_REPORT}`",
        "",
    ]
    write_md(FINAL_REPORT, "\n".join(lines))
    _append_tracker(decision, bt, audit, cov)
    payload = {
        "status": "DONE",
        "decision": decision,
        "current_unchanged": "baseline_v5=R16_O01",
        "auto_promote_current": False,
        "do_not_lock_v6": True,
        "best_vs_v5": best,
        "closest_distinct_vs_v5": closest_payload,
        "final_report": str(FINAL_REPORT),
        "reports": reports,
        "summary_csv": str(SUMMARY_CSV),
        "zoo": zoo.get("n"),
    }
    dump_json(DOSSIER, payload)
    return payload
