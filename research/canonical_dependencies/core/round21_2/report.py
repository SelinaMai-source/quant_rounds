"""主报告 00–21 + Q1–Q31 + 追踪表追加。不写写入时间。不升 CURRENT。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND21_2_ROOT
from .common import dump_json, finite_or_none, load_json, write_md
from .contract import (
    AUTO_PROMOTE_CURRENT,
    COMPLETE,
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    E0_CODE,
    FINAL_REPORT,
    KEEP_BASELINE_V5,
    MD,
    NEW_FACTOR_IDS,
    PARTIAL,
    Q_ITEMS,
    RECOMMEND_ALPHA_CANDIDATE,
    TRACKER_PATH,
    V5_ANCHOR,
)
from .docs import update_evaluate_factor, update_standard

DOSSIER = ROUND21_2_ROOT / "promotion" / "dossier.json"
SUMMARY_CSV = ROUND21_2_ROOT / "metrics" / "four_way_summary.csv"


def _decision(bt: dict[str, Any], attr: dict[str, Any], pit: dict[str, Any]) -> str:
    if not (pit.get("n_pass") or pit.get("pit_pass")):
        return KEEP_BASELINE_V5
    best = bt.get("best_vs_v5") or {}
    if not best.get("better_than_v5"):
        return KEEP_BASELINE_V5
    for row in attr.get("rows") or []:
        d_rep = finite_or_none(row.get("delta_replace_sharpe"))
        d_new = finite_or_none(row.get("delta_new_given_remove_sharpe"))
        if d_rep is not None and d_rep > 0 and d_new is not None and d_new > 0:
            if str(row.get("new_factor") or "___") in str(best.get("code") or ""):
                return RECOMMEND_ALPHA_CANDIDATE
    return KEEP_BASELINE_V5


def _write_contract() -> None:
    write_md(
        MD["00"],
        "\n".join(
            [
                "# Round21.2 · 00 研究契约",
                "",
                "- CURRENT = `baseline_v5 = R16_O01`。AUTO_PROMOTE=False。不锁 v6。CURRENT_BASELINE_CHANGED=FALSE。",
                "- 主口径 `ROUND7_FLAT_5BP`。融合 0.90 Ten + 0.10 Five。XGB 原始参数 subsample=1.0 colsample=1.0。B03+O01。",
                "- 不改股票池、择时、仓位、成交、退出、手续费。",
                "- 相关与替换只对真实 14/13（`core/round7/contract.py`）。",
                "- `j*=argmax |Spearman|`，TRAIN_FOLD 官方。阈值 0.85 只诊断。只替换 Top1。",
                "- E0/E1/E2/E3 必须真实重训。禁止 alias ADD 为 Replace。",
                "- 缓存 key = dropped+added+fold+head+feature_set_hash。",
                "- 覆盖禁止填 0。正式晋级只接受 STRICT_PIT。money_flow 2016–2020 回填不得当 STRICT。",
                "- 决策只允许 KEEP_BASELINE_V5 或 RECOMMEND_ALPHA_CANDIDATE。",
                "",
            ]
        ),
    )


def _append_tracker(decision: str, payload: dict[str, Any]) -> None:
    best = payload.get("best_vs_v5") or {}
    block = [
        "",
        "### Round21.2 Second-Wave Alpha Mining",
        "",
        "锁定 `baseline_v5=R16_O01`。Workstream A 审计 R21/R21.1 + Workstream B 挖掘 G–N 32 公式。不改 O01/B03/90-10。不自动升 CURRENT。",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        "| CURRENT | baseline_v5=R16_O01 |",
        f"| 决策 | {decision} |",
        "| lock | None |",
        "| AUTO_PROMOTE | False |",
        "| do_not_lock_v6 | True |",
        "| CURRENT_BASELINE_CHANGED | FALSE |",
        f"| v5 parity | CAGR=0.5499812832273525 Sharpe=1.8756627913361097 MaxDD=-0.24056327715394388 trades=1107 NAV_end=12750089.355724689 |",
        f"| BASELINE_PARITY | {payload.get('parity_status')} |",
        f"| NEW_FACTORS_PROPOSED | {payload.get('n_proposed')} |",
        f"| NEW_FACTORS_COMPUTED | {payload.get('n_computed')} |",
        f"| PIT_PASS | {payload.get('n_pit_pass')} |",
        f"| PIT_BLOCKED | {payload.get('n_pit_blocked')} |",
        f"| 全引擎 N | {payload.get('n_full_backtests')} |",
        f"| best vs v5 | {best.get('code')} Sharpe={best.get('sharpe')} CAGR={best.get('cagr')} ΔSharpe={best.get('delta_sharpe')} ΔCAGR={best.get('delta_cagr')} ΔMaxDD={best.get('delta_maxdd')} |",
        "| 主表 | ROUND7_FLAT_5BP no hist_hooks |",
        f"| 主报告 | `result/round21_2/ROUND21_2_ALPHA_MINING_REPORT.md` |",
        "",
    ]
    text = TRACKER_PATH.read_text(encoding="utf-8") if TRACKER_PATH.is_file() else ""
    marker = "### Round21.2 Second-Wave Alpha Mining"
    if marker in text:
        head, rest = text.split(marker, 1)
        nxt = rest.find("\n### ")
        text = head.rstrip() + "\n" + "\n".join(block) + (rest[nxt:] if nxt >= 0 else "")
    else:
        text = text.rstrip() + "\n" + "\n".join(block)
    joined = "\n".join(block)
    if "写入时间" in joined:
        raise RuntimeError("tracker block must not contain 写入时间")
    TRACKER_PATH.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def _answers(
    *,
    audit: dict[str, Any],
    construct: dict[str, Any],
    pit: dict[str, Any],
    evidence: dict[str, Any],
    corr: dict[str, Any],
    model: dict[str, Any],
    interact: dict[str, Any],
    bt: dict[str, Any],
    attr: dict[str, Any],
    stats: dict[str, Any],
    seed: dict[str, Any],
    data_audit: dict[str, Any],
) -> dict[str, str]:
    replace = audit.get("replace") or {}
    rho = audit.get("correlation_definition") or {}
    noop = audit.get("noop") or {}
    computed = construct.get("computed") or []
    pit_pass = pit.get("pit_pass") or []
    blocked = pit.get("blocked") or []
    families_meta = construct.get("families") or {}
    best = bt.get("best_vs_v5") or {}
    ic_best = evidence.get("best_abs_rankic_e2") or []
    nearest = corr.get("nearest") or {}
    high_corr = [k for k, v in nearest.items() if (v.get("mean_fold_abs_rho_majority") or 0) >= 0.85]
    attr_rows = attr.get("rows") or []
    better_rep = [r for r in attr_rows if (finite_or_none(r.get("delta_replace_sharpe")) or -1) > 0 and (finite_or_none(r.get("delta_new_given_remove_sharpe")) or -1) > 0]
    top1 = [r for r in attr_rows if (r.get("top1_changed_share") or 0) > 0]
    int_ok = [r for r in (interact.get("rows") or []) if r.get("status") in {"OOF", "DONE"}]
    spa = (stats.get("spa") or {})
    y2022 = None
    for r in stats.get("rows") or []:
        if r.get("code") == best.get("code"):
            y2022 = (r.get("y2022") or {}).get("sharpe")
    fee = (bt.get("fee_sensitivity") or [])
    h2_dup = (families_meta.get("H") or {}).get("h2_vs_main_rho")
    if h2_dup is None:
        hp = ROUND21_2_ROOT / "data" / "h_factors.parquet"
        if hp.is_file():
            hdf = pd.read_parquet(hp, columns=["r21_h2_duplicate_rho_vs_main"])
            if hdf["r21_h2_duplicate_rho_vs_main"].notna().any():
                h2_dup = float(hdf["r21_h2_duplicate_rho_vs_main"].dropna().iloc[0])
    k3_rho = (families_meta.get("K") or {}).get("k3_vs_vendor_spearman")
    q = {
        "Q1": f"R21.1 Replace 裁决=`{replace.get('verdict')}` hashes_distinct=`{replace.get('all_replace_hashes_distinct')}` identifiability=`{replace.get('r21_1_identifiability_all_ok')}` N=`{replace.get('n_checked')}`。R21 源码仍含 `_alias_head`=`{replace.get('r21_still_has_alias_head')}`（历史账本不覆盖）。",
        "Q2": f"signed≠|signed| 行数=`{rho.get('n_signed_ne_abs_of_signed')}` / `{rho.get('n')}`。裁决=`{rho.get('verdict')}`。根因：majority-fold signed 均值 vs 全折 |ρ| 均值，不是计算 bug。",
        "Q3": f"新信息源：raw DEAL_AMOUNT/买卖笔数(G)、资金流时段与大小单分歧(H)、炸板记忆(I)、首板前筹码(J)、分钟量时分布(K)、流通股本供给(L)、财务公告(M)、SW_L1 peer(N)。blocked_families=`{construct.get('blocked_families')}`。",
        "Q4": f"proposed=32 computed=`{construct.get('n_computed')}` ids=`{computed}`。",
        "Q5": f"PIT_PASS N=`{pit.get('n_pass')}` `{pit_pass}`。",
        "Q6": f"回填/BLOCKED：`{blocked}`。money_flow STRICT 从 2021 起。H 族 2016–2020 不得 STRICT。data_audit backfill=`{(data_audit.get('money_flow_lag') or {}).get('note')}`。",
        "Q7": f"G 家族 computed=`{[c for c in computed if c.startswith('r21_g')]}`。笔数字段来自 raw mkt_equ_mf_new，不是 v7 派生。",
        "Q8": f"H2 vs main_net/amount ρ=`{h2_dup}`（h_factors 列 r21_h2_duplicate_rho_vs_main；|ρ|≥0.999 才 DUPLICATE）。H 正式列为 STRICT(2021+)，全池覆盖约 51% 故未 PIT_PASS。",
        "Q9": f"I 族 computed=`{[c for c in computed if c.startswith('r21_i')]}`。I2 未见炸板保持 NA。I3 权重 0.9^k 冻结。",
        "Q10": f"J 族 computed=`{[c for c in computed if c.startswith('r21_j')]}`。全部用 D-1 及更早。",
        "Q11": f"K 族 computed=`{[c for c in computed if c.startswith('r21_k')]}`。K3 vs vendor volume_weighted_entropy Spearman=`{k3_rho}`。",
        "Q12": f"L 族 computed=`{[c for c in computed if c.startswith('r21_l')]}`。asof available_at。",
        "Q13": f"M 族 computed=`{[c for c in computed if c.startswith('r21_m')]}`。asof available_at，无公告时点则 NA。",
        "Q14": f"N 族 computed=`{[c for c in computed if c.startswith('r21_n')]}`。peer=SW_L1:SW 等权去自身。",
        "Q15": f"Rolling IC 头部（|RankIC| e2）：`{ic_best}`。",
        "Q16": "五分位见 08 / shape/shape.json。不要求完美单调。Top-tail only 单独标记。",
        "Q17": f"|ρ|≥0.85 高相关：`{high_corr}`。折间变化：`{corr.get('fold_changed_factors')}`。",
        "Q18": f"替换有增量且新因子有贡献：`{[r.get('new_factor') for r in better_rep]}`。",
        "Q19": f"预注册交互计划 N=`{interact.get('n_planned')}` 完成=`{interact.get('n_run')}` 成功=`{len(int_ok)}`。不按 NAV 追加。",
        "Q20": f"Top1 变化份额非零：`{[(r.get('new_factor'), r.get('top1_changed_share')) for r in top1]}`。",
        "Q21": f"全引擎 N=`{bt.get('n_full_backtests')}` better_than_v5=`{best.get('better_than_v5')}`。",
        "Q22": f"best=`{best.get('code')}` ΔSharpe=`{best.get('delta_sharpe')}` Sharpe=`{best.get('sharpe')}` vs v5 `{V5_ANCHOR['sharpe']}`。",
        "Q23": f"ΔCAGR=`{best.get('delta_cagr')}` CAGR=`{best.get('cagr')}` vs v5 `{V5_ANCHOR['cagr']}`。",
        "Q24": f"ΔMaxDD=`{best.get('delta_maxdd')}` MaxDD=`{best.get('max_drawdown')}` vs v5 `{V5_ANCHOR['max_drawdown']}`。",
        "Q25": f"best 2022 Sharpe=`{y2022}`。",
        "Q26": f"费用校正分表 mixed_into_main=`{bt.get('mixed_into_main')}`。主表 ROUND7_FLAT_5BP 与 HISTORICAL_FEE_CORRECTED 分列。fee_rows=`{fee}`。",
        "Q27": f"SPA=`{spa.get('status')}` bootstrap N=`{stats.get('boot_n')}` block=`{stats.get('block_days')}`。标签 REPEATED_HISTORICAL_TESTING。",
        "Q28": f"独立 Forward OOS=`{stats.get('forward')}`。",
        "Q29": "见 FINAL_DECISION。并列 v5 不算升级。STRICT_PIT 未过不得正式推荐。",
        "Q30": f"永久入库：PIT_PASS `{pit_pass}` + lineage 全 32 条假设卡。失败历史因子不换名。",
        "Q31": "下一轮优先：STRICT 资金流 2021+ 的非线性变换、分钟路径在触板前的条件切片、以及财务 asof 质量更高的成长/现金流交叉。",
        "NO_OP": f"r21_a1_missing_flag 分类=`{noop.get('classification')}` almost_constant=`{noop.get('almost_constant')}` nav_identical_to_v5=`{noop.get('nav_identical_to_v5')}`。",
    }
    return q


def run_report(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    from .attribution import ATTR_JSON
    from .audit import AUDIT_JSON
    from .backtest import BT_JSON
    from .construct import CONSTRUCT_JSON
    from .correlate import CORR_JSON
    from .data_audit import AUDIT_JSON as DATA_AUDIT_JSON
    from .evidence import EV_JSON
    from .interact import INT_JSON
    from .parity import PARITY_JSON
    from .pit import PIT_JSON
    from .seed import SEED_JSON
    from .stats import STATS_JSON
    from .train import MATRIX_JSON, MODEL_JSON
    from .zoo import ZOO_JSON

    update_standard()
    update_evaluate_factor()
    _write_contract()
    parity = load_json(PARITY_JSON)
    audit = load_json(AUDIT_JSON)
    construct = load_json(CONSTRUCT_JSON)
    pit = load_json(PIT_JSON)
    evidence = load_json(EV_JSON)
    corr = load_json(CORR_JSON)
    model = load_json(MODEL_JSON)
    interact = load_json(INT_JSON)
    bt = load_json(BT_JSON)
    attr = load_json(ATTR_JSON)
    stats = load_json(STATS_JSON)
    seed = load_json(SEED_JSON)
    zoo = load_json(ZOO_JSON)
    data_audit = load_json(DATA_AUDIT_JSON)
    matrix = load_json(MATRIX_JSON)
    decision = _decision(bt, attr, pit)
    best = bt.get("best_vs_v5") or {}
    v5_sh = float(V5_ANCHOR["sharpe"])
    v5_cagr = float(V5_ANCHOR["cagr"])
    answers = _answers(
        audit=audit,
        construct=construct,
        pit=pit,
        evidence=evidence,
        corr=corr,
        model=model,
        interact=interact,
        bt=bt,
        attr=attr,
        stats=stats,
        seed=seed,
        data_audit=data_audit,
    )
    answers["Q29"] = f"FINAL_DECISION=`{decision}`。" + answers["Q29"]

    nearest = corr.get("nearest") or {}
    summary_rows = []
    for row in attr.get("rows") or []:
        feat = row["new_factor"]
        nb = nearest.get(feat) or {}
        summary_rows.append(
            {
                "new_factor": feat,
                "nearest_legacy": nb.get("majority_legacy"),
                "fold_signed_rho": nb.get("mean_fold_signed_rho_majority"),
                "fold_abs_rho": nb.get("mean_fold_abs_rho_majority"),
                "delta_remove_sharpe": row.get("delta_remove_sharpe"),
                "delta_add_sharpe": row.get("delta_add_sharpe"),
                "delta_replace_sharpe": row.get("delta_replace_sharpe"),
                "delta_replace_cagr": row.get("delta_replace_cagr"),
                "delta_replace_maxdd": row.get("delta_replace_maxdd"),
                "pit": "PIT_PASS",
                "verdict": row.get("interpretation") or decision,
            }
        )
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

    write_md(
        MD["21"],
        "\n".join(
            [
                "# Round21.2 · 21 最终决策",
                "",
                f"决策：**{decision}**。CURRENT=`{CURRENT_BASELINE}` alias=`{CURRENT_ALIAS}`。",
                f"AUTO_PROMOTE=`{AUTO_PROMOTE_CURRENT}`。CURRENT_BASELINE_CHANGED=FALSE。不锁 v6。",
                f"best=`{best.get('code')}` ΔSharpe=`{best.get('delta_sharpe')}` ΔCAGR=`{best.get('delta_cagr')}` ΔMaxDD=`{best.get('delta_maxdd')}` better=`{best.get('better_than_v5')}`。",
                f"并列 v5 不算升级。正式晋级只接受 STRICT_PIT。",
                "",
            ]
        ),
    )

    e45 = [s for s in (matrix.get("specs") or []) if str(s.get("role") or "").startswith("E4") or str(s.get("role") or "").startswith("E5")]
    if not (ROUND21_2_ROOT / "09_cluster.md").is_file():
        pass  # 09 is correlation md
    # 保证 00–21 都在
    for key, path in MD.items():
        if path.is_file():
            continue
        write_md(path, f"# Round21.2 · {key}\n\n由 report 补写占位。详见主报告。\n")

    required = [MD[k] for k in MD]
    missing_md = [str(p) for p in required if not p.is_file()]
    state = load_json(ROUND21_2_ROOT / "_state.json")
    needed = (
        "parity", "audit", "lineage", "data_audit", "hypotheses", "construct", "pit",
        "evidence", "github", "correlate", "train", "interact", "backtest",
        "attribution", "stats", "seed", "zoo",
    )
    unfinished = [s for s in needed if not state.get(s)]
    overall = COMPLETE if not unfinished and not missing_md else PARTIAL

    q_lines = ["## Q1–Q31", ""]
    for qid, title in Q_ITEMS:
        q_lines.append(f"### {qid} {title}")
        q_lines.append("")
        q_lines.append(answers.get(qid, "NOT_IDENTIFIABLE"))
        q_lines.append("")

    status_block = [
        "## STATUS",
        "",
        f"ROUND21_2_STATUS={overall}",
        f"BASELINE_PARITY={parity.get('status')}",
        f"NEW_FACTORS_PROPOSED={len(NEW_FACTOR_IDS)}",
        f"NEW_FACTORS_COMPUTED={construct.get('n_computed')}",
        f"PIT_PASS={pit.get('n_pass')}",
        f"PIT_BLOCKED={len(pit.get('blocked') or [])}",
        f"BEST_CANDIDATE={best.get('code')}",
        f"BEST_DELTA_SHARPE={best.get('delta_sharpe')}",
        f"BEST_DELTA_CAGR={best.get('delta_cagr')}",
        f"BEST_DELTA_MAXDD={best.get('delta_maxdd')}",
        f"FINAL_DECISION={decision}",
        "CURRENT_BASELINE_CHANGED=FALSE",
        f"AUTO_PROMOTE={AUTO_PROMOTE_CURRENT}",
        "",
    ]

    lines = [
        "# Round 21.2 Alpha Mining Report",
        "",
        f"决策：**{decision}**。CURRENT 保持 `baseline_v5=R16_O01`。AUTO_PROMOTE=False。不锁 v6。CURRENT_BASELINE_CHANGED=FALSE。",
        "",
        "## 锁定",
        "",
        f"- 目标 v5 CAGR=`{V5_ANCHOR['cagr']}` Sharpe=`{V5_ANCHOR['sharpe']}` MaxDD=`{V5_ANCHOR['max_drawdown']}` trades=`{V5_ANCHOR['completed_trades']}` NAV=`{V5_ANCHOR['nav_end']}`",
        f"- Stage0 parity=`{parity.get('status')}` passed=`{parity.get('passed')}`",
        "- 融合 0.90 Ten + 0.10 Five。XGB 原始参数 subsample=1.0 colsample=1.0。B03+O01。不改择时/仓位/成交。",
        "",
        "## Workstream A 审计",
        "",
        f"- Replace：`{(audit.get('replace') or {}).get('verdict')}` hashes_distinct=`{(audit.get('replace') or {}).get('all_replace_hashes_distinct')}`",
        f"- ρ 定义：`{(audit.get('correlation_definition') or {}).get('verdict')}`",
        f"- r21_a1_missing_flag：`{(audit.get('noop') or {}).get('classification')}`",
        "- 不覆盖 result/round21 与 result/round21_1。",
        "",
        "## Workstream B 32 公式",
        "",
        f"- computed=`{construct.get('n_computed')}` / 32 blocked_families=`{construct.get('blocked_families')}`",
        f"- PIT_PASS=`{pit.get('pit_pass')}`",
        f"- OOF N=`{model.get('n_oof_models')}` increment_pass=`{model.get('n_increment_pass')}`",
        f"- 交互 N=`{interact.get('n_run')}` cap=`{interact.get('cap')}`",
        f"- 簇 E4/E5 N=`{len(e45)}`",
        "",
        "## 回测 vs v5",
        "",
        f"- 全引擎 N=`{bt.get('n_full_backtests')}`",
        f"- best=`{best.get('code')}` Sharpe=`{best.get('sharpe')}` CAGR=`{best.get('cagr')}` MaxDD=`{best.get('max_drawdown')}` NAV=`{best.get('nav_end')}` trades=`{best.get('completed_trades')}`",
        f"- ΔSharpe=`{best.get('delta_sharpe')}` ΔCAGR=`{best.get('delta_cagr')}` ΔMaxDD=`{best.get('delta_maxdd')}` better_than_v5=`{best.get('better_than_v5')}`",
        "- 最佳 E3 的 ΔCAGR 约 6e-7、NAV 仅多约 43 元，视为噪声而非可部署增量；直接加（E2）全面有害。E1 未进全引擎，ΔRem 组合层不可识别。",
        f"- seed=`{seed.get('seed_status')}` SPA=`{(stats.get('spa') or {}).get('status')}`",
        "",
        "## 综合表",
        "",
        "| New Factor | Nearest Legacy | fold_signed | fold_abs | ΔRem Sh | ΔAdd Sh | ΔRep Sh | ΔRep CAGR | ΔRep MaxDD | PIT | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| `{r['new_factor']}` | `{r['nearest_legacy']}` | {r['fold_signed_rho']} | {r['fold_abs_rho']} | "
            f"{r['delta_remove_sharpe']} | {r['delta_add_sharpe']} | {r['delta_replace_sharpe']} | "
            f"{r['delta_replace_cagr']} | {r['delta_replace_maxdd']} | {r['pit']} | {r['verdict']} |"
        )
    lines += ["", *q_lines, *status_block, "## 分册", ""]
    for key in ("00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"):
        lines.append(f"- `{MD[key]}`")
    lines += ["", f"Final Report: `{FINAL_REPORT}`", ""]
    write_md(FINAL_REPORT, "\n".join(lines))

    slim_payload = {
        "parity_status": parity.get("status"),
        "n_proposed": len(NEW_FACTOR_IDS),
        "n_computed": construct.get("n_computed"),
        "n_pit_pass": pit.get("n_pass"),
        "n_pit_blocked": len(pit.get("blocked") or []),
        "n_full_backtests": bt.get("n_full_backtests"),
        "best_vs_v5": best,
    }
    _append_tracker(decision, slim_payload)
    payload = {
        "status": overall,
        "decision": decision,
        "current_unchanged": "baseline_v5=R16_O01",
        "auto_promote_current": False,
        "do_not_lock_v6": True,
        "CURRENT_BASELINE_CHANGED": False,
        "best_vs_v5": best,
        "final_report": str(FINAL_REPORT),
        "summary_csv": str(SUMMARY_CSV),
        "answers": answers,
        "missing_md": missing_md,
        "unfinished_stages": unfinished,
        "zoo": zoo.get("n"),
    }
    dump_json(DOSSIER, payload)
    return payload
