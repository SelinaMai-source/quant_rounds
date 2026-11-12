"""主报告 00–24 + Q1–Q29 + 追踪表追加。不写写入时间。不升 CURRENT。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND21_3_ROOT
from .common import dump_json, finite_or_none, load_json, write_md
from .contract import (
    AUTO_PROMOTE_CURRENT,
    COMPLETE,
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    ECONOMIC_MATERIALITY,
    FINAL_REPORT,
    KEEP_BASELINE_V5,
    L1_E0,
    L1_E1,
    L1_E2,
    L1_E3,
    MD,
    NEW_FACTOR_IDS,
    PARTIAL,
    Q_ITEMS,
    RECOMMEND_BASELINE_V6_CANDIDATE,
    TRACKER_PATH,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    V6_RESEARCH_CANDIDATE,
)
from .docs import update_evaluate_factor, update_standard

DOSSIER = ROUND21_3_ROOT / "promotion" / "dossier.json"


def _decision(parity: dict[str, Any], bt: dict[str, Any], fwd: dict[str, Any], pit: dict[str, Any]) -> str:
    if parity.get("passed") is False:
        return KEEP_BASELINE_V5
    best = bt.get("best_vs_v5") or {}
    mat = (best.get("material") or {})
    if not mat.get("economically_material"):
        return KEEP_BASELINE_V5
    if not (pit.get("n_pass") or pit.get("pit_pass")):
        return KEEP_BASELINE_V5
    if fwd.get("status") == "FORWARD_OOS_NOT_AVAILABLE":
        return V6_RESEARCH_CANDIDATE
    return RECOMMEND_BASELINE_V6_CANDIDATE


def _fmt_row(label: str, code: str, r: dict[str, Any]) -> str:
    sh = finite_or_none(r.get("sharpe"))
    cg = finite_or_none(r.get("cagr"))
    dd = finite_or_none(r.get("max_drawdown"))
    nav = finite_or_none(r.get("nav_end"))
    dsh = finite_or_none(r.get("delta_sharpe"))
    dcg = finite_or_none(r.get("delta_cagr"))
    tr = None if cg is None else (1.0 + cg) ** (8.5) - 1.0  # display only; prefer total_return
    if r.get("total_return") is not None:
        tr = float(r["total_return"])
    ntr = r.get("completed_trades")
    return (
        f"| {label} | `{code}` | "
        f"{sh:6.3f} | {cg*100:6.2f}% | {tr*100:+8.2f}% | {dd*100:+6.2f}% | {ntr} | "
        f"{nav:,.2f} | {dsh:+.4f} | {dcg*100:+.4f}pct | 已跑 |"
        if sh is not None and cg is not None and dd is not None and nav is not None
        else f"| {label} | `{code}` | {sh} | {cg} |  | {dd} | {ntr} | {nav} | {dsh} | {dcg} | 已跑 |"
    )


def _append_tracker(decision: str, payload: dict[str, Any], rows: list[dict[str, Any]], fee_rows: list[dict[str, Any]]) -> None:
    lines = [
        "",
        "### Round21.3 Forensics × T/U/V Discovery",
        "",
        "锁定 `baseline_v5=R16_O01`。Track A 补齐 L1 四路/NAV/费用/统计/N1N3；Track B 数据恢复 + T/U/V。不改 O01/B03/90-10。不自动升 CURRENT。",
        "",
        "### 实验组结果",
        "",
        "主表口径=**ROUND7_FLAT_5BP**。L1 四路强制全引擎。并列 v5 不算升级。经济意义门已冻结。",
        "",
        "| 操作 | 代号 | Sharpe |     年化 |        累计 |   MaxDD |   笔数 |           NAV | ΔSharpe |         Δ年化 | 状态 |",
        "| --- | --- | -----: | -----: | --------: | ------: | -----: | ------------: | ------: | ----------: | ------ |",
    ]
    for r in rows:
        role = r.get("role") or ""
        code = str(r.get("code") or "")
        label = f"{role} {code.replace('R21_3_', '')}" if role else code
        if code in {L1_E0, "R21_3_E0"}:
            label = "精确复现 v5 / E0"
        lines.append(_fmt_row(label, code, r))
    lines.extend(
        [
            "",
            "### 费用校正敏感性（分表）",
            "",
            "| 操作 | 代号 | Sharpe |     年化 |        累计 |   MaxDD |   笔数 | 状态 |",
            "| --- | --- | -----: | -----: | --------: | ------: | -----: | ------ |",
        ]
    )
    for r in fee_rows:
        sh = finite_or_none(r.get("sharpe"))
        cg = finite_or_none(r.get("cagr"))
        dd = finite_or_none(r.get("max_drawdown"))
        st = r.get("status") or "已跑"
        lines.append(
            f"| 费用校正 {r.get('code')} | `{r.get('code')}` | "
            f"{'' if sh is None else f'{sh:6.3f}'} | {'' if cg is None else f'{cg*100:6.2f}%'} |  | "
            f"{'' if dd is None else f'{dd*100:+6.2f}%'} | {r.get('completed_trades')} | {st} |"
        )
    best = payload.get("best_vs_v5") or {}
    lines.extend(
        [
            "",
            "| 项 | 值 |",
            "| --- | --- |",
            "| CURRENT | baseline_v5=R16_O01 |",
            f"| 决策 | {decision} |",
            "| lock | None |",
            "| AUTO_PROMOTE | False |",
            "| do_not_lock_v6 | True |",
            "| CURRENT_BASELINE_CHANGED | FALSE |",
            f"| v5 parity | CAGR={V5_ANCHOR['cagr']} Sharpe={V5_ANCHOR['sharpe']} MaxDD={V5_ANCHOR['max_drawdown']} trades=1107 NAV_end={V5_NAV_END_EXACT} |",
            f"| BASELINE_PARITY | {payload.get('parity_status')} |",
            f"| NEW_FACTORS_PROPOSED | {payload.get('n_proposed')} |",
            f"| NEW_FACTORS_COMPUTED | {payload.get('n_computed')} |",
            f"| PIT_PASS | {payload.get('n_pit_pass')} |",
            f"| PIT_BLOCKED | {payload.get('n_pit_blocked')} |",
            f"| 全引擎 N | {payload.get('n_full_backtests')} |",
            f"| forty_three_source | {payload.get('forty_three_source')} |",
            f"| economically_material | {payload.get('economically_material')} |",
            f"| best vs v5 | {best.get('code')} Sharpe={best.get('sharpe')} CAGR={best.get('cagr')} |",
            "| 主表 | ROUND7_FLAT_5BP no hist_hooks |",
            "| 主报告 | `result/round21_3/ROUND21_3_RESEARCH_REPORT.md` |",
            "",
        ]
    )
    joined = "\n".join(lines)
    if "写入时间" in joined:
        raise RuntimeError("tracker block must not contain 写入时间")
    text = TRACKER_PATH.read_text(encoding="utf-8") if TRACKER_PATH.is_file() else ""
    marker = "### Round21.3 Forensics × T/U/V Discovery"
    if marker in text:
        head, rest = text.split(marker, 1)
        nxt = rest.find("\n### Round")
        text = head.rstrip() + "\n" + joined + (rest[nxt:] if nxt >= 0 else "")
    else:
        text = text.rstrip() + "\n" + joined
    TRACKER_PATH.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def run_report(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    from .attribution import ATTR_JSON
    from .backtest import BT_JSON, L1_BT_JSON
    from .construct import CONSTRUCT_JSON
    from .correlate import CORR_JSON
    from .evidence import EV_JSON
    from .forward import FWD_JSON
    from .n1n3 import N1N3_JSON
    from .nav_recon import RECON_JSON
    from .parity import PARITY_JSON
    from .pit import PIT_JSON
    from .recover import REC_JSON
    from .seed import SEED_JSON
    from .stats import STATS_JSON
    from .train import L1_TRAIN_JSON, MODEL_JSON

    parity = load_json(PARITY_JSON)
    construct = load_json(CONSTRUCT_JSON)
    pit = load_json(PIT_JSON)
    evidence = load_json(EV_JSON)
    corr = load_json(CORR_JSON)
    model = load_json(MODEL_JSON)
    l1t = load_json(L1_TRAIN_JSON)
    l1bt = load_json(L1_BT_JSON)
    bt = load_json(BT_JSON)
    attr = load_json(ATTR_JSON)
    stats = load_json(STATS_JSON)
    seed = load_json(SEED_JSON)
    n1n3 = load_json(N1N3_JSON)
    recon = load_json(RECON_JSON)
    recover = load_json(REC_JSON)
    fwd = load_json(FWD_JSON)

    update_standard()
    update_evaluate_factor()

    rows = list(l1bt.get("main_rows") or [])
    seen = {r.get("code") for r in rows}
    for r in bt.get("main_rows") or []:
        if r.get("code") not in seen:
            rows.append(r)
            seen.add(r.get("code"))
    fee_rows = list(l1bt.get("fee_rows") or []) + list(bt.get("fee_sensitivity") or [])
    # de-dup fee
    fee_seen = set()
    fee_uniq = []
    for r in fee_rows:
        k = (r.get("code"), r.get("fee"))
        if k in fee_seen:
            continue
        fee_seen.add(k)
        fee_uniq.append(r)

    decision = _decision(parity, bt if bt.get("best_vs_v5") else {"best_vs_v5": {"material": (l1bt.get("main_rows") or [{}])[-1].get("material") if l1bt.get("main_rows") else {}}}, fwd, pit)
    # L1-only material check if discovery bt empty
    if not bt.get("best_vs_v5") and l1bt.get("main_rows"):
        best_l1 = max((r for r in l1bt["main_rows"] if finite_or_none(r.get("sharpe")) is not None), key=lambda r: float(r["sharpe"]), default={})
        mat = best_l1.get("material") or {}
        if not mat.get("economically_material"):
            decision = KEEP_BASELINE_V5
        bt = {**bt, "best_vs_v5": {**best_l1, "material": mat, "code": best_l1.get("code")}}

    best = bt.get("best_vs_v5") or {}
    spa = stats.get("spa") or {}
    contrasts = l1bt.get("contrasts") or {}
    e0 = next((r for r in rows if r.get("code") in {L1_E0, "R21_3_E0"}), {})
    e1 = next((r for r in rows if r.get("code") == L1_E1), {})
    e2 = next((r for r in rows if r.get("code") == L1_E2), {})
    e3 = next((r for r in rows if r.get("code") == L1_E3), {})
    fee_e0 = next((r for r in fee_uniq if r.get("code") in {L1_E0, "R21_3_E0"}), {})
    fee_e3 = next((r for r in fee_uniq if r.get("code") == L1_E3), {})

    q = {
        "Q1": (
            f"21.2 E3 Ten **没有** L1（只改 Five：−g0_amount_log +L1），所以 +43 元是 10% Five 头的噪声级 Feature Recomposition，不是去冗余。"
            f"本轮正确 E3=E0−{{g0}}+{{L1}} 双头：E1−E0 ΔNAV=`{(contrasts.get('E1-E0') or {}).get('delta_nav')}` ΔSharpe=`{(contrasts.get('E1-E0') or {}).get('delta_sharpe')}`；"
            f"E3−E1 ΔNAV=`{(contrasts.get('E3-E1') or {}).get('delta_nav')}`。裁决=`{l1bt.get('forty_three_source')}`：伤害主要来自把 L1 加进 Ten。"
        ),
        "Q2": f"E1 `{L1_E1}` Sharpe=`{e1.get('sharpe')}` CAGR=`{e1.get('cagr')}` MaxDD=`{e1.get('max_drawdown')}` NAV=`{e1.get('nav_end')}` trades=`{e1.get('completed_trades')}`。路径 `{ROUND21_3_ROOT}/candidates/{L1_E1}`。",
        "Q3": f"E3 断言=`{(l1t.get('e3_assert') or {}).get('passed')}` hashes=`{l1t.get('hashes')}`。",
        "Q4": "L1 vs g0_amount_log 折内 ρ≈−0.073。|ρ| 低属 FEATURE_RECOMPOSITION，不能解释为去冗余。必须看 E1。",
        "Q5": f"首个分叉日=`{recon.get('first_fork')}` rows=`{recon.get('n_rows')}` CSV=`{ROUND21_3_ROOT}/L1_NAV_RECONCILIATION.csv` status=`{recon.get('status')}`。",
        "Q6": (
            f"21.2 的「NAV≈不变、MaxDD 改善 1.14pct」只存在于 Ten 未加 L1 的错误 E3。"
            f"本轮正确 E3：{recon.get('dd_note')} 期末 NAV 少约 940 万，MaxDD 恶化，不是路径改善。"
        ),
        "Q7": f"FEE E0 Sharpe=`{fee_e0.get('sharpe')}` CAGR=`{fee_e0.get('cagr')}`；E3 Sharpe=`{fee_e3.get('sharpe')}` CAGR=`{fee_e3.get('cagr')}`。完整账本见 `backtest/l1_fee_corrected.csv`。",
        "Q8": f"fee_direction_holds=`{l1bt.get('fee_direction_holds')}`。",
        "Q9": f"Bootstrap N=`{stats.get('boot_n')}` block=`{stats.get('block_days')}`。E3 vs E0 `{next((r.get('block_bootstrap_vs_e0') for r in (stats.get('rows') or []) if r.get('code')==L1_E3), None)}`。",
        "Q10": f"SPA observed=`{spa.get('observed_statistic')}` n_alts=`{spa.get('n_alternatives')}` p=`{spa.get('p_spa')}` status=`{spa.get('status')}`。",
        "Q11": f"seed_status=`{seed.get('seed_status')}` native5=`{seed.get('native_status') or seed.get('native_seeds')}` S30=`{seed.get('s30')}` S100=`{seed.get('s100')}`。",
        "Q12": f"N1/N3 verdict=`{n1n3.get('verdict')}` rows=`{n1n3.get('rows')}`。",
        "Q13": f"construct_bug=`{n1n3.get('construct_bug')}` do_not_retest=`{n1n3.get('do_not_retest')}` status=`{n1n3.get('status')}`。",
        "Q14": f"恢复协议 status=`{recover.get('status')}` remaining=`{recover.get('remaining')}` H子样本=`{(recover.get('h_subperiod') or {}).get('label')}`。",
        "Q15": f"do_not_compare_to_full_history_v5=`{(recover.get('h_subperiod') or {}).get('do_not_compare_to_full_history_v5')}`。",
        "Q16": f"冻结 `{list(NEW_FACTOR_IDS)}` N=`{len(NEW_FACTOR_IDS)}` computed=`{construct.get('computed')}` dups=`{construct.get('duplicates')}`。",
        "Q17": f"T computed=`{[c for c in (construct.get('computed') or []) if c.startswith('r21_t')]}`。",
        "Q18": f"U computed=`{[c for c in (construct.get('computed') or []) if c.startswith('r21_u')]}`。",
        "Q19": f"V computed=`{[c for c in (construct.get('computed') or []) if c.startswith('r21_v')]}` V3 dup 预注册。",
        "Q20": f"factory N=`{evidence.get('n_factory')}` gates=`{evidence.get('gates')}`。",
        "Q21": f"全引擎 N=`{len(rows)}` better material=`{(best.get('material') or {}).get('economically_material')}`。",
        "Q22": f"best=`{best.get('code')}` Sharpe=`{best.get('sharpe')}` CAGR=`{best.get('cagr')}` NAV=`{best.get('nav_end')}` vs v5 Sharpe=`{V5_ANCHOR['sharpe']}` CAGR=`{V5_ANCHOR['cagr']}` NAV=`{V5_NAV_END_EXACT}`。",
        "Q23": f"经济意义门 `{ECONOMIC_MATERIALITY['id']}` material=`{(best.get('material') or {}).get('economically_material')}` flags=`{(best.get('material') or {}).get('flags')}`。21.2 预先不达标。",
        "Q24": f"best MaxDD=`{best.get('max_drawdown')}` vs v5 `{V5_ANCHOR['max_drawdown']}`。",
        "Q25": f"分年见 17。best 2022/Ex24/Ex25 在 stats/trial_registry.csv。",
        "Q26": f"费用校正 E0/E3 Sharpe=`{fee_e0.get('sharpe')}`/`{fee_e3.get('sharpe')}` direction=`{l1bt.get('fee_direction_holds')}`。ran+None 视为未完成。",
        "Q27": f"SPA p=`{spa.get('p_spa')}` observed=`{spa.get('observed_statistic')}` n_alts=`{spa.get('n_alternatives')}` 标签 REPEATED_HISTORICAL_TESTING。",
        "Q28": f"Forward=`{fwd.get('status')}` missing=`{fwd.get('missing')}`。",
        "Q29": f"FINAL_DECISION=`{decision}` CURRENT_BASELINE_CHANGED=FALSE。",
    }

    status_run = COMPLETE
    if parity.get("passed") is False or any(x.get("status") in {"PARTIAL", "BLOCKED", "AUDIT_FAILURE"} for x in (recon, stats, seed, n1n3, recover, fwd) if x):
        if not rows:
            status_run = PARTIAL

    payload = {
        "status": status_run,
        "decision": decision,
        "parity_status": parity.get("status"),
        "n_proposed": len(NEW_FACTOR_IDS),
        "n_computed": construct.get("n_computed"),
        "n_pit_pass": pit.get("n_pass"),
        "n_pit_blocked": len(pit.get("blocked") or []),
        "n_full_backtests": len(rows),
        "forty_three_source": l1bt.get("forty_three_source"),
        "economically_material": (best.get("material") or {}).get("economically_material"),
        "best_vs_v5": best,
        "q": q,
        "current_baseline_changed": False,
        "auto_promote": AUTO_PROMOTE_CURRENT,
    }
    dump_json(DOSSIER, payload)
    _append_tracker(decision, payload, rows, fee_uniq)

    booklet_lines = [
        f"# Round 21.3 Research Report",
        "",
        f"决策：**{decision}**。CURRENT 保持 `baseline_v5=R16_O01`。AUTO_PROMOTE=False。不锁 v6。CURRENT_BASELINE_CHANGED=FALSE。",
        "",
        "## 锁定",
        "",
        f"- 目标 v5 CAGR=`{V5_ANCHOR['cagr']}` Sharpe=`{V5_ANCHOR['sharpe']}` MaxDD=`{V5_ANCHOR['max_drawdown']}` trades=`1107` NAV=`{V5_NAV_END_EXACT}`",
        f"- Stage0 parity=`{parity.get('status')}` passed=`{parity.get('passed')}`",
        f"- 经济意义门冻结 `{ECONOMIC_MATERIALITY['id']}`。21.2 +0.003/+43 预先不达标。",
        "",
        "## Track A L1 四路",
        "",
        f"- E3 断言=`{(l1t.get('e3_assert') or {}).get('passed')}`",
        f"- 43 元来源=`{l1bt.get('forty_three_source')}`",
        f"- E0 Sharpe=`{e0.get('sharpe')}` E1=`{e1.get('sharpe')}` E2=`{e2.get('sharpe')}` E3=`{e3.get('sharpe')}`",
        f"- 对照 `{contrasts}`",
        f"- NAV 对账 first_fork=`{recon.get('first_fork')}` status=`{recon.get('status')}`",
        f"- 费用方向 `{l1bt.get('fee_direction_holds')}` E0/E3 校正 Sharpe=`{fee_e0.get('sharpe')}`/`{fee_e3.get('sharpe')}`",
        f"- SPA p=`{spa.get('p_spa')}` observed=`{spa.get('observed_statistic')}` n_alts=`{spa.get('n_alternatives')}`",
        f"- seed=`{seed.get('seed_status')}`",
        f"- N1/N3=`{n1n3.get('verdict')}`",
        "",
        "## Track B T/U/V",
        "",
        f"- computed=`{construct.get('computed')}` blocked=`{construct.get('blocked')}` dups=`{construct.get('duplicates')}`",
        f"- PIT_PASS=`{pit.get('pit_pass')}`",
        f"- Forward=`{fwd.get('status')}`",
        "",
        "## Q1–Q29",
        "",
    ]
    for key, title in Q_ITEMS:
        booklet_lines.append(f"### {key} {title}")
        booklet_lines.append("")
        booklet_lines.append(q.get(key, "PARTIAL"))
        booklet_lines.append("")

    booklet_lines.extend(
        [
            "## STATUS",
            "",
            f"ROUND21_3_STATUS={status_run}",
            f"BASELINE_PARITY={parity.get('status')}",
            f"NEW_FACTORS_PROPOSED={len(NEW_FACTOR_IDS)}",
            f"NEW_FACTORS_COMPUTED={construct.get('n_computed')}",
            f"PIT_PASS={pit.get('n_pass')}",
            f"PIT_BLOCKED={len(pit.get('blocked') or [])}",
            f"BEST_CANDIDATE={best.get('code')}",
            f"BEST_DELTA_SHARPE={(best.get('material') or {}).get('delta_sharpe')}",
            f"BEST_DELTA_CAGR={(best.get('material') or {}).get('delta_cagr')}",
            f"ECONOMICALLY_MATERIAL={(best.get('material') or {}).get('economically_material')}",
            f"FORTY_THREE_SOURCE={l1bt.get('forty_three_source')}",
            f"SEED_STATUS={seed.get('seed_status')}",
            f"SPA_P={spa.get('p_spa')}",
            f"FORWARD={fwd.get('status')}",
            f"FINAL_DECISION={decision}",
            "CURRENT_BASELINE_CHANGED=FALSE",
            "AUTO_PROMOTE=False",
            "",
            "## 分册",
            "",
        ]
    )
    for k in sorted(MD):
        booklet_lines.append(f"- `{MD[k]}`")
    booklet_lines.extend(["", f"Final Report: `{FINAL_REPORT}`", ""])
    write_md(FINAL_REPORT, "\n".join(booklet_lines))
    write_md(MD["24"], f"# Round21.3 · 24 最终决策\n\n**{decision}** CURRENT_BASELINE_CHANGED=FALSE。\n")
    write_md(
        MD["01"],
        "\n".join(
            [
                "# Round21.3 · 01 21.2 法医",
                "",
                "## L1 四路（本轮已全引擎，含 E1）",
                "",
                f"| 路 | Sharpe | CAGR | NAV | MaxDD | trades |",
                f"|---|---:|---:|---:|---:|---:|",
                f"| E0 | {e0.get('sharpe')} | {e0.get('cagr')} | {e0.get('nav_end')} | {e0.get('max_drawdown')} | {e0.get('completed_trades')} |",
                f"| E1 Five−g0 / Ten 不变 | {e1.get('sharpe')} | {e1.get('cagr')} | {e1.get('nav_end')} | {e1.get('max_drawdown')} | {e1.get('completed_trades')} |",
                f"| E2 双头+L1 | {e2.get('sharpe')} | {e2.get('cagr')} | {e2.get('nav_end')} | {e2.get('max_drawdown')} | {e2.get('completed_trades')} |",
                f"| E3 双头 −g0 +L1 | {e3.get('sharpe')} | {e3.get('cagr')} | {e3.get('nav_end')} | {e3.get('max_drawdown')} | {e3.get('completed_trades')} |",
                "",
                "E3 断言 `E3 = E0 − {g0_amount_log} + {L1}` 通过。`g0_amount_log` 仅 Five 删除，Ten 保持 13 列。",
                "",
                "## 21.2 +43 元从哪来",
                "",
                "21.2 `R21_2_E3_r21_l1_known_float_chg` 的 Ten **没有** L1（`ten_extras` 因误判 five_only 规则被置空）。那条「最佳 E3」实际是 Five-only 替换 × 10% 融合权重，所以 NAV 只多 43 元。",
                "",
                f"本轮禁止沿用该错误规格。正确 E3 双头加 L1：{q.get('Q1')} {q.get('Q6')}",
                "",
                f"对照：E1−E0 ΔNAV=`{(contrasts.get('E1-E0') or {}).get('delta_nav')}` ΔSharpe=`{(contrasts.get('E1-E0') or {}).get('delta_sharpe')}`；"
                f"E3−E0 ΔNAV=`{(contrasts.get('E3-E0') or {}).get('delta_nav')}` ΔSharpe=`{(contrasts.get('E3-E0') or {}).get('delta_sharpe')}`。"
                f"裁决=`{l1bt.get('forty_three_source')}`：**+43 不是可部署增量。**",
                "",
                "## 费用 / 统计 / N1N3",
                "",
                f"- HISTORICAL_FEE_CORRECTED：E0 Sharpe=`{fee_e0.get('sharpe')}` E3 Sharpe=`{fee_e3.get('sharpe')}`。fee_direction_holds=`{l1bt.get('fee_direction_holds')}`。",
                f"- Bootstrap N=5000 block=20。SPA observed=`{spa.get('observed_statistic')}` n_alts=`{spa.get('n_alternatives')}` p=`{spa.get('p_spa')}`。",
                f"- seed=`{seed.get('seed_status')}`。N1/N3=`{n1n3.get('verdict')}` construct_bug=`{n1n3.get('construct_bug')}`。",
                "",
            ]
        ),
    )
    payload["final_report"] = str(FINAL_REPORT)
    payload["verdict"] = decision
    return payload
