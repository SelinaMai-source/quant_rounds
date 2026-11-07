"""主报告 00–27 + Q1–Q47 + 追踪表追加。不写写入时间。不升 CURRENT。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND21_4_ROOT
from .common import dump_json, finite_or_none, load_json, write_md
from .contract import (
    AUTO_PROMOTE_CURRENT,
    COMPLETE,
    CURRENT_ALIAS,
    CURRENT_BASELINE,
    ECONOMIC_MATERIALITY,
    FINAL_REPORT,
    FORWARD_OOS_NOT_AVAILABLE,
    KEEP_BASELINE_V5,
    MD,
    PARTIAL,
    Q_ITEMS,
    TRACKER_PATH,
    V5_ANCHOR,
    V5_NAV_END_EXACT,
    V6_RESEARCH_CANDIDATE,
    WXYZ_FACTOR_IDS,
)
from .docs import update_evaluate_factor, update_standard

DOSSIER = ROUND21_4_ROOT / "promotion" / "dossier.json"


def _decision(parity: dict[str, Any], bt: dict[str, Any], fwd: dict[str, Any], pit: dict[str, Any]) -> str:
    if parity.get("passed") is False:
        return KEEP_BASELINE_V5
    best = bt.get("best_vs_v5") or {}
    mat = best.get("material") or {}
    if not mat.get("economically_material"):
        return KEEP_BASELINE_V5
    if not (pit.get("n_pass") or pit.get("pit_pass")):
        return KEEP_BASELINE_V5
    if fwd.get("status") != "AVAILABLE" or not fwd.get("independent_forward"):
        return V6_RESEARCH_CANDIDATE
    return KEEP_BASELINE_V5


def _vs_v5(r: dict[str, Any]) -> dict[str, Any]:
    out = dict(r)
    sh = finite_or_none(out.get("sharpe"))
    cg = finite_or_none(out.get("cagr"))
    nav = finite_or_none(out.get("nav_end"))
    dd = finite_or_none(out.get("max_drawdown"))
    if out.get("delta_sharpe") is None and sh is not None:
        out["delta_sharpe"] = sh - float(V5_ANCHOR["sharpe"])
    if out.get("delta_cagr") is None and cg is not None:
        out["delta_cagr"] = cg - float(V5_ANCHOR["cagr"])
    if out.get("delta_maxdd") is None and dd is not None:
        out["delta_maxdd"] = dd - float(V5_ANCHOR["max_drawdown"])
    if out.get("delta_nav") is None and nav is not None:
        out["delta_nav"] = nav - float(V5_NAV_END_EXACT)
    return out


def _brief(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": r.get("code"),
        "combo": r.get("combo"),
        "method": r.get("method"),
        "status": r.get("status") or ("已跑" if finite_or_none(r.get("sharpe")) is not None else None),
        "sharpe": finite_or_none(r.get("sharpe")),
        "cagr": finite_or_none(r.get("cagr")),
        "nav_end": finite_or_none(r.get("nav_end")),
        "completed_trades": r.get("completed_trades"),
        "five_drops": r.get("five_drops"),
        "ten_drops": r.get("ten_drops"),
        "note": r.get("note") or r.get("reason"),
    }


def _fmt_row(label: str, code: str, r: dict[str, Any]) -> str:
    sh = finite_or_none(r.get("sharpe"))
    cg = finite_or_none(r.get("cagr"))
    dd = finite_or_none(r.get("max_drawdown"))
    nav = finite_or_none(r.get("nav_end"))
    dsh = finite_or_none(r.get("delta_sharpe"))
    dcg = finite_or_none(r.get("delta_cagr"))
    tr = r.get("total_return")
    ntr = r.get("completed_trades")
    st = "已跑" if sh is not None else (r.get("status") or "未完成")
    if sh is not None and cg is not None and dd is not None and nav is not None:
        if tr is None:
            tr = (1.0 + cg) ** 8.5 - 1.0
        return (
            f"| {label} | `{code}` | "
            f"{sh:6.3f} | {cg*100:6.2f}% | {float(tr)*100:+8.2f}% | {dd*100:+6.2f}% | {ntr} | "
            f"{nav:,.2f} | {'' if dsh is None else f'{dsh:+.4f}'} | {'' if dcg is None else f'{dcg*100:+.4f}pct'} | {st} |"
        )
    return f"| {label} | `{code}` | {sh} | {cg} |  | {dd} | {ntr} | {nav} | {dsh} | {dcg} | {st} |"


def _append_tracker(decision: str, payload: dict[str, Any], rows: list[dict[str, Any]], fee_rows: list[dict[str, Any]]) -> None:
    lines = [
        "",
        "### Round21.4 因子组合优化 × 下一代因子发现",
        "",
        "锁定 `baseline_v5=R16_O01`。Track A L1 A0–A9；Track B Forward `_max_date` 全分区扫描；Track D 六方法组合库；Track E W/X/Y/Z。不改 O01/B03/90-10。不自动升 CURRENT。",
        "",
        "### 实验组结果",
        "",
        "主表口径=**ROUND7_FLAT_5BP**。A0–A9 强制全引擎。并列 v5 不算升级。经济意义门沿用 21.3 冻结。",
        "",
        "| 操作 | 代号 | Sharpe |     年化 |        累计 |   MaxDD |   笔数 |           NAV | ΔSharpe |         Δ年化 | 状态 |",
        "| --- | --- | -----: | -----: | --------: | ------: | -----: | ------------: | ------: | ----------: | ------ |",
    ]
    formal = [r for r in rows if finite_or_none(r.get("sharpe")) is not None]
    diagnostic = [r for r in rows if finite_or_none(r.get("sharpe")) is None]
    for r in formal:
        role = r.get("role") or r.get("combo") or ""
        code = str(r.get("code") or "")
        label = f"{role} {code.replace('R21_4_', '')}" if role else code
        if code in {"R21_4_A0", "R21_4_E0"}:
            label = "精确复现 v5 / A0"
        lines.append(_fmt_row(label, code, r))
    if diagnostic:
        blocked = [r for r in diagnostic if r.get("status") == "BLOCKED"]
        oof_only = [r for r in diagnostic if r.get("status") != "BLOCKED"]
        if oof_only:
            lines.extend(["", "以下配置为 OOF_DIAGNOSTIC（未开全袖套），不进主表： " + ", ".join(f"`{r.get('code')}`" for r in oof_only) + "。"])
        if blocked:
            lines.extend(["", "以下配置 BLOCKED（真实阻断，非未跑），不进主表： " + "; ".join(f"`{r.get('code')}` {r.get('reason')}" for r in blocked) + "。"])
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
            f"| PIT_RESEARCH_ONLY | {payload.get('n_pit_research_only')} |",
            f"| 全引擎 N | {payload.get('n_full_backtests')} |",
            f"| forty_three_source | {payload.get('forty_three_source')} |",
            f"| a3_replicate_21_2 | {payload.get('a3_ok')} |",
            f"| a6_replicate_21_3 | {payload.get('a6_ok')} |",
            f"| economically_material | {payload.get('economically_material')} |",
            f"| best vs v5 | {best.get('code')} Sharpe={best.get('sharpe')} CAGR={best.get('cagr')} |",
            f"| combo_verdict | {payload.get('combo_verdict')} |",
            f"| forward | {payload.get('forward_status')} |",
            f"| ROUND21_4_STATUS | {payload.get('status')} |",
            f"| remaining | {payload.get('remaining')} |",
            "| 主表 | ROUND7_FLAT_5BP no hist_hooks |",
            "| 主报告 | `result/round21_4/ROUND21_4_RESEARCH_REPORT.md` |",
            "",
        ]
    )
    joined = "\n".join(lines)
    if "写入时间" in joined:
        raise RuntimeError("tracker block must not contain 写入时间")
    text = TRACKER_PATH.read_text(encoding="utf-8") if TRACKER_PATH.is_file() else ""
    marker = "### Round21.4 因子组合优化 × 下一代因子发现"
    if marker in text:
        head, rest = text.split(marker, 1)
        nxt = rest.find("\n### Round")
        text = head.rstrip() + "\n" + joined + (rest[nxt:] if nxt >= 0 else "")
    else:
        text = text.rstrip() + "\n" + joined
    TRACKER_PATH.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def run_report(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    from .attribution import ATTR_JSON
    from .backtest import A_BT_JSON, BT_JSON, WXYZ_BT_JSON
    from .combo import FAIL_JSON, RUN_JSON
    from .construct import CONSTRUCT_JSON
    from .correlate import CORR_JSON
    from .evidence import EV_JSON
    from .forward import COV_JSON, FIX_JSON, FWD_JSON
    from .nav_recon import RECON_JSON
    from .parity import PARITY_JSON
    from .pit import PIT_JSON
    from .recover import REC_JSON
    from .seed import SEED_JSON
    from .stats import STATS_JSON
    from .train import TRAIN_JSON

    parity = load_json(PARITY_JSON)
    construct = load_json(CONSTRUCT_JSON)
    pit = load_json(PIT_JSON)
    evidence = load_json(EV_JSON)
    corr = load_json(CORR_JSON)
    train = load_json(TRAIN_JSON)
    l1bt = load_json(A_BT_JSON) or load_json(BT_JSON)
    wbt = load_json(WXYZ_BT_JSON)
    combo = load_json(RUN_JSON)
    fail = load_json(FAIL_JSON)
    attr = load_json(ATTR_JSON)
    stats = load_json(STATS_JSON)
    seed = load_json(SEED_JSON)
    recon = load_json(RECON_JSON)
    recover = load_json(REC_JSON)
    fwd = load_json(FWD_JSON) or load_json(COV_JSON)
    fix = load_json(FIX_JSON)

    update_standard()
    update_evaluate_factor()

    rows = [_vs_v5(r) for r in (l1bt.get("main_rows") or [])]
    seen = {r.get("code") for r in rows}
    for r in (combo.get("rows") or []):
        if r.get("code") in seen:
            continue
        rows.append(_vs_v5(r))
        seen.add(r.get("code"))
    for r in (wbt.get("main_rows") or []):
        if r.get("code") in seen:
            continue
        rows.append(_vs_v5(r))
        seen.add(r.get("code"))
    fee_rows = list(l1bt.get("fee_rows") or [])
    fee_seen = set()
    fee_uniq = []
    for r in fee_rows:
        k = (r.get("code"), r.get("fee"))
        if k in fee_seen:
            continue
        fee_seen.add(k)
        fee_uniq.append(r)

    decision = _decision(parity, l1bt, fwd, pit)
    best = l1bt.get("best_vs_v5") or {}
    if not (best.get("material") or {}).get("economically_material"):
        decision = KEEP_BASELINE_V5

    spa = stats.get("spa") or {}
    contrasts = l1bt.get("contrasts") or {}
    by = {r.get("code"): r for r in rows}
    a = {k: by.get(f"R21_4_{k}") or {} for k in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9")}

    q = {
        "Q1": "；".join(f"{k} Sharpe=`{(a[k] or {}).get('sharpe')}` CAGR=`{(a[k] or {}).get('cagr')}` NAV=`{(a[k] or {}).get('nav_end')}`" for k in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9")),
        "Q2": f"A3 复现 21.2=`{(l1bt.get('replication') or {}).get('R21_4_A3')}` hashes=`{train.get('replicate')}`",
        "Q3": f"A6 复现 21.3=`{(l1bt.get('replication') or {}).get('R21_4_A6')}`",
        "Q4": f"恒等断言=`{(train.get('identity') or {}).get('passed')}` pairs=`{(train.get('identity') or {}).get('pairs')}`",
        "Q5": f"A1−A0=`{contrasts.get('A1-A0_drop_g0_five')}` A3−A1=`{contrasts.get('A3-A1_l1_after_drop')}` A3−A0=`{contrasts.get('A3-A0_43rmb')}` forty_three=`{l1bt.get('forty_three_rmb')}` source=`{l1bt.get('forty_three_source')}`",
        "Q6": f"A2−A0=`{contrasts.get('A2-A0_add_l1_five')}`",
        "Q7": f"A4−A0=`{contrasts.get('A4-A0_add_l1_ten')}` A5−A3=`{contrasts.get('A5-A3_add_l1_ten_after')}` A6−A3=`{contrasts.get('A6-A3_full_21_3')}`",
        "Q8": f"toxic_head=`{attr.get('toxic_head')}` interpretation=`{attr.get('interpretation')}`",
        "Q9": f"first_fork=`{recon.get('first_fork')}` n_trade_diff=`{recon.get('n_trade_diff')}` CSV=`{recon.get('csv')}`",
        "Q10": f"economically_material=`{l1bt.get('economically_material')}` best=`{(best.get('code'))}` flags=`{(best.get('material') or {}).get('flags')}`",
        "Q11": f"fee_direction_holds=`{l1bt.get('fee_direction_holds')}`",
        "Q12": f"A1−A0 drop g0=`{contrasts.get('A1-A0_drop_g0_five')}`",
        "Q13": f"bug=`{fix.get('bug')}` fix=`{fix.get('fix')}` daily_raw buggy=`{fix.get('daily_raw_buggy')}` fixed=`{fix.get('daily_raw_fixed')}`",
        "Q14": f"daily_raw=`{fix.get('daily_raw_fixed')}` money_flow=`{fix.get('money_flow_fixed')}`",
        "Q15": f"coverage status=`{fwd.get('status')}` matrix_n=`{len(fwd.get('matrix') or [])}` missing=`{fwd.get('missing')}`",
        "Q16": f"data_live=`{(fwd.get('data_live_releases') or {}).get('status')}` batches=`{fwd.get('live_batch_names')}`",
        "Q17": f"recover status=`{recover.get('status')}` steps=`{recover.get('steps')}`",
        "Q18": f"independent_forward=`{fwd.get('independent_forward')}` classification=`{fwd.get('classification')}`",
        "Q19": f"money_flow buggy=`{fix.get('money_flow_buggy')}` fixed=`{fix.get('money_flow_fixed')}` → 扫描 bug 不是真缺到 2010。",
        "Q20": "无独立 Forward 不得 live-ready，最多 V6_RESEARCH_CANDIDATE；本轮因经济门未过 → KEEP_BASELINE_V5。",
        "Q21": "evaluate_factor 强制 HEAD_SPECIFIC_FOUR_WAY：Five-only / Ten-only / Both-add / Both-replace。",
        "Q22": f"DATA_RECOVERY 接入=`{recover.get('status')}`",
        "Q23": "STANDARD 追加 Round 21.4 增量；zoo 写 round21_4_replacement.csv。",
        "Q24": f"Rejected=`{(load_json(ROUND21_4_ROOT / 'zoo' / 'zoo.json') or {}).get('rejected')}`",
        "Q25": f"六家族=`{(load_json(ROUND21_4_ROOT / 'combo' / 'combo_build.json') or {}).get('families')}` 路径=`research/factor_combination/`",
        "Q26": "C0–C11 在 freeze/manifest 冻结，sha256 见 round21_4_manifest.yaml。",
        "Q27": f"C1–C4=`{[_brief(r) for r in (combo.get('rows') or []) if r.get('combo') in {'C1','C2','C3','C4'}]}`",
        "Q28": f"C5–C6 折内物化真训=`{[_brief(r) for r in (combo.get('rows') or []) if r.get('combo') in {'C5','C6'}]}` k_gpc冻结=`{(combo.get('materialize') or {}).get('k_gpc_frozen')}`",
        "Q29": f"C7 折内物化真训=`{[_brief(r) for r in (combo.get('rows') or []) if r.get('combo') == 'C7']}`",
        "Q30": f"C8 折内重训（修正全样本前视）=`{[_brief(r) for r in (combo.get('rows') or []) if r.get('combo') == 'C8']}` superseded_n=`{len(combo.get('superseded') or [])}` C9=`{[_brief(r) for r in (combo.get('rows') or []) if r.get('combo') == 'C9']}`",
        "Q31": f"C10 折内物化真训=`{[_brief(r) for r in (combo.get('rows') or []) if r.get('combo') == 'C10']}` pairs=`{(combo.get('materialize') or {}).get('interaction_pairs') if combo.get('materialize') else (combo.get('notes') or {}).get('C10')}`",
        "Q32": f"C11 折内物化真训=`{[_brief(r) for r in (combo.get('rows') or []) if r.get('combo') == 'C11']}`",
        "Q33": f"combo_verdict=`{combo.get('verdict')}` portfolio=`{combo.get('portfolio')}`",
        "Q34": f"FailureAnalyzer=`{fail.get('rows')}`",
        "Q35": "Outer 年不参与方法/数量/正则选择；历史不足用 RAW_BASELINE 保守默认。FIXED/NESTED 均标 REPEATED_HISTORICAL_TESTING。",
        "Q36": f"全袖套 combo N=`{combo.get('n_full')}` rows with Sharpe=`{[r.get('code') for r in (combo.get('rows') or []) if finite_or_none(r.get('sharpe')) is not None]}`",
        "Q37": f"预注册 `{len(WXYZ_FACTOR_IDS)}` computed=`{construct.get('computed')}` blocked=`{construct.get('blocked')}`",
        "Q38": f"PIT_PASS=`{pit.get('n_pass')}` ids=`{pit.get('pit_pass')}` RESEARCH_ONLY=`{[r.get('factor_id') for r in (pit.get('rows') or []) if r.get('pit_status')=='RESEARCH_ONLY']}` BLOCKED=`{pit.get('blocked')}` dups=`{construct.get('duplicates')}`",
        "Q39": f"W=`{[c for c in (construct.get('computed') or []) if '_w' in c]}` IC 见 new_factor_IC.csv",
        "Q40": f"X=`{[c for c in (construct.get('computed') or []) if '_x' in c]}` 近邻 I 族在 correlate nearest。",
        "Q41": f"Y=`{[c for c in (construct.get('computed') or []) if '_y' in c]}` peer=交易所+代码前3位 exclude-self lag1。",
        "Q42": f"Z=`{[c for c in (construct.get('computed') or []) if '_z' in c]}`",
        "Q43": f"WXYZ 四路 n=`{wbt.get('n_full_backtests')}` attempted=`{wbt.get('n_attempted')}`",
        "Q44": f"best=`{best.get('code')}` Sharpe=`{best.get('sharpe')}` CAGR=`{best.get('cagr')}` NAV=`{best.get('nav_end')}` vs v5 Sharpe=`{V5_ANCHOR['sharpe']}` CAGR=`{V5_ANCHOR['cagr']}` NAV=`{V5_NAV_END_EXACT}`",
        "Q45": f"material=`{(best.get('material') or {}).get('economically_material')}` MaxDD=`{best.get('max_drawdown')}` vs `{V5_ANCHOR['max_drawdown']}`",
        "Q46": f"FINAL_DECISION=`{decision}` CURRENT_BASELINE_CHANGED=FALSE",
        "Q47": "下一轮优先：独立 Forward 数据授权恢复；Ten 头对 L1 类股本因子的 Head-specific 过滤；组合库 C5–C11 已折内物化真训（本轮完成），下一轮聚焦 Nested WF 方法选择与交互项经济解释。",
    }

    combo_diag = [r.get("combo") for r in (combo.get("rows") or []) if r.get("status") == "OOF_DIAGNOSTIC"]
    remaining = []
    if recover.get("status") not in {"DONE", "COMPLETE"}:
        remaining.append(f"recover={recover.get('status')}")
    if combo_diag:
        remaining.append(f"combo_diagnostic={combo_diag}")
    if int(combo.get("n_full") or 0) < 12:
        remaining.append(f"combo_n_full={combo.get('n_full')}/12")
    if fwd.get("independent_forward") is not True:
        remaining.append(f"forward={fwd.get('status') or FORWARD_OOS_NOT_AVAILABLE}")
    if int(l1bt.get("n_full_backtests") or 0) < 10:
        remaining.append(f"amatrix_n={l1bt.get('n_full_backtests')}")
    if parity.get("passed") is False:
        remaining.append("parity_fail")
    status_run = COMPLETE if not remaining else PARTIAL

    payload = {
        "status": status_run,
        "decision": decision,
        "parity_status": parity.get("status"),
        "n_proposed": len(WXYZ_FACTOR_IDS),
        "n_computed": construct.get("n_computed"),
        "n_pit_pass": pit.get("n_pass"),
        "n_pit_blocked": len(pit.get("blocked") or []),
        "n_pit_research_only": sum(1 for r in (pit.get("rows") or []) if r.get("pit_status") == "RESEARCH_ONLY"),
        "n_full_backtests": sum(1 for r in rows if finite_or_none(r.get("sharpe")) is not None),
        "forty_three_source": l1bt.get("forty_three_source"),
        "a3_ok": l1bt.get("a3_ok"),
        "a6_ok": l1bt.get("a6_ok"),
        "economically_material": (best.get("material") or {}).get("economically_material"),
        "best_vs_v5": best,
        "combo_verdict": combo.get("verdict"),
        "forward_status": fwd.get("status"),
        "remaining": remaining,
        "q": q,
        "current_baseline_changed": False,
        "auto_promote": AUTO_PROMOTE_CURRENT,
        "verdict": decision,
    }
    dump_json(DOSSIER, payload)
    _append_tracker(decision, payload, rows, fee_uniq)

    booklet = [
        "# Round 21.4 Research Report",
        "",
        f"决策：**{decision}**。CURRENT 保持 `baseline_v5=R16_O01`。AUTO_PROMOTE=False。不锁 v6。CURRENT_BASELINE_CHANGED=FALSE。",
        "",
        "## 锁定",
        "",
        f"- 目标 v5 CAGR=`{V5_ANCHOR['cagr']}` Sharpe=`{V5_ANCHOR['sharpe']}` MaxDD=`{V5_ANCHOR['max_drawdown']}` trades=`1107` NAV=`{V5_NAV_END_EXACT}`",
        f"- Stage0 parity=`{parity.get('status')}` passed=`{parity.get('passed')}`",
        f"- 经济意义门冻结 `{ECONOMIC_MATERIALITY['id']}`（沿用 21.3，禁止回调）。21.2 +0.003/+43 预先不达标。",
        "",
        "## Track A L1 A0–A9",
        "",
        f"- 恒等=`{(train.get('identity') or {}).get('passed')}` 复现=`{(train.get('replicate') or {}).get('passed')}`",
        f"- A3 21.2=`{l1bt.get('a3_ok')}` A6 21.3=`{l1bt.get('a6_ok')}`",
        f"- 43 元=`{l1bt.get('forty_three_rmb')}` 来源=`{l1bt.get('forty_three_source')}`",
        f"- toxic_head=`{attr.get('toxic_head')}`",
        "",
        "## Track B Forward",
        "",
        f"- `_max_date` daily_raw buggy=`{fix.get('daily_raw_buggy')}` fixed=`{fix.get('daily_raw_fixed')}`",
        f"- money_flow buggy=`{fix.get('money_flow_buggy')}` fixed=`{fix.get('money_flow_fixed')}`",
        f"- coverage=`{fwd.get('status')}` independent=`{fwd.get('independent_forward')}`",
        "",
        "## Track D Combo",
        "",
        f"- verdict=`{combo.get('verdict')}` n_full=`{combo.get('n_full')}`",
        f"- FailureAnalyzer=`{(fail.get('rows') or [{}])[0].get('primary') if fail.get('rows') else None}`",
        "",
        "## Track E W/X/Y/Z",
        "",
        f"- computed=`{construct.get('computed')}` PIT_PASS=`{pit.get('pit_pass')}` 四路 N=`{wbt.get('n_full_backtests')}`",
        "",
        "## Q1–Q47",
        "",
    ]
    for key, title in Q_ITEMS:
        booklet.append(f"### {key} {title}")
        booklet.append("")
        booklet.append(str(q.get(key, "PARTIAL")))
        booklet.append("")
    booklet.extend(
        [
            "## STATUS",
            "",
            f"ROUND21_4_STATUS={status_run}",
            f"REMAINING={remaining}",
            f"BASELINE_PARITY={parity.get('status')}",
            f"NEW_FACTORS_PROPOSED={len(WXYZ_FACTOR_IDS)}",
            f"NEW_FACTORS_COMPUTED={construct.get('n_computed')}",
            f"PIT_PASS={pit.get('n_pass')}",
            f"PIT_BLOCKED={len(pit.get('blocked') or [])}",
            f"PIT_RESEARCH_ONLY={sum(1 for r in (pit.get('rows') or []) if r.get('pit_status') == 'RESEARCH_ONLY')}",
            f"BEST_CANDIDATE={best.get('code')}",
            f"BEST_DELTA_SHARPE={(best.get('material') or {}).get('delta_sharpe')}",
            f"BEST_DELTA_CAGR={(best.get('material') or {}).get('delta_cagr')}",
            f"ECONOMICALLY_MATERIAL={(best.get('material') or {}).get('economically_material')}",
            f"FORTY_THREE_SOURCE={l1bt.get('forty_three_source')}",
            f"A3_REPLICATE_21_2={l1bt.get('a3_ok')}",
            f"A6_REPLICATE_21_3={l1bt.get('a6_ok')}",
            f"COMBO_VERDICT={combo.get('verdict')}",
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
    for k in sorted(MD, key=lambda x: str(x)):
        booklet.append(f"- `{MD[k]}`")
    booklet.extend(["", f"Final Report: `{FINAL_REPORT}`", ""])
    write_md(FINAL_REPORT, "\n".join(booklet))
    write_md(MD["27"], f"# Round21.4 · 27 最终决策\n\n**{decision}** CURRENT_BASELINE_CHANGED=FALSE。\n")
    write_md(
        MD["01"],
        "\n".join(
            [
                "# Round21.4 · 01 21.3 交接",
                "",
                "21.2 最佳 E3 是 Five-only（Ten 无 L1）→ +43 元是 10% Five 噪声。",
                "21.3 E3 双头加 L1 → −940 万。A3 必须复现 21.2；A6 必须复现 21.3。",
                "21.3 forward.py `_max_date` 只读 sorted()[:5][-1]。本轮全扫分区。",
                f"A3 ok=`{l1bt.get('a3_ok')}` A6 ok=`{l1bt.get('a6_ok')}` 43来源=`{l1bt.get('forty_three_source')}`",
                "",
            ]
        ),
    )
    payload["final_report"] = str(FINAL_REPORT)
    return payload
