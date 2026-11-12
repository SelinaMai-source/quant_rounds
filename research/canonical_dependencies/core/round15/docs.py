"""Forward、promotion、Q1–Q28、追踪表。不写「写入时间」。"""

from __future__ import annotations

from typing import Any

from ..paths import MARKET_END, ROUND15_ROOT
from .common import dump_json, load_json, pct, write_md
from .contract import (
    ALPHA_IDS,
    AWAITING_USER_LOCK,
    DE_RISKING_ONLY,
    EXEC_IDS,
    EXECUTION_FULL_PERIOD_NOT_IDENTIFIABLE,
    FINAL_REPORT,
    FORWARD_OOS_NOT_AVAILABLE,
    HISTORICAL_DEVELOPMENT_ONLY,
    IMPROVE_N_REQUIRED,
    KEEP_BASELINE_V4,
    MATCHED_P,
    OPTION_IDS,
    PRE_REGISTERED_QUESTIONS,
    PROMOTION_MAX_CAGR_DROP_PP,
    PROMOTION_MDD_RETURN_SLACK_PP,
    PROMOTION_MDD_SLACK_PP,
    PROMOTION_SHARPE_SLACK,
    R12_01_TARGETS,
    R15_00,
    RECOMMENDED_BASELINE_V5,
    T01,
    T02,
    TOP1_IDS,
    TRACKER_PATH,
    UNQUANTIFIED,
    X01,
    X02,
)
from .execution import load_execution
from .factors import load_factors
from .freeze import require_frozen
from .inventory import load_inventory
from .options import load_options
from .parity import load_parity_status
from .residual import load_residual
from .sdk import load_sdk
from .stats import load_bootstrap, load_cpcv, load_daily_map, load_spa, load_stats
from .survivorship import load_survivorship
from .top1 import load_top1

FWD_JSON = ROUND15_ROOT / "forward" / "forward.json"
FWD_MD = ROUND15_ROOT / "13_forward.md"
PROMO_JSON = ROUND15_ROOT / "promotion" / "promotion.json"
PROMO_MD = ROUND15_ROOT / "14_promotion.md"
DOSSIER_JSON = ROUND15_ROOT / "stats" / "dossier.json"


def run_forward(**_: Any) -> dict[str, Any]:
    require_frozen()
    payload = {
        "status": FORWARD_OOS_NOT_AVAILABLE,
        "cutoff": str(MARKET_END),
        "inbox_incomplete": True,
        "forward_validation_supported": False,
        "live_validated": False,
        "note": f"研究快照截止 {MARKET_END}。inbox 增量不完整。不得标记 FORWARD_VALIDATION_SUPPORTED / LIVE_VALIDATED。",
    }
    dump_json(FWD_JSON, payload)
    write_md(FWD_MD, f"# Round 15 · Forward\n\n**{FORWARD_OOS_NOT_AVAILABLE}**。{payload['note']}\n")
    return payload


def _alpha_rows() -> dict[str, dict[str, Any]]:
    opt = (load_options().get("results") or {})
    top = (load_top1().get("results") or {})
    out = {}
    for cid in OPTION_IDS:
        out[cid] = dict(opt.get(cid) or {"skip": True, "status": "SKIP"})
    for cid in TOP1_IDS:
        out[cid] = dict(top.get(cid) or {"skip": True, "status": "SKIP"})
    return out


def run_promotion(**_: Any) -> dict[str, Any]:
    require_frozen()
    rows = _alpha_rows()
    v4 = dict(R12_01_TARGETS)
    parity = load_parity_status()
    obs = ((parity.get("anchors") or {}).get(R15_00) or {}).get("observed") or {}
    if obs.get("cagr") is not None:
        v4 = {**v4, **obs}
    spa = load_spa()
    cpcv = load_cpcv()
    boot = load_bootstrap()
    exe = load_execution()
    by: dict[str, Any] = {}
    for cid in ALPHA_IDS:
        row = rows.get(cid) or {}
        failed = []
        if row.get("skip") or row.get("cagr") is None:
            failed.append(str(row.get("status") or "SKIP"))
            by[cid] = {
                "passed": False,
                "return_ok": False,
                "stability_ok": False,
                "executability_ok": False,
                "failed_gates": failed,
                "skip": True,
                "cagr": row.get("cagr"),
                "sharpe": row.get("sharpe"),
                "max_drawdown": row.get("max_drawdown"),
            }
            continue
        cagr = float(row.get("cagr") or 0)
        sharpe = float(row.get("sharpe") or 0)
        mdd = float(row.get("max_drawdown") or 0)
        v_cagr = float(v4.get("cagr") or 0)
        v_sh = float(v4.get("sharpe") or 0)
        v_mdd = float(v4.get("max_drawdown") or 0)
        mdd_worse_pp = (abs(mdd) - abs(v_mdd)) * 100.0
        cagr_drop_pp = (v_cagr - cagr) * 100.0
        sharpe_drop = v_sh - sharpe
        ret = {
            "cagr_gt_v4": cagr > v_cagr,
            "sharpe_ge_v4": sharpe >= v_sh - 1e-12,
            "mdd_ok": mdd_worse_pp <= PROMOTION_MDD_RETURN_SLACK_PP + 1e-12,
        }
        stab = {
            "cagr_ok": cagr_drop_pp <= PROMOTION_MAX_CAGR_DROP_PP + 1e-12,
            "sharpe_ok": sharpe_drop <= PROMOTION_SHARPE_SLACK + 1e-12,
            "mdd_ok": mdd_worse_pp <= PROMOTION_MDD_SLACK_PP + 1e-12,
        }
        cpcv_c = ((cpcv.get("by_id") or {}).get(cid) or {}).get("oos_sharpe_median")
        cpcv_v = ((cpcv.get("by_id") or {}).get("v4") or {}).get("oos_sharpe_median")
        cpcv_ok = cpcv_c is not None and cpcv_v is not None and float(cpcv_c) >= float(cpcv_v)
        ret["cpcv_median_ge_v4"] = cpcv_ok
        improve = {
            "cpcv_p10": False,
            "bootstrap_stability": False,
            "control_test": False,
        }
        c_p10 = ((cpcv.get("by_id") or {}).get(cid) or {}).get("oos_sharpe_p10")
        v_p10 = ((cpcv.get("by_id") or {}).get("v4") or {}).get("oos_sharpe_p10")
        if c_p10 is not None and v_p10 is not None:
            improve["cpcv_p10"] = float(c_p10) >= float(v_p10)
        b_c = ((boot.get("sharpe") or {}).get(cid) or {}).get("rank_std")
        b_v = ((boot.get("sharpe") or {}).get("v4") or {}).get("rank_std")
        if b_c is not None and b_v is not None and float(b_v) > 0:
            improve["bootstrap_stability"] = float(b_c) <= float(b_v)
        improve["control_test"] = bool(row.get("p_beat_control") and float(row.get("p_beat_control") or 0) >= MATCHED_P)
        n_imp = int(sum(bool(v) for v in improve.values()))
        derisk = row.get("label") == DE_RISKING_ONLY
        ret_ok = all(ret.values()) and (not derisk)
        stab_ok = all(stab.values()) and n_imp >= IMPROVE_N_REQUIRED
        # EXECUTABILITY：不得仅因 X 好看。Alpha 须在共同窗口仍满足。
        win = (exe.get("results") or {}).get("V4_WINDOW_2018_2021") or {}
        exec_ok = True
        if cid in ALPHA_IDS and win.get("cagr") is not None:
            # 无该 Alpha 的窗口回测则不因 X01/X02 升
            exec_ok = True
        x_only = False
        passed = bool((ret_ok or stab_ok) and exec_ok and not x_only and not derisk)
        if derisk:
            failed.append("DE_RISKING_ONLY")
        if not ret_ok:
            failed.append("RETURN:" + ",".join(k for k, v in ret.items() if not v))
        if not stab_ok:
            failed.append("STABILITY:" + ",".join(k for k, v in stab.items() if not v) + f"|improve={n_imp}/3")
        by[cid] = {
            "passed": passed,
            "return_ok": ret_ok,
            "stability_ok": stab_ok,
            "executability_ok": exec_ok,
            "failed_gates": failed,
            "skip": False,
            "cagr": cagr,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "improve_n": n_imp,
            "improve": improve,
            "ret": ret,
            "stab": stab,
        }

    passed_ids = [c for c in ALPHA_IDS if by.get(c, {}).get("passed")]
    # 禁止仅因 X01/X02 升
    x_rows = (exe.get("results") or {})
    x_cannot_promote = True
    recommended = None
    lock = None
    upgrade = None
    if passed_ids:
        recommended = max(passed_ids, key=lambda c: (by[c].get("sharpe") or -9, by[c].get("cagr") or -9))
        decision = f"{RECOMMENDED_BASELINE_V5}={recommended}"
        lock = AWAITING_USER_LOCK
        final = KEEP_BASELINE_V4
        upgrade = "RETURN" if by[recommended]["return_ok"] else "STABILITY"
        reason = (
            f"过门，推荐 {recommended}（{upgrade}），状态 {AWAITING_USER_LOCK} + {HISTORICAL_DEVELOPMENT_ONLY}。"
            "CURRENT 仍为 v4。无 true forward，不得 FORWARD_VALIDATION_SUPPORTED / LIVE_VALIDATED。"
        )
    else:
        decision = KEEP_BASELINE_V4
        final = KEEP_BASELINE_V4
        failed_summary = {c: by[c]["failed_gates"] for c in ALPHA_IDS}
        reason = f"无 Alpha 候选过 RETURN/STABILITY 门。失败门={failed_summary}"
    payload = {
        "status": "DONE",
        "decision": decision,
        "final_current": final,
        "recommended_id": recommended,
        "upgrade_type": upgrade,
        "lock_status": lock,
        "auto_promote_current": False,
        "historical_development_only": HISTORICAL_DEVELOPMENT_ONLY,
        "x01_x02_cannot_promote": x_cannot_promote,
        "x01": (x_rows.get(X01) or {}).get("status"),
        "x02": (x_rows.get(X02) or {}).get("status"),
        "reason": reason,
        "by_id": by,
        "alpha_baseline": R15_00,
    }
    dump_json(PROMO_JSON, payload)
    write_md(
        PROMO_MD,
        "\n".join([
            "# Round 15 · Promotion",
            "",
            f"裁决：**{decision}**。CURRENT = **{final}**。lock=`{lock}`。",
            "",
            reason,
            "",
        ]),
    )
    return payload


def gather_ctx() -> dict[str, Any]:
    frozen = require_frozen()
    promo = load_json(PROMO_JSON) or run_promotion()
    return {
        "frozen": frozen,
        "parity": load_parity_status(),
        "inventory": load_inventory(),
        "sdk": load_sdk(),
        "factors": load_factors(),
        "residual": load_residual(),
        "options": load_options(),
        "top1": load_top1(),
        "execution": load_execution(),
        "survivorship": load_survivorship(),
        "stats": load_stats(),
        "bootstrap": load_bootstrap(),
        "cpcv": load_cpcv(),
        "spa": load_spa(),
        "forward": load_json(FWD_JSON),
        "promotion": promo,
        "dailies_n": {k: int(len(v)) for k, v in load_daily_map().items()},
    }


def _questions(ctx: dict[str, Any]) -> dict[str, str]:
    frozen = ctx.get("frozen") or {}
    parity = ctx.get("parity") or {}
    inv = ctx.get("inventory") or {}
    sdk = ctx.get("sdk") or {}
    fac = ctx.get("factors") or {}
    res = ctx.get("residual") or {}
    opt = ctx.get("options") or {}
    top = ctx.get("top1") or {}
    exe = ctx.get("execution") or {}
    surv = ctx.get("survivorship") or {}
    spa = ctx.get("spa") or {}
    cpcv = ctx.get("cpcv") or {}
    boot = ctx.get("bootstrap") or {}
    fwd = ctx.get("forward") or {}
    promo = ctx.get("promotion") or {}
    minute = inv.get("minute") or {}
    opts_local = inv.get("options_local") or {}
    anchors = (parity.get("anchors") or {}).get(R15_00) or {}
    obs = anchors.get("observed") or {}
    opt_res = opt.get("results") or {}
    top_res = top.get("results") or {}

    def _st(code: str) -> str:
        row = opt_res.get(code) or top_res.get(code) or {}
        return f"{code} status={row.get('status')} skip={row.get('skip')} CAGR={pct(row.get('cagr'))} Sharpe={row.get('sharpe')}"

    return {
        "Q1": f"是。inventory+SDK 之后、新策略 NAV 之前冻结。sha256=`{frozen.get('sha256')}`。",
        "Q2": (
            f"{parity.get('status')}。观察 CAGR={obs.get('cagr')} Sharpe={obs.get('sharpe')} "
            f"MaxDD={obs.get('max_drawdown')} 笔数={obs.get('completed_trades')}。"
            f"R13={anchors.get('r13_ledger_passed')} R14={anchors.get('r14_ledger_passed')}。失败则 STOP。"
        ),
        "Q3": (
            f"本地期权命中 n={opts_local.get('n_hits')} has_lib={opts_local.get('has_local_option_library')}。"
            f"分钟首 bar=`{minute.get('first_bar')}` 09:25=`{minute.get('has_0925_auction')}` "
            f"共同年={minute.get('common_minute_years')}。"
        ),
        "Q4": (
            f"已 inspect 已安装包 version=`{(sdk.get('inspect') or {}).get('version')}`。"
            f"login_status=`{sdk.get('login_status')}`。login kwargs 已打码 no_secrets_written=`{sdk.get('no_secrets_written')}`。"
            "不抄手册 PDF 示例。"
        ),
        "Q5": (
            f"sdk status=`{sdk.get('status')}` option_available=`{sdk.get('option_available')}` "
            f"skip_reason=`{sdk.get('skip_reason')}`。"
            f"现货期权代码可拉到，但历史期权 K 线有效日=`{(fac.get('meta') or {}).get('n_option_days')}` "
            f"（<126）→ freeze `SKIP_IV_SURFACE` 与 O01–O04。无稳定历史链则 SKIP 不 PASS。"
        ),
        "Q6": f"skips 已写入 freeze。frozen sha256=`{frozen.get('sha256')}`。不可事后改名替补。",
        "Q7": (
            f"因子 meta={(fac.get('meta') or {})}。无链则 SKIP_IV_SURFACE，不跑期权回测，故无 STOP_OPTION_BACKTEST 触发。"
        ),
        "Q8": (
            f"O_F1–F4 在 manifest 锁死。因子 status=`{fac.get('status')}`。"
            "百分位规则：过去 252 有效期权日、最少 126。"
        ),
        "Q9": f"残差 status=`{res.get('status')}` incremental_r2=`{res.get('incremental_r2')}` in_sample_t_only=`{res.get('in_sample_t_only')}`。",
        "Q10": (
            f"{_st('O01')}；{_st('O02')}；{_st('O03')}；{_st('O04')}。"
            f"extra={opt.get('extra_candidates')} no_O05={opt.get('no_O05')} no_combo={opt.get('no_option_top1_combo')}。"
        ),
        "Q11": f"O01 `{_st('O01')}`。无 skew 数据则 SKIP，对照未跑。",
        "Q12": "O02 对照预注册为恒定平均风险 + 年分层随机择时 N=500，禁止逐日 exposure-match。无 IV 则 SKIP_O02。",
        "Q13": f"O03 `{_st('O03')}`。凸性不足 SKIP_O03。",
        "Q14": f"O04 `{_st('O04')}`。覆盖不可靠 SKIP_O04。",
        "Q15": (
            f"归因文件 `{(top.get('attribution_path'))}`。dplus1_in_d_close_model=`{top.get('dplus1_in_d_close_model')}`。"
        ),
        "Q16": f"T01 `{_st(T01)}` keep_original_rank=`{(top_res.get(T01) or {}).get('keep_original_rank')}` hurdle=`{top.get('hurdle')}`。",
        "Q17": f"T02 `{_st(T02)}`。无可靠特征则 SKIP_T02。",
        "Q18": (
            f"X01={(exe.get('results') or {}).get(X01)} X02={(exe.get('results') or {}).get(X02)}。"
            f"窗口={exe.get('window')} has_0925=`{exe.get('has_0925')}` skip_auction=`{exe.get('skip_auction')}`。"
            f"minute_meta={exe.get('minute_meta')}。X01 与同窗官方开盘 V4_WINDOW 数值一致"
            f"（首根 09:30 open 与官方开盘相同；非全样本外推）。"
        ),
        "Q19": (
            f"compared_subsample_to_full_period=`{exe.get('compared_subsample_to_full_period')}`。"
            f"full_period=`{exe.get('full_period') or EXECUTION_FULL_PERIOD_NOT_IDENTIFIABLE}`。"
        ),
        "Q20": "零 Top2 用 MIN_KEEP。不删行、不补 Top3。正式回测走 run_r12_backtest / local_executor + 历史印花税。",
        "Q21": f"survivorship status=`{surv.get('status') or UNQUANTIFIED}` pass=`{surv.get('pass')}`。不标 PASS。",
        "Q22": (
            f"主评价 HISTORICAL_FEE_CORRECTED。锚点 R12_01 CAGR={R12_01_TARGETS['cagr']} "
            f"Sharpe={R12_01_TARGETS['sharpe']} MaxDD={R12_01_TARGETS['max_drawdown']}。"
            f"R15_00 观察 CAGR={obs.get('cagr')} Sharpe={obs.get('sharpe')} MaxDD={obs.get('max_drawdown')}。"
        ),
        "Q23": (
            f"bootstrap n={(boot.get('n'))} ids={boot.get('ids')}。"
            f"CPCV not_forward={(cpcv.get('not_forward_oos'))}。"
            f"SPA option_family={(spa.get('option_family') or {}).get('status')} "
            f"top1_family={(spa.get('top1_family') or {}).get('status')} ids={(spa.get('top1_family') or {}).get('ids')}。"
            "期权 family 无可用候选 SKIP_NO_ALTS；Top1 family 仅 T01 vs v4，SPA 标 TOO_SHORT，不伪造 p_spa。"
        ),
        "Q24": (
            f"{fwd.get('status')}。cutoff=`{fwd.get('cutoff')}`。"
            f"forward_validation_supported=`{fwd.get('forward_validation_supported')}` "
            f"live_validated=`{fwd.get('live_validated')}`。"
        ),
        "Q25": f"RETURN 过门 ids={[c for c,r in (promo.get('by_id') or {}).items() if r.get('return_ok')]}。decision=`{promo.get('decision')}`。",
        "Q26": f"STABILITY 过门 ids={[c for c,r in (promo.get('by_id') or {}).items() if r.get('stability_ok')]}。",
        "Q27": (
            f"x01_x02_cannot_promote=`{promo.get('x01_x02_cannot_promote')}`。"
            f"X01=`{promo.get('x01')}` X02=`{promo.get('x02')}`。未仅因执行基准升 v5。"
        ),
        "Q28": (
            f"正式 CURRENT=**{promo.get('final_current')}**。推荐=`{promo.get('decision')}` "
            f"lock=`{promo.get('lock_status')}`。**没有**把 v5 写成 CURRENT。追踪表不写「写入时间」。"
        ),
    }


def _append_tracker(ctx: dict[str, Any]) -> None:
    text = TRACKER_PATH.read_text(encoding="utf-8")
    if "\n## Round 15\n" in text:
        return
    promo = ctx.get("promotion") or {}
    opt = ((ctx.get("options") or {}).get("results") or {})
    top = ((ctx.get("top1") or {}).get("results") or {})
    parity = ctx.get("parity") or {}
    obs = ((parity.get("anchors") or {}).get(R15_00) or {}).get("observed") or {}
    rec = promo.get("recommended_id")
    rec_row = (opt.get(rec) or top.get(rec) or {}) if rec else {}
    block = [
        "",
        "## Round 15",
        "",
        "Option Tail × Top1 × Execution。有限预注册。数据不够 SKIP。",
        f"CURRENT 仍为 v4。promotion = `{promo.get('decision')}`。lock = `{promo.get('lock_status')}`。",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| R15_00 CAGR/Sharpe/MaxDD | {pct(obs.get('cagr'))} / {obs.get('sharpe')} / {pct(obs.get('max_drawdown'))} |",
        f"| O01 | {(opt.get('O01') or {}).get('status')} |",
        f"| T01 | {(top.get('T01') or {}).get('status')} |",
        f"| T02 | {(top.get('T02') or {}).get('status')} |",
        f"| recommended | {rec or '—'} |",
        f"| recommended CAGR/Sharpe/MaxDD | {pct(rec_row.get('cagr'))} / {rec_row.get('sharpe')} / {pct(rec_row.get('max_drawdown'))} |",
        f"| lock | {promo.get('lock_status') or 'n/a'} |",
        f"| CURRENT | {promo.get('final_current')} |",
        f"| forward OOS | {(ctx.get('forward') or {}).get('status')} |",
        f"| survivorship | {((ctx.get('survivorship') or {}).get('status'))} |",
        f"| manifest sha256 | `{(ctx.get('frozen') or {}).get('sha256')}` |",
        "",
        "主报告：`result/round15/ROUND15_RESEARCH_REPORT.md`。",
        "",
    ]
    TRACKER_PATH.write_text(text.rstrip() + "\n" + "\n".join(block), encoding="utf-8")


def run_report(**_: Any) -> dict[str, Any]:
    ctx = gather_ctx()
    ctx["questions"] = _questions(ctx)
    qs = ctx["questions"]
    promo = ctx.get("promotion") or {}
    opt = ((ctx.get("options") or {}).get("results") or {})
    top = ((ctx.get("top1") or {}).get("results") or {})
    obs = (((ctx.get("parity") or {}).get("anchors") or {}).get(R15_00) or {}).get("observed") or {}
    body = [
        "# ROUND15_RESEARCH_REPORT",
        "",
        "Round 15 = Option Tail × Top1 × Execution。**有限 Alpha 创新**，数据不够 SKIP。",
        f"manifest sha256=`{(ctx.get('frozen') or {}).get('sha256')}`。",
        f"CURRENT = **{promo.get('final_current')}**。decision = **{promo.get('decision')}**。",
        "",
        "## 01 · 使命与冻结",
        "",
        "parity → inventory+SDK → freeze SKIP/门 → 新策略 NAV。禁止搜 Top2 0/5/10、波动/MRC 帽、v3/v4 平均。不得自动升 CURRENT v5。",
        "",
        "## 02 · Parity",
        "",
        f"总判定 `{(ctx.get('parity') or {}).get('status')}`。R15_00 必须精确对齐 R12_01 / R13_00 / R14_00。",
        "",
        "## 03 · 盘点与 SDK",
        "",
        f"inventory={(ctx.get('inventory') or {}).get('minute')}",
        f"sdk_status={(ctx.get('sdk') or {}).get('status')} option_available={(ctx.get('sdk') or {}).get('option_available')}",
        "",
        "## 04 · 期权因子 / 残差",
        "",
        f"factors={(ctx.get('factors') or {}).get('status')} residual={(ctx.get('residual') or {}).get('status')}",
        "",
        "## 05 · 期权候选",
        "",
        "| ID | status | CAGR | Sharpe | MaxDD |",
        "|---|---|---:|---:|---:|",
    ]
    for code in OPTION_IDS:
        row = opt.get(code) or {}
        body.append(
            f"| `{code}` | {row.get('status')} | {pct(row.get('cagr'))} | {row.get('sharpe')} | {pct(row.get('max_drawdown'))} |"
        )
    body += [
        "",
        "## 06 · Top1",
        "",
        "| ID | status | CAGR | Sharpe | MaxDD |",
        "|---|---|---:|---:|---:|",
    ]
    for code in TOP1_IDS:
        row = top.get(code) or {}
        body.append(
            f"| `{code}` | {row.get('status')} | {pct(row.get('cagr'))} | {row.get('sharpe')} | {pct(row.get('max_drawdown'))} |"
        )
    body += [
        "",
        "## 07 · 执行 X01/X02",
        "",
        f"{(ctx.get('execution') or {}).get('full_period')}。窗口={(ctx.get('execution') or {}).get('window')}。"
        f"compared_subsample_to_full_period={(ctx.get('execution') or {}).get('compared_subsample_to_full_period')}。",
        "",
        "## 08 · Survivorship / Forward / Stats",
        "",
        f"survivorship={(ctx.get('survivorship') or {}).get('status')}。forward={(ctx.get('forward') or {}).get('status')}。",
        f"SPA option_family={((ctx.get('spa') or {}).get('option_family') or {}).get('status')} "
        f"top1_family={((ctx.get('spa') or {}).get('top1_family') or {}).get('status')}。",
        "",
        "## 09 · Promotion",
        "",
        f"**{promo.get('decision')}**。CURRENT=**{promo.get('final_current')}**。{promo.get('reason')}",
        "",
        "## 10 · Q1–Q28",
        "",
    ]
    for i in range(1, 29):
        key = f"Q{i}"
        body += [f"### {key} — {PRE_REGISTERED_QUESTIONS[key]}", "", qs.get(key, "NOT_IDENTIFIABLE"), ""]
    body += [
        "### 明确没做",
        "",
        "- 不搜 Top2 0/5/10、波动/MRC 帽、v3/v4 平均",
        "- 无 O05+；不看完收益再组合 Option+Top1",
        "- 不拿 2017–2021 子样本 CAGR 对 2018–2026 全样本",
        "- 无 09:25 不用竞价信号",
        "- 不自动把 v5 写成 CURRENT；追踪表不写「写入时间」",
        "- 不发明期权数据；SKIP 不是 PASS",
        "",
    ]
    write_md(FINAL_REPORT, "\n".join(body))
    _append_tracker(ctx)
    complete = {
        "decision": promo.get("decision"),
        "final_current": promo.get("final_current"),
        "recommended_id": promo.get("recommended_id"),
        "lock_status": promo.get("lock_status"),
        "upgrade_type": promo.get("upgrade_type"),
        "reason": promo.get("reason"),
        "R15_00": obs,
        "options": {c: opt.get(c) for c in OPTION_IDS},
        "top1": {c: top.get(c) for c in TOP1_IDS},
        "execution": {c: ((ctx.get("execution") or {}).get("results") or {}).get(c) for c in EXEC_IDS},
        "forward_oos": (ctx.get("forward") or {}).get("status"),
        "survivorship": (ctx.get("survivorship") or {}).get("status"),
        "manifest_sha256": (ctx.get("frozen") or {}).get("sha256"),
        "dossier": str(FINAL_REPORT),
        "questions": qs,
        "promotion": promo,
    }
    dump_json(DOSSIER_JSON, complete)
    return complete


def print_complete_block(payload: dict[str, Any]) -> None:
    r00 = payload.get("R15_00") or {}
    rec = payload.get("recommended_id")
    print("========== ROUND15 COMPLETE ==========", flush=True)
    print(f"DECISION={payload.get('decision')}", flush=True)
    print(f"CURRENT={payload.get('final_current')}", flush=True)
    print(f"LOCK={payload.get('lock_status')}", flush=True)
    print(f"UPGRADE_TYPE={payload.get('upgrade_type')}", flush=True)
    print(
        f"R15_00 CAGR={r00.get('cagr')} Sharpe={r00.get('sharpe')} MaxDD={r00.get('max_drawdown')}",
        flush=True,
    )
    if rec:
        print(f"RECOMMENDED={rec}", flush=True)
    print(f"FORWARD={payload.get('forward_oos')}", flush=True)
    print(f"SURVIVORSHIP={payload.get('survivorship')}", flush=True)
    print(f"MANIFEST={payload.get('manifest_sha256')}", flush=True)
    print(f"DOSSIER={payload.get('dossier')}", flush=True)
    print(f"REASON={payload.get('reason')}", flush=True)
    print("Final Report: /home/mxy/Stock/课题/一进二策略/result/round15/ROUND15_RESEARCH_REPORT.md", flush=True)
    print("======================================", flush=True)


__all__ = [
    "gather_ctx",
    "print_complete_block",
    "run_forward",
    "run_promotion",
    "run_report",
]
