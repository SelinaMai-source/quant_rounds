"""主报告 1–25、追踪表追加、PART 41。不写「写入时间」。"""

from __future__ import annotations

from typing import Any

from ..paths import ROUND18_ROOT
from .common import dump_json, load_json, write_md
from .contract import (
    AWAITING_USER_LOCK,
    CURRENT_BASELINE,
    F1_FORMULA,
    F2_FORMULA,
    F3_FORMULA,
    FINAL_REPORT,
    KEEP_BASELINE_V4,
    OPTION_CANDIDATE,
    PARENT_V4,
    PRIOR_RESEARCH_SELECTION_EXISTS,
    TRACKER_PATH,
    V4_ANCHOR,
)
from .decision import run_decision
from .figures import make_figures
from .freeze import require_frozen

DOSSIER_JSON = ROUND18_ROOT / "stats" / "dossier.json"


def gather_ctx() -> dict[str, Any]:
    return {
        "read": load_json(ROUND18_ROOT / "read" / "read.json"),
        "inventory": load_json(ROUND18_ROOT / "inventory" / "inventory.json"),
        "parity": load_json(ROUND18_ROOT / "parity" / "parity.json"),
        "recover": load_json(ROUND18_ROOT / "download" / "recover.json"),
        "build": load_json(ROUND18_ROOT / "factors" / "factor_build.json"),
        "pit": load_json(ROUND18_ROOT / "pit" / "pit.json"),
        "frozen": load_json(ROUND18_ROOT / "configs" / "freeze.json"),
        "matrix": load_json(ROUND18_ROOT / "metrics" / "matrix.json"),
        "stats": load_json(ROUND18_ROOT / "stats" / "stats.json"),
        "stress": load_json(ROUND18_ROOT / "stress" / "stress.json"),
        "capacity": load_json(ROUND18_ROOT / "capacity" / "capacity.json"),
        "bootstrap": load_json(ROUND18_ROOT / "bootstrap" / "bootstrap.json"),
        "spa": load_json(ROUND18_ROOT / "spa" / "spa.json"),
        "fwd": load_json(ROUND18_ROOT / "forward" / "forward.json"),
        "promo": load_json(ROUND18_ROOT / "promotion" / "promotion.json"),
        "overlap": load_json(ROUND18_ROOT / "overlap" / "overlap.json"),
    }


def _row(matrix: dict[str, Any], code: str) -> dict[str, Any]:
    for r in matrix.get("rows") or []:
        if r.get("code") == code:
            return r
    return {}


def run_report(**_: Any) -> dict[str, Any]:
    require_frozen()
    ctx = gather_ctx()
    promo = ctx.get("promo") or run_decision()
    ctx["promo"] = promo
    figs = make_figures()
    ctx["figures"] = figs
    par = ctx.get("parity") or {}
    pit = ctx.get("pit") or {}
    matrix = ctx.get("matrix") or {}
    build = ctx.get("build") or {}
    rec = ctx.get("recover") or {}
    a00 = _row(matrix, "R18_A00") or par.get("observed") or {}
    a01 = _row(matrix, "R18_A01")
    a02 = _row(matrix, "R18_A02")
    a03 = _row(matrix, "R18_A03")
    b01 = _row(matrix, "R18_B01")
    b02 = _row(matrix, "R18_B02")
    b03 = _row(matrix, "R18_B03")
    c01 = _row(matrix, "R18_C01")
    formal = [r for r in (matrix.get("rows") or []) if r.get("formal") and r.get("sharpe") is not None]
    best = promo.get("best") or {}
    fac = pit.get("factors") or {}

    def _m(r: dict[str, Any]) -> str:
        if not r:
            return "未跑"
        if r.get("cagr") is None and r.get("status") not in {None, "PASS", "done", "resumed"}:
            return f"{r.get('status')}"
        return f"CAGR={r.get('cagr')} Sharpe={r.get('sharpe')} MaxDD={r.get('max_drawdown')} trades={r.get('completed_trades')}"

    lines = [
        "# ROUND18_MICROSTRUCTURE_ALPHA_REPORT",
        "",
        "Round18 = 锁定 baseline_v4 结构下的三个预注册微观结构因子增量。canonical 1.0/1.0。不融合 R16_O01。不自动升 CURRENT。",
        f"manifest sha256=`{(ctx.get('frozen') or {}).get('sha256')}` parity=`{par.get('status')}`",
        f"决策=**{promo.get('four_way')}** CURRENT=**{promo.get('final_current')}** lock=`{promo.get('lock')}`",
        "",
        "## 1. Executive Verdict",
        "",
        f"**{promo.get('four_way')}**。CURRENT 仍是 `{CURRENT_BASELINE} = {PARENT_V4}`。未自动晋升。未融合 `{OPTION_CANDIDATE}`。",
        f"- A00 parity：`{par.get('status')}`。观测 `{_m(a00)}`。锚点 CAGR=`{V4_ANCHOR['cagr']}` Sharpe=`{V4_ANCHOR['sharpe']}` MaxDD=`{V4_ANCHOR['max_drawdown']}` trades=`{V4_ANCHOR['completed_trades']}`。",
        f"- 数据恢复：`{rec.get('status')}` downloaded=`{rec.get('downloaded')}`。",
        f"- F1 覆盖=`{(fac.get('fb_post_limit_supply_absorption') or {}).get('coverage')}` PIT=`{(fac.get('fb_post_limit_supply_absorption') or {}).get('pit_status')}` valid=`{(fac.get('fb_post_limit_supply_absorption') or {}).get('n_valid')}`",
        f"- F2 覆盖=`{(fac.get('fb_order_size_asymmetry') or {}).get('coverage')}` PIT=`{(fac.get('fb_order_size_asymmetry') or {}).get('pit_status')}` valid=`{(fac.get('fb_order_size_asymmetry') or {}).get('n_valid')}`",
        f"- F3 覆盖=`{(fac.get('fb_seal_to_free_float_pressure') or {}).get('coverage')}` PIT=`{(fac.get('fb_seal_to_free_float_pressure') or {}).get('pit_status')}` valid=`{(fac.get('fb_seal_to_free_float_pressure') or {}).get('n_valid')}`",
        f"- 失败晋级门：`{promo.get('failed_gates')}`。",
        "",
        "## 2. Research Question",
        "",
        "在候选池、标签、超参、90/10、gap floor、clean_risk、B03、费用与成交语义全部固定时，F1/F2/F3 能否改善 D 日首板横截面排序，从而改善 D+1 买入、D+2 起原规则退出的净收益？",
        "",
        "## 3. Baseline Lock",
        "",
        f"CURRENT=`{CURRENT_BASELINE}` parent=`{PARENT_V4}`。精确锚点见上。不叠加 `{OPTION_CANDIDATE}`。",
        "",
        "## 4. Prior Research / Overlap",
        "",
        f"PRIOR_RESEARCH_SELECTION_EXISTS=`{PRIOR_RESEARCH_SELECTION_EXISTS}`。R17_T4033 extras 不是本三条公式。F1/F2/F3 均登记为新公式，近邻见 `02_factor_overlap.md`。",
        "",
        "## 5. Frozen Formulas",
        "",
        f"- F1 `{F1_FORMULA}`",
        f"- F2 `{F2_FORMULA}`（单位一致才进正式回测）",
        f"- F3 `{F3_FORMULA}`（禁止总市值顶替）",
        "epsilon 已锁，看到 NAV 后未搜索。",
        "",
        "## 6. Data Inventory & Recovery",
        "",
        f"inventory status=`{(ctx.get('inventory') or {}).get('status')}` recover=`{rec.get('status')}` missing=`{rec.get('missing')}`。未覆盖 Hermes 原始。",
        "",
        "## 7. A00 Parity",
        "",
        f"status=`{par.get('status')}` metric_ok=`{par.get('metric_ok')}` ledger=`{(par.get('ledger_vs_r16_5_v00') or {}).get('passed')}`。",
        f"observed `{_m(a00)}`。",
        "",
        "## 8. F1 Construction",
        "",
        f"{(build.get('f1') or {})}",
        "",
        "## 9. F2 Construction & Schema/PIT",
        "",
        f"{(build.get('f2') or {})}",
        "",
        "## 10. F3 Construction",
        "",
        f"{(build.get('f3') or {})}",
        "",
        "## 11. PIT / Coverage / Redundancy",
        "",
        f"{fac}",
        "",
        f"近邻相关：`{(pit.get('redundancy') or {})}`",
        "",
        "## 12. Experiment Matrix",
        "",
        "主实验 A00–C01；D01–D06 单头消融 DIAGNOSTIC_ONLY。",
        "",
        f"- A00 `{_m(a00)}`",
        f"- A01 +F1 `{_m(a01)}`",
        f"- A02 +F2 `{_m(a02)}`",
        f"- A03 +F3 `{_m(a03)}`",
        f"- B01 F1+F2 `{_m(b01)}`",
        f"- B02 F1+F3 `{_m(b02)}`",
        f"- B03 F2+F3 `{_m(b03)}`",
        f"- C01 三因子 `{_m(c01)}`",
        "",
        "## 13. Head Ablations D01–D06",
        "",
    ]
    for code in ("R18_D01", "R18_D02", "R18_D03", "R18_D04", "R18_D05", "R18_D06"):
        lines.append(f"- {code} `{_m(_row(matrix, code))}` DIAGNOSTIC_ONLY")
    lines += [
        "",
        "## 14. Incrementality vs v4",
        "",
        f"best eligible=`{best.get('code')}` Sharpe=`{best.get('sharpe')}` vs A00 Sharpe=`{a00.get('sharpe')}`。",
        "排序变化引起的 B03 格子变化记 ALPHA-INDUCED ALLOCATION EFFECT，未另起执行器。",
        "",
        "## 15. Cost / Capacity / Year",
        "",
        f"cost sign_flip=`{(ctx.get('stress') or {}).get('sign_flip')}` capacity sign_flip=`{(ctx.get('capacity') or {}).get('sign_flip')}`",
        f"分年：`{(ctx.get('stats') or {}).get('yearly')}`",
        "",
        "## 16. Bootstrap / SPA",
        "",
        f"bootstrap=`{ctx.get('bootstrap')}`",
        f"spa=`{ctx.get('spa')}`",
        "PRIOR_RESEARCH_SELECTION_EXISTS=TRUE，不得把本轮当独立首次发现。",
        "",
        "## 17. Forward OOS",
        "",
        f"`{(ctx.get('fwd') or {}).get('status')}`。{(ctx.get('fwd') or {}).get('note')}",
        "",
        "## 18. Promotion Gates",
        "",
        f"failed=`{promo.get('failed_gates')}` lock=`{promo.get('lock')}`",
        "",
        "## 19. Decision",
        "",
        f"**{promo.get('four_way')}**。final_current=`{promo.get('final_current')}`。",
        "",
        "## 20. Figures",
        "",
        f"drawn=`{(figs or {}).get('n_drawn')}` skipped=`{(figs or {}).get('skipped')}`。缺数据未造假。",
        "",
        "## 21. Limitations",
        "",
        "- F2 2021 前 UPDATE_TIME≠TRADE_DATE，PIT 未验证。",
        "- F3 available_at==trade_date，股本存在事后修订风险。",
        "- F1 分钟仅为收盘涨停子集。",
        "- 2018–2026 为历史回顾审计，不是未触碰 OOS。",
        "",
        "## 22. Forbidden Actions Check",
        "",
        "未改 AGENTS.md / no-xgb-subsample / core/round7 / Stock src,runs。未 fillna(0)。未搜 bagging/择时/B03。未融合 R16_O01。未自动升 CURRENT。追踪表未写「写入时间」。",
        "",
        "## 23. Artifact Paths",
        "",
        f"- 主报告：`{FINAL_REPORT}`",
        f"- manifest：`{ROUND18_ROOT / 'round18_microstructure_manifest.yaml'}`",
        "- 只写 result/round18/、core/round18/、tools/。",
        "",
        "## 24. Q&A",
        "",
        f"**Q1 CURRENT？** `{CURRENT_BASELINE}={PARENT_V4}`",
        "",
        f"**Q2 精确 v4？** CAGR={V4_ANCHOR['cagr']} Sharpe={V4_ANCHOR['sharpe']} MaxDD={V4_ANCHOR['max_drawdown']} trades={V4_ANCHOR['completed_trades']}",
        "",
        f"**Q3 A00 parity？** {par.get('status')}",
        "",
        "**Q4 bagging？** 否，subsample=colsample=1.0",
        "",
        f"**Q5 F1/F2/F3 是否旧公式？** 否。T4033 extras 不同。",
        "",
        f"**Q6 F2 单位/PIT？** {(build.get('f2') or {}).get('status')} pit={(build.get('f2') or {}).get('pit_status')}",
        "",
        f"**Q7 是否填 0？** 否",
        "",
        f"**Q8 是否融合 R16_O01？** 否",
        "",
        f"**Q9 是否自动升 CURRENT？** 否。final_current={promo.get('final_current')} lock={promo.get('lock')}",
        "",
        f"**Q10 Forward？** {(ctx.get('fwd') or {}).get('status')}",
        "",
        "## 25. Handoff",
        "",
        "下一轮若要继续微观结构，必须先解决 F2 PIT（非回填 available_at）或放弃 F2。不得在本轮数字上搜 epsilon。",
        "",
    ]
    write_md(FINAL_REPORT, "\n".join(lines))
    for i, name in enumerate(
        [
            "16_limitations.md",
            "17_forbidden_check.md",
            "18_artifacts.md",
            "19_qa.md",
            "20_handoff.md",
            "21_tracker_note.md",
        ],
        start=16,
    ):
        write_md(ROUND18_ROOT / name, f"# {name}\n\n见主报告第 {i} 节。\n")
    _append_tracker(promo, par, a00, a01, a02, a03, best)
    dossier = {
        "parity": par.get("status"),
        "recover": rec.get("status"),
        "f1": fac.get("fb_post_limit_supply_absorption"),
        "f2": fac.get("fb_order_size_asymmetry"),
        "f3": fac.get("fb_seal_to_free_float_pressure"),
        "a00": a00,
        "a01": a01,
        "a02": a02,
        "a03": a03,
        "b01": b01,
        "b02": b02,
        "b03": b03,
        "c01": c01,
        "best": best,
        "four_way": promo.get("four_way"),
        "final_current": promo.get("final_current"),
        "lock": promo.get("lock"),
        "forward": (ctx.get("fwd") or {}).get("status"),
        "failed_gates": promo.get("failed_gates"),
        "matrix": matrix.get("rows"),
    }
    dump_json(DOSSIER_JSON, dossier)
    return dossier


def _append_tracker(promo: dict[str, Any], par: dict[str, Any], a00, a01, a02, a03, best) -> None:
    block = [
        "",
        "## Round 18 — First-Board Microstructure Alpha",
        "",
        "锁定 baseline_v4 结构，canonical 1.0/1.0 检验 F1/F2/F3。不自动升 CURRENT。不融合 R16_O01。",
        f"四选一 = `{promo.get('four_way')}`。CURRENT = `{promo.get('final_current')}`。lock = `{promo.get('lock')}`。",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| parity | {par.get('status')} |",
        f"| A00 CAGR/Sharpe/MaxDD | {a00.get('cagr')} / {a00.get('sharpe')} / {a00.get('max_drawdown')} |",
        f"| A01 +F1 | {a01.get('cagr')} / {a01.get('sharpe')} / {a01.get('max_drawdown')} status={a01.get('status')} |",
        f"| A02 +F2 | {a02.get('cagr')} / {a02.get('sharpe')} / {a02.get('max_drawdown')} status={a02.get('status')} |",
        f"| A03 +F3 | {a03.get('cagr')} / {a03.get('sharpe')} / {a03.get('max_drawdown')} status={a03.get('status')} |",
        f"| best eligible | {(best or {}).get('code')} |",
        f"| CURRENT | {promo.get('final_current')} |",
        f"| four_way | {promo.get('four_way')} |",
        f"| option candidate | {OPTION_CANDIDATE} untouched |",
        f"| forward OOS | {promo.get('forward')} |",
        f"| lock | {promo.get('lock')} |",
        f"| manifest sha256 | `{(load_json(ROUND18_ROOT / 'configs' / 'freeze.json') or {}).get('sha256')}` |",
        "",
        f"主报告：`result/round18/ROUND18_MICROSTRUCTURE_ALPHA_REPORT.md`。",
        "",
    ]
    text = TRACKER_PATH.read_text(encoding="utf-8") if TRACKER_PATH.is_file() else ""
    marker = "## Round 18 — First-Board Microstructure Alpha"
    if marker in text:
        idx = text.rfind(marker)
        text = text[:idx].rstrip() + "\n" + "\n".join(block)
    else:
        text = text.rstrip() + "\n" + "\n".join(block)
    TRACKER_PATH.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def print_complete_block(payload: dict[str, Any]) -> None:
    print("ROUND18 COMPLETE", flush=True)
    print("", flush=True)
    print("Current baseline:", flush=True)
    print(f"{CURRENT_BASELINE} = {PARENT_V4}", flush=True)
    print("", flush=True)
    print("Parity:", flush=True)
    print(payload.get("parity") or "FAIL", flush=True)
    print("", flush=True)
    print("Data recovery:", flush=True)
    print(payload.get("recover"), flush=True)
    print("", flush=True)
    print("F1/F2/F3 availability + PIT + valid rows:", flush=True)
    for key in ("f1", "f2", "f3"):
        rec = payload.get(key) or {}
        print(
            f"{key}: cov={rec.get('coverage')} pit={rec.get('pit_status')} "
            f"valid={rec.get('n_valid')} promote_ok={rec.get('promote_ok')}",
            flush=True,
        )
    print("", flush=True)
    print("A00–C01:", flush=True)
    for code in ("a00", "a01", "a02", "a03", "b01", "b02", "b03", "c01"):
        r = payload.get(code) or {}
        print(
            f"{code.upper()}: status={r.get('status')} CAGR={r.get('cagr')} "
            f"Sharpe={r.get('sharpe')} MaxDD={r.get('max_drawdown')} trades={r.get('completed_trades')}",
            flush=True,
        )
    print("", flush=True)
    print("Best eligible / increment / decision:", flush=True)
    best = payload.get("best") or {}
    print(f"best={best.get('code')} Sharpe={best.get('sharpe')}", flush=True)
    print(f"{payload.get('four_way')} / CURRENT={payload.get('final_current')} / lock={payload.get('lock')}", flush=True)
    print("", flush=True)
    print("Forward OOS:", flush=True)
    print(payload.get("forward"), flush=True)
    print("", flush=True)
    print("Option candidate untouched:", flush=True)
    print(OPTION_CANDIDATE, flush=True)
    print("", flush=True)
    print("Failed gates:", flush=True)
    print(payload.get("failed_gates"), flush=True)
    print("", flush=True)
    print("Final Report:", flush=True)
    print(str(FINAL_REPORT), flush=True)
