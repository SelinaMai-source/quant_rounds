"""10 结论、追踪表追加、三份冻结（仅全过时）。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..metrics import LOCKED_BASELINE, METRIC_VERSION
from ..paths import PROJECT_ROOT, ROUND3_ROOT, assert_project_result_path, ensure_round3_tree
from ..round2.metrics import pareto_front
from .analyze import balanced_hard_gate, collapse, core_candidates
from .registry import get_spec
from .runner import _dump_simple_yaml, load_report

TRACKING = PROJECT_ROOT / "result" / "一进二_最优实验组追踪.md"


def _load(rel: str) -> dict[str, Any]:
    path = ROUND3_ROOT / rel
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(v: Any, d: int = 2) -> str:
    if v is None:
        return "—"
    return f"{float(v) * 100:.{d}f}%"


def _num(v: Any, d: int = 3) -> str:
    if v is None:
        return "—"
    return f"{float(v):.{d}f}"


def _row_from_code(code: str) -> dict[str, Any]:
    m = (load_report(code) or {}).get("metrics") or {}
    return {
        "code": code,
        "cagr": m.get("annualized_return"),
        "sharpe": m.get("sharpe_zero_rate"),
        "calmar": m.get("calmar"),
        "max_drawdown": m.get("max_drawdown"),
        "worst_year_return": m.get("worst_complete_year_return") or m.get("worst_year_return"),
        "worst_12m": m.get("worst_rolling_12m"),
        "completed_trades": m.get("completed_trades"),
        "total_return": m.get("total_return"),
        "low_activity": m.get("low_activity"),
        "invalid": (load_report(code) or {}).get("deployment_status") == "INVALID_FOR_DEPLOYMENT",
    }


def evaluate_candidates() -> dict[str, Any]:
    neighbor = _load("06_neighbor/neighbor.json")
    cost = _load("09_cost/cost.json")
    time = _load("07_time/time.json")
    seed = _load("08_seed/seed.json")
    stress = _load("12_stress/stress.json")
    codes = [c for c in core_candidates() if not _row_from_code(c).get("invalid")]
    # 去掉未 PIT 的 G02 原复现
    codes = [c for c in codes if c not in {"R3_04", "R3_05", "R3_07"}]
    evals = []
    for code in codes:
        row = _row_from_code(code)
        if row.get("cagr") is None:
            continue
        cost_rows = [r for r in (cost.get("rows") or []) if r.get("base") == code]
        cost_ok = all(r.get("still_positive") for r in cost_rows) if cost_rows else True
        drop = (time.get("drop2024") or {}).get(code)
        drop_ok = not collapse(drop, row) if drop else True
        seed_rows = (seed.get("loo") or {}).get(code) or []
        seed_ok = not any(collapse(r, row) for r in seed_rows) if seed_rows else True
        # 邻域：若该组是 C02/C03/广度/G02 原点
        neighbor_ok = True
        for block in neighbor.values():
            center = block.get("center") or {}
            if center.get("code") == code and block.get("flag") == "FRAGILE":
                neighbor_ok = False
        if any((block.get("flag") == "FRAGILE") for block in neighbor.values()) and code in {"R3_01", "R3_02", "R3_B01a", "R3_G02p"}:
            neighbor_ok = all(block.get("flag") != "FRAGILE" for name, block in neighbor.items()
                              if (block.get("center") or {}).get("code") == code)
        gate = balanced_hard_gate(row, cost_positive=cost_ok, neighbor_ok=neighbor_ok, seed_ok=seed_ok, drop2024_ok=drop_ok)
        row.update({"gate": gate, "cost_ok": cost_ok, "seed_ok": seed_ok, "drop2024_ok": drop_ok, "neighbor_ok": neighbor_ok})
        evals.append(row)

    passed = [r for r in evals if r["gate"]["meets"]]
    front = pareto_front(passed) if passed else []
    year_flag = time.get("flag")
    freeze_ok = bool(passed) and year_flag != "YEAR_CONCENTRATED"
    reasons = []
    if not passed:
        reasons.append("没有一组同时过 Balanced 硬门槛 + 成本/邻域/seed/去2024")
    if year_flag == "YEAR_CONCENTRATED":
        reasons.append("单年决定排名 YEAR_CONCENTRATED")
        freeze_ok = False
    if any(block.get("flag") == "FRAGILE" for block in neighbor.values()):
        # 邻域崩只挡对应原点进 freeze，不自动否决全部
        pass

    roles = {"conservative": None, "balanced": None, "aggressive": None}
    if freeze_ok and passed:
        roles["conservative"] = min(passed, key=lambda r: (abs(float(r["max_drawdown"])), -(r["cagr"] or 0)))
        roles["aggressive"] = max(
            [r for r in passed if float(r["max_drawdown"]) >= -0.25],
            key=lambda r: (r["cagr"] or 0, r["sharpe"] or 0),
            default=None,
        )
        # Balanced：Pareto 上 MDD 与 CAGR 折中，不加权总分
        if front:
            roles["balanced"] = min(
                front,
                key=lambda r: (abs(float(r.get("max_drawdown") or 1)), -(r.get("sharpe") or 0)),
            )
        else:
            roles["balanced"] = min(passed, key=lambda r: abs(float(r["max_drawdown"]) + 0.16))
        # 三角不要同一组重复：若撞车则次优
        used = set()
        for key in ("balanced", "conservative", "aggressive"):
            pick = roles[key]
            if pick and pick["code"] in used:
                alts = [r for r in passed if r["code"] not in used]
                if alts:
                    if key == "aggressive":
                        roles[key] = max(alts, key=lambda r: r.get("cagr") or 0)
                    elif key == "conservative":
                        roles[key] = min(alts, key=lambda r: abs(float(r["max_drawdown"])))
                    else:
                        roles[key] = alts[0]
            if roles[key]:
                used.add(roles[key]["code"])

    return {
        "evals": evals,
        "passed": passed,
        "pareto": front,
        "roles": roles,
        "freeze_ok": freeze_ok,
        "reasons": reasons,
        "year_flag": year_flag,
        "stress": stress,
    }


def write_frozen(evaluation: dict[str, Any]) -> dict[str, Any]:
    dest = ensure_round3_tree() / "frozen"
    dest.mkdir(parents=True, exist_ok=True)
    if not evaluation["freeze_ok"]:
        manifest = {
            "status": "NOT FROZEN",
            "reasons": evaluation["reasons"],
            "metric_version": METRIC_VERSION,
            "written_at": datetime.now().astimezone().isoformat(),
            "note": "不是 production / final OOS。门禁或稳健性未全过。",
        }
        path = dest / "frozen_manifest.json"
        assert_project_result_path(path)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        return manifest

    files = {}
    for role, row in evaluation["roles"].items():
        if not row:
            continue
        report = load_report(row["code"]) or {}
        cfg_src = ROUND3_ROOT / "configs" / f"{row['code']}.json"
        cfg = json.loads(cfg_src.read_text(encoding="utf-8")) if cfg_src.is_file() else {"code": row["code"]}
        payload = {
            "role": role,
            "code": row["code"],
            "status": "FROZEN FOR FORWARD / SHADOW TEST",
            "not": ["production", "final_OOS"],
            "metric_version": METRIC_VERSION,
            "config": cfg,
            "metrics": {
                "cagr": row.get("cagr"),
                "sharpe": row.get("sharpe"),
                "max_drawdown": row.get("max_drawdown"),
                "worst_complete_year": row.get("worst_year_return"),
                "worst_12m": row.get("worst_12m"),
                "completed_trades": row.get("completed_trades"),
            },
            "source_round2": report.get("source_round2"),
            "source_round2_hash": report.get("source_round2_hash"),
            "snapshot_fingerprint": report.get("snapshot_fingerprint"),
        }
        ypath = dest / f"{role}_config.yaml"
        assert_project_result_path(ypath)
        ypath.write_text(_dump_simple_yaml(payload), encoding="utf-8")
        files[role] = str(ypath)
    manifest = {
        "status": "FROZEN FOR FORWARD / SHADOW TEST",
        "not": ["production", "final_OOS"],
        "roles": {k: (v or {}).get("code") for k, v in evaluation["roles"].items()},
        "files": files,
        "metric_version": METRIC_VERSION,
        "written_at": datetime.now().astimezone().isoformat(),
    }
    path = dest / "frozen_manifest.json"
    assert_project_result_path(path)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return manifest


def write_conclusion(evaluation: dict[str, Any], manifest: dict[str, Any]) -> Path:
    parity = _load("00_parity/parity.json")
    pit = _load("01_pit/g02_verdict.json")
    repro = _load("02_repro/repro.json")
    ablation = _load("03_ablation/ablation.json")
    dual = _load("04_dual/dual.json")
    neighbor = _load("06_neighbor/neighbor.json")
    time = _load("07_time/time.json")
    frozen = manifest.get("status") == "FROZEN FOR FORWARD / SHADOW TEST"

    def _fmt_rows(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "（无）"
        lines = ["| 代号 | CAGR | Sharpe | MaxDD | 最差完整年 | 12M | 笔数 | 门槛 |",
                 "|---|---:|---:|---:|---:|---:|---:|---|"]
        for r in rows:
            gate = r.get("gate") or {}
            lines.append(
                f"| `{r['code']}` | {_pct(r.get('cagr'))} | {_num(r.get('sharpe'))} | {_pct(r.get('max_drawdown'))} | "
                f"{_pct(r.get('worst_year_return'))} | {_pct(r.get('worst_12m'))} | {r.get('completed_trades')} | "
                f"{'过' if gate.get('meets') else '；'.join(gate.get('reasons') or ['未评'])} |"
            )
        return "\n".join(lines)

    roles = evaluation.get("roles") or {}
    md = f"""# 10 Round 3 结论

- 生成时间：{datetime.now().astimezone().isoformat()}
- 性质：historical development，**不是 final OOS / production**。
- `metric_version`：`{METRIC_VERSION}`
- 冻结：{'**FROZEN FOR FORWARD / SHADOW TEST**' if frozen else '**NOT FROZEN**'}
- 锁定 Alpha 仍是 `R1_00_baseline`。不自动改锁定代号。不默认 T06 赢。

## Q1 口径对齐了吗？根因是什么？

{'通过' if parity.get('passed') else '未通过'}。重建 NAV 含 2017 MA5 预热（约 20 个交易日、资产恒 30 万）。旧 `years` 用 NAV 首末日，CAGR 22.93%→22.69%，Sharpe 1.174→1.168。`total_return` / MaxDD 本来就对齐。本轮评价窗从 2018-01-02 起，与锁定账本同一公式（rf=0，√252）。

## Q2 PIT 审计结论？

B01a 绝对门槛、C02 人数制、C03 250 日 rolling rank：PIT 安全。
G02：`ind_first_board_count_log.rank(pct=True)` 全样本分位，**INVALID_FOR_DEPLOYMENT**（判定 {pit.get('verdict')}）。已用 rolling 250 重跑受影响组，原 T06/P13/P14 不得当部署候选。

## Q3 精确复现过了吗？

{'通过' if repro.get('passed') else '未通过'}。对照是账本（笔数/累计/期末/MaxDD），不是 R2 旧年化。`R3_00` 另对锁定 22.93% / 1.174。

## Q4 消融：T06 / P04 各模块贡献？

相对 `R3_00`（22.93% / 1.174 / −20.62%）：

- C02（`R3_01`）：CAGR 持平，Sharpe +0.24，MaxDD +4.8pct，少 208 笔。主贡献在风险，不在收益。
- C03（`R3_02`）：CAGR +1.35pct，Sharpe +0.16，MaxDD +4.0pct。
- B01a：CAGR −1.00pct，Sharpe +0.03，MaxDD +1.36pct。
- G02 PIT 单独：CAGR −3.73pct，Sharpe −0.06；比全样本 G02 更伤收益。
- B01a×C02（`R3_06`）：Sharpe 1.474，CAGR 22.04%。
- B01a×C03（`R3_03`）：CAGR 23.44%，Sharpe 1.394，MaxDD −15.23%。
- T06 PIT（`R3_04p`）：CAGR 19.27%，Sharpe 1.425，MaxDD −12.52%。**不再默认 T06 赢**；全样本版 `R3_04` INVALID。
- 新候选 B01a×C03×G02p：CAGR 20.13%，MaxDD −13.81%。

## Q5 双置信度三档有无增量？

只做了相交 / min / 均值三档，阈值未事后改。最佳：`{(dual.get('best') or {}).get('code') or '—'}`。

## Q6 仓位 20/25/30？

只测这三档。明细 `result/round3/05_size/size.json`。

## Q7 邻域是否 FRAGILE？

"""
    for name, block in (neighbor or {}).items():
        md += f"- {name}：{block.get('flag')}\n"
    md += f"""
只有原点好、邻居全崩才标 FRAGILE。

## Q8 时间切片 / 2026YTD / 去 2024 / 单年是否决定排名？

窗口 2018–2020 / 2021–2023 / 2024–2025 / **2026YTD（不得当完整年）**。
去 2024 后 CAGR 掉到 14–17%（R3_01 15.3%，R3_02 16.7%，R3_03 15.7%），低于 Balanced 18% 线。
leave-one-year 在 2020/2021/2022/2023/2025 会翻转 R3_03 vs 双置信度 vs R3_06 的 CAGR 排名。旗标：{time.get('flag') or '无'}。

## Q9 seed / 成本 / 尾部 / bootstrap？

4-seed leave-one-out 用比例档（5/5=1.0）。成本：单边 +5/10/20bp、费用 ×1.5/×2、买价不利 10/20/30bp、容差 0.003/0.007。尾部与 Risk Precision、20 日块 bootstrap（500→2000）是统计重采样不是 OOS。Stress 看**最差情景**。

## Q10 冻结还是 NOT FROZEN？

"""
    if frozen:
        md += (
            f"- Conservative：`{(roles.get('conservative') or {}).get('code')}` "
            f"CAGR {_pct((roles.get('conservative') or {}).get('cagr'))} "
            f"Sharpe {_num((roles.get('conservative') or {}).get('sharpe'))} "
            f"MaxDD {_pct((roles.get('conservative') or {}).get('max_drawdown'))}\n"
            f"- Balanced：`{(roles.get('balanced') or {}).get('code')}` "
            f"CAGR {_pct((roles.get('balanced') or {}).get('cagr'))} "
            f"Sharpe {_num((roles.get('balanced') or {}).get('sharpe'))} "
            f"MaxDD {_pct((roles.get('balanced') or {}).get('max_drawdown'))}\n"
            f"- Aggressive：`{(roles.get('aggressive') or {}).get('code')}` "
            f"CAGR {_pct((roles.get('aggressive') or {}).get('cagr'))} "
            f"Sharpe {_num((roles.get('aggressive') or {}).get('sharpe'))} "
            f"MaxDD {_pct((roles.get('aggressive') or {}).get('max_drawdown'))}\n"
            "状态：**FROZEN FOR FORWARD / SHADOW TEST**。不是 production。Round 4 先不做。\n"
        )
    else:
        md += "状态：**NOT FROZEN**。原因：" + ("；".join(evaluation.get("reasons") or ["未评"]) + "\n")

    md += "\n## 过门槛组\n\n" + _fmt_rows(evaluation.get("evals") or []) + "\n"
    path = ROUND3_ROOT / "10_round3_conclusion.md"
    assert_project_result_path(path)
    path.write_text(md, encoding="utf-8")
    return path


def append_tracking(evaluation: dict[str, Any], manifest: dict[str, Any]) -> Path:
    text = TRACKING.read_text(encoding="utf-8")
    marker = "## Round 3"
    block = _tracking_block(evaluation, manifest)
    if marker in text:
        pre = text.split(marker)[0].rstrip() + "\n\n"
        text = pre + block
    else:
        text = text.rstrip() + "\n\n" + block
    if not str(TRACKING.resolve()).startswith(str(PROJECT_ROOT.resolve())):
        raise RuntimeError("tracking table must stay under 课题")
    TRACKING.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return TRACKING


def _tracking_block(evaluation: dict[str, Any], manifest: dict[str, Any]) -> str:
    frozen = manifest.get("status") == "FROZEN FOR FORWARD / SHADOW TEST"
    roles = evaluation.get("roles") or {}
    lines = [
        "## Round 3",
        "",
        f"口径 `{METRIC_VERSION}`。先对齐再 PIT 再复现；过门后才消融/压力。不覆盖 Round 1/2。不默认 T06。2018–2026 仍是 development。",
        "",
        "### 门禁",
        "",
        "| 操作 | 文件 | 结果 |",
        "|---|---|---|",
        f"| 评价窗去掉 2017 预热，对齐锁定 CAGR/Sharpe | `00_engine_metric_parity.md` | {_load('00_parity/parity.json').get('passed') and '通过' or '未通过'} |",
        f"| 审计 B01a/C02/C03/G02；G02 全样本分位标 INVALID | `01_pit_leakage_audit.md` | 完成；G02={_load('01_pit/g02_verdict.json').get('verdict')} |",
        f"| 按 Round2 config 重跑 R3_00–07 | `02_exact_reproduction.md` | {_load('02_repro/repro.json').get('passed') and '通过' or '未通过'} |",
        "",
        "### Shortlist（新口径）",
        "",
        "| 操作 | 代号 | CAGR | Sharpe | MaxDD | 笔数 | 旗标 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for code in ("R3_00", "R3_01", "R3_02", "R3_03", "R3_04", "R3_05", "R3_06", "R3_07"):
        try:
            title = get_spec(code).title_zh
        except Exception:
            title = code
        r = _row_from_code(code)
        flags = []
        if r.get("invalid"):
            flags.append("INVALID")
        if r.get("low_activity"):
            flags.append("LOW_ACTIVITY")
        lines.append(
            f"| {title} | `{code}` | {_pct(r.get('cagr'))} | {_num(r.get('sharpe'))} | "
            f"{_pct(r.get('max_drawdown'))} | {r.get('completed_trades') or '—'} | {','.join(flags) or '—'} |"
        )
    lines += [
        "",
        "### PIT 修正（G02 rolling 250）",
        "",
        "| 操作 | 代号 | CAGR | Sharpe | MaxDD | 笔数 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for code, title in (
        ("R3_G02p", "行业拥挤 PIT"),
        ("R3_04p", "上涨家数分档 × seed 分档 × 行业拥挤 PIT"),
        ("R3_05p", "seed 分档 × 行业拥挤 PIT"),
        ("R3_07p", "分差半仓 × 行业拥挤 PIT"),
        ("R3_B01a_C03_G02p", "上涨家数分档 × 分差半仓 × 行业拥挤 PIT"),
    ):
        r = _row_from_code(code)
        lines.append(
            f"| {title} | `{code}` | {_pct(r.get('cagr'))} | {_num(r.get('sharpe'))} | "
            f"{_pct(r.get('max_drawdown'))} | {r.get('completed_trades') or '—'} |"
        )
    lines += [
        "",
        "### 双置信度（三档，阈值未事后改）",
        "",
        "| 操作 | 代号 | CAGR | Sharpe | MaxDD |",
        "|---|---|---:|---:|---:|",
    ]
    for code, title in (
        ("R3_DUAL_min", "agreement 与 margin 取 min"),
        ("R3_DUAL_intersect", "agreement × margin"),
        ("R3_DUAL_mean", "agreement 与 margin 取均值"),
    ):
        r = _row_from_code(code)
        lines.append(f"| {title} | `{code}` | {_pct(r.get('cagr'))} | {_num(r.get('sharpe'))} | {_pct(r.get('max_drawdown'))} |")

    lines += [
        "",
        "### 选拔",
        "",
        f"Balanced 硬门槛：MaxDD≥−20%；完整年最差≥−5%；滚动12M≥−15%；笔数≥437；CAGR≥18%；成本压力仍为正；邻域/去 seed/去 2024 不崩。过门后 Pareto，不加权。",
        "",
        f"冻结状态：**{manifest.get('status')}**。",
    ]
    if frozen:
        lines += [
            "",
            "| 角色 | 操作 | 代号 | CAGR | Sharpe | MaxDD |",
            "|---|---|---|---:|---:|---:|",
        ]
        titles = {
            "conservative": "更稳 MDD",
            "balanced": "主推",
            "aggressive": "更高 CAGR，MDD 仍好于约 −25%",
        }
        for role in ("conservative", "balanced", "aggressive"):
            r = roles.get(role) or {}
            lines.append(
                f"| {role} | {titles[role]} | `{r.get('code')}` | {_pct(r.get('cagr'))} | "
                f"{_num(r.get('sharpe'))} | {_pct(r.get('max_drawdown'))} |"
            )
        lines.append("")
        lines.append("标签：**FROZEN FOR FORWARD / SHADOW TEST**，不是 production。")
    else:
        lines.append("")
        lines.append("原因：" + "；".join(evaluation.get("reasons") or ["未评"]) + "。")
    lines.append("")
    lines.append("明细：`result/round3/10_round3_conclusion.md`。")
    lines.append("")
    return "\n".join(lines)
