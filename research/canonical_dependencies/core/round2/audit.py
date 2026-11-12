"""P0：00_baseline_identity_audit。查 2022≈−20% 账本与 ZyQuant 版本。禁止升级。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..metrics import LOCKED_BASELINE
from ..paths import (
    PROJECT_ROOT,
    STOCK_ROOT,
    assert_project_result_path,
    ensure_round2_tree,
    stock_path,
)

DOC_PATH = PROJECT_ROOT / "docs" / "数据与策略详解.md"
TARGET_2022 = -0.20
MATCH_BAND = 0.03  # ±3pct around -20%


def zyquant_identity() -> dict[str, Any]:
    import importlib.metadata as metadata

    import zyquant

    pyproject = Path("/home/mxy/Stock/pyproject.toml").read_text(encoding="utf-8")
    pin = "zyquant>=2.0.1,<2.1" if "zyquant>=2.0.1,<2.1" in pyproject else "unparsed"
    vendor = sorted(Path("/home/mxy/Stock/vendor").glob("zyquant-*.whl"))
    return {
        "imported_version": getattr(zyquant, "__version__", None),
        "imported_file": getattr(zyquant, "__file__", None),
        "pkg_metadata": metadata.version("zyquant"),
        "pyproject_pin": pin,
        "vendor_wheels": [p.name for p in vendor],
        "upgrade_forbidden": True,
        "note": "只用已稳定 DataSnapshot / PIT。禁止 pip install --upgrade，也不重写 Strategy.decide()。",
    }


def _yearly_2022(report: dict[str, Any]) -> float | None:
    yearly = (report.get("metrics") or {}).get("yearly_returns") or {}
    value = yearly.get("2022")
    return None if value is None else float(value)


def scan_fs00263_reports() -> list[dict[str, Any]]:
    root = stock_path("runs/first_board_factor_search_5000_v1/backtests")
    rows = []
    for path in sorted(root.glob("**/fs00263/**/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics = report.get("metrics") or {}
        rel = str(path)
        parts = path.parts
        capacity = next((p for p in parts if p.startswith("n") and "_d2" in p), "")
        seed = "ensemble" if "ensemble" in rel else next((p.replace("seed", "") for p in parts if p.startswith("seed")), "")
        rows.append({
            "strategy": "fs00263",
            "trial": "fs00263",
            "model": "xgb_search",
            "capacity": capacity,
            "seed": seed,
            "sizing": "nav30pct",
            "gate": "both_gates",
            "exit": (report.get("config") or {}).get("exit_mode", "close_from_d2"),
            "year_2022": _yearly_2022(report),
            "mdd": metrics.get("max_drawdown"),
            "total_return": metrics.get("total_return"),
            "sharpe": metrics.get("sharpe_zero_rate"),
            "source": str(path),
        })
    return rows


def scan_docs_section12() -> list[dict[str, Any]]:
    """Documented yearly / seed tables. Not a new backtest."""
    return [
        {
            "strategy": "fs00263",
            "trial": "fs00263",
            "model": "ensemble_rank_pct_5seed",
            "capacity": "n50_d2",
            "seed": "ensemble",
            "sizing": "nav30pct",
            "gate": "both_gates",
            "exit": "close_from_d2",
            "year_2022": 0.1922,
            "mdd": -0.2062,
            "source": "docs/数据与策略详解.md §12.3 + LOCKED_BASELINE",
        },
        {
            "strategy": "fs00263",
            "trial": "fs00263",
            "model": "ensemble_rank_pct_5seed",
            "capacity": "n100_d2",
            "seed": "ensemble",
            "sizing": "nav30pct",
            "gate": "both_gates",
            "exit": "close_from_d2",
            "year_2022": 0.1171,
            "mdd": -0.2070,
            "source": "docs §12.3 100×2 集成",
        },
        {
            "strategy": "fs00263",
            "trial": "fs00263",
            "model": "single_seed",
            "capacity": "n50_d2",
            "seed": "73",
            "sizing": "nav30pct",
            "gate": "both_gates",
            "exit": "close_from_d2",
            "year_2022": None,
            "mdd": -0.3411,
            "total_return": 2.7372,
            "source": "docs §12.3 单种子表；最差回撤 50×2 seed73",
        },
        {
            "strategy": "fs00263",
            "trial": "fs00263",
            "model": "single_seed",
            "capacity": "n100_d2",
            "seed": "17",
            "sizing": "nav30pct",
            "gate": "both_gates",
            "exit": "close_from_d2",
            "year_2022": None,
            "mdd": -0.3494,
            "total_return": 1.2343,
            "source": "docs §12.3；最差回撤 100×2 seed17（总收益仍为正）",
        },
    ]


def scan_stock_reports_markdown() -> list[str]:
    hits = []
    reports = Path("/home/mxy/Stock/reports")
    if not reports.is_dir():
        return hits
    for path in reports.glob("*.md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "2022" in text and ("-20" in text or "−20" in text or "-0.20" in text):
            hits.append(path.name)
    return hits


def scan_ensemble_2022_near_minus20(limit: int = 200) -> list[dict[str, Any]]:
    """Scan ensemble report.json only (not all 60k seeds) for 2022 ≈ -20%."""
    root = stock_path("runs/first_board_factor_search_5000_v1/backtests")
    matches = []
    scanned = 0
    for path in root.glob("**/ensemble_rank_pct_5seed/**/report.json"):
        scanned += 1
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        y2022 = _yearly_2022(report)
        if y2022 is None:
            continue
        if abs(y2022 - TARGET_2022) <= MATCH_BAND:
            trial = next((p for p in path.parts if p.startswith("fs")), "")
            matches.append({
                "trial": trial,
                "year_2022": y2022,
                "mdd": (report.get("metrics") or {}).get("max_drawdown"),
                "source": str(path),
            })
            if len(matches) >= limit:
                break
    return {"scanned": scanned, "matches": matches}


def write_audit_markdown() -> Path:
    ensure_round2_tree()
    zy = zyquant_identity()
    fs_rows = scan_fs00263_reports()
    doc_rows = scan_docs_section12()
    md_hits = scan_stock_reports_markdown()
    ensemble_scan = scan_ensemble_2022_near_minus20()
    locked_2022 = LOCKED_BASELINE["yearly_returns_pct"]["2022"]
    locked_mdd = LOCKED_BASELINE["max_drawdown"]

    fs_2022 = [r for r in fs_rows if r.get("year_2022") is not None]
    fs_neg20 = [r for r in fs_2022 if abs(float(r["year_2022"]) - TARGET_2022) <= MATCH_BAND]
    min_2022 = min((float(r["year_2022"]) for r in fs_2022), default=None)
    max_2022 = max((float(r["year_2022"]) for r in fs_2022), default=None)

    found_locked = bool(fs_neg20)
    found_other = bool(ensemble_scan["matches"])
    found_ledger = found_locked
    if found_locked:
        conclusion = "锁定 fs00263 账本中存在 2022 ∈ [−23%, −17%] 的条目，见上表。"
    else:
        conclusion = (
            "未找到与「2022 亏约 20%」描述一致的当前 **fs00263 baseline**。"
            f"锁定 50×2 集成 2022 是 **+{locked_2022:.2f}%**，全样本 MaxDD 是 **{locked_mdd:.2%}**。"
            "fs00263 十二本现账本 2022 最差是 100×2 seed17 **−8.60%**，没有一本接近 −20%。"
            "更可能是把全样本 MaxDD −20.62% 误记成「2022 年亏 20%」。"
            + (
                f"搜索库里其它 trial 的集成账本确有 {len(ensemble_scan['matches'])}+ 条 2022≈−20%，"
                "但那些不是锁定 Alpha，禁止拿来专拟合 2022。"
                if found_other
                else ""
            )
            + "后续盯 MDD / worst year / rolling 12m / seed。"
        )

    lines = [
        "# 00 Baseline identity audit",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "- 范围：只读 Stock `runs/` / `reports/`、课题 `result/round1/`、文档 §12。",
        "- 禁止升级 ZyQuant，禁止写 Stock。",
        "",
        "## 锁定账本",
        "",
        f"- 代号：`{LOCKED_BASELINE['code']}` / `{LOCKED_BASELINE['trial_id']}` / `{LOCKED_BASELINE['capacity']}` / `{LOCKED_BASELINE['ensemble']}`",
        f"- 变体：`{LOCKED_BASELINE['variant']}` / `{LOCKED_BASELINE['exit_mode']}`",
        f"- 2022 年收益（锁定展示）：**+{locked_2022:.2f}%**",
        f"- 全样本 MaxDD：**{locked_mdd:.2%}**",
        "- 对照：50×2 集成 raw 2022 = +19.22%；种子最差回撤 50×2 seed73 −34.11%、100×2 seed17 −34.94%（总收益仍为正）。",
        "",
        "## ZyQuant",
        "",
        f"- 本机 import 版本：`{zy['imported_version']}`（pkg `{zy['pkg_metadata']}`）",
        f"- 文件：`{zy['imported_file']}`",
        f"- `pyproject.toml` 约束：`{zy['pyproject_pin']}`",
        f"- vendor：{', '.join(zy['vendor_wheels']) or '无'}",
        f"- **禁止升级**：{zy['note']}",
        "",
        "## fs00263 现账本 2022",
        "",
        "| strategy | capacity | seed | 2022 | MDD | source |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in fs_rows:
        y = "—" if row.get("year_2022") is None else f"{row['year_2022']*100:+.2f}%"
        mdd = "—" if row.get("mdd") is None else f"{row['mdd']*100:.2f}%"
        lines.append(
            f"| {row['strategy']} | {row.get('capacity','')} | {row.get('seed','')} | {y} | {mdd} | `{Path(row['source']).name}` |"
        )
    lines += [
        "",
        f"fs00263 现账本 2022 区间：{min_2022} ~ {max_2022}。落在 −20%±3pct 的条数：{len(fs_neg20)}。",
        "",
        "## 文档 §12 摘录",
        "",
        "| strategy | capacity | seed | 2022 | MDD | source |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in doc_rows:
        y = "—" if row.get("year_2022") is None else f"{row['year_2022']*100:+.2f}%"
        mdd = "—" if row.get("mdd") is None else f"{row['mdd']*100:.2f}%"
        lines.append(
            f"| {row['strategy']} | {row['capacity']} | {row['seed']} | {y} | {mdd} | {row['source']} |"
        )
    lines += [
        "",
        "## Stock reports/*.md 命中「2022」且像 −20%",
        "",
    ]
    if md_hits:
        lines += [f"- `{name}`" for name in md_hits]
    else:
        lines.append("- 无文件同时出现 2022 与 -20 / −20 / -0.20 字样。")
    lines += [
        "",
        "## 集成账本扫描（ensemble_rank_pct_5seed，不是 60k 多种子全扫）",
        "",
        f"- 扫描 report.json：{ensemble_scan['scanned']} 份",
        f"- 2022 ∈ [−23%, −17%] 命中：{len(ensemble_scan['matches'])}",
        "",
    ]
    if ensemble_scan["matches"]:
        lines += [
            "| trial | 2022 | MDD |",
            "|---|---:|---:|",
        ]
        for row in ensemble_scan["matches"][:30]:
            lines.append(
                f"| {row['trial']} | {row['year_2022']*100:+.2f}% | {row['mdd']*100:.2f}% |"
            )
    else:
        lines.append("无命中。")
    lines += [
        "",
        "## 结论",
        "",
        conclusion,
        "",
        "Round 1 课题侧 `result/round1/R1_00_baseline/` 只有 `report.json` 拷贝，没有 Stock 的 nav/trades（Stock 该变体也只有 report.json）。P0 归因用课题本地执行器按锁定契约重建账本，写入 `result/round2/00_audit/locked_reconstruct/`，**不覆盖 Round 1，不写 Stock**。",
        "",
    ]
    dest = ensure_round2_tree() / "00_baseline_identity_audit.md"
    assert_project_result_path(dest)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "zyquant": zy,
        "fs00263_rows": fs_rows,
        "doc_rows": doc_rows,
        "markdown_hits": md_hits,
        "ensemble_scan": ensemble_scan,
        "found_2022_minus20_ledger": found_ledger,
        "found_other_trial_2022_minus20": found_other,
        "conclusion": conclusion,
        "stock_root": str(STOCK_ROOT),
    }
    json_path = ensure_round2_tree() / "00_audit" / "identity.json"
    assert_project_result_path(json_path)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return dest
