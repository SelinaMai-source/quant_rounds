"""PIT / information timeline。UNKNOWN 不得标 SAFE。不改策略。"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..paths import PROJECT_ROOT, ROUND11_ROOT, stock_path
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, TEN_FEATURES
from .common import dump_json, stage_dir, write_md
from .contract import BASELINES

PIT_MD = ROUND11_ROOT / "02_pit_information_timeline.md"


def _row(
    name: str,
    known_at: str,
    used_at: str,
    classification: str,
    note: str,
    status: str,
) -> dict[str, str]:
    if classification == "UNKNOWN":
        status = "UNQUANTIFIED" if status == "SAFE" else status
    return {
        "name": name,
        "known_at": known_at,
        "used_at": used_at,
        "classification": classification,
        "note": note,
        "status": status,
    }


def _label_maturity() -> dict[str, Any]:
    folds = []
    path = PROJECT_ROOT / "result/round7/models/five/seed17/training_report.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for fold in payload.get("folds") or []:
            folds.append({
                "name": fold.get("name") or fold.get("fold"),
                "train_end": fold.get("train_end") or fold.get("label_end") or fold.get("train_years"),
                "predict_year": fold.get("predict_year") or fold.get("test_year"),
                "raw": {k: fold.get(k) for k in list(fold)[:12]},
            })
    return {
        "walk_forward": "exact prior 3 calendar years train, next year predict",
        "target": "label_e2_net",
        "label_end_column": "next2_date",
        "rule_zh": "标签须在 train_end 年及以前成熟；预测年标签不得进训练折。",
        "folds_from_seed17": folds,
        "source": str(path) if path.is_file() else "MISSING",
        "status": "DOCUMENTED" if folds else "UNQUANTIFIED",
    }


def _survivorship() -> dict[str, Any]:
    panel = stock_path(
        "data/first_board_hermes/aligned_original_candidate_full44_plus_d1_open_gap_v1/candidate_panel.parquet"
    )
    note = (
        "候选宇宙来自历史 first-board 面板，不是全 A 股可投资全集。"
        "无法在本机证明已退市/从未上市股票的完整覆盖，故 survivorship = UNQUANTIFIED。"
    )
    return {
        "candidate_panel": str(panel),
        "exists": panel.is_file(),
        "status": "UNQUANTIFIED",
        "note": note,
    }


def build_timeline() -> list[dict[str, str]]:
    rows = [
        _row(
            "daily_raw OHLCV / paused / limit_up / limit_down",
            "D close (PIT snapshot cutoff)",
            "D+1 open execution + D+2.. sell",
            "MARKET_PIT",
            "ZyQuant DataSnapshot cutoff=as_of_date",
            "SAFE",
        ),
        _row(
            "first-board candidate universe",
            "D close after board event",
            "D score / D+1 trade",
            "EVENT_CONDITIONED",
            "只在当日首板候选内，不是全市场",
            "DOCUMENTED",
        ),
        _row(
            "d1_open_gap",
            "D+1 open auction / open print",
            "D close ranker + D+1 open filter/execution",
            "AUCTION_CONDITIONED_STRATEGY",
            "EXECUTION_TIME feature；不是 D-close 可部署量。回测按 D+1 开盘成交。",
            "AUCTION_CONDITIONED",
        ),
        _row(
            "five/ten ranker features except d1_open_gap",
            "D close or earlier (feature contract)",
            "D score",
            "FEATURE_PIT_ASSUMED",
            "训练报告称 point-in-time feature-complete；本轮未逐列重证 as-of",
            "DOCUMENTED",
        ),
        _row(
            "label_e2_net / next2_date",
            "label end = next2_date (D+2 close path)",
            "train fold only after maturity",
            "LABEL",
            "walk-forward：过去 3 个日历年训 → 下一年预测",
            "DOCUMENTED",
        ),
        _row(
            "B01a market breadth overlay",
            "D close market context",
            "D allocation multiplier",
            "FEATURE_PIT_ASSUMED",
            "市场层乘子，不改 fused score",
            "DOCUMENTED",
        ),
        _row(
            "top1-median / agreement / separation geometry",
            "D same-day cross-section of fused scores",
            "D Top2 multiplier",
            "SAME_DAY_CROSS_SECTION",
            "rolling250 PIT；几何在 fused raw 上算",
            "DOCUMENTED",
        ),
        _row(
            "industry L1 membership",
            "Hermes known_at if available",
            "Round10 pair (SKIP if not PIT)",
            "UNKNOWN",
            "Round10 已 SKIP 非 PIT 行业回填。本轮不重开 pair。",
            "UNQUANTIFIED",
        ),
        _row(
            "delisted / never-listed names",
            "UNKNOWN",
            "universe construction",
            "UNKNOWN",
            "证不了就 UNQUANTIFIED，不得标 SAFE",
            "UNQUANTIFIED",
        ),
    ]
    for feat in FIVE_FEATURES:
        cls = "AUCTION_CONDITIONED_STRATEGY" if feat == "d1_open_gap" else "FEATURE_PIT_ASSUMED"
        used = "D+1 open + D score" if feat == "d1_open_gap" else "D score"
        known = "D+1 open" if feat == "d1_open_gap" else "D close / earlier"
        status = "AUCTION_CONDITIONED" if feat == "d1_open_gap" else "DOCUMENTED"
        rows.append(_row(f"feature:{feat}", known, used, cls, "five-head contract", status))
    for feat in TEN_FEATURES:
        if feat == "d1_open_gap":
            continue
        rows.append(_row(f"ten:{feat}", "D close / earlier", "D score", "FEATURE_PIT_ASSUMED", "ten-head contract", "DOCUMENTED"))
    return rows


def run_pit(**_: Any) -> dict[str, Any]:
    dest = stage_dir("pit")
    rows = build_timeline()
    frame = pd.DataFrame(rows)
    csv_path = dest / "information_timeline.csv"
    frame.to_csv(csv_path, index=False)
    maturity = _label_maturity()
    surv = _survivorship()
    unknown_safe = [
        r for r in rows if r["classification"] == "UNKNOWN" and r["status"] == "SAFE"
    ]
    d1 = [r for r in rows if r["name"] in {"d1_open_gap", "feature:d1_open_gap"}]
    payload = {
        "status": "PASS" if not unknown_safe else "FAIL_UNKNOWN_MARKED_SAFE",
        "n_rows": int(len(rows)),
        "d1_open_gap": {
            "classification": "AUCTION_CONDITIONED_STRATEGY",
            "execution_time": "D+1 open",
            "tag": ["AUCTION_CONDITIONED_STRATEGY", "EXECUTION_TIME"],
            "rows": d1,
        },
        "label_maturity": maturity,
        "survivorship": surv,
        "unknown_marked_safe": unknown_safe,
        "five_panel": str(FIVE_HEAD.panel_dir),
        "baselines": {k: spec.code for k, spec in BASELINES.items()},
        "csv": str(csv_path),
        "post_hoc_diagnostic": True,
    }
    dump_json(dest / "pit.json", payload)
    lines = [
        "# Round 11 · PIT / information timeline",
        "",
        "UNKNOWN 不得标 SAFE。`d1_open_gap` = **AUCTION_CONDITIONED_STRATEGY** + **EXECUTION_TIME**。",
        "本文件是 freeze 后检验，标 `POST_HOC_DIAGNOSTIC`。不改策略。",
        "",
        f"- timeline 行数：{len(rows)}",
        f"- UNKNOWN 被标 SAFE：{len(unknown_safe)}（必须为 0）",
        f"- 标签成熟：`{maturity.get('status')}`；walk-forward = {maturity.get('walk_forward')}",
        f"- survivorship：`{surv.get('status')}`",
        "",
        "## d1_open_gap",
        "",
        "已知时点 = D+1 开盘（集合竞价/开盘价）。用于 ranker 特征与 ≥ -1% 预滤，回测按 D+1 开盘成交。",
        "数字不能写成纯 D-close 可部署结果。",
        "",
        "## 表",
        "",
        "| 名称 | 已知 | 使用 | 分类 | 状态 |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if r["name"].startswith("ten:"):
            continue
        lines.append(
            f"| `{r['name']}` | {r['known_at']} | {r['used_at']} | {r['classification']} | {r['status']} |"
        )
    lines += [
        "",
        "## 标签 vs fold",
        "",
        f"- target=`label_e2_net`，成熟列=`next2_date`",
        f"- 训练报告：`{maturity.get('source')}`",
        f"- 折数：{len(maturity.get('folds_from_seed17') or [])}",
        "",
        "## Survivorship",
        "",
        surv["note"],
        "",
        f"候选面板存在 = {surv.get('exists')}。",
        "",
    ]
    write_md(PIT_MD, "\n".join(lines))
    return payload


__all__ = ["run_pit"]
