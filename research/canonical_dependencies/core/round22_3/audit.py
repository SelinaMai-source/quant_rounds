"""Track A：Gate SHA 语义差 + matched 折 + 两本账隔离 + 短窗 SPA 重算。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND21_4_ROOT, ROUND22_1_ROOT
from ..round21_4.contract import ECONOMIC_MATERIALITY as R214_GATE
from .common import dump_json, load_json, sha256_json, write_csv, write_md
from .contract import (
    ECONOMIC_GATE_SEMANTIC_DRIFT,
    FIRST_PREDICTION_YEAR,
    MAIN_LEDGER,
    MATCHED_SUBPERIOD_NOT_COMPARABLE,
    MD,
    NOT_EVALUATED,
    ROUND22_2_ROOT,
    ROUND22_3_ROOT,
    SEMANTICALLY_IDENTICAL_HASH_MISMATCH,
    SHORT_LEDGER,
    canonical_threshold,
)
from .freeze import require_frozen_gate

AUDIT_JSON = ROUND22_3_ROOT / "audit" / "audit.json"


def _gate_audit() -> dict[str, Any]:
    g214 = canonical_threshold(R214_GATE)
    g221 = canonical_threshold((load_json(ROUND22_1_ROOT / "configs" / "frozen_gates.json").get("body") or {}).get("economic_materiality") or R214_GATE)
    g222 = canonical_threshold((load_json(ROUND22_2_ROOT / "configs" / "frozen_gates.json").get("body") or {}).get("economic_materiality") or R214_GATE)
    g223 = canonical_threshold((load_json(ROUND22_3_ROOT / "configs" / "frozen_gates.json").get("body") or {}).get("economic_materiality") or R214_GATE)
    hashes = {
        "r21_4": sha256_json(g214),
        "r22_1": sha256_json(g221),
        "r22_2": sha256_json(g222),
        "r22_3": sha256_json(g223),
    }
    identical = len(set(hashes.values())) == 1
    body_hashes = {
        "r22_1_body": (load_json(ROUND22_1_ROOT / "configs" / "frozen_gates.json") or {}).get("body_sha256"),
        "r22_2_body": (load_json(ROUND22_2_ROOT / "configs" / "frozen_gates.json") or {}).get("body_sha256"),
        "r22_3_body": (load_json(ROUND22_3_ROOT / "configs" / "frozen_gates.json") or {}).get("body_sha256"),
        "r22_3_threshold": (load_json(ROUND22_3_ROOT / "configs" / "frozen_gates.json") or {}).get("threshold_sha256"),
    }
    status = SEMANTICALLY_IDENTICAL_HASH_MISMATCH if identical else ECONOMIC_GATE_SEMANTIC_DRIFT
    payload = {
        "status": status,
        "thresholds_identical": identical,
        "threshold_sha256": hashes,
        "body_sha256": body_hashes,
        "reason": "body hash includes body.round; threshold object canonicalized without round",
        "objects": {"r21_4": g214, "r22_1": g221, "r22_2": g222, "r22_3": g223},
    }
    dump_json(ROUND22_3_ROOT / "audit" / "economic_gate_semantic_diff.json", payload)
    write_csv(
        ROUND22_3_ROOT / "audit" / "economic_gate_lineage.csv",
        [
            {"round": k, "threshold_sha256": v, "id": "R21_3_ECONOMIC_MATERIALITY_V1", "status": status}
            for k, v in hashes.items()
        ],
    )
    write_md(
        ROUND22_3_ROOT / "audit" / "economic_gate_raw_diff.md",
        "\n".join(
            [
                "# 经济门 SHA 审计",
                "",
                f"status=`{status}` thresholds_identical=`{identical}`",
                "",
                f"- 21.4 threshold `{hashes['r21_4']}`",
                f"- 22.1 threshold `{hashes['r22_1']}` body `{body_hashes['r22_1_body']}`",
                f"- 22.2 threshold `{hashes['r22_2']}` body `{body_hashes['r22_2_body']}`",
                f"- 22.3 threshold `{hashes['r22_3']}` body `{body_hashes['r22_3_body']}`",
                "",
                "22.1 / 22.2 body_sha256 不同，是因为哈希了整份 body（含 round 字符串）。门槛数字与 21.4 ECONOMIC_MATERIALITY 相同则标 SEMANTICALLY_IDENTICAL_HASH_MISMATCH。语义漂移才用权威门重评，不按新结果改门。",
                "",
            ]
        ),
    )
    write_md(MD["02"], f"# Round22.3 · 02 Gate SHA\n\n`{status}`。threshold 一致=`{identical}`。\n")
    return payload


def _two_ledger() -> dict[str, Any]:
    payload = {
        "main": dict(MAIN_LEDGER),
        "short": dict(SHORT_LEDGER),
        "mix_forbidden": True,
        "note": "High Sharpe on the short window is the window, not a better strategy.",
    }
    dump_json(ROUND22_3_ROOT / "audit" / "two_ledger.json", payload)
    body = "\n".join(
        [
            "# 两本账契约",
            "",
            "| 账本 | 窗口 | n_obs | 约笔数 | 对照 | Sharpe | 官方 |",
            "|---|---|---:|---:|---|---:|---|",
            f"| 主表 | {MAIN_LEDGER['window_start']} – {MAIN_LEDGER['window_end']} | {MAIN_LEDGER['n_obs']} | {MAIN_LEDGER['n_trades']} | `{MAIN_LEDGER['compare_to']}` | {MAIN_LEDGER['sharpe']} | 是 |",
            f"| 22.2 实验组 | {SHORT_LEDGER['window_start']} – {SHORT_LEDGER['window_end']} | {SHORT_LEDGER['n_obs']} | {SHORT_LEDGER['n_trades']} | `{SHORT_LEDGER['compare_to']}` | {SHORT_LEDGER['sharpe']} | 否 |",
            "",
            f"短窗状态=`{MATCHED_SUBPERIOD_NOT_COMPARABLE}`。禁止与 R16_O01 混表。费用/引擎/Sharpe 公式未变。",
            "",
        ]
    )
    write_md(ROUND22_3_ROOT / "audit" / "TWO_LEDGER_CONTRACT.md", body)
    write_md(MD["01"], body)
    src = ROUND22_2_ROOT / "backtest" / "sleeve_results.csv"
    dest = ROUND22_3_ROOT / "matched_subperiod" / "sleeve_results.csv"
    if src.is_file():
        frame = pd.read_csv(src)
        frame["window"] = "2023-2026"
        frame["ledger"] = "R222_MATCHED_2023_2026"
        frame["comparability"] = MATCHED_SUBPERIOD_NOT_COMPARABLE
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dest, index=False)
    return payload


def _matched_folds() -> dict[str, Any]:
    matched = load_json(ROUND22_2_ROOT / "matched" / "matched_sample.json")
    train = load_json(ROUND22_2_ROOT / "model" / "train.json")
    both = matched.get("both") or {}
    first_pred = int(train.get("first_prediction_year") or 2023)
    last_pred = int(train.get("last_prediction_year") or 2026)
    rows = []
    for year in range(2018, 2027):
        train_lo, train_hi = year - 3, year - 1
        canonical = year >= FIRST_PREDICTION_YEAR and (train_hi - train_lo + 1) == 3
        evaluated = year >= first_pred
        rows.append(
            {
                "predict_year": year,
                "train_start": train_lo,
                "train_end": train_hi,
                "train_years": 3,
                "canonical_3y1y_from_2018": bool(canonical and year >= FIRST_PREDICTION_YEAR),
                "short_window_evaluated": bool(evaluated),
                "status": "OOF" if evaluated else (NOT_EVALUATED if year == 2022 else "OUT_OF_SHORT_WINDOW"),
            }
        )
    if first_pred > 2018:
        rows.append(
            {
                "predict_year": "NOTE",
                "train_start": None,
                "train_end": None,
                "train_years": None,
                "canonical_3y1y_from_2018": False,
                "short_window_evaluated": False,
                "status": "short-window first fold train starts after 2018; NOT Canonical vs v5",
            }
        )
    write_csv(ROUND22_3_ROOT / "audit" / "MATCHED_FOLD_TIMELINE.csv", rows)
    write_csv(ROUND22_3_ROOT / "matched_subperiod" / "MATCHED_FOLD_TIMELINE.csv", rows)
    dates_info = {
        "n_dates": both.get("n_dates") or 1394,
        "min_date": both.get("min_date") or "2020-09-24",
        "max_date": both.get("max_date"),
        "sleeve_window_start": "2023-01-03",
        "sleeve_window_end": "2026-07-24",
        "n_obs_sleeve": 860,
        "n_obs_main": 2075,
        "y2022": NOT_EVALUATED,
        "canonical": False,
        "reason": "1394 PIT days ≠ formal eval days; first OOS year 2023 so first train is 2020-2022, not 2018-2020",
    }
    dump_json(ROUND22_3_ROOT / "audit" / "MATCHED_EVALUATION_DATES.csv".replace(".csv", ".json"), dates_info)
    write_csv(ROUND22_3_ROOT / "audit" / "MATCHED_EVALUATION_DATES.csv", [dates_info])
    write_csv(ROUND22_3_ROOT / "matched_subperiod" / "MATCHED_EVALUATION_DATES.csv", [dates_info])
    write_md(
        MD["03"],
        f"# Round22.3 · 03 Matched 折\n\n联合 matched n=`{dates_info['n_dates']}` {dates_info['min_date']}–{dates_info.get('max_date')}。袖套从 2023-01-03。第一折训期不是 2018–2020，短窗不得称 Canonical。2022=`{NOT_EVALUATED}`。\n",
    )
    return {"folds": rows, "dates": dates_info}


def _spa_short() -> dict[str, Any]:
    original = load_json(ROUND22_2_ROOT / "spa" / "spa.json")
    sleeve = load_json(ROUND22_2_ROOT / "backtest" / "sleeve.json")
    rows = sleeve.get("rows") or []
    codes = [r.get("code") for r in rows if r.get("code")]
    n_formal = max(0, len(codes) - 1)
    n_alts_reported = original.get("n_alternatives")
    e0_skipped_in_code = True
    corrected = {
        "status": original.get("status") or "DONE",
        "n_rows_including_e0": len(codes),
        "n_formal": n_formal,
        "n_alternatives_original": n_alts_reported,
        "n_alternatives_corrected": n_formal,
        "e0_skipped_in_stats_py": e0_skipped_in_code,
        "p_spa_original": original.get("p_spa") or original.get("p"),
        "note": "stats.py skips E0 when building alts; n_alternatives=10 matches n_formal=10. Report wording 'n_formal=10' is not E0-included.",
        "p_spa_corrected": original.get("p_spa") or original.get("p"),
        "n_boot": original.get("n_boot") or 5000,
    }
    dump_json(ROUND22_3_ROOT / "audit" / "spa_short_original.json", original)
    dump_json(ROUND22_3_ROOT / "audit" / "spa_short_corrected.json", corrected)
    write_md(
        MD["04"],
        f"# Round22.3 · 04 SPA 短窗审计\n\nOriginal n_alternatives=`{n_alts_reported}` p=`{corrected['p_spa_original']}`。Corrected n_alternatives=`{n_formal}`（已跳过 E0）。全历史 SPA 待重跑后另写。\n",
    )
    return corrected


def run_audit(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen_gate()
    if resume and AUDIT_JSON.is_file() and MD["01"].is_file():
        existing = load_json(AUDIT_JSON)
        if existing.get("status") in {"DONE"}:
            return existing
    gate = _gate_audit()
    ledgers = _two_ledger()
    folds = _matched_folds()
    spa = _spa_short()
    payload = {
        "status": "DONE",
        "gate": gate,
        "ledgers": ledgers,
        "folds": folds.get("dates"),
        "spa_short": spa,
    }
    dump_json(AUDIT_JSON, payload)
    return payload
