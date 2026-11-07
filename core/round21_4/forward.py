"""Track B：修复 _max_date 全分区扫描；Forward 覆盖矩阵。

21.3 bug：sorted(rglob)[:5] 再取 files[-1]，Hermes daily_raw 被锁在 2010-05-31。
本轮扫 ALL parquet 分区，按日期聚合 max，不只 5 片。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from ..paths import MARKET_END, ROUND21_4_ROOT
from ..round21.contract import HERMES_REBUILD_RAW, HERMES_V7
from .common import dump_json, load_json, write_md
from .contract import (
    BLOCKED_PERMISSION,
    DATA_LIVE_RELEASES,
    FORWARD_OOS_NOT_AVAILABLE,
    INCOMPLETE,
    MD,
    PREVIOUSLY_INSPECTED,
    UNSEEN_FORWARD_EVALUATION,
)

FWD_JSON = ROUND21_4_ROOT / "forward" / "forward.json"
FIX_JSON = ROUND21_4_ROOT / "forward" / "maxdate_fix.json"
COV_CSV = ROUND21_4_ROOT / "FORWARD_DATA_COVERAGE_MATRIX.csv"
COV_JSON = ROUND21_4_ROOT / "coverage" / "coverage.json"

_DATE_COLS = ("trade_date", "date", "datetime", "available_at", "effective_from", "asof_date", "ann_date", "end_date")


def _max_date(path: Path, col_candidates: tuple[str, ...] = _DATE_COLS, *, sample_files: int | None = None) -> dict[str, Any]:
    """扫描全部 parquet 分区，按日期列取全局 max。sample_files=None 表示全扫。"""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False, "max_date": None, "n_files": 0}
    try:
        if path.is_file():
            files = [path]
        else:
            files = sorted(path.rglob("*.parquet"))
        if not files:
            return {"path": str(path), "exists": True, "max_date": None, "n_files": 0, "note": "empty dir"}
        n_all = len(files)
        if sample_files is not None:
            files = files[: int(sample_files)]
        mx = None
        col_used = None
        n_ok = 0
        errors = 0
        for fp in files:
            try:
                schema = set(pq.read_schema(fp).names)
                col = next((c for c in col_candidates if c in schema), None)
                if col is None:
                    continue
                s = pd.read_parquet(fp, columns=[col])[col]
                cur = pd.to_datetime(s, errors="coerce").max()
                if pd.isna(cur):
                    continue
                n_ok += 1
                col_used = col_used or col
                if mx is None or cur > mx:
                    mx = cur
            except Exception:
                errors += 1
                continue
        return {
            "path": str(path),
            "exists": True,
            "max_date": None if mx is None else str(pd.Timestamp(mx).date()),
            "col": col_used,
            "n_files": n_all,
            "n_scanned": len(files),
            "n_ok": n_ok,
            "errors": errors,
            "sample_files": sample_files,
        }
    except Exception as exc:
        return {"path": str(path), "exists": True, "error": str(exc), "max_date": None}


def _buggy_max_date(path: Path, col_candidates: tuple[str, ...] = ("trade_date", "date", "datetime")) -> dict[str, Any]:
    """21.3 原实现：只读 sorted()[:5] 的最后一片。用于对照。"""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False, "max_date": None}
    try:
        if path.is_dir():
            files = sorted(path.rglob("*.parquet"))[:5]
            if not files:
                return {"path": str(path), "exists": True, "max_date": None, "note": "empty dir"}
            path = files[-1]
        schema = set(pq.read_schema(path).names)
        col = next((c for c in col_candidates if c in schema), None)
        if col is None:
            return {"path": str(path), "exists": True, "max_date": None, "cols": list(schema)[:12]}
        s = pd.read_parquet(path, columns=[col])[col]
        mx = pd.to_datetime(s, errors="coerce").max()
        return {
            "path": str(path),
            "exists": True,
            "max_date": None if pd.isna(mx) else str(pd.Timestamp(mx).date()),
            "col": col,
            "bug": "sorted()[:5] then files[-1]",
        }
    except Exception as exc:
        return {"path": str(path), "exists": True, "error": str(exc)}


def run_forward_fix(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and FIX_JSON.is_file() and MD["02"].is_file():
        existing = load_json(FIX_JSON)
        if existing.get("status") == "DONE":
            return existing
    targets = {
        "eod_daily_raw": HERMES_V7 / "daily_raw",
        "money_flow": HERMES_V7 / "daily_money_flow",
        "share_capital": HERMES_V7 / "share_capital",
        "adj_factor": HERMES_V7 / "adj_factor" if (HERMES_V7 / "adj_factor").exists() else HERMES_V7 / "daily_adj",
    }
    rows = []
    for name, path in targets.items():
        cols = ("available_at", "effective_from", "trade_date", "date") if "share" in name else _DATE_COLS
        buggy = _buggy_max_date(path, cols[:3] if name != "share_capital" else ("available_at", "effective_from", "trade_date"))
        fixed = _max_date(path, cols)
        rows.append({"field": name, "buggy": buggy, "fixed": fixed, "path": str(path)})
        print(f"[round21.4] maxdate {name} buggy={buggy.get('max_date')} fixed={fixed.get('max_date')} n_files={fixed.get('n_files')}", flush=True)
    payload = {
        "status": "DONE",
        "bug": "sorted(rglob)[:5][-1] locks daily_raw at first-5-shards last file",
        "fix": "scan ALL parquet partitions and take global max date",
        "rows": rows,
        "daily_raw_buggy": next((r["buggy"].get("max_date") for r in rows if r["field"] == "eod_daily_raw"), None),
        "daily_raw_fixed": next((r["fixed"].get("max_date") for r in rows if r["field"] == "eod_daily_raw"), None),
        "money_flow_buggy": next((r["buggy"].get("max_date") for r in rows if r["field"] == "money_flow"), None),
        "money_flow_fixed": next((r["fixed"].get("max_date") for r in rows if r["field"] == "money_flow"), None),
    }
    dump_json(FIX_JSON, payload)
    write_md(
        MD["02"],
        "\n".join(
            [
                "# Round21.4 · 02 Forward `_max_date` 修复",
                "",
                "21.3 `forward.py` `_max_date` 对目录 `sorted(rglob('*.parquet'))[:5]` 再取 `files[-1]`。",
                "Hermes v7 `daily_raw` 分区按时间升序，前 5 片落在 2010 年初，所以 max 被锁在 `2010-05-31`；`money_flow` 同理可能 `null`。",
                "",
                f"- daily_raw buggy=`{payload['daily_raw_buggy']}` fixed=`{payload['daily_raw_fixed']}`",
                f"- money_flow buggy=`{payload['money_flow_buggy']}` fixed=`{payload['money_flow_fixed']}`",
                "",
                "本轮扫全部分区。这是扫描 bug，不是数据真到 2010。",
                "",
            ]
        ),
    )
    return payload


def _try_listdir(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {"path": str(path), "exists": False, "entries": [], "status": "MISSING"}
        entries = sorted(p.name for p in path.iterdir())
        return {"path": str(path), "exists": True, "n": len(entries), "entries": entries[:40], "status": "OK"}
    except PermissionError as exc:
        return {"path": str(path), "exists": True, "status": BLOCKED_PERMISSION, "error": str(exc)}
    except Exception as exc:
        return {"path": str(path), "exists": True, "status": "ERROR", "error": str(exc)}


def run_coverage(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and COV_JSON.is_file() and COV_CSV.is_file() and MD["03"].is_file():
        existing = load_json(COV_JSON)
        if existing.get("status") in {"DONE", "PARTIAL"}:
            return existing
    run_forward_fix(resume=True)
    cutoff = str(MARKET_END)
    need_after = "2026-07-25"
    checks = {
        "eod_daily_raw": _max_date(HERMES_V7 / "daily_raw"),
        "adj_factor": _max_date(HERMES_V7 / "adj_factor") if (HERMES_V7 / "adj_factor").exists() else _max_date(HERMES_V7 / "daily_adj"),
        "limit_up": _max_date(HERMES_V7 / "daily_raw", ("limit_up", "trade_date")),
        "candidate_panel": _max_date(Path("/data1/Stock/data/first_board_hermes")),
        "minute": _max_date(Path("/data1/Stock/data/factor_cache")),
        "money_flow": _max_date(HERMES_V7 / "daily_money_flow"),
        "share_capital": _max_date(HERMES_V7 / "share_capital", ("available_at", "effective_from", "trade_date")),
        "announcement": _max_date(HERMES_V7 / "announcement") if (HERMES_V7 / "announcement").exists() else {"path": str(HERMES_V7 / "announcement"), "exists": False, "max_date": None},
        "o01": {"note": "O01 overlay uses cached v4 through MARKET_END", "market_end": cutoff, "max_date": cutoff},
        "exit_labels": _max_date(Path("/data1/Stock/data/first_board_hermes")),
    }
    live = _try_listdir(DATA_LIVE_RELEASES)
    live_dates = []
    if live.get("status") == "OK":
        for name in live.get("entries") or []:
            digits = "".join(ch for ch in name if ch.isdigit())
            if len(digits) >= 6:
                live_dates.append(name)
    matrix = []
    missing = []
    for field, v in checks.items():
        mx = v.get("max_date") if isinstance(v, dict) else None
        after = bool(mx and str(mx) >= need_after)
        cls = PREVIOUSLY_INSPECTED if mx and str(mx) <= cutoff else (UNSEEN_FORWARD_EVALUATION if after else INCOMPLETE)
        if mx is None or str(mx) <= cutoff:
            missing.append({"field": field, "max_date": mx, "need_after": need_after})
            if mx is None:
                cls = INCOMPLETE
        matrix.append(
            {
                "field": field,
                "max_date": mx,
                "cutoff": cutoff,
                "need_after": need_after,
                "covers_after_cutoff": after,
                "n_files": v.get("n_files") if isinstance(v, dict) else None,
                "exists": v.get("exists") if isinstance(v, dict) else True,
                "classification": cls,
                "path": v.get("path") if isinstance(v, dict) else None,
            }
        )
    independent = all(r.get("covers_after_cutoff") for r in matrix if r["field"] in {"eod_daily_raw", "candidate_panel", "exit_labels"})
    status = "AVAILABLE" if independent else FORWARD_OOS_NOT_AVAILABLE
    payload = {
        "status": status,
        "cutoff": cutoff,
        "need_after": need_after,
        "checks": checks,
        "matrix": matrix,
        "missing": missing,
        "data_live_releases": live,
        "live_batch_names": live_dates,
        "independent_forward": bool(independent),
        "classification": UNSEEN_FORWARD_EVALUATION if independent else (INCOMPLETE if missing else PREVIOUSLY_INSPECTED),
        "note": "只有同日同池同引擎同标签成熟才算独立 Forward。覆盖不足不得 live-ready。",
        "rebuild_raw": str(HERMES_REBUILD_RAW),
        "hermes_v7": str(HERMES_V7),
    }
    dump_json(COV_JSON, payload)
    dump_json(FWD_JSON, payload)
    pd.DataFrame(matrix).to_csv(COV_CSV, index=False)
    write_md(
        MD["03"],
        "\n".join(
            [
                "# Round21.4 · 03 Forward 覆盖矩阵",
                "",
                f"status=`{status}` cutoff=`{cutoff}` independent=`{independent}` classification=`{payload['classification']}`",
                f"data_live/releases status=`{live.get('status')}` n=`{live.get('n')}`",
                f"缺失=`{missing}`",
                "",
                "字段：EOD / 复权 / 涨停 / 候选 / 分钟 / 资金流 / 股本 / 公告 / 特征 / O01 / 退出标签。",
                "",
            ]
        ),
    )
    return payload


def run_forward(*, resume: bool = True, **kwargs: Any) -> dict[str, Any]:
    return run_coverage(resume=resume, **kwargs)
