"""Track C：HN / BT / MG / FS。去重 Round21，PIT/恢复，少量冻结公式。新数据只写 result/round22_2/data_recovery/。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..round21.contract import BLOCK_TRADING_PARQUET, HERMES_REBUILD_RAW, HERMES_V7, HOLDER_NUM_PARQUET
from ..round21.panel import (
    KEYS,
    asof_by_instrument,
    load_candidate_keys,
    load_hermes_table,
    load_trade_calendar,
    market_to_instrument,
    next_trading_day,
    parse_yyyymmdd,
)
from .common import dump_json, load_json, write_csv, write_md
from .contract import (
    BLOCKED_DATA,
    BLOCKED_PIT,
    DUPLICATE,
    MD,
    NEW_FACTOR_SPECS,
    PIT_MIN_FINITE,
    PIT_PASS,
    ROUND22_2_ROOT,
)
from .freeze import require_frozen_gate

REC = ROUND22_2_ROOT / "data_recovery"
DISC_JSON = REC / "discover.json"
STALE_DAYS = 365 * 2


def _finite(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _search_root(root: Path, needles: tuple[str, ...]) -> dict[str, Any]:
    if not root.exists():
        return {"path": str(root), "exists": False, "hits": []}
    hits = []
    try:
        for p in root.iterdir():
            low = p.name.lower()
            if any(n in low for n in needles):
                hits.append(p.name)
    except PermissionError:
        return {"path": str(root), "exists": True, "status": "BLOCKED_ACCESS", "hits": []}
    return {"path": str(root), "exists": True, "hits": hits}


def _hn() -> dict[str, Any]:
    out: dict[str, Any] = {
        "holder_path": str(HOLDER_NUM_PARQUET),
        "exists": HOLDER_NUM_PARQUET.is_file(),
        "hn1": {
            "factor_id": "r22_2_hn1_neg_log_holder_ratio",
            "status": DUPLICATE,
            "lineage": "r21_d1_log_holder_ratio",
            "formula": "-log(HOLDER_NUM_t / HOLDER_NUM_{t-1})",
            "note": "HN1 ≈ −r21_d1。已 PIT_PASS、弱、DROP_WEAK。不当新 Alpha。",
            "values_constructed": False,
            "model_eligible": False,
        },
        "hn2": {"factor_id": "r22_2_hn2_avg_holding_chg", "status": BLOCKED_DATA},
    }
    if not HOLDER_NUM_PARQUET.is_file():
        out["hn2"]["reason"] = "holder_num/all.parquet 不存在"
        return out
    raw = pd.read_parquet(HOLDER_NUM_PARQUET)
    cols = list(raw.columns)
    out["holder_columns"] = cols
    if "ANN_DT" not in raw.columns or "HOLDER_NUM" not in raw.columns:
        out["hn2"]["status"] = BLOCKED_PIT
        out["hn2"]["reason"] = f"缺 ANN_DT/HOLDER_NUM 列={cols}"
        return out
    calendar = load_trade_calendar()
    keys = load_candidate_keys()[KEYS].drop_duplicates()
    keys["trade_date"] = pd.to_datetime(keys["trade_date"])
    keys["instrument_id"] = keys["instrument_id"].astype(str)
    frame = pd.DataFrame(
        {
            "instrument_id": raw["MARKET_CODE"].map(market_to_instrument),
            "ann_dt": parse_yyyymmdd(raw["ANN_DT"]),
            "holder_num": _finite(raw["HOLDER_NUM"]),
        }
    )
    frame = frame.dropna(subset=["instrument_id", "ann_dt"])
    frame = frame.loc[frame["holder_num"].gt(0)].copy()
    frame = frame.sort_values(["instrument_id", "ann_dt"]).drop_duplicates(["instrument_id", "ann_dt"], keep="last")
    g = frame.groupby("instrument_id", sort=False)
    prev_h = g["holder_num"].shift(1)
    prev_ann = g["ann_dt"].shift(1)
    # 不默认 15:00 已知。ANN_DT 次一交易日才可得。
    frame["known_at"] = next_trading_day(frame["ann_dt"], calendar)
    days = (frame["ann_dt"] - prev_ann).dt.days
    frame["stale"] = days.gt(STALE_DAYS)
    ids = keys["instrument_id"].unique()
    val = load_hermes_table(
        "daily_valuation",
        KEYS + ["circulating_shares", "available_at"],
        instruments=ids,
    )
    val["circulating_shares"] = _finite(val["circulating_shares"])
    val["available_at"] = pd.to_datetime(val["available_at"], errors="coerce")
    work = asof_by_instrument(
        keys,
        frame,
        left_on="trade_date",
        right_on="known_at",
        columns=["holder_num", "known_at", "ann_dt", "stale"],
    )
    work = asof_by_instrument(
        work,
        val,
        left_on="trade_date",
        right_on="available_at",
        columns=["circulating_shares", "available_at"],
    )
    circ = _finite(work["circulating_shares"])
    hold = _finite(work["holder_num"])
    avg = pd.Series(np.nan, index=work.index, dtype="float64")
    ok = hold.gt(0) & circ.gt(0)
    avg.loc[ok] = circ.loc[ok] / hold.loc[ok]
    work = work.sort_values(["instrument_id", "trade_date"])
    work["_avg"] = avg
    prev_avg = work.groupby("instrument_id", sort=False)["_avg"].shift(1)
    hn2 = pd.Series(np.nan, index=work.index, dtype="float64")
    both = avg.gt(0) & prev_avg.gt(0)
    hn2.loc[both] = np.log(avg.loc[both] / prev_avg.loc[both])
    # 多年未更新标 stale，不填 0
    stale = work["stale"].fillna(False).astype(bool)
    hn2.loc[stale] = np.nan
    work["r22_2_hn2_avg_holding_chg"] = hn2
    dest = REC / "hn2_panel.parquet"
    keep = KEYS + ["r22_2_hn2_avg_holding_chg", "ann_dt", "known_at", "available_at", "stale"]
    work[keep].to_parquet(dest, index=False)
    finite = float(pd.to_numeric(hn2, errors="coerce").notna().mean()) if len(hn2) else 0.0
    pit = PIT_PASS if finite >= PIT_MIN_FINITE else "PIT_FAIL_COVERAGE"
    out["hn2"].update(
        {
            "status": pit,
            "formula": "Δlog(then_known_circ_shares / HOLDER_NUM); known_at=ANN_DT next trading day; float.available_at",
            "values_constructed": True,
            "finite_share": finite,
            "n": int(len(work)),
            "n_stale_nan": int(stale.sum()),
            "filled_with_zero": False,
            "assume_1500_known": False,
            "panel": str(dest),
            "model_eligible": pit == PIT_PASS,
            "lineage": "new vs r21_d1/d3",
        }
    )
    dump_json(ROUND22_2_ROOT / "factors" / "r22_2_hn2_avg_holding_chg" / "pit.json", out["hn2"])
    return out


def _bt() -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(BLOCK_TRADING_PARQUET),
        "exists": BLOCK_TRADING_PARQUET.is_file(),
        "factor_id": "r22_2_bt1_block_discount",
        "lineage": "r21_e1_block_discount",
        "status": BLOCKED_PIT,
        "model_eligible": False,
        "filled_no_event_as_zero": False,
    }
    if not BLOCK_TRADING_PARQUET.is_file():
        out["reason"] = "block_trading/all.parquet 不存在"
        return out
    sample = pd.read_parquet(BLOCK_TRADING_PARQUET)
    cols = list(sample.columns)
    out["columns"] = cols
    time_cols = [c for c in cols if any(k in c.upper() for k in ("ANN", "PUBLISH", "AVAILABLE", "TIME", "NOTICE", "DECLARE"))]
    # TRADE_DATE 不算公告时点
    time_cols = [c for c in time_cols if c.upper() not in {"TRADE_DATE"}]
    out["candidate_timestamp_cols"] = time_cols
    if not time_cols:
        out["status"] = BLOCKED_PIT
        out["reason"] = "大宗只有 TRADE_DATE，未恢复公告时点。人为延迟不称 PIT-safe。覆盖未证实时「无大宗」不得填 0。"
        out["duplicate"] = DUPLICATE
        dump_json(ROUND22_2_ROOT / "factors" / "r22_2_bt1_block_discount" / "pit.json", out)
        write_csv(REC / "bt_blocked.csv", [{"factor_id": "r22_2_bt1_block_discount", "status": BLOCKED_PIT}])
        return out
    out["status"] = "TIMESTAMP_CANDIDATE"
    out["reason"] = f"发现可疑时点列 {time_cols}，需证明是公告时点才进模型。本轮不默认 PIT_PASS。"
    out["model_eligible"] = False
    dump_json(ROUND22_2_ROOT / "factors" / "r22_2_bt1_block_discount" / "pit.json", out)
    return out


def _mg() -> dict[str, Any]:
    out: dict[str, Any] = {
        "factor_id": "r22_2_mg1_fin_buy_intensity",
        "status": DUPLICATE,
        "lineage": "r21_f3_fin_buy_intensity",
        "pit_legacy": "PIT_FAIL_COVERAGE",
        "formula": "financing_buy_value / amount with daily_margin.available_at ≤ cutoff",
        "changed_v5_universe": False,
        "filled_non_margin_zero": False,
        "model_eligible": False,
        "note": "MG1 近 r21_f3。标 DUPLICATE，不当新 Alpha。非标的 / 缺值 / 合法零 三分开。",
    }
    try:
        keys = load_candidate_keys()[KEYS].drop_duplicates()
        ids = keys["instrument_id"].unique()
        margin = load_hermes_table(
            "daily_margin",
            KEYS + ["financing_buy_value", "available_at"],
            instruments=ids,
        )
        out["n_margin_rows"] = int(len(margin))
        out["available_at_present"] = "available_at" in margin.columns
        out["coverage_vs_keys"] = float(len(margin) / max(len(keys), 1))
        out["non_margin_stays_null"] = True
    except Exception as exc:
        out["audit_error"] = type(exc).__name__
    dump_json(ROUND22_2_ROOT / "factors" / "r22_2_mg1_fin_buy_intensity" / "pit.json", out)
    return out


def _fs() -> dict[str, Any]:
    needles = ("fdmt", "income", "cash", "cashflow", "profit", "financial", "fund")
    searches = {
        "hermes_v7": _search_root(HERMES_V7, needles),
        "hermes_raw": _search_root(HERMES_REBUILD_RAW, needles),
    }
    out: dict[str, Any] = {
        "factor_id": "r22_2_fs1_accrual_minus_ocf",
        "priority": "below_C_and_HN",
        "search": searches,
        "status": BLOCKED_DATA,
        "model_eligible": False,
        "values_constructed": False,
        "filled_with_zero": False,
    }
    hits = []
    for block in searches.values():
        hits.extend(block.get("hits") or [])
    out["hits"] = hits
    # 需要利润与经营现金流同报告期，且有 published_at/available_at
    income_names = [n for n in hits if any(k in n.lower() for k in ("income", "profit", "is_"))]
    cash_names = [n for n in hits if "cash" in n.lower()]
    if not income_names or not cash_names:
        out["reason"] = (
            "未同时找到带 published_at/available_at 的利润表与经营现金流表。"
            f" hits={hits[:20]}。不猜列、不填 0。优先级低于 C 与 HN。"
        )
        dump_json(ROUND22_2_ROOT / "factors" / "r22_2_fs1_accrual_minus_ocf" / "pit.json", out)
        return out
    # 再核一列：必须有时点
    picked = None
    for name in income_names + cash_names:
        for root in (HERMES_V7, HERMES_REBUILD_RAW):
            p = root / name
            if not p.exists():
                continue
            sample = next(p.rglob("*.parquet"), None) if p.is_dir() else (p if p.suffix == ".parquet" else None)
            if sample is None:
                continue
            try:
                cols = list(pq.read_schema(sample).names)
            except Exception:
                continue
            time_ok = any(c.upper() in {"PUBLISHED_AT", "AVAILABLE_AT", "ANN_DT", "PUBLISH_DATE"} for c in cols)
            if time_ok:
                picked = {"table": name, "columns": cols, "path": str(sample)}
                break
        if picked:
            break
    if picked is None:
        out["status"] = BLOCKED_PIT
        out["reason"] = "命中财务表名但没有 published_at/available_at/ANN_DT，不得 PIT_PASS。"
        dump_json(ROUND22_2_ROOT / "factors" / "r22_2_fs1_accrual_minus_ocf" / "pit.json", out)
        return out
    out["status"] = BLOCKED_PIT
    out["reason"] = (
        f"发现 {picked['table']} 且含时点列，但未证明利润与经营现金流同报告期同口径可对齐。"
        "不在未验证口径上硬造应计。优先级低于 C 与 HN。"
    )
    out["picked"] = picked
    dump_json(ROUND22_2_ROOT / "factors" / "r22_2_fs1_accrual_minus_ocf" / "pit.json", out)
    return out


def run_discover(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen_gate()
    if resume and DISC_JSON.is_file() and MD["08"].is_file():
        existing = load_json(DISC_JSON)
        if existing.get("status") in {"DONE", "PARTIAL"}:
            return existing
    REC.mkdir(parents=True, exist_ok=True)
    hn = _hn()
    dump_json(REC / "hn.json", hn)
    bt = _bt()
    dump_json(REC / "bt.json", bt)
    mg = _mg()
    dump_json(REC / "mg.json", mg)
    fs = _fs()
    dump_json(REC / "fs.json", fs)
    rows = [
        {"factor_id": spec["factor_id"], "family": spec["family"], "priority": spec["priority"]}
        for spec in NEW_FACTOR_SPECS
    ]
    rows[0]["status"] = hn["hn1"]["status"]
    rows[1]["status"] = hn["hn2"]["status"]
    rows[2]["status"] = bt["status"]
    rows[3]["status"] = mg["status"]
    rows[4]["status"] = fs["status"]
    write_csv(REC / "new_factor_status.csv", rows)
    n_new = int(hn["hn2"].get("values_constructed") is True and hn["hn2"].get("status") == PIT_PASS)
    payload = {
        "status": "PARTIAL" if n_new == 0 or bt["status"] == BLOCKED_PIT or fs["status"].startswith("BLOCKED") else "DONE",
        "write_root": str(REC),
        "overwrote_hermes_v7": False,
        "overwrote_round22": False,
        "overwrote_round22_1": False,
        "hn": hn,
        "bt": bt,
        "mg": mg,
        "fs": fs,
        "n_duplicate": 2 + (1 if bt.get("duplicate") == DUPLICATE else 0),
        "n_new_constructed": n_new,
        "today_download_is_not_2018_available": True,
    }
    dump_json(DISC_JSON, payload)
    write_md(MD["08"], f"# Round22.2 · 08 HN\n\nHN1 `{hn['hn1']['status']}` {hn['hn1']['note']}\n\nHN2 `{hn['hn2'].get('status')}` finite=`{hn['hn2'].get('finite_share')}` stale_nan=`{hn['hn2'].get('n_stale_nan')}` 未填 0。ANN_DT 次一交易日，不默认 15:00。\n")
    write_md(MD["09"], f"# Round22.2 · 09 BT\n\n`{bt['status']}` {bt.get('reason')}\n\n无大宗不填 0。未恢复公告时点不得进模型。\n")
    write_md(MD["10"], f"# Round22.2 · 10 MG\n\n`{mg['status']}` lineage=`{mg['lineage']}`。未改 v5 全池。非标的保持 NULL。\n")
    write_md(MD["11"], f"# Round22.2 · 11 FS\n\n`{fs['status']}` {fs.get('reason')}\n\n优先级低于 C 与 HN。新数据只写 `{REC}`。\n")
    return payload
