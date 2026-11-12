"""PIT 宇宙：daily_raw spine + instruments + special_treatment(known_at≤day)。禁止当前证券列表。"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ..paths import MARKET_END, PREDICTION_START, ROUND12_ROOT, stock_path
from ..round11.contract import SNAPSHOT_ID
from ..stock_readonly import SNAPSHOT_RELATIVE, ensure_stock_src_on_path
from .common import dump_json, write_md
from .contract import CHINEXT_20PCT_FROM
from .runner import load_v4_overlayed

UNIVERSE_MD = ROUND12_ROOT / "02_pit_universe.md"
PIT_PATH = ROUND12_ROOT / "universe" / "pit_universe_daily.parquet"
AUDIT_JSON = ROUND12_ROOT / "universe" / "universe.json"


def _board_of(instrument_id: str) -> str:
    iid = str(instrument_id)
    if iid.endswith(".XBEI"):
        return "BJ"
    code = iid.split(".")[0]
    if code.startswith("688"):
        return "STAR"
    if code.startswith(("300", "301")):
        return "CHINEXT"
    if code.startswith(("8", "4")) and iid.endswith(".XBEI"):
        return "BJ"
    return "MAIN"


def _limit_rule(board: str, st_flag: bool, day) -> float:
    if st_flag:
        return 0.05
    if board == "STAR":
        return 0.20
    if board == "CHINEXT":
        return 0.20 if pd.Timestamp(day) >= pd.Timestamp(CHINEXT_20PCT_FROM) else 0.10
    if board == "BJ":
        return 0.30
    return 0.10


def _active_st_by_day(st: pd.DataFrame, days: list[pd.Timestamp]) -> dict[pd.Timestamp, set[str]]:
    """复用 first_board_hermes.universe 的 known_at≤day 语义，不用 membership.effective_to。"""
    ensure_stock_src_on_path()
    from first_board_hermes.universe import _normalise_special_treatment

    trans = _normalise_special_treatment(st)
    out: dict[pd.Timestamp, set[str]] = {}
    if trans.empty:
        return {d: set() for d in days}
    for day in days:
        visible = trans[trans["effective_from"].le(day) & trans["known_at"].le(day)]
        if visible.empty:
            out[day] = set()
            continue
        latest = visible.sort_values(
            ["instrument_id", "effective_from", "known_at", "_order"]
        ).drop_duplicates("instrument_id", keep="last")
        flagged = latest["name"].fillna("").astype(str).str.contains("ST", regex=False)
        out[day] = set(latest.loc[flagged, "instrument_id"].astype(str))
    return out


def _load_snapshot_tables() -> dict[str, Any]:
    ensure_stock_src_on_path()
    from zyquant.data import DataSnapshot

    snapshot = DataSnapshot(stock_path(SNAPSHOT_RELATIVE), verify_hashes=False)
    inst = snapshot.table("instruments", cutoff=snapshot.metadata.as_of_date)
    st = snapshot.table("special_treatment", cutoff=snapshot.metadata.as_of_date)
    bars = snapshot.table(
        "daily_raw",
        start=PREDICTION_START,
        end=MARKET_END,
        cutoff=snapshot.metadata.as_of_date,
        fields=[
            "trade_date", "instrument_id", "open", "high", "low", "close",
            "pre_close", "volume", "amount", "paused", "limit_up", "limit_down",
        ],
    )
    return {"instruments": inst, "special_treatment": st, "daily_raw": bars, "snapshot": snapshot}


def build_pit_universe() -> pd.DataFrame:
    loaded = _load_snapshot_tables()
    bars = loaded["daily_raw"].copy()
    inst = loaded["instruments"].copy()
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
    bars["instrument_id"] = bars["instrument_id"].astype(str)
    inst["instrument_id"] = inst["instrument_id"].astype(str)
    inst["list_date"] = pd.to_datetime(inst["list_date"], errors="coerce")
    inst["delist_date"] = pd.to_datetime(inst["delist_date"], errors="coerce")
    keep = inst[["instrument_id", "list_date", "delist_date", "exchange", "name"]].drop_duplicates("instrument_id")
    work = bars.merge(keep, on="instrument_id", how="left")
    work["listed"] = work["list_date"].notna() & work["trade_date"].ge(work["list_date"])
    work["delisted_by_instrument"] = work["delist_date"].notna() & work["trade_date"].ge(work["delist_date"])
    work["in_daily_raw"] = True
    work["delisted"] = ~work["in_daily_raw"]
    work["board"] = work["instrument_id"].map(_board_of)
    work["exclude_bj"] = work["board"].eq("BJ") | work["exchange"].astype(str).eq("XBEI")
    days = sorted(pd.to_datetime(work["trade_date"].unique()))
    st_map = _active_st_by_day(loaded["special_treatment"], days)
    st_rows = [
        {"trade_date": day, "instrument_id": iid, "ST_status": True}
        for day, names in st_map.items()
        for iid in names
    ]
    if st_rows:
        st_df = pd.DataFrame(st_rows)
        work = work.merge(st_df, on=["trade_date", "instrument_id"], how="left")
        work["ST_status"] = work["ST_status"].fillna(False).astype(bool)
    else:
        work["ST_status"] = False
    work["suspended"] = work["paused"].fillna(False).astype(bool)
    vol = pd.to_numeric(work["volume"], errors="coerce")
    work["tradable"] = (~work["suspended"]) & vol.gt(0)
    work["price_limit_rule"] = [
        _limit_rule(b, st, d) for b, st, d in zip(work["board"], work["ST_status"], work["trade_date"])
    ]
    pre = pd.to_numeric(work["pre_close"], errors="coerce")
    lu = pd.to_numeric(work["limit_up"], errors="coerce")
    implied = (lu / pre) - 1.0
    work["implied_limit"] = implied
    work["limit_rule_match"] = (implied - work["price_limit_rule"]).abs() <= 0.015
    work["pit_eligible"] = (
        work["listed"]
        & ~work["delisted_by_instrument"]
        & ~work["exclude_bj"]
        & work["tradable"]
        & work["in_daily_raw"]
    )
    cols = [
        "trade_date", "instrument_id", "listed", "delisted", "delisted_by_instrument",
        "in_daily_raw", "ST_status", "suspended", "tradable", "board",
        "price_limit_rule", "implied_limit", "limit_rule_match", "exclude_bj",
        "pit_eligible", "volume", "amount", "paused", "open", "pre_close",
        "limit_up", "list_date", "delist_date",
    ]
    return work[cols]


def _candidate_panel() -> pd.DataFrame:
    panel = stock_path(
        "data/first_board_hermes/aligned_original_candidate_full44_plus_d1_open_gap_v1/candidate_panel.parquet"
    )
    frame = pd.read_parquet(panel, columns=["trade_date", "instrument_id"])
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame["instrument_id"] = frame["instrument_id"].astype(str)
    frame = frame.loc[
        (frame["trade_date"] >= pd.Timestamp(PREDICTION_START))
        & (frame["trade_date"] <= pd.Timestamp(MARKET_END))
    ]
    return frame.drop_duplicates()


def audit_vs_candidates(pit: pd.DataFrame) -> dict[str, Any]:
    cands = _candidate_panel()
    overlayed, _, _ = load_v4_overlayed()
    ov = overlayed.copy()
    ov["signal_date"] = pd.to_datetime(ov["signal_date"]).dt.normalize()
    ov["instrument_id"] = ov["instrument_id"].astype(str)
    ov["_rank"] = ov.groupby("signal_date")["score"].rank(method="first", ascending=False)
    tops = ov.loc[ov["_rank"].isin([1, 2]), ["signal_date", "instrument_id", "_rank"]].copy()
    pit_key = pit[["trade_date", "instrument_id", "pit_eligible", "ST_status", "exclude_bj", "tradable", "listed"]].copy()
    pit_key = pit_key.rename(columns={"trade_date": "signal_date"})
    merged_c = cands.merge(
        pit_key, left_on=["trade_date", "instrument_id"], right_on=["signal_date", "instrument_id"], how="left"
    )
    in_pit = merged_c["pit_eligible"].fillna(False)
    only_legacy = merged_c.loc[~in_pit, ["trade_date", "instrument_id"]]
    cand_keys = set(zip(cands["trade_date"].astype("int64"), cands["instrument_id"]))
    pit_elig = pit.loc[pit["pit_eligible"], ["trade_date", "instrument_id"]]
    # only_pit vs first-board candidates is expected huge (full A vs event set); report overlap only
    merged_t = tops.merge(pit_key, on=["signal_date", "instrument_id"], how="left")
    top_hit = merged_t["pit_eligible"].fillna(False)
    top1 = merged_t.loc[merged_t["_rank"].eq(1)]
    top2 = merged_t.loc[merged_t["_rank"].eq(2)]
    n_cand = int(len(cands))
    n_cand_in_pit = int(in_pit.sum())
    n_top_excluded = int((~top_hit).sum())
    independent_delist_pit = False
    if not independent_delist_pit:
        verdict = "UNQUANTIFIED"
        reason = "缺独立退市公告 PIT；daily_raw 缺失只能作辅助。按契约不得标 SURVIVORSHIP_PASS。"
    elif n_top_excluded > 0 or (n_cand and n_cand_in_pit / n_cand < 0.98):
        verdict = "MATERIAL_DIFFERENCE"
        reason = "PIT 过滤改变候选或 Top1/Top2。"
    else:
        verdict = "SURVIVORSHIP_PASS"
        reason = "PIT 与候选/Top 对齐且有独立退市 PIT。"
    quality_ok = verdict == "SURVIVORSHIP_PASS"
    return {
        "n_candidate_rows": n_cand,
        "n_candidate_in_pit": n_cand_in_pit,
        "candidate_overlap": float(n_cand_in_pit / n_cand) if n_cand else None,
        "n_candidate_only_legacy": int(len(only_legacy)),
        "n_top_rows": int(len(merged_t)),
        "n_top_excluded": n_top_excluded,
        "n_top1_excluded": int((~top1["pit_eligible"].fillna(False)).sum()),
        "n_top2_excluded": int((~top2["pit_eligible"].fillna(False)).sum()),
        "top_excluded_sample": merged_t.loc[~top_hit, ["signal_date", "instrument_id", "_rank"]].head(20).astype(str).to_dict("records"),
        "only_legacy_sample": only_legacy.head(20).astype(str).to_dict("records"),
        "independent_delist_announcement_pit": independent_delist_pit,
        "used_current_security_list": False,
        "used_membership_effective_to": False,
        "verdict": verdict,
        "quality_ok_for_R12_02": quality_ok,
        "reason": reason,
    }


def run_universe(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    dest = ROUND12_ROOT / "universe"
    dest.mkdir(parents=True, exist_ok=True)
    if resume and PIT_PATH.is_file() and AUDIT_JSON.is_file():
        existing = json_load_if(AUDIT_JSON)
        if existing.get("status") == "DONE":
            return existing
    pit = build_pit_universe()
    pit.to_parquet(PIT_PATH, index=False)
    audit = audit_vs_candidates(pit)
    payload = {
        "status": "DONE",
        "snapshot_id": SNAPSHOT_ID,
        "path": str(PIT_PATH),
        "n_rows": int(len(pit)),
        "n_days": int(pit["trade_date"].nunique()),
        "n_instruments": int(pit["instrument_id"].nunique()),
        "n_eligible": int(pit["pit_eligible"].sum()),
        "n_bj_excluded": int(pit["exclude_bj"].sum()),
        "n_st": int(pit["ST_status"].sum()),
        "limit_rule_match_rate": float(pit.loc[pit["implied_limit"].notna(), "limit_rule_match"].mean()) if pit["implied_limit"].notna().any() else None,
        "delist_date_lookahead_note": "instruments.delist_date 只作辅助并标注前视争议；主 delisted 定义=当日无 daily_raw。",
        "audit": audit,
        "R12_02_will_run": bool(audit.get("quality_ok_for_R12_02")),
    }
    dump_json(AUDIT_JSON, payload)
    lines = [
        "# Round 12 · PIT 宇宙",
        "",
        "spine = Hermes v7 `daily_raw`。`listed` = `trade_date >= list_date`。",
        "`delisted` 优先=当日无 daily_raw；`delist_date` 只作辅助（前视争议）。",
        "`ST_status` = `special_treatment` 中 `known_at <= day` 的最新状态名含 ST。",
        "复用 `first_board_hermes.universe._normalise_special_treatment`。",
        "**未使用** 当前证券列表，**未使用** `universe_membership.effective_to`。北交所排除。",
        "",
        f"- rows = **{payload['n_rows']}**；days = {payload['n_days']}；instruments = {payload['n_instruments']}",
        f"- eligible = {payload['n_eligible']}；BJ excluded rows = {payload['n_bj_excluded']}；ST rows = {payload['n_st']}",
        f"- limit_rule vs limit_up/pre_close match = {payload['limit_rule_match_rate']}",
        f"- 审计终审 = **{audit['verdict']}**",
        f"- R12_02 是否跑 = **{payload['R12_02_will_run']}**（{audit['reason']}）",
        "",
        f"- 候选重叠 = {audit['n_candidate_in_pit']} / {audit['n_candidate_rows']} = {audit['candidate_overlap']}",
        f"- Top1/Top2 被 PIT 排除 = {audit['n_top_excluded']}（Top1 {audit['n_top1_excluded']} / Top2 {audit['n_top2_excluded']}）",
        "",
        "缺独立退市公告 PIT → 不得 `SURVIVORSHIP_PASS`。",
        "",
    ]
    write_md(UNIVERSE_MD, "\n".join(lines))
    return payload


def json_load_if(path) -> dict[str, Any]:
    import json
    from pathlib import Path

    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def load_universe() -> dict[str, Any]:
    return json_load_if(AUDIT_JSON)


def load_pit() -> pd.DataFrame:
    if not PIT_PATH.is_file():
        return pd.DataFrame()
    return pd.read_parquet(PIT_PATH)


__all__ = ["load_pit", "load_universe", "run_universe"]
