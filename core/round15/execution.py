"""X01/X02：仅共同分钟窗口（2018–2021）对比 v4。禁止拿子样本 CAGR 对全样本。无 09:25。"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from ..paths import ROUND15_ROOT
from ..round12.contract import MINUTE_ROOT
from .common import dump_json, load_json, observed_from_report, pct, write_md
from .contract import (
    AUCTION_HM,
    EXECUTION_FULL_PERIOD_NOT_IDENTIFIABLE,
    MINUTE_YEARS,
    R15_00,
    X01,
    X02,
    X02_HM,
    experiment_dir,
    parity_dir,
)
from .freeze import load_skips, require_frozen
from .runner import build_version_overlayed, hist_hooks, load_market, run_r15_backtest

EXEC_JSON = ROUND15_ROOT / "execution" / "execution.json"
EXEC_MD = ROUND15_ROOT / "08_execution.md"
WIN_START = date(2018, 1, 1)
WIN_END = date(2021, 12, 31)


def _iid_to_stem(instrument_id: str) -> str:
    iid = str(instrument_id)
    code, exch = iid.split(".")
    prefix = "sh" if exch == "XSHG" else "sz"
    return f"{prefix}{code}"


def _v4_window_trades() -> pd.DataFrame:
    path = parity_dir(R15_00) / "trades.parquet"
    t = pd.read_parquet(path)
    t["entry_date"] = pd.to_datetime(t["entry_date"]).dt.normalize()
    t["instrument_id"] = t["instrument_id"].astype(str)
    t["year"] = t["entry_date"].dt.year
    return t.loc[t["year"].isin(MINUTE_YEARS)].copy()


def _load_minute(trades: pd.DataFrame) -> pd.DataFrame:
    parts = MINUTE_ROOT / "parts"
    frames = []
    needed = trades[["instrument_id", "entry_date", "year"]].drop_duplicates()
    for year, g in needed.groupby("year"):
        yp = parts / f"year={int(year)}"
        if not yp.is_dir():
            continue
        dates = set(pd.to_datetime(g["entry_date"]).dt.strftime("%Y-%m-%d"))
        stems = {_iid_to_stem(i): i for i in g["instrument_id"]}
        for stem, iid in stems.items():
            fp = yp / f"{stem}.parquet"
            if not fp.is_file():
                continue
            raw = pd.read_parquet(fp)
            raw["entry_date"] = pd.to_datetime(raw["交易日期"]).dt.normalize()
            raw["bar_hm"] = pd.to_datetime(raw["bar_time"]).dt.strftime("%H:%M")
            raw = raw.loc[raw["entry_date"].dt.strftime("%Y-%m-%d").isin(dates)]
            if raw.empty:
                continue
            raw["instrument_id"] = str(iid)
            frames.append(raw[["instrument_id", "entry_date", "bar_hm", "open"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _scheduled_entry_keys(overlayed: pd.DataFrame, prepared, start, end) -> pd.DataFrame:
    work = overlayed.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.normalize()
    work["instrument_id"] = work["instrument_id"].astype(str)
    work["_rank"] = work.groupby("signal_date", sort=False)["score"].rank(method="first", ascending=False)
    top = work.loc[work["_rank"].le(2), ["signal_date", "instrument_id"]]
    cal = pd.DatetimeIndex(prepared.calendar).normalize().sort_values()
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    top["entry_date"] = top["signal_date"].map(nxt)
    top = top.dropna(subset=["entry_date"])
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    return top.loc[top["entry_date"].between(lo, hi), ["entry_date", "instrument_id"]].drop_duplicates()


def _official_opens(keys: pd.DataFrame, prepared=None) -> pd.DataFrame:
    if prepared is not None and getattr(prepared, "bars", None) is not None:
        bars = prepared.bars.copy()
        bars["entry_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
        bars["instrument_id"] = bars["instrument_id"].astype(str)
        bars["official_open"] = pd.to_numeric(bars["open"], errors="coerce")
        return bars[["entry_date", "instrument_id", "official_open"]]
    from ..paths import stock_path
    from ..stock_readonly import SNAPSHOT_RELATIVE, ensure_stock_src_on_path

    ensure_stock_src_on_path()
    from zyquant.data import DataSnapshot

    snapshot = DataSnapshot(stock_path(SNAPSHOT_RELATIVE), verify_hashes=False)
    days = sorted(pd.to_datetime(keys["entry_date"].unique()))
    bars = snapshot.table(
        "daily_raw",
        start=min(days).date(),
        end=max(days).date(),
        cutoff=snapshot.metadata.as_of_date,
        fields=["trade_date", "instrument_id", "open"],
    )
    bars["entry_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
    bars["instrument_id"] = bars["instrument_id"].astype(str)
    bars["official_open"] = pd.to_numeric(bars["open"], errors="coerce")
    return bars[["entry_date", "instrument_id", "official_open"]]


def _overrides(minute: pd.DataFrame, trades: pd.DataFrame, how: str, prepared=None) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta = {"how": how, "has_0925": False, "n": 0, "n_minute": 0, "n_fallback_official_open": 0, "coverage": 0.0}
    keys = trades[["entry_date", "instrument_id"]].drop_duplicates().copy()
    keys["entry_date"] = pd.to_datetime(keys["entry_date"]).dt.normalize()
    keys["instrument_id"] = keys["instrument_id"].astype(str)
    picked = pd.DataFrame(columns=["instrument_id", "entry_date", "open"])
    if not minute.empty:
        work = minute.sort_values(["instrument_id", "entry_date", "bar_hm"])
        meta["has_0925"] = bool((work["bar_hm"] == AUCTION_HM).any())
        if how == "first":
            picked = work.groupby(["instrument_id", "entry_date"], as_index=False).first()
        else:
            picked = work.loc[work["bar_hm"].eq(X02_HM)].drop_duplicates(["instrument_id", "entry_date"])
    official = _official_opens(keys, prepared=prepared)
    official = official.drop_duplicates(["entry_date", "instrument_id"], keep="last")
    need_dates = set(keys["entry_date"])
    need_ids = set(keys["instrument_id"])
    official = official.loc[official["entry_date"].isin(need_dates) & official["instrument_id"].isin(need_ids)]
    merged = keys.merge(
        picked.rename(columns={"open": "minute_px"})[["instrument_id", "entry_date", "minute_px"]],
        on=["instrument_id", "entry_date"],
        how="left",
    )
    merged = merged.merge(official, on=["instrument_id", "entry_date"], how="left")
    merged["minute_px"] = pd.to_numeric(merged["minute_px"], errors="coerce")
    merged["official_open"] = pd.to_numeric(merged["official_open"], errors="coerce")
    merged["entry_reference_price"] = merged["minute_px"].where(merged["minute_px"].gt(0), merged["official_open"])
    out = merged.loc[merged["entry_reference_price"].gt(0), ["entry_date", "instrument_id", "entry_reference_price"]].copy()
    out = out.drop_duplicates(["entry_date", "instrument_id"], keep="last")
    out["entry_date"] = pd.to_datetime(out["entry_date"]).dt.normalize()
    out["instrument_id"] = out["instrument_id"].astype(str)
    meta["n"] = int(len(out))
    meta["n_minute"] = int(merged["minute_px"].gt(0).sum())
    meta["n_fallback_official_open"] = int((merged["minute_px"].isna() | ~merged["minute_px"].gt(0)).sum())
    meta["coverage"] = float(meta["n_minute"] / len(merged)) if len(merged) else 0.0
    # 引擎若收到 overrides 必须覆盖每一笔；缺分钟用官方 open，不发明价。
    return out, meta


def _run_window(tag: str, ov, overrides, extra, resume, loaded) -> dict[str, Any]:
    dest = ROUND15_ROOT / "execution" / tag
    ran = run_r15_backtest(
        dest,
        overlayed=ov,
        validation_hooks=hist_hooks(),
        extra_report=extra,
        resume=resume,
        prepared_market=loaded["prepared"],
        market_meta=loaded,
        entry_price_overrides=overrides if overrides is not None and len(overrides) else None,
        start_date=WIN_START,
        end_date=WIN_END,
    )
    obs = observed_from_report(ran.get("report") or {})
    obs.update({"code": tag, "status": ran.get("status"), "out_dir": str(dest), "window": "2018-2021"})
    return obs


def run_execution(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen()
    if resume and EXEC_JSON.is_file():
        existing = load_json(EXEC_JSON)
        if existing.get("status") in {"DONE", "SKIP"}:
            return existing
    skips = load_skips()
    loaded = load_market()
    ov4, _, _ = build_version_overlayed("v4")
    trades = _v4_window_trades()
    keys = _scheduled_entry_keys(ov4, loaded["prepared"], WIN_START, WIN_END)
    keys_m = keys.copy()
    keys_m["year"] = pd.to_datetime(keys_m["entry_date"]).dt.year
    minute = _load_minute(keys_m)
    first_ov, meta1 = _overrides(minute, keys, "first", prepared=loaded["prepared"])
    x35_ov, meta2 = _overrides(minute, keys, "0935", prepared=loaded["prepared"])
    meta1["n_scheduled_keys"] = int(len(keys))
    meta2["n_scheduled_keys"] = int(len(keys))

    results: dict[str, Any] = {}
    # matched-window v4（官方 open，同一 2018–2021 窗）
    results["V4_WINDOW_2018_2021"] = _run_window(
        "V4_WINDOW_2018_2021",
        ov4,
        None,
        {"family": "execution", "role": "matched_window_v4", "not_full_period": True},
        resume,
        loaded,
    )

    if skips.get("SKIP_X01") or meta1["n"] == 0:
        results[X01] = {
            "code": X01,
            "status": "SKIP_X01",
            "skip": True,
            "reason": "no first executable bar on common minute window",
        }
    else:
        results[X01] = _run_window(
            X01,
            ov4,
            first_ov,
            {"family": "execution", "role": "first_causally_executable", "not_alpha": True, **meta1},
            resume,
            loaded,
        )
        results[X01]["skip"] = False

    if skips.get("SKIP_X02") or meta2["n"] == 0:
        results[X02] = {
            "code": X02,
            "status": "SKIP_X02",
            "skip": True,
            "reason": "no 09:35 bar",
        }
    else:
        results[X02] = _run_window(
            X02,
            ov4,
            x35_ov,
            {"family": "execution", "role": "fixed_0935", "not_alpha": True, **meta2},
            resume,
            loaded,
        )
        results[X02]["skip"] = False

    payload = {
        "status": "DONE",
        "window": {"start": str(WIN_START), "end": str(WIN_END), "years": list(MINUTE_YEARS)},
        "full_period": EXECUTION_FULL_PERIOD_NOT_IDENTIFIABLE,
        "compared_subsample_to_full_period": False,
        "has_0925": bool(meta1.get("has_0925") or meta2.get("has_0925")),
        "skip_auction": True,
        "not_alpha_family": True,
        "minute_meta": {"first": meta1, "x0935": meta2, "n_window_trades": int(len(trades))},
        "results": results,
        "note": "只在共同分钟年份对比。X01/X02 不计 Alpha family，不得仅因此升 v5。",
    }
    dump_json(EXEC_JSON, payload)
    lines = [
        "# Round 15 · 执行基准 X01/X02",
        "",
        f"窗口 = 2018–2021。**{EXECUTION_FULL_PERIOD_NOT_IDENTIFIABLE}**。未用子样本 CAGR 对全样本。",
        f"09:25=`{payload['has_0925']}` → SKIP_AUCTION_SIGNAL。",
        "",
        "| ID | status | CAGR | Sharpe | MaxDD |",
        "|---|---|---:|---:|---:|",
    ]
    for code in ("V4_WINDOW_2018_2021", X01, X02):
        row = results.get(code) or {}
        lines.append(
            f"| `{code}` | {row.get('status')} | {pct(row.get('cagr'))} | {row.get('sharpe')} | {pct(row.get('max_drawdown'))} |"
        )
    write_md(EXEC_MD, "\n".join(lines) + "\n")
    return payload


def load_execution() -> dict[str, Any]:
    return load_json(EXEC_JSON)


__all__ = ["experiment_dir", "load_execution", "run_execution"]
