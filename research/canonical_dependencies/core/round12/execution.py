"""分钟盘点 + 2018-2019 校准 / 2020-2021 验证。不选最好分钟。不可识别则 fallback。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND9_ROOT, ROUND12_ROOT, stock_path
from ..stock_readonly import SNAPSHOT_RELATIVE, ensure_stock_src_on_path
from .common import dump_json, write_md
from .contract import (
    CHINEXT_20PCT_FROM,
    FALLBACK_P50_BPS,
    FALLBACK_P75_BPS,
    FALLBACK_P90_BPS,
    FILL_SEMANTICS,
    MINUTE_ROOT,
    PARENT_V4,
    T_DECISION,
)

INV_MD = ROUND12_ROOT / "execution" / "data_inventory.md"
CAL_MD = ROUND12_ROOT / "03_execution_calibration.md"
EXEC_JSON = ROUND12_ROOT / "execution" / "execution.json"

CAL_YEARS = (2018, 2019)
VAL_YEARS = (2020, 2021)


def _iid_to_file_stem(instrument_id: str) -> str:
    iid = str(instrument_id)
    code, exch = iid.split(".")
    prefix = "sh" if exch == "XSHG" else "sz"
    return f"{prefix}{code}"


def _inventory() -> dict[str, Any]:
    parts = MINUTE_ROOT / "parts"
    years = {}
    for y in range(2017, 2025):
        yp = parts / f"year={y}"
        if yp.is_dir():
            files = list(yp.glob("*.parquet"))
            years[str(y)] = {
                "n_files": len(files),
                "bytes": int(sum(f.stat().st_size for f in files)),
                "full_market": y <= 2021 and len(files) > 1000,
            }
        else:
            years[str(y)] = {"n_files": 0, "bytes": 0, "full_market": False}
    sample = None
    probe = parts / "year=2018" / "sz000672.parquet"
    first_bar = None
    has_0925 = False
    if probe.is_file():
        df = pd.read_parquet(probe)
        sample = {"columns": list(df.columns), "n": int(len(df))}
        if "bar_time" in df.columns:
            times = pd.to_datetime(df["bar_time"]).dt.strftime("%H:%M").value_counts().head(8).to_dict()
            sample["time_head"] = {str(k): int(v) for k, v in times.items()}
            first_bar = str(pd.to_datetime(df["bar_time"]).dt.strftime("%H:%M").min())
            has_0925 = bool((pd.to_datetime(df["bar_time"]).dt.strftime("%H:%M") == "09:25").any())
    return {
        "root": str(MINUTE_ROOT),
        "years": years,
        "first_bar": first_bar or "09:30",
        "has_0925_auction": has_0925,
        "coverage_note": "2017–2021 全量；2022+ 几乎不可用（每年约 10 个文件）。无 09:25 竞价价。",
        "sample": sample,
        "t_decision": T_DECISION,
        "causal_chain": "information_available_at <= T_DECISION < order_submission <= fill",
        "baseline_fill": "同一 official daily_raw.open + 12.3bp",
        "fill_semantics": FILL_SEMANTICS,
    }


def _v4_trades() -> pd.DataFrame:
    path = ROUND9_ROOT / PARENT_V4 / "trades.parquet"
    t = pd.read_parquet(path)
    t["signal_date"] = pd.to_datetime(t["signal_date"]).dt.normalize()
    t["entry_date"] = pd.to_datetime(t["entry_date"]).dt.normalize()
    t["instrument_id"] = t["instrument_id"].astype(str)
    t["_rank"] = t.groupby("signal_date")["score"].rank(method="first", ascending=False)
    t["year"] = t["entry_date"].dt.year
    return t


def _load_minute_for(trades: pd.DataFrame) -> pd.DataFrame:
    parts = MINUTE_ROOT / "parts"
    frames = []
    needed = trades.loc[trades["year"].isin(CAL_YEARS + VAL_YEARS), ["instrument_id", "entry_date", "year"]].drop_duplicates()
    for year, g in needed.groupby("year"):
        yp = parts / f"year={int(year)}"
        if not yp.is_dir():
            continue
        stems = {_iid_to_file_stem(i) for i in g["instrument_id"]}
        dates = set(pd.to_datetime(g["entry_date"]).dt.strftime("%Y-%m-%d"))
        for stem in stems:
            fp = yp / f"{stem}.parquet"
            if not fp.is_file():
                continue
            raw = pd.read_parquet(fp)
            raw["entry_date"] = pd.to_datetime(raw["交易日期"]).dt.normalize()
            raw["bar_hm"] = pd.to_datetime(raw["bar_time"]).dt.strftime("%H:%M")
            raw = raw.loc[raw["entry_date"].dt.strftime("%Y-%m-%d").isin(dates)]
            if raw.empty:
                continue
            raw["instrument_id"] = str(next(i for i in g["instrument_id"] if _iid_to_file_stem(i) == stem))
            frames.append(raw[["instrument_id", "entry_date", "bar_hm", "open", "high", "low", "close", "volume", "amount"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _pick_bars(minute: pd.DataFrame) -> pd.DataFrame:
    if minute.empty:
        return minute
    work = minute.sort_values(["instrument_id", "entry_date", "bar_hm"])
    first = work.groupby(["instrument_id", "entry_date"], as_index=False).first().rename(
        columns={"open": "first_open", "bar_hm": "first_hm"}
    )
    def _at(hm: str, name: str) -> pd.DataFrame:
        sub = work.loc[work["bar_hm"].eq(hm), ["instrument_id", "entry_date", "open"]].rename(columns={"open": name})
        return sub.drop_duplicates(["instrument_id", "entry_date"])
    out = first[["instrument_id", "entry_date", "first_open", "first_hm"]]
    out = out.merge(_at("09:31", "px_0931"), on=["instrument_id", "entry_date"], how="left")
    out = out.merge(_at("09:35", "px_0935"), on=["instrument_id", "entry_date"], how="left")
    return out


def _daily_open(trades: pd.DataFrame) -> pd.DataFrame:
    ensure_stock_src_on_path()
    from zyquant.data import DataSnapshot

    snapshot = DataSnapshot(stock_path(SNAPSHOT_RELATIVE), verify_hashes=False)
    days = sorted(pd.to_datetime(trades["entry_date"].unique()))
    bars = snapshot.table(
        "daily_raw",
        start=min(days).date(),
        end=max(days).date(),
        cutoff=snapshot.metadata.as_of_date,
        fields=["trade_date", "instrument_id", "open", "pre_close", "amount", "limit_up", "volume", "paused"],
    )
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
    bars["instrument_id"] = bars["instrument_id"].astype(str)
    return bars.rename(columns={"trade_date": "entry_date", "open": "official_open", "amount": "d_amount"})


def _valuation(trades: pd.DataFrame) -> pd.DataFrame:
    ensure_stock_src_on_path()
    from zyquant.data import DataSnapshot

    snapshot = DataSnapshot(stock_path(SNAPSHOT_RELATIVE), verify_hashes=False)
    days = sorted(pd.to_datetime(trades["entry_date"].unique()))
    try:
        val = snapshot.table(
            "daily_valuation",
            start=min(days).date(),
            end=max(days).date(),
            cutoff=snapshot.metadata.as_of_date,
            fields=["trade_date", "instrument_id", "circulating_market_cap", "available_at"],
        )
    except Exception:
        return pd.DataFrame(columns=["entry_date", "instrument_id", "mcap"])
    val["entry_date"] = pd.to_datetime(val["trade_date"]).dt.normalize()
    val["instrument_id"] = val["instrument_id"].astype(str)
    val["mcap"] = pd.to_numeric(val["circulating_market_cap"], errors="coerce")
    return val[["entry_date", "instrument_id", "mcap"]]


def _bps(num, den) -> pd.Series:
    a = pd.to_numeric(num, errors="coerce")
    b = pd.to_numeric(den, errors="coerce")
    return (a / b - 1.0) * 10_000.0


def _percentiles(s: pd.Series) -> dict[str, float | None]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return {"p50": None, "p75": None, "p90": None, "n": 0, "mean": None}
    return {
        "p50": float(np.nanpercentile(x, 50)),
        "p75": float(np.nanpercentile(x, 75)),
        "p90": float(np.nanpercentile(x, 90)),
        "n": int(len(x)),
        "mean": float(x.mean()),
    }


def _stable(cal: dict[str, Any], val: dict[str, Any]) -> bool:
    if not cal.get("n") or not val.get("n"):
        return False
    if cal["n"] < 80 or val["n"] < 80:
        return False
    for key in ("p50", "p75"):
        if cal.get(key) is None or val.get(key) is None:
            return False
        if abs(float(cal[key]) - float(val[key])) > 15.0:
            return False
    return True


def run_execution(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    dest = ROUND12_ROOT / "execution"
    dest.mkdir(parents=True, exist_ok=True)
    if resume and EXEC_JSON.is_file():
        import json

        existing = json.loads(EXEC_JSON.read_text(encoding="utf-8"))
        if existing.get("status") == "DONE":
            return existing
    inv = _inventory()
    trades = _v4_trades()
    focus = trades.loc[trades["year"].isin(CAL_YEARS + VAL_YEARS)].copy()
    minute = _load_minute_for(focus)
    bars = _pick_bars(minute)
    daily = _daily_open(focus)
    val = _valuation(focus)
    work = focus.merge(daily, on=["entry_date", "instrument_id"], how="left")
    work = work.merge(bars, on=["entry_date", "instrument_id"], how="left")
    work = work.merge(val, on=["entry_date", "instrument_id"], how="left")
    work["open_gap"] = pd.to_numeric(work["official_open"], errors="coerce") / pd.to_numeric(work["pre_close"], errors="coerce") - 1.0
    work["limit_open"] = (
        pd.to_numeric(work["official_open"], errors="coerce")
        >= pd.to_numeric(work["limit_up"], errors="coerce") - 0.005
    )
    for src, col in (("first_open", "drift_first_bps"), ("px_0931", "drift_0931_bps"), ("px_0935", "drift_0935_bps")):
        work[col] = _bps(work[src], work["official_open"])
        work[col.replace("drift_", "adverse_")] = work[col].clip(lower=0.0)
    work["liq"] = pd.to_numeric(work["d_amount"], errors="coerce")
    work.to_parquet(dest / "trade_minute_join.parquet", index=False)

    primary = "adverse_0931_bps"
    coverage = float(work[primary].notna().mean()) if len(work) else 0.0
    cal = work.loc[work["year"].isin(CAL_YEARS)]
    vali = work.loc[work["year"].isin(VAL_YEARS)]
    by_slice = {}
    for label, frame in (("cal_2018_2019", cal), ("val_2020_2021", vali), ("all_2018_2021", work)):
        by_slice[label] = {
            "0931": _percentiles(frame[primary]),
            "0935": _percentiles(frame["adverse_0935_bps"]),
            "first": _percentiles(frame["adverse_first_bps"]),
            "signed_0931": _percentiles(frame["drift_0931_bps"]),
            "n_trades": int(len(frame)),
            "n_joined": int(frame[primary].notna().sum()),
        }
    groups = {}
    for name, mask in (
        ("top1", work["_rank"].eq(1)),
        ("top2", work["_rank"].eq(2)),
        ("limit_open", work["limit_open"].fillna(False)),
        ("not_limit_open", ~work["limit_open"].fillna(False)),
    ):
        groups[name] = _percentiles(work.loc[mask, primary])

    identifiable = coverage >= 0.40 and _stable(by_slice["cal_2018_2019"]["0931"], by_slice["val_2020_2021"]["0931"])
    if identifiable:
        exec_p = {
            "P50": float(by_slice["val_2020_2021"]["0931"]["p50"]),
            "P75": float(by_slice["val_2020_2021"]["0931"]["p75"]),
            "P90": float(by_slice["val_2020_2021"]["0931"]["p90"]),
            "source": "validation_2020_2021_adverse_0931",
            "label": "EMPIRICAL_MINUTE",
        }
        classification = "OPTIMISTIC"
        fallback = False
    else:
        exec_p = {
            "P50": FALLBACK_P50_BPS,
            "P75": FALLBACK_P75_BPS,
            "P90": FALLBACK_P90_BPS,
            "source": "FIXED_STRESS_FALLBACK",
            "label": "EXECUTION_MODEL_NOT_IDENTIFIABLE",
        }
        classification = "NOT_IDENTIFIABLE"
        fallback = True
    if identifiable and not inv["has_0925_auction"]:
        classification = "OPTIMISTIC"

    payload = {
        "status": "DONE",
        "inventory": inv,
        "n_focus_trades": int(len(focus)),
        "n_minute_rows": int(len(minute)),
        "coverage_0931": coverage,
        "by_slice": by_slice,
        "groups": groups,
        "exec": exec_p,
        "fallback": fallback,
        "classification": classification,
        "primary_bar": "09:31",
        "did_not_pick_best_minute": True,
        "features_allowed": ["open_gap", "liquidity", "mcap", "rank", "limit_state"],
        "no_hyperparam_search": True,
        "extrapolation_2022_2026": "分钟不可识别；只把校准/fallback bp 当全样本情景。",
        "same_open_execution_optimism": True,
        "causal_chain": inv["causal_chain"],
    }
    dump_json(EXEC_JSON, payload)
    dump_json(dest / "inventory.json", inv)
    write_md(
        INV_MD,
        "\n".join([
            "# Round 12 · 成交数据盘点",
            "",
            f"- 分钟根目录：`{MINUTE_ROOT}`",
            f"- 首 bar = **{inv['first_bar']}**；09:25 竞价 = **{inv['has_0925_auction']}**",
            f"- T_DECISION = `{T_DECISION}`",
            f"- 因果链：`{inv['causal_chain']}`",
            f"- 基线成交：同一 `daily_raw.open` + 12.3bp → **{FILL_SEMANTICS}**",
            "",
            "## 分年文件数",
            "",
            "| 年 | 文件数 | 全市场 |",
            "|---|---:|---|",
            *[
                f"| {y} | {info['n_files']} | {'是' if info['full_market'] else '否'} |"
                for y, info in inv["years"].items()
            ],
            "",
            inv["coverage_note"],
            "",
            "涨停分钟快照只有 D 日，不能量 D+1 买入漂移。",
            "",
        ]),
    )
    write_md(
        CAL_MD,
        "\n".join([
            "# Round 12 · 分钟校准",
            "",
            "校准 2018–2019，验证 2020–2021。时间切，不随机切。不选最好分钟。",
            f"主观察 bar = 09:31（T_DECISION=09:25 之后第一根可对齐分钟；09:30 接近官方 open）。",
            "",
            f"- coverage 09:31 = **{coverage:.2%}**",
            f"- 可识别 = **{identifiable}**",
            f"- EXEC_P50/P75/P90 = **{exec_p['P50']} / {exec_p['P75']} / {exec_p['P90']}** bp（{exec_p['label']}）",
            f"- 分类 = **{classification}**",
            f"- fallback = {fallback}",
            "",
            "## 分片百分位（买侧不利漂移 bp）",
            "",
            "| 片 | n | 09:31 P50/P75/P90 | 09:35 P50/P75/P90 | first P50 |",
            "|---|---:|---|---|---|",
            *[
                "| {k} | {n} | {a} | {b} | {c} |".format(
                    k=k,
                    n=v["0931"].get("n"),
                    a=f"{v['0931'].get('p50')}/{v['0931'].get('p75')}/{v['0931'].get('p90')}",
                    b=f"{v['0935'].get('p50')}/{v['0935'].get('p75')}/{v['0935'].get('p90')}",
                    c=v["first"].get("p50"),
                )
                for k, v in by_slice.items()
            ],
            "",
            "2022–2026：分钟不可识别，只把上述 bp 当全样本情景。",
            "",
        ]),
    )
    return payload


def load_execution() -> dict[str, Any]:
    import json

    return json.loads(EXEC_JSON.read_text(encoding="utf-8")) if EXEC_JSON.is_file() else {}


__all__ = ["load_execution", "run_execution"]
