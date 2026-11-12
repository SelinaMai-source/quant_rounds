"""期权覆盖：按 underlying × year，不是只报总行数。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND16_ROOT
from .common import dump_json, load_json, write_md
from .contract import ETF_UNDERLYINGS, R15_DATA
from .download import load_all_basic, load_all_kline, load_download, load_etf

COV_JSON = ROUND16_ROOT / "coverage" / "coverage.json"
COV_MD = ROUND16_ROOT / "01_option_data_coverage.md"
FIG_YEAR = ROUND16_ROOT / "figures" / "coverage_by_year.png"
FIG_UND = ROUND16_ROOT / "figures" / "coverage_by_underlying.png"


def underlying_from_exchange(ex: Any) -> str | None:
    s = str(ex)
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < 6:
        return None
    pref = digits[:6]
    if pref.startswith("5"):
        return f"{pref}.SH"
    if pref.startswith("1"):
        return f"{pref}.SZ"
    return None


def _r15_before() -> dict[str, Any]:
    kdir = R15_DATA / "option_kline_day"
    frames = []
    if kdir.is_dir():
        for p in sorted(kdir.glob("part-*.parquet")):
            df = pd.read_parquet(p)
            if not df.empty:
                frames.append(df)
    n_days = n_codes = n_rows = 0
    if frames:
        k = pd.concat(frames, ignore_index=True)
        n_rows = int(len(k))
        if "kline_time" in k.columns:
            n_days = int(pd.to_datetime(k["kline_time"], errors="coerce").dt.normalize().nunique())
        if "code" in k.columns:
            n_codes = int(k["code"].nunique())
    return {"n_valid_days": n_days, "n_contracts": n_codes, "n_quote_rows": n_rows}


def run_coverage(**_: Any) -> dict[str, Any]:
    before = _r15_before()
    kline = load_all_kline()
    basic = load_all_basic()
    etf = load_etf()
    dl = load_download()
    if kline.empty:
        payload = {
            "status": "EMPTY",
            "before": before,
            "n_valid_days": 0,
            "n_contracts": 0,
            "n_quote_rows": 0,
            "by_underlying_year": [],
            "download_status": dl.get("status"),
        }
        dump_json(COV_JSON, payload)
        write_md(COV_MD, "# Round16 · 01 覆盖\n\n无日 K。\n")
        return payload
    work = kline.copy()
    work["_date"] = pd.to_datetime(work.get("kline_time"), errors="coerce").dt.normalize()
    work["_code"] = work["code"].astype(str) if "code" in work.columns else work.get("_key", pd.Series(dtype=str)).astype(str)
    work = work.loc[work["_date"].notna() & work["_code"].ne("")]
    if not basic.empty and "MARKET_CODE" in basic.columns:
        b = basic.copy()
        b["_code"] = b["MARKET_CODE"].astype(str)
        b["_und"] = b["EXCHANGE_CODE"].map(underlying_from_exchange) if "EXCHANGE_CODE" in b.columns else None
        work = work.merge(b[["_code", "_und"]].drop_duplicates("_code"), on="_code", how="left")
    else:
        work["_und"] = None
    work["_year"] = work["_date"].dt.year
    rows = []
    for (und, year), g in work.groupby(["_und", "_year"], dropna=False):
        days = g["_date"].nunique()
        rows.append({
            "underlying": str(und) if pd.notna(und) else "UNKNOWN",
            "year": int(year) if pd.notna(year) else None,
            "n_quote_rows": int(len(g)),
            "n_contracts": int(g["_code"].nunique()),
            "valid_contract_days": int(days),
            "first_option_date": str(g["_date"].min().date()),
            "last_option_date": str(g["_date"].max().date()),
        })
    by = pd.DataFrame(rows)
    n_days = int(work["_date"].nunique())
    n_codes = int(work["_code"].nunique())
    years = sorted(int(y) for y in work["_year"].dropna().unique())
    payload = {
        "status": "DONE",
        "before": before,
        "after": {"n_valid_days": n_days, "n_contracts": n_codes, "n_quote_rows": int(len(work))},
        "n_valid_days": n_days,
        "n_contracts": n_codes,
        "n_quote_rows": int(len(work)),
        "kline_min": str(work["_date"].min().date()) if n_days else None,
        "kline_max": str(work["_date"].max().date()) if n_days else None,
        "year_span": years,
        "n_calendar_years": len(years),
        "by_underlying_year": rows,
        "underlyings_present": sorted({r["underlying"] for r in rows}),
        "etf_rows": int(len(etf)),
        "pre_registered_underlyings": list(ETF_UNDERLYINGS),
        "download_status": dl.get("status"),
    }
    dump_json(COV_JSON, payload)
    _plots(by)
    _write_md(payload, by)
    return payload


def _plots(by: pd.DataFrame) -> None:
    if by.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIG_YEAR.parent.mkdir(parents=True, exist_ok=True)
    year = by.groupby("year", dropna=True)["valid_contract_days"].max() if "year" in by else pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4))
    if not year.empty:
        ax.bar([str(int(i)) for i in year.index], year.values)
    ax.set_title("Option Coverage × Year (max valid days)")
    ax.set_ylabel("valid contract days")
    fig.tight_layout()
    fig.savefig(FIG_YEAR, dpi=120)
    plt.close(fig)
    und = by.groupby("underlying")["valid_contract_days"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(und.index.astype(str), und.values)
    ax.set_title("Option Coverage × Underlying (sum valid days)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIG_UND, dpi=120)
    plt.close(fig)


def _write_md(payload: dict[str, Any], by: pd.DataFrame) -> None:
    b = payload.get("before") or {}
    lines = [
        "# Round 16 · 01 期权数据覆盖",
        "",
        f"修复前：有效日=`{b.get('n_valid_days')}` 合约=`{b.get('n_contracts')}` 行=`{b.get('n_quote_rows')}`。",
        f"修复后：有效日=`{payload.get('n_valid_days')}` 合约=`{payload.get('n_contracts')}` 行=`{payload.get('n_quote_rows')}`。",
        f"区间 `{payload.get('kline_min')}` → `{payload.get('kline_max')}` 自然年=`{payload.get('year_span')}`。",
        "",
        "图：`figures/coverage_by_year.png`、`figures/coverage_by_underlying.png`。",
        "",
        "| underlying | year | first | last | valid_days | n_contracts | n_rows |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for r in payload.get("by_underlying_year") or []:
        lines.append(
            f"| {r.get('underlying')} | {r.get('year')} | {r.get('first_option_date')} | {r.get('last_option_date')} | "
            f"{r.get('valid_contract_days')} | {r.get('n_contracts')} | {r.get('n_quote_rows')} |"
        )
    lines += ["", "不能只看总行数。ETF 上市前年份记为无覆盖，不假设全年有期权。", ""]
    write_md(COV_MD, "\n".join(lines))


def load_coverage() -> dict[str, Any]:
    return load_json(COV_JSON)
