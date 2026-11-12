"""DISCOVER → RECOVER → VALIDATE → RECOMPUTE。新数据只写版本化独立目录。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from ..paths import ROUND21_2_ROOT, ROUND21_3_ROOT
from ..round18.contract import HERMES_V7, MIN1_2013, MIN1_2022
from ..round21.contract import HERMES_REBUILD_RAW, HERMES_V7 as HERMES_V7_R21
from .common import dump_json, write_md
from .contract import BLOCKED_DATA, MD, RESEARCH_ONLY, STRICT_PIT, STRICT_PIT_SUBPERIOD_EVIDENCE

REC_JSON = ROUND21_3_ROOT / "data_recovery" / "recovery.json"
REC_DIR = ROUND21_3_ROOT / "data_recovery"
VERSIONED = ROUND21_3_ROOT / "data" / "recovered_v1"

SEARCH_ROOTS = (
    Path("/data1/Stock/data/acquisitions"),
    Path("/data1/Stock/data/factor_cache"),
    Path("/home/mxy/AmazingData/data"),
    HERMES_REBUILD_RAW,
    HERMES_V7,
    MIN1_2013,
    MIN1_2022,
)


def _size(path: Path, *, deep: bool = False) -> int:
    try:
        if path.is_file():
            return int(path.stat().st_size)
        if path.is_dir() and deep:
            return int(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()))
        if path.is_dir():
            return int(path.stat().st_size)
    except OSError:
        return 0
    return 0


def _exists(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "bytes": _size(path) if path.exists() else 0}


def _schema_sample(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return out
    try:
        if path.is_file() and path.suffix == ".parquet":
            schema = pq.read_schema(path)
            out["columns"] = list(schema.names)
            out["bytes"] = _size(path)
            return out
        files = sorted(path.glob("*.parquet"))
        if not files:
            # 年分区：有界下探前两个 parquet，禁止分钟树全量 rglob
            found: list[Path] = []
            try:
                for p in path.rglob("*.parquet"):
                    found.append(p)
                    if len(found) >= 2:
                        break
            except OSError:
                found = []
            parts = []
            try:
                parts = [p for p in path.iterdir() if p.is_dir()]
            except OSError:
                parts = []
            out["n_partitions"] = len(parts)
            files = found
        out["n_parquet_top"] = len(files)
        out["bytes"] = _size(path)
        if files:
            schema = pq.read_schema(files[0])
            out["sample_file"] = str(files[0])
            out["columns"] = list(schema.names)
            out["last_file"] = str(files[-1])
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _date_span(path: Path, cols: tuple[str, ...] = ("TRADE_DATE", "trade_date", "available_at", "date")) -> dict[str, Any]:
    info = _schema_sample(path)
    if not path.exists():
        return info
    try:
        target = path
        if path.is_dir():
            files = []
            try:
                for p in path.rglob("*.parquet"):
                    files.append(p)
                    if len(files) >= 8:
                        break
            except OSError:
                files = []
            if not files:
                return info
            target = files[0] if len(files) == 1 else files[-1]
        names = set(pq.read_schema(target).names)
        col = next((c for c in cols if c in names), None)
        if col is None:
            return info
        # 首尾各读一文件取 span，避免全表
        files = [target] if target.is_file() else []
        if path.is_dir():
            files = [sorted(path.rglob("*.parquet"))[0], sorted(path.rglob("*.parquet"))[-1]]
        mins, maxs = [], []
        for f in files:
            s = pd.read_parquet(f, columns=[col])[col]
            ts = pd.to_datetime(s, errors="coerce")
            if ts.notna().any():
                mins.append(ts.min())
                maxs.append(ts.max())
        if mins:
            info["min_date"] = str(pd.Timestamp(min(mins)).date())
            info["max_date"] = str(pd.Timestamp(max(maxs)).date())
            info["date_col"] = col
    except Exception as exc:
        info["date_error"] = str(exc)
    return info


def _gh_cov() -> dict[str, Any]:
    g = ROUND21_2_ROOT / "data" / "g_factors.parquet"
    h = ROUND21_2_ROOT / "data" / "h_factors.parquet"
    out: dict[str, Any] = {}
    if g.is_file():
        gf = pd.read_parquet(g)
        gf["trade_date"] = pd.to_datetime(gf["trade_date"])
        cols = [c for c in gf.columns if c.startswith("r21_g") and not c.endswith("_strict")]
        for c in cols:
            s = pd.to_numeric(gf[c], errors="coerce")
            out[c] = {
                "coverage": float(s.notna().mean()),
                "n": int(len(s)),
                "valid": int(s.notna().sum()),
                "min_year": int(gf.loc[s.notna(), "trade_date"].dt.year.min()) if s.notna().any() else None,
                "max_year": int(gf.loc[s.notna(), "trade_date"].dt.year.max()) if s.notna().any() else None,
                "cov_2021plus": float(s.loc[gf["trade_date"].dt.year.ge(2021)].notna().mean()) if (gf["trade_date"].dt.year >= 2021).any() else None,
            }
    if h.is_file():
        hf = pd.read_parquet(h)
        hf["trade_date"] = pd.to_datetime(hf["trade_date"])
        cols = [c for c in hf.columns if c.startswith("r21_h") and not c.endswith(("_strict", "_rho", "duplicate"))]
        for c in cols:
            s = pd.to_numeric(hf[c], errors="coerce")
            years = hf.loc[s.notna(), "trade_date"].dt.year
            out[c] = {
                "coverage": float(s.notna().mean()),
                "n": int(len(s)),
                "valid": int(s.notna().sum()),
                "min_year": int(years.min()) if len(years) else None,
                "max_year": int(years.max()) if len(years) else None,
                "cov_2021plus": float(s.loc[hf["trade_date"].dt.year.ge(2021)].notna().mean()) if (hf["trade_date"].dt.year >= 2021).any() else None,
            }
    return out


def _write_subperiod(src: Path, dest: Path, year: int = 2021) -> dict[str, Any] | None:
    if not src.is_file():
        return None
    frame = pd.read_parquet(src)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    sub = frame.loc[frame["trade_date"].dt.year.ge(year)].copy()
    dest.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(dest, index=False)
    return {
        "path": str(dest),
        "n": int(len(sub)),
        "label": STRICT_PIT_SUBPERIOD_EVIDENCE,
        "window": f"{year}+",
        "do_not_compare_to_full_history_v5": True,
        "source": str(src),
        "overwrote_source": False,
    }


def run_recover(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and REC_JSON.is_file() and MD["02"].is_file():
        from .common import load_json

        existing = load_json(REC_JSON)
        if existing.get("status") in {"DONE", "PARTIAL", BLOCKED_DATA}:
            return existing
    REC_DIR.mkdir(parents=True, exist_ok=True)
    VERSIONED.mkdir(parents=True, exist_ok=True)

    # DISCOVER：分钟树禁止 rglob 全扫
    sources = []
    skip_rglob = {MIN1_2013.resolve(), MIN1_2022.resolve()} if MIN1_2013.exists() else set()
    if MIN1_2022.exists():
        skip_rglob.add(MIN1_2022.resolve())
    for root in SEARCH_ROOTS:
        sources.append(_exists(root))
        if not (root.exists() and root.is_dir()):
            continue
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved in skip_rglob or "minute_bars" in root.parts or "min1" in str(root).lower():
            child = root / "minute_bars"
            if child.exists():
                sources.append(_exists(child))
            continue
        for name in (
            "mkt_equd",
            "mkt_equ_mf_new",
            "daily_money_flow",
            "share_capital",
            "mkt_equ_mf",
            "daily_raw",
            "daily_limit_events",
            "minute_bars",
        ):
            hits = []
            try:
                direct = root / name
                if direct.exists():
                    hits = [direct]
                else:
                    hits = list(root.glob(name))[:6]
            except OSError:
                continue
            for p in hits:
                sources.append(_exists(p))

    discovered: dict[str, Any] = {
        "mkt_equd": _date_span(HERMES_REBUILD_RAW / "mkt_equd"),
        "mkt_equ_mf_new": _date_span(HERMES_REBUILD_RAW / "mkt_equ_mf_new"),
        "share_capital": _date_span(HERMES_V7 / "share_capital", ("available_at", "effective_from", "trade_date")),
        "daily_raw": _date_span(HERMES_V7 / "daily_raw"),
        "daily_limit_events": _date_span(HERMES_V7 / "daily_limit_events"),
        "min1_2013": _schema_sample(MIN1_2013 / "minute_bars"),
        "min1_2022": _schema_sample(MIN1_2022 / "minute_bars"),
        "amazingdata": _exists(Path("/home/mxy/AmazingData/data")),
        "factor_cache": _exists(Path("/data1/Stock/data/factor_cache")),
        "acquisitions": _exists(Path("/data1/Stock/data/acquisitions")),
        "hermes_v7": _exists(HERMES_V7_R21),
    }
    dump_json(VERSIONED / "discovered_inventory.json", discovered)

    ch = {"attempted": False, "status": "NOT_CONFIGURED"}
    if os.environ.get("CLICKHOUSE_HOST") or os.environ.get("HERMES_CLICKHOUSE_HOST"):
        ch = {"attempted": True, "status": "ENV_PRESENT_SKIP_UNBOUNDED", "note": "有界限速未开会话，避免全表拉取"}

    before = _gh_cov()

    # RECOVER：只写 recovered_v1，不覆盖源
    remaining: list[str] = []
    h_sub = _write_subperiod(ROUND21_2_ROOT / "data" / "h_factors.parquet", VERSIONED / "h_strict_2021plus.parquet")
    g_sub = _write_subperiod(ROUND21_2_ROOT / "data" / "g_factors.parquet", VERSIONED / "g_strict_2021plus.parquet")
    if h_sub is None:
        remaining.append("H source parquet missing; cannot cut 2021+ STRICT_PIT_SUBPERIOD_EVIDENCE")
    else:
        remaining.append("H 2016-2020 STRUCTURAL backfill → RESEARCH_ONLY, not STRICT; 2021+ in recovered_v1")
    if g_sub is None:
        remaining.append("G source parquet missing")
    else:
        remaining.append("G full-history UPDATE_TIME same-day coverage不足 → RESEARCH_ONLY; 2021+ subperiod in recovered_v1")

    # 字段可用性校验（不把 raw 整表拷进课题）
    field_checks: dict[str, Any] = {}
    for key, path, need in (
        ("mkt_equd.DEAL_AMOUNT", HERMES_REBUILD_RAW / "mkt_equd", ("DEAL_AMOUNT", "TRADE_DATE", "UPDATE_TIME")),
        ("mkt_equ_mf_new.orders", HERMES_REBUILD_RAW / "mkt_equ_mf_new", ("BUY_ORD_S", "BUY_ORD_XL", "BUY_VOL_XL", "SELL_VOL_XL", "SELL_ORD_XL")),
        ("share_capital", HERMES_V7 / "share_capital", ("available_at", "effective_from", "tradable_shares")),
        ("daily_raw.amount", HERMES_V7 / "daily_raw", ("amount", "volume", "trade_date", "instrument_id")),
    ):
        info = _schema_sample(path)
        cols = set(info.get("columns") or [])
        field_checks[key] = {
            **info,
            "required": list(need),
            "missing_fields": [c for c in need if c not in cols] if cols else list(need),
            "ok": bool(cols) and all(c in cols for c in need),
        }
    dump_json(VERSIONED / "field_checks.json", field_checks)

    g_note = {
        "g_computed_in_21_2": True,
        "pit_pass_in_21_2": False,
        "reason": "raw UPDATE_TIME 同日才 STRICT；全历史同日覆盖不足 → 未 PIT_PASS",
        "action": "keep RESEARCH_ONLY for non-same-day; do not abandon; T family recomputes from same raw with relative transforms",
        "subperiod": g_sub,
        "do_not_compare_to_full_history_v5": True,
    }

    # VALIDATE
    after = _gh_cov()
    validate = {
        "h_2021plus_written": h_sub is not None,
        "g_2021plus_written": g_sub is not None,
        "deal_amount_ok": bool((field_checks.get("mkt_equd.DEAL_AMOUNT") or {}).get("ok")),
        "orders_ok": bool((field_checks.get("mkt_equ_mf_new.orders") or {}).get("ok")),
        "share_capital_ok": bool((field_checks.get("share_capital") or {}).get("ok")),
        "daily_raw_ok": bool((field_checks.get("daily_raw.amount") or {}).get("ok")),
        "minutes_2013": (MIN1_2013 / "minute_bars").is_dir(),
        "minutes_2022": (MIN1_2022 / "minute_bars").is_dir(),
    }
    blocked = [k for k, v in validate.items() if v is False]
    if blocked:
        remaining.append(f"VALIDATE gaps: {blocked}")

    # RECOMPUTE 由 construct T/U/V 从 raw 重算；此处只落协议状态
    status = "DONE" if not blocked else "PARTIAL"
    payload = {
        "status": status,
        "protocol": ["DISCOVER", "RECOVER", "VALIDATE", "RECOMPUTE"],
        "sources": sources,
        "discovered": discovered,
        "field_checks": field_checks,
        "validate": validate,
        "clickhouse": ch,
        "before_coverage": before,
        "after_coverage": after,
        "versioned_dir": str(VERSIONED),
        "overwrote_source": False,
        "g": g_note,
        "h_subperiod": h_sub,
        "g_subperiod": g_sub,
        "remaining": remaining,
        "structural_na_vs_true_missing": {
            "H_2016_2020": "STRUCTURAL_NA vendor backfill available_at",
            "G_no_DEAL_AMOUNT": "TRUE_DATA_MISSING",
            "money_flow_pre_2021": "STRUCTURAL_NA / RESEARCH_ONLY",
        },
        "pit_labels": {
            STRICT_PIT: "same-day UPDATE_TIME or asof available_at post-2020",
            RESEARCH_ONLY: "backfill years",
            STRICT_PIT_SUBPERIOD_EVIDENCE: "H/G 2021+ only, not vs 2018-2026 v5",
        },
        "do_not_compare_gh_subperiod_to_v5_sharpe": True,
    }
    dump_json(REC_JSON, payload)
    dump_json(REC_DIR / "before_coverage.json", before)
    dump_json(REC_DIR / "after_coverage.json", after)
    dump_json(REC_DIR / "sources.json", sources)
    dump_json(VERSIONED / "recovery_manifest.json", {k: payload[k] for k in ("status", "validate", "remaining", "versioned_dir", "overwrote_source")})
    write_md(
        MD["02"],
        "\n".join(
            [
                "# Round21.3 · 02 数据恢复",
                "",
                "协议：DISCOVER → RECOVER → VALIDATE → RECOMPUTE。新数据只写 `data/recovered_v1/`，不覆盖源。",
                f"H/G 2021+ 子样本 `{STRICT_PIT_SUBPERIOD_EVIDENCE}`，**禁止**与 2018–2026 全历史 v5 Sharpe 直接比。",
                f"G 保持 RESEARCH_ONLY（非同日 UPDATE_TIME）。remaining=`{remaining}`",
                f"VALIDATE=`{validate}` ClickHouse `{ch}`。",
                f"status=`{status}` blocked=`{blocked}`",
                "",
            ]
        ),
    )
    return payload
