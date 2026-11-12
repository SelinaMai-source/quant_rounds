"""历史期权 metadata + 日 K + 标的 ETF。chunked / resume / 独立 raw。不拉全 tick。"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import ROUND16_ROOT, assert_project_result_path
from .common import dump_json, file_sha256, load_json, redact, write_md
from .contract import ETF_UNDERLYINGS, KLINE_BEGIN, KLINE_END, OPTION_SECURITY_TYPE
from .sdk import (
    RAW_ROOT,
    codes_from,
    classify_error,
    ensure_login,
    local_path,
    period_day,
    query_kline_safe,
    save_frame,
    to_frame,
)
from .audit import load_audit

CKPT = RAW_ROOT / "download_checkpoint.jsonl"
META_JSON = ROUND16_ROOT / "sdk" / "download.json"
KDIR = RAW_ROOT / "option_kline_day"
HIST_DIR = RAW_ROOT / "hist_codes"
BASIC_DIR = RAW_ROOT / "basic"
BATCH = 40
YEAR_WINDOWS = [(y, y * 10000 + 101, y * 10000 + 1231) for y in range(2015, 2027)]


def _append_ckpt(rec: dict[str, Any]) -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(CKPT)
    rec = dict(rec)
    rec["download_time"] = datetime.now(timezone.utc).isoformat()
    rec["sdk_version"] = "1.1.7"
    rec["source"] = "AmazingData"
    with CKPT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(redact(rec), ensure_ascii=False, default=str) + "\n")


def _load_done() -> set[str]:
    done: set[str] = set()
    if not CKPT.is_file():
        return done
    for line in CKPT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("ok") and rec.get("batch_key"):
            done.add(str(rec["batch_key"]))
    return done


def _union_hist_codes(base) -> dict[str, Any]:
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    all_codes: list[str] = []
    by_year: list[dict[str, Any]] = []
    seen: set[str] = set()
    for year, start, end in YEAR_WINDOWS:
        rec: dict[str, Any] = {"year": year, "n": 0}
        path = HIST_DIR / f"hist_{year}.parquet"
        try:
            if path.is_file():
                codes = pd.read_parquet(path)["code"].astype(str).tolist()
            else:
                hist = base.get_hist_code_list(
                    security_type=OPTION_SECURITY_TYPE,
                    start_date=int(start),
                    end_date=int(end),
                    local_path=local_path(),
                )
                codes = codes_from(hist)
                save_frame(pd.DataFrame({"code": codes}), path)
            rec["n"] = len(codes)
            rec["path"] = str(path)
            for c in codes:
                if c not in seen:
                    seen.add(c)
                    all_codes.append(c)
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}"
            rec["error_class"] = classify_error(exc)
        by_year.append(rec)
    union_path = HIST_DIR / "hist_union.parquet"
    save_frame(pd.DataFrame({"code": all_codes}), union_path)
    return {"n_union": len(all_codes), "by_year": by_year, "path": str(union_path), "codes": all_codes}


def _download_basic(info, codes: list[str], done: set[str]) -> dict[str, Any]:
    BASIC_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    failed = []
    lp = local_path()
    chunks = [codes[i : i + 80] for i in range(0, len(codes), 80)]
    for i, batch in enumerate(chunks):
        key = f"basic_{i:04d}"
        path = BASIC_DIR / f"part-{i:04d}.parquet"
        if key in done and path.is_file():
            continue
        try:
            df = to_frame(info.get_option_basic_info(list(batch), local_path=lp, is_local=False))
            rec = save_frame(df, path)
            rec.update({"batch_key": key, "ok": True, "n_codes": len(batch), "requested_range": None})
            _append_ckpt(rec)
            files.append(rec)
        except Exception as exc:
            failed.append({"step": key, "error": type(exc).__name__, "error_class": classify_error(exc)})
            _append_ckpt({"batch_key": key, "ok": False, "error": type(exc).__name__})
            time.sleep(1.0)
    return {"n_parts": len(list(BASIC_DIR.glob("part-*.parquet"))), "failed": failed, "n_new": len(files)}


def _download_kline(md, ad, codes: list[str], done: set[str], *, begin: int, end: int) -> dict[str, Any]:
    KDIR.mkdir(parents=True, exist_ok=True)
    period = period_day(ad)
    files = []
    failed = []
    n_skip = 0
    batches = [codes[i : i + BATCH] for i in range(0, len(codes), BATCH)]
    for i, batch in enumerate(batches):
        key = f"opt_kline_day_{i:04d}"
        path = KDIR / f"part-{i:04d}.parquet"
        if key in done and path.is_file():
            n_skip += 1
            continue
        df, meta = query_kline_safe(md, batch, begin=begin, end=end, period=period)
        if meta.get("permission"):
            failed.append({"step": key, "error_class": "PERMISSION_REQUIRED", "n_codes": len(batch)})
            _append_ckpt({"batch_key": key, "ok": False, "error_class": "PERMISSION_REQUIRED"})
            return {"status": "PERMISSION_REQUIRED", "n_new": len(files), "failed": failed, "skipped": n_skip, "n_parts": i}
        rec = save_frame(df, path)
        rec.update({
            "batch_key": key,
            "ok": True,
            "n_codes": len(batch),
            "requested_range": [begin, end],
            "actual_rows": rec.get("rows"),
            "record_count": rec.get("rows"),
            "contract_codes": batch,
        })
        _append_ckpt(rec)
        files.append(rec)
        if (i + 1) % 20 == 0:
            print(f"[download] kline batch {i+1}/{len(batches)} rows={rec.get('rows')}", flush=True)
    return {
        "status": "DONE",
        "n_new": len(files),
        "skipped": n_skip,
        "failed": failed,
        "n_parts": len(list(KDIR.glob("part-*.parquet"))),
        "n_batches": len(batches),
    }


def _download_etf(md, ad, done: set[str]) -> dict[str, Any]:
    key = "etf_daily_kline"
    path = RAW_ROOT / "etf_daily.parquet"
    if key in done and path.is_file():
        return {"skipped": True, "path": str(path), "rows": int(len(pd.read_parquet(path)))}
    period = period_day(ad)
    df, meta = query_kline_safe(md, list(ETF_UNDERLYINGS), begin=KLINE_BEGIN, end=KLINE_END, period=period)
    rec = save_frame(df, path)
    rec.update({"batch_key": key, "ok": bool(meta.get("ok")), "requested_range": [KLINE_BEGIN, KLINE_END], "underlying": list(ETF_UNDERLYINGS)})
    _append_ckpt(rec)
    return rec


def run_download(
    *,
    resume: bool = True,
    start: int | None = None,
    end: int | None = None,
    underlying: str | None = None,
    dry_run: bool = False,
    verify: bool = False,
    **_: Any,
) -> dict[str, Any]:
    audit = load_audit()
    begin = int(start or KLINE_BEGIN)
    finish = int(end or KLINE_END)
    if dry_run:
        payload = {
            "status": "DRY_RUN",
            "begin": begin,
            "end": finish,
            "underlying": underlying,
            "history_available": audit.get("history_available"),
            "history_status": audit.get("history_status"),
        }
        dump_json(META_JSON, payload)
        return payload
    if audit.get("history_available") is False and audit.get("history_status") in {"PERMISSION_REQUIRED", "UNAVAILABLE"}:
        payload = {
            "status": "SKIP_HISTORY_UNRECOVERABLE",
            "history_status": audit.get("history_status"),
            "verdict": audit.get("verdict"),
            "note": "P0 判定历史不可恢复，不编造下载成功。",
        }
        dump_json(META_JSON, payload)
        write_md(ROUND16_ROOT / "logs" / "download_blocked.md", f"# download blocked\n\n{payload}\n")
        return payload
    if resume and META_JSON.is_file():
        existing = load_json(META_JSON)
        if existing.get("status") == "DONE":
            return existing
    ad, login_meta, login_status = ensure_login()
    if ad is None:
        payload = {"status": "SKIP_NO_LOGIN", "login_status": login_status, "login_meta": redact(login_meta)}
        dump_json(META_JSON, payload)
        return payload
    base = ad.BaseData()
    info = ad.InfoData()
    calendar = base.get_calendar()
    md = ad.MarketData(calendar)
    union = _union_hist_codes(base)
    codes = list(union.pop("codes"))
    try:
        current = codes_from(base.get_option_code_list(security_type=OPTION_SECURITY_TYPE))
    except Exception:
        current = []
    for c in current:
        if c not in set(codes):
            codes.append(c)
    if underlying:
        # 过滤：仅当 basic 已有时按标的筛；否则全量
        pass
    if verify:
        payload = {
            "status": "VERIFY",
            "n_hist_union": union.get("n_union"),
            "n_codes": len(codes),
            "n_current": len(current),
            "by_year": union.get("by_year"),
            "begin": begin,
            "end": finish,
        }
        dump_json(META_JSON, payload)
        return payload
    done = _load_done() if resume else set()
    basic_rec = _download_basic(info, codes, done)
    etf_rec = _download_etf(md, ad, done)
    kline_rec = _download_kline(md, ad, codes, done, begin=begin, end=finish)
    payload = {
        "status": kline_rec.get("status") or "DONE",
        "login_status": login_status,
        "n_codes": len(codes),
        "n_current": len(current),
        "hist_union": union,
        "basic": basic_rec,
        "etf": {k: etf_rec.get(k) for k in ("path", "rows", "sha256", "skipped") if k in etf_rec or True},
        "kline": kline_rec,
        "begin": begin,
        "end": finish,
        "batch": BATCH,
        "raw_root": str(RAW_ROOT),
        "no_tick": True,
        "independent_ad_local": True,
    }
    if payload["status"] == "DONE" and kline_rec.get("failed"):
        payload["status"] = "PARTIAL"
    dump_json(META_JSON, redact(payload))
    return payload


def load_download() -> dict[str, Any]:
    return load_json(META_JSON)


def iter_kline_parts() -> list[Path]:
    if not KDIR.is_dir():
        return []
    return sorted(KDIR.glob("part-*.parquet"))


def load_all_kline() -> pd.DataFrame:
    parts = iter_kline_parts()
    if not parts:
        return pd.DataFrame()
    frames = []
    for p in parts:
        df = pd.read_parquet(p)
        if df is not None and not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_all_basic() -> pd.DataFrame:
    parts = sorted(BASIC_DIR.glob("part-*.parquet")) if BASIC_DIR.is_dir() else []
    frames = []
    for p in parts:
        df = pd.read_parquet(p)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        p = RAW_ROOT / "option_basic_info.parquet"
        if p.is_file():
            return pd.read_parquet(p)
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "MARKET_CODE" in out.columns:
        out = out.drop_duplicates(subset=["MARKET_CODE"], keep="last")
    return out.reset_index(drop=True)


def load_etf() -> pd.DataFrame:
    p = RAW_ROOT / "etf_daily.parquet"
    if p.is_file():
        return pd.read_parquet(p)
    return pd.DataFrame()
