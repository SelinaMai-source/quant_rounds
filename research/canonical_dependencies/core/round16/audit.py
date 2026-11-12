"""P0：Round15 仅 63 个有效期权日的根因探针。写出 00_option_data_root_cause.md。"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import ROUND16_ROOT, assert_project_result_path, ensure_round16_tree
from .common import dump_json, file_sha256, load_json, redact, write_md
from .contract import (
    ETF_UNDERLYINGS,
    KLINE_BEGIN,
    KLINE_END,
    OPTION_SECURITY_TYPE,
    R15_DATA,
)
from .sdk import (
    AD_LOCAL,
    codes_from,
    classify_error,
    ensure_login,
    inspect_sdk,
    local_path,
    period_day,
    query_kline_safe,
    save_frame,
    to_frame,
)

AUDIT_JSON = ROUND16_ROOT / "audit" / "root_cause.json"
AUDIT_MD = ROUND16_ROOT / "00_option_data_root_cause.md"
SNAP_DIR = ROUND16_ROOT / "data" / "round15_snapshot"
SNAP_JSON = SNAP_DIR / "snapshot_sha256.json"

YEAR_WINDOWS = (
    (2015, 20150101, 20151231),
    (2018, 20180101, 20181231),
    (2019, 20190101, 20191231),
    (2021, 20210101, 20211231),
    (2023, 20230101, 20231231),
    (2025, 20250101, 20251231),
)
PROBE_YEARS = (2019, 2021, 2023, 2025)


def snapshot_round15() -> dict[str, Any]:
    """只读复制 Round15 期权数据并记 sha256。不删除原文件。"""
    ensure_round16_tree()
    assert_project_result_path(SNAP_DIR)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    sources = [
        R15_DATA / "option_basic_info.parquet",
        R15_DATA / "option_hist_code_list.parquet",
        R15_DATA / "etf_daily.parquet",
        R15_DATA / "download_checkpoint.jsonl",
    ]
    kdir = R15_DATA / "option_kline_day"
    if kdir.is_dir():
        sources.extend(sorted(kdir.glob("part-*.parquet")))
    sample = R15_DATA / "sample"
    if sample.is_dir():
        sources.extend(sorted(sample.glob("*")))
    for src in sources:
        if not src.is_file():
            files.append({"src": str(src), "copied": False, "sha256": None})
            continue
        rel = src.relative_to(R15_DATA)
        dest = SNAP_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)
        sha = file_sha256(dest)
        files.append({
            "src": str(src),
            "dest": str(dest),
            "copied": True,
            "bytes": dest.stat().st_size,
            "sha256": sha,
        })
    payload = {
        "n_files": sum(1 for x in files if x.get("copied")),
        "files": files,
        "note": "Round15 旧数据只读快照。未复制 ad_local 缓存。未删除任何 Round15 文件。",
    }
    dump_json(SNAP_JSON, payload)
    return payload


def _r15_facts() -> dict[str, Any]:
    basic_p = R15_DATA / "option_basic_info.parquet"
    hist_p = R15_DATA / "option_hist_code_list.parquet"
    kdir = R15_DATA / "option_kline_day"
    facts: dict[str, Any] = {
        "n_kline_days": 0,
        "kline_min": None,
        "kline_max": None,
        "n_kline_rows": 0,
        "n_kline_codes": 0,
        "n_basic": 0,
        "basic_start_min": None,
        "basic_start_max": None,
        "basic_expiry_min": None,
        "basic_expiry_max": None,
        "hist_n": 0,
        "kline_code_cap": 400,
        "requested_begin": KLINE_BEGIN,
        "requested_end": KLINE_END,
        "code_locations": {
            "get_option_code_list": "core/round15/sdk.py:_resume_download",
            "kline_cap": "core/round15/sdk.py:_resume_download max_batches=40 × 10",
            "hist_probe": "core/round15/sdk.py:audit_option_coverage 仅 20180101-20181231",
            "hist_unused": "hist 代码从未进入 query_kline",
        },
    }
    if basic_p.is_file():
        basic = pd.read_parquet(basic_p)
        facts["n_basic"] = int(len(basic))
        if "START_TRADE_DATE" in basic.columns:
            facts["basic_start_min"] = str(basic["START_TRADE_DATE"].min())
            facts["basic_start_max"] = str(basic["START_TRADE_DATE"].max())
        if "EXPIRY_DATE" in basic.columns:
            facts["basic_expiry_min"] = str(basic["EXPIRY_DATE"].min())
            facts["basic_expiry_max"] = str(basic["EXPIRY_DATE"].max())
        if "EXCHANGE_CODE" in basic.columns:
            und = sorted({str(x)[:6] for x in basic["EXCHANGE_CODE"].astype(str) if str(x)[:6].isdigit()})
            facts["basic_underlying_prefixes"] = und
    if hist_p.is_file():
        hist = pd.read_parquet(hist_p)
        facts["hist_n"] = int(len(hist))
        facts["hist_sample"] = [str(x) for x in hist.iloc[:5, 0].tolist()]
    frames = []
    if kdir.is_dir():
        for p in sorted(kdir.glob("part-*.parquet")):
            df = pd.read_parquet(p)
            if not df.empty:
                frames.append(df)
    if frames:
        k = pd.concat(frames, ignore_index=True)
        facts["n_kline_rows"] = int(len(k))
        tcol = "kline_time" if "kline_time" in k.columns else None
        if tcol:
            ts = pd.to_datetime(k[tcol], errors="coerce")
            facts["n_kline_days"] = int(ts.dt.normalize().nunique())
            facts["kline_min"] = str(ts.min().date()) if ts.notna().any() else None
            facts["kline_max"] = str(ts.max().date()) if ts.notna().any() else None
        if "code" in k.columns:
            facts["n_kline_codes"] = int(k["code"].nunique())
    return facts


def _hist_codes(base, year: int, start: int, end: int) -> dict[str, Any]:
    rec: dict[str, Any] = {"year": year, "start": start, "end": end, "n": 0, "error": None, "permission": False}
    try:
        hist = base.get_hist_code_list(
            security_type=OPTION_SECURITY_TYPE,
            start_date=int(start),
            end_date=int(end),
            local_path=local_path(),
        )
        codes = codes_from(hist)
        rec["n"] = int(len(codes))
        rec["sample"] = codes[:8]
        rec["kind"] = type(hist).__name__
        dest = ROUND16_ROOT / "data" / "raw" / "hist_codes" / f"hist_{year}.parquet"
        save_frame(pd.DataFrame({"code": codes}), dest)
        rec["path"] = str(dest)
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}"
        rec["error_class"] = classify_error(exc)
        rec["permission"] = classify_error(exc) == "PERMISSION_REQUIRED"
    return rec


def _probe_expired(ad, md, info, codes: list[str], year: int) -> dict[str, Any]:
    out: dict[str, Any] = {"year": year, "n_probe_codes": len(codes), "codes": codes}
    if not codes:
        out["reason"] = "NO_CODES"
        return out
    lp = local_path()
    try:
        basic = to_frame(info.get_option_basic_info(list(codes), local_path=lp, is_local=False))
        out["basic_rows"] = int(len(basic))
        out["basic_cols"] = list(basic.columns)[:20]
        if not basic.empty:
            save_frame(basic, ROUND16_ROOT / "audit" / f"probe_basic_{year}.parquet")
            for c in ("START_TRADE_DATE", "LAST_TRADE_DATE", "EXPIRY_DATE"):
                if c in basic.columns:
                    out[c.lower() + "_min"] = str(basic[c].min())
                    out[c.lower() + "_max"] = str(basic[c].max())
    except Exception as exc:
        out["basic_error"] = f"{type(exc).__name__}"
        out["basic_error_class"] = classify_error(exc)
        out["basic_permission"] = classify_error(exc) == "PERMISSION_REQUIRED"
        basic = pd.DataFrame()
    period = period_day(ad)
    kdf, kmeta = query_kline_safe(md, codes, begin=KLINE_BEGIN, end=KLINE_END, period=period)
    out["kline"] = kmeta
    if not kdf.empty:
        save_frame(kdf, ROUND16_ROOT / "audit" / f"probe_kline_{year}.parquet")
        tcol = "kline_time" if "kline_time" in kdf.columns else None
        if tcol:
            ts = pd.to_datetime(kdf[tcol], errors="coerce")
            out["kline_min"] = str(ts.min().date()) if ts.notna().any() else None
            out["kline_max"] = str(ts.max().date()) if ts.notna().any() else None
            out["kline_days"] = int(ts.dt.normalize().nunique())
        out["kline_rows"] = int(len(kdf))
        out["kline_codes"] = int(kdf["code"].nunique()) if "code" in kdf.columns else None
        # 跨年？过期合约若能拉到当年日 K，则历史可恢复
        if out.get("kline_min") and str(year) in str(out.get("kline_min")) + str(out.get("kline_max")):
            out["covers_probe_year"] = True
        elif out.get("kline_min"):
            ymin = int(str(out["kline_min"])[:4])
            ymax = int(str(out["kline_max"])[:4])
            out["covers_probe_year"] = ymin <= year <= ymax
        else:
            out["covers_probe_year"] = False
    else:
        out["kline_rows"] = 0
        out["covers_probe_year"] = False
    try:
        snap = to_frame(md.query_snapshot(list(codes[:3]), begin_date=int(f"{year}0601"), end_date=int(f"{year}0610")))
        out["snapshot_rows"] = int(len(snap))
        out["snapshot_cols"] = list(snap.columns)[:20]
    except Exception as exc:
        out["snapshot_error"] = f"{type(exc).__name__}"
        out["snapshot_error_class"] = classify_error(exc)
        out["snapshot_permission"] = classify_error(exc) == "PERMISSION_REQUIRED"
    return out


def _independent_cache_probe(ad, md, codes: list[str]) -> dict[str, Any]:
    """独立 ad_local + is_local=False。对比 Round15 part parquet hash。"""
    rec: dict[str, Any] = {"ad_local": local_path(), "is_local": False}
    if not codes:
        rec["reason"] = "NO_CODES"
        return rec
    period = period_day(ad)
    kdf, kmeta = query_kline_safe(md, codes[:5], begin=KLINE_BEGIN, end=KLINE_END, period=period)
    rec["kline"] = kmeta
    rec["rows"] = int(len(kdf))
    old_parts = sorted((R15_DATA / "option_kline_day").glob("part-*.parquet"))
    rec["round15_part_hashes"] = [
        {"name": p.name, "sha256": file_sha256(p), "bytes": p.stat().st_size} for p in old_parts[:8]
    ]
    rec["cache_dir_exists"] = AD_LOCAL.exists()
    rec["note"] = "Round16 使用独立 result/round16/data/raw/ad_local，不复用 Round15 缓存目录。"
    return rec


def _pick_verdict(payload: dict[str, Any]) -> dict[str, Any]:
    r15 = payload.get("round15_facts") or {}
    hist = payload.get("hist_universe") or []
    probes = payload.get("expired_probes") or []
    perm = bool(payload.get("permission_hit"))
    hist_ok = any(int(x.get("n") or 0) > 0 and not x.get("permission") for x in hist)
    hist_n_total = sum(int(x.get("n") or 0) for x in hist)
    kline_ok = any(int(x.get("kline_rows") or 0) > 0 and x.get("covers_probe_year") for x in probes)
    kline_any = any(int(x.get("kline_rows") or 0) > 0 for x in probes)
    date_range_ok = int(r15.get("requested_begin") or 0) == KLINE_BEGIN and int(r15.get("requested_end") or 0) == KLINE_END
    current_only = (
        int(r15.get("n_kline_days") or 0) == 63
        and str(r15.get("kline_min")) == "2026-04-23"
        and str(r15.get("basic_start_min") or "").startswith("2026")
    )
    cap = int(r15.get("kline_code_cap") or 400)
    unused_hist = int(r15.get("hist_n") or 0) > 0

    causes = []
    if current_only:
        causes.append("A.CURRENT_CONTRACT_ONLY_BUG")
    if not date_range_ok:
        causes.append("B.DATE_RANGE_BUG")
    if cap < int(r15.get("n_basic") or 0):
        causes.append("C.CACHE_OR_PAGINATION_BUG")
    if unused_hist and not kline_ok:
        # mapping unused is a process bug, not mapping API fail
        pass
    if perm and not kline_ok:
        causes.append("E.SOURCE_PERMISSION_LIMIT")
    if hist_ok and not kline_any and not perm:
        causes.append("F.DATA_PROVIDER_HISTORY_LIMIT")
    if unused_hist:
        causes.append("D.HISTORICAL_CONTRACT_MAPPING_BUG")

    history_available = bool(hist_ok and (kline_ok or kline_any) and not (perm and not kline_any))
    if perm and not kline_any:
        code = "E.SOURCE_PERMISSION_LIMIT"
        history_status = "PERMISSION_REQUIRED"
    elif current_only and hist_ok and (kline_ok or kline_any):
        code = "G.MULTIPLE_CAUSES" if (cap < int(r15.get("n_basic") or 1080) or unused_hist) else "A.CURRENT_CONTRACT_ONLY_BUG"
        if cap < int(r15.get("n_basic") or 1080) or unused_hist:
            code = "G.MULTIPLE_CAUSES"
        history_status = "AVAILABLE" if kline_ok else "PARTIAL"
        history_available = True
    elif current_only and hist_ok and not kline_any:
        code = "F.DATA_PROVIDER_HISTORY_LIMIT" if not perm else "E.SOURCE_PERMISSION_LIMIT"
        history_status = "UNAVAILABLE" if not perm else "PERMISSION_REQUIRED"
        history_available = False
    elif current_only:
        code = "A.CURRENT_CONTRACT_ONLY_BUG"
        history_status = "PARTIAL" if hist_ok else "UNAVAILABLE"
        history_available = bool(hist_ok and kline_any)
    else:
        code = "H.ROOT_CAUSE_NOT_IDENTIFIED"
        history_status = "UNAVAILABLE"
        history_available = False

    letter = code.split(".", 1)[0]
    return {
        "verdict": code,
        "verdict_letter": letter,
        "causes": causes,
        "history_available": history_available,
        "history_status": history_status,
        "hist_n_total_unique_years": hist_n_total,
        "why_63_days": (
            "Round15 `_resume_download` 只用 `get_option_code_list` 当前合约（1080 个，"
            f"START_TRADE {r15.get('basic_start_min')}–{r15.get('basic_start_max')}），"
            f"再加 kline_code_cap={cap}。query_kline 请求 {r15.get('requested_begin')}–{r15.get('requested_end')}，"
            f"返回区间 {r15.get('kline_min')}–{r15.get('kline_max')} 共 {r15.get('n_kline_days')} 日，"
            "因为当前合约寿命约三个月，不是请求日期被截成 63 天。"
            f" freeze 已拿到 2018 hist_n={r15.get('hist_n')} 但从未用这些代码拉 K 线。"
        ),
    }


def run_audit(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and AUDIT_JSON.is_file():
        existing = load_json(AUDIT_JSON)
        if existing.get("status") == "DONE" and existing.get("verdict"):
            return existing
    ensure_round16_tree()
    snap = snapshot_round15()
    r15 = _r15_facts()
    inspect_payload = None
    inspect_error = None
    try:
        inspect_payload = inspect_sdk()
    except Exception as exc:
        inspect_error = f"{type(exc).__name__}: {exc}"
    ad, login_meta, login_status = ensure_login()
    hist_rows: list[dict[str, Any]] = []
    current_codes: list[str] = []
    probes: list[dict[str, Any]] = []
    cache_rec: dict[str, Any] = {}
    permission_hit = False
    extra_etf: list[str] = []
    if ad is None:
        payload = {
            "status": "DONE",
            "login_status": login_status,
            "login_meta": redact(login_meta),
            "round15_facts": r15,
            "snapshot": {"n_files": snap.get("n_files")},
            "permission_hit": True,
            "inspect": inspect_payload,
            "inspect_error": inspect_error,
        }
        verdict = _pick_verdict(payload)
        payload.update(verdict)
        dump_json(AUDIT_JSON, redact(payload))
        _write_md(payload)
        return payload

    base = ad.BaseData()
    info = ad.InfoData()
    calendar = base.get_calendar()
    md = ad.MarketData(calendar)
    try:
        raw_cur = base.get_option_code_list(security_type=OPTION_SECURITY_TYPE)
        current_codes = codes_from(raw_cur)
    except Exception as exc:
        permission_hit = classify_error(exc) == "PERMISSION_REQUIRED"
        current_codes = []
    for year, start, end in YEAR_WINDOWS:
        rec = _hist_codes(base, year, start, end)
        hist_rows.append(rec)
        permission_hit = permission_hit or bool(rec.get("permission"))
    by_year = {int(x["year"]): x for x in hist_rows}
    for year in PROBE_YEARS:
        rec = by_year.get(year) or {}
        sample = list(rec.get("sample") or [])[:6]
        if not sample and rec.get("n"):
            p = Path(rec.get("path") or "")
            if p.is_file():
                sample = pd.read_parquet(p)["code"].astype(str).head(6).tolist()
        probe = _probe_expired(ad, md, info, sample, year)
        probes.append(probe)
        if probe.get("basic_permission") or probe.get("kline", {}).get("permission") or probe.get("snapshot_permission"):
            permission_hit = True
    cache_rec = _independent_cache_probe(ad, md, current_codes[:10] or (probes[0].get("codes") if probes else []))
    und = list(r15.get("basic_underlying_prefixes") or [])
    extra_etf = [u for u in und if f"{u}.SH" not in ETF_UNDERLYINGS and f"{u}.SZ" not in ETF_UNDERLYINGS]

    payload = {
        "status": "DONE",
        "login_status": login_status,
        "login_meta": redact(login_meta),
        "inspect": inspect_payload,
        "inspect_error": inspect_error,
        "sdk_version": (inspect_payload or {}).get("version"),
        "is_local": False,
        "ad_local": local_path(),
        "round15_facts": r15,
        "snapshot": {"n_files": snap.get("n_files"), "path": str(SNAP_DIR)},
        "current_n": len(current_codes),
        "current_sample": current_codes[:8],
        "hist_universe": hist_rows,
        "expired_probes": probes,
        "cache_probe": cache_rec,
        "permission_hit": permission_hit,
        "underlyings_pre_registered": list(ETF_UNDERLYINGS),
        "underlyings_discovered_extra_prefixes": extra_etf,
        "etf_missed_in_round15": ["588080.SH"],
        "no_secrets_written": True,
    }
    verdict = _pick_verdict(payload)
    payload.update(verdict)
    dump_json(AUDIT_JSON, redact(payload))
    _write_md(payload)
    return payload


def _write_md(payload: dict[str, Any]) -> None:
    r15 = payload.get("round15_facts") or {}
    v = payload.get("verdict")
    lines = [
        "# Round 16 · 00 期权数据根因",
        "",
        f"裁决：**{v}**（字母 `{payload.get('verdict_letter')}`）。",
        f"历史可访问：`{payload.get('history_status')}` history_available=`{payload.get('history_available')}`。",
        f"login=`{payload.get('login_status')}` SDK=`{payload.get('sdk_version')}` is_local=`{payload.get('is_local')}`。",
        "",
        "## 为什么只有 63 天",
        "",
        payload.get("why_63_days") or "",
        "",
        "## Round15 事实（只读）",
        "",
        f"- n_kline_days=`{r15.get('n_kline_days')}` range=`{r15.get('kline_min')}` → `{r15.get('kline_max')}` rows=`{r15.get('n_kline_rows')}` codes=`{r15.get('n_kline_codes')}`",
        f"- n_basic=`{r15.get('n_basic')}` START_TRADE=`{r15.get('basic_start_min')}`–`{r15.get('basic_start_max')}` EXPIRY=`{r15.get('basic_expiry_min')}`–`{r15.get('basic_expiry_max')}`",
        f"- hist_n(2018 only)=`{r15.get('hist_n')}` kline_code_cap=`{r15.get('kline_code_cap')}`",
        f"- query_kline 请求 `{r15.get('requested_begin')}`–`{r15.get('requested_end')}`（不是 63 日截断）",
        f"- 代码位置：`{r15.get('code_locations')}`",
        f"- Round15 漏标的：588080.SH。当前 basic 前缀：`{r15.get('basic_underlying_prefixes')}`",
        "",
        "## 当前 vs 历史宇宙",
        "",
        f"当前 `get_option_code_list` n=`{payload.get('current_n')}` sample=`{payload.get('current_sample')}`",
        "",
        "| year | n_hist | error | permission | sample |",
        "|---:|---:|---|---|---|",
    ]
    for row in payload.get("hist_universe") or []:
        lines.append(
            f"| {row.get('year')} | {row.get('n')} | {row.get('error')} | {row.get('permission')} | `{row.get('sample')}` |"
        )
    lines += [
        "",
        "## 过期合约探针（2019/2021/2023/2025）",
        "",
        "| year | n_codes | basic_rows | kline_rows | kline_days | range | covers_year | perm |",
        "|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for p in payload.get("expired_probes") or []:
        k = p.get("kline") or {}
        lines.append(
            f"| {p.get('year')} | {p.get('n_probe_codes')} | {p.get('basic_rows')} | {p.get('kline_rows')} | "
            f"{p.get('kline_days')} | {p.get('kline_min')}→{p.get('kline_max')} | {p.get('covers_probe_year')} | "
            f"{k.get('permission') or p.get('basic_permission') or p.get('snapshot_permission')} |"
        )
    lines += [
        "",
        "## 缓存 / 权限",
        "",
        f"独立缓存 `{payload.get('ad_local')}`。permission_hit=`{payload.get('permission_hit')}`。",
        f"cache_probe=`{(payload.get('cache_probe') or {}).get('note')}` rows=`{(payload.get('cache_probe') or {}).get('rows')}`。",
        "",
        "## 伴随原因",
        "",
        f"`{payload.get('causes')}`",
        "",
        "禁止把当前代码表回填到过去。本文件不含 password/token/host 明文。",
        "",
    ]
    write_md(AUDIT_MD, "\n".join(lines))


def load_audit() -> dict[str, Any]:
    return load_json(AUDIT_JSON)
