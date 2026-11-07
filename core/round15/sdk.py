"""AmazingData SDK：按已安装包签名 inspect，小样本后 resume 下载。不打印 password/token。"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import ROUND15_ROOT, assert_project_result_path
from .common import dump_json, load_json, redact, write_md
from .contract import ETF_UNDERLYINGS, OPTION_SECURITY_TYPE

SDK_JSON = ROUND15_ROOT / "sdk" / "sdk.json"
SDK_MD = ROUND15_ROOT / "02_sdk.md"
DATA_ROOT = ROUND15_ROOT / "data"
CKPT = DATA_ROOT / "download_checkpoint.jsonl"
SAMPLE_DIR = DATA_ROOT / "sample"


def _parse_dotenv(fp: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not fp.is_file():
        return out
    for raw in fp.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'").strip('"')
        if key:
            out[key] = val
    return out


def _login_kwargs() -> dict[str, Any]:
    """与低换手 `amazingdata_login_kwargs` 同一套 env 文件；不把低换手 core 插入 sys.path。"""
    import os

    ad_root = Path(os.environ.get("AMAZINGDATA_ROOT", "/home/mxy/AmazingData")).expanduser()
    env_files = [
        ad_root / ".env",
        ad_root / "config.env",
        Path("/home/mxy/Stock/课题/低换手策略1") / ".env.amazingdata",
        Path("/home/mxy/Stock/课题/低换手策略1") / "secrets" / "amazingdata.env",
    ]
    for fp in env_files:
        for key, val in _parse_dotenv(fp).items():
            if not val or key in os.environ:
                continue
            os.environ[key] = val
    os.environ.setdefault("AMAZINGDATA_ROOT", str(ad_root))
    user = (os.environ.get("AMAZINGDATA_USERNAME") or "").strip()
    pwd = (os.environ.get("AMAZINGDATA_PASSWORD") or "").strip()
    host = (os.environ.get("AMAZINGDATA_HOST") or "").strip()
    port_s = (os.environ.get("AMAZINGDATA_PORT") or "").strip()
    out: dict[str, Any] = {}
    if user:
        out["username"] = user
    if pwd:
        out["password"] = pwd
    if host:
        out["host"] = host
    if port_s:
        out["port"] = int(port_s)
    return out


def _inspect_sdk() -> dict[str, Any]:
    import AmazingData as ad

    def _sig(cls, name: str) -> dict[str, Any]:
        fn = getattr(cls, name, None)
        if fn is None:
            return {"missing": True}
        try:
            sig = str(inspect.signature(fn))
        except (TypeError, ValueError):
            sig = "UNAVAILABLE"
        return {
            "signature": sig,
            "doc_head": ((inspect.getdoc(fn) or "").splitlines() or [""])[0][:240],
        }

    period_members = []
    try:
        period_members = list(ad.constant.Period.__members__.keys())
    except Exception as exc:
        period_members = [f"inspect_fail:{type(exc).__name__}"]
    return {
        "package": getattr(ad, "__file__", None),
        "version": getattr(ad, "__version__", None),
        "classes": [x for x in dir(ad) if not x.startswith("_")],
        "login": _sig(ad, "login") if not isinstance(ad.login, type) else {
            "signature": str(inspect.signature(ad.login)),
        },
        "BaseData": {
            "init": _sig(ad.BaseData, "__init__"),
            "get_option_code_list": _sig(ad.BaseData, "get_option_code_list"),
            "get_hist_code_list": _sig(ad.BaseData, "get_hist_code_list"),
            "get_code_list": _sig(ad.BaseData, "get_code_list"),
            "get_calendar": _sig(ad.BaseData, "get_calendar"),
        },
        "InfoData": {
            "get_option_basic_info": _sig(ad.InfoData, "get_option_basic_info"),
            "get_option_mon_ctr_specs": _sig(ad.InfoData, "get_option_mon_ctr_specs"),
            "get_option_std_ctr_specs": _sig(ad.InfoData, "get_option_std_ctr_specs"),
            "get_history_stock_status": _sig(ad.InfoData, "get_history_stock_status"),
            "get_treasury_yield": _sig(ad.InfoData, "get_treasury_yield"),
        },
        "MarketData": {
            "query_kline": _sig(ad.MarketData, "query_kline"),
            "query_snapshot": _sig(ad.MarketData, "query_snapshot"),
        },
        "Period_members": period_members,
        "note": "签名来自已安装 AmazingData 1.x 包，不抄手册 PDF 示例。",
    }


def _to_frame(raw: Any) -> pd.DataFrame:
    if raw is None:
        return pd.DataFrame()
    if isinstance(raw, pd.DataFrame):
        return raw
    if isinstance(raw, dict):
        frames = []
        for k, v in raw.items():
            if isinstance(v, pd.DataFrame) and not v.empty:
                part = v.copy()
                part["_key"] = str(k)
                frames.append(part)
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, pd.DataFrame) and not v2.empty:
                        part = v2.copy()
                        part["_key"] = str(k2)
                        frames.append(part)
        if frames:
            return pd.concat(frames, ignore_index=True)
    if isinstance(raw, (list, tuple)) and raw:
        if isinstance(raw[0], pd.DataFrame):
            return pd.concat([x for x in raw if isinstance(x, pd.DataFrame)], ignore_index=True)
        if isinstance(raw[0], dict):
            return pd.DataFrame(list(raw))
    return pd.DataFrame()


def _file_sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _append_ckpt(rec: dict[str, Any]) -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(CKPT)
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


class _RedactStream:
    """拦截 SDK 把 Token 打到 stdout。"""

    def __init__(self, inner):
        self.inner = inner

    def write(self, s):
        if not s:
            return 0
        text = str(s)
        if "Token" in text or "password" in text.lower():
            text = "[REDACTED_SDK_LOGON]\n"
        return self.inner.write(text)

    def flush(self):
        return self.inner.flush()

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _login():
    import contextlib
    import sys

    kw = _login_kwargs()
    missing = {"username", "password", "host", "port"} - set(kw)
    meta = {
        "has_username": bool(kw.get("username")),
        "has_password": bool(kw.get("password")),
        "has_host": bool(kw.get("host")),
        "has_port": kw.get("port") is not None,
        "missing": sorted(missing),
        "redacted": redact(kw),
    }
    if missing:
        return None, meta, "LOGIN_FIELDS_MISSING"
    import AmazingData as ad

    with contextlib.redirect_stdout(_RedactStream(sys.stdout)), contextlib.redirect_stderr(_RedactStream(sys.stderr)):
        ad.login(username=kw["username"], password=kw["password"], host=kw["host"], port=kw["port"])
    return ad, meta, "LOGGED_IN"


def _period_day(ad) -> Any:
    period = ad.constant.Period
    if hasattr(period, "day"):
        return period.day.value
    return 86400000


def _small_sample(ad, local_path: str) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False}
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(SAMPLE_DIR)
    try:
        base = ad.BaseData()
        codes = base.get_option_code_list(security_type=OPTION_SECURITY_TYPE)
        if isinstance(codes, dict):
            code_list = list(codes.keys()) if codes else []
            out["code_list_kind"] = "dict"
        elif isinstance(codes, (list, tuple)):
            code_list = [str(x) for x in codes]
            out["code_list_kind"] = "list"
        else:
            code_list = []
            out["code_list_kind"] = type(codes).__name__
        out["n_option_codes"] = len(code_list)
        out["sample_codes"] = code_list[:5]
        if not code_list:
            out["reason"] = "EMPTY_OPTION_CODE_LIST"
            return out
        info = ad.InfoData()
        probe = code_list[:3]
        basic = info.get_option_basic_info(probe, local_path=local_path, is_local=False)
        basic_df = _to_frame(basic)
        basic_path = SAMPLE_DIR / "option_basic_info_sample.parquet"
        if not basic_df.empty:
            basic_df.to_parquet(basic_path, index=False)
        out["basic_rows"] = int(len(basic_df))
        out["basic_cols"] = list(basic_df.columns)[:40]
        out["basic_sha256"] = _file_sha(basic_path)
        calendar = base.get_calendar()
        md = ad.MarketData(calendar)
        period = _period_day(ad)
        kraw = md.query_kline([probe[0]], begin_date=20240102, end_date=20240110, period=period)
        kdf = _to_frame(kraw)
        kpath = SAMPLE_DIR / "option_kline_sample.parquet"
        if not kdf.empty:
            kdf.to_parquet(kpath, index=False)
        out["kline_rows"] = int(len(kdf))
        out["kline_cols"] = list(kdf.columns)[:40]
        out["kline_sha256"] = _file_sha(kpath)
        out["ok"] = bool(len(basic_df) or len(kdf) or code_list)
        out["reason"] = None if out["ok"] else "SAMPLE_EMPTY"
    except Exception as exc:
        out["ok"] = False
        out["reason"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()[-1500:]
    return out


def _save_frame(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(path)
    if df is None or df.empty:
        return {"path": str(path), "rows": 0, "sha256": None}
    df.to_parquet(path, index=False)
    return {"path": str(path), "rows": int(len(df)), "sha256": _file_sha(path)}


def _resume_download(ad, local_path: str, sample: dict[str, Any]) -> dict[str, Any]:
    """基础信息 + 日 K。不先拉全市场高频。checkpoint/resume。"""
    done = _load_done()
    downloaded: dict[str, Any] = {"files": [], "skipped": 0, "failed": []}
    base = ad.BaseData()
    info = ad.InfoData()
    calendar = base.get_calendar()
    md = ad.MarketData(calendar)
    period = _period_day(ad)

    codes = sample.get("sample_codes") or []
    try:
        raw_codes = base.get_option_code_list(security_type=OPTION_SECURITY_TYPE)
        if isinstance(raw_codes, dict):
            codes = list(raw_codes.keys())
        elif isinstance(raw_codes, (list, tuple)):
            codes = [str(x) for x in raw_codes]
    except Exception as exc:
        downloaded["failed"].append({"step": "refresh_code_list", "error": f"{type(exc).__name__}: {exc}"})

    downloaded["n_codes"] = len(codes)
    if not codes:
        downloaded["status"] = "NO_CODES"
        return downloaded

    key = "option_basic_info_all"
    basic_path = DATA_ROOT / "option_basic_info.parquet"
    if key not in done:
        try:
            basic = info.get_option_basic_info(list(codes), local_path=local_path, is_local=False)
            rec = _save_frame(_to_frame(basic), basic_path)
            rec.update({"batch_key": key, "ok": True})
            _append_ckpt(rec)
            downloaded["files"].append(rec)
        except Exception as exc:
            downloaded["failed"].append({"step": key, "error": f"{type(exc).__name__}: {exc}"})
            _append_ckpt({"batch_key": key, "ok": False, "error": type(exc).__name__})
    else:
        downloaded["skipped"] += 1

    for spec_name, fn in (
        ("option_mon_ctr_specs", info.get_option_mon_ctr_specs),
        ("option_std_ctr_specs", info.get_option_std_ctr_specs),
    ):
        if spec_name in done:
            downloaded["skipped"] += 1
            continue
        try:
            raw = fn(list(codes)[:50], local_path=local_path, is_local=False)
            rec = _save_frame(_to_frame(raw), DATA_ROOT / f"{spec_name}.parquet")
            rec.update({"batch_key": spec_name, "ok": True})
            _append_ckpt(rec)
            downloaded["files"].append(rec)
        except Exception as exc:
            downloaded["failed"].append({"step": spec_name, "error": f"{type(exc).__name__}: {exc}"})
            _append_ckpt({"batch_key": spec_name, "ok": False, "error": type(exc).__name__})

    etf_codes = [c.replace(".SH", ".SH").replace(".SZ", ".SZ") for c in ETF_UNDERLYINGS]
    etf_key = "etf_daily_kline"
    if etf_key not in done:
        try:
            kraw = md.query_kline(etf_codes, begin_date=20150101, end_date=20260724, period=period)
            rec = _save_frame(_to_frame(kraw), DATA_ROOT / "etf_daily.parquet")
            rec.update({"batch_key": etf_key, "ok": True})
            _append_ckpt(rec)
            downloaded["files"].append(rec)
        except Exception as exc:
            downloaded["failed"].append({"step": etf_key, "error": f"{type(exc).__name__}: {exc}"})
            _append_ckpt({"batch_key": etf_key, "ok": False, "error": type(exc).__name__})
    else:
        downloaded["skipped"] += 1

    # 日 K：分批 resume。先拉有限批次验证覆盖，不先拉全市场高频。
    max_batches = 40
    batches = [codes[i : i + 10] for i in range(0, min(len(codes), max_batches * 10), 10)]
    downloaded["kline_code_cap"] = min(len(codes), max_batches * 10)
    kdir = DATA_ROOT / "option_kline_day"
    kdir.mkdir(parents=True, exist_ok=True)
    for i, batch in enumerate(batches):
        bkey = f"opt_kline_day_{i:04d}"
        if bkey in done:
            downloaded["skipped"] += 1
            continue
        try:
            kraw = md.query_kline(list(batch), begin_date=20150101, end_date=20260724, period=period)
            rec = _save_frame(_to_frame(kraw), kdir / f"part-{i:04d}.parquet")
            rec.update({"batch_key": bkey, "ok": True, "n_codes": len(batch)})
            _append_ckpt(rec)
            downloaded["files"].append(rec)
        except Exception as exc:
            downloaded["failed"].append({"step": bkey, "error": f"{type(exc).__name__}: {exc}"})
            _append_ckpt({"batch_key": bkey, "ok": False, "error": type(exc).__name__})
            time.sleep(1.0)
    downloaded["status"] = "DONE"
    downloaded["n_kline_parts"] = len(list(kdir.glob("part-*.parquet")))
    return downloaded


def run_sdk(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and SDK_JSON.is_file():
        existing = load_json(SDK_JSON)
        if existing.get("status") in {"DONE", "SKIP_NO_LOGIN", "SKIP_NO_OPTION_DATA"}:
            return existing
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    inspect_payload = None
    inspect_error = None
    try:
        inspect_payload = _inspect_sdk()
    except Exception as exc:
        inspect_error = f"{type(exc).__name__}: {exc}"
    ad = None
    login_meta: dict[str, Any] = {}
    login_status = "NOT_ATTEMPTED"
    try:
        ad, login_meta, login_status = _login()
    except Exception as exc:
        login_status = f"LOGIN_FAIL:{type(exc).__name__}"
        login_meta = {"error": str(exc), "redacted": True}

    sample: dict[str, Any] = {}
    download: dict[str, Any] = {}
    skip_reason = None
    option_available = False
    local_path = str((DATA_ROOT / "ad_local").resolve())
    Path(local_path).mkdir(parents=True, exist_ok=True)

    if ad is None:
        skip_reason = login_status
        status = "SKIP_NO_LOGIN"
    else:
        sample = _small_sample(ad, local_path)
        if not sample.get("ok"):
            skip_reason = sample.get("reason") or "SAMPLE_FAIL"
            status = "SKIP_NO_OPTION_DATA"
        else:
            download = _resume_download(ad, local_path, sample)
            n_rows = sum(int(f.get("rows") or 0) for f in download.get("files") or [])
            option_available = bool(sample.get("n_option_codes")) and (
                n_rows > 0 or bool(sample.get("basic_rows")) or bool(sample.get("kline_rows"))
            )
            status = "DONE" if option_available else "SKIP_NO_OPTION_DATA"
            if not option_available:
                skip_reason = "DOWNLOAD_EMPTY"

    payload = {
        "status": status,
        "inspect": inspect_payload,
        "inspect_error": inspect_error,
        "login_status": login_status,
        "login_meta": redact(login_meta),
        "sample": redact(sample),
        "download": redact(download),
        "option_available": option_available,
        "skip_reason": skip_reason,
        "data_root": str(DATA_ROOT),
        "no_secrets_written": True,
        "note": "无期权且无法下载 → 期权组 SKIP，不填补、不伪造。",
    }
    dump_json(SDK_JSON, payload)
    lines = [
        "# Round 15 · AmazingData SDK",
        "",
        f"status=**{status}**。login=`{login_status}`。option_available=`{option_available}`。",
        f"SDK version=`{(inspect_payload or {}).get('version')}`。",
        f"小样本 n_option_codes=`{sample.get('n_option_codes')}` reason=`{sample.get('reason')}`。",
        f"skip_reason=`{skip_reason}`。",
        "",
        "login kwargs 已打码。不写 password/token。签名来自已安装包，不抄手册 PDF。",
        "",
    ]
    write_md(SDK_MD, "\n".join(lines))
    return payload


def load_sdk() -> dict[str, Any]:
    return load_json(SDK_JSON)


def require_sdk() -> dict[str, Any]:
    payload = load_sdk()
    if payload.get("status") not in {"DONE", "SKIP_NO_LOGIN", "SKIP_NO_OPTION_DATA"}:
        raise RuntimeError("SDK_STAGE_MISSING STOP")
    return payload


def option_data_available() -> bool:
    return bool(load_sdk().get("option_available"))


def audit_option_coverage() -> dict[str, Any]:
    """freeze 前的覆盖/PIT 审计。历史链不够则诚实 SKIP，不填补。"""
    basic_p = DATA_ROOT / "option_basic_info.parquet"
    kdir = DATA_ROOT / "option_kline_day"
    etf_p = DATA_ROOT / "etf_daily.parquet"
    out: dict[str, Any] = {
        "has_basic": basic_p.is_file(),
        "has_etf": etf_p.is_file(),
        "n_kline_rows": 0,
        "n_kline_days": 0,
        "kline_min": None,
        "kline_max": None,
        "n_basic": 0,
        "has_strike": False,
        "has_cp": False,
        "has_expiry": False,
        "hist_attempted": False,
        "hist_n": None,
        "hist_error": None,
        "stable_chain": False,
        "bilateral_25d": False,
        "has_10d_put": False,
        "chain_dense": False,
        "reason": None,
    }
    if basic_p.is_file():
        basic = pd.read_parquet(basic_p)
        out["n_basic"] = int(len(basic))
        cols = {c.upper() for c in basic.columns}
        out["has_strike"] = "EXERCISE_PRICE" in cols
        out["has_cp"] = "CONTRACT_TYPE" in cols
        out["has_expiry"] = "EXPIRY_DATE" in cols
    frames = []
    if kdir.is_dir():
        for p in sorted(kdir.glob("part-*.parquet")):
            df = pd.read_parquet(p)
            if not df.empty:
                frames.append(df)
    if frames:
        k = pd.concat(frames, ignore_index=True)
        out["n_kline_rows"] = int(len(k))
        tcol = "kline_time" if "kline_time" in k.columns else None
        if tcol:
            ts = pd.to_datetime(k[tcol], errors="coerce")
            out["n_kline_days"] = int(ts.dt.normalize().nunique())
            out["kline_min"] = str(ts.min().date()) if ts.notna().any() else None
            out["kline_max"] = str(ts.max().date()) if ts.notna().any() else None
    # 当前代码列表几乎都是近月。尝试历史代码。
    if out["n_kline_days"] < 126:
        out["hist_attempted"] = True
        try:
            import multiprocessing as mp

            def _hist_job(q):
                try:
                    ad, _meta, st = _login()
                    if ad is None:
                        q.put({"hist_login": st, "hist_error": "NO_AD"})
                        return
                    base = ad.BaseData()
                    hist = base.get_hist_code_list(
                        security_type=OPTION_SECURITY_TYPE,
                        start_date=20180101,
                        end_date=20181231,
                        local_path=str(DATA_ROOT / "ad_local"),
                    )
                    if isinstance(hist, dict):
                        n = len(hist)
                        frame = pd.DataFrame({"code": list(hist.keys())})
                    elif isinstance(hist, (list, tuple)):
                        n = len(hist)
                        frame = pd.DataFrame({"code": list(hist)})
                    else:
                        frame = _to_frame(hist)
                        n = int(len(frame))
                    rec = _save_frame(frame, DATA_ROOT / "option_hist_code_list.parquet")
                    q.put({"hist_login": st, "hist_n": n, "hist_file": rec})
                except Exception as exc:
                    q.put({"hist_error": f"{type(exc).__name__}: {exc}"})

            q: mp.Queue = mp.Queue()
            proc = mp.Process(target=_hist_job, args=(q,))
            proc.start()
            proc.join(90)
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
                out["hist_error"] = "TIMEOUT_90s_get_hist_code_list"
            elif not q.empty():
                out.update(q.get())
            else:
                out["hist_error"] = "HIST_NO_RESULT"
        except Exception as exc:
            out["hist_error"] = f"{type(exc).__name__}: {exc}"
    out["stable_chain"] = bool(
        out["has_basic"]
        and out["has_strike"]
        and out["has_cp"]
        and out["n_kline_days"] >= 126
        and out["n_kline_rows"] >= 500
    )
    # 25D 双边 / 10D put 需要足够密的链 + IV。日数不够则否。
    out["bilateral_25d"] = bool(out["stable_chain"] and out["has_cp"])
    out["has_10d_put"] = bool(out["stable_chain"] and out["has_cp"])
    out["chain_dense"] = bool(out["stable_chain"] and out["n_basic"] >= 50)
    if not out["stable_chain"]:
        out["reason"] = (
            f"NO_STABLE_HISTORICAL_CHAIN n_days={out['n_kline_days']} "
            f"range={out['kline_min']}..{out['kline_max']} hist_n={out['hist_n']} hist_error={out['hist_error']}"
        )
    return out


__all__ = [
    "DATA_ROOT",
    "_login",
    "_save_frame",
    "_to_frame",
    "audit_option_coverage",
    "load_sdk",
    "option_data_available",
    "require_sdk",
    "run_sdk",
]
