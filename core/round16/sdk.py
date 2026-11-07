"""AmazingData SDK 1.1.7 封装。独立 ad_local。不打印 password/token。禁止 pip upgrade。"""

from __future__ import annotations

import contextlib
import inspect
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import ROUND16_ROOT, assert_project_result_path
from .common import dump_json, file_sha256, redact
from .contract import ETF_UNDERLYINGS, KLINE_BEGIN, KLINE_END, OPTION_SECURITY_TYPE

RAW_ROOT = ROUND16_ROOT / "data" / "raw"
AD_LOCAL = RAW_ROOT / "ad_local"
SDK_JSON = ROUND16_ROOT / "sdk" / "sdk.json"


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


def login_kwargs() -> dict[str, Any]:
    """与低换手 amazingdata_login_kwargs 同一套 env；不把低换手 core 插入 sys.path。"""
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


class _RedactStream:
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


def inspect_sdk() -> dict[str, Any]:
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
            "doc_head": ((inspect.getdoc(fn) or "").splitlines() or [""])[0][:400],
        }

    period_members = []
    try:
        period_members = list(ad.constant.Period.__members__.keys())
    except Exception as exc:
        period_members = [f"inspect_fail:{type(exc).__name__}"]
    return {
        "package": getattr(ad, "__file__", None),
        "version": getattr(ad, "__version__", None),
        "login": _sig(ad, "login"),
        "BaseData": {
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
        "note": "签名来自已安装 AmazingData 1.1.7，不抄手册 PDF，不 pip upgrade。",
    }


def to_frame(raw: Any) -> pd.DataFrame:
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
        try:
            return pd.DataFrame(raw)
        except Exception:
            return pd.DataFrame()
    if isinstance(raw, (list, tuple)) and raw:
        if isinstance(raw[0], pd.DataFrame):
            return pd.concat([x for x in raw if isinstance(x, pd.DataFrame)], ignore_index=True)
        if isinstance(raw[0], dict):
            return pd.DataFrame(list(raw))
        return pd.DataFrame({"code": [str(x) for x in raw]})
    return pd.DataFrame()


def codes_from(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [str(x) for x in raw.keys()]
    if isinstance(raw, (list, tuple, set, pd.Index)):
        out = []
        for x in raw:
            if isinstance(x, dict):
                for k in ("code", "CODE", "instrument_id"):
                    if k in x:
                        out.append(str(x[k]))
                        break
            else:
                out.append(str(x))
        return out
    if isinstance(raw, pd.DataFrame) and not raw.empty:
        for c in ("code", "CODE", "MARKET_CODE"):
            if c in raw.columns:
                return [str(x) for x in raw[c].astype(str).tolist()]
        return [str(x) for x in raw.iloc[:, 0].astype(str).tolist()]
    return []


def save_frame(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_project_result_path(path)
    if df is None or df.empty:
        return {"path": str(path), "rows": 0, "sha256": None, "exists": False}
    df.to_parquet(path, index=False)
    return {"path": str(path), "rows": int(len(df)), "sha256": file_sha256(path), "exists": True}


def period_day(ad) -> Any:
    period = ad.constant.Period
    if hasattr(period, "day"):
        return period.day.value
    return 86400000


def classify_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    perm_keys = ("permission", "unauthorized", "no right", "no privilege", "无权", "权限", "forbidden", "not allowed")
    if any(k in text for k in perm_keys):
        return "PERMISSION_REQUIRED"
    if "timeout" in text:
        return "TIMEOUT"
    return f"{type(exc).__name__}"


def login():
    kw = login_kwargs()
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


_AD = None
_LOGIN_META: dict[str, Any] = {}
_LOGIN_STATUS = "NOT_ATTEMPTED"


def ensure_login():
    global _AD, _LOGIN_META, _LOGIN_STATUS
    if _AD is not None and _LOGIN_STATUS == "LOGGED_IN":
        return _AD, _LOGIN_META, _LOGIN_STATUS
    AD_LOCAL.mkdir(parents=True, exist_ok=True)
    try:
        _AD, _LOGIN_META, _LOGIN_STATUS = login()
    except Exception as exc:
        _AD = None
        _LOGIN_STATUS = f"LOGIN_FAIL:{type(exc).__name__}"
        _LOGIN_META = {"error": str(exc), "redacted": True}
    return _AD, _LOGIN_META, _LOGIN_STATUS


def local_path() -> str:
    AD_LOCAL.mkdir(parents=True, exist_ok=True)
    return str(AD_LOCAL.resolve())


def query_kline_safe(md, codes: list[str], *, begin: int, end: int, period, retries: int = 3) -> tuple[pd.DataFrame, dict[str, Any]]:
    last_err = None
    for i in range(retries):
        try:
            raw = md.query_kline(list(codes), begin_date=int(begin), end_date=int(end), period=period)
            df = to_frame(raw)
            return df, {"ok": True, "rows": int(len(df)), "n_codes": len(codes), "begin": begin, "end": end}
        except Exception as exc:
            last_err = exc
            kind = classify_error(exc)
            if kind == "PERMISSION_REQUIRED":
                return pd.DataFrame(), {
                    "ok": False,
                    "permission": True,
                    "error_class": kind,
                    "error": f"{type(exc).__name__}",
                }
            time.sleep(min(8.0, 1.0 * (i + 1)))
    return pd.DataFrame(), {
        "ok": False,
        "permission": False,
        "error_class": classify_error(last_err) if last_err else "EMPTY",
        "error": f"{type(last_err).__name__}" if last_err else "NONE",
        "retries": retries,
    }


def etf_underlyings() -> list[str]:
    return list(ETF_UNDERLYINGS)


__all__ = [
    "AD_LOCAL",
    "RAW_ROOT",
    "SDK_JSON",
    "codes_from",
    "classify_error",
    "ensure_login",
    "etf_underlyings",
    "inspect_sdk",
    "local_path",
    "login",
    "login_kwargs",
    "period_day",
    "query_kline_safe",
    "save_frame",
    "to_frame",
]
