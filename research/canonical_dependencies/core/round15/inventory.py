"""服务器实有数据盘点。不发明期权。不写 Stock。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import ROUND15_ROOT
from ..round12.contract import MINUTE_ROOT
from .common import dump_json, load_json, write_md

INV_JSON = ROUND15_ROOT / "inventory" / "inventory.json"
INV_MD = ROUND15_ROOT / "01_inventory.md"

OPTION_NAME_HINTS = ("option_", "期权", "implied_vol", "iv_surface", "etf_op", "opt_chain")
SCAN_ROOTS = (
    Path("/home/mxy/AmazingData"),
    Path("/home/mxy/AmazingData/data"),
    Path("/home/mxy/Stock/data"),
    Path("/data1/stock-mxy/data"),
    Path("/home/mxy/Stock/docs"),
    Path("/home/mxy/docs"),
)


def _minute_inventory() -> dict[str, Any]:
    parts = MINUTE_ROOT / "parts"
    years = {}
    for y in range(2017, 2027):
        yp = parts / f"year={y}"
        if yp.is_dir():
            files = list(yp.glob("*.parquet"))
            years[str(y)] = {
                "n_files": len(files),
                "bytes": int(sum(f.stat().st_size for f in files if f.is_file())),
                "full_market": y <= 2021 and len(files) > 1000,
            }
        else:
            years[str(y)] = {"n_files": 0, "bytes": 0, "full_market": False}
    first_bar = None
    has_0925 = False
    sample = None
    probe = parts / "year=2018" / "sz000672.parquet"
    if not probe.is_file():
        for cand in (parts / "year=2018").glob("*.parquet") if (parts / "year=2018").is_dir() else []:
            probe = cand
            break
    if probe.is_file():
        df = pd.read_parquet(probe)
        sample = {"columns": list(df.columns), "n": int(len(df)), "file": probe.name}
        time_col = "bar_time" if "bar_time" in df.columns else None
        if time_col:
            hm = pd.to_datetime(df[time_col]).dt.strftime("%H:%M")
            first_bar = str(hm.min()) if len(hm) else None
            has_0925 = bool((hm == "09:25").any())
            sample["time_head"] = {str(k): int(v) for k, v in hm.value_counts().head(8).items()}
    return {
        "root": str(MINUTE_ROOT),
        "years": years,
        "first_bar": first_bar or "UNKNOWN",
        "has_0925_auction": has_0925,
        "common_minute_years": [y for y in range(2017, 2022) if years.get(str(y), {}).get("full_market")],
        "coverage_note": "2017–2021 全市场分钟；2022+ 几乎无。首 bar 预期 09:30。无 09:25 则 SKIP_AUCTION_SIGNAL。",
        "sample": sample,
    }


def _iter_limited(root: Path, max_depth: int = 3):
    skip = {".git", "node_modules", "__pycache__", "runs", "logs", ".cache"}
    if not root.exists():
        return
    yield root
    frontier = [(root, 0)]
    while frontier:
        cur, depth = frontier.pop()
        if depth >= max_depth:
            continue
        try:
            children = list(cur.iterdir())
        except (PermissionError, OSError):
            continue
        for p in children:
            if p.name in skip:
                continue
            yield p
            if p.is_dir():
                frontier.append((p, depth + 1))


def _scan_option_dirs() -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    manuals: list[str] = []
    known_pdf = Path("/home/mxy/AmazingData/AmazingData开发手册.pdf")
    if known_pdf.is_file():
        manuals.append(str(known_pdf))
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for p in _iter_limited(root, 3 if "stock-mxy" in str(root) or "Stock/data" in str(root) else 4):
            name = p.name.lower()
            if p.is_file() and p.suffix.lower() == ".pdf" and ("amazing" in name or "开发手册" in p.name):
                if str(p) not in manuals:
                    manuals.append(str(p))
            if not any(h in name for h in OPTION_NAME_HINTS):
                continue
            if p.is_dir() or p.suffix.lower() in {".parquet", ".csv", ".h5", ".hdf5", ".pkl"}:
                try:
                    sz = int(p.stat().st_size) if p.is_file() else None
                except OSError:
                    sz = None
                hits.append({"path": str(p), "is_dir": p.is_dir(), "bytes": sz})
            if len(hits) >= 80:
                break
    return {
        "hits": hits,
        "n_hits": len(hits),
        "manuals": manuals,
        "has_local_option_library": bool(hits),
        "note": "数据清单无期权域。此处只记录路径命中，不把 PC ratio 叫 RN kurtosis。",
    }


def run_inventory(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and INV_JSON.is_file():
        existing = load_json(INV_JSON)
        if existing.get("status") == "DONE":
            return existing
    minute = _minute_inventory()
    options = _scan_option_dirs()
    payload = {
        "status": "DONE",
        "minute": minute,
        "options_local": options,
        "forward_inbox": "incomplete_default_FORWARD_OOS_NOT_AVAILABLE",
        "snapshot_end": "2026-07-24",
        "no_invent": True,
    }
    dump_json(INV_JSON, payload)
    lines = [
        "# Round 15 · 数据盘点",
        "",
        f"本地期权命中数 = **{options['n_hits']}**。has_local_option_library=`{options['has_local_option_library']}`。",
        f"分钟根 = `{minute['root']}`。首 bar=`{minute['first_bar']}`。09:25=`{minute['has_0925_auction']}`。",
        f"共同全市场分钟年 = {minute['common_minute_years']}。",
        "",
        f"手册 PDF = {options['manuals'] or '未找到'}。SDK 用法以已安装包签名为准，不抄手册笔误。",
        "",
        "无期权库不填补。下载与否见 sdk stage。",
        "",
    ]
    write_md(INV_MD, "\n".join(lines))
    return payload


def load_inventory() -> dict[str, Any]:
    return load_json(INV_JSON)


def require_inventory() -> dict[str, Any]:
    inv = load_inventory()
    if inv.get("status") != "DONE":
        raise RuntimeError("INVENTORY_MISSING STOP")
    return inv


__all__ = ["load_inventory", "require_inventory", "run_inventory"]
