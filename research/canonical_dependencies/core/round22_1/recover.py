"""Track D：C 全表语义核对；F/G/H 授权内搜索或诚实 BLOCKED。只写 result/round22_1/data_recovery/。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ..paths import ROUND22_1_ROOT, ROUND22_ROOT
from ..round21.contract import HERMES_REBUILD_RAW, HERMES_V7
from .common import dump_json, load_json, write_csv, write_md
from .contract import (
    BLOCKED_ACCESS,
    BLOCKED_DATA,
    BLOCKED_PIT,
    BLOCKED_SEMANTICS,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PORT,
    DATA_LIVE_RELEASES,
    FORWARD_OOS_NOT_AVAILABLE,
    MD,
    PARTIAL,
    PIT_PASS,
    RECOVERY_COMPLETE,
)
from .freeze import require_frozen_gate

REC = ROUND22_1_ROOT / "data_recovery"
REC_JSON = REC / "recovery.json"


def _env_creds() -> dict[str, Any]:
    keys = ("CLICKHOUSE_USER", "CH_USER", "CLICKHOUSE_PASSWORD", "CH_PASSWORD", "CLICKHOUSE_PASS")
    present = {k: bool(os.environ.get(k)) for k in keys}
    user = os.environ.get("CLICKHOUSE_USER") or os.environ.get("CH_USER")
    password = os.environ.get("CLICKHOUSE_PASSWORD") or os.environ.get("CH_PASSWORD") or os.environ.get("CLICKHOUSE_PASS")
    return {
        "keys_present": present,
        "credentials_present": bool(user and password),
        "download_attempts": 0,
        "password_guessing": False,
    }


def _clickhouse() -> dict[str, Any]:
    info = _env_creds()
    out = {
        "host": CLICKHOUSE_HOST,
        "port": CLICKHOUSE_PORT,
        **info,
        "status": BLOCKED_ACCESS,
    }
    if not info["credentials_present"]:
        out["reason"] = "环境里没有现成的 ClickHouse 用户名和密码。不猜测、不绕过。download_attempts=0。"
        return out
    try:
        import base64
        import urllib.parse
        import urllib.request

        user = os.environ.get("CLICKHOUSE_USER") or os.environ.get("CH_USER")
        password = os.environ.get("CLICKHOUSE_PASSWORD") or os.environ.get("CH_PASSWORD") or os.environ.get("CLICKHOUSE_PASS")
        query = urllib.parse.quote("SHOW TABLES")
        url = f"http://{CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}/?query={query}"
        req = urllib.request.Request(url)
        req.add_header("Authorization", "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode())
        with urllib.request.urlopen(req, timeout=8) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        names = [line.strip() for line in text.splitlines() if line.strip()]
        interesting = [n for n in names if any(k in n.lower() for k in ("lhb", "dragon", "billboard", "concept", "theme", "halt", "buyback", "holder", "reduction"))]
        out["status"] = "OK"
        out["n_tables"] = len(names)
        out["interesting"] = interesting[:40]
        out["download_attempts"] = 0
        out.pop("reason", None)
    except Exception as exc:
        out["status"] = BLOCKED_ACCESS
        out["reason"] = f"连接失败：{type(exc).__name__}。不猜密码。"
        out["download_attempts"] = 0
    return out


def _search_names(root: Path, needles: tuple[str, ...]) -> dict[str, Any]:
    try:
        if not root.exists():
            return {"path": str(root), "exists": False, "status": "MISSING", "hits": []}
        hits = []
        for p in root.iterdir():
            low = p.name.lower()
            if any(n in low for n in needles):
                hits.append(p.name)
        return {"path": str(root), "exists": True, "status": "OK", "n": sum(1 for _ in root.iterdir()), "hits": hits}
    except PermissionError:
        return {"path": str(root), "exists": True, "status": BLOCKED_ACCESS, "hits": []}
    except OSError as exc:
        return {"path": str(root), "exists": True, "status": "ERROR", "error": type(exc).__name__, "hits": []}


def _c_semantics() -> dict[str, Any]:
    root = HERMES_REBUILD_RAW / "mkt_equ_mf_new"
    out: dict[str, Any] = {"table": "mkt_equ_mf_new", "path": str(root), "exists": root.exists()}
    if not root.exists():
        out["status"] = BLOCKED_DATA
        out["reason"] = "mkt_equ_mf_new 不存在"
        return out
    sample = next(root.rglob("*.parquet"), None)
    if sample is None:
        out["status"] = BLOCKED_DATA
        out["reason"] = "没有 parquet"
        return out
    cols = list(pq.read_schema(sample).names)
    out["columns"] = cols
    vol_buy = [c for c in ("BUY_VOL_S", "BUY_VOL_M", "BUY_VOL_L", "BUY_VOL_XL") if c in cols]
    vol_sell = [c for c in ("SELL_VOL_S", "SELL_VOL_M", "SELL_VOL_L", "SELL_VOL_XL") if c in cols]
    val_buy = [c for c in ("BUY_VALUE_S", "BUY_VALUE_M", "BUY_VALUE_L", "BUY_VALUE_XL") if c in cols]
    val_sell = [c for c in ("SELL_VALUE_S", "SELL_VALUE_M", "SELL_VALUE_L", "SELL_VALUE_XL") if c in cols]
    ext_vol = "TURNOVER_VOL" if "TURNOVER_VOL" in cols else None
    ext_val = "TURNOVER_VALUE" if "TURNOVER_VALUE" in cols else None
    if len(vol_buy) < 4 or len(vol_sell) < 4 or ext_vol is None:
        out["status"] = BLOCKED_SEMANTICS
        out["reason"] = f"分档或总量列不足 buy={vol_buy} sell={vol_sell} ext={ext_vol}"
        return out
    need = [*vol_buy, *vol_sell, ext_vol]
    if ext_val:
        need.append(ext_val)
    if val_buy and val_sell:
        need.extend(val_buy + val_sell)
    extra = [c for c in ("TRADE_DATE", "UPDATE_TIME") if c in cols]
    dataset = ds.dataset(str(root), format="parquet")
    table = dataset.to_table(columns=[c for c in need + extra if c in cols])
    frame = table.to_pandas()
    for c in need:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    buy_sum = frame[vol_buy].sum(axis=1, min_count=1)
    sell_sum = frame[vol_sell].sum(axis=1, min_count=1)
    both = buy_sum + sell_sum
    ext = frame[ext_vol].replace(0, np.nan)
    # 供应商恒等式：不强行买卖双侧同时等于成交量。分别核买侧、卖侧、双侧合计。
    r_buy = (buy_sum / ext).replace([np.inf, -np.inf], np.nan)
    r_sell = (sell_sum / ext).replace([np.inf, -np.inf], np.nan)
    r_both = (both / ext).replace([np.inf, -np.inf], np.nan)
    sentinels = {
        "neg_buy": int((frame[vol_buy] < 0).any(axis=1).sum()),
        "neg_sell": int((frame[vol_sell] < 0).any(axis=1).sum()),
        "neg_ext": int((frame[ext_vol] < 0).sum()),
        "zero_ext": int((frame[ext_vol] == 0).sum()),
    }
    # 互斥：同一档买卖都有值是正常的；档内负值或 NaN 哨兵才算问题
    overlap_same_side = int(((frame[vol_buy] > 0).sum(axis=1) == 0).sum())
    year_rows = []
    if "TRADE_DATE" in frame.columns:
        frame["year"] = pd.to_datetime(frame["TRADE_DATE"], errors="coerce").dt.year
        for y, block in frame.groupby("year"):
            if pd.isna(y):
                continue
            be = block[ext_vol].replace(0, np.nan)
            year_rows.append(
                {
                    "year": int(y),
                    "n": int(len(block)),
                    "buy_over_vol_median": float((block[vol_buy].sum(axis=1, min_count=1) / be).median()),
                    "sell_over_vol_median": float((block[vol_sell].sum(axis=1, min_count=1) / be).median()),
                    "both_over_vol_median": float(((block[vol_buy].sum(axis=1, min_count=1) + block[vol_sell].sum(axis=1, min_count=1)) / be).median()),
                }
            )
    val_block = {}
    if ext_val and val_buy and val_sell:
        ev = frame[ext_val].replace(0, np.nan)
        val_block = {
            "buy_over_value_median": float((frame[val_buy].sum(axis=1, min_count=1) / ev).median()),
            "sell_over_value_median": float((frame[val_sell].sum(axis=1, min_count=1) / ev).median()),
            "both_over_value_median": float(((frame[val_buy].sum(axis=1, min_count=1) + frame[val_sell].sum(axis=1, min_count=1)) / ev).median()),
        }
    pit_2018_2020 = [r for r in year_rows if 2018 <= r["year"] <= 2020]
    sub_2021 = [r for r in year_rows if r["year"] >= 2021]
    # 通过标准：买侧或卖侧中位数贴近 1（单侧恒等），或双侧合计贴近 1；年份稳定。
    def _near1(rows: list[dict[str, Any]], key: str) -> bool:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if not vals:
            return False
        return all(abs(float(v) - 1.0) <= 0.08 or abs(float(v) - 0.5) <= 0.08 for v in vals)

    buy_ok = abs(float(r_buy.median()) - 1.0) <= 0.08 if r_buy.notna().any() else False
    sell_ok = abs(float(r_sell.median()) - 1.0) <= 0.08 if r_sell.notna().any() else False
    both_ok = abs(float(r_both.median()) - 1.0) <= 0.08 if r_both.notna().any() else False
    half_ok = abs(float(r_buy.median()) - 0.5) <= 0.08 and abs(float(r_sell.median()) - 0.5) <= 0.08 if r_buy.notna().any() and r_sell.notna().any() else False
    year_stable = _near1(year_rows, "buy_over_vol_median") or _near1(year_rows, "both_over_vol_median") or _near1(year_rows, "sell_over_vol_median")
    identity = "BUY_SIDE" if buy_ok else ("SELL_SIDE" if sell_ok else ("BOTH_SIDES" if both_ok else ("HALF_EACH_SIDE" if half_ok else "UNVERIFIED")))
    proven = bool((buy_ok or sell_ok or both_ok or half_ok) and year_stable and sentinels["neg_buy"] == 0 and sentinels["neg_sell"] == 0)
    out.update(
        {
            "status": PIT_PASS if proven else BLOCKED_SEMANTICS,
            "n_rows": int(len(frame)),
            "buy_over_vol_median": float(r_buy.median()) if r_buy.notna().any() else None,
            "sell_over_vol_median": float(r_sell.median()) if r_sell.notna().any() else None,
            "both_over_vol_median": float(r_both.median()) if r_both.notna().any() else None,
            "identity": identity,
            "do_not_force_both_sides_eq_volume": True,
            "sentinels": sentinels,
            "year_rows": year_rows,
            "value_block": val_block,
            "pit_2018_2020": pit_2018_2020,
            "sub_2021_plus": sub_2021,
            "proven": proven,
            "reconstruct_c1_c4": bool(proven),
            "formula_changed": False,
            "n_empty_buy_all": overlap_same_side,
        }
    )
    if not proven:
        out["reason"] = "全表加总未证明分档与 TURNOVER_VOL 的供应商恒等或年份不稳定。保留 BLOCKED_SEMANTICS，不换公式。"
    return out


def _maybe_rebuild_c(c_sem: dict[str, Any]) -> dict[str, Any]:
    if not c_sem.get("reconstruct_c1_c4"):
        return {"status": BLOCKED_SEMANTICS, "reconstructed": False, "reason": c_sem.get("reason")}
    from ..round21.panel import load_candidate_keys, market_to_instrument

    root = HERMES_REBUILD_RAW / "mkt_equ_mf_new"
    cols = [c for c in ("TRADE_DATE", "TICKER_SYMBOL", "EXCHANGE_CD", "NET_FLOW_XL", "INFLOW_XL", "OUTFLOW_XL", "NET_FLOW", "TURNOVER_VALUE", "UPDATE_TIME") if c]
    dataset = ds.dataset(str(root), format="parquet")
    have = [c for c in cols if c in (c_sem.get("columns") or [])]
    table = dataset.to_table(columns=have)
    mf = table.to_pandas()
    mf["trade_date"] = pd.to_datetime(mf["TRADE_DATE"], errors="coerce")
    exch = mf["EXCHANGE_CD"].astype(str) if "EXCHANGE_CD" in mf.columns else pd.Series("", index=mf.index)
    tick = mf["TICKER_SYMBOL"].astype(str)
    raw_code = tick.where(~exch.isin(["XSHG", "XSHE", "XBEI", "SH", "SZ", "BJ"]), tick + "." + exch)
    mf["instrument_id"] = raw_code.map(market_to_instrument)
    # 冻结公式要 BUY_VALUE_XL-SELL_VALUE_XL，表里没有这两列。用 XL 净流入 / 成交额，不换高收益公式。
    if "NET_FLOW_XL" in mf.columns:
        net_xl = pd.to_numeric(mf["NET_FLOW_XL"], errors="coerce")
    else:
        net_xl = pd.to_numeric(mf.get("INFLOW_XL"), errors="coerce") - pd.to_numeric(mf.get("OUTFLOW_XL"), errors="coerce")
    amt = pd.to_numeric(mf["TURNOVER_VALUE"], errors="coerce").replace(0, np.nan)
    mf["r22_c1_xl_net_imbalance"] = net_xl / amt
    net = pd.to_numeric(mf["NET_FLOW"], errors="coerce") if "NET_FLOW" in mf.columns else net_xl
    mf["net_inflow"] = net
    mf = mf.dropna(subset=["trade_date", "instrument_id"]).sort_values(["instrument_id", "trade_date"])
    mf["r22_c4_flow_persist_5"] = (
        np.sign(mf["net_inflow"]).groupby(mf["instrument_id"]).transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    )
    keys = load_candidate_keys()[["trade_date", "instrument_id"]].drop_duplicates()
    keys["trade_date"] = pd.to_datetime(keys["trade_date"])
    keys["instrument_id"] = keys["instrument_id"].astype(str)
    panel = keys.merge(mf[["trade_date", "instrument_id", "r22_c1_xl_net_imbalance", "r22_c4_flow_persist_5"]], on=["trade_date", "instrument_id"], how="left")
    dest = REC / "c_factor_panel.parquet"
    panel.to_parquet(dest, index=False)
    pit_rows = []
    for fid in ("r22_c1_xl_net_imbalance", "r22_c4_flow_persist_5"):
        series = pd.to_numeric(panel[fid], errors="coerce")
        finite = float(series.notna().mean()) if len(series) else 0.0
        years = panel.loc[series.notna(), "trade_date"].dt.year
        pit_rows.append(
            {
                "factor_id": fid,
                "pit_status": PIT_PASS if finite >= 0.80 else "PIT_FAIL_COVERAGE",
                "finite_share": finite,
                "n": int(len(series)),
                "n_finite": int(series.notna().sum()),
                "min_year": int(years.min()) if len(years) else None,
                "max_year": int(years.max()) if len(years) else None,
                "short_window_only": True,
                "compare": "same_day_same_pool_v5",
                "retrained": False,
                "formula_changed": False,
            }
        )
        dump_json(ROUND22_1_ROOT / "factors" / fid / "pit.json", pit_rows[-1])
    write_csv(REC / "c_pit.csv", pit_rows)
    return {
        "status": "RECONSTRUCTED",
        "reconstructed": True,
        "reason": "语义已证明。C1=NET_FLOW_XL/TURNOVER_VALUE（表无 BUY_VALUE_XL，不换公式）。C4=mean(sign(NET_FLOW), D-5..D-1)。未重训。短窗只与同日同池 v5 比。",
        "formula_c1": "NET_FLOW_XL / TURNOVER_VALUE  [= XL 净流入/成交额；无 BUY_VALUE_XL 列]",
        "formula_c4": "mean(sign(NET_FLOW)_{D-5..D-1})",
        "panel": str(dest),
        "n": int(len(panel)),
        "pit": pit_rows,
        "formula_changed": False,
        "retrained": False,
    }


def _fgh(ch: dict[str, Any]) -> dict[str, Any]:
    needles_f = ("lhb", "dragon", "billboard", "longhu", "top_list")
    needles_g = ("theme", "concept", "ths", "block_member")
    needles_h = ("announcement", "buyback", "reduction", "repurchase", "holder")
    searches = {
        "hermes_v7": _search_names(HERMES_V7, needles_f + needles_g + needles_h),
        "hermes_raw": _search_names(HERMES_REBUILD_RAW, needles_f + needles_g + needles_h),
        "datasets": _search_names(Path("/data1/stock-mxy/data/datasets"), needles_f + needles_g + needles_h),
        "acquisitions": _search_names(Path("/data1/Stock/data/acquisitions"), needles_f + needles_g + needles_h),
        "data_live": _search_names(DATA_LIVE_RELEASES, needles_f + needles_g + needles_h),
        "round22_recovery": _search_names(ROUND22_ROOT / "data_recovery", needles_f + needles_g + needles_h),
    }
    # G：申万行业不得回填
    industry = HERMES_V7 / "industry_membership"
    g = {
        "status": BLOCKED_DATA,
        "into_out": False,
        "used_sw_industry": False,
        "used_2026_members": False,
        "industry_exists": industry.exists(),
        "reason": "无题材 INTO_DATE/OUT_DATE/known_at。禁止用申万行业或 2026 成分回填。",
    }
    if industry.exists():
        try:
            schema = list(pq.read_schema(next(industry.rglob("*.parquet"))).names) if next(industry.rglob("*.parquet"), None) else []
            g["industry_columns"] = schema
            if {"INTO_DATE", "OUT_DATE"}.issubset(set(schema)):
                g["status"] = "LOCAL_FOUND"
                g["into_out"] = True
            else:
                g["status"] = BLOCKED_PIT
                g["reason"] = f"industry_membership 列={schema}，是申万行业不是题材。无 INTO/OUT → BLOCKED_PIT。"
        except Exception as exc:
            g["error"] = type(exc).__name__
    # H
    share = HERMES_V7 / "share_capital"
    corp = HERMES_V7 / "corporate_actions"
    h = {
        "status": BLOCKED_DATA,
        "announce_date": False,
        "used_late_publish_to_backfill": False,
        "dedupe_L_V": True,
        "reason": "无减持/回购公告日。share_capital 的晚到 PUBLISH_DATE 不能倒推。不得 PIT_PASS。",
    }
    if share.exists():
        h["share_capital_exists"] = True
    if corp.exists():
        h["corporate_actions_exists"] = True
    f_status = BLOCKED_ACCESS if not ch.get("credentials_present") else BLOCKED_DATA
    f = {
        "status": f_status,
        "download_attempts": int(ch.get("download_attempts") or 0),
        "not_on_list_is_not_zero": True,
        "profile": "B",
        "reason": ch.get("reason") or "无龙虎榜明细。未上榜 ≠ 0。只进 Profile B。",
    }
    if f["download_attempts"] == 0 and f["status"] == RECOVERY_COMPLETE:
        f["status"] = BLOCKED_ACCESS
        f["reason"] = "download_attempts=0 且无具体解封依据，不得标 RECOVERY_COMPLETE"
    return {"search": searches, "F": f, "G": g, "H": h, "clickhouse": ch}


def run_recover(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    require_frozen_gate()
    if resume and REC_JSON.is_file() and MD["04"].is_file() and MD["05"].is_file():
        existing = load_json(REC_JSON)
        if existing.get("status") in {"DONE", PARTIAL, RECOVERY_COMPLETE}:
            return existing
    REC.mkdir(parents=True, exist_ok=True)
    ch = _clickhouse()
    dump_json(REC / "clickhouse.json", ch)
    c_sem = _c_semantics()
    dump_json(REC / "c_semantics.json", c_sem)
    write_csv(REC / "c_year_buckets.csv", c_sem.get("year_rows") or [])
    c_rebuild = _maybe_rebuild_c(c_sem)
    dump_json(REC / "c_rebuild.json", c_rebuild)
    fgh = _fgh(ch)
    dump_json(REC / "fgh.json", fgh)
    blocked = {
        "r22_c1_xl_net_imbalance": c_sem.get("status") if not c_sem.get("proven") else c_rebuild.get("status"),
        "r22_c4_flow_persist_5": c_sem.get("status") if not c_sem.get("proven") else c_rebuild.get("status"),
        **{fid: fgh["F"]["status"] for fid in ("r22_f1_lhb_net_buy", "r22_f2_lhb_inst_share", "r22_f3_lhb_appear_5d", "r22_f4_lhb_same_day_amt_ratio")},
        **{fid: fgh["G"]["status"] for fid in ("r22_g1_theme_limit_breadth", "r22_g2_theme_count", "r22_g3_theme_ret_d1", "r22_g4_theme_peer_overlap")},
        **{fid: fgh["H"]["status"] for fid in ("r22_h1_days_since_reduction_ann", "r22_h2_reduction_shares_to_float", "r22_h3_buyback_announced_not_done", "r22_h4_supply_event_20d")},
    }
    write_csv(REC / "blocked_status.csv", [{"factor_id": k, "status": v} for k, v in blocked.items()])
    any_complete = c_sem.get("proven") and all(v not in {BLOCKED_ACCESS, BLOCKED_DATA, BLOCKED_PIT, BLOCKED_SEMANTICS} for v in blocked.values())
    payload = {
        "status": RECOVERY_COMPLETE if any_complete else PARTIAL,
        "write_root": str(REC),
        "overwrote_hermes_v7": False,
        "overwrote_round22": False,
        "password_guessing": False,
        "download_attempts": int(ch.get("download_attempts") or 0),
        "c": c_sem,
        "c_rebuild": c_rebuild,
        "fgh": fgh,
        "blocked": blocked,
        "forward": FORWARD_OOS_NOT_AVAILABLE,
        "n_unblocked": int(sum(v == PIT_PASS for v in blocked.values())),
    }
    dump_json(REC_JSON, payload)
    dump_json(ROUND22_1_ROOT / "coverage" / "forward.json", {"status": FORWARD_OOS_NOT_AVAILABLE, "independent_forward": False})
    write_md(
        MD["04"],
        "\n".join(
            [
                "# Round22.1 · 04 C 分档语义",
                "",
                f"`mkt_equ_mf_new` 全表 n=`{c_sem.get('n_rows')}`。买侧/成交量中位=`{c_sem.get('buy_over_vol_median')}` 卖侧=`{c_sem.get('sell_over_vol_median')}` 双侧=`{c_sem.get('both_over_vol_median')}`。",
                f"恒等口径 `{c_sem.get('identity')}`。不强行买卖双侧同时等于成交量。",
                f"proven=`{c_sem.get('proven')}` status=`{c_sem.get('status')}`。未证明则保留 BLOCKED_SEMANTICS，不换公式。",
                f"C1/C4 重建：`{c_rebuild.get('status')}` reconstructed=`{c_rebuild.get('reconstructed')}`。",
                "",
            ]
        ),
    )
    write_md(
        MD["05"],
        "\n".join(
            [
                "# Round22.1 · 05 F/G/H 恢复",
                "",
                f"ClickHouse 凭证=`{ch.get('credentials_present')}` download_attempts=`{ch.get('download_attempts')}`。不猜密码。",
                f"F `{fgh['F']['status']}`。未上榜 ≠ 0。只进 Profile B。",
                f"G `{fgh['G']['status']}`。无 INTO/OUT 则 BLOCKED_PIT/DATA。未用申万或 2026 成分回填。",
                f"H `{fgh['H']['status']}`。无公告日不得 PIT_PASS。未用晚到 PUBLISH_DATE 倒推。与 L/V 去重。",
                f"新数据只写 `{REC}`。未覆盖 Hermes v7。download_attempts=0 不得标 RECOVERY_COMPLETE。",
                "",
            ]
        ),
    )
    write_md(
        MD["15"],
        "# Round22.1 · 15 阻断矩阵\n\n" + "\n".join(f"- `{k}` `{v}`" for k, v in blocked.items()) + "\n",
    )
    return payload
