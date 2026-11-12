"""构造 C 与 F/G/H。源缺失或 PIT 不成立时诚实阻断，不填 0。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND22_ROOT
from ..round21.panel import KEYS, load_candidate_keys
from ..round21_2.construct import _load_raw_table
from .common import dump_json, load_json, write_md
from .contract import (
    BLOCKED_ACCESS,
    BLOCKED_DATA,
    BLOCKED_PIT,
    BLOCKED_SEMANTICS,
    DUPLICATE_EXPRESSION,
    FACTOR_SPECS,
    MD,
    STRICT_PIT_SUBPERIOD_RESEARCH,
)

DATA = ROUND22_ROOT / "data"
PANEL_ABDE = DATA / "factor_panel_abde.parquet"
PANEL_PATH = DATA / "factor_panel.parquet"
META_PATH = ROUND22_ROOT / "construct" / "cfgh.json"
EQ_PATH = ROUND22_ROOT / "construct" / "formula_equivalence.csv"

FORMULA_DUPLICATES = {
    "r22_c2_ticket_size_innov": "r21_t1_ticket_vs_med20",
    "r22_c3_buy_sell_count_gap": "r21_g3_order_count_imbalance",
}


def _status_row(spec: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
    return {
        "factor_id": spec["factor_id"],
        "family": spec["family"],
        "profile": spec["profile"],
        "status": status,
        "reason": reason,
        "values_constructed": False,
        "formula": spec["formula"],
    }


def _bucket_verdict() -> tuple[str, str]:
    """分档必须能对照独立总量。对不上或没有独立总量，记 BLOCKED_SEMANTICS。"""
    from ..round21.contract import HERMES_REBUILD_RAW
    import pyarrow.parquet as pq

    root = HERMES_REBUILD_RAW / "mkt_equ_mf_new"
    if not root.exists():
        return BLOCKED_DATA, "mkt_equ_mf_new 不存在"
    sample = next(root.rglob("*.parquet"), None)
    if sample is None:
        return BLOCKED_DATA, "mkt_equ_mf_new 没有 parquet"
    cols = list(pq.read_schema(sample).names)
    buckets = [c for c in cols if any(k in c.upper() for k in ("_S", "_M", "_L", "_XL"))]
    external = [c for c in cols if c.upper() in {"VOLUME", "TURNOVER_VOL", "TOT_VOL", "DEAL_VOL", "TRADE_VOL"}]
    if len(buckets) < 4:
        return BLOCKED_DATA, f"分档列不足：{cols[:20]}"
    if not external:
        return BLOCKED_SEMANTICS, "S/M/L/XL 没有独立总量字段可对照，不能证明分档互斥。BLOCKED_SEMANTICS，不把重叠档位算成订单流 Alpha。"
    return "UNVERIFIED", f"发现外部总量 {external}，本轮未做全表加总核对，仍不把未验证分档放进全历史模型"


def _try_c(keys: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    del keys
    rows = []
    out = pd.DataFrame(columns=KEYS)
    for fid, legacy in FORMULA_DUPLICATES.items():
        spec = next(s for s in FACTOR_SPECS if s["factor_id"] == fid)
        rows.append({**_status_row(spec, DUPLICATE_EXPRESSION, f"公式与 {legacy} 相同，不注册为新 Alpha"), "duplicate_of": legacy})
    status, reason = _bucket_verdict()
    if status == "UNVERIFIED":
        status = BLOCKED_SEMANTICS
    for fid in ("r22_c1_xl_net_imbalance", "r22_c4_flow_persist_5"):
        spec = next(s for s in FACTOR_SPECS if s["factor_id"] == fid)
        rows.append(_status_row(spec, status, reason))
    _ = _load_raw_table
    return out, rows


def run_construct_cfgh(*, resume: bool = True, jobs: int = 4, **_: Any) -> dict[str, Any]:
    del jobs
    if resume and PANEL_PATH.is_file() and META_PATH.is_file():
        existing = load_json(META_PATH)
        if existing.get("status") in {"DONE", "PARTIAL"}:
            return existing
    if not PANEL_ABDE.is_file():
        raise RuntimeError("ABDE panel missing")
    rec = load_json(ROUND22_ROOT / "data_recovery" / "recovery.json")
    classified = rec.get("classified") or {}
    keys = load_candidate_keys()[KEYS].drop_duplicates()
    keys["trade_date"] = pd.to_datetime(keys["trade_date"])
    keys["instrument_id"] = keys["instrument_id"].astype(str)
    c_panel, rows = _try_c(keys)
    ch = (rec.get("clickhouse") or {}).get("status")
    f_status = classified.get("dragon_tiger_status") or BLOCKED_DATA
    if f_status == "LOCAL_FOUND":
        f_status = BLOCKED_DATA
        f_reason = "本地命中表名但没有可证明的上榜明细列，不把未上榜填成 0"
    elif ch == BLOCKED_ACCESS or f_status == BLOCKED_ACCESS:
        f_status = BLOCKED_ACCESS
        f_reason = "龙虎榜无本地可用明细，且 ClickHouse 127.0.0.1:18123 没有现成凭证或连接失败。不绕过权限。"
    else:
        f_status = BLOCKED_DATA
        f_reason = "未发现龙虎榜历史明细"
    for spec in FACTOR_SPECS:
        if spec["family"] != "F":
            continue
        rows.append(_status_row(spec, f_status, f_reason + " 只允许 Profile B。"))
    g_status = classified.get("theme_status") or BLOCKED_DATA
    g_reason = "没有历史 INTO_DATE/OUT_DATE，禁止用当前成分回填 2018" if g_status == BLOCKED_PIT else "未发现可 PIT 的题材成分表"
    if g_status not in {BLOCKED_PIT, BLOCKED_DATA, BLOCKED_ACCESS}:
        g_status = BLOCKED_PIT
        g_reason = "题材表缺少历史纳入/剔除日期"
    for spec in FACTOR_SPECS:
        if spec["family"] == "G":
            rows.append(_status_row(spec, g_status, g_reason))
    h_status = classified.get("announcement_status") or BLOCKED_DATA
    h_reason = "没有公告日，只有生效日或表不存在时不得 PIT_PASS。计划上限不等于已成交减持。与 L/V 去重。"
    if h_status not in {BLOCKED_PIT, BLOCKED_DATA, BLOCKED_ACCESS}:
        h_status = BLOCKED_PIT
    for spec in FACTOR_SPECS:
        if spec["family"] == "H":
            rows.append(_status_row(spec, h_status, h_reason))
    abde = pd.read_parquet(PANEL_ABDE)
    abde["trade_date"] = pd.to_datetime(abde["trade_date"])
    abde["instrument_id"] = abde["instrument_id"].astype(str)
    extra = [c for c in c_panel.columns if c not in KEYS]
    panel = abde.merge(c_panel[KEYS + extra], on=KEYS, how="left") if extra else abde
    panel.to_parquet(PANEL_PATH, index=False)
    eq = pd.DataFrame(rows)
    EQ_PATH.parent.mkdir(parents=True, exist_ok=True)
    eq.to_csv(EQ_PATH, index=False)
    payload = {
        "status": "PARTIAL",
        "n_rows": int(len(eq)),
        "counts": eq["status"].value_counts().to_dict() if not eq.empty else {},
        "filled_with_zero": False,
        "profile_b_separate": True,
    }
    dump_json(META_PATH, payload)
    write_md(
        MD["10"],
        "\n".join(
            [
                "# Round22 · 10 构造 C / F / G / H",
                "",
                f"状态计数=`{payload['counts']}`。",
                "C2 与 `r21_t1_ticket_vs_med20` 同式，C3 与 `r21_g3_order_count_imbalance` 同式，记 DUPLICATE_EXPRESSION。",
                "C1/C4 只有 UPDATE_TIME 不晚于 D 日 15:00 的记录保留。不能证明 2018 前 PIT 的，不进 2018–2026 全历史模型。",
                "F 只属于 Profile B。未上榜不记为 0。G 没有历史纳入/剔除日则 BLOCKED_PIT。H 没有公告日则不得 PIT_PASS。",
                "",
            ]
        ),
    )
    return payload
