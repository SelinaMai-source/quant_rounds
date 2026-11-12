"""四路 Δ 归因 + Top1/Top2 逐笔。未来 label 只用于事后。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND21_2_ROOT
from .common import dump_json, finite_or_none, load_json, write_md
from .contract import E0_CODE, KEYS, MD, exp_dir
from .panels import pit_safe_factors
from .train import MODEL_JSON

ATTR_JSON = ROUND21_2_ROOT / "attribution" / "attribution.json"
ATTR_CSV = ROUND21_2_ROOT / "attribution" / "four_way_delta.csv"
TRADE_CSV = ROUND21_2_ROOT / "attribution" / "top1_top2_trades.csv"


def _q(row: dict[str, Any], field: str) -> float | None:
    if field in {"cagr", "sharpe", "max_drawdown"}:
        return finite_or_none(row.get(field))
    if field == "oof_five":
        return finite_or_none((row.get("five_oof") or {}).get("oof_rankic_mean"))
    if field == "oof_ten":
        return finite_or_none((row.get("ten_oof") or {}).get("oof_rankic_mean"))
    return finite_or_none(row.get(field))


def _delta(a: dict[str, Any] | None, b: dict[str, Any] | None, field: str) -> float | None:
    if not a or not b:
        return None
    va, vb = _q(a, field), _q(b, field)
    if va is None or vb is None:
        return None
    return va - vb


def _top1_map(code: str) -> pd.DataFrame:
    path = exp_dir(code) / "daily_top1_top2.parquet"
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def run_attribution(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and ATTR_JSON.is_file() and ATTR_CSV.is_file() and MD["16"].is_file():
        existing = load_json(ATTR_JSON)
        if existing.get("status") == "DONE":
            return existing
    model = load_json(MODEL_JSON)
    by = {r["code"]: r for r in (model.get("rows") or [])}
    for code, row in list(by.items()):
        summary = load_json(exp_dir(code) / "summary.json")
        if summary.get("sharpe") is not None:
            row.update({k: summary.get(k) for k in ("cagr", "sharpe", "max_drawdown", "completed_trades", "nav_end")})
    e0 = by.get(E0_CODE) or {}
    factors = pit_safe_factors(resume=True)
    rows = []
    trade_rows = []
    for feat in factors:
        e1, e2, e3 = by.get(f"R21_2_E1_{feat}"), by.get(f"R21_2_E2_{feat}"), by.get(f"R21_2_E3_{feat}")
        rec = {
            "new_factor": feat,
            "nearest_legacy": (e3 or e1 or {}).get("five_drops") or (e3 or e1 or {}).get("ten_drops"),
            "delta_add_sharpe": _delta(e2, e0, "sharpe"),
            "delta_remove_sharpe": _delta(e1, e0, "sharpe"),
            "delta_replace_sharpe": _delta(e3, e0, "sharpe"),
            "delta_new_given_remove_sharpe": _delta(e3, e1, "sharpe"),
            "delta_add_cagr": _delta(e2, e0, "cagr"),
            "delta_remove_cagr": _delta(e1, e0, "cagr"),
            "delta_replace_cagr": _delta(e3, e0, "cagr"),
            "delta_replace_maxdd": _delta(e3, e0, "max_drawdown"),
            "delta_add_oof_five": _delta(e2, e0, "oof_five"),
            "delta_add_oof_ten": _delta(e2, e0, "oof_ten"),
            "delta_remove_oof_five": _delta(e1, e0, "oof_five"),
            "delta_replace_oof_five": _delta(e3, e0, "oof_five"),
            "delta_replace_oof_ten": _delta(e3, e0, "oof_ten"),
            "delta_new_given_remove_oof_five": _delta(e3, e1, "oof_five"),
        }
        d_add, d_rep, d_rem, d_new = rec["delta_add_sharpe"], rec["delta_replace_sharpe"], rec["delta_remove_sharpe"], rec["delta_new_given_remove_sharpe"]
        note = None
        if d_add is not None and d_rep is not None and d_add < 0 and d_rep > 0:
            note = "直接加有害、替换有增量"
        elif d_rem is not None and d_new is not None and d_rem > 0 and abs(d_new) < 1e-6:
            note = "收益来自删旧因子，新因子未证明贡献"
        elif d_new is not None and d_rep is not None and d_new > 0 and d_rep < 0:
            note = "E3>E1 但 E3<E0，不升级"
        rec["interpretation"] = note
        rows.append(rec)

        old_top = _top1_map(E0_CODE)
        new_top = _top1_map(f"R21_2_E3_{feat}")
        if old_top.empty or new_top.empty:
            dest = ROUND21_2_ROOT / "factors" / feat
            dest.mkdir(parents=True, exist_ok=True)
            dump_json(dest / "attribution.json", rec)
            continue
        for head in ("five", "ten"):
            o = old_top.loc[old_top["head"].eq(head)]
            n = new_top.loc[new_top["head"].eq(head)]
            o1 = o.loc[o["rank"].eq(1), ["trade_date", "instrument_id", "score"]].rename(
                columns={"instrument_id": "old_top1", "score": "old_top1_score"}
            )
            n1 = n.loc[n["rank"].eq(1), ["trade_date", "instrument_id", "score"]].rename(
                columns={"instrument_id": "new_top1", "score": "new_top1_score"}
            )
            o2 = o.loc[o["rank"].eq(2), ["trade_date", "instrument_id"]].rename(columns={"instrument_id": "old_top2"})
            n2 = n.loc[n["rank"].eq(2), ["trade_date", "instrument_id"]].rename(columns={"instrument_id": "new_top2"})
            merged = o1.merge(n1, on="trade_date", how="outer").merge(o2, on="trade_date", how="left").merge(n2, on="trade_date", how="left")
            merged["head"] = head
            merged["new_factor"] = feat
            merged["top1_changed"] = merged["old_top1"].astype(str) != merged["new_top1"].astype(str)
            from .panels import DATA

            cand_path = DATA / ("five_candidate.parquet" if head == "five" else "ten_candidate.parquet")
            if cand_path.is_file():
                cand = pd.read_parquet(cand_path)
                cand["trade_date"] = pd.to_datetime(cand["trade_date"])
                lab = "label_e2_net" if head == "five" else "label_exit_rule_net"
                if lab in cand.columns:
                    lab_map = cand[KEYS + [lab]].rename(columns={lab: "realized_net_pnl"})
                    merged = merged.merge(
                        lab_map.rename(columns={"instrument_id": "new_top1"}),
                        on=["trade_date", "new_top1"],
                        how="left",
                    )
            merged["severe_loss"] = pd.to_numeric(merged.get("realized_net_pnl"), errors="coerce") <= -0.08
            trade_rows.extend(merged.to_dict(orient="records"))
        feat_trades = [t for t in trade_rows if t.get("new_factor") == feat]
        rec["top1_changed_share"] = float(pd.Series([t.get("top1_changed") for t in feat_trades]).mean()) if feat_trades else None
        rec["severe_loss_n"] = int(sum(1 for t in feat_trades if t.get("severe_loss")))
        dest = ROUND21_2_ROOT / "factors" / feat
        dest.mkdir(parents=True, exist_ok=True)
        dump_json(dest / "attribution.json", rec)

    ATTR_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ATTR_CSV, index=False)
    if trade_rows:
        pd.DataFrame(trade_rows).to_csv(TRADE_CSV, index=False)
        pd.DataFrame(trade_rows).to_parquet(ROUND21_2_ROOT / "attribution" / "top1_top2_trades.parquet", index=False)
    payload = {
        "status": "DONE",
        "n": len(rows),
        "future_label_post_hoc_only": True,
        "csv": str(ATTR_CSV),
        "trades": str(TRADE_CSV) if trade_rows else None,
        "rows": rows,
    }
    dump_json(ATTR_JSON, payload)
    lines = [
        "# Round21.2 · 16 Top1/Top2 归因",
        "",
        "ΔAdd=Q(E2)-Q(E0)；ΔRemove=Q(E1)-Q(E0)；ΔReplace=Q(E3)-Q(E0)；ΔNewGivenRemove=Q(E3)-Q(E1)。未来 label 只事后。",
        "",
        "| factor | ΔAdd Sh | ΔRem Sh | ΔRep Sh | ΔNew|Rem Sh | ΔRep CAGR | note |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['new_factor']}` | {r['delta_add_sharpe']} | {r['delta_remove_sharpe']} | "
            f"{r['delta_replace_sharpe']} | {r['delta_new_given_remove_sharpe']} | {r['delta_replace_cagr']} | {r.get('interpretation')} |"
        )
    write_md(MD["16"], "\n".join(lines))
    return payload
