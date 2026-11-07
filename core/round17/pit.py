"""工程 A–F PIT 门。高度相关只在 development 上诊断，不按全历史 CAGR 删因子。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..paths import ROUND17_ROOT
from .common import dump_json, load_json, write_md
from .factors import FACTOR_JSON, FACTOR_SPECS, load_new_factors

PIT_MD = ROUND17_ROOT / "03_pit_gate.md"
PIT_JSON = ROUND17_ROOT / "pit" / "pit.json"
MIN_FINITE = 0.80


def run_pit(**_: Any) -> dict[str, Any]:
    facts = load_json(FACTOR_JSON) or {}
    frame = load_new_factors()
    dev = frame.loc[frame["trade_date"].between("2018-01-01", "2021-12-31")].copy()
    valid = []
    skipped = []
    for spec in FACTOR_SPECS:
        name = spec["name"]
        s = pd.to_numeric(dev[name], errors="coerce")
        share = float(np.isfinite(s).mean()) if len(s) else 0.0
        reason = None
        if spec["family"] == "minute_new":
            reason = "SKIP_FACTOR minute family cannot support 2018-2026 without universe change"
        if share < MIN_FINITE:
            reason = f"SKIP_FACTOR finite_share={share:.4f} < {MIN_FINITE} (no fill-0)"
        # constant
        finite = s[np.isfinite(s)]
        if finite.nunique(dropna=True) <= 1:
            reason = "SKIP_FACTOR degenerate constant on development"
        rec = {**spec, "finite_share_dev": share, "nunique_dev": int(finite.nunique()) if len(finite) else 0}
        if reason:
            rec["status"] = "SKIP_FACTOR"
            rec["reason"] = reason
            skipped.append(rec)
        else:
            rec["status"] = "PIT_VALID"
            rec["reason"] = "decision-time or lagged; event-history uses matured labels only; no fill-0"
            valid.append(rec)
    # development-only correlation diagnostic, do not drop by full-history CAGR
    names = [r["name"] for r in valid]
    corr_pairs = []
    if names:
        mat = dev[names].apply(pd.to_numeric, errors="coerce")
        c = mat.corr()
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                v = c.loc[a, b]
                if pd.notna(v) and abs(float(v)) >= 0.95:
                    corr_pairs.append({"a": a, "b": b, "corr_dev": float(v)})
    payload = {
        "status": "gated",
        "n_specified": len(FACTOR_SPECS),
        "n_valid": len(valid),
        "n_skipped": len(skipped),
        "min_finite": MIN_FINITE,
        "filled_with_zero": False,
        "valid": valid,
        "skipped": skipped,
        "high_corr_dev_only": corr_pairs,
        "note": "相关诊断仅 development 2018–2021；不按全历史 CAGR 删因子。经验贝叶斯只用成熟 D+1/D+2 标签，先验 20。",
        "factor_build": {k: facts.get(k) for k in ("n_rows", "sha256", "filled_with_zero")},
    }
    dump_json(PIT_JSON, payload)
    lines = [
        "# 03 PIT 门",
        "",
        f"specified=`{len(FACTOR_SPECS)}` valid=`{len(valid)}` skipped=`{len(skipped)}`",
        "",
        "## Valid",
        "",
        "| name | family | finite_share | nunique |",
        "|---|---|---|---|",
    ]
    for rec in valid:
        lines.append(f"| `{rec['name']}` | {rec['family']} | {rec['finite_share_dev']:.4f} | {rec['nunique_dev']} |")
    lines += ["", "## Skipped", ""]
    if not skipped:
        lines.append("无。")
    for rec in skipped:
        lines.append(f"- `{rec['name']}`: {rec.get('reason')}")
    lines += ["", "## Development |corr|≥0.95（不删）", ""]
    if not corr_pairs:
        lines.append("无。")
    for p in corr_pairs:
        lines.append(f"- `{p['a']}` vs `{p['b']}` corr=`{p['corr_dev']:.4f}`")
    write_md(PIT_MD, "\n".join(lines))
    return payload


def valid_factor_names() -> list[str]:
    payload = run_pit() if not PIT_JSON.is_file() else __import__("json").loads(PIT_JSON.read_text(encoding="utf-8"))
    return [r["name"] for r in payload.get("valid") or []]
