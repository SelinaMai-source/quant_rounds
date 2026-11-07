"""已有因子盘点。不发明不存在的数据。完成前不看策略收益。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import ROUND17_ROOT
from ..round7.contract import FIVE_FEATURES, FIVE_HEAD, MARKET_CONTEXT, TEN_FEATURES, TEN_HEAD
from ..round7.contract import LIMIT_DOWN_CONTEXT
from ..stock_readonly import ensure_stock_src_on_path
from .common import dump_json, write_md
from .factors import FACTOR_SPECS

INV_MD = ROUND17_ROOT / "00_existing_factor_inventory.md"
INV_JSON = ROUND17_ROOT / "inventory" / "existing.json"
REG_CSV = ROUND17_ROOT / "inventory" / "factor_registry.csv"


def _panel_cols() -> list[str]:
    path = FIVE_HEAD.panel_dir / "candidate_panel.parquet"
    return list(pd.read_parquet(path, columns=[]).columns) if False else list(pd.read_parquet(path).columns)


def run_inventory(**_: Any) -> dict[str, Any]:
    ensure_stock_src_on_path()
    from first_board_hermes.single_model import SINGLE_MODEL_FEATURES

    cand = pd.read_parquet(FIVE_HEAD.panel_dir / "candidate_panel.parquet")
    cols = list(cand.columns)
    locked = set(FIVE_FEATURES) | set(TEN_FEATURES)
    sm44 = list(SINGLE_MODEL_FEATURES)
    unused44 = [c for c in sm44 if c not in locked]
    mkt_cols = list(pd.read_parquet(MARKET_CONTEXT).columns)
    down_cols = list(pd.read_parquet(LIMIT_DOWN_CONTEXT).columns)
    rows = []
    for name in FIVE_FEATURES:
        rows.append({
            "name": name, "source": "baseline_five", "baseline": True, "head": "five+ten" if name in TEN_FEATURES else "five_only",
            "available_at": "D close except d1_open_gap=D+1 09:25", "pit": "inherited_round7",
            "old_trial_id": "R7_FS01122 / R9_B03",
        })
    for name in unused44:
        rows.append({
            "name": name, "source": "single_model44_unused", "baseline": False, "head": "panel_available",
            "available_at": "D close if in panel", "pit": "panel_complete_case",
            "old_trial_id": "first_board_factor_search_5000_v1 possible member; NOT re-searched as subset",
        })
    for spec in FACTOR_SPECS:
        rows.append({
            "name": spec["name"], "source": spec["family"], "baseline": False, "head": "new_r17",
            "available_at": spec["src"], "pit": "gated_in_pit_stage", "old_trial_id": "NEW",
        })
    frame = pd.DataFrame(rows)
    REG_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REG_CSV, index=False)
    payload = {
        "n_panel_cols": len(cols),
        "n_five": len(FIVE_FEATURES),
        "n_ten": len(TEN_FEATURES),
        "n_single_model44": len(sm44),
        "n_unused44": len(unused44),
        "n_new_specified": len(FACTOR_SPECS),
        "unused44": unused44,
        "market_context_cols": mkt_cols,
        "limit_down_cols": down_cols,
        "five_panel": str(FIVE_HEAD.panel_dir),
        "ten_panel": str(TEN_HEAD.panel_dir),
        "note": "Round17 搜索是 baseline 锁定 + 1–4 个新因子，不是 44 子集重放。",
        "registry_csv": str(REG_CSV),
    }
    dump_json(INV_JSON, payload)
    lines = [
        "# 00 已有因子盘点",
        "",
        "完成盘点/PIT 之前不看策略收益。不发明不存在的列。",
        "",
        f"- five 锁定 {len(FIVE_FEATURES)} 列：`{'`, `'.join(FIVE_FEATURES)}`",
        f"- ten 锁定 {len(TEN_FEATURES)} 列（five 去掉 g0_amount_log）",
        f"- SINGLE_MODEL_FEATURES 44，其中未进 baseline {len(unused44)} 列：不当作本轮随机子集搜索空间",
        f"- 新因子规格 {len(FACTOR_SPECS)}（见 08_new_families.md）",
        f"- market_context：{mkt_cols}",
        f"- limit_down_context：{down_cols}",
        "",
        "d1_open_gap 为 D+1 09:25 竞价信息，与 Round7 契约一致。",
        "分钟族若无法覆盖 2018–2026 主宇宙则 SKIP_FACTOR；本轮不新采集分钟。",
        "",
        f"registry：`{REG_CSV}`",
    ]
    write_md(INV_MD, "\n".join(lines))
    payload["status"] = "done"
    return payload
