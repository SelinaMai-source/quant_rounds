"""只读适配旧模块；记录来源文件 hash，不改旧实现。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .cache_identity import file_sha256


def source_hashes() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    rels = [
        "round22_3/contract.py",
        "r225241_models.py",
        "r225241_sleeve.py",
        "r225241_stats.py",
        "r225241_io.py",
        "local_executor.py",
        "round22_5/pipeline.py",
        "round23_1/pipeline.py",
        "round24_1/pipeline.py",
        "round24_1/features.py",
    ]
    out = {}
    for rel in rels:
        path = root / rel
        out[rel] = {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size if path.is_file() else None}
    return out


def fee_components() -> dict[str, Any]:
    from core.local_executor import round7_fixed_sleeve_config

    cfg = round7_fixed_sleeve_config(exit_mode="close_from_d2")
    return {
        "schedule_name": "ROUND7_FLAT_5BP",
        "slippage_bps": float(cfg.costs.slippage_bps),
        "commission_bps": float(cfg.costs.commission_bps),
        "transfer_fee_bps": float(cfg.costs.transfer_fee_bps),
        "sell_tax_bps": float(cfg.costs.sell_tax_bps),
        "note": "契约名含5BP，真实分项如上；买卖合计远不止5bp",
        "source": "core.local_executor.round7_fixed_sleeve_config",
    }


def module_digest(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()
