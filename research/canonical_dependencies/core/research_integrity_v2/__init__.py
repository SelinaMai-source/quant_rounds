"""公共研究完整性修复 v2。不改写旧 round 源码；旧门保留为 LEGACY。"""

from .status import EvidenceStatus, as_bool, require_bool
from .nav_delta import nav_improvement, signed_nav_metrics
from .economic_gate import evaluate_economic_gate, freeze_economic_gate
from .cache_identity import ExperimentIdentity, cache_valid, fingerprint
from .merge_rank import explicit_merge, rank_unique
from .time_contract import LEGACY_OPEN_INFORMATION_LANE, shanghai_now, shanghai_ts
from .ledger import nav_tolerance, reconcile_nav
from .stats_v2 import delta_sharpe_of_paths, paired_block_bootstrap_v2, spa_hansen_or_blocked

__all__ = [
    "EvidenceStatus",
    "as_bool",
    "require_bool",
    "nav_improvement",
    "signed_nav_metrics",
    "evaluate_economic_gate",
    "freeze_economic_gate",
    "ExperimentIdentity",
    "cache_valid",
    "fingerprint",
    "explicit_merge",
    "rank_unique",
    "LEGACY_OPEN_INFORMATION_LANE",
    "shanghai_now",
    "shanghai_ts",
    "nav_tolerance",
    "reconcile_nav",
    "delta_sharpe_of_paths",
    "paired_block_bootstrap_v2",
    "spa_hansen_or_blocked",
]
