"""兼容别名：wxyz → construct/pit/evidence/backtest/combo。"""

from .backtest import run_wxyz_fourway as run_fourway
from .combo import run_wxyz_combo
from .construct import run_construct
from .evidence import run_evidence
from .pit import run_pit

__all__ = ["run_construct", "run_pit", "run_evidence", "run_fourway", "run_wxyz_combo"]
