"""兼容别名：track_a → train/backtest/attribution/nav_recon。"""

from .attribution import run_attribution
from .backtest import run_amatrix_backtest
from .nav_recon import run_recon43
from .train import run_amatrix_train

__all__ = ["run_amatrix_train", "run_amatrix_backtest", "run_attribution", "run_recon43"]
