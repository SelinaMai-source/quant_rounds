"""课题路径。禁止把回测产物写到 Stock/src 或 Stock/runs；本课题 result/ 可写。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STOCK_ROOT = Path("/home/mxy/Stock")
STOCK_SRC = STOCK_ROOT / "src"
RESULT_ROOT = PROJECT_ROOT / "result"
ROUND1_ROOT = RESULT_ROOT / "round1"
ROUND2_ROOT = RESULT_ROOT / "round2"
ROUND3_ROOT = RESULT_ROOT / "round3"
ROUND4_ROOT = RESULT_ROOT / "round4"
ROUND5_ROOT = RESULT_ROOT / "round5"
ROUND6_ROOT = RESULT_ROOT / "round6"
ROUND7_ROOT = RESULT_ROOT / "round7"
ROUND8_ROOT = RESULT_ROOT / "round8"
ROUND9_ROOT = RESULT_ROOT / "round9"
ROUND10_ROOT = RESULT_ROOT / "round10"
ROUND11_ROOT = RESULT_ROOT / "round11"
ROUND12_ROOT = RESULT_ROOT / "round12"
ROUND13_ROOT = RESULT_ROOT / "round13"
ROUND14_ROOT = RESULT_ROOT / "round14"
ROUND15_ROOT = RESULT_ROOT / "round15"
ROUND16_ROOT = RESULT_ROOT / "round16"
ROUND16_5_ROOT = RESULT_ROOT / "round16_5"
ROUND17_ROOT = RESULT_ROOT / "round17"
ROUND18_ROOT = RESULT_ROOT / "round18"
ROUND19_ROOT = RESULT_ROOT / "round19"
ROUND20_ROOT = RESULT_ROOT / "round20"
ROUND21_ROOT = RESULT_ROOT / "round21"
ROUND21_1_ROOT = RESULT_ROOT / "round21_1"
ROUND21_2_ROOT = RESULT_ROOT / "round21_2"
ROUND21_3_ROOT = RESULT_ROOT / "round21_3"
ROUND21_4_ROOT = RESULT_ROOT / "round21_4"
ROUND22_ROOT = RESULT_ROOT / "round22"
ROUND22_1_ROOT = RESULT_ROOT / "round22_1"
REPLAY_FS01122_ROOT = RESULT_ROOT / "replay_fs01122"
REPLAY_ROUND7_ROOT = RESULT_ROOT / "replay_round7"
STOCK_ROUND7_ROOT = ROUND7_ROOT

ROUND2_SHARED_DIRS: tuple[str, ...] = (
    "00_audit",
    "01_attr",
    "02_research",
    "03_single",
    "04_pairwise",
    "05_full",
    "06_conclusion",
    "configs",
    "metrics",
    "trades",
    "nav",
    "figures",
    "attribution",
    "research",
    "models",
)

ROUND3_SHARED_DIRS: tuple[str, ...] = (
    "00_parity",
    "01_pit",
    "02_repro",
    "03_ablation",
    "04_dual",
    "05_size",
    "06_neighbor",
    "07_time",
    "08_seed",
    "09_cost",
    "10_tail",
    "11_bootstrap",
    "12_stress",
    "configs",
    "metrics",
    "trades",
    "nav",
    "figures",
    "frozen",
)

ROUND4_SHARED_DIRS: tuple[str, ...] = (
    "00_parity",
    "01_attr",
    "02_ecology",
    "03_margin",
    "04_twodim",
    "05_cross",
    "06_fdiag",
    "07_cvar",
    "08_single",
    "09_pair",
    "10_triple",
    "11_robust",
    "12_conclusion",
    "configs",
    "metrics",
    "trades",
    "nav",
    "figures",
)

ROUND5_SHARED_DIRS: tuple[str, ...] = (
    "00_parity",
    "01_diagnose",
    "02_audit",
    "03_univariate",
    "04_train",
    "05_objective",
    "06_label",
    "07_twostage",
    "08_combo",
    "09_finalists",
    "10_robust",
    "11_conclusion",
    "configs",
    "metrics",
    "trades",
    "nav",
    "figures",
    "factors",
    "models",
    "predictions",
)

ROUND6_SHARED_DIRS: tuple[str, ...] = (
    "00_parity",
    "01_forensics",
    "02_horizon",
    "03_repr",
    "04_pairwise",
    "05_label",
    "06_meta",
    "07_ensemble",
    "08_context",
    "09_combo",
    "10_robust",
    "11_conclusion",
    "configs",
    "metrics",
    "trades",
    "nav",
    "figures",
    "models",
    "predictions",
    "reproducibility",
    "canonical",
)

ROUND7_SHARED_DIRS: tuple[str, ...] = (
    "models",
    "predictions",
    "diagnostics",
    "comparison",
)

ROUND8_SHARED_DIRS: tuple[str, ...] = (
    "metrics",
    "nav",
    "trades",
    "exposure",
    "bootstrap",
    "configs",
    "figures",
    "logs",
    "models",
    "predictions",
)

ROUND9_SHARED_DIRS: tuple[str, ...] = (
    "metrics",
    "nav",
    "trades",
    "exposure",
    "bootstrap",
    "configs",
    "figures",
    "logs",
    "models",
    "predictions",
    "ledger",
)

ROUND10_SHARED_DIRS: tuple[str, ...] = (
    "metrics",
    "nav",
    "trades",
    "exposure",
    "bootstrap",
    "configs",
    "figures",
    "logs",
    "models",
    "predictions",
    "ledger",
    "diagnostics",
    "pair",
    "shrinkage",
    "capacity",
    "perturbation",
    "shadow",
)

ROUND11_SHARED_DIRS: tuple[str, ...] = (
    "parity",
    "contracts",
    "metrics",
    "nav",
    "trades",
    "configs",
    "logs",
    "diagnostics",
    "figures",
    "pit",
    "engine",
    "cost",
    "time",
    "model",
    "placebo",
    "trials",
    "alphaforge",
    "stats",
    "capacity",
    "dossier",
)

ROUND12_SHARED_DIRS: tuple[str, ...] = (
    "parity",
    "universe",
    "execution",
    "deploy",
    "selection",
    "selection/recovered_returns",
    "trials",
    "dsr",
    "pbo",
    "alphaforge",
    "placebo",
    "timing_placebo",
    "staircase",
    "capacity",
    "shadow_protocol",
    "controls",
    "stats",
    "logs",
    "configs",
    "metrics",
    "nav",
    "trades",
    "diagnostics",
    "figures",
)

ROUND13_SHARED_DIRS: tuple[str, ...] = (
    "parity",
    "weights",
    "candidates",
    "scenarios",
    "bootstrap",
    "oob",
    "cpcv",
    "spa",
    "placebo",
    "timing",
    "execution",
    "capacity",
    "forward",
    "promotion",
    "stats",
    "controls",
    "logs",
    "configs",
    "metrics",
    "nav",
    "trades",
    "diagnostics",
    "figures",
)

ROUND14_SHARED_DIRS: tuple[str, ...] = (
    "parity",
    "risk",
    "weights",
    "candidates",
    "controls",
    "scenarios",
    "bootstrap",
    "matched_bootstrap",
    "cpcv",
    "spa",
    "execution",
    "capacity",
    "tail",
    "forward",
    "promotion",
    "stats",
    "logs",
    "configs",
    "metrics",
    "nav",
    "trades",
    "diagnostics",
    "figures",
    "tests",
)

ROUND15_SHARED_DIRS: tuple[str, ...] = (
    "parity",
    "inventory",
    "sdk",
    "data",
    "factors",
    "residual",
    "options",
    "top1",
    "execution",
    "survivorship",
    "candidates",
    "controls",
    "bootstrap",
    "matched_bootstrap",
    "cpcv",
    "spa",
    "forward",
    "promotion",
    "stats",
    "logs",
    "configs",
    "metrics",
    "nav",
    "trades",
    "diagnostics",
    "figures",
    "tests",
    "weights",
)

ROUND16_SHARED_DIRS: tuple[str, ...] = (
    "parity",
    "audit",
    "sdk",
    "data",
    "data/round15_snapshot",
    "data/raw",
    "data/raw/option_kline_day",
    "data/raw/hist_codes",
    "data/raw/basic",
    "data/raw/ad_local",
    "data/processed",
    "factors",
    "incremental",
    "redundancy",
    "options",
    "strategies",
    "candidates",
    "controls",
    "bootstrap",
    "matched_bootstrap",
    "cpcv",
    "spa",
    "forward",
    "promotion",
    "stats",
    "logs",
    "configs",
    "metrics",
    "nav",
    "trades",
    "diagnostics",
    "figures",
    "tests",
    "weights",
    "stress",
    "coverage",
    "chain",
    "quality",
    "iv",
)

ROUND16_5_SHARED_DIRS: tuple[str, ...] = (
    "parity",
    "candidates",
    "controls",
    "placebo",
    "attribution",
    "equity",
    "orthogonal",
    "quote",
    "pit",
    "prediction",
    "bootstrap",
    "matched_bootstrap",
    "cpcv",
    "spa",
    "forward",
    "promotion",
    "stats",
    "logs",
    "configs",
    "metrics",
    "nav",
    "trades",
    "diagnostics",
    "figures",
    "tests",
    "weights",
    "stress",
    "inventory",
    "read",
)

ROUND17_SHARED_DIRS: tuple[str, ...] = (
    "parity",
    "preflight",
    "inventory",
    "factors",
    "pit",
    "data",
    "configs",
    "search",
    "search/trials",
    "models",
    "predictions",
    "nav",
    "trades",
    "ledger",
    "metrics",
    "select",
    "validate",
    "fourway",
    "diversity",
    "attribution",
    "bootstrap",
    "cpcv",
    "spa",
    "pbo",
    "stress",
    "capacity",
    "forward",
    "promotion",
    "stats",
    "logs",
    "figures",
    "weights",
    "read",
)

ROUND18_SHARED_DIRS: tuple[str, ...] = (
    "parity",
    "inventory",
    "overlap",
    "data",
    "data/raw",
    "data/processed",
    "factors",
    "pit",
    "configs",
    "models",
    "predictions",
    "nav",
    "trades",
    "ledger",
    "metrics",
    "bootstrap",
    "spa",
    "cpcv",
    "stress",
    "capacity",
    "attribution",
    "forward",
    "promotion",
    "stats",
    "logs",
    "figures",
    "weights",
    "read",
    "download",
    "diagnostics",
    "candidates",
)

ROUND19_SHARED_DIRS: tuple[str, ...] = (
    "parity",
    "read",
    "configs",
    "metrics",
    "nav",
    "trades",
    "ledger",
    "logs",
    "figures",
    "weights",
    "diagnostics",
    "candidates",
    "controls",
    "placebo",
    "attribution",
    "equity",
    "pit",
    "quote",
    "prediction",
    "bootstrap",
    "matched_bootstrap",
    "cpcv",
    "spa",
    "forward",
    "promotion",
    "stats",
    "stress",
    "inventory",
    "models",
    "predictions",
    "tests",
    "track_a",
    "track_a/replay",
    "track_a/events",
    "track_a/attribution",
    "track_a/concentration",
    "track_a/controls",
    "track_a/placebo",
    "track_a/pit",
    "track_a/quotes",
    "track_a/prediction",
    "track_a/stats",
    "track_b",
    "track_b/retrain",
    "track_b/parity",
    "track_b/audit",
    "track_b/heads",
    "track_b/repair",
    "track_c",
    "track_c/forward",
    "track_c/decision",
)

ROUND20_SHARED_DIRS: tuple[str, ...] = (
    "parity",
    "read",
    "inventory",
    "pit",
    "ic",
    "shape",
    "redundancy",
    "conditional",
    "combo",
    "configs",
    "metrics",
    "nav",
    "trades",
    "ledger",
    "logs",
    "figures",
    "weights",
    "diagnostics",
    "candidates",
    "models",
    "predictions",
    "seed",
    "bootstrap",
    "matched_bootstrap",
    "cpcv",
    "spa",
    "pbo",
    "forward",
    "promotion",
    "stats",
    "stress",
    "factors",
    "data",
    "tests",
)

ROUND21_SHARED_DIRS: tuple[str, ...] = (
    "parity",
    "read",
    "github",
    "source_audit",
    "hypotheses",
    "inventory",
    "pit",
    "ic",
    "shape",
    "redundancy",
    "conditional",
    "combo",
    "configs",
    "metrics",
    "nav",
    "trades",
    "ledger",
    "logs",
    "figures",
    "weights",
    "diagnostics",
    "candidates",
    "models",
    "predictions",
    "seed",
    "bootstrap",
    "matched_bootstrap",
    "cpcv",
    "spa",
    "pbo",
    "forward",
    "promotion",
    "stats",
    "stress",
    "factors",
    "data",
    "tests",
    "zoo",
)

ROUND21_3_SHARED_DIRS: tuple[str, ...] = (
    "parity",
    "audit",
    "lineage",
    "source_audit",
    "hypotheses",
    "coverage",
    "correlation",
    "conditional",
    "attribution",
    "interaction",
    "github",
    "evidence",
    "configs",
    "metrics",
    "nav",
    "trades",
    "ledger",
    "logs",
    "figures",
    "weights",
    "diagnostics",
    "candidates",
    "models",
    "predictions",
    "seed",
    "bootstrap",
    "spa",
    "forward",
    "promotion",
    "stats",
    "factors",
    "data",
    "data_recovery",
    "zoo",
    "pit",
    "ic",
    "shape",
    "n1n3",
    "construct",
    "backtest",
    "fee_sensitivity",
    "rejected",
)

ROUND21_2_SHARED_DIRS: tuple[str, ...] = (
    "parity",
    "audit",
    "lineage",
    "source_audit",
    "hypotheses",
    "coverage",
    "correlation",
    "conditional",
    "attribution",
    "interaction",
    "github",
    "evidence",
    "configs",
    "metrics",
    "nav",
    "trades",
    "ledger",
    "logs",
    "figures",
    "weights",
    "diagnostics",
    "candidates",
    "models",
    "predictions",
    "seed",
    "bootstrap",
    "spa",
    "forward",
    "promotion",
    "stats",
    "factors",
    "data",
    "zoo",
    "pit",
    "ic",
    "shape",
)

ROUND21_1_SHARED_DIRS: tuple[str, ...] = (
    "parity",
    "coverage",
    "audit",
    "correlation",
    "conditional",
    "attribution",
    "configs",
    "metrics",
    "nav",
    "trades",
    "ledger",
    "logs",
    "figures",
    "weights",
    "diagnostics",
    "candidates",
    "models",
    "predictions",
    "seed",
    "bootstrap",
    "spa",
    "forward",
    "promotion",
    "stats",
    "factors",
    "data",
    "zoo",
    "hypotheses",
)

ROUND22_SHARED_DIRS: tuple[str, ...] = (
    "parity",
    "configs",
    "candidates",
    "weights",
    "model",
    "data",
    "data_recovery",
    "forward",
    "coverage",
    "hypotheses",
    "literature",
    "prior",
    "construct",
    "pit",
    "ic",
    "evidence",
    "correlation",
    "backtest",
    "fee_sensitivity",
    "factors",
    "zoo",
    "stats",
    "seed",
    "bootstrap",
    "spa",
    "ledger",
    "logs",
    "metrics",
    "profile_b",
    "diagnostics",
    "rejected",
    "tracker",
    "report",
)

ROUND21_4_SHARED_DIRS: tuple[str, ...] = (
    "parity",
    "configs",
    "candidates",
    "weights",
    "model",
    "data",
    "data_recovery",
    "forward",
    "coverage",
    "attribution",
    "backtest",
    "fee_sensitivity",
    "factors",
    "construct",
    "pit",
    "ic",
    "evidence",
    "correlation",
    "combo",
    "combo/candidates",
    "zoo",
    "stats",
    "seed",
    "bootstrap",
    "spa",
    "ledger",
    "logs",
    "metrics",
    "nav",
    "trades",
    "predictions",
    "diagnostics",
    "figures",
    "rejected",
    "tracker",
    "report",
    "promotion",
)

SEARCH_RUN = "first_board_factor_search_5000_v1"
TRIAL_ID = "fs00263"
CAPACITY = "n50_d2"
ENSEMBLE_METHOD = "ensemble_rank_pct_5seed"
BACKTEST_VARIANT = "top1_both_gates_nav30pct_v1"
SNAPSHOT_RELATIVE = Path("data/datasets/hermes-cn-a-2010-20260724-v7-all-features")
CANDIDATE_PANEL_RELATIVE = Path(
    "data/first_board_hermes/"
    "aligned_original_candidate_full44_2014warmup_2018oos_v1/candidate_panel.parquet"
)
ENSEMBLE_RELATIVE = Path(
    f"runs/{SEARCH_RUN}/ensembles/{CAPACITY}/{ENSEMBLE_METHOD}/{TRIAL_ID}"
)
BASELINE_REPORT_RELATIVE = Path(
    f"runs/{SEARCH_RUN}/backtests/{CAPACITY}/{ENSEMBLE_METHOD}/"
    f"{TRIAL_ID}/{BACKTEST_VARIANT}/report.json"
)

PREDICTION_START = date(2018, 1, 1)
SIGNAL_END = date(2026, 6, 1)
MARKET_END = date(2026, 7, 24)

FS00263_FEATURES: tuple[str, ...] = (
    "fb_limit_close_minute_ratio",
    "h5_close_limit_retention",
    "h1_last15m_volume_ratio",
    "g0_amt_vs_5",
    "ind_first_board_count_log",
    "ind_stock_seal_quality_rank",
)

FS01122_RUN = "first_board_fs01122_d1_open_gap_v1"
FS01122_TRIAL_ID = "fs01122"
FS01122_CAPACITY = "n100_d2"
FS01122_ENSEMBLE = "ensemble_prefilter_gap_floor"
FS01122_BACKTEST = "rank25_15"
FS01122_PREDICTIONS_RELATIVE = Path(
    f"runs/{FS01122_RUN}/{FS01122_ENSEMBLE}/predictions.parquet"
)
FS01122_SEED_PRED_RELATIVE = Path(f"runs/{FS01122_RUN}/predictions/{FS01122_CAPACITY}")
FS01122_STOCK_REPORT_RELATIVE = Path(
    f"runs/{FS01122_RUN}/backtests_prefilter/{FS01122_BACKTEST}/report.json"
)
FS01122_PANEL_RELATIVE = Path(
    "data/first_board_hermes/aligned_original_candidate_full44_plus_d1_open_gap_v1"
)
FS01122_FEATURES: tuple[str, ...] = (
    "g_prim_px_vol_corr",
    "fb_close_seal_streak_norm",
    "fb_last30_amount_ratio",
    "h5_last_first_limit_value_change_sym",
    "h1_consistent_buy_ratio",
    "h1_last15m_volume_ratio",
    "h4_large_net_to_turnover",
    "h4_retail_net_to_turnover",
    "g0_amount_log",
    "g0_amt_vs_5",
    "g1_ret_5",
    "g3_days_since_limit",
    "ind_seal_rate_excess",
    "d1_open_gap",
)
FS01122_TOP_K = 2
FS01122_RANK_NAV_FRACTIONS: tuple[float, ...] = (0.25, 0.15)
FS01122_OPEN_GAP_MIN = -0.01
FS01122_SEEDS: tuple[int, ...] = (17, 42, 73, 101, 202)

FS00263_FEATURE_NAMES_ZH: dict[str, str] = {
    "fb_limit_close_minute_ratio": "收盘封住分钟占比",
    "h5_close_limit_retention": "收盘封单留存",
    "h1_last15m_volume_ratio": "尾盘 15 分钟量比",
    "g0_amt_vs_5": "当日成交额 / 近 5 日均额",
    "ind_first_board_count_log": "行业首板数量对数",
    "ind_stock_seal_quality_rank": "个股封板质量行业内排名",
}


def stock_path(relative: str | Path) -> Path:
    """Resolve a path under the read-only Stock root."""
    return STOCK_ROOT / relative


def round1_dir(code: str) -> Path:
    """Round 1 artifact directory inside the 课题 folder (never Stock/runs)."""
    return ROUND1_ROOT / str(code).strip()


def round2_dir(code: str) -> Path:
    """Round 2 artifact directory inside the 课题 folder (never Stock/runs)."""
    return ROUND2_ROOT / str(code).strip()


def ensure_round2_tree() -> Path:
    """Create result/round2 shared folders. Never touches Stock."""
    ROUND2_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND2_SHARED_DIRS:
        (ROUND2_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND2_ROOT


def round3_dir(code: str) -> Path:
    """Round 3 artifact directory inside the 课题 folder (never Stock/runs)."""
    return ROUND3_ROOT / str(code).strip()


def ensure_round3_tree() -> Path:
    """Create result/round3 shared folders. Never touches Stock."""
    ROUND3_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND3_SHARED_DIRS:
        (ROUND3_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND3_ROOT


def round4_dir(code: str) -> Path:
    """Round 4 artifact directory inside the 课题 folder (never Stock/runs)."""
    return ROUND4_ROOT / str(code).strip()


def ensure_round4_tree() -> Path:
    """Create result/round4 shared folders. Never touches Stock."""
    ROUND4_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND4_SHARED_DIRS:
        (ROUND4_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND4_ROOT


def round5_dir(code: str) -> Path:
    """Round 5 artifact directory inside the 课题 folder (never Stock/runs)."""
    return ROUND5_ROOT / str(code).strip()


def ensure_round5_tree() -> Path:
    """Create result/round5 shared folders. Never touches Stock."""
    ROUND5_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND5_SHARED_DIRS:
        (ROUND5_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND5_ROOT


def round6_dir(code: str) -> Path:
    """Round 6 artifact directory inside the 课题 folder (never Stock/runs)."""
    return ROUND6_ROOT / str(code).strip()


def ensure_round6_tree() -> Path:
    """Create result/round6 shared folders. Never touches Stock src/runs."""
    ROUND6_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND6_SHARED_DIRS:
        (ROUND6_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND6_ROOT


def round7_dir(code: str) -> Path:
    """Round 7 candidate artifact directory inside this 课题."""
    return ROUND7_ROOT / str(code).strip()


def ensure_round7_tree() -> Path:
    """Create the Round 7 tree. Never touches Stock src/runs."""
    ROUND7_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND7_SHARED_DIRS:
        (ROUND7_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND7_ROOT


def replay_fs01122_dir(round_name: str, code: str = "") -> Path:
    """result/replay_fs01122/{round_name}/{code}/. Never Stock/runs."""
    base = REPLAY_FS01122_ROOT / str(round_name).strip()
    if code:
        return base / str(code).strip()
    return base


def ensure_replay_fs01122_tree() -> Path:
    """Create result/replay_fs01122/{round1…round6} plus shared folders."""
    REPLAY_FS01122_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("round1", "round2", "round3", "round4", "round5", "round6", "parity"):
        (REPLAY_FS01122_ROOT / name).mkdir(parents=True, exist_ok=True)
        for sub in ("configs", "metrics", "trades", "nav", "models", "predictions"):
            (REPLAY_FS01122_ROOT / name / sub).mkdir(parents=True, exist_ok=True)
    return REPLAY_FS01122_ROOT


def replay_round7_dir(round_name: str, code: str = "") -> Path:
    """result/replay_round7/{round_name}/{code}/. Never Stock/runs."""
    base = REPLAY_ROUND7_ROOT / str(round_name).strip()
    if code:
        return base / str(code).strip()
    return base


def ensure_replay_round7_tree() -> Path:
    """Create result/replay_round7/{round1…round6} plus shared folders."""
    REPLAY_ROUND7_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("round1", "round2", "round3", "round4", "round5", "round6", "parity"):
        (REPLAY_ROUND7_ROOT / name).mkdir(parents=True, exist_ok=True)
        for sub in ("configs", "metrics", "trades", "nav", "models", "predictions"):
            (REPLAY_ROUND7_ROOT / name / sub).mkdir(parents=True, exist_ok=True)
    return REPLAY_ROUND7_ROOT


def round8_dir(code: str = "") -> Path:
    """result/round8/{code}/. Never Stock/runs."""
    if code:
        return ROUND8_ROOT / str(code).strip()
    return ROUND8_ROOT


def ensure_round8_tree() -> Path:
    """Create result/round8 shared folders. Never touches Stock."""
    ROUND8_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND8_SHARED_DIRS:
        (ROUND8_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND8_ROOT


def round9_dir(code: str = "") -> Path:
    """result/round9/{code}/. Never Stock/runs."""
    if code:
        return ROUND9_ROOT / str(code).strip()
    return ROUND9_ROOT


def ensure_round9_tree() -> Path:
    """Create result/round9 shared folders. Never touches Stock."""
    ROUND9_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND9_SHARED_DIRS:
        (ROUND9_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND9_ROOT


def round10_dir(code: str = "") -> Path:
    """result/round10/{code}/. Never Stock/runs."""
    if code:
        return ROUND10_ROOT / str(code).strip()
    return ROUND10_ROOT


def ensure_round10_tree() -> Path:
    """Create result/round10 shared folders. Never touches Stock."""
    ROUND10_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND10_SHARED_DIRS:
        (ROUND10_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND10_ROOT


def round11_dir(code: str = "") -> Path:
    """result/round11/{code}/. Never Stock/runs."""
    if code:
        return ROUND11_ROOT / str(code).strip()
    return ROUND11_ROOT


def ensure_round11_tree() -> Path:
    """Create result/round11 shared folders. Never touches Stock or legacy roundN."""
    ROUND11_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND11_SHARED_DIRS:
        (ROUND11_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND11_ROOT


def round12_dir(code: str = "") -> Path:
    """result/round12/{code}/. Never Stock/runs."""
    if code:
        return ROUND12_ROOT / str(code).strip()
    return ROUND12_ROOT


def ensure_round12_tree() -> Path:
    """Create result/round12 shared folders. Never touches Stock or legacy roundN."""
    ROUND12_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND12_SHARED_DIRS:
        (ROUND12_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND12_ROOT


def round13_dir(code: str = "") -> Path:
    """result/round13/{code}/. Never Stock/runs."""
    if code:
        return ROUND13_ROOT / str(code).strip()
    return ROUND13_ROOT


def ensure_round13_tree() -> Path:
    """Create result/round13 shared folders. Never touches Stock or legacy roundN."""
    ROUND13_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND13_SHARED_DIRS:
        (ROUND13_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND13_ROOT


def round14_dir(code: str = "") -> Path:
    """result/round14/{code}/. Never Stock/runs."""
    if code:
        return ROUND14_ROOT / str(code).strip()
    return ROUND14_ROOT


def ensure_round14_tree() -> Path:
    """Create result/round14 shared folders. Never touches Stock or legacy roundN."""
    ROUND14_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND14_SHARED_DIRS:
        (ROUND14_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND14_ROOT


def round15_dir(code: str = "") -> Path:
    """result/round15/{code}/. Never Stock/runs."""
    if code:
        return ROUND15_ROOT / str(code).strip()
    return ROUND15_ROOT


def ensure_round15_tree() -> Path:
    """Create result/round15 shared folders. Never touches Stock or legacy roundN."""
    ROUND15_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND15_SHARED_DIRS:
        (ROUND15_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND15_ROOT


def round16_dir(code: str = "") -> Path:
    """result/round16/{code}/. Never Stock/runs."""
    if code:
        return ROUND16_ROOT / str(code).strip()
    return ROUND16_ROOT


def ensure_round16_tree() -> Path:
    """Create result/round16 shared folders. Never touches Stock or other 课题."""
    ROUND16_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND16_SHARED_DIRS:
        (ROUND16_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND16_ROOT


def round16_5_dir(code: str = "") -> Path:
    """result/round16_5/{code}/. Never Stock/runs."""
    if code:
        return ROUND16_5_ROOT / str(code).strip()
    return ROUND16_5_ROOT


def ensure_round16_5_tree() -> Path:
    """Create result/round16_5 shared folders. Never touches Stock or other 课题."""
    ROUND16_5_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND16_5_SHARED_DIRS:
        (ROUND16_5_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND16_5_ROOT


def round17_dir(code: str = "") -> Path:
    """result/round17/{code}/. Never Stock/runs."""
    if code:
        return ROUND17_ROOT / str(code).strip()
    return ROUND17_ROOT


def ensure_round17_tree() -> Path:
    """Create result/round17 shared folders. Never touches Stock or other 课题."""
    ROUND17_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND17_SHARED_DIRS:
        (ROUND17_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND17_ROOT


def round18_dir(code: str = "") -> Path:
    """result/round18/{code}/. Never Stock/runs."""
    if code:
        return ROUND18_ROOT / str(code).strip()
    return ROUND18_ROOT


def ensure_round18_tree() -> Path:
    """Create result/round18 shared folders. Never touches Stock or other 课题."""
    ROUND18_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND18_SHARED_DIRS:
        (ROUND18_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND18_ROOT


def round19_dir(code: str = "") -> Path:
    """result/round19/{code}/. Never Stock/runs."""
    if code:
        return ROUND19_ROOT / str(code).strip()
    return ROUND19_ROOT


def ensure_round19_tree() -> Path:
    """Create result/round19 shared folders. Never touches Stock or other 课题."""
    ROUND19_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND19_SHARED_DIRS:
        (ROUND19_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND19_ROOT


def round20_dir(code: str = "") -> Path:
    """result/round20/{code}/. Never Stock/runs."""
    if code:
        return ROUND20_ROOT / str(code).strip()
    return ROUND20_ROOT


def ensure_round20_tree() -> Path:
    """Create result/round20 shared folders. Never touches Stock or other 课题."""
    ROUND20_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND20_SHARED_DIRS:
        (ROUND20_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND20_ROOT


def round21_dir(code: str = "") -> Path:
    """result/round21/{code}/. Never Stock/runs."""
    if code:
        return ROUND21_ROOT / str(code).strip()
    return ROUND21_ROOT


def ensure_round21_tree() -> Path:
    """Create result/round21 shared folders. Never touches Stock or other 课题."""
    ROUND21_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND21_SHARED_DIRS:
        (ROUND21_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND21_ROOT


def round21_1_dir(code: str = "") -> Path:
    """result/round21_1/{code}/. Never Stock/runs. Never overwrite round21 ledgers."""
    if code:
        return ROUND21_1_ROOT / str(code).strip()
    return ROUND21_1_ROOT


def ensure_round21_1_tree() -> Path:
    """Create result/round21_1 shared folders. Never touches result/round21 ledgers."""
    ROUND21_1_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND21_1_SHARED_DIRS:
        (ROUND21_1_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND21_1_ROOT


def round21_2_dir(code: str = "") -> Path:
    """result/round21_2/{code}/. Never Stock/runs. Never overwrite round21 / round21_1 ledgers."""
    if code:
        return ROUND21_2_ROOT / str(code).strip()
    return ROUND21_2_ROOT


def ensure_round21_2_tree() -> Path:
    """Create result/round21_2 shared folders. Never touches result/round21 or round21_1 ledgers."""
    ROUND21_2_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND21_2_SHARED_DIRS:
        (ROUND21_2_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND21_2_ROOT


def round21_3_dir(code: str = "") -> Path:
    """result/round21_3/{code}/. Never Stock/runs. Never overwrite round21 / 21.1 / 21.2 ledgers."""
    if code:
        return ROUND21_3_ROOT / str(code).strip()
    return ROUND21_3_ROOT


def ensure_round21_3_tree() -> Path:
    """Create result/round21_3 shared folders. Never touches result/round21, 21.1, or 21.2 ledgers."""
    ROUND21_3_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND21_3_SHARED_DIRS:
        (ROUND21_3_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND21_3_ROOT


def round21_4_dir(code: str = "") -> Path:
    """result/round21_4/{code}/. Never Stock/runs. Never overwrite round21 / 21.1 / 21.2 / 21.3 ledgers."""
    if code:
        return ROUND21_4_ROOT / str(code).strip()
    return ROUND21_4_ROOT


def ensure_round21_4_tree() -> Path:
    """Create result/round21_4 shared folders. Never touches result/round21, 21.1, 21.2, or 21.3 ledgers."""
    ROUND21_4_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND21_4_SHARED_DIRS:
        (ROUND21_4_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND21_4_ROOT


def round22_dir(code: str = "") -> Path:
    """result/round22/{code}/. Never overwrite round21* ledgers. Never touch CURRENT."""
    if code:
        return ROUND22_ROOT / str(code).strip()
    return ROUND22_ROOT


def ensure_round22_tree() -> Path:
    """Create result/round22 shared folders. Never touches result/round21* ledgers."""
    ROUND22_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND22_SHARED_DIRS:
        (ROUND22_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND22_ROOT


ROUND22_1_SHARED_DIRS: tuple[str, ...] = (
    "parity",
    "configs",
    "candidates",
    "weights",
    "data",
    "data_recovery",
    "ic",
    "evidence",
    "factors",
    "ledger",
    "nav",
    "construct",
    "pit",
    "correlation",
    "backtest",
    "stats",
    "bootstrap",
    "spa",
    "seed",
    "zoo",
    "tests",
    "report",
    "logs",
    "diagnostics",
    "rejected",
    "tracker",
    "coverage",
    "forward",
    "hypotheses",
)


def round22_1_dir(code: str = "") -> Path:
    """result/round22_1/{code}/. Never overwrite round22 or round21* ledgers. Never touch CURRENT."""
    if code:
        return ROUND22_1_ROOT / str(code).strip()
    return ROUND22_1_ROOT


def ensure_round22_1_tree() -> Path:
    """Create result/round22_1 shared folders. Never touches result/round22 or result/round21*."""
    ROUND22_1_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ROUND22_1_SHARED_DIRS:
        (ROUND22_1_ROOT / name).mkdir(parents=True, exist_ok=True)
    return ROUND22_1_ROOT


def assert_project_result_path(path: Path) -> Path:
    """Refuse writes outside this 课题 result/. Stock src/runs and other 课题 stay forbidden.

    This project lives under /home/mxy/Stock/课题, so a blanket Stock-root ban would
    block the local result tree. Only RESULT_ROOT is the writable exception.
    """
    resolved = path.resolve()
    result_root = RESULT_ROOT.resolve()
    if resolved == result_root or result_root in resolved.parents:
        return resolved
    raise RuntimeError(f"refusing to write outside {result_root}: {resolved}")
