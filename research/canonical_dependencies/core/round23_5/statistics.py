"""配对日收益的区块自助法。SPA 只调用 arch.bootstrap.SPA，不把 White RC 改名。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def _maxdd(net: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peak - 1.0))


def _ulcer(net: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + net)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    return float(np.sqrt(np.mean(np.square(drawdown))))


def _sharpe(net: np.ndarray) -> float:
    vol = float(np.std(net, ddof=1))
    if vol <= 0:
        return 0.0
    return float(np.mean(net) / vol * np.sqrt(252.0))


def _cagr(net: np.ndarray) -> float:
    wealth = float(np.prod(1.0 + net))
    years = max(len(net) / 252.0, 1e-9)
    return float(wealth ** (1.0 / years) - 1.0)


def block_indices(n_obs: int, block: int, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_obs / block))
    starts = rng.integers(0, n_obs, size=(draws, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n_obs
    return idx.reshape(draws, n_blocks * block)[:, :n_obs]


def paired_bootstrap(base: pd.Series, cand: pd.Series, *, draws: int = 5000, block: int = 20, seed: int = 17) -> dict:
    joined = pd.concat([base.rename("base"), cand.rename("cand")], axis=1, join="inner").dropna()
    if len(joined) < block * 2:
        return {"status": "TOO_SHORT", "n": int(len(joined))}
    b = joined["base"].to_numpy(dtype=float)
    c = joined["cand"].to_numpy(dtype=float)
    idx = block_indices(len(joined), block, draws, seed)
    metrics = {"mean": [], "sharpe": [], "cagr": [], "maxdd": [], "ulcer": []}
    for sample in idx:
        bb = b[sample]
        cc = c[sample]
        metrics["mean"].append(float(cc.mean() - bb.mean()))
        metrics["sharpe"].append(_sharpe(cc) - _sharpe(bb))
        metrics["cagr"].append(_cagr(cc) - _cagr(bb))
        metrics["maxdd"].append(_maxdd(cc) - _maxdd(bb))
        metrics["ulcer"].append(_ulcer(cc) - _ulcer(bb))
    out = {"status": "OK", "n": int(len(joined)), "block": int(block), "draws": int(draws)}
    for name, values in metrics.items():
        arr = np.asarray(values)
        out[f"delta_{name}"] = float(np.mean(arr))
        out[f"delta_{name}_p05"] = float(np.quantile(arr, 0.05))
        out[f"delta_{name}_p95"] = float(np.quantile(arr, 0.95))
    out["point_delta_mean"] = float(c.mean() - b.mean())
    out["point_delta_sharpe"] = _sharpe(c) - _sharpe(b)
    out["point_delta_cagr"] = _cagr(c) - _cagr(b)
    out["point_delta_maxdd"] = _maxdd(c) - _maxdd(b)
    out["point_delta_ulcer"] = _ulcer(c) - _ulcer(b)
    return out


def spa_test(base: pd.Series, others: dict[str, pd.Series], *, block: int = 20, reps: int = 5000) -> dict:
    try:
        from arch.bootstrap import SPA
    except Exception:
        return _spa_in_isolated_env(base, others, block=block, reps=reps)
    joined = pd.DataFrame({"base": base})
    for name, series in others.items():
        joined[name] = series
    joined = joined.dropna()
    if joined.shape[1] < 2 or len(joined) < block * 2:
        return {"status": "TOO_SHORT", "n": int(len(joined))}
    benchmark = -joined["base"].to_numpy(dtype=float)
    models = -joined.drop(columns=["base"]).to_numpy(dtype=float)
    if models.ndim == 1:
        models = models.reshape(-1, 1)
    try:
        test = SPA(benchmark, models, block_size=block, reps=reps, bootstrap="circular", seed=17)
        test.compute()
        pvalue = test.pvalues
        return {
            "status": "OK",
            "implementation": "arch.bootstrap.SPA",
            "not_white_rc": True,
            "n": int(len(joined)),
            "block": int(block),
            "reps": int(reps),
            "pvalues": {str(key): float(value) for key, value in pvalue.items()},
            "columns": list(joined.drop(columns=["base"]).columns),
            "null": "benchmark is not inferior to the best model; loss is negative daily net return",
        }
    except Exception as exc:
        return {"status": "BLOCKED_STAT_TEST", "reason": str(exc), "implementation": "arch.bootstrap.SPA"}


def _spa_in_isolated_env(base: pd.Series, others: dict[str, pd.Series], *, block: int, reps: int) -> dict:
    """arch 不在当前解释器里时，只调用隔离目录里的 arch，不改生产环境。"""
    import json
    import os
    import subprocess
    import sys
    import tempfile

    python = Path(sys.executable)
    arch_path = Path("/tmp/r235_arch_pkg")
    if not (arch_path / "arch").is_dir() and not Path("/tmp/r235_arch_venv/bin/python").is_file():
        return {"status": "BLOCKED_STAT_TEST", "reason": "isolated arch env missing"}
    if not (arch_path / "arch").is_dir():
        python = Path("/tmp/r235_arch_venv/bin/python")
    joined = pd.DataFrame({"base": base})
    for name, series in others.items():
        joined[name] = series
    joined = joined.dropna()
    if joined.shape[1] < 2 or len(joined) < block * 2:
        return {"status": "TOO_SHORT", "n": int(len(joined))}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "losses.csv"
        joined.to_csv(path, index=False)
        script = f"""
import json
import pandas as pd
from arch.bootstrap import SPA
frame = pd.read_csv({str(path)!r})
benchmark = (-frame['base']).to_numpy()
models = (-frame.drop(columns=['base'])).to_numpy()
test = SPA(benchmark, models, block_size={int(block)}, reps={int(reps)}, bootstrap='circular', seed=17)
test.compute()
print(json.dumps({{
    'status': 'OK',
    'implementation': 'arch.bootstrap.SPA',
    'not_white_rc': True,
    'n': int(len(frame)),
    'block': {int(block)},
    'reps': {int(reps)},
    'pvalues': {{str(k): float(v) for k, v in test.pvalues.items()}},
    'columns': list(frame.drop(columns=['base']).columns),
    'arch_version': __import__('arch').__version__,
}}))
"""
        env = os.environ.copy()
        if arch_path.is_dir():
            env["PYTHONPATH"] = str(arch_path) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        done = subprocess.run([str(python), "-c", script], text=True, capture_output=True, env=env)
    if done.returncode != 0:
        return {"status": "BLOCKED_STAT_TEST", "reason": done.stderr[-500:], "implementation": "arch.bootstrap.SPA"}
    return json.loads(done.stdout.strip().splitlines()[-1])
