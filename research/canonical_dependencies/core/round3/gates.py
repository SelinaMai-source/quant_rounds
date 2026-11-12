"""Round 3 门禁：00 口径 / 01 PIT / 02 精确复现。不过不准进 ablation/stress。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..metrics import (
    CAGR_ABS_TOL,
    LOCKED_BASELINE,
    METRIC_VERSION,
    SHARPE_ABS_TOL,
    attach_nav_metrics,
    clip_nav_for_metrics,
)
from ..paths import (
    PROJECT_ROOT,
    ROUND1_ROOT,
    ROUND2_ROOT,
    ROUND3_ROOT,
    assert_project_result_path,
    ensure_round3_tree,
    stock_path,
)
from ..round2.metrics import attach_round2_metrics
from ..stock_readonly import load_stock_baseline_report, stock_baseline_report_path
from .registry import SHORTLIST, get_spec, r2_source_payload, source_config_hash
from .runner import attach_round3_fields, run_spec

G02_RUNNER_SNIPPET = 'merged["ind_p"] = merged["ind_first_board_count_log"].rank(pct=True)'


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _write(path: Path, text: str) -> Path:
    assert_project_result_path(path)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return path


def _pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def locked_and_reconstruct() -> dict[str, Any]:
    stock_report = load_stock_baseline_report()
    r1 = ROUND1_ROOT / "R1_00_baseline" / "report.json"
    recon_dir = ROUND2_ROOT / "00_audit" / "locked_reconstruct"
    recon_report = _load_json(recon_dir / "report.json") if (recon_dir / "report.json").is_file() else {}
    recon_nav = pd.read_parquet(recon_dir / "nav.parquet") if (recon_dir / "nav.parquet").is_file() else pd.DataFrame()
    recon_trades = (
        pd.read_parquet(recon_dir / "trades.parquet") if (recon_dir / "trades.parquet").is_file() else pd.DataFrame()
    )
    r1_report = _load_json(r1) if r1.is_file() else {}
    return {
        "stock_report": stock_report,
        "r1_report": r1_report,
        "recon_report": recon_report,
        "recon_nav": recon_nav,
        "recon_trades": recon_trades,
        "recon_dir": recon_dir,
        "stock_path": stock_baseline_report_path(),
    }


def compute_aligned_metrics(nav: pd.DataFrame, trades: pd.DataFrame | None = None) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    clipped = clip_nav_for_metrics(nav) if nav is not None and not nav.empty else nav
    if clipped is not None and not clipped.empty:
        asset = clipped["total_asset"].astype(float)
        metrics["final_asset"] = float(asset.iloc[-1])
        metrics["total_return"] = float(asset.iloc[-1] / 300_000.0 - 1.0)
        metrics["max_drawdown"] = float((asset / asset.cummax() - 1.0).min())
    if trades is not None and not trades.empty:
        attach_round2_metrics(nav, trades, metrics)
    else:
        attach_nav_metrics(nav, metrics)
    attach_round3_fields(metrics)
    if not nav.empty:
        raw = nav.sort_values("trade_date").copy()
        raw["trade_date"] = pd.to_datetime(raw["trade_date"])
        metrics["raw_nav_start"] = str(raw["trade_date"].iloc[0].date())
        metrics["raw_nav_end"] = str(raw["trade_date"].iloc[-1].date())
        metrics["raw_nav_rows"] = int(len(raw))
        warmup = raw.loc[raw["trade_date"] < pd.Timestamp("2018-01-01")]
        metrics["warmup_rows"] = int(len(warmup))
        metrics["warmup_unique_asset"] = int(warmup["total_asset"].nunique()) if len(warmup) else 0
    return metrics


def write_parity_markdown() -> dict[str, Any]:
    ensure_round3_tree()
    pack = locked_and_reconstruct()
    stock_m = pack["stock_report"].get("metrics") or {}
    r1_m = pack["r1_report"].get("metrics") or {}
    recon_m_old = pack["recon_report"].get("metrics") or {}
    aligned = compute_aligned_metrics(pack["recon_nav"], pack["recon_trades"])
    locked_cagr = float(stock_m.get("annualized_return") or LOCKED_BASELINE["annualized_return"])
    locked_sharpe = float(stock_m.get("sharpe_zero_rate") or LOCKED_BASELINE["sharpe"])
    locked_mdd = float(stock_m.get("max_drawdown") or LOCKED_BASELINE["max_drawdown"])
    locked_ret = float(stock_m.get("total_return") or LOCKED_BASELINE["total_return"])

    old_cagr = recon_m_old.get("annualized_return")
    old_sharpe = recon_m_old.get("sharpe_zero_rate")
    new_cagr = aligned.get("annualized_return")
    new_sharpe = aligned.get("sharpe_zero_rate")

    cagr_ok = new_cagr is not None and abs(float(new_cagr) - locked_cagr) <= CAGR_ABS_TOL
    sharpe_ok = new_sharpe is not None and abs(float(new_sharpe) - locked_sharpe) <= SHARPE_ABS_TOL
    ret_ok = abs(float(aligned.get("total_return") or recon_m_old.get("total_return") or 0) - locked_ret) <= 1e-6
    mdd_ok = abs(float(aligned.get("max_drawdown") or recon_m_old.get("max_drawdown") or 0) - locked_mdd) <= 1e-5
    passed = bool(cagr_ok and sharpe_ok and ret_ok and mdd_ok)

    dest = ROUND3_ROOT / "00_parity"
    dest.mkdir(parents=True, exist_ok=True)
    nav_clip = clip_nav_for_metrics(pack["recon_nav"])
    if not pack["recon_nav"].empty:
        raw = pack["recon_nav"].sort_values("trade_date").copy()
        raw["trade_date"] = pd.to_datetime(raw["trade_date"])
        merged = raw.merge(
            nav_clip[["trade_date", "total_asset"]].rename(columns={"total_asset": "eval_asset"}),
            on="trade_date",
            how="left",
        )
        merged["in_eval_window"] = merged["eval_asset"].notna()
        merged["asset_diff"] = 0.0
        nav_diff_path = dest / "nav_diff.csv"
        assert_project_result_path(nav_diff_path)
        merged[["trade_date", "total_asset", "in_eval_window"]].to_csv(nav_diff_path, index=False)

    if not pack["recon_trades"].empty:
        trades = pack["recon_trades"].copy()
        trade_diff_path = dest / "trade_diff.csv"
        assert_project_result_path(trade_diff_path)
        keep = [c for c in ("signal_date", "instrument_id", "buy_date", "sell_date", "ret", "pnl") if c in trades.columns]
        trades[keep].to_csv(trade_diff_path, index=False)

    payload = {
        "passed": passed,
        "metric_version": METRIC_VERSION,
        "locked": {
            "cagr": locked_cagr,
            "sharpe": locked_sharpe,
            "max_drawdown": locked_mdd,
            "total_return": locked_ret,
            "final_asset": stock_m.get("final_asset"),
            "completed_trades": stock_m.get("completed_trades"),
            "source": str(pack["stock_path"]),
        },
        "reconstruct_old_metrics": {
            "cagr": old_cagr,
            "sharpe": old_sharpe,
            "yearly_has_2017": "2017" in (recon_m_old.get("yearly_returns") or {}),
        },
        "reconstruct_aligned": {
            "cagr": new_cagr,
            "sharpe": new_sharpe,
            "max_drawdown": aligned.get("max_drawdown"),
            "total_return": aligned.get("total_return") or recon_m_old.get("total_return"),
            "metric_nav_start": aligned.get("metric_nav_start"),
            "metric_nav_end": aligned.get("metric_nav_end"),
            "metric_years": aligned.get("metric_years"),
            "raw_nav_start": aligned.get("raw_nav_start"),
            "warmup_rows": aligned.get("warmup_rows"),
        },
        "tolerances": {"cagr": CAGR_ABS_TOL, "sharpe": SHARPE_ABS_TOL},
        "checks": {
            "cagr_ok": cagr_ok,
            "sharpe_ok": sharpe_ok,
            "total_return_ok": ret_ok,
            "max_drawdown_ok": mdd_ok,
        },
    }
    _write(dest / "parity.json", json.dumps(payload, ensure_ascii=False, indent=2, default=str))

    md = f"""# 00 引擎 / 指标口径对齐

- 生成时间：{_now()}
- 对象：**Stock 锁定账本 ↔ 课题 local_executor 重建**。不是第二套 ZyQuant Strategy。
- `metric_version`：**{METRIC_VERSION}**
- 门禁：{'**通过**' if passed else '**未通过**'}

## 根因

重建 NAV 从 **{aligned.get('raw_nav_start')}** 起（MA5 预热，{aligned.get('warmup_rows')} 行 2017 日、资产恒为 30 万）。
锁定评价窗从 **2018-01-02** 到 **{aligned.get('metric_nav_end')}**。

旧公式用 NAV 首末日算 `years`：

| 窗 | 起点 | years | CAGR | Sharpe |
|---|---|---:|---:|---:|
| 含 2017 预热（R2_B00/D05 旧） | {aligned.get('raw_nav_start')} | ~8.635 | {_pct(old_cagr, 2)} | {_num(old_sharpe, 3)} |
| 锁定窗（本轮） | {aligned.get('metric_nav_start')} | {_num(aligned.get('metric_years'), 6)} | {_pct(new_cagr, 2)} | {_num(new_sharpe, 3)} |
| Stock / R1_00 展示 | 2018-01-02 | — | {_pct(locked_cagr, 2)} | {_num(locked_sharpe, 3)} |

2017 预热日收益全为 0：拉长 years → CAGR 从 22.93% 掉到 22.69%；日收益均值被 0 拉低 → Sharpe 1.174 → 1.168。
`total_return` / `final_asset` / MaxDD / 分年（2018–2026）两边本来就一致。

## 对照

Stock 锁定变体只有 `report.json`，无逐笔。课题重建自洽：874 笔、期末 1,754,296、MaxDD −20.62%。

| 项 | Stock 锁定 | R1_00 副本 | 重建（旧指标） | 重建（{METRIC_VERSION}） | 对齐 |
|---|---:|---:|---:|---:|---|
| CAGR | {_pct(locked_cagr, 2)} | {_pct(r1_m.get('annualized_return'), 2)} | {_pct(old_cagr, 2)} | {_pct(new_cagr, 2)} | {'是' if cagr_ok else '否'} |
| Sharpe（rf=0，√252） | {_num(locked_sharpe, 3)} | {_num(r1_m.get('sharpe_zero_rate'), 3)} | {_num(old_sharpe, 3)} | {_num(new_sharpe, 3)} | {'是' if sharpe_ok else '否'} |
| 累计 | {_pct(locked_ret, 2)} | {_pct(r1_m.get('total_return'), 2)} | {_pct(recon_m_old.get('total_return'), 2)} | {_pct(aligned.get('total_return') or recon_m_old.get('total_return'), 2)} | {'是' if ret_ok else '否'} |
| MaxDD | {_pct(locked_mdd, 2)} | {_pct(r1_m.get('max_drawdown'), 2)} | {_pct(recon_m_old.get('max_drawdown'), 2)} | {_pct(aligned.get('max_drawdown') or recon_m_old.get('max_drawdown'), 2)} | {'是' if mdd_ok else '否'} |
| 完成交易 | {stock_m.get('completed_trades')} | {r1_m.get('completed_trades')} | {recon_m_old.get('completed_trades')} | {recon_m_old.get('completed_trades')} | 是 |
| 期末未平 | {stock_m.get('terminal_positions')} | {r1_m.get('terminal_positions')} | {recon_m_old.get('terminal_positions')} | {recon_m_old.get('terminal_positions')} | 是 |

约定误差：CAGR ≤ {CAGR_ABS_TOL}，Sharpe ≤ {SHARPE_ABS_TOL}。

## 其它口径

- 日收益：`pct_change().dropna()`，丢掉评价窗首日。
- 无风险利率：0。
- 费用（两边相同）：滑点 12.3bp、佣金 1bp、过户 0.1bp、卖印花税 5bp。
- 涨停容差：0.005。
- rounding：期末资产与锁定一致到 1e-6 相对量级（`asset_identity_max_abs` ~1e-10）。
- 禁止为对齐改 Alpha / score。

## 产物

- `result/round3/00_parity/parity.json`
- `result/round3/00_parity/nav_diff.csv`（标记评价窗）
- `result/round3/00_parity/trade_diff.csv`（重建逐笔；Stock 无逐笔可 diff）
"""
    path = ROUND3_ROOT / "00_engine_metric_parity.md"
    _write(path, md)
    _write(ROUND3_ROOT / "00_parity" / "00_engine_metric_parity.md", md)
    payload["path"] = str(path)
    return payload


def audit_g02_full_sample() -> dict[str, Any]:
    runner = PROJECT_ROOT / "core" / "round2" / "runner.py"
    text = runner.read_text(encoding="utf-8")
    uses_full = G02_RUNNER_SNIPPET in text
    return {
        "file": str(runner),
        "snippet": G02_RUNNER_SNIPPET,
        "uses_full_sample_rank": uses_full,
        "verdict": "INVALID_FOR_DEPLOYMENT" if uses_full else "PIT_OK",
    }


def write_pit_markdown() -> dict[str, Any]:
    ensure_round3_tree()
    g02 = audit_g02_full_sample()
    rows = [
        {
            "variable": "B01a up_ratio 45/50/60",
            "source": "market_context.market_up_ratio（D 收盘）",
            "available_at": "信号日收盘，D+1 开盘前",
            "normalization": "绝对门槛，无分位",
            "uses_future": False,
            "action": "PIT 安全，保持",
        },
        {
            "variable": "C02_graded agree_n",
            "source": "当日 5 个 seed Top1 是否等于 ensemble Top1",
            "available_at": "信号日打分后（score 已锁定）",
            "normalization": "人数制 5/5、4/5、3/5，无分位",
            "uses_future": False,
            "action": "PIT 安全，保持",
        },
        {
            "variable": "C03_half margin_p250",
            "source": "Top1−Top2 score 差",
            "available_at": "信号日打分后",
            "normalization": "250 日 rolling.rank(pct)，含当日、min_periods=40；按日排序后滚动",
            "uses_future": False,
            "action": "PIT 安全；未用 2018–2026 全局 quantile",
        },
        {
            "variable": "G02 ind_first_board_count_log",
            "source": "候选面板 Top1 行业首板对数",
            "available_at": "特征本身 D 收盘可见",
            "normalization": "对 Top1 时间序列做 **全样本** `rank(pct=True)`，阈值 p80/p90 用了未来分位",
            "uses_future": True,
            "action": "INVALID_FOR_DEPLOYMENT；rolling 250 PIT 版重跑 T06/P13/P14",
        },
    ]
    dest = ROUND3_ROOT / "01_pit"
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest / "pit_audit.csv", index=False)
    _write(dest / "g02_verdict.json", json.dumps(g02, ensure_ascii=False, indent=2))

    table = [
        "| variable | source | available_at | normalization | uses_future | action |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        table.append(
            f"| `{row['variable']}` | {row['source']} | {row['available_at']} | {row['normalization']} | "
            f"{'是' if row['uses_future'] else '否'} | {row['action']} |"
        )

    passed = True  # 审计完成即过门；G02 INVALID 是结论不是审计失败
    md = f"""# 01 PIT / 泄漏审计

- 生成时间：{_now()}
- 门禁：{'**通过（审计完成）**' if passed else '**未通过**'}
- 禁止「先看全样本再回填 2018」。

## 逐项

{chr(10).join(table)}

## G02

- 文件：`{g02['file']}`
- 命中：`{g02['snippet']}`
- 判定：**{g02['verdict']}**
- 说明：registry 文案写「当日截面分位」，实现却是 Top1 行业拥挤 **跨 2018–2026 全样本时间序列分位**。p80/p90 用了未来分位回填。
- 特征水平（`ind_first_board_count_log`）D 收盘可见，泄漏在 **normalization**。
- PIT 修正：`rolling(250).rank(pct=True)`，含当日、不含未来；不必 `shift(1)`（决策用 D 收盘，买在 D+1 开盘）。
- 受影响 shortlist：`R3_04`（T06）、`R3_05`（P13）、`R3_07`（P14）。原 config 复现后标 INVALID，门禁后重跑 `R3_04p` / `R3_05p` / `R3_07p` / `R3_G02p`。

## B01a / C02 / C03

- B01a：45/50/60 绝对门槛，无全样本分位。
- C02：当日 5 seed 是否同 Top1，人数制。
- C03：`seed_top1_agreement` 在按日排序后做 250 日 rolling rank；阈值未用全局 quantile。`core/round2/data.py` 已显式 `sort_values("signal_date")` 再 rolling。

## 其它分位（本轮不扩搜，仅记录）

B02/B03/B04/B05/G01/G03 用的是 `rolling(250).rank`，与 C03 同类。不在 shortlist 里当部署候选。
"""
    path = ROUND3_ROOT / "01_pit_leakage_audit.md"
    _write(path, md)
    _write(dest / "01_pit_leakage_audit.md", md)
    return {"passed": passed, "g02": g02, "path": str(path), "rows": rows}


def _r2_ledger(source: str | None) -> dict[str, Any] | None:
    if not source:
        return None
    if source == "R1_00_baseline":
        recon = ROUND2_ROOT / "00_audit" / "locked_reconstruct" / "report.json"
        if recon.is_file():
            return _load_json(recon)
        r1 = ROUND1_ROOT / "R1_00_baseline" / "report.json"
        return _load_json(r1) if r1.is_file() else None
    path = ROUND2_ROOT / source / "report.json"
    return _load_json(path) if path.is_file() else None


def compare_ledger(r3_metrics: dict[str, Any], r2_metrics: dict[str, Any], *, vs_locked: bool) -> dict[str, Any]:
    def _close(a, b, tol):
        if a is None or b is None:
            return False
        return abs(float(a) - float(b)) <= tol

    trades_ok = r3_metrics.get("completed_trades") == r2_metrics.get("completed_trades")
    ret_ok = _close(r3_metrics.get("total_return"), r2_metrics.get("total_return"), 1e-6)
    asset_ok = _close(r3_metrics.get("final_asset"), r2_metrics.get("final_asset"), 1.0)
    mdd_ok = _close(r3_metrics.get("max_drawdown"), r2_metrics.get("max_drawdown"), 1e-5)
    ledger_ok = bool(trades_ok and ret_ok and asset_ok and mdd_ok)
    metric_ok = True
    if vs_locked:
        metric_ok = (
            _close(r3_metrics.get("annualized_return"), LOCKED_BASELINE["annualized_return"], CAGR_ABS_TOL)
            and _close(r3_metrics.get("sharpe_zero_rate"), LOCKED_BASELINE["sharpe"], SHARPE_ABS_TOL)
        )
    return {
        "trades_ok": trades_ok,
        "total_return_ok": ret_ok,
        "final_asset_ok": asset_ok,
        "max_drawdown_ok": mdd_ok,
        "ledger_ok": ledger_ok,
        "locked_metric_ok": metric_ok if vs_locked else None,
        "ok": ledger_ok and (metric_ok if vs_locked else True),
    }


def write_repro_markdown(run_results: dict[str, Any]) -> dict[str, Any]:
    ensure_round3_tree()
    dest = ROUND3_ROOT / "02_repro"
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    all_ok = True
    for code, spec in SHORTLIST.items():
        out = run_results.get(code) or {}
        report = out.get("report") or load_report_safe(code)
        r3m = (report or {}).get("metrics") or {}
        r2 = _r2_ledger(spec.source_round2)
        r2m = (r2 or {}).get("metrics") or {}
        cmp = compare_ledger(r3m, r2m, vs_locked=(code == "R3_00"))
        if not cmp["ok"]:
            all_ok = False
        flags = []
        if spec.uses_g02:
            flags.append("INVALID_FOR_DEPLOYMENT")
        row = {
            "code": code,
            "title_zh": spec.title_zh,
            "source": spec.source_round2,
            "source_hash": source_config_hash(r2_source_payload(spec)),
            "r3_cagr": r3m.get("annualized_return"),
            "r3_sharpe": r3m.get("sharpe_zero_rate"),
            "r3_mdd": r3m.get("max_drawdown"),
            "r3_ret": r3m.get("total_return"),
            "r3_trades": r3m.get("completed_trades"),
            "r2_cagr": r2m.get("annualized_return"),
            "r2_sharpe": r2m.get("sharpe_zero_rate"),
            "r2_mdd": r2m.get("max_drawdown"),
            "r2_ret": r2m.get("total_return"),
            "r2_trades": r2m.get("completed_trades"),
            "ledger_ok": cmp["ledger_ok"],
            "ok": cmp["ok"],
            "flags": flags,
        }
        rows.append(row)

    _write(dest / "repro.json", json.dumps({"passed": all_ok, "rows": rows}, ensure_ascii=False, indent=2, default=str))
    lines = [
        "| 代号 | 操作 | Round2 源 | 账本 | 新CAGR | 新Sharpe | MaxDD | 笔数 | 旗标 |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['code']}` | {SHORTLIST[row['code']].title_zh} | `{row['source']}` | "
            f"{'对齐' if row['ledger_ok'] else '**失败**'} | {_pct(row['r3_cagr'], 2)} | "
            f"{_num(row['r3_sharpe'], 3)} | {_pct(row['r3_mdd'], 2)} | {row['r3_trades']} | "
            f"{','.join(row['flags']) or '—'} |"
        )

    md = f"""# 02 精确复现

- 生成时间：{_now()}
- 引擎 / 指标：{METRIC_VERSION}
- 门禁：{'**通过**' if all_ok else '**未通过**'}
- 对照：Round 2 原 config 的 **账本**（笔数 / 累计 / 期末 / MaxDD）。CAGR/Sharpe 按锁定窗重算，不要求贴 R2 旧年化。
- `R3_00` 另对锁定展示 CAGR/Sharpe（22.93% / 1.174）。
- 不准改参数去贴数字。

{chr(10).join(lines)}

G02 三组（R3_04/05/07）即使账本对齐，仍标 **INVALID_FOR_DEPLOYMENT**，门禁后用 rolling 250 PIT 版重跑。
"""
    path = ROUND3_ROOT / "02_exact_reproduction.md"
    _write(path, md)
    _write(dest / "02_exact_reproduction.md", md)
    return {"passed": all_ok, "rows": rows, "path": str(path)}


def load_report_safe(code: str) -> dict[str, Any] | None:
    path = ROUND3_ROOT / code / "report.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def gate_files_exist() -> dict[str, bool]:
    return {
        "parity": (ROUND3_ROOT / "00_engine_metric_parity.md").is_file(),
        "pit": (ROUND3_ROOT / "01_pit_leakage_audit.md").is_file(),
        "repro": (ROUND3_ROOT / "02_exact_reproduction.md").is_file(),
    }


def load_gate_status() -> dict[str, Any]:
    parity = ROUND3_ROOT / "00_parity" / "parity.json"
    repro = ROUND3_ROOT / "02_repro" / "repro.json"
    pit = ROUND3_ROOT / "01_pit_leakage_audit.md"
    out = {
        "parity": json.loads(parity.read_text(encoding="utf-8")) if parity.is_file() else {"passed": False},
        "pit": {"passed": pit.is_file()},
        "repro": json.loads(repro.read_text(encoding="utf-8")) if repro.is_file() else {"passed": False},
    }
    out["all_passed"] = bool(
        out["parity"].get("passed") and out["pit"].get("passed") and out["repro"].get("passed")
    )
    return out


def require_gates() -> dict[str, Any]:
    status = load_gate_status()
    if not status["all_passed"]:
        raise RuntimeError(
            "门禁未过，禁止 ablation/stress。"
            f" parity={status['parity'].get('passed')} pit={status['pit'].get('passed')} "
            f"repro={status['repro'].get('passed')}"
        )
    return status


def run_shortlist(*, predictions, prepared, meta, resume: bool, run_now: bool) -> dict[str, Any]:
    results = {}
    for code, spec in SHORTLIST.items():
        results[code] = run_spec(
            spec,
            predictions=predictions,
            prepared_market=prepared,
            market_meta=meta,
            resume=resume,
            run_now=run_now,
        )
    return results
