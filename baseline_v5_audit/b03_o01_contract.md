# B03 / O01 / clean_risk 契约

## 权威 v5 执行链（源码）

```
dual90_10_gap_floor（融合分）
  → B03_conservative overlay（含 B01a×截面 date_mult + 九格 Top2）
  → O01 apply_top2_veto
  → Fixed-Sleeve / local_executor
```

证据：`core/replay_round7/inject.py`（`load_locked_alpha` 读 dual90_10，**不用** clean_risk pred）；`USED_LOCKED_ROUND7_RISK=False`；`core/round20/parity.py`；`tools/replay_r7_caliber.py`。

## BASELINE_CONTRACT_MISMATCH：clean_risk

| 声明方 | 内容 |
|---|---|
| 发布文档 / `baseline_v5/contract.py` PORTFOLIO | 写「90/10 + **clean_risk** + B03 + O01」或 `clean_risk=True` |
| 源码事实 | v5 权威回放**不消费** `round7` clean_risk 预测文件；clean_risk 规则仍存在于 `core/round7/risk.py`，但**不是** v5 父链一步 |

**不改代码强行对齐。** 审计以 inject/parity/replay 为准。

注：B01a 乘子集合也可产生 0.25/0.375，**不能**据此推断走了 clean_risk。

## clean_risk（仅规则库存，非 v5 链）

竞价/风格健康度缩放，乘数 ∈ {0.25, 0.375, 0.5, 0.75, 1.0}（`core/round7/risk.py`）。v5 冻结 NAV **未**经该预测乘子。

## B03 九格（Top2 NAV 份额）

B03 overlay 先有 **date_mult**（B01a × top1−median 截面硬半），再对 Top2 应用九格；Top1 只乘 date_mult。

来源 `core/round9` / `core/baseline_v5/contract.py::B03_CELL`：

| agreement \ separation | LOW | MID | HIGH |
|---|---:|---:|---:|
| A | 0.00 | 0.05 | 0.10 |
| B | 0.00 | 0.00 | 0.05 |
| C | 0.00 | 0.00 | 0.00 |

sep：`p≤0.20→LOW；p>0.50→HIGH；else MID`。

## O01

`PUT_SKEW25.shift(1) > rolling(252, min_periods=126).quantile(0.90) → apply_top2_veto`：信号日将 **rank=2** 置 `MIN_KEEP≈1e-12`，**Top1 保留**。

信号日清单：`result/round16/options/treat_dates.json`。
