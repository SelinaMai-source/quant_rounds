# FINAL · baseline_v5 冻结身份

## 冻结声明

| 字段 | 值 |
|---|---|
| identity | baseline_v5 |
| alias | R16_O01 |
| parent | R9_B03 |
| fee_schedule | ROUND7_FLAT_5BP |
| window | 2018-01-02 .. 2026-07-24 |
| initial_capital | 300000.0 |
| Sharpe | 1.8756627913361097 |
| CAGR | 0.5499812832273525 |
| MaxDD | -0.24056327715394388 |
| Terminal NAV | 12750089.355724689 |
| Completed Trades | 1107 |
| CURRENT_BASELINE_CHANGED | **FALSE** |
| PRODUCTION_CONFIG_CHANGED | **FALSE** |

费用分项：滑点 12.3bp + 佣金 1.0bp + 过户 0.1bp + 卖印花税 5.0bp（**不是**总成本 5bp）。

## 如何核验 parity

1. `python tools/run_baseline_v5.py --tests`
2. `python tools/run_baseline_v5.py --parity`
3. 人工打开 `result/replay_r7_caliber/v5_R16_O01/replay_summary.json`，确认五数字一致
4. 交叉：`result/round22_3/parity/R223_V5_R16_O01/replay_summary.json`

## O01 公式

`PUT_SKEW25.shift(1) > rolling(252, min_periods=126).quantile(0.90) → apply_top2_veto rank=2 MIN_KEEP`

信号日清单参考：`result/round16/options/treat_dates.json`。
