# baseline_v5 parity 报告

## 冻结参考（prompt） vs 源码产物

| 指标 | 参考 | 产物 | 匹配 |
|---|---:|---:|:---:|
| cagr | 0.5499812832273525 | 0.5499812832273525 | YES |
| sharpe | 1.8756627913361097 | 1.8756627913361097 | YES |
| max_drawdown | -0.24056327715394388 | -0.24056327715394388 | YES |
| nav_end | 12750089.355724689 | 12750089.355724689 | YES |
| initial_capital | 300000.0 | 300000.0 | YES |
| completed_trades | 1107 | 1107 | YES |
| window_start | 2018-01-02 | 2018-01-02 | YES |
| window_end | 2026-07-24 | 2026-07-24 | YES |

**全部数值匹配：True**

## BASELINE_CONTRACT_MISMATCH（非指标）
- `d1_open_gap` 注册表写 `(open/close_D)-1`；builder 为 `open/pre_close-1`。**未改代码。**

## 声明
未声称实盘成交已验证。CURRENT/生产配置未改。
