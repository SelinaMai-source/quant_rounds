# baseline_v5 回测契约

## 评价窗与资金

| 项 | 值 |
|---|---|
| 窗 | FULL_HISTORY `2018-01-02` → `2026-07-24` |
| 初始资金 | 300000 RMB |
| 袖套 | 2 × 150000 |
| TopK | 2 |
| 预算 | Top1=30% NAV，Top2=10% NAV；另有固定 name_budget=10000 诊断字段 |
| 退出 | `close_from_d2`（D+2 起首个未涨停收盘卖；涨停 `limit_up_hold`） |
| 引擎 | `local_executor` → `first_board_hermes.fixed_sleeve_backtest.run_fixed_dual_sleeve_top3_backtest` |
| 指标版本 | `r3_locked_window_v1` |

## 费用契约名 `ROUND7_FLAT_5BP`

**名称含 5BP，但不是总成本=5bp。** 实际分项（来自 `round7_fixed_sleeve_config` / 权威 report.json）：

| 分项 | bp |
|---|---:|
| 滑点 slippage（买卖对称） | 12.3 |
| 佣金 commission | 1.0 |
| 过户 transfer | 0.1 |
| 卖出印花税 sell_tax | 5.0 |

官方对照**默认不用** `HISTORICAL_FEE_CORRECTED`（2023-08-28 前卖税 10bp）。该口径另有 replay_v4_v5 数字，不可与下表混用。

## 权威冻结结果（ROUND7_FLAT_5BP）

| 指标 | 值 |
|---|---|
| Sharpe | 1.8756627913361097 |
| CAGR | 0.5499812832273525 |
| MaxDD | -0.24056327715394388 |
| Terminal NAV | 12750089.355724689 |
| Completed Trades | 1107 |
| n_obs（日收益行） | 2075 |

CAGR 定义：日历年 `(final/initial)**(1/years)-1`，`years=(end-start).days/365.25`。  
Sharpe：日收益均值/标准差 × √252，rf=0。

## NAV 对账

- 权威目录：`result/replay_r7_caliber/v5_R16_O01/`
- 交叉核对：`result/round22_3/parity/R223_V5_R16_O01/`
- `asset_identity_max_abs` 应接近 0；costs 分项与 trades/fills 行数见 report.json `frame_rows`
