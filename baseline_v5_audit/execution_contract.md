# baseline_v5 执行契约

## 引擎链

权威 v5 回放：`dual90_10_gap_floor` → B03 overlay → O01 veto → `local_executor` → `fixed_sleeve_backtest`。

研究入口（同执行器）：`research_sleeve.run_heads_sleeve` → `fuse_heads` → `apply_b03` → `apply_top2_veto(O01)` → `round20.parity._run_r7` → `local_executor`。

**不含** Round7 clean_risk 预测乘子（文档曾误写 → 见 `b03_o01_contract.md`）。

## 时点
| 阶段 | 时点 | 信息 |
|---|---|---|
| 特征（除 gap） | D close | 收盘可知 |
| d1_open_gap | D+1 09:25/open | OPEN_INFORMATION |
| 买入 | D+1 open | 同开盘成交假设 |
| 卖出 | D+2 起首个非涨停收盘 | close_from_d2；涨停顺延 |

## 成交假设（回测，非实盘核验）
- 开盘一字涨停拒买（容差 0.005）
- 无补单（no refill）
- 涨停持有 / 一字跌停延迟按 Stock 引擎
- **不声称已验证实盘成交**

## 组合
- 初始 300000；双袖套各 150000
- TopK=2；预算 Top1 30% NAV / Top2 10% NAV
- lot_size=100
