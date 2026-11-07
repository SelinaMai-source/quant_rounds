# 订单 / 成交 / 持仓谱系

```
overlayed scores (post B03/O01)
    → local_executor 构造 fixed-sleeve 订单意图
    → fixed_sleeve_backtest 按 D+1 open 尝试买入
    → 拒买：开盘一字涨停等
    → 持仓：双袖套独立现金
    → 卖出：close_from_d2 / limit_up_hold
    → fills/trades/nav parquet + report.json
```

权威回放目录：`result/replay_r7_caliber/v5_R16_O01/`  
交叉：`result/round22_3/parity/R223_V5_R16_O01/`（服务器课题树）。

分配乘数：clean_risk 缩放 × B03 Top2 份额 × O01 Top2 veto。  
**无补单。** 未做实盘成交核验。
