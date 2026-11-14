# D3 AlphaGen真实A股研究：运行中，未完成

已完成 8/42 个原始PPO生成折，候选Add回测0。仅匹配原生基准S0/S1/S2已实际执行；无可替代v5的新候选结论。

锁定作者原表达式求值、MseAlphaPool、LSTMSharedNet和MaskablePPO；预算化A股适配，不冒称原论文完整CSI300数值复现。RAW/RES目标和原始配置见CONTRACT.yaml，RAW/RES同预算，不因指标差延长或停止。

每折100000环境步或20000次原求值调用先到即停；调用数包括重复和原池拒绝，另报不同AST、原池准入、异常。最后完整PPO优化rollout与未优化尾部环境步分别保存，不将env steps当更新次数。

完整生成折（验证IC仅用于Outer前一年检查点选择，不能当策略成绩）：

```csv
arm,seed,year,env_steps,frozen_ASTs,complete_prediction_rows,missing_factor_rows,selected_validation_IC_not_strategy_performance
AG_RAW,17,2020,100000,20,5863,0,0.11482954770326614
AG_RES,17,2020,100000,20,5863,0,0.09414810687303543
AG_RAW,17,2021,100000,20,5749,0,0.0610123872756958
AG_RES,17,2021,100000,20,5749,0,0.02776629664003849
AG_RAW,17,2022,100000,20,5600,0,0.0323031060397625
AG_RES,17,2022,100000,20,5600,0,0.02832738868892193
AG_RES,17,2023,100000,20,4027,0,0.04973046854138374
AG_RAW,17,2023,100000,20,4027,0,0.07777590304613113
```

匹配原生基准，仅基准成绩（2023-01-03—2026-07-24，独立300000初资、原overlay预热、同原生执行器）：

```csv
experiment_id,CAGR,Sharpe,MaxDD,trades
AG_A0,1.0819480044645982,2.627762557774374,-0.2322636727293504,478
AG_A0_S1,1.0251773871129457,2.537050851917624,-0.236468738044302,478
AG_A0_S2,0.9711932326718802,2.4489772623881505,-0.2409246017992237,478
```

市场OHLCV先在完整股票历史rolling再映射D已知候选。63696条历史候选PIT-ST重算通过；预测不按未来标签筛选。原算子/零分母语义沿用作者实现。未来目标、未成熟目标和历史输入prefix测试已按真实数据入口执行；完整PPO六个反事实重训协议已提交，尚未执行时不得标PASS。

仍缺：全部历年/seed因子、六个完整PPO反事实、历史嵌套OOF入模、Five/Ten/Both Add、触发时Remove-only/Replace、条件OOF与完整风险/费用/Bootstrap/SPA，以及实际XGB入口反事实。不可把未交付改称外部数据阻断。

2010—2012基线预热微观/资金流恢复遇真实认证失败，仅限制更早预热扩展；不阻断当前OHLCV路径。权限清单严格单列。既存v5 next-open/high-low成交与供应商历史版本限制仍保留，不将parity/PIT区间重放当实盘证明。所有历史均为开发期walk-forward。

复现入口、原始协议、source lock、真实日志、generation_audits、完成模型/预测/资源及大checkpoint的精确路径与哈希均保留。三seed只做预定等权，不选赢家。

CURRENT_BASELINE_CHANGED=FALSE
PRODUCTION_CONFIG_CHANGED=FALSE

预测覆盖与控制回放：预测候选清单截至2026-07-01，2026-07-24是估值和退出尾部，并非新增预测覆盖。原v5冻结预测截至2026-06-01；同拟合器真实无因子控制回放新增655个键，公共16764键分数完全一致。主比较采用匹配覆盖的无因子控制，保留原v5参考；详见CANDIDATE_COVERAGE_CONTRACT.md和baseline_head_replay/coverage_difference_evidence.json。实际控制为8个head/year配置、40棵XGB模型及3个原生成本情景；该覆盖修复不算因子收益。
