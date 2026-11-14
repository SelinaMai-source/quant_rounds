# Round26.1执行状态

D0已执行（覆盖和限制见D0报告）。D3运行中，尚未完成：14/42个完整原始PPO生成折，候选Add回测0；匹配原生基准3种成本情景已执行。D5尚未实施/运行，不属于外部数据阻断。

已完成生成模型和精确资源见D3/completed_generation_models.csv；各模型冻结因子和完整预测见D3/completed_factors，审计见D3/generation_audits。D3/REPORT.md列出实际成绩范围及仍缺证据，STATUS/registry保留失败或未完成状态。当前没有已验收的v5替代方案。

CURRENT_BASELINE_CHANGED=FALSE
PRODUCTION_CONFIG_CHANGED=FALSE

当前完整PPO时间反事实状态：RAW/RES的2020、2023共四项已完成完整预算并通过AST、重估权重及全部预测精确一致性验证；2026两项正在运行，当前4/6。模型、预测、全部表达式尝试及大checkpoint精确路径/大小/SHA见各自generation_counterfactual_tests/*/*/completed_artifacts/manifest.json。该诊断不新增经济试验。

比较范围：预测候选截至2026-07-01，07-24是估值及退出尾部；原v5冻结预测截至06-01，真实无因子控制补入655键，公共16764键融合分数完全一致。主比较采用匹配覆盖的真实无因子控制（8个head/year配置、40棵XGB及3个原生成本情景），不将覆盖变化当因子收益。详见CANDIDATE_COVERAGE_CONTRACT.md及baseline_head_replay/coverage_difference_evidence.json。
