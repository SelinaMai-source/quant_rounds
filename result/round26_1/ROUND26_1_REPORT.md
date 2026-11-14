# Round26.1执行状态

D0已执行（覆盖和限制见D0报告）。D3运行中，尚未完成：12/42个完整原始PPO生成折，候选Add回测0；匹配原生基准3种成本情景已执行。D5尚未实施/运行，不属于外部数据阻断。

已完成生成模型和精确资源见D3/completed_generation_models.csv；各模型冻结因子和完整预测见D3/completed_factors，审计见D3/generation_audits。D3/REPORT.md列出实际成绩范围及仍缺证据，STATUS/registry保留失败或未完成状态。当前没有已验收的v5替代方案。

CURRENT_BASELINE_CHANGED=FALSE
PRODUCTION_CONFIG_CHANGED=FALSE

预测覆盖与控制回放：预测候选清单截至2026-07-01，2026-07-24是估值和退出尾部，并非新增预测覆盖。原v5冻结预测截至2026-06-01；同拟合器真实无因子控制回放新增655个键，公共16764键分数完全一致。主比较采用匹配覆盖的无因子控制，保留原v5参考；详见CANDIDATE_COVERAGE_CONTRACT.md和baseline_head_replay/coverage_difference_evidence.json。实际控制为8个head/year配置、40棵XGB模型及3个原生成本情景；该覆盖修复不算因子收益。
