# Round26.1执行状态

D0已执行（覆盖和限制见D0报告）。D3运行中，尚未完成：14/42个完整原始PPO生成折，候选Add回测0；匹配原生基准3种成本情景已执行。D5尚未实施/运行，不属于外部数据阻断。

已完成生成模型和精确资源见D3/completed_generation_models.csv；各模型冻结因子和完整预测见D3/completed_factors，审计见D3/generation_audits。D3/REPORT.md列出实际成绩范围及仍缺证据，STATUS/registry保留失败或未完成状态。当前没有已验收的v5替代方案。

CURRENT_BASELINE_CHANGED=FALSE
PRODUCTION_CONFIG_CHANGED=FALSE

阶段检查点：seed17的RAW/RES共14个年度生成折已完整生成并审计；全研究仍为14/42，尚无候选Add回测。六个完整PPO时间反事实现已启动2020年两臂，未完成时不得标PASS。真实干预记录见generation_counterfactual_tests/*/*/attempts。

预测覆盖与控制回放：预测候选清单截至2026-07-01，2026-07-24是估值和退出尾部，并非新增预测覆盖。原v5冻结预测截至2026-06-01；真实无因子控制重放新增655个键，公共16764键分数完全一致。主比较采用匹配覆盖的无因子控制，详见CANDIDATE_COVERAGE_CONTRACT.md及baseline_head_replay/coverage_difference_evidence.json；覆盖修复不算因子收益。实际控制为8个head/year配置、40棵XGB模型和3个原生成本情景。

完整PPO时间反事实检查点：RAW2020已完成100000环境步，AST、重估权重与完整预测同原运行精确一致；其余五项尚未完成。实际模型、原预测、配置、全部表达式尝试和大checkpoint精确路径/大小/SHA见generation_counterfactual_tests/AG_RAW/2020/completed_artifacts/manifest.json。未将该诊断作为新增Alpha或收益试验。
