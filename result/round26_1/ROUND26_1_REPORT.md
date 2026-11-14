# Round26.1

D0已执行。D3当前状态：PARTIAL_SEED_CHECKPOINT；原PPO模型 14，真实匹配回测 13。完整方法成绩、范围与未完成状态见D3/REPORT.md、D3/STATUS.json和all_results.csv。D5尚未在本提交执行。

CURRENT_BASELINE_CHANGED=FALSE
PRODUCTION_CONFIG_CHANGED=FALSE

计数口径：STATUS兼容字段effective_expression_evaluations表示原表达式求值调用次数，含重复及原池拒绝，不是不同AST数、优化更新数或新增经济试验数。不同AST、异常、池准入和PPO更新分别见completed_generation_models.csv及各工件审计。

<!-- LIVE_GENERATION_PROGRESS -->
当前原始生成完成18/42；上方入模、经济门与风险结果仍对应已完成的seed检查点17，不将新生成折冒称新经济回测。已完成当前模型合计1800000环境步、267160次原求值调用（含重复及原池拒绝）；不同AST、异常及PPO更新单列，失败/未完成尝试另存，不计入此完成模型合计。
完整PPO时间反事实六项已通过。原始生成工件审计只验证工件；研究总状态以本STATUS及真实结果表为准。
