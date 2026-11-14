# Round26.1执行状态

D0已执行（覆盖和限制见D0报告）。D3运行中，尚未完成：5/42个完整原始PPO生成折，候选Add回测0；匹配原生基准3种成本情景已执行。D5尚未实施/运行，不属于外部数据阻断。

已完成生成模型和精确资源见D3/completed_generation_models.csv；各模型冻结因子和完整预测见D3/completed_factors，审计见D3/generation_audits。D3/REPORT.md列出实际成绩范围及仍缺证据，STATUS/registry保留失败或未完成状态。当前没有已验收的v5替代方案。

CURRENT_BASELINE_CHANGED=FALSE
PRODUCTION_CONFIG_CHANGED=FALSE


已实际完成A0同口径控制：8个Head年度配置、40个原参数XGB模型以及3种原生成本回测。旧参考预测截至2026-06-01；真实无因子重放截至2026-07-01，增加2026-06-02—07-01的655个键，共同16764键的融合分数完全一致。该差异是预测覆盖，不作为新因子贡献。主比较使用MATCHED_REPAIRED_NO_FACTOR_BASELINE，旧参考保留在legacy_original_window_baseline_results.csv和BASELINE_PREFLIGHT。新S0 CAGR1.081948/Sharpe2.627763/MaxDD-0.232264/478笔；S1 CAGR1.025177，S2 CAGR0.971193。信号覆盖截至07-01，07-24为估值/退出窗口终点，不声称预测股池覆盖到07-24。无因子控制独立训练，无任何空/零占位因子。真正候选Add回测仍0。

调度协调仅暂停了本任务派发进程121.312秒，原PPO子进程未暂停，控制任务完成后已恢复派发；并发训练上限保持2。精确模型/覆盖差异/资源/原始日志及调度证据见baseline_head_replay。
