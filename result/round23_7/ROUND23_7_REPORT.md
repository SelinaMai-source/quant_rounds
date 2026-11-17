# Round23.7

D0、D1与D2已实际执行和审计，保留报告中的遗留与资源计量限制。N-U/N-R、P-EQUAL/P-FIXED均未通过冻结收益门。

|方法|CAGR|Sharpe|MaxDD|
|---|---:|---:|---:|
|v5|54.9981%|1.8757|-24.0563%|
|N-U|54.5645%|1.8573|-23.2892%|
|N-R|54.0999%|1.8554|-23.3221%|
|P-SCORE|55.3927%|1.9179|-24.4437%|
|P-EQUAL|38.6096%|1.7653|-23.0631%|
|P-FIXED|50.1003%|1.8354|-22.8893%|

完整交付报告：D0/REPORT.md、D1/REPORT.md、D2/REPORT.md及[D4主报告](D4/REPORT.md)。是否形成替代候选以各自冻结门与完整证据为准，不自动升级，不扩搜索。

CURRENT_BASELINE_CHANGED=FALSE
PRODUCTION_CONFIG_CHANGED=FALSE

<!-- D4_REAL_EXECUTION_CHECKPOINT -->
D4完整代码、集成、目标数据与缓存修复已实现；公开参考CPU8完整默认运行中，旧CPU2未完成尝试和一次未复现的单批精确断言失败均保留。真实A股目标拟合0、真实回测0、完整参考阶段0；非外部阻断，未标完成。当前无可靠替代v5的完整证据。

<!-- TARGET_2018_SIX_FULL_MODELS_COMPLETE_20261117 -->
2026-11-17T00:47:06.451219+00:00：seed17/2018 Five、Ten各自init/T0/T1共6次完整默认预算训练完成，详见TARGET_PROGRESS_2018_COMPLETE.csv及各FIT_AUDIT/checkpoint/forecast。两个Head的T0/T1分别共享同一预训练骨干，且冻结权重均未改变。每个模型均输出完整3241条Outer预测，无有效输入预测缺失。模型关键源码8文件再次逐字节SHA核验通过，见RUNNING_TARGET_CODE_LOCK_AUDIT_20261117.json。2019年两路init已开始。这里只是中间检查点，完整174模型、6个真实冷拟合案例、7个原生反事实账户及24个经济配置尚未全部完成；TARGET_OOF_COMPLETE/BACKTEST_COMPLETE/EVIDENCE_COMPLETE仍为false，不宣称A股绩效或baseline改进。历史0模型段落为此前带日期的运行快照，不是当前计数。

<!-- TARGET_2019_SIX_FULL_MODELS_COMPLETE_20261117 -->
2026-11-17T01:26:23.404730+00:00：seed17/2019 Five/Ten各自init/T0/T1六次完整训练均已完成，累计2018–2019十二次。详见TARGET_PROGRESS_2019_COMPLETE.csv及各模型完整checkpoint/forecast/FIT_AUDIT。每个模型保留完整4711条Outer预测，无有效输入预测缺失；各Head的T0/T1共享骨干且冻结权重未改变。关键8源码锁再次SHA全部匹配。2020年两路init已继续运行。完整年度/seed OOF、真实冷拟合和账户回测仍未完成，不据此宣称A股绩效或baseline改进。

<!-- TARGET_2020_SIX_FULL_MODELS_COMPLETE_20261117 -->
2026-11-17T03:31:18.300736+00:00：seed17/2020 Five/Ten各自init/T0/T1六次完整默认训练完成，累计2018–2020十八次。详见TARGET_PROGRESS_2020_COMPLETE.csv和各模型checkpoint/forecast/FIT_AUDIT。每个模型保留完整5863条Outer预测，无有效输入预测缺失；各Head的T0/T1共享骨干且冻结权重未改变。关键8源码锁再次SHA全部匹配。2021年两路init已继续运行。Ten单头长时训练完整保留默认500cap/20patience预算，没有因耗时缩短轮次。完整年度/seed OOF、真实冷拟合与原生账户仍未完成，不据此宣称A股绩效或baseline改进。

<!-- TARGET_2021_SIX_FULL_MODELS_COMPLETE_20261117 -->
2026-11-17T05:53:51.245639+00:00：seed17/2021 Five/Ten各自init/T0/T1六次完整默认训练完成，累计2018–2021二十四次。详见TARGET_PROGRESS_2021_COMPLETE.csv及各模型checkpoint/forecast/FIT_AUDIT。每个模型保留完整5749条Outer预测，无有效输入预测缺失；各Head的T0/T1共享骨干且冻结权重未改变。关键8源码锁再次SHA全部匹配。2022年两路init已继续运行。完整年度/seed OOF、真实冷拟合与原生账户仍未完成，不据此宣称A股绩效或baseline改进。

<!-- TARGET_2022_SIX_FULL_MODELS_COMPLETE_20261117 -->
2026-11-17T09:09:55.598640+00:00：seed17/2022 Five/Ten各自init/T0/T1六次完整默认训练完成，累计2018–2022三十次。详见TARGET_PROGRESS_2022_COMPLETE.csv及各模型checkpoint/forecast/FIT_AUDIT。每个模型保留完整5600条Outer预测，无有效输入预测缺失；各Head的T0/T1共享骨干且冻结权重未改变。关键8源码锁再次SHA全部匹配。2023年两路init已继续运行。完整年度/seed OOF、真实冷拟合与原生账户仍未完成，不据此宣称A股绩效或baseline改进。

<!-- TARGET_2023_SIX_FULL_MODELS_COMPLETE_20261117 -->
2026-11-17T12:10:03.732454+00:00：seed17/2023 Five/Ten各自init/T0/T1六次完整默认训练完成，累计2018–2023三十六次。详见TARGET_PROGRESS_2023_COMPLETE.csv及各模型checkpoint/forecast/FIT_AUDIT。每个模型保留完整4027条Outer预测，无有效输入预测缺失；各Head的T0/T1共享骨干且冻结权重未改变。关键8源码锁再次SHA全部匹配。2024年两路init已继续运行。完整年度/seed OOF、真实冷拟合与原生账户仍未完成，不据此宣称A股绩效或baseline改进。

<!-- TARGET_2024_SIX_FULL_MODELS_COMPLETE_20261117 -->
2026-11-17T15:04:24.156816+00:00：seed17/2024 Five/Ten各自init/T0/T1六次完整默认训练完成，累计2018–2024四十二次。详见TARGET_PROGRESS_2024_COMPLETE.csv及各模型checkpoint/forecast/FIT_AUDIT。每个模型保留完整6792条Outer预测，无有效输入预测缺失；各Head的T0/T1共享骨干且冻结权重未改变。关键8源码锁再次SHA全部匹配。2025年两路init已继续运行。完整年度/seed OOF、真实冷拟合与原生账户仍未完成，不据此宣称A股绩效或baseline改进。

<!-- PROCESS_INTERRUPTION_RECOVERY_20261118 -->
北京时间2026-11-18续跑审计：此前控制器和两路未完成学习进程已不存在，确切退出信号/退出码不可得；保留两路partial状态与原日志，不计为完整模型或外部阻断。见PROCESS_INTERRUPTION_RECOVERY_20261117_213956.json。新控制器3445952通过原CLI --resume启动，公开完整参考未重训，原45个完整目标模型逐个验证指纹/工件全部命中缓存，见RESUME_CACHE_REENTRY_AUDIT_20261118.json。中断的2025 Five/Ten T0按原seed、初始化权重和完整默认停止条件重跑；旧partial均保留。当前完整目标模型45个，完整OOF/反事实冷拟合/原生账户回测尚未完成，不宣称A股收益改进。
