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

<!-- AUTHOR_TRAIN_LOG_LOSS_SEMANTICS -->
作者train_epoch的训练日志loss为各batch标量损失之和再除以总预测样本数（有路由时含原OT项），并非总体逐行加权MSE，不能用它的小数量级宣称泛化预测更好。该作者原行为未修改；默认早停仍按validation IC。正式公开参考MSE/MAE来自原evaluate(pred)，与训练日志字段分开。源码SHA与行号见AUTHOR_TRAIN_LOG_LOSS_SEMANTICS_AUDIT.json；未新增训练或改动参数。

<!-- ENTIRE_INTERRUPTED_OBSERVED_PREFIX_PARITY_20261118 -->
中断恢复后，已覆盖旧Five截至epoch143的432条、旧Ten截至epoch73的220条已完成训练/验证损失与指标日志，逐条字符串及选定记录SHA完全一致，见RESTART_FULL_INTERRUPTED_PREFIX_AUDIT_20261118.json。该证据仅为可观察日志前缀一致性，不等于逐epoch参数张量一致性，更不替代规定的完整未来标签冷拟合或原生订单回放。没有新增模型或修改学习参数；当前完整目标模型仍45个，完整OOF/回测/证据仍未完成。

<!-- TARGET_2025_SIX_FULL_MODELS_COMPLETE_20261118 -->
北京时间2026-11-18：seed17/2025 Five/Ten各自init/T0/T1六次完整默认训练完成，累计2018–2025四十八次。详见TARGET_PROGRESS_2025_COMPLETE.csv及各模型checkpoint/forecast/FIT_AUDIT。每个模型保留完整6584条Outer预测，无有效输入预测缺失；各Head的T0/T1共享骨干且冻结权重未改变。关键8源码锁再次SHA全部匹配。2026年两路init已继续运行。中断的两路旧partial单独保留，不计完整模型；原45个完整缓存全部验证复用。完整年度/seed OOF、真实冷拟合与原生账户仍未完成，不据此宣称A股绩效或baseline改进。


续跑核验（2026-11-18T01:04:29.110694+00:00）：seed17 已完成51个完整默认停止模型；新增2026 Five init为47轮/893步，Five T0为35轮/350步，二者各3667条预测且有效输入无缺失；T0骨干冻结哈希一致。8项学习代码锁均一致。Five T1、Ten T0仍在运行；完整真实冷启动反事实、原生账户与A股收益证据尚未完成。详见 D4/CONTINUATION_MODEL_AUDIT_20261118_0105.json。


阶段核验（2026-11-18T02:14:06.821834+00:00）：seed17 的2018–2026全部54个常规模型已完整停止；全部有效输入预测无缺失，36个T0/T1骨干保持冻结，同年同Head骨干一致。2026完整记录见 D4/TARGET_PROGRESS_2026_COMPLETE.csv；8项学习代码锁均未变化。六组完整预算冷启动反事实已启动（2018 Five/Ten init），尚未通过；原生账户回放、经济回测与其余seed尚未完成，不作TRA盈利判断。


真实冷启动阶段记录（2026-11-18T02:59:01.194591+00:00）：2018 Ten 完整init/T1重训、权重/3241行预测/默认停止一致性与逐日未知标签扰动和删除全Outer回放通过，见 D4/counterfactual_tests/2018/ten/RESULT.json。六组模型病例仅完成1组；原生账户回放尚未开始，不能将其记为全部反事实或经济证据通过。


真实冷启动阶段记录（2026-11-18T03:50:54.708803+00:00）：2018 Five/Ten 两组均完成完整init/T1重训、checkpoint张量/全3241行预测/停止规则严格一致，以及原始/未知标签扰动/未知标签删除三次全Outer回放严格一致。见 D4/counterfactual_tests/PROGRESS_TWO_CASES.json。当前2/6组模型验证通过；另四组仍在运行或待运行，原生账户回放及经济评价未完成。


真实冷启动阶段记录（2026-11-18T04:35:46.504832+00:00）：新增2022 Ten完整init/T1重训、checkpoint张量/全5600行预测/停止规则严格一致，原始/未知标签扰动/未知标签删除三次全Outer回放严格一致。模型病例3/6通过，另三组待完成；原生账户回放和经济评价尚未完成。见 D4/counterfactual_tests/PROGRESS_THREE_CASES.json。


真实冷启动阶段记录（2026-11-18T05:30:09.327538+00:00）：新增2022 Five完整init/T1重训、checkpoint张量/全5600行预测/停止规则严格一致，原始/未知标签扰动/未知标签删除三次全Outer回放严格一致。模型病例4/6通过，2026两组待完成；原生账户回放和经济评价尚未完成。见 D4/counterfactual_tests/PROGRESS_FOUR_CASES.json。
