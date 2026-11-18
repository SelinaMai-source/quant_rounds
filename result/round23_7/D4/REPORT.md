# D4 TRA — 实际进行中

当前不是完成交付：锁定旧作者版完整参考正在运行；A股完整训练、反事实冷拟合和经济回测尚未开始。不是数据权限或资源阻断，不将未实现组件标为外部阻断。

已推送冻结协议并核验公开数据2512790行。参考使用作者ALSTM init3状态→TRA10状态两阶段、256隐藏维/2层/attention/dropout0.1、500epoch上限/验证早停20、每轮100训练batch，未缩短预算。公开参考保留原实现，股票切片边界缺陷单列披露，不据此声称A股PIT有效。实际进度见full_author_reference.log和服务器TRA_reference/*/*/LIVE.json；单batch fixture不计参考完成。

A股六通道数据适配已准备58830个D日已知候选的60个历史交易观测，全部序列有效。先前4270个不完整序列全部来自将零成交报价误计为交易观测；已在任何目标训练/经济运行前记录DATA_OBSERVATION_CLARIFICATION.json，排除无交易报价，不填造VWAP、不删候选，原工件完整保留。实际21条成交VWAP异常源记录仍按缺失处理，不以其他价格替代；未落入当前候选窗口。数据版本、完整源码和各分区SHA见target_data/contract.json及server_only_inputs_manifest.json；不声称证明供应商历史修订PIT。

目标适配使用原作者MSE、LSTM/TRA、Gumbel路由、Sinkhorn OT、优化器和停止条件，只适配数据维度、成熟反馈和单股票形状兼容。原源码的多余squeeze在batch1报错；目标端去除该冗余形状操作，公开参考源码不变。原代码/适配方法的完整文本及SHA见ADAPTER_SOURCE_PROVENANCE.json。失败日志causal_queue_replay_integration.log保留。

集成测试已实际通过：原实现和适配的单批MSE/OT更新全参数精确一致；T0/T1相同冻结预训练骨干；超过21日的可变成熟延迟不会提前释放，NaT不释放，反馈使用保存的专家输出。70个单股票日期完整预测回放、50次非空的当时不可用标签扰动，预测完全一致。2026年清空标签后3667个预测节点全部保留，32行真实输入模型预测入口完全一致。这些均不冒称完整A股拟合或六个完整冷拟合诊断。

比较窗口固定2018-01-02..2026-07-24、初始现金300000；信号候选截止2026-07-01，之后为估值/退出尾段。匹配原XGB历史OOF控制的38003个旧分数精确一致，新增655个真实可预测键，21个新增日期。最初预检误计67个特征不完整的空预测占位为新增预测，已按原特征完整条件纠正且保留旧预检工件，没有经济运行使用该版本。全量比较及源文件SHA见matched_control；覆盖修复不计TRA alpha，正式v5单列且不修改。

一次缺失Head值会使整日恢复匹配v5，其他日期完全一致，已由真实分数集成测试验证；不会删行后重排。后端代码继续调用原Stock执行器，真实执行一致性和压力回测将在完整参考后运行。

CURRENT_BASELINE_CHANGED=FALSE
PRODUCTION_CONFIG_CHANGED=FALSE

完整目标控制器现已实现并接入统一入口：先等待当前完整参考，固定17→42→73，逐年两个Head共享预训练骨干后分别完整T0/T1拟合；共162个原配置目标拟合。seed17另有早/中/晚两Head共六组、12个完整预算冷拟合，并从实际保存的预测初始队列和Torch随机状态重放Outer未来/未成熟标签扰动。真实计数当前仍为0，不能把实现当运行完成。

已通过原默认停止条件的合成三阶段IO集成：init37、T021、T152个epoch，仅证明端到端工程路径，不计真实目标模型。原作者对相同冻结权重求均值产生约3.725e-9浮点漂移，目标端加入常量均值精确恒等兼容，保持T0/T1骨干一致；原失败日志和澄清均保留。验证/Outer各自冷启动队列，不携带训练内拟合误差；验证保留全部D日已知节点，只在合法成熟标签上计指标。

原始市场数据现已恢复到2010-01-04，共14522056行；58826个候选有完整60个有效观测，4个短历史按作者规则前置零padding。该次完整预热初版缓存为1405e913e6e35af0d2d37884c8a56e108ce0b91251d0e77d3b8eefc5a45fd59e（后续缓存调用闭包修复见下）；较早预热和元数据尝试归档保留，未用于真实拟合。

完整作者参考指标提取使用锁定Reports.ipynb原func/backtest函数，已通过实际AST提取与导入检查；原Top80减Bottom80、作者ret.pkl截尾、365日算术口径与Stock账户结果分开。尚无完整参考指标，需两阶段默认停止后实际运行。

公开参考CPU线程补充协议已于任何目标拟合前提交f997d2dbc0296e1715d90d89d36b8121238dcff7。原2线程尝试在init epoch4中断，非完整参考、非外部阻断；保留全部日志、原partial以及实测CPU/RSS资源，见reference_attempts/cpu2_incomplete和PUBLIC_REFERENCE_CPU2_INTERRUPTION.json。原作者LSTM合成输入仅运行性能检查：2/4/8线程分别0.535/0.320/0.251秒，输出逐值一致；不计研究模型、不用验证成绩选择线程。新公开参考使用8线程从同seed17完整重新训练，原500/20/100停止预算不变；A股目标仍2线程。新日志full_author_reference_cpu8.log，原日志不覆盖。

线程补充后的首次2线程工程复测出现一次严格memory数组相等断言失败，日志原样保留；10次默认及10次deterministic标志诊断均未复现，未更改算法或放宽数值容差，原因尚未证实。随后8线程实际公开数据加载与作者/观察器单批全参数精确一致测试通过。完整目标冷拟合反事实仍须实际通过，不用复测PASS掩盖这次失败；详见REFERENCE_PARITY_RETEST_AUDIT.json。

目标数据调用闭包在首个真实目标拟合前补齐参考路径模块、资源模块及round7路径模块，并显式绑定Python解释器版本；目标模型缓存也补充解释器版本。新实际缓存为dee8d2c4165ee7c5f84cf24b5233db87f43c6d0c736635e519abb0bf45bea6d1，首次真实prepare为miss，七个输入数据工件逐字节SHA与1405完全一致；第二次真实prepare命中缓存。没有改市场特征、标签、成熟时间、候选键或预算。旧数据manifest完整归档，见TARGET_INPUT_CLOSURE_FIX_AUDIT.json；公开参考运行源码未改动。

完整目标反事实现增加实际原Stock账户重放：一个原seed17 T1 S0账户与早/中/晚两个Head六个冷拟合标签变体，共七个诊断配置。要求全部overlay计划输入、成交、拒单、持仓、袖套、费用、现金流及NAV逐项精确一致；不是新增alpha候选，登记独立manifest并在SPA中注明与TRA_T1_17相同路径别名。当前均未运行，等待完整参考和真实seed17模型，不能将新增代码标为实际回放完成。

保存checkpoint、初始空队列和Torch RNG后的完整预测回放IO集成已用70个合成日期、每日1/2/3个节点实际通过；139个节点，1次原始预测生成加3次比较回放共4次完整前向。未来/未成熟标签扰动及当时不可用标签删除均非空且预测精确一致。权重为随机未拟合fixture，不计研究训练、公开参考或真实账户。日志synthetic_checkpoint_rng_replay_integration.log；首次3字段指比较回放，不含创建基线，审计JSON已明确总次数4。完整真实六组冷拟合和原账户测试仍为NOT_RUN。

目标CPU确定性运行标志已在真实目标模型0、原账户0时先提交冻结e2489d4843a5b241c0eff468e825d8aab5f3c189。目标拟合与回放固定2线程/2 interop、torch deterministic algorithms=True、MKLDNN deterministic=True；所有模型公式、优化器、seed和500/20/100预算不变。实际标志与补充协议逐项核验并绑定模型fingerprint和FIT_AUDIT，保存随机状态的回放也必须匹配该策略；公开参考8线程默认运行未改动。

该环境下重新实际通过因果核心集成（原作者/适配MSE及OT单步全参数精确一致、真实2026清空标签仍保留3667个节点）、完整合成默认停止IO（37/21/52epoch）及139节点checkpoint/RNG回放（变更和删除未来/未成熟标签非空且精确一致）。日志均以_deterministic_runtime命名。不是声称确定性标志已经解释或治愈先前一次未知原因的短暂断言失败；完整真实冷拟合及订单检验仍须通过，不放宽数值容差、不计这些fixture为真实模型。

<!-- ACTIVE_CODE_DATA_HASH_AUDIT -->
实际运行中复核：当前公开参考契约绑定的239个源码文件及6个输入文件均逐字节SHA匹配，见reference/RUNNING_SOURCE_DATA_AUDIT_20261115_*.json。该复核证明本次冻结文件未漂移，不代表完整参考或目标研究已完成；后续默认停止及全部真实检验仍须完成。

<!-- PUBLIC_INIT_COMPLETE_20261115 -->
公开作者参考init阶段已真实完成：49个完整epoch，best epoch=28，严格满足连续20次验证未改善的作者默认停止条件，global_step=4899；完整checkpoint、预测、49行日志、STOPPING_AUDIT及输入/模型manifest已保存。原作者最终测试指标MSE=0.1598477992、MAE=0.3234504645、IC=0.0409109791、ICIR=0.3025278028。实测本阶段wall=52839.772秒、CPU user=323564.129秒、CPU system=83893.442秒、采样峰值RSS=2419265536字节，GPU=0。这些仅是公开init结果，不是最终10-state路由参考，更不是A股成绩。完整route阶段已继续运行，REFERENCE_RUN_COMPLETE仍为false；真实A股模型及账户仍为0。

<!-- RUNNING_ROUTE_HASH_AUDIT_20261116 -->
route实际长时运行期间再次核验其契约：239个源码文件、6个输入文件以及已完成init的上游checkpoint，字节数与SHA256全部匹配，见reference/RUNNING_ROUTE_SOURCE_DATA_UPSTREAM_AUDIT_20261116_1418.json。该核验只证明冻结文件及上游工件未漂移；route默认停止、完整公开基准、真实A股模型、冷拟合及订单检验仍待完成，不据此提高完成状态。

<!-- FULL_PUBLIC_REFERENCE_COMPLETE -->
作者公开两阶段均已按原默认停止完成，原基准函数实际结果见reference/original_benchmark_results.csv。目标研究仍以STATUS分项为准。

```json
[
  {
    "phase": "init",
    "seed": 17,
    "MSE": 0.15984779916710504,
    "MAE": 0.32345046454939397,
    "IC": 0.040910979113555186,
    "ICIR": 0.30252780280239655,
    "AR": 0.08707035090625416,
    "AV": 0.13945541783802562,
    "SR": 0.6243597578072186,
    "MDD": 0.2299310703529045
  },
  {
    "phase": "route",
    "seed": 17,
    "MSE": 0.15912024499686586,
    "MAE": 0.32228303098260175,
    "IC": 0.04527631412970165,
    "ICIR": 0.34470982288736546,
    "AR": 0.10381869540071507,
    "AV": 0.13892658503974137,
    "SR": 0.7472917827139901,
    "MDD": 0.1983107318810653
  }
]
```

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
