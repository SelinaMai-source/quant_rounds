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
D4实际状态：PARTIAL_SEED_CHECKPOINT；已完成种子路径[17, 42]。完整作者参考、真实目标模型、T0/T1同骨干回测、时间反事实与全部风险见[D4主报告](D4/REPORT.md)。不把合成集成测试叫真实研究，不修改正式v5/CURRENT。


阶段核验（2026-11-19T02:02:56.204991+00:00）：seed42完整54个常规模型、2018–2026全期预测、6个新增原生经济配置及全部配对成本/风险/Bootstrap/SPA证据已完成；两seed累计108个常规模型，另有12个完整反事实模型、7个原生诊断账户，经济配置18个（含3个正式v5控制）。匹配TRA_A0 S0 CAGR/Sharpe/MaxDD=0.5472179502/1.8646649834/-0.2405632772；T0_42=-0.0390128557/-0.1892120214/-0.4734816453；T1_42=-0.0141643726/0.0108551971/-0.4504897298；两者均REJECT_CURRENT_CONFIGURATION。T1相对T0有所改善但仍显著弱于匹配基准，不构成baseline升级。真实已成交改选身份T0/T1为1235/1236，S1/S2同口径成本增量方向均未通过；正式v5旧锚点独立保留。模型、预测、原生账本和结果已自动发布ff5ff1d8dfba195d3693e7a7a08980afc1f12a08，4023个远端manifest工件逐项一致。seed73的2018 Five/Ten init已开始；D4总体仍未完成，不因17/42结果不盈利而跳过73。D5尚未实施，不作为外部阻断。详见D4/all_results.csv、economic_decisions.csv、TARGET_PROGRESS_SEED42_2026_COMPLETE.csv及D4/REPORT.md。
