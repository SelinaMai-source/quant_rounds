# 双研究线执行记录：PARTIAL

本次未按完整规范完成。冻结协议已先提交；正式 baseline_v5 未修改，未运行二进三。

round23_6 已执行基线一致性验证、P1 分数融合、Super Learner/BOA 历史训练与控制组、27 次当前经济回测及统计分析；另保留 3 次时间戳修复前 BOA 回测和 2 次基线跟踪诊断。真实 P1 共享账户组合、TRA/DoubleAdapt 完整 L1/L2 与完整集成验收未完成。不能把分数融合称为 P1 组合复现。

round26 独立执行 1 次基线回测，进行了原源码核心或组件的有限合成输入验证；完整 L1 为 0，新增 A 股因子包、Add/Remove/Replace 和新增候选回测均为 0。AlphaForge 仅组件更新。其余可执行工作未完成，不归因于权限阻断。

真实阻断分别包括业绩预告表认证失败、缺少已授权配置的 RD-Agent LLM 端点、A 股历史样本仅五档而非 DeepLOB 十档。下载失败记录与未执行状态均保留。权限清单仅包含实际认证失败的数据表。

round23_6 已核验远端提交：`efcb7da982566e6aaf1e0be8a1ce8353835569ac`。
round26 最终提交通过 Git 当前分支 HEAD 及外部发布后核验 JSON 解析，避免在提交树内自引用自身 SHA。

主报告：result/round23_6/ROUND23_6_ENSEMBLE_REPORT.md；result/round26/ROUND26_ALPHA_REPORT.md。
逐论文状态：execution_status_by_paper.csv；实验计数：actual_execution_counts.json；未完成与偏差：protocol_deviations_and_unfinished.json；集成状态：integration_acceptance.json。

COMPLETE=FALSE
LIVE_READY=FALSE
AUTO_PROMOTE=FALSE
ROUND24_EXECUTED=FALSE
