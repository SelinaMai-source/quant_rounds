# 论文证据审计

用户于 2026-11-08 指定：下列正文就是先前 Deep Research 的 10 篇论文报告。停止搜索替代书目。Hansen SPA 与 White Reality Check 不计入这 10 篇。

评级在本轮任何正式回测收益之前冻结。等级来自算法是否明确、数据是否够用，不用预期 Sharpe。

Hansen SPA 与 White Reality Check 只作为统计工具，不计入 10 篇。

## P01_NARDARI_2024 B

- 标题：Ensembles of Portfolio Rules
- 作者：Federico Nardari; Rainer Alexander Schüssler
- 年份：2024
- 来源：SSRN 4217088 / https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4217088
- 核对：VERIFIED_SSRN_WORKING_PAPER。SSRN 4217088 与 2024-12-19 EFMA 稿核对到作者、标题与式(7)(5)。无期刊 DOI，不编造。
- 算法：式(7)：在伪样本外组合总收益上最大化贴现幂效用，γ=3；式(5) 权重非负且和为1；遗忘因子 α 在原文网格上自适应。组合对象是规则的资产权重，不是分数回归。
- 超参：γ=3（原文写明）。α 网格 {0.90:0.01:1.00} 为原文搜索，本轮按主提示不展开该网格。120个月是候选规则的估计窗引用，不是式(7)的唯一窗长。
- 运行：SELECTED
- 限制：A股单袖套不能同时持有多套完整仓位。本轮用规则分数的凸组合进入同一执行器，并标明 METHOD_REPLICATION_ON_NEW_DATA。

## P02_SHI_2026 C

- 标题：Integrating minds: An ensemble approach to portfolio optimization
- 作者：Fangquan Shi; Lianjie Shu; Xinhua Gu; Jiaping Qiu
- 年份：2026
- 来源：10.1016/j.jbankfin.2026.107815 / https://doi.org/10.1016/j.jbankfin.2026.107815
- 核对：VERIFIED。JBF DOI 与 SSRN 4884029 核对。期刊作者顺序为 Shi, Qiu, Gu, Shu；与用户简称 Shi, Shu, Gu & Qiu 是同一组作者。
- 算法：把资产随机分成 G 组，在组内做 GMV，重复 B=1000 次后对基模型权重取平均。
- 超参：B=1000。G 由 LOO 选择，单点默认值 UNSPECIFIED。
- 运行：NOT_SELECTED_WRONG_OBJECT
- 限制：本策略是少数排序头的 Top1/Top2 袖套，不是高维资产 GMV。

## P03_JENSEN_2026 B

- 标题：Machine Learning and the Implementable Efficient Frontier
- 作者：Theis Ingerslev Jensen; Bryan Kelly; Semyon Malamud; Lasse Heje Pedersen
- 年份：2026
- 来源：10.1093/rfs/hhag022 / https://doi.org/10.1093/rfs/hhag022
- 核对：VERIFIED。RFS DOI 10.1093/rfs/hhag022 与 SFI 22-63 核对。
- 算法：把交易成本写进组合目标，直接学习可执行仓位（Portfolio-ML / 广义 Gârleanu-Pedersen），而不是先预测收益再忽略成本。
- 超参：原文网络与惩罚的完整默认值在本轮锁定正文中为 UNSPECIFIED。本轮不补写一套 Portfolio-ML。
- 运行：SELECTED_DECISION_LAYER
- 限制：完整 Portfolio-ML 会引入新特征和大模型。预注册实验只做离散动作选择，不能声称复现了 RFS 数值。

## P04_BAI_2026 B

- 标题：Machine Learning Classification and Portfolio Construction: Does the Loss Function Matter?
- 作者：Yang Bai; Kuntara Pukthuanthong
- 年份：2026
- 来源：10.1080/0015198X.2026.2726133 / https://arxiv.org/abs/2108.02283
- 核对：VERIFIED。FAJ DOI、arXiv:2108.02283 与 CFA Institute 目录核对。
- 算法：GBT、随机森林、神经网络分别做分类，再用 stacking 组合；对照为回归。线性基准是多项 logistic。
- 超参：树深度网格 1/2/4/8，树数上限 1000，神经网络层数网格。本轮不搜索该网格，也不上大 NN。
- 运行：SELECTED
- 限制：不用大 NN；类别按预注册改成训练折三分位；基座优先用已有 OOF，另训一个 RF。

## P05_GKX_2020 B

- 标题：Empirical Asset Pricing via Machine Learning
- 作者：Shihao Gu; Bryan Kelly; Dacheng Xiu
- 年份：2020
- 来源：10.1093/rfs/hhaa009 / https://doi.org/10.1093/rfs/hhaa009
- 核对：VERIFIED。RFS 33(5) 2223-2273，DOI 10.1093/rfs/hhaa009。
- 算法：在统一的时间切分上训练多种机器学习模型，用验证集 R^2 选择或比较模型，而不是用测试段收益挑模型。
- 超参：各模型在验证集上选择。本轮不重开他们的模型动物园。
- 运行：NOT_SELECTED_AUDIT_ONLY
- 限制：不是本轮三个预注册候选。R23.3 用的是内层经济权重，不是验证集 R^2，因此不能标成已经测过同一算法。

## P06_YANG_2020 D

- 标题：Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy
- 作者：Hongyang Yang; Xiao-Yang Liu; Shan Zhong; Anwar Walid
- 年份：2020
- 来源：10.1145/3383455.3422540 / https://doi.org/10.1145/3383455.3422540
- 核对：VERIFIED。ACM DOI 10.1145/3383455.3422540，ICAIF 2020。
- 算法：分别训练 A2C、PPO、DDPG，在滚动验证段按 Sharpe 选择 agent，集成其交易动作。
- 超参：网络与奖励的完整默认值 UNSPECIFIED，不补写。
- 运行：BLOCKED
- 限制：会替换整个交易策略，不是现有 OOF 的组合层。

## P07_DEEPTRADER_2021 D

- 标题：DeepTrader: A Deep Reinforcement Learning Approach for Risk-Return Balanced Portfolio Management with Market Conditions Embedding
- 作者：Zhicheng Wang; Biwei Huang; Shikui Tu; Kun Zhang; Lei Xu
- 年份：2021
- 来源：AAAI-21 paper 643 / https://ojs.aaai.org/index.php/AAAI/article/view/16140
- 核对：VERIFIED_IDENTITY_HYPERPARAMETERS_UNSPECIFIED。AAAI 2021 论文身份与作者已核对。结构与超参保持 UNSPECIFIED。
- 算法：个股评估单元加市场状态风险预算；风险单元奖励为负的最大回撤。图结构与网络宽度在锁定正文中为 UNSPECIFIED。
- 超参：UNSPECIFIED
- 运行：BLOCKED
- 限制：用户禁止在未公开结构上自行补写。

## P08_MASAAT_2024 D

- 标题：Developing An Attention-Based Ensemble Learning Framework for Financial Portfolio Optimisation
- 作者：Zhenglong Li; Vincent W. L. Tam
- 年份：2024
- 来源：10.1109/IJCNN60899.2024.10650941 / https://doi.org/10.1109/IJCNN60899.2024.10650941
- 核对：VERIFIED。DOI 10.1109/IJCNN60899.2024.10650941 与作者核对。
- 算法：多个 agent 读取价格与方向变化，用注意力做截面和时间分析，再由 portfolio generator 汇总成组合。
- 超参：层宽、头数、学习率 UNSPECIFIED
- 运行：BLOCKED
- 限制：输出是新的组合生成器，不是 v5 分数组合。

## P09_LIU_JIA_2023 D

- 标题：A hybrid approach for portfolio construction: Combining two-stage ensemble forecasting with portfolio optimization
- 作者：Wei Chen; Zinuo Liu; Lifen Jia
- 年份：2023
- 来源：10.1111/coin.12617 / https://doi.org/10.1111/coin.12617
- 核对：VERIFIED。DOI 10.1111/coin.12617。完整作者是 Chen, Liu, Jia；用户简称 Liu & Jia 指向同一篇，不另找论文。
- 算法：第一层 LSTM/GRU/DMLP/RF，第二层 MLP/GRU/LSTM 融合，再时变 OLS，最后 DMVF 组合优化。
- 超参：网络宽度与时变窗 UNSPECIFIED
- 运行：BLOCKED
- 限制：基学习器不是现有 XGB/LGBM OOF。重训 LSTM/GRU 会变成新的模型搜索。

## P10_AAF_2024 C

- 标题：Advancing Markowitz: Asset Allocation Forest
- 作者：Luis Ormonde Bettencourt; Anastasija Tetereva; Alla Petukhina
- 年份：2024
- 来源：SSRN 4781685 / https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4781685
- 核对：VERIFIED_SSRN_WORKING_PAPER。SSRN 4781685 核对作者、标题与 Sharpe 节点目标。无期刊 DOI。
- 算法：随机森林按宏观状态递归划分，节点目标是 Sharpe 组合目标，输出状态依赖的组合权重。
- 超参：树数、深度、叶节点最小样本 UNSPECIFIED
- 运行：NOT_SELECTED_MISSING_STATE
- 限制：本项目没有登记他们的多资产宏观状态面板。不编造宏观因子。

与 Round23.3 / 23.4 的关系：固定 90/10、LGBM 权重网格和回撤倍率都不是上述任何一篇的原算法，所以不标 ALREADY_TESTED，也不把旧结果重训后称为新论文方法。R23.3 的决策仍是 KEEP_BASELINE_V5，最优记录 E1_T090，见 `result/round23_3/decision.json`。
