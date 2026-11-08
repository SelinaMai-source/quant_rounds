# Round 23.5 研究报告

本轮只跑用户锁定的 10 篇论文，不替换书目，不把 Hansen SPA 或 White Reality Check 算进这 10 篇，也不跑 Round24。三个正式方法在看到任何外层净值之前已经写进 `PRE_REGISTERED_EXPERIMENTS.yaml`。评级写在 `literature_registry.py`，冻结号 `R235_FEASIBILITY_GRADES_V1`。看到收益之后没有改评级、没有改经济门、没有按外层夏普补选方法。

比较通道是 `LEGACY_OPEN_INFORMATION_COMPARISON`。基线 parity 只说明在原开盘信息契约下，候选键、融合分数、成交笔数和日净值与 v5 一致。它不证明 v5 可以在真实开盘成交。

## 基线

`B0_V5` 与 Round23.3 的 v5 账本对齐：Sharpe 1.8756627913361097，CAGR 0.5499812832273525，MaxDD -0.240563277153944，期末净值 12750089.355724689，成交 1107。分数最大差 1.11e-16，日净值差 0。`baseline_parity.json` 为 PASS。

经济门沿用 `R21_3_ECONOMIC_MATERIALITY_V1`：ΔCAGR≥0.01，ΔSharpe≥0.05，净值相对差≥1%，MaxDD 恶化不超过 0.02，Top1 或 Top2 必须变化，并且 2024 与 2025 以外的财富也要好于 v5。费用表没有改，记为 `SAME_FEE_SCHEDULE`。三个主方法都没有过这道门。

## 十篇论文

1. Nardari & Schüssler，*Ensembles of Portfolio Rules*，SSRN 4217088，工作论文，无期刊 DOI。评级 B，已跑。原文式(7)是 γ=3 的幂效用，式(5)要求权重非负且和为 1。公开材料里的 120 个月是候选规则估计窗，不是式(7)的唯一窗长。α 的原文网格是 0.90 到 1.00、步长 0.01；主提示第 9 节禁止这种网格，所以只预注册 α=0.99、1.00、0.90。A 股这里是单袖套，不能同时持有四套完整仓位，交易时把规则分数做凸组合后再进入同一个执行器。这是方法复现，不是原文数值复现。

2. Shi, Shu, Gu & Qiu，*Integrating Minds*，Journal of Banking & Finance 2026，192:107815，DOI 10.1016/j.jbankfin.2026.107815。评级 C。原文是随机分组 GMV，B=1000。本策略是 Top1/Top2 袖套，不是高维协方差组合。没有用 Ridge 冒充，没有跑。

3. Jensen, Kelly, Malamud & Pedersen，*Machine Learning and the Implementable Efficient Frontier*，Review of Financial Studies 2026，DOI 10.1093/rfs/hhag022。评级 B。完整 Portfolio-ML 需要他们的特征和仓位网络，本轮没有实现，也不声称复现了 RFS 数值。预注册的只是离散可实现效用选择器。

4. Bai & Pukthuanthong，*Machine Learning Classification and Portfolio Construction*，Financial Analysts Journal 2026，DOI 10.1080/0015198X.2026.2726133，arXiv 2108.02283。评级 B，已跑。原文还有大神经网络和十分位；本轮按预注册只用三分位、已有 XGB/LGBM 分数、一棵随机森林和多项 logistic。RF 的 `n_estimators=100` 是 sklearn 默认，不是原文的 1000 棵树上限。

5. Gu, Kelly & Xiu，*Empirical Asset Pricing via Machine Learning*，RFS 2020，33(5):2223–2273，DOI 10.1093/rfs/hhaa009。评级 B，只审计。验证集 R² 元策略和 Round23.3 的权重网格不是同一个算法，所以不标 ALREADY_TESTED，也不重训。Round23.3 的简单融合对照仍是 `E1_T090`，也就是 v5 本身。

6. Yang, Liu, Zhong & Walid，ICAIF 2020，DOI 10.1145/3383455.3422540。评级 D。A2C/PPO/DDPG 不在已有 OOF 里，没有训练强化学习。

7. Wang, Huang, Tu, Zhang & Xu，DeepTrader，AAAI 2021，35(1):643–650。身份已核对。结构和未公开超参保持 UNSPECIFIED，没有补写。评级 D，阻断。

8. Li & Tam，MASAAT，IJCNN 2024，DOI 10.1109/IJCNN60899.2024.10650941。评级 D。多 agent 加 attention 的组合生成器没有现成 OOF，没有跑大模型。

9. Chen, Liu & Jia，*Computational Intelligence* 2023，DOI 10.1111/coin.12617。用户简称 Liu & Jia 指向这一篇，没有另换论文。LSTM/GRU/DMLP/RF 到 MLP/GRU/LSTM 再到时变 OLS 和 DMVF，这些基学习器不是现成 OOF。评级 D，阻断。

10. Bettencourt, Tetereva & Petukhina，*Asset Allocation Forest*，SSRN 4781685。评级 C。树节点最大化 Sharpe 组合目标，超参 UNSPECIFIED，也没有登记过的宏观状态面板。没有编造宏观因子。

A=0，B=4，C=2，D=4。实际进入回测的是 B 档里的三篇预注册方法。另外 7 篇没有重训。

## P1 Nardari

每年只用该年 1 月 1 日之前的规则月度总收益估计权重。2018 年历史不足，回退 v5。对照是 v5、1/N 和历史单规则最优。没有把全历史净值拟合进组合。

基准 α=0.99：Sharpe 1.9178625690151945，CAGR 0.5539270332022608，MaxDD -0.24443715512547703，净值 13030475.150611492，成交 1114。相对 v5，ΔSharpe +0.0422，ΔCAGR +0.0039，MaxDD 恶化 0.0039，净值多 280386 元。Top1 变化 77 个信号日，Top2 变化 168 个信号日。新增成交的净收益率之和是 0.0295，被换掉成交的净收益率之和是 -0.834。权重多数年份压在 v5 上，五因子和 LGBM 只在部分年份分到权重，十因子权重基本为 0。

这组点估计没有过经济门：夏普增量不到 0.05，CAGR 增量不到 0.01。区块长度 20、5000 次的配对自助区间里，ΔSharpe 的 5% 到 95% 分位是 -0.108 到 0.197，覆盖 0。长度 10 和 40 各 1000 次，结论相同。

α=1.00 的 Sharpe 是 1.8705，低于 v5。α=0.90 的 Sharpe 是 1.5849，净值降到 586 万。1/N 的 Sharpe 是 1.8141。历史最优单规则每一年都选中 v5，所以对照账本与 v5 相同。`P1_ADAPT_V5` 与基准权重相同，约束没有再改变分数，不把它写成另一个算法。

买价再滑 10bp / 20bp 之后，Sharpe 降到 1.820 / 1.722。S3 不另造一套净值：引擎已经在开盘涨停时拒绝买单。该账本里 `open_limit_up` 拒绝 301 次，计划名单没有被重排或回补。

## P2 Bai

标签是成熟的 `label_exit_rule_net`，成熟日是 `actual_exit_date`，没有用 `next2_date`。每个外层年份的随机森林和多项 logistic 只用该年之前已经成熟的样本。2018 年回退 v5。缺特征的股票保留在池子里，分数回退 v5。RF 特征只有原来的 14 个 Five 特征，含 `d1_open_gap`。没有上大神经网络，也没有把训练内拟合送进 stacker。

默认深度：Sharpe 0.4296，CAGR 0.0561，MaxDD -0.3475，净值 478413，成交 1214。Top1 变化 856 日，Top2 变化 879 日。被换掉的 v5 成交净收益率之和约 12.98，换进来的成交净收益率之和为负。深度 4 和深度 8 更差，深度 8 的 CAGR 为负。`P2_ADAPT_V5` 用的是同一套默认分数。

自助法里 ΔSharpe 的 95% 分位仍是 -0.64，损失不是抽样噪声。买价再滑 10bp / 20bp 后 Sharpe 降到 0.278 / 0.120。开盘涨停拒绝 60 次。2019 年及之后的样本外分类分数没有形成可交易增量，失败来自排序本身，不是只来自 2024 或 2025。

## P3 Jensen

动作只有：保持 V5、只买 Top1、Top1+Top2、交换 Top2/Top3、空仓。Top1+Top2 在 `top_k=2` 下就是保持 V5，没有重复记账。选择只用该年之前的实际净收益、换手和溃疡指数。λ 只有三组，全部披露，没有在外层搜索：基准是换手惩罚 0、回撤惩罚 1；敏感性是 (0,0) 和 (1,1)。费用已经在净收益里，基准不再把换手惩罚加第二次。

基准 λ 下，2018 年保持 V5，2019 到 2026 年全部空仓。Sharpe 0.5946，CAGR 0.0382，MaxDD -0.1456，净值 413496，成交 140。这 140 笔是 2018 年的持仓，2019 年还有一点退出尾差，之后日收益为 0。平均敞口 0.016，v5 是 0.136。回撤惩罚取 1 时，溃疡指数大约 0.09，日均净收益大约 0.002，空仓的效用高于继续持有。这是预注册目标的尺度结果，不是事后改出来的。λ=(1,1) 同样从 2019 年起空仓。λ=(0,0) 在若干年改成只买 Top1，Sharpe 1.7888，净值 10210241，仍低于 v5。

匹配敞口对照把基准路径的乘数按训练期敞口缩放，Sharpe 0.4569，净值 400473。基准相对 v5 的差异来自少交易，不是来自更好的持仓。经济门不通过。

## 统计与执行

三个主方法一起做 Hansen SPA。损失是日净收益的相反数，基准是 v5，区块 20，circular bootstrap，5000 次，`arch` 7.2.0 的 `arch.bootstrap.SPA`。consistent 与 lower 的 p 值都是 0.4678，upper 是 0.7678。不能拒绝“v5 不差于这三个模型中的最好者”。这不是 White Reality Check 换名。区块 10 和 40 的自助只用了 1000 次，已在 `bootstrap_spa.csv` 写明。

多重比较只覆盖这三个预注册主方法。敏感性全部入账，但没有用外层夏普去挑赢家。没有 forward，不能标 LIVE_READY。S0 到 S3 都记在 `execution_sensitivity.csv`。S3 与 S0 净值相同，因为开盘涨停拒绝本来就在执行器里。

## 决定

`NO_PAPER_ENSEMBLE_PROMOTED`。`BEST_PAPER_METHOD=NONE`。`BEST_V5_ADAPTATION=NONE`。

P1 的点估计略高于 v5，但没有过冻结的经济门，自助区间也覆盖 0。P2 明显差于 v5。P3 的基准配置把大部分年份空仓，匹配敞口之后仍然没有增量。

`CURRENT_BASELINE_CHANGED=FALSE`。`PRODUCTION_CONFIG_CHANGED=FALSE`。`AUTO_PROMOTE=FALSE`。`FORWARD_OOS_STATUS=NOT_AVAILABLE`。`LIVE_READY=FALSE`。
