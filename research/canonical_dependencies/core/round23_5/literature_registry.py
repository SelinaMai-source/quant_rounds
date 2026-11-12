"""锁定的 10 篇论文登记。不得增删论文，不得把 Hansen SPA / White RC 算进 10 篇。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

SOURCE_NOTE = (
    "用户于 2026-11-08 指定：下列正文就是先前 Deep Research 的 10 篇论文报告。"
    "停止搜索替代书目。Hansen SPA 与 White Reality Check 不计入这 10 篇。"
)

USER_REPORT = """\
# Deep Research 模型组合文献报告（用户锁定正文）

来源：用户于 2026-11-08 明确指定，本文件就是先前 Deep Research 的 10 篇论文报告。
不得替换书目，不得把一篇拆成多篇，不得把 Hansen SPA 或 White Reality Check 计入 10 篇。

1. Nardari & Schüssler, Ensembles of Portfolio Rules, 2024 SSRN/working paper. 规则层 utility ensemble，非负权重和为1，近期表现可更高权重；公开资料提到120个月滚动窗。未明确超参记 UNSPECIFIED。
2. Shi, Shu, Gu & Qiu, Integrating Minds, JBF 2026 Vol.192 Art.107815；SSRN 2024. 随机分组 GMV + B=1000 aggregation。
3. Jensen, Kelly, Malamud & Pedersen, Machine Learning and the Implementable Efficient Frontier, RFS 2026；SFI 22-63/AQR. 交易成本写进经济目标，直接学习可实现仓位。
4. Bai & Pukthuanthong, Machine Learning Classification and Portfolio Construction, FAJ 2026. 分类 vs 回归 + GBT/RF/NN stacking；连续12个月 validation 再12个月 OOS。
5. Gu, Kelly & Xiu, Empirical Asset Pricing via Machine Learning, RFS 2020, 33(5) 2223–2273. 统一时间切分与 validation R^2 meta-strategy。
6. Yang, Liu, Zhong & Walid, Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy, ICAIF 2020. A2C/PPO/DDPG，rolling validation 按 Sharpe 选 agent。
7. Wang et al., DeepTrader, AAAI 2021, 35(1) 643–650. 个股评估 + 市场状态风险预算（负最大回撤 reward）。未公开项目 UNSPECIFIED，禁止自行补写。
8. Li & Tam, MASAAT, IJCNN 2024, DOI 10.1109/IJCNN60899.2024.10650941. 多 agent + attention portfolio generator。
9. Liu & Jia, two-stage ensemble + DMVF, Computational Intelligence 2023. LSTM/GRU/DMLP/RF → MLP/GRU/LSTM 融合 → 时变 OLS → DMVF。
10. Bettencourt, Tetereva & Petukhina, Asset Allocation Forest, SSRN 2024–2026. 树节点目标为 Sharpe portfolio objective。

文献预注册的三个复现候选（登记于看收益之前，不是看完收益后再选）：
- P1 Nardari：Portfolio Rules Ensemble。仅用历史 OOF 完整策略日净收益，非负且和为1的规则权重。对照 V5、1/N、历史最优单规则。不要把全历史 NAV 拟合进 ensemble。
- P2 Bai：Classification Stacking。每折只用训练成熟净收益做 3 档（训练折 33%/67% 分位）；基座用已有 XGB-Five、XGB-Ten、LGBM，加一个 RFClassifier；Meta 只读历史 OOF，多项 Logistic。禁止训练内拟合进 Stacker。不要上大 NN。
- P3 Jensen：离散 Implementable-Utility Selector。动作只允许：保持V5、Top1-only、Top1+Top2、交换Top2/3、abstain。目标必须来自同一执行器的实际净收益/换手/回撤惩罚，不能用原始 label 替代。lambda 只做三组预注册值并全部披露，禁止 Outer 搜索。若提升完全来自降低敞口，必须做 matched-exposure 对照。
"""

# 评级在任何正式回测收益之前冻结。禁止事后改等级。
FROZEN_BEFORE_PNL = True
GRADE_FREEZE_ID = "R235_FEASIBILITY_GRADES_V1"

PAPERS: list[dict] = [
    {
        "paper_id": "P01_NARDARI_2024",
        "short": "Nardari & Schüssler",
        "title": "Ensembles of Portfolio Rules",
        "authors": "Federico Nardari; Rainer Alexander Schüssler",
        "year": "2024",
        "doi_or_arxiv": "SSRN 4217088",
        "original_url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4217088",
        "institution": "University of Melbourne; University of Rostock",
        "method_name": "utility ensemble of portfolio rules",
        "original_algorithm": "式(7)：在伪样本外组合总收益上最大化贴现幂效用，γ=3；式(5) 权重非负且和为1；遗忘因子 α 在原文网格上自适应。组合对象是规则的资产权重，不是分数回归。",
        "original_objective": "power utility of the convex combination of pseudo-OOS portfolio gross returns",
        "training_protocol": "每个时点只用 τ 到 t-1 的伪样本外收益重估权重；α 由历史自适应选择",
        "required_inputs": "各规则的资产权重与伪样本外收益",
        "original_hyperparameters": "γ=3（原文写明）。α 网格 {0.90:0.01:1.00} 为原文搜索，本轮按主提示不展开该网格。120个月是候选规则的估计窗引用，不是式(7)的唯一窗长。",
        "source_code_url": "UNSPECIFIED",
        "replication_limitations": "A股单袖套不能同时持有多套完整仓位。本轮用规则分数的凸组合进入同一执行器，并标明 METHOD_REPLICATION_ON_NEW_DATA。",
        "verification_status": "VERIFIED_SSRN_WORKING_PAPER",
        "verification_note": "SSRN 4217088 与 2024-12-19 EFMA 稿核对到作者、标题与式(7)(5)。无期刊 DOI，不编造。",
        "grade": "B",
        "run_status": "SELECTED",
        "already_tested": "",
        "questions": {
            "q01_problem": "把异质组合规则按联合效用组合，而不是只选单一规则。",
            "q02_family": "预测组合之外的规则层效用组合，带时变权重。",
            "q03_inputs": "规则的伪样本外组合收益。",
            "q04_objective": "贴现幂效用，γ=3。",
            "q05_base_learners": "需要多条完整组合规则，不是单个打分头。",
            "q06_oof": "是，必须用规则形成之后的伪样本外收益。",
            "q07_state": "原文用遗忘因子，不另加市场状态变量。",
            "q08_portfolio_returns": "是，需要完整规则的收益序列。",
            "q09_negative_weights": "不允许。",
            "q10_sum_to_one": "是。",
            "q11_online": "每个决策点用截至当时的历史重估。",
            "q12_future_labels": "权重只用过去收益。",
            "q13_missing_data": "不需要原文的美股库；需要本策略规则的历史净值。",
            "q14_universe": "适配层保持 v5 股票池。",
            "q15_prior_rounds": "不同于 R23.3 的固定权重网格，也不同于 R23.4 的回撤倍率。",
            "q16_faithful": "效用与单纯形约束可实现；资产权重混合改成同一袖套的分数凸组合，必须披露。",
            "q17_v5_adapt": "可以，只替换组合层。",
            "q18_point_in_time": "可以，按年只用该年之前的规则收益。",
            "q19_cost": "四条冻结规则各一次全历史回测，再加少量组合回测。",
            "q20_failure": "规则收益不足 12 个月、优化失败、或把全历史净值拟合进权重。",
        },
    },
    {
        "paper_id": "P02_SHI_2026",
        "short": "Shi, Shu, Gu & Qiu",
        "title": "Integrating minds: An ensemble approach to portfolio optimization",
        "authors": "Fangquan Shi; Lianjie Shu; Xinhua Gu; Jiaping Qiu",
        "year": "2026",
        "doi_or_arxiv": "10.1016/j.jbankfin.2026.107815",
        "original_url": "https://doi.org/10.1016/j.jbankfin.2026.107815",
        "institution": "Journal of Banking & Finance, Vol.192, Article 107815; SSRN 4884029 (2024-07-03)",
        "method_name": "random-grouping GMV ensemble",
        "original_algorithm": "把资产随机分成 G 组，在组内做 GMV，重复 B=1000 次后对基模型权重取平均。",
        "original_objective": "降低高维协方差估计误差下的全局最小方差组合风险",
        "training_protocol": "G 由权重的留一交叉验证方差选择；B=1000",
        "required_inputs": "大规模资产收益率协方差",
        "original_hyperparameters": "B=1000。G 由 LOO 选择，单点默认值 UNSPECIFIED。",
        "source_code_url": "UNSPECIFIED",
        "replication_limitations": "本策略是少数排序头的 Top1/Top2 袖套，不是高维资产 GMV。",
        "verification_status": "VERIFIED",
        "verification_note": "JBF DOI 与 SSRN 4884029 核对。期刊作者顺序为 Shi, Qiu, Gu, Shu；与用户简称 Shi, Shu, Gu & Qiu 是同一组作者。",
        "grade": "C",
        "run_status": "NOT_SELECTED_WRONG_OBJECT",
        "already_tested": "",
        "questions": {
            "q01_problem": "高维协方差误差下的 GMV 权重估计。",
            "q02_family": "风险组合，不是排序模型融合。",
            "q03_inputs": "资产收益率，用来估计协方差。",
            "q04_objective": "组内全局最小方差，再对基模型平均。",
            "q05_base_learners": "需要大量随机分组基模型，B=1000。",
            "q06_oof": "不是本项目的预测 OOF 融合。",
            "q07_state": "不需要市场状态标签。",
            "q08_portfolio_returns": "需要资产收益面板。",
            "q09_negative_weights": "GMV 原文未在本报告中强制单纯形；不臆造约束。",
            "q10_sum_to_one": "组合权重通常和为1，但分组细节未在公开摘要里给全。",
            "q11_online": "滚动估计协方差，具体再平衡 UNSPECIFIED。",
            "q12_future_labels": "协方差只用历史收益才点时正确。",
            "q13_missing_data": "缺少原文那种大规模资产协方差实验。",
            "q14_universe": "若改成资产 GMV 会改变本策略股票池与下单。",
            "q15_prior_rounds": "R23.3/R23.4 没有做过 B=1000 随机分组 GMV。",
            "q16_faithful": "不能在四条规则上假装这就是该文算法。",
            "q17_v5_adapt": "不能在不改变决策层的前提下做忠实适配。",
            "q18_point_in_time": "原则上可以，但对象不匹配。",
            "q19_cost": "B=1000 的资产分组对本袖套没有对应对象。",
            "q20_failure": "没有高维资产协方差面板，禁止用 Ridge 替代。",
        },
    },
    {
        "paper_id": "P03_JENSEN_2026",
        "short": "Jensen, Kelly, Malamud & Pedersen",
        "title": "Machine Learning and the Implementable Efficient Frontier",
        "authors": "Theis Ingerslev Jensen; Bryan Kelly; Semyon Malamud; Lasse Heje Pedersen",
        "year": "2026",
        "doi_or_arxiv": "10.1093/rfs/hhag022",
        "original_url": "https://doi.org/10.1093/rfs/hhag022",
        "institution": "Review of Financial Studies; Swiss Finance Institute Research Paper 22-63",
        "method_name": "implementable-utility portfolio learning",
        "original_algorithm": "把交易成本写进组合目标，直接学习可执行仓位（Portfolio-ML / 广义 Gârleanu-Pedersen），而不是先预测收益再忽略成本。",
        "original_objective": "扣除交易成本后的风险收益，即 implementable efficient frontier",
        "training_protocol": "在经济目标中训练仓位函数；原文是特征到仓位的机器学习，不是五动作枚举。",
        "required_inputs": "证券特征、交易成本、可交易仓位",
        "original_hyperparameters": "原文网络与惩罚的完整默认值在本轮锁定正文中为 UNSPECIFIED。本轮不补写一套 Portfolio-ML。",
        "source_code_url": "UNSPECIFIED",
        "replication_limitations": "完整 Portfolio-ML 会引入新特征和大模型。预注册实验只做离散动作选择，不能声称复现了 RFS 数值。",
        "verification_status": "VERIFIED",
        "verification_note": "RFS DOI 10.1093/rfs/hhag022 与 SFI 22-63 核对。",
        "grade": "B",
        "run_status": "SELECTED_DECISION_LAYER",
        "already_tested": "",
        "questions": {
            "q01_problem": "让组合目标知道交易成本，直接学习可实现仓位。",
            "q02_family": "决策优化，不是纯预测融合。",
            "q03_inputs": "原文是特征与成本；本轮预注册只用执行器净收益、换手和回撤。",
            "q04_objective": "净收益减换手与回撤惩罚。λ 只允许三组预注册值。",
            "q05_base_learners": "原文需要仓位模型。本轮动作是冻结规则，不再训练新的大模型。",
            "q06_oof": "动作收益必须来自执行器，不能用样本内标签代替。",
            "q07_state": "本轮动作集不含未来市场状态。",
            "q08_portfolio_returns": "是，必须用同一执行器的净值。",
            "q09_negative_weights": "动作不使用负权重。",
            "q10_sum_to_one": "每个时点只选一个动作，不是权重向量。",
            "q11_online": "按年用该年之前的执行器收益选择动作。",
            "q12_future_labels": "禁止用 Y 年收益选择 Y 年动作。",
            "q13_missing_data": "完整 Portfolio-ML 缺少其公开特征实现；离散动作不缺数据。",
            "q14_universe": "动作不引入新股票；Top1-only、交换和空仓会改变下单。",
            "q15_prior_rounds": "R23.4 改的是连续敞口倍率，不是这五个离散动作。",
            "q16_faithful": "完整论文算法本轮阻断。预注册的是决策层原则适配。",
            "q17_v5_adapt": "可以，作为独立决策层实验。",
            "q18_point_in_time": "可以。2018 年历史不足则回退 V5。",
            "q19_cost": "每个冻结动作一次回测，再加三条 λ 路径。",
            "q20_failure": "若增益只来自降低敞口而未做匹配敞口对照，则不能声称增量。",
        },
    },
    {
        "paper_id": "P04_BAI_2026",
        "short": "Bai & Pukthuanthong",
        "title": "Machine Learning Classification and Portfolio Construction: Does the Loss Function Matter?",
        "authors": "Yang Bai; Kuntara Pukthuanthong",
        "year": "2026",
        "doi_or_arxiv": "10.1080/0015198X.2026.2726133",
        "original_url": "https://arxiv.org/abs/2108.02283",
        "institution": "Financial Analysts Journal, 2026, Volume 82 Issue 4; arXiv:2108.02283",
        "method_name": "classification stacking",
        "original_algorithm": "GBT、随机森林、神经网络分别做分类，再用 stacking 组合；对照为回归。线性基准是多项 logistic。",
        "original_objective": "分类交叉熵，而不是回归损失。",
        "training_protocol": "每年六月扩展训练集，随后 12 个月验证调参，再 12 个月 OOS。",
        "required_inputs": "股票特征与收益类别",
        "original_hyperparameters": "树深度网格 1/2/4/8，树数上限 1000，神经网络层数网格。本轮不搜索该网格，也不上大 NN。",
        "source_code_url": "UNSPECIFIED",
        "replication_limitations": "不用大 NN；类别按预注册改成训练折三分位；基座优先用已有 OOF，另训一个 RF。",
        "verification_status": "VERIFIED",
        "verification_note": "FAJ DOI、arXiv:2108.02283 与 CFA Institute 目录核对。",
        "grade": "B",
        "run_status": "SELECTED",
        "already_tested": "",
        "questions": {
            "q01_problem": "分类损失是否比回归更能分开收益档。",
            "q02_family": "预测组合，stacking。",
            "q03_inputs": "特征与收益类别；meta 读基模型的样本外输出。",
            "q04_objective": "多项分类。",
            "q05_base_learners": "是。原文是 GBT/RF/NN。本轮用已有 XGB/LGBM OOF 加一个 RF。",
            "q06_oof": "是。禁止把训练集内拟合预测送进 stacker。",
            "q07_state": "不需要额外市场状态。",
            "q08_portfolio_returns": "标签用成熟净收益，不用 next2_date 代理。",
            "q09_negative_weights": "logistic 系数可正可负，但那是 meta 系数，不是仓位权重。",
            "q10_sum_to_one": "类别概率和为1。",
            "q11_online": "按年扩展，模型在预测年之前拟合。",
            "q12_future_labels": "训练只使用 actual_exit_date 已成熟的标签。",
            "q13_missing_data": "已有特征和 OOF。不新增因子。",
            "q14_universe": "不改变 v5 股票池。",
            "q15_prior_rounds": "R23.3 是分数加权，不是分类 stacking。",
            "q16_faithful": "核心 stacking 与多项 logistic 可实现；省略 NN 和十分位必须披露。",
            "q17_v5_adapt": "可以。",
            "q18_point_in_time": "可以。2018 元历史不足则回退。",
            "q19_cost": "一个 RF 的按年 OOF，数据量约 6 万行。",
            "q20_failure": "标签成熟日缺失、或 meta 用了样本内 RF 预测。",
        },
    },
    {
        "paper_id": "P05_GKX_2020",
        "short": "Gu, Kelly & Xiu",
        "title": "Empirical Asset Pricing via Machine Learning",
        "authors": "Shihao Gu; Bryan Kelly; Dacheng Xiu",
        "year": "2020",
        "doi_or_arxiv": "10.1093/rfs/hhaa009",
        "original_url": "https://doi.org/10.1093/rfs/hhaa009",
        "institution": "Review of Financial Studies, 2020, 33(5), 2223-2273",
        "method_name": "validation R^2 model selection",
        "original_algorithm": "在统一的时间切分上训练多种机器学习模型，用验证集 R^2 选择或比较模型，而不是用测试段收益挑模型。",
        "original_objective": "样本外预测 R^2 与由此形成的组合",
        "training_protocol": "训练集、验证集、测试集按时间顺序滚动",
        "required_inputs": "大规模特征面板与收益",
        "original_hyperparameters": "各模型在验证集上选择。本轮不重开他们的模型动物园。",
        "source_code_url": "UNSPECIFIED",
        "replication_limitations": "不是本轮三个预注册候选。R23.3 用的是内层经济权重，不是验证集 R^2，因此不能标成已经测过同一算法。",
        "verification_status": "VERIFIED",
        "verification_note": "RFS 33(5) 2223-2273，DOI 10.1093/rfs/hhaa009。",
        "grade": "B",
        "run_status": "NOT_SELECTED_AUDIT_ONLY",
        "already_tested": "NOT_IDENTICAL_TO_R23_3",
        "questions": {
            "q01_problem": "用机器学习测量风险溢价，并用验证集选择模型。",
            "q02_family": "预测组合 / 模型选择。",
            "q03_inputs": "特征与收益。",
            "q04_objective": "验证集 R^2。",
            "q05_base_learners": "原文比较多种模型。",
            "q06_oof": "验证段必须晚于训练。",
            "q07_state": "不把未来状态写进选择。",
            "q08_portfolio_returns": "经济评估用组合，选择标准是 R^2。",
            "q09_negative_weights": "不是单纯形融合。",
            "q10_sum_to_one": "不适用。",
            "q11_online": "滚动再训练。",
            "q12_future_labels": "测试年不参与选择。",
            "q13_missing_data": "可以做小规模版本，但本轮不跑。",
            "q14_universe": "若重训新模型会扩大搜索。",
            "q15_prior_rounds": "R23.3 权重网格不是 validation R^2。见 result/round23_3/decision.json，最优仍是 E1_T090。",
            "q16_faithful": "算法清楚，但不是预注册的三个框架。",
            "q17_v5_adapt": "可以，但本轮不运行。",
            "q18_point_in_time": "可以。",
            "q19_cost": "重开模型比较会变成新的搜索。",
            "q20_failure": "若用 2018–2026 收益挑选模型，就不再是该文的验证协议。",
        },
    },
    {
        "paper_id": "P06_YANG_2020",
        "short": "Yang, Liu, Zhong & Walid",
        "title": "Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy",
        "authors": "Hongyang Yang; Xiao-Yang Liu; Shan Zhong; Anwar Walid",
        "year": "2020",
        "doi_or_arxiv": "10.1145/3383455.3422540",
        "original_url": "https://doi.org/10.1145/3383455.3422540",
        "institution": "ICAIF 2020",
        "method_name": "A2C/PPO/DDPG ensemble",
        "original_algorithm": "分别训练 A2C、PPO、DDPG，在滚动验证段按 Sharpe 选择 agent，集成其交易动作。",
        "original_objective": "最大化投资收益，验证段用 Sharpe 选 agent",
        "training_protocol": "深度强化学习，滚动验证",
        "required_inputs": "价格与账户状态，连续或离散交易动作",
        "original_hyperparameters": "网络与奖励的完整默认值 UNSPECIFIED，不补写。",
        "source_code_url": "UNSPECIFIED",
        "replication_limitations": "会替换整个交易策略，不是现有 OOF 的组合层。",
        "verification_status": "VERIFIED",
        "verification_note": "ACM DOI 10.1145/3383455.3422540，ICAIF 2020。",
        "grade": "D",
        "run_status": "BLOCKED",
        "already_tested": "",
        "questions": {
            "q01_problem": "用三个 actor-critic 算法学习交易策略再集成。",
            "q02_family": "决策优化 / 动态专家选择。",
            "q03_inputs": "行情与持仓状态。",
            "q04_objective": "投资收益，验证段 Sharpe。",
            "q05_base_learners": "需要三个强化学习 agent。",
            "q06_oof": "不是现成 OOF 分数。",
            "q07_state": "账户状态是必须的。",
            "q08_portfolio_returns": "策略自己产生收益。",
            "q09_negative_weights": "动作空间由算法决定。",
            "q10_sum_to_one": "不适用。",
            "q11_online": "滚动验证选 agent。",
            "q12_future_labels": "用 Sharpe 选 agent 时不能看测试段。",
            "q13_missing_data": "可以训练，但那是新策略而不是组合层。",
            "q14_universe": "会改变下单。",
            "q15_prior_rounds": "没有训练过这三个 agent。",
            "q16_faithful": "在本契约下不能忠实复现。",
            "q17_v5_adapt": "不能只替换组合层。",
            "q18_point_in_time": "训练本身可以按时间切，但计算和决策层都不匹配。",
            "q19_cost": "三个深度强化学习智能体，超出本轮预注册。",
            "q20_failure": "用验证段 Sharpe 在全历史里挑 agent 会污染测试。",
        },
    },
    {
        "paper_id": "P07_DEEPTRADER_2021",
        "short": "Wang et al.",
        "title": "DeepTrader: A Deep Reinforcement Learning Approach for Risk-Return Balanced Portfolio Management with Market Conditions Embedding",
        "authors": "Zhicheng Wang; Biwei Huang; Shikui Tu; Kun Zhang; Lei Xu",
        "year": "2021",
        "doi_or_arxiv": "AAAI-21 paper 643",
        "original_url": "https://ojs.aaai.org/index.php/AAAI/article/view/16140",
        "institution": "AAAI 2021, 35(1), 643-650",
        "method_name": "DeepTrader",
        "original_algorithm": "个股评估单元加市场状态风险预算；风险单元奖励为负的最大回撤。图结构与网络宽度在锁定正文中为 UNSPECIFIED。",
        "original_objective": "收益与负最大回撤的风险收益平衡",
        "training_protocol": "深度强化学习。未公开项目细节禁止补写。",
        "required_inputs": "个股序列、市场状态、资产关系图",
        "original_hyperparameters": "UNSPECIFIED",
        "source_code_url": "UNSPECIFIED",
        "replication_limitations": "用户禁止在未公开结构上自行补写。",
        "verification_status": "VERIFIED_IDENTITY_HYPERPARAMETERS_UNSPECIFIED",
        "verification_note": "AAAI 2021 论文身份与作者已核对。结构与超参保持 UNSPECIFIED。",
        "grade": "D",
        "run_status": "BLOCKED",
        "already_tested": "",
        "questions": {
            "q01_problem": "在市场状态下平衡收益与回撤。",
            "q02_family": "决策优化。",
            "q03_inputs": "个股序列与市场状态。",
            "q04_objective": "价格涨幅奖励与负最大回撤奖励。",
            "q05_base_learners": "两个神经网络单元，不是现有 OOF。",
            "q06_oof": "不适用。",
            "q07_state": "是，市场状态是方法的一部分。",
            "q08_portfolio_returns": "策略自行生成。",
            "q09_negative_weights": "原文含多空资金比例，细节 UNSPECIFIED。",
            "q10_sum_to_one": "组合权重由生成器给出。",
            "q11_online": "强化学习策略。",
            "q12_future_labels": "不能用未来状态训练当前权重。",
            "q13_missing_data": "未公开结构，禁止补写。",
            "q14_universe": "会改变仓位。",
            "q15_prior_rounds": "没有实现 DeepTrader。",
            "q16_faithful": "不能。",
            "q17_v5_adapt": "不能在不编造结构的情况下适配。",
            "q18_point_in_time": "结构不明，无法证明时点正确。",
            "q19_cost": "未运行。",
            "q20_failure": "任何自行补写的网络都不是该文。",
        },
    },
    {
        "paper_id": "P08_MASAAT_2024",
        "short": "Li & Tam",
        "title": "Developing An Attention-Based Ensemble Learning Framework for Financial Portfolio Optimisation",
        "authors": "Zhenglong Li; Vincent W. L. Tam",
        "year": "2024",
        "doi_or_arxiv": "10.1109/IJCNN60899.2024.10650941",
        "original_url": "https://doi.org/10.1109/IJCNN60899.2024.10650941",
        "institution": "IJCNN 2024",
        "method_name": "MASAAT",
        "original_algorithm": "多个 agent 读取价格与方向变化，用注意力做截面和时间分析，再由 portfolio generator 汇总成组合。",
        "original_objective": "平衡收益与风险的集成组合",
        "training_protocol": "多 agent 注意力模型",
        "required_inputs": "价格序列与方向变化事件",
        "original_hyperparameters": "层宽、头数、学习率 UNSPECIFIED",
        "source_code_url": "UNSPECIFIED",
        "replication_limitations": "输出是新的组合生成器，不是 v5 分数组合。",
        "verification_status": "VERIFIED",
        "verification_note": "DOI 10.1109/IJCNN60899.2024.10650941 与作者核对。",
        "grade": "D",
        "run_status": "BLOCKED",
        "already_tested": "",
        "questions": {
            "q01_problem": "用多 agent 注意力降低有偏交易信号。",
            "q02_family": "决策优化。",
            "q03_inputs": "价格与方向变化。",
            "q04_objective": "集成组合的收益风险。",
            "q05_base_learners": "多个自建 agent。",
            "q06_oof": "不是现成 OOF。",
            "q07_state": "方向变化是额外数据。",
            "q08_portfolio_returns": "生成器直接输出组合。",
            "q09_negative_weights": "UNSPECIFIED。",
            "q10_sum_to_one": "组合生成器通常输出权重。",
            "q11_online": "模型训练协议细节 UNSPECIFIED。",
            "q12_future_labels": "不能用未来价格训练当前权重。",
            "q13_missing_data": "缺少可复核的公开实现细节。",
            "q14_universe": "会改变仓位。",
            "q15_prior_rounds": "没有实现 MASAAT。",
            "q16_faithful": "不能在不编造结构时复现。",
            "q17_v5_adapt": "不能只换组合层。",
            "q18_point_in_time": "结构不明。",
            "q19_cost": "未运行。",
            "q20_failure": "把注意力网络写成通用 Ridge。",
        },
    },
    {
        "paper_id": "P09_LIU_JIA_2023",
        "short": "Liu & Jia",
        "title": "A hybrid approach for portfolio construction: Combining two-stage ensemble forecasting with portfolio optimization",
        "authors": "Wei Chen; Zinuo Liu; Lifen Jia",
        "year": "2023",
        "doi_or_arxiv": "10.1111/coin.12617",
        "original_url": "https://doi.org/10.1111/coin.12617",
        "institution": "Computational Intelligence, published 2023-12-15",
        "method_name": "two-stage ensemble plus DMVF",
        "original_algorithm": "第一层 LSTM/GRU/DMLP/RF，第二层 MLP/GRU/LSTM 融合，再时变 OLS，最后 DMVF 组合优化。",
        "original_objective": "价格预测误差加带预测误差项和二范数成本的均值方差",
        "training_protocol": "两阶段集成后再做组合优化",
        "required_inputs": "价格序列与上述基学习器",
        "original_hyperparameters": "网络宽度与时变窗 UNSPECIFIED",
        "source_code_url": "UNSPECIFIED",
        "replication_limitations": "基学习器不是现有 XGB/LGBM OOF。重训 LSTM/GRU 会变成新的模型搜索。",
        "verification_status": "VERIFIED",
        "verification_note": "DOI 10.1111/coin.12617。完整作者是 Chen, Liu, Jia；用户简称 Liu & Jia 指向同一篇，不另找论文。",
        "grade": "D",
        "run_status": "BLOCKED",
        "already_tested": "",
        "questions": {
            "q01_problem": "用两阶段集成预测股价，再做分散均值方差。",
            "q02_family": "预测组合加风险组合。",
            "q03_inputs": "价格序列。",
            "q04_objective": "集成预测后的 DMVF。",
            "q05_base_learners": "需要 LSTM/GRU/DMLP/RF 和新的融合网络。",
            "q06_oof": "必须样本外预测，但现成 OOF 不是这些模型。",
            "q07_state": "时变 OLS 用历史误差。",
            "q08_portfolio_returns": "优化阶段需要。",
            "q09_negative_weights": "DMVF 约束细节不在锁定正文中，不补写。",
            "q10_sum_to_one": "组合优化通常要求和为1。",
            "q11_online": "时变权重。",
            "q12_future_labels": "不能用未来价格训练。",
            "q13_missing_data": "缺少这些序列模型的已训练 OOF。",
            "q14_universe": "会改变选股与权重。",
            "q15_prior_rounds": "没有训练过这条栈。",
            "q16_faithful": "不能用现有排序头冒充 LSTM 栈。",
            "q17_v5_adapt": "不能在不训练新模型时做忠实适配。",
            "q18_point_in_time": "若重训必须嵌套，但本轮不重训。",
            "q19_cost": "多个序列模型，超出预注册。",
            "q20_failure": "把现有 XGB 叫作该文的第一阶段。",
        },
    },
    {
        "paper_id": "P10_AAF_2024",
        "short": "Bettencourt, Tetereva & Petukhina",
        "title": "Advancing Markowitz: Asset Allocation Forest",
        "authors": "Luis Ormonde Bettencourt; Anastasija Tetereva; Alla Petukhina",
        "year": "2024",
        "doi_or_arxiv": "SSRN 4781685",
        "original_url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4781685",
        "institution": "SSRN working paper, 2024-04-02, revised 2026-03-02",
        "method_name": "Asset Allocation Forest",
        "original_algorithm": "随机森林按宏观状态递归划分，节点目标是 Sharpe 组合目标，输出状态依赖的组合权重。",
        "original_objective": "节点内 Sharpe portfolio objective",
        "training_protocol": "在宏观预测变量上生长集成树",
        "required_inputs": "多资产收益与宏观状态变量",
        "original_hyperparameters": "树数、深度、叶节点最小样本 UNSPECIFIED",
        "source_code_url": "UNSPECIFIED",
        "replication_limitations": "本项目没有登记他们的多资产宏观状态面板。不编造宏观因子。",
        "verification_status": "VERIFIED_SSRN_WORKING_PAPER",
        "verification_note": "SSRN 4781685 核对作者、标题与 Sharpe 节点目标。无期刊 DOI。",
        "grade": "C",
        "run_status": "NOT_SELECTED_MISSING_STATE",
        "already_tested": "",
        "questions": {
            "q01_problem": "让组合权重随可观察宏观状态变化。",
            "q02_family": "风险组合 / 状态依赖权重。",
            "q03_inputs": "宏观状态与多资产收益。",
            "q04_objective": "叶节点 Sharpe。",
            "q05_base_learners": "树的集成，但是资产配置树，不是分类器。",
            "q06_oof": "需要样本外状态，不是现成个股 OOF。",
            "q07_state": "是，宏观状态是方法核心。",
            "q08_portfolio_returns": "是。",
            "q09_negative_weights": "受组合约束，细节 UNSPECIFIED。",
            "q10_sum_to_one": "是组合权重。",
            "q11_online": "按状态预测权重。",
            "q12_future_labels": "不能用未来宏观标签。",
            "q13_missing_data": "没有登记原文的多资产宏观面板。",
            "q14_universe": "对象是多资产，不是本股票池。",
            "q15_prior_rounds": "R23.4 的回撤规则不是 Asset Allocation Forest。",
            "q16_faithful": "缺状态变量，不能忠实复现。",
            "q17_v5_adapt": "不能在不新增宏观因子时适配。",
            "q18_point_in_time": "若有状态数据可以，但本轮没有。",
            "q19_cost": "未运行。",
            "q20_failure": "用个股因子冒充宏观状态。",
        },
    },
]


def papers() -> list[dict]:
    if len(PAPERS) != 10:
        raise RuntimeError(f"paper count must be 10, got {len(PAPERS)}")
    ids = [row["paper_id"] for row in PAPERS]
    if len(set(ids)) != 10:
        raise RuntimeError("paper_id not unique")
    return PAPERS


def write_literature(root: Path) -> dict:
    root = Path(root)
    lit = root / "literature"
    lit.mkdir(parents=True, exist_ok=True)
    (lit / "model_ensemble_10_papers.md").write_text(USER_REPORT, encoding="utf-8")
    fields = [
        "paper_id", "title", "authors", "year", "doi_or_arxiv", "original_url",
        "method_name", "original_algorithm", "original_objective", "training_protocol",
        "required_inputs", "original_hyperparameters", "source_code_url",
        "replication_limitations", "verification_status", "grade", "run_status",
    ]
    with (lit / "paper_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in papers():
            writer.writerow({key: row.get(key, "") for key in fields})
    q_fields = ["paper_id", "grade", "run_status"] + [f"q{i:02d}" for i in range(1, 21)]
    with (lit / "paper_reproduction_feasibility.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=q_fields)
        writer.writeheader()
        for row in papers():
            item = {"paper_id": row["paper_id"], "grade": row["grade"], "run_status": row["run_status"]}
            for key, value in row["questions"].items():
                item[key.split("_", 1)[0]] = value
            writer.writerow(item)
    mapping_fields = ["paper_id", "experiment_id", "role", "reason"]
    mapping = [
        ("P01_NARDARI_2024", "P1_BASE_A099", "PRIMARY", "预注册规则效用组合"),
        ("P01_NARDARI_2024", "P1_SENS_A100", "SENSITIVITY", "原文无遗忘边界"),
        ("P01_NARDARI_2024", "P1_SENS_A090", "SENSITIVITY", "原文网格下界"),
        ("P01_NARDARI_2024", "P1_ADAPT_V5", "V5_ADAPTATION", "缺规则日回退 v5，不缩小股票池"),
        ("P04_BAI_2026", "P2_BASE_RF_DEFAULT", "PRIMARY", "预注册三分位 stacking"),
        ("P04_BAI_2026", "P2_SENS_DEPTH4", "SENSITIVITY", "原文深度网格中的一个点"),
        ("P04_BAI_2026", "P2_SENS_DEPTH8", "SENSITIVITY", "原文深度网格上界"),
        ("P04_BAI_2026", "P2_ADAPT_V5", "V5_ADAPTATION", "缺失预测回退 v5"),
        ("P03_JENSEN_2026", "P3_BASE_DD", "PRIMARY_DECISION_LAYER", "预注册离散可实现效用"),
        ("P03_JENSEN_2026", "P3_SENS_NET", "SENSITIVITY", "只使用执行器净收益"),
        ("P03_JENSEN_2026", "P3_SENS_BOTH", "SENSITIVITY", "净收益、换手、回撤三项都惩罚"),
        ("P03_JENSEN_2026", "P3_ADAPT_MATCHED", "MATCHED_EXPOSURE", "敞口变化时的匹配对照"),
    ]
    for row in papers():
        if row["run_status"] in {"SELECTED", "SELECTED_DECISION_LAYER"}:
            continue
        mapping.append((row["paper_id"], "", "AUDIT_ONLY", row["run_status"]))
    with (lit / "paper_to_experiment_mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=mapping_fields)
        writer.writeheader()
        for paper_id, experiment_id, role, reason in mapping:
            writer.writerow({
                "paper_id": paper_id,
                "experiment_id": experiment_id,
                "role": role,
                "reason": reason,
            })
    manifest = {
        "source_note": SOURCE_NOTE,
        "grade_freeze_id": GRADE_FREEZE_ID,
        "frozen_before_any_round23_5_pnl": FROZEN_BEFORE_PNL,
        "paper_count": 10,
        "excluded_from_ten": ["Hansen SPA", "White Reality Check"],
        "papers": [
            {
                "paper_id": row["paper_id"],
                "verification_status": row["verification_status"],
                "verification_note": row["verification_note"],
                "grade": row["grade"],
                "source_code_url": row["source_code_url"],
            }
            for row in papers()
        ],
    }
    (lit / "paper_source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 论文证据审计",
        "",
        SOURCE_NOTE,
        "",
        "评级在本轮任何正式回测收益之前冻结。等级来自算法是否明确、数据是否够用，不用预期 Sharpe。",
        "",
        "Hansen SPA 与 White Reality Check 只作为统计工具，不计入 10 篇。",
        "",
    ]
    for row in papers():
        lines.append(f"## {row['paper_id']} {row['grade']}")
        lines.append("")
        lines.append(f"- 标题：{row['title']}")
        lines.append(f"- 作者：{row['authors']}")
        lines.append(f"- 年份：{row['year']}")
        lines.append(f"- 来源：{row['doi_or_arxiv']} / {row['original_url']}")
        lines.append(f"- 核对：{row['verification_status']}。{row['verification_note']}")
        lines.append(f"- 算法：{row['original_algorithm']}")
        lines.append(f"- 超参：{row['original_hyperparameters']}")
        lines.append(f"- 运行：{row['run_status']}")
        lines.append(f"- 限制：{row['replication_limitations']}")
        lines.append("")
    lines.append("与 Round23.3 / 23.4 的关系：固定 90/10、LGBM 权重网格和回撤倍率都不是上述任何一篇的原算法，所以不标 ALREADY_TESTED，也不把旧结果重训后称为新论文方法。R23.3 的决策仍是 KEEP_BASELINE_V5，最优记录 E1_T090，见 `result/round23_3/decision.json`。")
    lines.append("")
    (lit / "PAPER_EVIDENCE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "paper_count": 10,
        "grades": {row["paper_id"]: row["grade"] for row in papers()},
        "grade_freeze_id": GRADE_FREEZE_ID,
    }
