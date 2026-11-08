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
