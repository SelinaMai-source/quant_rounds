# CODEX MASTER PROMPT
# Round23.6：P1 延伸与模型组合框架复现
# Round26：独立 Alpha 发现、关系/序列模型及增量验证

你是本项目的量化研究执行工程师。下面已固定研究范围、对照、数据规则、实验步骤和交付格式。你的职责是实现和执行，不是另起研究方向。

用户已经批准本次研究代码、数据恢复、训练、回测及向指定私有仓库的新分支 push；没有批准购买新数据、开通付费服务、增加云计算资源、修改实盘策略或向外部模型发送私有数据。

本任务只研究一进二。二进三 Round24 暂停，不运行、不修改、不继续下载专门为二进三准备的数据。

最重要的五条约束：
1. 组合线与 Alpha 线分开，不让一条线本轮挑出的赢家进入另一条线。
2. 本文列出的十篇新框架全部进入执行清单，不再只挑三篇；可运行的真实运行，缺权限/缺关键实现的逐项阻断，不用近似方法冒充成功。
3. 作者原实现复现、方法在新数据上运行、v5 适配是三个不同层次，分别报告。
4. 数据不足先实际查找、恢复和合法下载；不得用全 NaN 列、空文件或硬编码 PASS 冒充完成。
5. 最终权限清单仅含“数据名称、数据路径”，严格按第18节格式输出。

================================================================
0. 项目、源版本与两条新分支
================================================================

Repository：
https://github.com/SelinaMai-source/quant_rounds.git

Canonical project：
/data1/stock-mxy/课题/一进二策略

已知软链：
/home/mxy/Stock/课题/一进二策略

先用 realpath/inode 核验，不要误用另一份 /home/mxy/课题/一进二策略。

本次核查的版本：
- baseline_v5：feda6870a26f6efe5d722fd9c3b11130ca266d85
- round23_5：2c28c2083097a28cda921c59c28b254c30cd0976
- round23_4：6fee129f424593763c4c32755524d494a05edd8e
- round23_3：0d20428fb85003a0a8f383a0e38cefaefe19e961

按仓库现有序号，组合线从23.5顺延为23.6，不创建含义重复的25.6。

新分支：
- round23_6：P1继续研究 + Super Learner / BOA / TRA / DoubleAdapt
- round26：AlphaGen / AlphaForge / RSR / MASTER / RD-Agent-Quant / DeepLOB

各用独立 worktree。round23_6 从上面的 round23_5 派生；round26 同样可从该只读锚点派生以取得完整执行依赖，但不得继承本轮23.6筛出的新模型/权重。baseline_v5分支作为因子与策略契约来源。

如果远端已有更新，记录差异，禁止静默改研究起点。如果目标分支已存在且不是本任务续跑，停止该分支发布，不覆盖。

目录：
core/round23_6/
result/round23_6/
core/round26/
result/round26/
core/research_integrity_v4/
research/literature_round23_6_26/
research/vendor_sources/
result/round23_6_round26_master/

读 AGENTS.md、.cursor/rules/、原Factor Evidence Factory标准、已有因子/实验注册表、baseline_v5审计和原执行器。代码路径可能在发布镜像子目录，按source manifest定位。

================================================================
1. 锁定基线；不把研究比较误称为实盘证明
================================================================

基线：baseline_v5 = R16_O01。
原评价：2018-01-02 .. 2026-07-24。
初始资金：300000元。
参考：
Sharpe = 1.8756627913361097
CAGR = 0.5499812832273525
MaxDD = -0.24056327715394388
TerminalNAV = 12750089.355724689
CompletedTrades = 1107

P1历史挑战者：P1_BASE_A099，来自round23_5；保留原结果不改名覆盖。
它是utility-weighted SCORE fusion，不是忠实的portfolio-weight FLEXPOOL。

冻结原Five14/Ten13成员与顺序、目标、XGB参数、3年训练/1年预测、90%Ten+10%Five、实际B03/O01、槽位、资金、费用、退出。
以真实baseline调用链为准，不因旧Prompt提到clean_risk就额外叠加一个原链路没有使用的模块。

输出BASELINE_PARITY：
候选键、分数、排名、计划订单、成交、持仓、费用、逐日NAV、交易笔数。
不能只用Sharpe接近判断通过。

严禁：
CURRENT自动更新；生产配置修改；自动实盘下单；删除旧实验。

信息lane：
- LEGACY_V5_COMPARISON：保持v5的原开盘信息契约，用于公平历史比较。包含d1_open_gap，不声称D-close或同开盘成交已获证明。
- NEW_SIGNAL_D22：Round26新生成信号只使用D日22:00 Asia/Shanghai前合法已知信息。加入原v5后的组合仍属于LEGACY比较，不自动变成严格D-close策略。
- PAPER_REFERENCE：作者公开基准及其原协议。
- TARGET_PIT_SUBPERIOD：新数据只能支持较短历史时，使用完全匹配基线，不能直接对比全历史55% CAGR。

没有验证完的数据时间链必须标PARTIALLY_VERIFIED。不能为了满足“无未来函数”把状态直接写成PASS。

================================================================
2. 固定书目：十篇新框架全部执行，另继续P1
================================================================

以下链接是本任务的已指定来源，不需要Codex自行重新选论文。下载论文/源码/配置并锁commit、版本、许可证和hash。不得只读博客。不得把下面的作者方法换成同名库。

COMBO-01 Super Learner（2007）
Paper: https://doi.org/10.2202/1544-6115.1309
Code: https://github.com/ecpolley/SuperLearner
重点：R/SuperLearner.R、R目录中的实际method.NNLS实现、官方测试与vignette。

COMBO-02 Optimal Learning with Bernstein Online Aggregation（2017）
Paper: https://arxiv.org/abs/1404.1356
Code: https://github.com/Dralliag/opera
Reference: R/BOA.R、R/mixture.R及loss实现。
API基准：model='BOA', loss.type='square'，采用原实现gradient设置；不以普通指数加权冒充BOA。

COMBO-03 Temporal Routing Adaptor / TRA（KDD2021）
Paper: https://arxiv.org/abs/2106.12950
Code: https://github.com/microsoft/qlib/tree/main/examples/benchmarks/TRA
论文复现路径优先：目录内run.sh、configs/config_alstm.yaml、src。
维护版qlib.contrib.model.pytorch_tra若被使用，必须单列版本差异。

COMBO-04 DoubleAdapt（KDD2023）
Paper: https://arxiv.org/abs/2306.09862
Code: https://github.com/SJTU-DMTai/DoubleAdapt
作者指出原结果稳定复现还可用：https://github.com/SJTU-Quant/qlib
入口：main.py、run.sh、src；只用一个作者提供的GRU配置作为本轮主配置。

ALPHA-01 AlphaGen（KDD2023）
Paper: https://arxiv.org/abs/2306.12964
Code: https://github.com/ICT-FinD-Lab/alphagen
重点：alphagen、AlphaCalculator、alpha pool、RL训练脚本。
当前仓库含后续HARLA/LLM扩展；本轮锁定KDD2023 AlphaGen，不把后续扩展当原算法。

ALPHA-02 AlphaForge（AAAI2025）
Paper: https://arxiv.org/abs/2406.18394
Code: https://github.com/DulyHao/AlphaForge
重点：gan、alphagen、combine_AFF.py、官方实验notebook。
这不是xiejinglover/alpha-forge。后者仅作为已有Factor Evidence Factory参考，不能替代学术AlphaForge算法。

ALPHA-03 Temporal Relational Ranking / RSR（TOIS2019）
Paper: https://arxiv.org/abs/1809.09441
Code: https://github.com/fulifeng/Temporal_Relational_Stock_Ranking
入口：preprocess、training/rank_lstm.py、training/relation_rank_lstm.py。
原版使用旧TensorFlow，放独立环境；不要为安装方便改成另一种网络。

ALPHA-04 MASTER（AAAI2024）
Paper: https://arxiv.org/abs/2312.15235
Code: https://github.com/SJTU-DMTai/MASTER
入口：main.py、master.py、base_model.py、qlib-update。
必须执行官方README关于验证集/infer_processor的修正；不能按未来标签删除验证/测试节点。

ALPHA-05 R&D-Agent-Quant（NeurIPS2025）
Paper: https://arxiv.org/abs/2505.15155
Code: https://github.com/microsoft/RD-Agent
量化场景：官方qlib research loop、factor/model joint workflow。
它是研究自动化框架，不是一个单独数值因子。

ALPHA-06 DeepLOB（IEEE TSP2019）
Paper: https://arxiv.org/abs/1808.03668
Code: https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
原版公开基准：FI-2010，采用仓库作者给出的数据与模型示例。
A股适配必须是真实LOB，不得用OHLCV分钟拼成十档盘口。

P1继续研究，不算新十篇之一：
Ensembles of Portfolio Rules
已审计版本PDF：https://www.oru.se/contentassets/84372fcaa4784b22b5bd7d15e645a560/schussler---paper.pdf
保存该PDF版本与hash。使用式(5)/(6)/(7)/(9)作为方法核对依据。
现有nardari.py是项目实现，不冒充作者官方代码。

生成paper_registry.csv：每一条必须有paper/code/version/license/entrypoint/reference_config/status。
不把第三方代码整个复制进私有仓库后删除许可证。
没有明确再分发许可时保存下载脚本、URL、hash和独立adapter，不擅自vendoring。

================================================================
3. 复现层次：全部登记，不能因缺数据就只写“已审计”
================================================================

每篇必须有三个独立状态：

L0_ALGORITHM：官方源码、核心算法测试、实际反向/更新步骤。
L1_REFERENCE：作者公开基准和选定原配置的训练/评价。
L2_V5_ADAPTATION：本文规定的A股一进二适配与完整回测。

有公开合规基准就实际下载、运行L1。不能仅因本地没有FI-2010就跳过DeepLOB原基准。
不存在原文完整数据/硬件条件时，准确写REFERENCE_PARTIAL或BLOCKED，不宣称复现原文全部数值。
官方小样例、smoke test不等于完整L1。

默认配置冻结规则：
- L1只使用本任务点名的一个作者配置，不跑作者全部参数网格。
- 架构、损失、优化器、学习率从该配置机械导出到resolved_author_config.json，必须在查看本轮结果前提交。
- 同名README脚本不存在时，允许调用官方notebook中相同核心模块，写薄入口；缺失核心模块则BLOCKED_UPSTREAM_CODE，不能凭空发明。
- 为兼容Python/框架版本的修补单列diff，必须有原/新实现的小样例数值对照。
- 原基准静态行业、特征、预训练文件仅用于PAPER_REFERENCE，不能带入历史A股适配。
- 所有L2随机方法固定seeds={0,1,2}。主结果报告三个独立运行的均值、范围、每seed账本；不选最佳seed。
- 不把三seed再平均作为新方法，除非该实验明确要求。
- 确定性算法不假跑三次制造样本量。

所有十篇均执行上述流程。只有客观的权限、数据语义、核心代码或计算资源阻断可以终止相应分支，且必须有真实证据。
不采用“只有前三篇有戏，其他不做”的规则。

================================================================
4. 数据恢复、权限与来源
================================================================

用户数据清单是2026-10-04盘点，不当成今天的实时状态。将原文保存为只读输入，保持hash。
当前聊天中的 /mnt/data/数据清单.md 不是服务器必然存在的路径：在服务器按文件名、data_inventory、GitHub已上传副本定位；没有原文件时从本Prompt列出的真实目录和机器清单开始核验，不伪造已读取。

优先核查：
/data1/stock-mxy/data/datasets/hermes-cn-a-2010-20260724-v7-all-features
/data1/Stock/data/acquisitions/hermes-cn-a-2010-20260724-v7-rebuild/raw
/data1/stock-mxy/data/source_snapshots
/data1/Stock/data_live/inbox/batches
/data1/Stock/data_live/inbox/server_relay
/data1/Stock/.jq_stage
/home/mxy/AmazingData/data/amazingdata_raw
/data1/stock-mxy/data/factor_cache
/data1/Stock/.zyquant/cache/factors-pit-ex-st-v2
result/data_inventory/

所需数据源与实际来源映射：
D01 日线/复权/成交笔数：v7 daily_raw、daily_post_adjusted；raw/mkt_equd。
D02 资金流与订单分档：v7 daily_money_flow；raw/mkt_equ_mf_new。
D03 涨停事件：daily_limit_events、raw/mkt_limit、raw/mkt_limit_ind。
D04 D日分钟：source_snapshots/amazingdata-cn-a-close-limit-min1-20130104-20211231-v1/minute_bars；对应20220104-20260724-v1/minute_bars。
D05 D+1至实际退出分钟：/home/mxy/AmazingData/data/amazingdata_raw/kline_min1/parts/year=YYYY/。
D06 行业历史：v7 industry_membership、/data1/Stock/.jq_stage/get_industry；必须核对历史分类版本。
D07 财报/公司事件：financial_reports、financial_facts、fundamental_metrics、share_capital、corporate_actions、special_treatment、md_sec_halt。
D08 公告与供给扩展：源库hermes.fdmt_ef_new、fdmt_ee、equ_shares_float、equ_sh_schg、equ_share_buy_back。
D09 题材/龙虎榜扩展：hermes.mkt_themed、md_sec_type、mkt_rank_list_stocks/sales/invs/spec。
D10 指数/市场上下文：csi300.parquet、聚宽指数缓存；hermes.mkt_idxd等实际schema确认的表。
D11 真实LOB/逐笔/竞价过程：清单未发现本地源。检查现有已授权供应商产品，绝不从分钟代理生成。
D12 新增历史分区：恢复2026-07-24后缺口，但隔离为quarantine，不供本轮调参。

恢复顺序：
1. 检查已有文件、schema、代码映射和日期解析，排除“有数据但Join错”。
2. 检查原始分区、缓存、历史快照、采集状态库和合法项目配置。
3. 通过现有已授权ClickHouse/Hermes、AmazingData、聚宽接口实际查询和下载。
4. 公开基准按作者许可下载。
5. 真实认证失败、表授权失败、套餐不包含时才登记权限阻断。

不能只检查几个环境变量，然后断言整个服务器没有权限。
允许使用当前用户本来可读、已配置的credential helper；不得读取root 600文件、提权、猜密码、进入其他用户目录或绕过权限。
不得向日志输出凭证。
不自动购买数据或开通新的API费用。市场数据/LLM服务额外付费默认预算为0。

只读必要股票×日期×字段，限速、分批、checksum、断点续传。授权拒绝不反复刷接口。
下载后重新计算覆盖与PIT。下载成功不等于历史可得。

生成data_recovery_jobs.jsonl，记录真实尝试、错误码、恢复行数、源URI、目标路径、覆盖前后和PIT结论。

分类：
PERMISSION_DENIED / AUTH_REQUIRED / SUBSCRIPTION_REQUIRED
DATA_NOT_FOUND
UNVERIFIED_PIT
BLOCKED_SCHEMA
BLOCKED_UPSTREAM_CODE
BLOCKED_RUNTIME_OR_LICENSE
RESOURCE_BUDGET_REACHED

只有前三种权限类别进入用户要求的“无权限数据.md”。其他问题保存在详细JSON，不能塞进该简洁MD。

================================================================
5. 共同时间安全与历史评价协议
================================================================

构建唯一candidate_manifest：signal_date/instrument_id/decision_at/eligibility_version。
股票池只能由当时信息确定。不能因未来标签为空、无法成交、未退出或亏损而删除预测行。
必须分开D_CLOSE_UNIVERSE与LEGACY_OPEN_ELIGIBILITY：Round26在D22构建的截面rank、行业聚合、图节点、残差和因子搜索，不得提前使用D+1 gap筛过的股票集合。新信号先在D时可知的首板集合生成，之后才由原legacy执行资格模块做匹配比较。
训练集和预测集分开建；预测集不可来自dropna(target)后的训练表。

时间字段：event_time、known_at、retrieved_at、revision_time分开。
派生值known_at取全部输入可得时间上界。
市场原始观测可以根据可靠的原记录和发布机制建立历史可得证据；不能因今天获取就全部判未来，也不能把供应商最新修订自动当历史原值。
资金流/财务/行业必须as-of取当时版本。不得先argMax全历史UPDATE_TIME。

原面板含标签和未来价：严格特征白名单；agent/搜索器不得拿到未来列。
新因子严禁negative shift、centered rolling、backfill未来值、全历史归一化、全历史行业成员。
所有滚动先在完整个股交易日历计算，再选首板；不能在稀疏首板行上rolling。

标签：
- 保留原Five/Ten各自目标，分别追踪actual exit和label_available_at。
- 比较历史v5可使用LEGACY标签，但必须明确原执行假设；不新造虚假的固定D+2成熟时间。
- 若现有Ten标签无法证明成熟时间，恢复真实执行记录；仍无法证明则阻断涉及该目标的Meta训练，不阻断已验证的另一个Head。
- 未成交、未成熟、停牌、右删失与真实0收益必须分开。
- 费用标签不能在组合NAV再重复扣费。

Outer：预测年Y训练窗口Y-3..Y-1，评价Y；原2018..2026历史属于反复开发样本，不叫untouched。

Round26生成/选择：
- Y-3..Y-2：生成和拟合候选；
- Y-1：验证、有限选择和方向确定；
- 冻结表达式/候选后，可按已锁定规则用完整三年重估下游参数；
- Y年：只评价一次；不得反馈给该年候选生成。

Round23.6 Meta：使用同一三年内真正历史OOF预测。
每条OOF必须有模型hash、fit_cutoff、训练日期、最大训练标签成熟时刻。
所有校准、特征筛选、归一化也必须时间交叉拟合，不能只保证Base Model OOF。

最早OOF不够：先恢复预热历史；仍不足则当年明确fallback到v5。另报方法真实生效期，所有对照相同起点/资本重跑。
不能拿模型2018空仓而v5正常交易的差异当公平模型结果。

在线更新：在任何决策前只消费当时已成熟的反馈。保存预测时刻原始预测、权重和最终成熟时刻；不能标签到来后重算过去预测。
同一天所有股票先完成预测再更新。不可依赖任意股票行排序制造日内“学习”。

================================================================
6. Round23.6-A：P1对象对齐，不用分数组合冒充仓位组合
================================================================

保留两个历史锚点：v5及P1_SCORE_A099。
四个现有专家为v5、Five-only、Ten-only、LGBM双Head。不得给它们新增因子。

先验证：在score空间v5=0.1Five+0.9Ten，记录重复自由度；在完整策略空间，Top2和资金约束使收益不一定线性重复。

固定实验：
C00_V5：原v5。
C01_P1_SCORE_A099：精确重放原P1，年频权重。
C02_PORT_EQUAL：真实资产权重/现金权重的等权组合。
C03_PORT_A099_ANNUAL：与C01同一组冻结年频权重，只改变执行对象为资产权重。
C04_PORT_STATPOOL_MONTHLY：alpha=1，月频更新。
C05_PORT_FLEXPOOL_MONTHLY：原文alpha集合{0.90,0.91,...,1.00}，按固定PDF式(7)/(9)的历史递归效用规则选择；不是看Outer收益挑alpha。
C06_PORT_A099_MONTHLY：固定0.99月频，用于分离“更新频率”和“自适应alpha”。

不增加0.995等网格。gamma=3。所有收益历史截止到上月已完成、已知的记录。
原论文式(9)的索引和初始tau0照固定PDF逐字数学转写，保存equation_to_code.md；不可用另一个自创validation收益选择器替代。
没有足够12个月专家历史则回退v5；自适应alpha缺少初始评价历史时先用alpha=1，单独记WARMUP，不算FLEXPOOL已生效。

仓位组合必须是真实新模块：
- 专家独立输出当时可得的目标权重Omega_t（含现金），而不是未来成交后才知道的最终持仓。
- q_target=Omega_t*w_t；不得再次用Top2截断该组合后还称原论文方法。
- 一个共享现金账户，禁止四份300000元当成一份300000元。
- 相同证券净额化后形成外部订单；成本按真实净额订单计算，不简单平均四条已经扣费NAV。
- 保留虚拟专家、原持仓和退出意图的归属，用于审计。
- T+1锁定、涨跌停、停牌、整手、最低费用、公司行动和剩余现金实际处理。
- 不可卖出的持仓不得因专家权重变0凭空消失。记录target/actual deviation。
- 计划股数用当时已知价格/预算；合法执行价格在成交后才用于结算。

两个lane：
PAPER_IDEAL：分数股/可即时再平衡仅作式(6)数值验证。
TRADABLE_CONSTRAINED：真实整手、资金和持仓约束，是主要经济结果。
前者不能用于替换baseline。

仓位组合允许多于2只股票，所以C02等权同广度对照是强制项；同时报告持仓数、敞口、换手、容量。禁止将分散化效果全部归因于最优化算法。
单专家权重=1的路径必须通过对应该专家的资金和交易Parity。
优化和执行使用不同成本模型时必须披露，不能声称净收益目标精确相等。

================================================================
7. Round23.6-B：Super Learner与BOA
================================================================

逐Head构建共同目标预测银行，不能把Five与Ten不同标签混成同一收益目标。
每个Head专家库固定：原XGB、对应LGBM、Ridge(alpha=10)和历史均值预测。
v5不额外作为与Five/Ten确定性重复的独立预测列。
新Ridge只用该Head原特征，属于低复杂度对照，不扩展特征。

原树模型是ranking score，不是收益率。先在历史训练块上对score进行Ridge(alpha=10)收益校准，再生成未来块预测。
Meta训练的校准值也必须cross-fit，不能用整个Meta训练集标签拟合校准器后直接拟合同一批Meta标签。

C10_SL：
- 官方SuperLearner默认method.NNLS在合成/公开例子上实际跑通。
- A股适配显式构造只用过去训练的OOF矩阵，再调用同一NNLS系数和归一化逻辑。
- 不能仅设置validRows后让训练取验证集补集：补集仍可能含未来。
- 保存归一前/后系数、所有专家预测和同目标损失。
- 两个Head各自组合，再按原90/10融合；交易规则不变。

C11_BOA：
- 官方opera BOA square loss作为数值reference，含二阶修正，不以Hedge代替。
- 同一Head所有输入校准为同单位收益。训练期q01/q99用于学习损失的有界变换，固定后应用未来；真实回测收益绝不截断。
- 年度变换变动时用该年截止前历史重新初始化/回放，不能把不同尺度的累计状态硬拼。
- 初始专家权重均匀；只用作者默认更新，不搜索learning rate。
- 反馈延迟版必须保存预测时的专家向量及组合预测。只在真实成熟时计算原预测的损失/梯度并更新状态。
- 同日成熟反馈等权按日期聚合，避免一日股票多就任意占更大权重。记录这是delayed/batched adaptation，不能声称原即时反馈定理自动适用。
- 无延迟构造样例上必须与官方R实现权重/预测数值一致。
- 没有新成熟反馈时保持权重，不用未来标签补空窗。

两方法对照：对应单专家、简单等权、原v5。
每日模型缺失按整日一致fallback策略处理；不将[0,2]分类分数与[0,1]基线rank逐股票混排。
所有配置完整回测，不因为初期IC差就私自删除。

================================================================
8. Round23.6-C：TRA与DoubleAdapt
================================================================

它们需要可训练的预测器，不是假装在冻结XGB上训练一个Ridge就叫原框架。

先完成作者公开基准L1。A股适配统一使用作者支持的Alpha360六字段历史序列，严格在完整日线上构建、训练期归一化。
这是新序列专家，不是新增原始数据；控制组使用完全相同输入与骨干。
两个Head分别预测各自成熟标签，原90/10融合。

TRA固定组：
C20_SINGLE：作者同一骨干，一个预测器、无router。
C21_TRA：同骨干+作者多专家/OT/router原配置。
C22_TRA_NO_LOSS_MEMORY：只关闭历史误差输入，其余不变。
历史误差memory只能写入已经成熟的样本；训练时OT可以用训练标签，验证/预测router不得看本条或尚未成熟标签。
不允许跨fold复用未来memory或用test loss优化router。

DoubleAdapt固定组：
C30_STATIC：同一GRU、每Outer窗口训练一次。
C31_NAIVE_INCREMENTAL：同GRU、同更新批次，无DoubleAdapt数据/模型adapters。
C32_DOUBLEADAPT：作者数据与模型双适配器。
固定每5个交易日安排一次更新；实际只用已成熟数据。标签不足时不更新，不回填未来。
该5日调度是本项目适配，不声称原论文唯一设定。
作者README提示评价IC需回到raw labels：同时记录训练归一化目标和真实收益目标上的指标，不混称。

主回测报告新专家独立策略。
对C21和C32各增加一个事前固定的10%新分数+90%冻结v5分数对照（同日rank到统一尺度），作为低扰动增量测试，不再调比例。
同样加入C20/C30的10%组合对照，区分新骨干与框架的贡献。

Round23.6不接收Round26本轮新Alpha，后续跨线组合须新立实验。

================================================================
9. Round26共同目标：不是只找高IC，要验证相对v5增量
================================================================

主要对照始终为原v5。P1只作为冻结的第二参照，不替代基线。
不将Round23.6最好模型与Round26最好因子直接组合再报告最高Sharpe。

Round26两类输出：
FAMILY_FORMULA：表达式、源字段、known_at、可重复计算的因子值。
FAMILY_LEARNED：关系/序列/LOB模型在历史OOF生成的分数；当成学习因子进入v5。

先复用原Factor Evidence Factory，再补以下必做环节：
PIT → 年度覆盖 → 缺失/常数 → IC/RankIC → 负向与非单调形状 → 顶部样本 → 条件信息 → 冗余 → Head-specific Add/Remove/Replace → 完整交易增量。

每个候选至少：
- 同一fold与相同候选的raw IC、rolling 60/120/252；不足日期明确NA。
- 日横截面少于5个有效样本时不硬算稳定IC；记录可评价日比例。
- 稳定负IC允许作为预测信号，方向只在训练/验证期确定。
- 条件信息：训练折Ridge残差化（分别拟合新因子与标签对旧特征）；测试折残差相关只作诊断。
- 真正非线性增量以匹配OOF/交易对照判定，不能把普通相关改名conditional IC。
- 与旧因子和历史R17..R23失败表达式的语义/数值去重，不能重命名当新Alpha。

候选包：每个论文框架、每个Head、每个Outer折最多保留5个因子/输出。
选择只看验证期等日期权重指标，排序依次为：有效增量、稳定性、较低冗余、较低复杂度；规则冻结，不看Outer。
不删除未入选和失败记录。

模型矩阵（对每个合格候选包）：
A0 匹配v5。
A1 Five Add，Ten不变。
A2 Ten Add，Five不变。
A3 Both Add。

训练折内|Spearman rho|>=0.60且旧因子确在对应Head时，追加Remove-only/Replace配对；删除对象只在训练折选。
同一包最多删1个旧因子/Head，避免展开组合爆炸。
必须比较Replace vs Remove，不只vs A0。
不强求新因子单独正收益才允许加入；联合生成因子包可作为整体检验。
学习因子训练行也必须OOF。不得用上游模型对自身训练样本的fitted score训练下游v5。

================================================================
10. Round26-A：AlphaGen与AlphaForge
================================================================

L1采用官方CSI300公开/可授权数据与原入口，公开原数据获取失败要明确状态，不能伪造原文数值。
L2不使用作者静态“今天CSI300成分”回填我们的历史一进二股票池。

基础字段白名单G0：
open/high/low/close/pre_close、合法后复权收益、volume、amount、DEAL_AMOUNT、可核验float shares/turnover。
若某字段单位或PIT未证实就不进生成器，其他合法字段继续。

扩展字段白名单G1：
已验证known_at的资金流各档amount/volume/order-count；D日分钟的合法事件路径；涨停事件字段。
不能只因清单标交易日，就默认所有资金流和封单历史修订PIT安全。
金额与股/手单位不清先验证，不能计算虚假的“平均订单金额”。

算子限制：
原作者算子库为基础；禁止负shift、中心窗、未来聚合；防除零/溢出，NaN不能全部填0。
时间窗口固定{1,5,10,20,60}，表达式最多12个算子节点。
本轮每方法/seed/Outer折最多5000个去重有效表达式评估、候选pool最多20；完整日志必须计数。
这是本项目有界预算适配，不声称原论文相同规模。

ALPHA-01固定组：
G_RAW：原协同因子池奖励，目标为对应Head收益，框架不改成单因子独立排序。
G_RESIDUAL：以严格cross-fit的v5校准残差为目标，在相同预算下生成协同因子池。
残差e=y-E[y|v5_score]；校准只在先前训练块拟合。不能收益减rank。
奖励只访问开发训练区；验证区只用于有限最终筛选；Outer文件对生成进程不可见。

ALPHA-02固定组：
F_STATIC：作者生成器+固定因子组合。
F_DYNAMIC：相同因子池+作者动态选择/组合模块。
保持因子生成预算、源字段、训练窗口一致，区分生成与组合贡献。
动态权重的绩效反馈必须由已成熟标签组成，不得对D日因子使用未来完整窗口IC。
保留作者原生成器、表示和组合核心；不得把AlphaGen算法复制改名成AlphaForge。

两方法都输出原始表达式AST、标准化后的AST、源字段、有效区间、known_at、每fold权重和因子包。
不能只给“alpha_001”及一个IC数值。
所有包进入第9节匹配矩阵；不只回测最高IC一条。

================================================================
11. Round26-B：RSR与MASTER——跨股票信息
================================================================

目标：检验跨股票信息，而不是通过换一个深度模型混淆贡献。

REFERENCE：
RSR跑作者公开NASDAQ默认示例及同骨干无关系对照；不能将静态美股关系用于A股。
MASTER优先用作者修正后的公开CSI300数据与core code；原业务数据无法取得则标公开数据复现，不冒充原商业数据数值。

MASTER必须修正官方已指出的验证/测试标签过滤问题：完整股票节点前向计算，标签mask仅在loss/metric使用。

A股适配共同输入：完整日频时间序列；目标为下一有效交易日后复权close-to-close收益，真实成熟时间是下一收盘数据可得时点。
这是用于形成新学习因子的共同预测任务，不把该收益当实际可成交策略利润。
之后把它的OOF分数加入v5原Five/Ten目标检验。

预测支持股票池：
每Outer年用此前60日合法成交额选前800只作为固定context池，另加入当日实际首板候选。
训练日也按同一历史规则构建。不能预先把未来一年所有会首板的股票放进池子。
每日集合根据当时信息确定；不足800只用真实数量，不按未来存活筛选。
所有当日首板候选都必须预测，不能为了固定tensor大小删除候选。
动态节点使用mask；不能按test label缺失删除上下文。
该支持池是明确的工程适配，报告其与原论文股票池的区别。

RSR固定组：
R0：相同时间骨干、无关系传播。
R1：历史行业关系传播；要求行业as-of可证。
R2：事前固定的纯价格关系适配：用截至D-1的60日收益相关，正相关top10邻居，对称化、不含自身；不使用D+1。
R1行业不可用则阻断R1，不能拿当前行业替代；R0/R2仍可真实运行。
原图网络核心与损失保留，不把普通同行均值叫RSR。

另做透明、非论文模型诊断：用R2图对当日其他股票收益/涨停状态做leave-self-out加权聚合；作为“复杂图网络是否必要”的对照，不冒充论文方法。

MASTER固定组：
M0：同架构去掉跨股票信息，保留时间模块。
M1：完整MASTER。
M2：完整MASTER但市场门控固定常量，其他不变。
官方市场输入优先恢复真实CSI300/500/800历史指数与amount。
若确实无授权，另用全A等权/市值加权、历史规模分组的公开可重建指数，必须新ID MARKET_PROXY_ADAPTATION；不得声称原63维市场信号完整复现。
历史规模分组只用前一日合法市值，不用全期排名。

每种输出按日期、股票、fit_cutoff保存为学习因子，进入第9节Add/Remove/Replace。
同一控制/方法使用同样节点、特征、目标和预算，不根据test结果换数据池。

================================================================
12. Round26-C：R&D-Agent-Quant，允许自动研究但不允许看答案
================================================================

L1运行作者量化场景的最小完整循环：假设→代码→训练→评价→下一步，并保留所有轨迹。
L2使用用户当前策略的Factor Evidence Factory，分两种清楚标注：
- OFFICIAL_JOINT_REFERENCE：作者因子+模型共同优化，在独立公开基准上。
- V5_FACTOR_ONLY_ADAPTATION：固定v5结构，只允许生成因子代码。不是完整原联合优化的同义词。

每Outer折/seed最多20次研究提案；每提案最多3次代码修复。所有失败与重复均计入预算。
不允许agent修改：数据截止、费用、股票池、标签、验收器、参考基线、测试集。

执行隔离：
- 只挂载训练开发数据、匿名schema和训练反馈。
- 不挂载Outer测试标签/最终NAV、历史赢家报告、私有凭证或Git写权限。
- 代码运行限制目录和网络，调用可信评价器。
- 验证集反馈最多对最终5个候选返回，不能无限轮追逐验证IC。
- Outer只在全部候选冻结后统一评价一次。

LLM使用：
优先已配置且获授权的本地端点；模型ID/版本、temperature、token预算记录。
不得把用户真实交易账本/供应商原始行/秘密上传外部模型。
没有已授权端点或预算则BLOCKED_LLM_ACCESS，而不是偷偷调用付费API。
LLM/API权限不是“数据”，不得写进无权限数据.md；写详细blockers.json。
不能以静态手工写5个因子代替RD-Agent运行后声称完成。

================================================================
13. Round26-D：DeepLOB，不混淆可公开复现与A股可落地
================================================================

L1必须尝试作者FI-2010数据与作者网络。原训练/验证/测试划分和归一化，固定一个作者demo horizon；不根据结果挑horizon。
滚动窗口不得跨股票、跨交易日、跨train/test边界；标签预测区间跨边界要purge。
官方数据许可不能确认则阻断分发，允许本地依法使用。

A股适配需要真实至少十档bid/ask价格和数量、事件时间、证券标识及清晰快照语义。
盘口数据与逐笔订单是不同产品；缺订单ID不能宣称重建完整排队队列。

若有合法数据：
- 每个D日收盘前最后100个合法book状态，生成direction probabilities。
- 训练微观目标固定10事件后的中间价方向，标签只在其成熟后使用，不从未来交易日补到当前日。
- 方向阈值按训练数据/作者配置冻结，不能全历史计算。
- 新因子固定p_up-p_down，及预测熵作为诊断。
- D-close模型只用截至该时刻状态，不能引入D+1盘口。
- 做同数据线性盘口基线对照。
- 该微观方向预测不是多日收益保证，必须进入v5增量矩阵实测。

没有A股LOB时：
完成能完成的L1；L2标BLOCKED_LOB_DATA。
不拿分钟bar补10档、不拿FI-2010准确率写成A股收益、不删该研究任务。
数据确因账号权限被拒绝，按第18节输出实际源路径；只是本地找不到而无已知授权来源时记DATA_NOT_FOUND，不能编造权限路径。

================================================================
14. Round26补充机制组：使用已有数据，不把旧失败因子换名
================================================================

这些属于前次研究规划中的机制对照，不冒充上述论文的新贡献。
先与历史因子注册表比对；已存在的原样复用作控制，不能当新增Alpha重复选优。

E1 原始订单结构：将合法raw BUY/SELL_ORD、各档金额/量和DEAL_AMOUNT纳入生成器扩展字段；单位未验证则禁止入池。分档不等于机构/游资身份。

E2 D日路径：固定可解释诊断——首次触板后最大离板幅度、开板后恢复时间、开板阶段成交金额占比。仅由D日分钟构建；准确区分首次触板、bar内high触板与close封住。缺bar不伪造。与既有封板长度、last30 ratio及R18/21表达式去重。

E3 公告供给与催化：先恢复已公告解禁/减持/回购/业绩预告，保留发布时点与修订版本。
固定两个研究输入：
- 在D时已公布、未来20交易日内计划解禁股数/当时已知自由流通股数；
- 过去5交易日已公开的回购/减持/业绩预告事件类型及距公告天数。
分母单位不明或公告时间不可靠则阻断；“未来计划”今天已公布可用，未来才公布不可用。
不自动让大模型阅读后来的新闻摘要。
本轮不追加任意窗口和文本模型。

这些组逐项走第9节证据与交易矩阵。

数据与执行补充：恢复所有合法首板候选（不只是历史买到者）的D+1至固定D+5分钟、及真实未退出持仓必要日期。
核验集合竞价API是否返回真实历史时间序列还是最终一条结果。
本轮只建立可执行性数据与固定压力对照，不再搜索分钟止盈止损规则。
需要采购的先列缺口/真实权限结果，不自动支付。

================================================================
15. 执行与统计：统一，不再只做漂亮summary
================================================================

所有单账户比较：相同初始资金、费用、候选、日历、槽位/持仓规则。
P1真实仓位组合改变持仓广度，另列scope并和同广度等权组比较。

S0 原lane执行。
S1 买入额外10bp不利滑点。
S2 买入额外20bp不利滑点。
S3 原计划开盘买单若触及合法涨停限制则拒绝，不替换、不重新rank、不覆盖仓位倍率。
若原执行器已经同样拒绝，标ALREADY_ENFORCED并用真实计划订单/拒单trace证明，不伪造另一套S3。
基线与候选都在同情景运行，比较delta；不能仅列候选加费后仍盈利。
后续日期现金分化导致计划数量变化是正常路径效应，不要求整段未来所有订单与S0一致。

完整归因：Top1/Top2真实净人民币PnL、数量、费用、未实现损益、替换/共同交易、资金与复利桥。
收益率求和不等于人民币PnL；trades.head(200)不等于完整归因。

风险与收益分账：
- Alpha：沿用原冻结经济门，符号正确。
- Risk：报告CAGR保留、UI、MaxDD、63日最差窗口、ES95和恢复时间，不强求所有指标同时上升。
不能为本轮赢家降低门槛。
fee_direction_ok必须从成对费用压力计算，不能“同费率”就硬写True。
nav_improvement=candidateNAV/baseNAV-1，不能绝对值。

统计：
5000次配对circular-block bootstrap，主block20日，10/40日各1000次敏感性。
抽样索引同时用于候选和基线；重算各自Sharpe后相减。
CAGR统一同一个时间年化定义；不能主表日历年、bootstrap252交易年混为相同数字。
回撤序列从初始NAV纳入峰值，避免忽略第一天亏损。
不dropna静默裁掉坏日期。日历不一致先报错。

SPA用arch.bootstrap.SPA，loss=-真实日净收益；同scope去重并登记所有候选。
控制主方法、适配、敏感性多重选择；原基准论文复现不与A股实测混到一个p-value。
每seed不是一个可挑赢家。主方法统计预先定义为各seed策略日收益等权的研究汇总，另保留每seed原策略；汇总不是自动部署的seed组合。
记录全搜索trial数量、被看过数据、失败和计算预算。
没有新独立Forward不能LIVE_READY。
曾用于历史研究的日期即使重新切成train/test，也不是从未看过的独立检验；研究报告必须保留这一限制。

================================================================
16. 计算预算、复现状态与执行顺序
================================================================

先完成每篇L0的真实smoke（含模型更新、预测和有限实算），再依次L1/L2；不能只写代码不运行。
神经模型使用作者单一配置的训练轮数与停止规则；本项目适配上限200epochs，原配置更小按原值；早停只看训练/验证，patience10且只在原框架允许的适配层使用。
预算截断必须写BUDGETED_ADAPTATION，不能声称原文完整数值复现。
生成方法按第10节表达式预算；RD按第12节提案预算。

最多1个GPU训练任务+1个CPU任务并行；最多使用当前可分配内存70%，不争抢其他用户资源。
启动前probe资源并记录。每篇每阶段设置checkpoint、可续跑的epochs/evaluations上限。
如果已有项目资源上限则遵守；否则单方法L1阶段24 GPU小时、L2全部fold/seed阶段96 GPU小时作为首轮硬上限；无GPU可运行部分CPU任务。
达到上限保存所有结果、状态和resume命令，继续其他独立框架，不宣称全完成、不自动租GPU。
不要预先拿这个预算作为不开始执行的理由。

执行批次：
0. 源冻结、共享数据/PIT/标签/账本修复；两个分支提交协议。
1. 23.6：P1各对象对照、Super Learner、BOA。
2. 26：AlphaGen、AlphaForge及机制组数据恢复。
3. 23.6：TRA、DoubleAdapt（含相同骨干对照）。
4. 26：RSR、MASTER。
5. 26：RD-Agent-Quant、DeepLOB；数据/服务阻断不阻止其他组。
6. 全部同口径回测、统计、权限清单、报告与远端校验。

十篇必须在执行状态表各有一行，L0/L1/L2分别记录。
训练/回测数量从实际manifest汇总，不靠预期数量填写。
“没有新baseline”是允许结论；“没有执行却写完成”不是。

================================================================
17. 必须通过的集成验收
================================================================

不是只检查函数存在或字段叫PASS：

A. 修改任何未来标签，历史候选、特征、已生成分数与计划订单不变。
B. 截至日期的前缀数据回放，与全数据as-of回放一致。
C. 插入known_at在未来的旧event_date修订，历史输入不变。
D. 未成熟标签不入训练、校准、router memory、BOA反馈或生成奖励。
E. 0新权重/单专家权重=1正确回退；不因二次rank或inner join改变基线。
F. 日截面不同股票排列不改变同日在线更新结果。
G. SuperLearner系数与官方方法固定样例一致；BOA无延迟流与官方实现一致。
H. P1式(6)仓位组合与score融合存在明确非等价构造测试。
I. Alpha表达式禁止未来运算；生成环境无法读Outer标签和NAV。
J. 学习因子训练输入必须OOF，不能模型内拟合再叠模型。
K. MASTER/RSR保留所有合法预测节点，标签mask只在损失/指标。
L. DeepLOB不能用分钟伪盘口；序列与标签不跨切分边界。
M. 原仓位缩放使用乘法，不覆盖0/半仓；资金总额不被复制。
N. S3保留计划身份及槽位，未成交不能补买下一名。
O. 标签成本与组合成本不重复扣除；现金+市值=NAV。
P. Gate失败真实停止依赖训练/晋级，其他独立组继续。
Q. 两条分支不接收对方本轮胜出工件；CURRENT不变。
R. 无权限数据.md严格通过第18节格式断言。

所有真实数据集成失败例保留脱敏小样本或schema+hash，不能删测试。

================================================================
18. 用户要求的“无权限数据.md”：严格只有名字和路径
================================================================

每个round输出自己的：
result/round23_6/无权限数据.md
result/round26/无权限数据.md

最终汇总去重版本：
result/round23_6_round26_master/无权限数据.md

内容只允许以下逐行格式：
1. 数据名称，数据路径：真实路径或源URI
2. 数据名称，数据路径：真实路径或源URI

不写标题、前言、原因、权限错误、解决建议、表格、时间、备注或结论。
条目必须是本轮确实需要且实际因访问/下载权限受阻的数据。

路径规则：
- 本地文件/目录：实际无法读取的绝对路径。
- 数据库：真实源URI，例如clickhouse://127.0.0.1:18123/hermes/实际表名。
- 供应商API：实际发出并被权限拒绝的endpoint/product URI，不包含密码、token、cookie。
- 不把规划中的下载目标路径冒充源路径。
- 不因“本地没有数据”编造一个不存在的数据目录。
- 同源不同字段确需区分时写进数据名称，不额外增加说明列。

没有权限阻断：生成空文件（0字节），不写“无”或“全部已获取”。
详细原因和技术错误只写data_recovery_jobs.jsonl/blockers.json。
LLM端点、软件许可证、计算预算不是数据，不写入本MD。
验证每一非空行匹配：^\d+\. .+，数据路径：.+$
确保行内无凭证、条目去重、编号连续。

不要预先根据旧清单把所有权限问题填进去。执行后的真实结果决定该文件。

================================================================
19. 必须交付的代码与产物
================================================================

运行入口：
tools/run_round23_6.py
tools/run_round26.py
tools/run_round23_6_round26.py

参数：
--phase {audit,reference,adapt,backtest,stats,report,publish,all}
--method
--outer-year
--seed
--resume
--jobs
--max-gpu-hours

支持依赖DAG与断点续跑。不能把所有任务塞成一个失败后全部中断的巨大函数。

共同产物：
paper_registry.csv
source_lock.json（repo URL/commit/license/配置/论文hash）
resolved_author_configs/
experiment_prereg.yaml
execution_status_by_paper.csv
reference_replication_results.csv
adaptation_differences.md
baseline_parity.json
candidate_and_fold_manifests/
oof_registry.parquet
model_registry.csv
data_recovery_jobs.jsonl
trial_registry.csv
all_results.csv
daily_returns.csv.gz
yearly_metrics.csv
top1_top2_actual_pnl.csv
risk_metrics.csv
execution_sensitivity.csv
bootstrap_results.csv
spa_results.json
failure_examples/
tests_summary.json
artifact_manifest.csv（路径/大小/SHA256/许可）
无权限数据.md
REPRODUCE.md
FINAL_DECISION.md

23.6额外：
ROUND23_6_ENSEMBLE_REPORT.md
score_vs_portfolio_ablation.csv
expert_capital_and_netting_ledger/
meta_weights_by_time.csv
online_feedback_ledger/

26额外：
ROUND26_ALPHA_REPORT.md
factor_AST_registry.jsonl
factor_source_and_known_at.csv
factor_evidence/
legacy_factor_dedup.csv
head_specific_add_remove_replace.csv
learned_factor_oof/
agent_trajectory_manifest.json
LOB_reference_vs_Ashare_status.json

主报告每篇都回答：真正执行什么、原版与适配差异、实际数据、真实模型数、回测数、是否改善v5、风险、未完成项。
不得只展示最佳候选；不能把paper-inspired方法称原论文完整复现。

================================================================
20. GitHub提交和实际Push
================================================================

使用独立发布clone/worktree，不擅自修改服务器项目origin。
新分支round23_6、round26分别维护。
不修改baseline_v5、round23_5、任何Round24或生产分支。

共享模块相同hash；必要时cherry-pick明确的共享基础设施提交，不合并整条赢家实验分支。
有未提交修改先保留，不reset/clean/force。

至少三次逻辑提交：
1. 文献源锁、协议、数据schema和基础测试。
2. 真实算法adapter、执行入口和新增测试。
3. 实际结果、完整报告、权限清单与复现manifest。

上传：自有完整源码依赖、配置、运行日志摘要、各fold参数与截止、全部成绩与统计、实际日收益/NAV摘要、因子定义、订单/成交归因、测试、许可证引用。
不能只上传report。

不要上传：
.env、API密钥、密码、SSH私钥；供应商未授权原始库；虚拟环境；临时缓存；私人账户资料。
大工件可分块压缩或使用已有合规LFS；不擅自开通付费存储。
本地保留完整工件，GitHub上传schema、hash、大小、合法派生摘要与生成命令。
source_dependency_manifest必须标明哪些数据只能在服务器获得，不能声称无数据也可完整复现。

第三方源码使用git submodule/锁commit下载器或按许可证保留版权；不得删除许可。研究不自动授权商业部署。

提交前：secret scan、large-file scan、git diff --check、白名单git add、测试状态。

在对应worktree分别执行：
git push -u origin HEAD:round23_6
git push -u origin HEAD:round26

禁止force push。
push被拒绝记录真实原因，不猜凭证、不覆盖远端。

分别比较：
git rev-parse HEAD
git ls-remote origin refs/heads/round23_6
git ls-remote origin refs/heads/round26

远端SHA必须与对应本地HEAD一致。
随后fetch远端并实际读取主报告、源码、测试、all_results.csv、无权限数据.md验证内容/hash。
不能只确认分支名称存在。

master汇总放round26分支的result/round23_6_round26_master/，记录两分支最终SHA和各自主报告链接。
如果部分客观阻断，仍推送真实PARTIAL产物，不把状态写COMPLETE。

================================================================
21. 最终终端与回复
================================================================

按两轮分别输出：
- 真实完成/部分完成/阻断状态。
- 十篇逐项L0/L1/L2状态。
- P1对象对齐结果。
- 实际训练配置数、fold模型数、回测数、失败与阻断数。
- 相同窗口v5、候选CAGR/Sharpe/MaxDD/UI/交易笔数及增量。
- 能否替换baseline，原因与证据。
- 全部论文方法是否真的运行，哪些只完成原公开基准。
- 无权限数据.md的路径（文件内部仍仅含名称和路径）。
- 两条远端分支URL、commit SHA、REMOTE_SHA_VERIFIED。

强制：
CURRENT_BASELINE_CHANGED=FALSE
PRODUCTION_CONFIG_CHANGED=FALSE
AUTO_PROMOTE=FALSE
ROUND24_EXECUTED=FALSE

执行前不要再次要求用户指定论文或设计实验。
本文已给定十篇、P1延伸、对照、数据与权限处理、预算和交付。
只有不可绕过的真实权限或资源问题才阻断对应任务，其余任务继续。

现在开始按依赖顺序执行，完成后实际push代码和结果到两个新分支。
