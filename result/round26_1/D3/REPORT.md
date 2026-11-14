# D3 AlphaGen真实A股执行

状态：PARTIAL_SEED_CHECKPOINT。这是锁定原表达式求值、MseAlphaPool协同优化、原LSTMSharedNet与MaskablePPO的预算化A股适配，不是随机公式、合成样本更新，也不冒称原论文完整CSI300数值复现。

当前已完成原始PPO生成模型20/42，完成模型环境步2000000。每折上限100000环境步/20000次原求值调用（含重复及原池拒绝），未为弱结果加搜索。val只用Outer前一年成熟标签选checkpoint；AST冻结后仅重估先前年份池权重。所有AST、权重、检查点、尝试/失效表达式与模型及资源manifest保留。

D日已知首板domain在完整股票OHLCV序列上先rolling再映射；无next1/next2/label进入表达式。未用未来标签决定预测池。RAW与RES使用相同历史有效目标行，RES是收益率减真正历史OOF校准收益，不是减百分位。原normalize/零分母语义保留，训练loss只用截止前成熟目标。

嵌套因子从2020起生成，每年只用先前3年；主Add模型自2023起，仅用2020-2022等历史滚动生成值，不把当年AST训练内拟合值当OOF。主比较是2023-01-03至2026-07-24同窗口300000初始资金的原生匹配无因子基线；其两个Head均经真实相同入口原特征重放，旧v5窗口参考保留，是否存在差异见baseline_head_replay/STATUS.json。正式v5不变。不能与2018全期基线混比。信号资格来自冻结candidate manifest，2026信号截至07-01；07-24是估值及退出尾部，不声称07-24仍有完整预测股池。旧v5窗口参考信号截至06-01；真实无因子重放补入06-02—07-01的655键，共同16764键融合分数完全一致，不将覆盖差异算作因子增量。详见baseline_head_replay/coverage_difference_evidence.json。这些年份是开发历史，不是未触碰测试。

每个可训练factor均真正完成Five/Ten/Both Add，原始IC不作为性能前置门。训练折|rho|>=.60时保存Remove-only与Replace，其余年份明确原Head回退。三seed只等权，标度来自各年度生成前训练历史，绝不挑最好seed。

```csv
experiment_id,CAGR,Sharpe,MaxDD,UI,trades
AG_A0,1.0819480044645982,2.627762557774374,-0.23226367272935045,0.06820905232387404,478
AG_RAW_17_A1_FIVE_ADD,0.9562058554648436,2.411737187416201,-0.25215107259248193,0.07839005879287118,481
AG_RAW_17_A2_TEN_ADD,1.0429333494303776,2.5847578594192253,-0.2471065724155872,0.07741548247377308,482
AG_RAW_17_A3_BOTH_ADD,0.7946192635380032,2.189262029663296,-0.2475586121631944,0.08369704166212807,484
AG_RES_17_A1_FIVE_ADD,1.0359869924943572,2.5502378218823183,-0.23397998733133163,0.06856317243062789,481
AG_RES_17_A2_TEN_ADD,0.8500138921408344,2.3206974787290497,-0.22677259069820344,0.0709665969545438,481
AG_RES_17_A3_BOTH_ADD,0.8631026741593313,2.3364585370834487,-0.2350376116657984,0.07247894442117685,479

```

真实入口反事实范围见head_entry_tests；完整PPO反事实重训状态与六个实际诊断模型见generation_counterfactual_tests/STATUS.json，未完成时不得称为通过；这些不是额外Alpha搜索。经济门逐项结果见economic_decisions.csv/json；风险改善分列，未做次要配置成本压力时费用证据为UNKNOWN。

完整raw/rolling/年度IC、缺失极值、尾部形状、训练相关去重、条件真实OOF增量、实际交易/人民币PnL、风险及paired成本/Bootstrap/SPA见结果表。S3原限制已生效，不复制一份成绩。具体执行量和未完成seed以STATUS和factor_run_registry为准，不由报告把NOT_IMPLEMENTED改成外部阻断。

CURRENT_BASELINE_CHANGED=FALSE
PRODUCTION_CONFIG_CHANGED=FALSE

计数口径：STATUS兼容字段effective_expression_evaluations表示原表达式求值调用次数，含重复及原池拒绝，不是不同AST数、优化更新数或新增经济试验数。不同AST、异常、池准入和PPO更新分别见completed_generation_models.csv及各工件审计。

<!-- LIVE_GENERATION_PROGRESS -->
当前原始生成完成20/42；上方入模、经济门与风险结果仍对应已完成的seed检查点17，不将新生成折冒称新经济回测。已完成当前模型合计2000000环境步、293899次原求值调用（含重复及原池拒绝）；不同AST、异常及PPO更新单列，失败/未完成尝试另存，不计入此完成模型合计。
完整PPO时间反事实六项已通过。原始生成工件审计只验证工件；研究总状态以本STATUS及真实结果表为准。
