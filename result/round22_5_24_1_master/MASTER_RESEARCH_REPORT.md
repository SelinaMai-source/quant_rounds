# Round22.5–24.1 总研究报告

## 1. 范围与锁定基线

一进二锁定基线仍是 baseline_v5=R16_O01，全历史 2018-01-02～2026-07-24，Sharpe=1.8756627913361097，CAGR=0.5499812832273525，MaxDD=-0.24056327715394388，NAV=12750089.355724689，笔数=1107，资金=300000。费用 ROUND7_FLAT_5BP。没有改执行器。
二进三是独立研究线。下面的表把两条线分开，不把它们的 Sharpe 排在同一张名次里，也不拿二进三 CAGR 去比一进二。

## 2. 假设与预注册

假设写在各轮 frozen_manifest.json，看结果之后没有改口径。每条假设只有 PASS、FAIL 或 BLOCKED。

## 3. 数据与产物

历史 result/round21* 到 result/round24 只读。新产物在 result/round22_5、result/round23_1、result/round24_1 和本目录。缺文件记 MISSING_ARTIFACT，没有用报告里的数字编造预测或成交。

## 4. Round22.5 事实表与多重检验

experiment_fact_table.parquet 来自 Round21.3–24 的真实 summary。缺 Top1/Top2 为 NOT_IDENTIFIABLE。多重检验登记在 multiple_testing_registry.csv。自助区间不是后验。

## 5. C1/C4 覆盖与滚动

C4 公式未改。n_valid_3=5116，n_valid_4=5116，n_valid_5=11778270。滚动按行，不把日历空档插进窗口。

## 6. 空值、打乱与截面置换

ALLNAN 净值与 v5 相同：True。SHUFFLE 对齐冻结 Sharpe：True（读到 1.8125087429942996）。截面置换种子 17/42/73 不计 alpha。

## 7. 归因与净值桥

C1 裁决=C1_NO_STABLE_INCREMENTAL_ALPHA。PARTIAL 财富比重算=1.2636868422206864，FULL_TRAIN 财富比重算=0.8067108503924177。
C4 全历史={'sharpe': 1.885273629050742, 'cagr': 0.5557948579127749, 'delta_sharpe': None, 'delta_cagr': None, 'nav_end': 13165091.68084055, 'max_drawdown': -0.24056327715394388, 'economically_material': None}。净值桥=RECONCILED，残差/净值=1.4608879179302726e-16。

## 8. FS2 与执行压力

FS2=PREDICTIVE_NOT_ECONOMIC。公式是经营现金流/平均资产。执行压力 7 行，主敏感性没有当日替补。

## 9. Round22.5 决策

KEEP_BASELINE_V5。C1=ARCHIVE_C1_NO_STABLE_INCREMENTAL_ALPHA。C4=WEAK_RESEARCH_SIGNAL_ONLY。FS2=PREDICTIVE_NOT_ECONOMIC。

## 10. Round23.1 目标与嵌套权重

正式主目标是 O4。O1 的日 Spearman 只作为失败对照，Ten=0 不是正式候选。内折不足 120 个信号日则 Ten=0.90。没有在外折 Sharpe 上重调。
选出的十头权重={'2018': 0.9, '2019': 0.75, '2020': 0.75, '2021': 0.9, '2022': 1.0, '2023': 0.9, '2024': 0.75, '2025': 0.5, '2026': 0.75}。与 0.90 相同=False。

## 11. LightGBM、多样性与规则法医

多样性 Spearman=0.8405585121740321，Top1 不一致=0.39803439803439805，结论=DIVERSITY_PASS。未过门则没有融合袖套。规则没有重调阈值。

## 12. Round23.1 决策

PREDICTIVE_ONLY。经济门、统计、Top 变化和执行同时成立才是 V6_RESEARCH_CANDIDATE。

## 13. Round24.1 二板状态、标签与特征

板块审计 board_ok=True，检查行=105，不符=0。特征集=REDUCED，阻断=['BLOCKED_PIT']。
label_2to3_net 的 0.001 留在诊断标签里。净值由执行器按 ROUND7_FLAT_5BP 从价格计算，不把标签成本再扣一次。

## 14. 阶段模型、弃权与执行

阶段 A 是可交易且已成熟样本上的涨跌分类。阶段 B 用 P 排序，或用同一树参数的回归预测已实现净收益。弃权阈值只在内折按已选可交易机会的平均已实现净收益选择。不足 5 个名字的日子不做 RankIC。

## 15. Round24.1 决策

SECOND_TO_THIRD_RESEARCH_ONLY。门={'level': False, 'paired_increment': False, 'fee_robust': True, 'catastrophic': True, 'execution_dependent': False, 'passed': False}。没有替换 baseline_2to3_v0。

## 16. 分线对照、测试与生产配置

### 一进二（只在这条线内比较，对照 v5）

| 项 | 结果 |
| --- | --- |
| Round22.5 | KEEP_BASELINE_V5 |
| Round23.1 | PREDICTIVE_ONLY |
| v5 Sharpe | 1.8756627913361097 |

### 二进三（独立，不和上一张表的 Sharpe 排名）

| 项 | 结果 |
| --- | --- |
| Round24.1 | SECOND_TO_THIRD_RESEARCH_ONLY |
| 特征集 | REDUCED |

二进三袖套指标见下表，只在二进三内部阅读。

```
          code                  status    sharpe      cagr  max_drawdown       nav_end  completed_trades                                                                            source new_backtest  nav_start  total_return  n_obs window_start window_end    fee_schedule       compare_to restarted_capital scenario counts_as_alpha  round                       scope  initial_cash economically_material           gate_id  delta_cagr_vs_v5_not_used  v5_nav_reference_not_a_benchmark
     T0_REUSED REUSED_ROUND24_ARTIFACT -0.224850 -0.144760     -0.899318  78719.003204              2671 /data1/stock-mxy/课题/一进二策略/result/round24/candidates/baseline_2to3_v0/summary.json        False        NaN           NaN    NaN          NaN        NaN             NaN              NaN               NaN      NaN             NaN    NaN                         NaN           NaN                   NaN               NaN                        NaN                               NaN
     T2_REUSED REUSED_ROUND24_ARTIFACT  0.026013 -0.049594     -0.701943 194140.935840              2794       /data1/stock-mxy/课题/一进二策略/result/round24/candidates/T2_RETRAIN/summary.json        False        NaN           NaN    NaN          NaN        NaN             NaN              NaN               NaN      NaN             NaN    NaN                         NaN           NaN                   NaN               NaN                        NaN                               NaN
     A1_P_RANK                     ran -0.458882 -0.187293     -0.841753  50878.894694              2859                                                                               NaN          NaN   300000.0     -0.830404 2075.0   2018-01-02 2026-07-24 ROUND7_FLAT_5BP baseline_2to3_v0             False    False            True   24.1 SECOND_TO_THIRD_INDEPENDENT      300000.0                 False NOT_THE_1TO2_GATE                  -0.737274                      1.275009e+07
  A2_REGRESSOR                     ran -0.056436 -0.069018     -0.660611 162700.436215              2520                                                                               NaN          NaN   300000.0     -0.457665 2075.0   2018-01-02 2026-07-24 ROUND7_FLAT_5BP baseline_2to3_v0             False    False            True   24.1 SECOND_TO_THIRD_INDEPENDENT      300000.0                 False NOT_THE_1TO2_GATE                  -0.619000                      1.275009e+07
  A3_ABSTAIN_P                     ran  0.003810 -0.023658     -0.521033 244432.433345              1244                                                                               NaN          NaN   300000.0     -0.185225 2075.0   2018-01-02 2026-07-24 ROUND7_FLAT_5BP baseline_2to3_v0             False    False            True   24.1 SECOND_TO_THIRD_INDEPENDENT      300000.0                 False NOT_THE_1TO2_GATE                  -0.573639                      1.275009e+07
A4_ABSTAIN_REG                     ran -0.199688 -0.063540     -0.613718 171075.656263              1146                                                                               NaN          NaN   300000.0     -0.429748 2075.0   2018-01-02 2026-07-24 ROUND7_FLAT_5BP baseline_2to3_v0             False    False            True   24.1 SECOND_TO_THIRD_INDEPENDENT      300000.0                 False NOT_THE_1TO2_GATE                  -0.613522                      1.275009e+07
```

测试：通过 30，失败 0，合计 30。
假设：PASS=11，FAIL=5，BLOCKED=0。

CURRENT 基线未改。PRODUCTION_CONFIG_CHANGED=false。AUTO_PROMOTE=false。LIVE_READY=false。没有独立 Forward，因此没有 LIVE_READY。
