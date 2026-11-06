# Round22.5 研究报告

决策：**KEEP_BASELINE_V5**。不自动升 CURRENT。C1=ARCHIVE_C1_NO_STABLE_INCREMENTAL_ALPHA。C4=WEAK_RESEARCH_SIGNAL_ONLY。FS2=PREDICTIVE_NOT_ECONOMIC。

本轮没有新因子、没有新的 C 公式、没有重调 XGB。经济门仍是 `R21_3_ECONOMIC_MATERIALITY_V1`。短窗 Sharpe 不与 v5 的 1.8756627913361097 混排。

## 锁定锚点

baseline_v5=R16_O01，2018-01-02～2026-07-24，Sharpe=1.8756627913361097，CAGR=0.5499812832273525，MaxDD=-0.24056327715394388，NAV=12750089.355724689，笔数=1107，资金=300000。

## 事实表

真实账本来自 Round21.3 到 Round24 的 summary.json。缺 Top1/Top2 的行记为 NOT_IDENTIFIABLE。见 `experiment_fact_table.parquet` 与 `multiple_testing_registry.csv`。

## C1 / C4 覆盖与滚动

C4 公式未改：`mean(sign(NET_FLOW).shift(1).rolling(5, min_periods=3))`。按行滚动，不插入日历空档。n_valid_3=5116，n_valid_4=5116，n_valid_5=11778270。

## 空值、打乱、截面置换

ALLNAN 与 v5 净值相同：True。SHUFFLE Sharpe=1.8125087429942996，对齐冻结值 1.8125087429942996：True。
日内截面置换种子 17/42/73 是诊断，不是 alpha。

```
 seed    sharpe      cagr       nav_end  counts_as_alpha
   17 -0.660722 -0.110653 109998.407703            False
   42 -0.112775 -0.034986 221205.082800            False
   73 -0.469101 -0.099915 121893.423960            False
```

## C1 FULL_TRAIN

重算 PARTIAL 财富比=1.2636868422206864，FULL_TRAIN 财富比=0.8067108503924177。历史引用 1.263687 与 0.806711 只作核对，不代替重算。裁决：C1_NO_STABLE_INCREMENTAL_ALPHA。没有新的 C1 组合。

## C4 FULL_TRAIN

2024、2025 分开报。2026 不是完整年。条件 IC、NDCG@1/@2、变更后的 Top1/Top2 成熟标签、块长 20 的 5000 次配对自助都在 `c4_full_train_attribution.json`。自助区间不是后验。

全历史摘要：{'sharpe': 1.885273629050742, 'cagr': 0.5557948579127749, 'delta_sharpe': None, 'delta_cagr': None, 'nav_end': 13165091.68084055, 'max_drawdown': -0.24056327715394388, 'economically_material': None}

自助：{'status': 'OK', 'n': 619, 'block': 20, 'n_boot': 5000, 'mean_excess': 5.472508412938327e-05, 'ci95_low': -0.00022764616581552553, 'ci95_high': 0.00036194042599038066, 'share_boot_mean_gt_0': 0.6368, 'interpretation': 'block bootstrap interval, not a posterior'}

## 净值桥

状态=RECONCILED。现金+市值=总资产：True。费用已含在成交现金里，总额=446540.19993530144。残差/净值=1.4608879179302726e-16。终点净值同时包含未实现市值和复利后的总资产。

## FS2

公式是经营现金流/平均资产，不是现金转换变化。裁决=PREDICTIVE_NOT_ECONOMIC。PIT={'formula': 'OperatingCashFlow/AverageAssets', 'not_cash_conversion_change': True, 'future_known_at_rows_before_filter': 0, 'asof': 'known_at<=trade_date 后按标的和交易日取最近版本', 'version_collapse': 'latest known_at', 'staleness_days_median': 56.0, 'staleness_days_p90': 133.0, 'filled_with_zero': False, 'pit_pass': 'True'}。筛选={'five': {'pass': True, 'complete_outer_years': 8, 'top2_mean': 0.0024290370176174338, 'same_sign_share': 0.75, 'years': [{'year': 2018, 'top2_mean': -0.0022898310342823118, 'n': 486}, {'year': 2019, 'top2_mean': 0.0056071145927694185, 'n': 488}, {'year': 2020, 'top2_mean': 0.008263612739938843, 'n': 486}, {'year': 2021, 'top2_mean': 0.002102890563322676, 'n': 486}, {'year': 2022, 'top2_mean': 0.0013187659735216044, 'n': 483}, {'year': 2023, 'top2_mean': 0.0014413252048300338, 'n': 484}, {'year': 2024, 'top2_mean': -0.0002510113754580212, 'n': 484}, {'year': 2025, 'top2_mean': 0.0032394294762972285, 'n': 486}]}, 'ten': {'pass': True, 'complete_outer_years': 8, 'top2_mean': 0.0022652321850551767, 'same_sign_share': 0.625, 'years': [{'year': 2018, 'top2_mean': -0.003353725178230656, 'n': 486}, {'year': 2019, 'top2_mean': -0.0026055432188405027, 'n': 488}, {'year': 2020, 'top2_mean': 0.011130800063776352, 'n': 486}, {'year': 2021, 'top2_mean': 0.003895533956029492, 'n': 486}, {'year': 2022, 'top2_mean': -0.0024723064962896735, 'n': 483}, {'year': 2023, 'top2_mean': 0.0029099146729706397, 'n': 484}, {'year': 2024, 'top2_mean': 0.0029396188456089533, 'n': 484}, {'year': 2025, 'top2_mean': 0.005677564835416809, 'n': 486}]}}。没有 Remove/Replace，没有双头同时加，没有新参数。

## 执行压力

S0 复用 R223_E0。S1 +10bp 买滑点，S2 +20bp。S3 用真实涨停价把开盘封死记为未成交。S4 缺口≥7% 成交 50%。S5 缺口≥9% 成交 0%。S4/S5 是压力，不是板块规则。主敏感性没有当日替补。替补只在 S3_REPLACEMENT。

开盘封死频率：{'n_top2_slots': 2096, 'open_limit_up_share': 0.14599236641221375, 'gap_ge_7_share': 0.3139312977099237, 'gap_ge_9_share': 0.22089694656488548, 'limit_price_source': 'market limit_up when finite and not sentinel; no flat 1.099 rule', 'main_sensitivity': 'NO_SAME_DAY_REPLACEMENT'}

## 假设

```
                          id                                      claim result                     artifact
H225_C1_FULL_TRAIN_INCREMENT C1 在 2024-01-01 起的 FULL_TRAIN 上相对 v5 有稳定增量   FAIL H225_C1_FULL_TRAIN_INCREMENT
            H225_C4_ECONOMIC                       C4 的全历史增量过 R21_3 经济门   FAIL             H225_C4_ECONOMIC
       H225_ALLNAN_INVARIANT                      全 NaN 对照的 NAV 与 v5 相同   PASS        H225_ALLNAN_INVARIANT
        H225_SHUFFLE_RECHECK                既有 SHUFFLE 袖套 Sharpe 等于冻结数字   PASS         H225_SHUFFLE_RECHECK
         H225_PERM_NOT_ALPHA                          日内截面置换不是可晋级 alpha   PASS          H225_PERM_NOT_ALPHA
         H225_FS2_OOF_SCREEN         FS2=OCF/平均资产 通过 PIT 与 OOF 筛选并可进入袖套   PASS          H225_FS2_OOF_SCREEN
         H225_NAV_RECONCILED                       v5 净值桥包含现金、费用、未实现和复利   PASS          H225_NAV_RECONCILED
     H225_C4_ROLLING_ROWWISE                         C4 滚动按行，不把日历空档插进窗口   PASS      H225_C4_ROLLING_ROWWISE
```

CURRENT 未改。AUTO_PROMOTE=false。没有独立 Forward，因此 LIVE_READY=false。
