# C4 滚动语义

原公式保持 `mean(sign(NET_FLOW).shift(1).rolling(5, min_periods=3))`。没有改写。
可得性起点 2020-09-24。
按行 rolling 不把周末或缺失交易日插进窗口。缺行时窗口吃到更早的已有行，日历跨度可以长于 5 个交易日。

```
 predict_year  train_rows  test_rows  r22_2_c1_xl_net_imbalance_lag1_train_finite  r22_2_c1_xl_net_imbalance_lag1_test_finite  r22_c4_flow_persist_5_train_finite  r22_c4_flow_persist_5_test_finite  r22_c1_xl_net_imbalance_train_finite  r22_c1_xl_net_imbalance_test_finite
         2018       12596       3241                                     0.000000                                    0.000000                            0.000000                           0.000000                              0.000000                             0.000000
         2019        9299       4711                                     0.000000                                    0.000000                            0.000000                           0.000000                              0.000000                             0.000000
         2020       10334       5863                                     0.000000                                    0.166127                            0.000000                           0.174484                              0.000000                             0.174484
         2021       13815       5749                                     0.070503                                    0.948513                            0.074050                           1.000000                              0.074050                             1.000000
         2022       16323       5600                                     0.393739                                    0.958393                            0.414875                           1.000000                              0.414875                             1.000000
         2023       17212       4027                                     0.685220                                    0.936926                            0.718801                           0.999752                              0.718801                             0.999752
         2024       15376       6792                                     0.949076                                    0.953033                            0.999935                           0.999853                              0.999935                             0.999853
         2025       16419       6584                                     0.950911                                    0.967345                            0.999878                           1.000000                              0.999878                             1.000000
         2026       17403       3667                                     0.954720                                    0.971094                            0.999885                           1.000000                              0.999885                             1.000000
```

```
n_valid_3                                                                     5116
n_valid_4                                                                     5116
n_valid_5                                                                 11778270
n_valid                                                                   11788502
status                                                                          OK
formula                          mean(sign(NET_FLOW).shift(1).rolling(5, min_pe...
source                           /data1/Stock/data/acquisitions/hermes-cn-a-201...
column                                                                    NET_FLOW
rolling_inserts_calendar_gaps                                                False
rolling_is_row_based                                                          True
calendar_gaps_present                                                         True
n_instruments                                                                 5116
n_flow_rows                                                               11803850
rewritten                                                                    False
```
