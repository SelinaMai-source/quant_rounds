# Round7 口径重跑 baseline_v4 / baseline_v5

费用=`ROUND7_FLAT_5BP`（卖印花税 5bp 一口价）。**不是** `HISTORICAL_FEE_CORRECTED`。CURRENT 锁未改：`baseline_v5=R16_O01` USER_LOCK。

## 口径

- engine=`local_executor → Stock run_fixed_dual_sleeve_top3_backtest`
- config=`round7_fixed_sleeve_config` Top2 `RANK_NAV=[0.3, 0.1]` 初始现金=`300000.0`
- costs slippage=12.3bp commission=1.0bp transfer=0.1bp sell_tax=5.0bp flat
- CAGR `(final_asset / initial_cash) ** (1 / years) - 1; years=(last-first).days/365.25`
- B03 只在 v4/v5 overlay，不在 Round7 入口本身

## v4 = R9_B03

- CAGR(日历年)=`0.5373748002477328`（对照 252 日 cagr_of=`0.5634215542184309`，非正式口径）
- Sharpe=`1.8433564243143739` MaxDD=`-0.24751010811952434`
- completed_trades=`1149` n_trading_observations=`2075` n_nav_rows=`2076`
- NAV_end=`11889633.46375984` total_return=`38.63211154586613`
- vs 旧 R9 5bp flat `MATCH`（1.843 / 53.74% 家族）

## v5 = R16_O01

- formula=`PUT_SKEW25.shift(1) > rolling(252, min_periods=126).quantile(0.90) → apply_top2_veto rank=2 MIN_KEEP` n_signal=`332`
- CAGR(日历年)=`0.5499812832273525`（对照 252 日 cagr_of=`0.5767445498514736`，非正式口径）
- Sharpe=`1.8756627913361097` MaxDD=`-0.24056327715394388`
- completed_trades=`1107` n_trading_observations=`2075` n_nav_rows=`2076`
- NAV_end=`12750089.355724689` total_return=`41.50029785241563`

## NAV

- NAV_end v4=`11889633.46375984` v5=`12750089.355724689` Δ(v5−v4)=`860455.8919648491`

## 分年

- v4 yearly=`{'2018': 0.37363452704771616, '2019': 0.10273711880857106, '2020': 0.3510273494805374, '2021': 0.5882104574912888, '2022': -0.07471527479878248, '2023': 0.507656924566041, '2024': 2.335369107611699, '2025': 1.4498508023732324, '2026': 0.069720625871309}`
- v5 yearly=`{'2018': 0.38463940409582253, '2019': 0.11444672040054482, '2020': 0.35444299783283406, '2021': 0.6198409168432124, '2022': -0.06551933499938356, '2023': 0.5091216072612235, '2024': 2.355684557967081, '2025': 1.4706428306695094, '2026': 0.07368584258455546}`

产物目录：`/data1/stock-mxy/课题/一进二策略/result/replay_r7_caliber`
