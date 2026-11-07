# baseline_v5 因子公式源码审计

身份：`baseline_v5 = R16_O01`。本文仅记录**源码核实**公式；不猜测近义名。

特征顺序哈希：Five14 / Ten13 与 `common_code_manifest.json` **一致**。

## BASELINE_CONTRACT_MISMATCH

| 项 | 已发布注册表 | 源码核实 |
|---|---|---|
| `d1_open_gap` | `(open_{D+1}/close_D)-1`（FACTOR_REGISTRY.md） | `tools/run_first_board_open_gap46.py`：`d1_open / d1_pre_close - 1` |

**不改代码强行对齐。** 审计以 builder 源码为准；注册表叙述视为过时。


`d1_open_gap` 标记为 **OPEN_INFORMATION**（D+1 开盘才可得）。


## `g_prim_px_vol_corr`

- 隶属：Five,Ten
- 公式状态：`VERIFIED_AS_VENDOR_CACHE_NO_LOCAL_FORMULA`
- 迁移类：`RECOMPUTE_ON_SECOND_BOARD`
- 源路径：`tools/combine_first_board_factor_caches.py (PRIMITIVES) + AmazingData minute factor cache; listed in tools/build_first_board_single_model_assets.py MINUTE_PRIMITIVE_FEATURES`
- 函数：`cache merge of vendor minute primitive; NO in-repo correlation computation`
- 精确公式：供应商分钟因子缓存列 g_prim_px_vol_corr = corr(minute_price, minute_volume)（本仓库无自算代码；docs/数据与策略详解.md §8.2）
- 表/列：AmazingData/minute factor cache: g_prim_px_vol_corr
- 回看：当日分钟序列；对齐：signal_date=D；available_at：`D_close`
- 缺失：NaN; train complete-case; 禁止 fillna(0)；归一：无额外截面归一；预期[-1,1]但供应商越界未在代码截断确认；单位：corr
- 一进二面板覆盖：1.0000 on five panel candidate_panel
- PIT：PIT_SAFE_IF_MINUTE_COMPLETE_AT_D_CLOSE；uses_D+1：False
- 首板特异性：分钟路径基于当日涨停股分钟；二板需同日分钟重算
- 二板变更：RECOMPUTE_ON_SECOND_BOARD: 需二板日分钟价量；首板分钟快照不可复用
- 源 SHA256：`7a5d952a6192920708c9155903524cbaf864b94cfdd89dcd46ea4fa1f27e684b`
- 测试参考：tests/test_first_board_single_model.py (feature list presence)

## `fb_close_seal_streak_norm`

- 隶属：Five,Ten
- 公式状态：`SOURCE_VERIFIED`
- 迁移类：`RECOMPUTE_ON_SECOND_BOARD`
- 源路径：`src/first_board_hermes/single_model.py`
- 函数：`compute_minute_extension_factors`
- 精确公式：trailing = 从最后一根分钟向前连续 (close>=limit_up-1e-4) 根数; value = trailing / n_bars
- 表/列：AmazingData minute bars: open/high/low/close/amount; selection.limit_up
- 回看：当日全部分钟；对齐：D；available_at：`D_close`
- 缺失：若无分钟日则该键缺失；构建要求 selection 完整；归一：除以当日分钟根数 → [0,1]；单位：ratio
- 一进二面板覆盖：1.0000
- PIT：PIT_SAFE_D_CLOSE；uses_D+1：False
- 首板特异性：语义=尾盘连续封板长度；二板日同样可算但连板语境不同
- 二板变更：在二板候选上用同公式重算；勿直接复用首板面板值
- 源 SHA256：`cde225d3c0755cfe5231afcf7913a5d864df305b79f4cf7d41208014f71570ad`
- 测试参考：tests/test_first_board_single_model.py

## `fb_last30_amount_ratio`

- 隶属：Five,Ten
- 公式状态：`SOURCE_VERIFIED`
- 迁移类：`RECOMPUTE_ON_SECOND_BOARD`
- 源路径：`src/first_board_hermes/single_model.py`
- 函数：`compute_minute_extension_factors`
- 精确公式：sum(amount[-30:]) / sum(amount_day) if total_amount>0 else NaN
- 表/列：AmazingData minute amount
- 回看：尾盘最多30根分钟；对齐：D；available_at：`D_close`
- 缺失：total_amount<=0 → NaN；归一：ratio；单位：ratio
- 一进二面板覆盖：1.0000
- PIT：PIT_SAFE_D_CLOSE；uses_D+1：False
- 首板特异性：通用分钟统计
- 二板变更：二板日分钟重算
- 源 SHA256：`cde225d3c0755cfe5231afcf7913a5d864df305b79f4cf7d41208014f71570ad`
- 测试参考：tests/test_first_board_single_model.py

## `h5_last_first_limit_value_change_sym`

- 隶属：Five,Ten
- 公式状态：`SOURCE_VERIFIED`
- 迁移类：`RECOMPUTE_ON_SECOND_BOARD`
- 源路径：`tools/build_first_board_single_model_assets.py + single_model.symmetric_change`
- 函数：`symmetric_change(last_limit_value, first_limit_value)`
- 精确公式：(last-first)/(|last|+|first|); denom==0 → 0; 结果∈[-1,1]
- 表/列：daily_limit_events[up]: first_limit_value, last_limit_value
- 回看：当日涨停事件；对齐：D；available_at：`D_close`
- 缺失：无 up 事件则 left merge → NaN；归一：对称变化已有界；单位：sym_change
- 一进二面板覆盖：0.9984
- PIT：PIT_SAFE_D_CLOSE（事件日收盘后）；uses_D+1：False
- 首板特异性：封单路径；二板涨停事件同样可得
- 二板变更：用二板日 limit_events 重算
- 源 SHA256：`c8abd1c195bd9270f758161ea0fae91f297f24d4ad94e8256d5d99d0ee5ab868`
- 测试参考：tests/test_first_board_single_model.py::symmetric_change

## `h1_consistent_buy_ratio`

- 隶属：Five,Ten
- 公式状态：`VENDOR_COLUMN_MAPPED_INTERNAL_FORMULA_UNVERIFIED`
- 迁移类：`BLOCKED_DATA`
- 源路径：`tools/build_first_board_single_model_assets.py MICRO_RENAME; tools/extract_first_board_microstructure_clickhouse.py`
- 函数：`argMax(t1.PCV, t1.UPDATE_TIME) AS PCV → rename PCV→h1_consistent_buy_ratio`
- 精确公式：供应商 equ_h2l_factor_t1.PCV（本仓库未展开供应商内部公式）；按 UPDATE_TIME 取 argMax
- 表/列：ClickHouse hermes equ_h2l_factor_t1.PCV, UPDATE_TIME
- 回看：当日微观；对齐：D；available_at：`D_close（依赖供应商 UPDATE_TIME；非今日下载即 PIT）`
- 缺失：缺失保持 NaN；formal 覆盖门槛曾要求≥0.98；归一：无；单位：**vendor score（观测可越出 [0,1]，勿当严格 ratio）**
- 一进二面板覆盖：0.9914
- PIT：PIT_CONDITIONAL_ON_UPDATE_TIME; today_download≠PIT_SAFE；uses_D+1：False
- 首板特异性：微观提取绑在首板 selection 键上
- 二板变更：需对二板候选重新抽取 PCV；BLOCKED_DATA 若无二板日 CH 抽取
- 源 SHA256：`88f3a8015fd1a63bf0521bc6e6dc2528f4b2fcf62c352f2ad4a23cadf928e21f`
- 测试参考：UNVERIFIED_local_test_for_PCV_formula

## `h1_last15m_volume_ratio`

- 隶属：Five,Ten
- 公式状态：`VENDOR_COLUMN_MAPPED_INTERNAL_FORMULA_UNVERIFIED`
- 迁移类：`BLOCKED_DATA`
- 源路径：`同 h1_consistent_buy_ratio`
- 函数：`argMax(t1.BCVP_15MIN, UPDATE_TIME) → h1_last15m_volume_ratio`
- 精确公式：供应商 equ_h2l_factor_t1.BCVP_15MIN（内部公式未在本仓库展开）
- 表/列：ClickHouse equ_h2l_factor_t1.BCVP_15MIN
- 回看：尾盘15分钟（供应商定义）；对齐：D；available_at：`D_close / UPDATE_TIME`
- 缺失：NaN complete-case；归一：无；单位：**vendor score（观测可负/可>1，勿当严格 volume ratio）**
- 一进二面板覆盖：0.9913
- PIT：PIT_CONDITIONAL_ON_UPDATE_TIME；uses_D+1：False
- 首板特异性：微观抽取绑首板 selection
- 二板变更：二板候选需重抽；缺数则 BLOCKED_DATA
- 源 SHA256：`88f3a8015fd1a63bf0521bc6e6dc2528f4b2fcf62c352f2ad4a23cadf928e21f`
- 测试参考：UNVERIFIED_vendor_formula

## `h4_large_net_to_turnover`

- 隶属：Five,Ten
- 公式状态：`SOURCE_VERIFIED`
- 迁移类：`RECOMPUTE_ON_SECOND_BOARD`
- 源路径：`src/first_board_hermes/native_factors.py / dataset.py`
- 函数：`_safe_ratio(net_inflow_l, turnover_value)`
- 精确公式：net_inflow_l / turnover_value；inf→NaN
- 表/列：daily_money_flow: net_inflow_l, turnover_value
- 回看：当日；对齐：D；available_at：`D_close（资金流日频；早期年覆盖不足）`
- 缺失：缺表/零分母→NaN；禁止填0；归一：ratio；单位：ratio
- 一进二面板覆盖：0.9658
- PIT：PIT_SAFE_D_CLOSE_IF_TABLE_ASOF；uses_D+1：False
- 首板特异性：通用资金流比
- 二板变更：对二板日 instrument 重算；需确认 money_flow 覆盖
- 源 SHA256：`3e74e234244d50e8b28fd99166abcd98a5e6c4f8ba703b2d60907126b888f417`
- 测试参考：tests/test_first_board_native_factors.py

## `h4_retail_net_to_turnover`

- 隶属：Five,Ten
- 公式状态：`SOURCE_VERIFIED`
- 迁移类：`RECOMPUTE_ON_SECOND_BOARD`
- 源路径：`src/first_board_hermes/native_factors.py`
- 函数：`_safe_ratio(retail_net_inflow, turnover_value)`
- 精确公式：retail_net_inflow / turnover_value
- 表/列：daily_money_flow: retail_net_inflow, turnover_value
- 回看：当日；对齐：D；available_at：`D_close`
- 缺失：NaN；归一：ratio；单位：ratio
- 一进二面板覆盖：0.9658
- PIT：PIT_SAFE_D_CLOSE_IF_TABLE_ASOF；uses_D+1：False
- 首板特异性：通用
- 二板变更：二板日重算
- 源 SHA256：`3e74e234244d50e8b28fd99166abcd98a5e6c4f8ba703b2d60907126b888f417`
- 测试参考：tests/test_first_board_native_factors.py

## `g0_amount_log`

- 隶属：Five_only
- 公式状态：`SOURCE_VERIFIED`
- 迁移类：`SEMANTIC_REDEFINITION`
- 源路径：`src/first_board_hermes/dataset.py`
- 函数：`_compute_daily_features`
- 精确公式：np.log1p(amount.where(amount>=0))；amount<0→NaN
- 表/列：daily_raw.amount
- 回看：当日；对齐：D；available_at：`D_close`
- 缺失：NaN；禁止 fillna(0)；归一：log1p；单位：log_CNY
- 一进二面板覆盖：1.0000
- PIT：PIT_SAFE_D_CLOSE；uses_D+1：False
- 首板特异性：通用规模；Round24 已有 r24_amount_log 适配名
- 二板变更：可用同公式；建议新名避免语义混淆（SEMANTIC_REDEFINITION→r24_amount_log）
- 源 SHA256：`1903f43e9358bf5f6122b4ff2baddfc36c2e9370a8dc6229c97371e30bbbb3d0`
- 测试参考：tests/test_first_board_hermes_dataset.py

## `g0_amt_vs_5`

- 隶属：Five,Ten
- 公式状态：`SOURCE_VERIFIED`
- 迁移类：`RECOMPUTE_ON_SECOND_BOARD`
- 源路径：`src/first_board_hermes/dataset.py`
- 函数：`_compute_daily_features`
- 精确公式：amount_t / mean(amount_{t-1..t-5}); amount_prev=shift(1); rolling mean min_periods=5; _safe_ratio
- 表/列：daily_raw.amount
- 回看：前5个该股交易日均额；对齐：D；available_at：`D_close`
- 缺失：不足5日或分母无效→NaN；归一：ratio；单位：ratio
- 一进二面板覆盖：1.0000
- PIT：PIT_SAFE_D_CLOSE；uses_D+1：False
- 首板特异性：通用；二板日可重算（Round24 有 r24_amt_vs_5）
- 二板变更：RECOMPUTE；或 SEMANTIC_REDEFINITION 为 second_board 命名
- 源 SHA256：`1903f43e9358bf5f6122b4ff2baddfc36c2e9370a8dc6229c97371e30bbbb3d0`
- 测试参考：tests/test_first_board_hermes_dataset.py

## `g1_ret_5`

- 隶属：Five,Ten
- 公式状态：`SOURCE_VERIFIED`
- 迁移类：`RECOMPUTE_ON_SECOND_BOARD`
- 源路径：`src/first_board_hermes/dataset.py`
- 函数：`_compute_daily_features`
- 精确公式：close_post_t / close_post_{t-5} - 1（后复权；按 instrument shift 5）
- 表/列：daily_post_adjusted.close_post
- 回看：5个该股交易日；对齐：D；available_at：`D_close`
- 缺失：历史不足→NaN；归一：return；单位：return
- 一进二面板覆盖：1.0000
- PIT：PIT_SAFE_D_CLOSE；uses_D+1：False
- 首板特异性：通用动量
- 二板变更：二板日重算（Round24: r24_ret_5）
- 源 SHA256：`1903f43e9358bf5f6122b4ff2baddfc36c2e9370a8dc6229c97371e30bbbb3d0`
- 测试参考：tests/test_first_board_hermes_dataset.py

## `g3_days_since_limit`

- 隶属：Five,Ten
- 公式状态：`SOURCE_VERIFIED`
- 迁移类：`SEMANTIC_REDEFINITION`
- 源路径：`src/first_board_hermes/dataset.py`
- 函数：`_compute_daily_features`
- 精确公式：row_number - last_limit_row(shift1 ffill of close-limit rows); 无历史→-1；is_close_limit=tradable & close>=limit_up-1e-4；v6 曾对旧0-based +1 repair
- 表/列：daily_raw close/limit_up/paused/volume
- 回看：该股历史收盘涨停；对齐：D；available_at：`D_close`
- 缺失：结构填充 -1（非 NaN）；归一：days_or_-1；单位：days
- 一进二面板覆盖：1.0000
- PIT：PIT_SAFE_D_CLOSE；uses_D+1：False
- 首板特异性：硬首板已要求近20日无涨停，此特征主要刻画>20日远近；在二板截面上常退化/语义变
- 二板变更：SEMANTIC_REDEFINITION：二板日‘距上次涨停’几乎总是1（昨首板），信息量变
- 源 SHA256：`1903f43e9358bf5f6122b4ff2baddfc36c2e9370a8dc6229c97371e30bbbb3d0`
- 测试参考：tools/repair_first_board_days_since_limit.py; tests dataset

## `ind_seal_rate_excess`

- 隶属：Five,Ten
- 公式状态：`SOURCE_VERIFIED`
- 迁移类：`RECOMPUTE_ON_SECOND_BOARD`
- 源路径：`tools/build_first_board_single_model_assets.py`
- 函数：`compute_industry_features`
- 精确公式：industry_seal_rate(trade_date, industry_id) - market_seal_rate(trade_date)；seal= up limit_events 是否落在收盘涨停 selection 键；industry=PIT map_pit_industry
- 表/列：daily_limit_events; selection close-limits; industry_membership
- 回看：当日截面；对齐：D；available_at：`D_close`
- 缺失：无行业映射→NaN；归一：rate_diff；单位：rate_diff
- 一进二面板覆盖：0.9995
- PIT：PIT_SAFE_IF_INDUSTRY_MEMBERSHIP_PIT；uses_D+1：False
- 首板特异性：构建时 first 子集用 is_first_board 过滤后挂到首板行；市场/行业封住率来自全市场 up 事件
- 二板变更：需在二板宇宙重算行业/市场封住率；不可直接搬首板值；行业 PIT 需复核
- 源 SHA256：`c8abd1c195bd9270f758161ea0fae91f297f24d4ad94e8256d5d99d0ee5ab868`
- 测试参考：docs/数据与策略详解.md §8.43

## `d1_open_gap`

- 隶属：Five,Ten
- 公式状态：`SOURCE_VERIFIED_WITH_REGISTRY_MISMATCH`
- 迁移类：`BLOCKED_PIT`
- 源路径：`tools/run_first_board_open_gap46.py`
- 函数：`build_panels`
- 精确公式：SOURCE_VERIFIED: where valid: d1_open/d1_pre_close - 1; else NaN. valid= ~paused & volume>0 & open>0 & pre_close>0. NOTE: FACTOR_REGISTRY.md 写 (open/close_D)-1 → BASELINE_CONTRACT_MISMATCH
- 表/列：daily_raw on next1_date: open, pre_close, volume, paused
- 回看：D+1 开盘；对齐：signal D, value asof D+1；available_at：`D+1_09:25 — OPEN_INFORMATION / LEGACY_OPEN_INFORMATION_LANE`
- 缺失：非有效→NaN；融合层再要求 finite & >= GAP_FLOOR(-0.01)；归一：return；融合预滤；单位：return
- 一进二面板覆盖：0.9943
- PIT：NOT_PIT_SAFE_AT_D_CLOSE; OPEN_INFORMATION；uses_D+1：True
- 首板特异性：执行时信息进模型
- 二板变更：BLOCKED_PIT：不可作为 D 收盘决策特征；若仅执行过滤则 EXECUTION_ONLY
- 源 SHA256：`351b7625033eb8299b0444e89a4b6cca17431581816f98f5b0464406ea15fe32`
- 测试参考：baseline.py EXECUTION_TIME_FEATURES; core/research_integrity_v2/time_contract.py

## 特别复核

### `g3_days_since_limit`

源码：`dataset._compute_daily_features` 行距差，无历史填 -1。硬首板近 20 日无涨停约束下信息偏远窗；二板截面几乎退化为常数量级 → `SEMANTIC_REDEFINITION`。

### `d1_open_gap`

OPEN_INFORMATION。源码用 D+1 `open/pre_close-1`，不是 D close 比。融合另加 `>= -0.01`。不可声称实盘开盘可执行性已验证。

### `ind_seal_rate_excess`

`compute_industry_features`：行业收盘封住率 − 市场收盘封住率；挂在首板子集行上。二板需重算。

### 分钟因子

`fb_*` 来自 `compute_minute_extension_factors`；`g_prim_px_vol_corr` 仅有供应商缓存列，无本地相关公式。

### 资金流因子

`h4_*` = money_flow 净流入 / turnover_value；2014 前覆盖不足见面板 coverage≈0.966。

