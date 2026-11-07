# baseline_v5 策略圣经（R16_O01）

## 1. 身份

- **baseline_v5 = R16_O01**
- 父基线结构：**R9_B03**（双头 90/10 + clean_risk + B03 allocator）
- 增量规则：**O01** Top2 veto（Top1 不变）
- 信息车道：`LEGACY_OPEN_INFORMATION_LANE`（特征含 `d1_open_gap`）
- **不**自动改 CURRENT / 生产配置（`CURRENT_BASELINE_CHANGED=FALSE`）

## 2. 股票池 / 候选定义（一进二）

- 宇宙：A 股非 ST（PIT `known_at`）硬**首板**候选
- 硬首板：当日收盘涨停，且近窗（通常 20 交易日）无涨停；排除一字板等过滤（见 `docs/数据与策略详解.md`）
- 涨跌停制度：主板 10%、创业板/科创历史变更、ST 规则等见 `core/limit_rules.py` 与 Hermes 涨停价
- 日历：交易日日历与 Hermes v7 对齐；评价窗截止 `2026-07-24`

## 3. 决策时点 vs 执行时点

| 阶段 | 时点 |
|---|---|
| 信号/决策 | D 日收盘后（除 `d1_open_gap`） |
| 执行买入 | D+1 开盘 |
| 计划卖出 | D+2 起第一个非涨停收盘 |

`d1_open_gap` 仅在 D+1 开盘可知 → 标记 **LEGACY_OPEN_INFORMATION_LANE**：历史契约可比较，**不能**声称真实开盘可执行性已验证。

## 4. 特征

见 `FACTOR_REGISTRY.md`。Five14 / Ten13 顺序哈希已冻结于 `common_code_manifest.json`。

## 5. 标签

| 标签 | 头 | 成熟规则 |
|---|---|---|
| `label_e2_net` | Five | 以 `next2_date` 为 label_end；净值口径含费用分项 |
| `label_exit_rule_net` | Ten | 以 `actual_exit_date` 为 label_end；对齐退出规则路径净收益 |

训练必须等标签成熟；禁止用未成熟标签训练。

## 6. 模型与融合

见 `MODEL_REGISTRY.md`。融合权重 **90% Ten + 10% Five**（先各头日内百分位，再加权）。

## 7. 组合、风控、退出

- clean_risk 缩放 → B03 九格 Top2 份额 → O01 信号日将 rank=2 置 `MIN_KEEP≈0`（Top1 保留）
- T+1：D 信号，D+1 买
- 涨停卖出顺延；一字跌停延迟；开盘一字涨停拒买（容差 0.005）
- 费用：见 `BACKTEST_CONTRACT.md`

## 8. 权威结果

见 `FINAL_BASELINE_V5.md`。源码契约：`core/baseline_v5/contract.py`。
