# baseline_v5 分支文件索引

## 必读文档

- `FINAL_BASELINE_V5.md` — 冻结身份与核验
- `BASELINE_V5_SPEC.md` — 策略圣经
- `FACTOR_REGISTRY.md` / `FACTOR_REGISTRY.csv` — 14/13 因子
- `MODEL_REGISTRY.md` — Five/Ten/XGB/融合/B03/O01
- `BACKTEST_CONTRACT.md` — 窗、资金、费用、指标
- `REPRODUCE.md` — 复现命令与阻断
- `external_dependencies.json` — 非完全自包含声明
- `common_code_manifest.json` — 代码/产物哈希

## 代码

- `core/baseline_v5/` — 冻结契约、parity、不变量测试
- `core/round7/` — 双头训练、融合、clean_risk
- `core/round8/` `core/round9/` — overlay / B03
- `core/round15/` `core/round16/` `core/round16_5/` — O01 veto 与信号
- `core/local_executor.py` / `core/research_sleeve.py` — 正式袖套入口
- `tools/run_baseline_v5.py` / `tools/replay_r7_caliber.py`

## 结果

- `result/replay_r7_caliber/v5_R16_O01/` — **权威 ROUND7_FLAT_5BP**
- `result/round22_3/parity/R223_V5_R16_O01/` — 交叉 parity
- `result/round7/` — 模型与融合产物
- `result/round16/candidates/R16_O01/` — 历史 HISTORICAL_FEE 口径（对照，非正式主锚）
- `result/baseline_v5/frozen_identity.json`

## 长文参考（课题 docs）

- `docs/数据与策略详解.md`
- `docs/一进二策略_多seed集成全流程.md`
- `AGENTS.md`

特征顺序哈希：Five14=`be6c7cf6ff7f7f18bb22bd31340c4a6ca7aebc6dbf0d77c106872e1027128e27`；Ten13=`5627ba42ed47b38334b34c56d2b946fd5cc8f4e84fd745b7814df23d34382c34`。
