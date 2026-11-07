# Alpha Factor Research Standard（Round21 起）

课题内可复用的因子研究契约。正式账户数字仍走 Stock Round7 5bp，不由本标准替换。

## 锁定

- CURRENT = `baseline_v5 = R16_O01`。不自动升 CURRENT。过门只写 `RECOMMEND_NEW_ALPHA_CANDIDATE` + 等用户锁。默认 `KEEP_BASELINE_V5`。
- `subsample = colsample = 1.0`。禁止改 O01 / B03 / 90-10 / clean_risk / 退出 / 仓位 / gap。
- 看任何新因子 NAV 之前冻结：假设卡（代码/yaml）+ `round21_manifest.yaml` + sha256。

## Factory API

`FactorDefinition → DataAudit → PIT → IC → Rolling → Shape → TopTail → Redundancy → ConditionalIC → Combination → ModelIncrementality → PortfolioEvidence → PromotionGate`

入口：

```text
python tools/evaluate_factor.py --list-steps
python tools/evaluate_factor.py --factor-id <id> --through PromotionGate
python tools/evaluate_factor.py --factor-id <id> --through PromotionGate --persist-gate
python tools/run_round21.py --stage zoo,report
```

`--through` 可停在任一步。默认 `PromotionGate`。缺下游产物必须写 `SKIPPED` + `reason`，**禁止伪造 PASS**，禁止把全局 combo/NAV 复制进未过门的单因子目录。

`GITHUB_EXACT_*` 与 `ROUND21_INTERNAL_W60/W120/W252` 必须分名。源码无 Score 则不伪造。无源码不得写 `EXACT_GITHUB_REPRODUCTION_PASS`。

## 假设卡

每个 `factor_id` 独立。公式必须在代码或 yaml，不只写 markdown。缺失不填 0。数据不足记 `BLOCKED_DATA` / `BLOCKED_PIT`。覆盖不足记 `PIT_FAIL_COVERAGE`，下游 `SKIPPED`。

近邻必须预登记，不得换名：

- A4 vs `fb_close_seal_streak_norm`
- B1 vs `r17_d_amt_xs_rank`
- C3 vs `h_min1_path_eff_full`
- A1 ≠ R18 F1（F1 = `V_seal / (A_below + 0.01*A_day)`）

## 证据

- 双标签：`label_e2_net` 与 `label_exit_rule_net`（maturity-aware，按 `actual_exit_date`）。
- 内部滚动窗 **60/120/252**，命名 `ROUND21_INTERNAL_W*`，不事后挑窗。
- GitHub `rolling-factor-evidence` 只覆盖边际证据。窗口/分数必须分名 `GITHUB_EXACT_*`。
- Spearman≥0.85 为簇边界，非唯一 DROP。簇内必须 Original / New / Add / Replace。
- 内部 Evidence 是向量/分层硬过滤 + Pareto，不优化人为加权总分。

## 组合与模型

- 入主组合新因子 ≤12。Pair≤66、Triple≤36、ortho/PCA/非线性≤24。超预算开下一批次。
- 同时报 `FIXED_FACTOR_SET_REPLAY` 与 `NESTED_WALK_FORWARD_FEATURE_SELECTION`；二者都不是独立 OOS。
- Seed：仅当预注册预测增量门槛过了才跑；native 5 退化则 group-date；不从 100/300 里挑最好 seed。
- 训练：`train_head_open` + 3y→1y；转换只在训练折拟合。

## 晋级（PromotionGate）

顺序：PIT+质量 → IC/滚动/条件 → Top-tail → 替换/互补 → OOF → 同口径净收益优于 v5 → 稳健性。

| 结果 | 何时写 |
|---|---|
| `SKIPPED` | 缺产物 / `BLOCKED_PIT` / `PIT_FAIL_COVERAGE` / 无正式组合行 |
| `FAIL` | 有正式回测但未优于 v5，或其它硬门未过 |
| `RECOMMEND_NEW_ALPHA_CANDIDATE` | 全部硬门过且同口径净收益优于 v5；仍等用户锁 |
| `PASS` | **不用于未过门因子**。禁止用全局结果冒充 |

过门只允许推荐 + `AWAITING_USER_REVIEW`。默认 `KEEP_BASELINE_V5`。`AUTO_PROMOTE=False`。不锁 v6。

## Factor Zoo

永久目录：`research/factor_zoo/`

必须落盘：`registry.parquet/csv`、`families.csv`、`lineage.parquet/csv`、`similarity.csv`、`evidence.parquet/csv`、`rejected_archive.csv`、`trial_registry.csv`、`registry.yaml`。

数字必须来自 `result/round*/` 已有产物，禁止手填 IC/NAV。

## 禁止

发明缺失表、全市场分钟均值、非两融融资填 0、把 R17/R18/R20 失败因子换名当新发现、追踪表写「写入时间」、改 CURRENT。

## Round 21.1 强制增量（替换法医）

每个新因子除 PIT / IC / Rolling / Top-tail / Conditional IC 外，正式晋级还必须完成：

| 步 | 含义 |
|---|---|
| `NEW_VS_BASELINE_CORRELATION` | 首板候选截面 vs **真实** 14/13（`core/round7/contract.py`），禁止用 KEEP-45 / ZyQuant 350 顶替 |
| `NEAREST_LEGACY_FACTOR_IDENTIFICATION` | `j* = argmax |Spearman|`，**TRAIN_FOLD** 官方；FULL_HISTORY 仅诊断 |
| `FOUR_WAY_REPLACEMENT_TEST` | E0 Control / E1 Remove / E2 Add / E3 Replace，必须真实重训，禁止 alias ADD |
| `CLUSTER_REPLACEMENT_IF_APPLICABLE` | 仅当训练窗 `|ρ|≥0.85` 且簇 ≥2 时做 E4/E5 |
| `OOF_INCREMENTALITY` | 四路 OOF NDCG / Rank IC |
| `FULL_PORTFOLIO_INCREMENTALITY` | 同口径 Round7 5bp 全引擎 |

未过 PIT 只出诊断，不得填 0 拉覆盖。E 族无公告时点仍 `BLOCKED_PIT`。

覆盖法医分类仅允许：`STRUCTURAL_NOT_APPLICABLE` / `EVENT_CONDITIONAL` / `TRUE_DATA_MISSING` / `SOURCE_ALIGNMENT_FAILURE` / `FORMULA_UNDEFINED` / `TIMESTAMP_UNVERIFIED`。必须同时报全池 Coverage、适用子样本 Conditional Coverage、真实缺失率。

训练缓存 key 必须包含 dropped + added + fold + head + `feature_set_hash`。`train_both_heads` 支持 `five_drops` / `ten_drops`，锁定列可以在实验层删除，但不得改 `FIVE_FEATURES` / `TEN_FEATURES` 常量。

Zoo 增补：`nearest_legacy_factor`、`correlation`、`correlation_stability`、`removed_factor`、`replace/add/remove_result`、`incremental_value_after_removal`、`replacement_verdict`。

## Round 21.2 强制增量（第二波 Alpha + 替换法医修补）

在 Round 21.1 替换法医之上，本轮额外强制：

| 步 | 含义 |
|---|---|
| `FORMULA_EQUIVALENCE_CHECK` | 新公式不得是历史失败因子换名；必须写 `round21_2_prior_factor_lineage.csv` |
| `REPLACE_HASH_INTEGRITY` | E0/E1/E2/E3 的 `feature_set_hash` / pred hash 必须互异；禁止 alias ADD=Replace |
| 显式 ρ 字段 | `full_sample_signed_rho` / `full_sample_abs_rho` / `fold_signed_rho` / `fold_abs_rho` / `mean_daily_rho` / `median_daily_rho` / `mean_absolute_daily_rho` |
| Coverage masks | `applicability_mask` / `missingness_indicator` / `strict_pit_mask` / full-universe XGB 原生 NA+flag；全池 80% 不是唯一门槛 |
| NO_OP 分列 | 近恒定稀有 flag（如 `r21_a1_missing_flag`）不得与有增量候选并列 |

G–N 32 个新公式必须从原始字段计算。缺失不填 0。`BLOCKED_DATA` 不得伪造。正式晋级只接受 `STRICT_PIT`。money_flow 2016–2020 回填年不得当 STRICT。决策只允许 `KEEP_BASELINE_V5` 或 `RECOMMEND_ALPHA_CANDIDATE`。`AUTO_PROMOTE=False`。`CURRENT_BASELINE_CHANGED=FALSE`。

## Round 21.3 强制增量（L1 法医 + T/U/V + 经济意义门）

在 Round 21.2 之上，本轮额外强制：

| 步 | 含义 |
|---|---|
| `DATA_RECOVERY_BEFORE_COVERAGE_REJECT` | 覆盖不足必须走 DISCOVER→RECOVER→VALIDATE→RECOMPUTE，禁止直接放弃 |
| `FORMULA_EQUIVALENCE_CHECK` | 代数≡G/H/K/L 记 `DUPLICATE`，不新开 ID |
| `TRUE_DELETE_ASSERTION` | E3 = E0 − {dropped} + {new}；`g0_amount_log` 仅 Five-only |
| `FEE_LEDGER_COMPLETENESS` | `ROUND7_FLAT_5BP` 与 `HISTORICAL_FEE_CORRECTED` 必须有真实 Sharpe；`ran`+`None` 视为未完成 |
| `BOOTSTRAP_SPA_NUMERICAL` | Bootstrap N=5000 block=20；SPA 必须输出观测统计量 / 备择数 / 数值 p |
| `SEED_STATUS_ENUM` | 只能是 COMPLETE / NOT_TRIGGERED / BLOCKED / FAILED |
| `ECONOMIC_MATERIALITY_GATE` | ΔCAGR≥+1.0pct 且 ΔSharpe≥+0.05 且 \|ΔNAV\|/NAV≥1% 且 Top1/Top2 变化>0 且 MaxDD 恶化≤2pct 且费用方向不翻转。21.2 +0.003/+43 预先不达标 |
| `FORWARD_AVAILABILITY` | 无独立 Forward 不得 live-ready，最多 `V6_RESEARCH_CANDIDATE` |

决策只允许 `KEEP_BASELINE_V5` / `RECOMMEND_BASELINE_V6_CANDIDATE` / `V6_RESEARCH_CANDIDATE`。`AUTO_PROMOTE=False`。`CURRENT_BASELINE_CHANGED=FALSE`。失败进 Rejected Archive。

## Round 21.4 强制增量（L1 Head-specific × Forward 修复 × 组合工厂 × W/X/Y/Z）

在 Round 21.3 之上，本轮额外强制：

| 步 | 含义 |
|---|---|
| `HEAD_SPECIFIC_FOUR_WAY` | 新因子必须跑 Five-only / Ten-only / Both-add / Both-replace，禁止只报双头均值 |
| `COMBINATION_FAILURE_ANALYSIS` | 单因子 IC 有效但 Add 变差必须按 8 类归因，给证据 |
| `DATA_RECOVERY_BEFORE_COVERAGE_REJECT` | Forward `_max_date` 必须扫全部分区；覆盖不足走 DISCOVER→RECOVER→VALIDATE→INTEGRATE→RETEST |
| `ECONOMIC_MATERIALITY_GATE` | 沿用 21.3，禁止回调。+0.003/+43 预先不达标 |

组合方法在 `research/factor_combination/`。C0–C11 不预设优于 Raw。无独立 Forward 最多 `V6_RESEARCH_CANDIDATE`。`AUTO_PROMOTE=False`。`CURRENT_BASELINE_CHANGED=FALSE`。

## Round 22 强制增量（Profile 分离 × Head-specific E0–E9 × 新公式）

在 Round 21.4 之上，本轮额外强制：

| 步 | 含义 |
|---|---|
| `PROFILE_SEPARATION` | Profile A 用 D 收盘可得信息。Profile B 用收盘后到次日开盘前。龙虎榜不得进入 Profile A |
| `HEAD_SPECIFIC_E0_E9` | E0 对照；E1/E2/E3 为 Five/Ten/Both Add；E4–E9 只在该头训练折 \|ρ\|≥0.60 时 Remove/Replace。低相关删除是 EXPLORATORY_DIAGNOSTIC |
| `FORMULA_EQUIVALENCE_CHECK` | 与 zoo、G/H/T/Y/L/V、封板持续或常数表达式相同则 `DUPLICATE_EXPRESSION` |
| `ECONOMIC_MATERIALITY_GATE` | 沿用 R21_3_ECONOMIC_MATERIALITY_V1，禁止回调 |
| `FORWARD_AVAILABILITY` | 无独立 Forward 不得 live-ready |

决策只允许 `KEEP_BASELINE_V5` 或 `RECOMMEND_V6_RESEARCH_CANDIDATE`。组合因子可测增量，但不能单独当成新的原始信息去升级。`AUTO_PROMOTE=False`。`CURRENT_BASELINE_CHANGED=FALSE`。

## Round 22.1 强制增量（IC 宇宙 × 强制证据 × Top1/Top2 账本 × 公式身份）

在 Round 22 之上，本轮额外强制：

| 步 | 含义 |
|---|---|
| `NO_SILENT_SKIP` | mandatory 阶段 `Rolling` / `Shape` / `TopTail` / `ConditionalIC` 禁止 `SKIP_IF_MISSING`。缺文件先跑，失败留日志 |
| `IC_UNIVERSE_AUDIT` | IC 必须写 universe / n_pairs / label。主宇宙是一进二 `candidate_panel`，不是全市场。日 n<5 跳过；累计有效日<60 才 `NOT_IDENTIFIABLE`。ICIR=mean/std(ddof=1)，未年化 |
| `MANDATORY_ROLLING_CONDITIONAL_SHAPE` | 11 个 PIT_PASS 必须有 rolling/conditional/shape/top_tail。缺任一 JSON 不得 `FULL_EVIDENCE_COMPLETE` |
| `TOP1_TOP2_LEDGER_TRACE` | 回测必须可追踪 Top1/Top2。`top_net_ok=False` 若因缺列不得解释为已算不合格 |
| `DATA_RECOVERY_BEFORE_COVERAGE_REJECT` | 覆盖拒绝前先恢复。C 必须全表分档核对。无凭证不猜密码。`download_attempts=0` 不得 `RECOVERY_COMPLETE` |
| `FORMULA_IDENTITY_PROVENANCE` | Original Prompt vs Frozen vs 实际代码。只标 `PRE-FREEZE_REVISION` / `POST-FREEZE_DEVIATION`，不借机改公式 |
| `ECONOMIC_MATERIALITY_GATE` | 沿用 R21_3_ECONOMIC_MATERIALITY_V1，禁止回调 |
| `FORWARD_AVAILABILITY` | 无独立 Forward 不得 live-ready。未完成不得 `FULL_RESEARCH_COMPLETE` |

决策只允许 `KEEP_BASELINE_V5` 或 `RECOMMEND_V6_RESEARCH_CANDIDATE`。`AUTO_PROMOTE=False`。`CURRENT_BASELINE_CHANGED=FALSE`。

## Round 22.2 强制增量（严格 PIT 时间线 × Matched 重训 × 去重新机制）

在 Round 22.1 之上，本轮额外强制：

| 步 | 含义 |
|---|---|
| `STRICT_PIT_TIMELINE` | C1/C4 必须按 `UPDATE_TIME`/`available_at` vs `D_close` 逐年审计。覆盖≥80% + 分档恒等 ≠ 全历史严格 PIT |
| `MATCHED_SAMPLE_RETRAIN` | 非全历史则建同日同池 matched v5 并重训。禁止截断全历史 NAV，禁止短窗 Sharpe 与 2018–2026 v5 混表。训练不足 3 年标 `INSUFFICIENT_HISTORY` |
| `NO_ALIAS_ADD` | 真删走 `five_drops`/`ten_drops`。Replace 仅当训练折 \|ρ\|≥0.60 |
| `DUPLICATE_LINEAGE_CHECK` | HN1/BT1/MG1 与 Round21 等价则 `DUPLICATE`，不当新 Alpha。必须滞后则新版本号 |
| `TOP1_TOP2_LEDGER_TRACE` | 净收益 = sell_proceeds+cash_dividend−buy_cost。缺列不得写成已算不合格 |
| `ECONOMIC_MATERIALITY_GATE` | 沿用 R21_3_ECONOMIC_MATERIALITY_V1，禁止回调 |
| `FORWARD_AVAILABILITY` | 无独立 Forward 不得 live-ready。短窗最多 `SUBPERIOD_ALPHA_RESEARCH_CANDIDATE` |

决策默认 `KEEP_BASELINE_V5`。禁止 `LIVE_READY_V6`。`AUTO_PROMOTE=False`。`CURRENT_BASELINE_CHANGED=FALSE`。

## Round 22.3 强制增量（两本账 × 全历史重跑 × canonical 门槛哈希）

在 Round 22.2 之上，本轮额外强制：

| 步 | 含义 |
|---|---|
| `TWO_LEDGER_ISOLATION` | 主账 2018–2026 n_obs≈2075 对照 R16_O01；22.2 短窗 2023–2026 n_obs=860 对照 R222_E0。禁止混表。短窗行标 `MATCHED_SUBPERIOD_NOT_COMPARABLE` |
| `CANONICAL_THRESHOLD_HASH` | 经济门对阈值对象做 canonical JSON。body.round 导致的 SHA 差标 `SEMANTICALLY_IDENTICAL_HASH_MISMATCH`，门槛漂移才 `ECONOMIC_GATE_SEMANTIC_DRIFT` |
| `FULL_HISTORY_REPLAY` | 22.2 全部正式实验组按规范 3y/1y 从 2018 起重训+全袖套。禁止把 2023+ 预测贴到 2018 NAV |
| `NAN_BEFORE_PIT_KEEP_POOL` | C1/C4 在 2020-09-24 前保持 NaN，不填 0，不删股票。XGB 按缺失处理 |
| `C4_DUAL_LEDGER_ATTRIBUTION` | 短窗 E1 ΔSharpe=+0.025 只作附录。主裁决用全历史 vs v5。\|ρ\| vs g1_ret_5≈0.576 不准降 Replace 门 |
| `HN2_IDENTITY_DECOMPOSITION` | 锁 22.2 公式并检验是否 Δlog(circ)−Δlog(holders)。H1–H3 全历史真训 |
| `FS_REPORT_ID_STATEMENT_ALIGN` | facts→reports 用 report_id；利润与现金流按 instrument+period+statement+available_at+revision。不同 statement 的 report_id 不得当同一 ID |

决策默认 `KEEP_BASELINE_V5`。禁止 `LIVE_READY_V6`。`AUTO_PROMOTE=False`。`CURRENT_BASELINE_CHANGED=FALSE`。无独立 Forward 不得 live-ready。
