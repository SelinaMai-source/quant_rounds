# baseline_v5 模型训练契约

## 身份
- baseline_v5 = R16_O01；父结构 R9_B03 + O01
- 源：`core/round7/contract.py`、`core/fold_ranker.py`、`core/baseline_v5/contract.py`

## 双头

| 头 | 特征数 | 目标 | label_end | rank_n_bins | eval_metric |
|---|---:|---|---|---:|---|
| Five | 14 | label_e2_net | next2_date | 5 | ndcg@1 |
| Ten | 13（去 g0_amount_log） | label_exit_rule_net | actual_exit_date | 10 | ndcg@2 |

## XGBRanker（两头相同，仅 seed）
见 `model_parameters.json`。硬约束 subsample=1.0、colsample_bytree=1.0。

## Seeds
(17, 42, 73, 101, 202)。多种子预测先做日内 rank 百分位再平均（`ensemble_rank_pct`）。

## Walk-forward
预测年 Y：训练窗日历年 Y-3..Y-1，且 label_end_column 的年 ≤ train_end；group/qid = trade_date；query size≥2。

## 融合
1. 各头 ensemble 分数  
2. 合并 d1_open_gap，保留 finite 且 ≥ GAP_FLOOR(-0.01)  
3. 当日截面 percentile：pct_ten、pct_five  
4. score = 0.90*pct_ten + 0.10*pct_five  
5. 稳定破并列：instrument_id 升序  

实现：`core/round7/blend.py` / `core/round17/backtest.py::fuse_heads`。

## 之后组合层（权威 v5）
`dual90_10` → **B03**（含 B01a×截面，**不含** clean_risk pred）→ O01 Top2 veto → local_executor。  
发布文档写 clean_risk 进父链 → **BASELINE_CONTRACT_MISMATCH**（见 `b03_o01_contract.md`）。
