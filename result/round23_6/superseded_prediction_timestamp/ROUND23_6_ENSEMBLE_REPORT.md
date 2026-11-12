# Round23.6 ensemble research — PARTIAL

The full requested research is not complete. Frozen protocol was committed and pushed before experiments. Source anchors and canonical inode were verified. Formal baseline_v5, CURRENT and production settings were not changed. Round24 was not executed.

## Actually executed

Original baseline replay matches candidate scores/ranks, original executor entry-plan intents/pre-trade capital, fills, holdings, costs, rejections and daily NAV. The P1_SCORE_A099 annual score-fusion replay also matches the old score/allocation output. Shared-account portfolio combination C02-C06 has no economic backtest: monthly optimization outputs are a mathematical weight study only. Fixed-PDF equation mapping, hash and 60+60-month FLEXPOOL warmup are documented in equation_to_code.md. The four expert ledgers available from 2018 do not activate FLEXPOOL alpha selection by July 2026; pre-2018 expert-ledger recovery is unfinished.

Super Learner NNLS and BOA default square-gradient updates match locked official R within 8e-16 on the fixed example. A synthetic official SuperLearner example is retained, not called full original-paper L1 reproduction. L2 trained original XGB/LGBM/Ridge/mean banks, historical OOF score calibration, per-head NNLS and delayed date-batched BOA. Prediction-time vectors, weights and maturity timestamps are retained. All single-expert/equal controls and S0/S1/S2 strategies ran; S3 verifies actual original plan identities and already-enforced opening limit rejection. No failed candidate was deleted or tuned after inspecting Outer outcomes.

| experiment_id | CAGR | Sharpe | MaxDD | UI | CompletedTrades |
| --- | --- | --- | --- | --- | --- |
| C00_V5 | 0.549981 | 1.875663 | -0.240563 | 0.092394 | 1107 |
| C01_P1_SCORE_A099 | 0.553927 | 1.917863 | -0.244437 | 0.090463 | 1114 |
| C10_SL | 0.005274 | 0.127089 | -0.672186 | 0.452670 | 1262 |
| C11_BOA | 0.033875 | 0.271635 | -0.569445 | 0.352198 | 1251 |

All metrics above use the same 2018-01-02..2026-07-24 window and 300000 initial capital. The baseline CAGR is 54.9981%, not an independent forward result. P1 score replay has delta CAGR only 0.3946 percentage points and delta Sharpe 0.0422, below the frozen 1-point/0.05 economic thresholds. Super Learner and BOA are materially worse. Full controls, yearly metrics, actual RMB PnL, costs, open-position residual bridge, risk, paired fee stresses, paired circular bootstrap and same-scope SPA are retained. Bootstrap uses 5000 paired draws at block20 and 1000 at blocks10/40 with the same calendar-year CAGR convention. SPA deduplicates numerically identical daily paths; the original byte-only dedup result is retained as an audit correction, not selected for a favorable p-value.

Actual execution counts: `{"base_bank_model_files": 168, "base_bank_unique_fit_configs": 14, "base_bank_head_year_fits": 24, "calibrator_fits": 44, "NNLS_head_year_fits": 18, "BOA_head_year_replays": 18, "round23_6_formal_backtests": 27, "round23_6_extra_baseline_trace_backtests": 2, "round26_baseline_backtests": 1, "round26_new_candidate_backtests": 0, "full_L1_paper_replications": 0, "new_neural_L2_fold_seed_models": 0, "round26_Ashare_factor_packages": 0, "diagnostic_future_mutation_model_fits": 2}`.

## Unfinished and limitations

TRA and DoubleAdapt each performed actual official finite model updates (including OT/router and both data/model adapters), but complete author-reference L1 and three-seed L2/controls were not run. The full author TRA feature/label/return pickle set (2,512,790 rows, 16 original features) was successfully downloaded and read; missing-data is not asserted as an excuse for its unexecuted training. P1 shared-cash netting/exits/one-expert execution parity are unimplemented. A real Five-head/2018/seed17 future-or-immature target mutation test trained two diagnostic models and compared exact predictions and overlay plan inputs; it passed within that scope. Full pipeline future-mutation/prefix/revision tests and label-maturity repair and strict price availability proofs are incomplete. The NAV identity passes, but the entire unrealized/corporate-action PnL residual has not been decomposed. The source bank records cutoff and maximum training maturity; this is not a blanket no-future-function PASS. `integration_acceptance.json` contains all missing gates.

The baseline remains a LEGACY_V5_COMPARISON. Original high/low-dependent modeled fill clipping is disclosed. New results are research only; no new untouched Forward exists. KEEP_BASELINE_V5; no promotion.

Permissions file: 无权限数据.md is empty (0 bytes) for this line. The announcement-source authentication rejection belongs to Round26, not this aggregation line.

CURRENT_BASELINE_CHANGED=FALSE
PRODUCTION_CONFIG_CHANGED=FALSE
AUTO_PROMOTE=FALSE
ROUND24_EXECUTED=FALSE
