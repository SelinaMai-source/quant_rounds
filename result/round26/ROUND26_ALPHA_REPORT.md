# Round26 alpha research — PARTIAL

The requested alpha research is unfinished. The exact round23_5 anchor and independent protocol/worktree were used; no Round23.6 winning weight/model was imported. An independent frozen A0_V5 full-history baseline was actually executed: `{"cagr": 0.5499812832273525, "sharpe": 1.8756627913361097, "max_drawdown": -0.240563277153944, "nav_end": 12750089.355724689, "nav_start": 300000.0, "total_return": 41.50029785241563, "n_obs": 2075, "window_start": "2018-01-02", "window_end": "2026-07-24", "code": "A0_V5", "completed_trades": 1107, "status": "ran", "fee_schedule": "ROUND7_FLAT_5BP", "compare_to": "R16_O01", "restarted_capital": false, "scenario": false, "counts_as_alpha": false, "round": "26", "scope": "LEGACY_V5_COMPARISON", "initial_cash": 300000.0, "economically_material": false, "gate_id": "NOT_THE_1TO2_GATE", "delta_cagr_vs_v5_not_used": 0.0, "v5_nav_reference_not_a_benchmark": 12750089.355724689}`.

## What really ran

AlphaGen: official expression evaluator, synergistic alpha-pool optimization and original LSTM PPO policy performed finite synthetic updates. No HARLA/LLM extension was used. This is L0, not a CSI300 L1 or A-share factor result.

AlphaForge: official NetG_DCGAN and NetP components performed finite updates. The core paper mining loss/collector/training loop, static/dynamic combination and complete L1/L2 were not run. train_AFF.py and its real configuration are available; no missing-core-code claim is fabricated.

RSR: original TensorFlow 1.15.5 graph construction for both RankLSTM and relational ranking performed finite forward/backward updates in Python3.7. No different network was substituted. The author 570MB pretrained embedding archive was actually downloaded and its tar contents inspected. Full NASDAQ relation/price data recovery and full reference training remain incomplete; pretraining downloads, source archive attempts and exact API-recovered code are recorded. R0/R1/R2 A-share execution is unimplemented.

MASTER: original full MASTER architecture performed an update, with all 40 synthetic prediction nodes retained and a node-permutation diagnostic. Full original/opensource L1 and M0/M1/M2 A-share controls are unexecuted. Author validation/infer_processor correction and label-only loss masking are recorded as required but have not been validated in a full dataset integration. No complete original business-data reproduction is claimed.

RD-Agent: blocked by absence of an authorized configured LLM endpoint and zero additional paid-service budget. Local direct model-endpoint probes are recorded. No trajectory, proposal or research loop is invented, and no manual five-factor replacement is called RD-Agent.

DeepLOB: the exact author PyTorch notebook network and original loss/optimizer performed a finite update. The official FI-2010 zip was actually requested repeatedly; timeout/partial-download failures are retained, and no partially downloaded archive is treated as a successful benchmark. A real authorized A-share historical sample returned 5046 five-level Level1 book states. It does not supply ten-level LOB, so L2 is BLOCKED_LOB_DATA. No minute-bar pseudo-book or FI-2010 accuracy-to-A-share-profit claim exists.

## Data and factor evidence

All specified local source roots were actually listed and sampled through parquet schemas; the original inventory was found and saved by hash. The ClickHouse announcement extension query for hermes.fdmt_ef_new actually returned authentication failure. The root-private .env was not opened. This single actual data authentication blocker is the only permission list entry. LOB schema gaps and LLM/runtime/network problems are not entered as fake data permissions.

No A-share formula/learned-factor package completed PIT, temporal generation/selection, dedup, head-specific Add/Remove/Replace or full incremental trading gates. Those steps are NOT_RUN/NOT_IMPLEMENTED, not PASS and not all excused by data permission. No all-NaN factor table or fabricated IC/portfolio result is produced. E1 order-structure unit checks, E2 mechanism dedup/path reconstruction, E3 supplied-announcement recovery, full D+1..D+5 minute recovery and D12 quarantine recovery are unfinished. Consequently there are zero new Round26 candidate backtests, zero qualified factor packages and no incremental SPA/IC claim.

Actual counts: `{"base_bank_model_files": 168, "base_bank_unique_file_fit_configs": 14, "base_bank_head_year_fits": 24, "historical_mean_fit_states": 24, "base_bank_fit_entities_including_mean": 192, "calibrator_fits": 44, "NNLS_head_year_fits": 18, "BOA_head_year_replays": 18, "round23_6_formal_backtests": 27, "superseded_BOA_backtests_retained": 3, "round23_6_economic_backtest_runs_including_superseded": 30, "round23_6_extra_baseline_trace_backtests": 2, "round26_baseline_backtests": 1, "round26_new_candidate_backtests": 0, "full_L1_paper_replications": 0, "new_neural_L2_fold_seed_models": 0, "round26_Ashare_factor_packages": 0, "diagnostic_future_mutation_model_fits": 2}`.

## Decision

KEEP_BASELINE_V5. The baseline was not shown to be improved by Round26 because target-market candidates have not reached executable evidence gates. Nine paper frameworks had finite algorithm/component tests across the two lines; zero complete paper-level L1 replications were finished. Core smoke success is not full reproduction.

Permissions: 无权限数据.md contains only the actual fdmt_ef_new source URI. Detailed reasons are in blockers.json/data_recovery_jobs.jsonl, outside that file.

CURRENT_BASELINE_CHANGED=FALSE
PRODUCTION_CONFIG_CHANGED=FALSE
AUTO_PROMOTE=FALSE
ROUND24_EXECUTED=FALSE
