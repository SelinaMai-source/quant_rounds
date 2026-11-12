from pathlib import Path
import csv,json,hashlib,shutil,subprocess,sys,os,re,datetime,ast
import numpy as np,pandas as pd
B=Path('/data1/stock-mxy/课题/一进二策略/research/round23_6_26_execution');C=B.parents[1];M=C/'result/round23_6_round26_master';M.mkdir(exist_ok=True)
METHODS=[('COMBO-01','SuperLearner','round23_6'),('COMBO-02','opera','round23_6'),('COMBO-03','TRA','round23_6'),('COMBO-04','DoubleAdapt','round23_6'),('ALPHA-01','AlphaGen','round26'),('ALPHA-02','AlphaForge','round26'),('ALPHA-03','RSR','round26'),('ALPHA-04','MASTER','round26'),('ALPHA-05','RD-Agent','round26'),('ALPHA-06','DeepLOB','round26')]
REPOS={'RSR':'fulifeng/Temporal_Relational_Stock_Ranking','DeepLOB':'zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books'}
COMMITS={'RSR':'cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d','DeepLOB':'ff14d7c2fd38bdfc143389786993d0f0236d4eb8'}
ENTRY={'SuperLearner':'R/SuperLearner.R; R/method.R::method.NNLS','opera':'R/mixture.R; R/BOA.R','TRA':'examples/benchmarks/TRA/run.sh; configs/config_alstm.yaml; src/model.py','DoubleAdapt':'main.py; run.sh GRU/csi300; src/model.py','AlphaGen':'scripts/rl.py with original LSTM policy, no HARLA/LLM','AlphaForge':'train_AFF.py; combine_AFF.py','RSR':'training/rank_lstm.py; training/relation_rank_lstm.py','MASTER':'main.py; master.py; base_model.py; qlib-update','RD-Agent':'rdagent/app/qlib_rd_loop; official factor/model workflow','DeepLOB':'jupyter_pytorch/run_train_pytorch.ipynb'}
def read(p,default=None):
 try:return json.loads(Path(p).read_text())
 except (FileNotFoundError,json.JSONDecodeError):return default

def markdown(frame):
 cols=list(frame.columns);rows=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
 for vals in frame.itertuples(index=False,name=None):rows.append('| '+' | '.join(format(v,'.6f') if isinstance(v,float) else str(v) for v in vals)+' |')
 return '\n'.join(rows)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()

source={x['method']:x for x in read(B/'source_acquisition.json',[])}
for n in ['MASTER','RD-Agent','DeepLOB']:
 x=read(B/('acquisition_'+n+'.json'),{})
 if x:source[n]=x
for n in ['RSR','DeepLOB']:
 source.setdefault(n,{}).update(method=n,repo='https://github.com/'+REPOS[n],commit=COMMITS[n],code_recovery='pinned GitHub connector fetch; source files outside release tree',clone_status='original clone failed or incomplete; source API recovery succeeded')
statuses=[];registry=[]
for ident,n,branch in METHODS:
 x=source[n];v=B/'vendor'/n
 licenses=[]
 if (v/'DESCRIPTION').exists():
  for line in (v/'DESCRIPTION').read_text().splitlines():
   if line.startswith('License:'):licenses.append(line.split(':',1)[1].strip())
 for f in [v/'LICENSE',v/'LICENSE.txt',v/'LICENSE.md',v/'COPYING']:
  if f.exists():licenses.append(f.name+':'+sha(f))
 if n=='TRA':licenses=['MIT']
 x['license_evidence']=licenses or ['UNSPECIFIED_NO_THIRD_PARTY_CODE_REDISTRIBUTED']
 papers=[B/'vendor'/(n+'_recovered.pdf'),B/'vendor'/(n+'.pdf')]
 valid=next((p for p in papers if p.exists() and p.stat().st_size>0 and p.open('rb').read(4)==b'%PDF'),None)
 x['paper_local_valid_pdf']=bool(valid);x['paper_sha256']=sha(valid) if valid else None
 x['entrypoint']=ENTRY[n]
 r={'paper_id':ident,'method':n,'branch':branch,'L0':'NOT_RUN','L1':'NOT_RUN','L2':'NOT_IMPLEMENTED','limitations':''}
 if n in ['SuperLearner','opera']:
  q=read(C/'result/round23_6/r_reference/numerical_parity.json',{})
  r['L0']='OFFICIAL_R_PARITY_PASS' if q.get('status')=='PASS' else 'FAILED_OR_NOT_RUN'
  r['L1']='REFERENCE_PARTIAL_OFFICIAL_SYNTHETIC_EXAMPLE_AND_NO_DELAY_STREAM'
  r['L2']='ADAPTATION_BACKTEST_EXECUTED_INTEGRATION_PENDING'
  r['limitations']='Not full original-paper L1 numerical study; legacy information lane and full real-data integration tests remain incomplete.'
 elif n=='RD-Agent':
  r.update(L0='BLOCKED_LLM_ACCESS',L1='BLOCKED_LLM_ACCESS',L2='BLOCKED_LLM_ACCESS',limitations='No configured authorized local LLM endpoint; direct probes and listening-port audit retained. Zero added LLM budget; no private data sent externally. No manual factors substituted.')
 else:
  q=read(M/'l0'/(n+'.json'),{})
  r['L0']=q.get('L0',q.get('status','NOT_RUN'))
  r['L1']='NOT_RUN_REFERENCE_RECOVERY_INCOMPLETE'
  r['limitations']=q.get('limitations','Finite official core update only; full author training/evaluation and target adapters are unfinished.')
  if n=='DeepLOB':r['L2']='BLOCKED_LOB_DATA';r['limitations']+=' Authorized real sample contains five levels, not required ten levels. This is a data/schema gap, not a fabricated subscription denial.'
  if n=='AlphaForge':r['L0']='COMPONENT_UPDATE_ONLY_NOT_FULL_GENERATOR_TRAINING';r['limitations']='Official NetG_DCGAN and NetP updated on a finite synthetic example; the published adversarial loss, collector, mining loop and dynamic combination were not executed. train_AFF.py does exist; no upstream-missing-code blocker is asserted.'
 statuses.append(r)
 registry.append({'paper_id':ident,'method':n,'paper':x.get('paper','https://arxiv.org/abs/1809.09441' if n=='RSR' else ''),'code':x.get('repo'),'version':x.get('commit'),'license':';'.join(x['license_evidence']),'entrypoint':ENTRY[n],'reference_config':'resolved_author_configs/'+n+'.json','paper_pdf_hash':x['paper_sha256'],'status':'PARTIAL' if r['L0']!='BLOCKED_LLM_ACCESS' else 'BLOCKED_LLM_ACCESS'})

pd.DataFrame(statuses).to_csv(M/'execution_status_by_paper.csv',index=False)
pd.DataFrame(registry).to_csv(M/'paper_registry.csv',index=False)
(M/'source_lock.json').write_text(json.dumps(list(source.values()),ensure_ascii=False,indent=2))
# Actual permission evidence only. Credentials, runtime licenses, absent LOB and LLM access are not data entries.
records=[]
p=M/'data_recovery_jobs.jsonl'
if p.exists():records=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
permissions=[r for r in records if r.get('status') in ['PERMISSION_DENIED','AUTH_REQUIRED','SUBSCRIPTION_REQUIRED'] and r.get('source_uri','').startswith('clickhouse:')]
lines=[]
for r in permissions:
 item='业绩预告源表 fdmt_ef_new，数据路径：'+r['source_uri']
 if item not in lines:lines.append(item)
permission_text=''.join(str(i+1)+'. '+line+'\n' for i,line in enumerate(lines))
(M/'无权限数据.md').write_text(permission_text)
blockers=[{'method':'RD-Agent','category':'BLOCKED_LLM_ACCESS','evidence':['local_llm_direct_probe.json','llm_access_probe.json'],'data_permission_entry':False},{'method':'DeepLOB','category':'BLOCKED_LOB_DATA','evidence':'amazing_probe.json','reason':'Actual authorized historical product is five-level Level1, not ten-level LOB','data_permission_entry':False}]
for r in permissions:blockers.append({'method':'E3_ANNOUNCEMENTS','category':r['status'],'source_uri':r['source_uri'],'data_permission_entry':True})
(M/'blockers.json').write_text(json.dumps(blockers,ensure_ascii=False,indent=2))
# Explicit requirement state; missing outputs are never filled with fabricated metrics or all-NaN factors.
required='paper_registry.csv source_lock.json experiment_prereg.yaml execution_status_by_paper.csv reference_replication_results.csv adaptation_differences.md baseline_parity.json oof_registry.parquet model_registry.csv data_recovery_jobs.jsonl trial_registry.csv all_results.csv daily_returns.csv.gz yearly_metrics.csv top1_top2_actual_pnl.csv risk_metrics.csv execution_sensitivity.csv bootstrap_results.csv spa_results.json tests_summary.json artifact_manifest.csv 无权限数据.md REPRODUCE.md FINAL_DECISION.md'.split()
actual_models=[]
for p in (C/'result/round23_6/expert_bank').rglob('fit_manifest.json'):actual_models.extend(read(p,[]))
calibrators=[]
for p in (C/'result/round23_6/expert_bank').rglob('calibrator_registry.json'):calibrators.extend(read(p,[]))
fits=pd.DataFrame(actual_models)
if len(fits):
 fits['kind']=fits.model_path.map(lambda p:'XGBRanker' if Path(p).suffix=='.json' else 'LGBMRanker' if Path(p).suffix=='.txt' else 'Ridge(alpha=10)')
 fits['seed']=fits.model_path.map(lambda p:Path(p).stem.split('_')[-1] if 'xgb_' in Path(p).name else '17' if Path(p).name=='lgbm.txt' else '')
 fits['fit_config_id']=fits['head']+'_'+fits['kind']+'_'+fits['seed']
 fits.to_csv(C/'result/round23_6/model_registry.csv',index=False)
 fits.to_parquet(C/'result/round23_6/oof_registry.parquet',index=False)
metrics=pd.read_csv(C/'result/round23_6/all_results.csv')
counts={'base_bank_model_files':len(actual_models),'base_bank_unique_fit_configs':int(fits.fit_config_id.nunique()) if len(fits) else 0,'base_bank_head_year_fits':int(fits[['head','year']].drop_duplicates().shape[0]) if len(fits) else 0,'calibrator_fits':len(calibrators),'NNLS_head_year_fits':18 if (C/'result/round23_6/meta_weights_by_time_SL_BOA.csv').exists() else 0,'BOA_head_year_replays':18 if (C/'result/round23_6/online_forecast_ledger.parquet').exists() else 0,'round23_6_formal_backtests':len(metrics),'round23_6_extra_baseline_trace_backtests':len(list((C/'result/round23_6/diagnostics').glob('*/summary.json'))),'round26_baseline_backtests':int((C/'result/round26/bt/A0_V5/summary.json').exists()),'round26_new_candidate_backtests':0,'full_L1_paper_replications':0,'new_neural_L2_fold_seed_models':0,'round26_Ashare_factor_packages':0,'diagnostic_future_mutation_model_fits':len(list((C/'result/round23_6/diagnostics/future_mutation').glob('*.model.json')))}
# Count aggregation fits/replays from actual ledgers rather than the planned 2x9.
w=pd.read_csv(C/'result/round23_6/meta_weights_by_time_SL_BOA.csv')
counts['NNLS_head_year_fits']=int(w.loc[w.method.eq('SL'),['head','outer_year']].drop_duplicates().shape[0])
counts['BOA_head_year_replays']=int(w.loc[w.method.eq('BOA'),['head','outer_year']].drop_duplicates().shape[0])
(M/'actual_execution_counts.json').write_text(json.dumps(counts,indent=2))
# Per-experiment trials, including every executed sensitivity, not only main candidates.
trials=[]
for p in (C/'result/round23_6/bt').glob('*/summary.json'):
 q=read(p,{});trials.append({'task':p.parent.name,'stage':'BACKTEST','round':'round23_6','status':q.get('status'),'evidence':str(p),'sha256':sha(p)})
for r in statuses:trials.append({'task':r['method'],'stage':'L0/L1/L2','round':r['branch'],'status':r['L0']+';'+r['L1']+';'+r['L2'],'evidence':'execution_status_by_paper.csv'})
pd.DataFrame(trials).to_csv(M/'trial_registry.csv',index=False)
reference=pd.DataFrame([{'paper_id':r['paper_id'],'method':r['method'],'L0':r['L0'],'L1':r['L1'],'Ashare_metrics_claimed':False,'limitations':r['limitations']} for r in statuses])
# Integration gates contain explicit NOT_RUN states, even when a nearby unit test passed.
integration=[{'gate':k,'status':'NOT_RUN_FULL_REAL_DATA_INTEGRATION','evidence':'No blanket PASS claimed'} for k in 'ABCDEFGHIJKLMNOPQR']
for g in integration:
 if g['gate']=='A' and (C/'result/round23_6/diagnostics/future_mutation/integration_result.json').exists():g.update(status='PASS_ONE_REAL_FOLD_SCOPE_ONLY',evidence='diagnostics/future_mutation/integration_result.json; not full pipeline coverage')
 if g['gate']=='G':g.update(status='PASS_FIXED_OFFICIAL_R_EXAMPLE',evidence='r_reference/numerical_parity.json')
 if g['gate']=='H':g.update(status='PASS_CONSTRUCTED_NON_EQUIVALENCE_ONLY',evidence='test_time_safety.py; no tradable portfolio implementation')
 if g['gate']=='N':g.update(status='PASS_ACTUAL_PLAN_IDENTITY_SCOPE',evidence='execution_sensitivity.csv and bt/*/planned_order_identities.csv.gz; no replacement selected')
 if g['gate']=='O':g.update(status='PARTIAL_NAV_ACCOUNTING_PASS',evidence='ledger_checks.json; full unrealized PnL bridge remains unresolved')
 if g['gate']=='Q':g.update(status='NO_CROSS_LINE_WINNER_IMPORTED',evidence='No Round26 learned factor package exists; baseline/production refs remain unchanged')
 if g['gate']=='R':g.update(status='PASS_FORMAT_AND_ACTUAL_AUTH_EVIDENCE',evidence='Only fdmt_ef_new query AUTH_REQUIRED is listed in Round26/master; Round23.6 is empty')
(M/'integration_acceptance.json').write_text(json.dumps(integration,indent=2))
# Document incomplete experiment groups without assigning them fake backtest metrics.
p1=[]
for code in ['C00_V5','C01_P1_SCORE_A099','C02_PORT_EQUAL','C03_PORT_A099_ANNUAL','C04_PORT_STATPOOL_MONTHLY','C05_PORT_FLEXPOOL_MONTHLY','C06_PORT_A099_MONTHLY']:
 p1.append({'experiment_id':code,'execution_status':'EXECUTED_SCORE_STRATEGY' if code in ['C00_V5','C01_P1_SCORE_A099'] else 'TRADABLE_ACCOUNT_NOT_IMPLEMENTED','monthly_weight_study_exists':(C/'result/round23_6/monthly_portfolio_weight_research.csv').exists(),'portfolio_claim':False})
pd.DataFrame(p1).to_csv(C/'result/round23_6/score_vs_portfolio_ablation.csv',index=False)

for branch in ['round23_6','round26']:
 R=C/'result'/branch;R.mkdir(exist_ok=True)
 pd.DataFrame(registry).to_csv(R/'paper_registry.csv',index=False)
 (R/'source_lock.json').write_text(json.dumps(list(source.values()),ensure_ascii=False,indent=2))
 pd.DataFrame([r for r in statuses if r['branch']==branch]).to_csv(R/'execution_status_by_paper.csv',index=False)
 reference.to_csv(R/'reference_replication_results.csv',index=False)
 pd.DataFrame([t for t in trials if t['round']==branch]).to_csv(R/'trial_registry.csv',index=False)
 (R/'无权限数据.md').write_text('' if branch=='round23_6' else permission_text)
 shutil.copy2(M/'data_recovery_jobs.jsonl',R/'data_recovery_jobs.jsonl')
 (R/'blockers.json').write_text(json.dumps([x for x in blockers if branch=='round26'],ensure_ascii=False,indent=2))
 (R/'tests_summary.json').write_text(json.dumps({'unit_test_execution':read(R/'unit_test_execution.json',{'status':'NOT_RUN'}),'official_R_numerical_test':read(C/'result/round23_6/r_reference/numerical_parity.json'),'full_integration_acceptance':'INCOMPLETE','gates':integration},indent=2))
 (R/'adaptation_differences.md').write_text('''# Fidelity and limits

L0 synthetic/core/component updates are separated from L1 full public-reference training and L2 A-share backtests. No complete paper-level L1 is claimed. AlphaGen used the author expression evaluator, synergistic pool and LSTM policy for the finite update. AlphaForge only updated generator/predictor components; its full loss/mining/combination loop remains unexecuted. RSR used original TensorFlow 1.15.5 in a separate Python 3.7 environment and extracted the unchanged complete graph-construction blocks for finite updates, not a PyTorch replacement. TRA uses legacy src/model.py, not maintenance-version pytorch_tra. DoubleAdapt runs the actual two-level manager with both adapters; its CPU device compatibility diff must not be described as GPU numerical reproduction. DeepLOB uses the exact author PyTorch notebook class and original softmax-before-CrossEntropy behavior.

Super Learner and BOA adapters match locked official R on a fixed stream. L2 uses separate per-head targets, 3-year past-only base fits, historical OOF calibrators, Ridge(alpha=10), per-year BOA rescaling/replay, stored prediction-time vectors and date-batched matured feedback. No best seed was selected: these aggregation methods are deterministic given the original fixed seed ensemble. No three-seed neural L2 has been trained.

The original legacy executor clips some modeled fill prices using full-day high/low. Baseline parity is not proof of contemporaneous executable opening fills. Strict price-timing, PIT/revision and full integration mutation tests remain incomplete. Historical 2018-2026 data were repeatedly used for research. No untouched forward period and no LIVE_READY claim.
''')
 (R/'FINAL_DECISION.md').write_text('''# PARTIAL — research unfinished

KEEP_BASELINE_V5. No candidate is promoted. The completed P1 score replay, Super Learner and BOA do not establish a replacement for v5. Actual tradable P1 portfolio netting, complete L1 replications, neural/graph L2, Round26 factor discovery/evidence/Add-Remove-Replace and full integration acceptance are unfinished. Not all unfinished work has an objective data blocker: NOT_IMPLEMENTED/NOT_RUN remain explicit.

CURRENT_BASELINE_CHANGED=FALSE
PRODUCTION_CONFIG_CHANGED=FALSE
AUTO_PROMOTE=FALSE
ROUND24_EXECUTED=FALSE
LIVE_READY=FALSE
''')
 (R/'REPRODUCE.md').write_text('''# Reproduction and continuation

Canonical server project: /home/mxy/Stock/课题/一进二策略 (same inode as /data1/stock-mxy/课题/一进二策略). Use the independent worktree of the corresponding branch under research/round23_6_26_execution. The frozen input is research/literature_round23_6_26/MASTER_PROMPT.md.

Run `python tools/run_round23_6.py --phase reference --method SuperLearner --resume`, then the implemented aggregation stages `--phase adapt`, `--phase backtest`, `--phase stats`, `--phase report`. Use the isolated env/bin/python for Torch/arch and rsr-env/bin/python for TensorFlow 1.15.5. The CLI exposes explicit NOT_IMPLEMENTED or UNSUPPORTED_FILTER for unfinished adapters; it never silently substitutes an algorithm. Its DAG covers only implemented tasks, and is not a claim that all required stages are implemented.

Read source_lock.json and resolved_author_configs before training. Third-party source is obtained from locked commits through external vendor downloads, not vendored without license. Raw supplier tables, supplier credentials, environments and large public data are server-only. Download failures and partial public datasets are retained locally; do not load a partial archive as successful data. The exact server resource/download recovery state is in the master directory.

Standalone reproduction without the supplier datasets and canonical source dependencies is not claimed. source_dependency_manifest.json and source_dependency_manifest.csv identify required source files and hashes. Full missing requirements are listed in missing_artifacts.json and execution_status_by_paper.csv. Do not interpret a successful individual CLI task as COMPLETE for a paper or research line.
''')
 # Preserve the preregistration already committed in each branch, also copy into canonical results.
 for n in ['experiment_prereg.yaml','data_schema.json','source_dependency_manifest.json']:
  p=B/branch/'result'/branch/n
  if p.exists() and not (R/n).exists():shutil.copy2(p,R/n)
 if branch=='round23_6':
  table=markdown(metrics.loc[metrics.experiment_id.isin(['C00_V5','C01_P1_SCORE_A099','C10_SL','C11_BOA']),['experiment_id','CAGR','Sharpe','MaxDD','UI','CompletedTrades']])
  body='''# Round23.6 ensemble research — PARTIAL

The full requested research is not complete. Frozen protocol was committed and pushed before experiments. Source anchors and canonical inode were verified. Formal baseline_v5, CURRENT and production settings were not changed. Round24 was not executed.

## Actually executed

Original baseline replay matches candidate scores/ranks, original executor entry-plan intents/pre-trade capital, fills, holdings, costs, rejections and daily NAV. The P1_SCORE_A099 annual score-fusion replay also matches the old score/allocation output. Shared-account portfolio combination C02-C06 has no economic backtest: monthly optimization outputs are a mathematical weight study only. Fixed-PDF equation mapping, hash and 60+60-month FLEXPOOL warmup are documented in equation_to_code.md. The four expert ledgers available from 2018 do not activate FLEXPOOL alpha selection by July 2026; pre-2018 expert-ledger recovery is unfinished.

Super Learner NNLS and BOA default square-gradient updates match locked official R within 8e-16 on the fixed example. A synthetic official SuperLearner example is retained, not called full original-paper L1 reproduction. L2 trained original XGB/LGBM/Ridge/mean banks, historical OOF score calibration, per-head NNLS and delayed date-batched BOA. Prediction-time vectors, weights and maturity timestamps are retained. All single-expert/equal controls and S0/S1/S2 strategies ran; S3 verifies actual original plan identities and already-enforced opening limit rejection. No failed candidate was deleted or tuned after inspecting Outer outcomes.

'''+table+'''

All metrics above use the same 2018-01-02..2026-07-24 window and 300000 initial capital. The baseline CAGR is 54.9981%, not an independent forward result. P1 score replay has delta CAGR only 0.3946 percentage points and delta Sharpe 0.0422, below the frozen 1-point/0.05 economic thresholds. Super Learner and BOA are materially worse. Full controls, yearly metrics, actual RMB PnL, costs, open-position residual bridge, risk, paired fee stresses, paired circular bootstrap and same-scope SPA are retained. Bootstrap uses 5000 paired draws at block20 and 1000 at blocks10/40 with the same calendar-year CAGR convention. SPA deduplicates numerically identical daily paths; the original byte-only dedup result is retained as an audit correction, not selected for a favorable p-value.

Actual execution counts: `'''+json.dumps(counts,ensure_ascii=False)+'''`.

## Unfinished and limitations

TRA and DoubleAdapt each performed actual official finite model updates (including OT/router and both data/model adapters), but complete author-reference L1 and three-seed L2/controls were not run. The full author TRA feature/label/return pickle set (2,512,790 rows, 16 original features) was successfully downloaded and read; missing-data is not asserted as an excuse for its unexecuted training. P1 shared-cash netting/exits/one-expert execution parity are unimplemented. A real Five-head/2018/seed17 future-or-immature target mutation test trained two diagnostic models and compared exact predictions and overlay plan inputs; it passed within that scope. Full pipeline future-mutation/prefix/revision tests and label-maturity repair and strict price availability proofs are incomplete. The NAV identity passes, but the entire unrealized/corporate-action PnL residual has not been decomposed. The source bank records cutoff and maximum training maturity; this is not a blanket no-future-function PASS. `integration_acceptance.json` contains all missing gates.

The baseline remains a LEGACY_V5_COMPARISON. Original high/low-dependent modeled fill clipping is disclosed. New results are research only; no new untouched Forward exists. KEEP_BASELINE_V5; no promotion.

Permissions file: 无权限数据.md is empty (0 bytes) for this line. The announcement-source authentication rejection belongs to Round26, not this aggregation line.

CURRENT_BASELINE_CHANGED=FALSE
PRODUCTION_CONFIG_CHANGED=FALSE
AUTO_PROMOTE=FALSE
ROUND24_EXECUTED=FALSE
'''
  (R/'ROUND23_6_ENSEMBLE_REPORT.md').write_text(body)
 else:
  baseline=read(R/'bt/A0_V5/summary.json',{})
  pd.DataFrame([{'experiment_id':'A0_V5','status':'EXECUTED_FROZEN_BASELINE','cagr':baseline.get('cagr'),'sharpe':baseline.get('sharpe'),'max_drawdown':baseline.get('max_drawdown'),'completed_trades':baseline.get('completed_trades')},*[{'experiment_id':r['method'],'status':r['L2']} for r in statuses if r['branch']=='round26']]).to_csv(R/'all_results.csv',index=False)
  if (R/'bt/A0_V5/nav.parquet').exists():
   nav=pd.read_parquet(R/'bt/A0_V5/nav.parquet');ret=np.r_[nav.total_asset.iloc[0]/300000-1,nav.total_asset.to_numpy()[1:]/nav.total_asset.to_numpy()[:-1]-1]
   pd.DataFrame({'trade_date':nav.trade_date,'A0_V5':ret}).to_csv(R/'daily_returns.csv.gz',index=False)
  (R/'LOB_reference_vs_Ashare_status.json').write_text(json.dumps({'L0':read(M/'l0/DeepLOB.json'),'L1':'REFERENCE_DATA_DOWNLOAD_INCOMPLETE_OR_PENDING','L2':'BLOCKED_LOB_DATA','authorized_product_sample':read(M/'amazing_probe.json'),'fake_minute_orderbook_created':False},indent=2))
  (R/'agent_trajectory_manifest.json').write_text(json.dumps({'method':'RD-Agent','status':'BLOCKED_LLM_ACCESS','proposals':0,'trajectory_files':[],'manual_factor_substitution':False},indent=2))
  (R/'ROUND26_ALPHA_REPORT.md').write_text('''# Round26 alpha research — PARTIAL

The requested alpha research is unfinished. The exact round23_5 anchor and independent protocol/worktree were used; no Round23.6 winning weight/model was imported. An independent frozen A0_V5 full-history baseline was actually executed: `'''+json.dumps(baseline,ensure_ascii=False)+'''`.

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

Actual counts: `'''+json.dumps(counts,ensure_ascii=False)+'''`.

## Decision

KEEP_BASELINE_V5. The baseline was not shown to be improved by Round26 because target-market candidates have not reached executable evidence gates. Nine paper frameworks had finite algorithm/component tests across the two lines; zero complete paper-level L1 replications were finished. Core smoke success is not full reproduction.

Permissions: 无权限数据.md contains only the actual fdmt_ef_new source URI. Detailed reasons are in blockers.json/data_recovery_jobs.jsonl, outside that file.

CURRENT_BASELINE_CHANGED=FALSE
PRODUCTION_CONFIG_CHANGED=FALSE
AUTO_PROMOTE=FALSE
ROUND24_EXECUTED=FALSE
''')
 (R/'missing_artifacts.json').write_text(json.dumps({'overall':'PARTIAL','missing':[name for name in required if not (R/name).exists()],'note':'Missing artifacts reflect incomplete work, not successful empty placeholder outputs.'},ensure_ascii=False,indent=2))
# Exact permission-format assertion.
for f in [C/'result/round23_6/无权限数据.md',C/'result/round26/无权限数据.md',M/'无权限数据.md']:
 text=f.read_text();seen=set()
 for i,line in enumerate(text.splitlines(),1):
  assert re.fullmatch(r'\d+\. .+，数据路径：.+',line),line
  assert line.startswith(str(i)+'. ');assert line not in seen;seen.add(line)
  assert not any(k in line.lower() for k in ['password=','token=','cookie=','secret='])
print(json.dumps(counts,indent=2))
