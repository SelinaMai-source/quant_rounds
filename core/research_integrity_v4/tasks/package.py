from pathlib import Path
import json,shutil,hashlib,subprocess,os,re,importlib.metadata,datetime
import pandas as pd
B=Path('/data1/stock-mxy/课题/一进二策略/research/round23_6_26_execution');C=B.parents[1];M=C/'result/round23_6_round26_master'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
# Probe the actual runtime, not expected capabilities.
import torch,psutil
runtime={'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':__import__('sys').version,'torch_version':torch.__version__,'cuda_available':torch.cuda.is_available(),'gpu_device_count':torch.cuda.device_count(),'cpu_count':os.cpu_count(),'memory_total':psutil.virtual_memory().total,'memory_available':psutil.virtual_memory().available,'memory_limit_fraction':.7,'configured_cpu_threads':2,'paid_data_or_llm_added_cost':0,'CURRENT_BASELINE_CHANGED':False,'PRODUCTION_CONFIG_CHANGED':False,'AUTO_PROMOTE':False,'ROUND24_EXECUTED':False}
for n in ['numpy','pandas','xgboost','lightgbm','scipy','scikit-learn','arch','higher','stable-baselines3','sb3-contrib','gdown','pyqlib']:
 try:runtime[n]=importlib.metadata.version(n)
 except importlib.metadata.PackageNotFoundError:runtime[n]='NOT_INSTALLED'
dump(M/'runtime_and_resources.json',runtime)
# Record every public download artifact, including partial/failed data, without shipping raw datasets.
files=[]
for p in (B/'public_data').rglob('*'):
 if not p.is_file():continue
 files.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'status':'PARTIAL_TRANSFER' if p.suffix=='.part' or p.name in ['RSR_source.zip','FI2010.zip','FI2010_master.zip'] else 'PRESENT_REQUIRES_SCHEMA_VALIDATION','redistributed':False})
dump(M/'public_download_artifacts.json',files)
for branch in ['round23_6','round26']:
 W=B/branch;R=C/'result'/branch;out=W/'result'/branch
 taskdir=W/'core/research_integrity_v4/tasks';taskdir.mkdir(parents=True,exist_ok=True)
 for name in ['audit_data','baseline','trace_baseline','p1_replay','p1_monthly','r_reference','smoke','rsr_smoke','statistics','probe_amazing','report']:
  p=B/(name+'.py')
  text=p.read_text().replace('B=Path('/data1/stock-mxy/课题/一进二策略/research/round23_6_26_execution')',"B=Path('/data1/stock-mxy/课题/一进二策略/research/round23_6_26_execution')")
  (taskdir/p.name).write_text(text)
 shutil.copy2(B/'round23_6/core/round23_6/meta.py',taskdir/'meta.py')
 # Keep exact committed canonical dependencies as a separate source snapshot,
 # rather than silently replacing the historical branch files.
 deps=[];srcdest=W/'research/canonical_dependencies'
 groups=[(C/'core','core'),(Path('/data1/stock-mxy/src/first_board_hermes'),'stock/first_board_hermes')]
 for root,prefix in groups:
  for p in root.rglob('*.py'):
   if '__pycache__' in p.parts or any(q.startswith('round24') for q in p.relative_to(root).parts):continue
   dest=srcdest/prefix/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
   deps.append({'server_path':str(p),'repo_path':str(dest.relative_to(W)),'sha256':sha(p),'bytes':p.stat().st_size,'role':'actual server source dependency; Stock remains read-only'})
 pd.DataFrame(deps).to_csv(out/'source_dependency_manifest.csv',index=False)
 # Copy outputs using a raw-data exclusion policy; original full artifacts remain on server.
 exported=[]
 for p in R.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(R)
  if p.name=='frozen_components.parquet':
   exported.append({'server_path':str(p),'sha256':sha(p),'size':p.stat().st_size,'export':'SERVER_ONLY_DERIVED_INPUT','reason':'contains candidate input gap alongside prediction components'});continue
  if 'expert_bank' in rel.parts and p.suffix=='.parquet':
   frame=pd.read_parquet(p);drop=[n for n in frame if n.startswith('label_') and n!='label_available_at']
   dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);frame.drop(columns=drop).to_parquet(dest,index=False)
   exported.append({'server_path':str(p),'sha256':sha(p),'size':p.stat().st_size,'repo_path':str(dest.relative_to(W)),'export':'DERIVED_PREDICTIONS_WITH_TARGET_VALUES_REMOVED'});continue
  if any(x in rel.parts for x in ['raw','raw_snapshot_sample','supplier_raw']):continue
  dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.resolve()!=p.resolve():shutil.copy2(p,dest)
 dump(out/'server_artifact_export_policy.json',exported)
 # Shared source/recovery snapshots are identical in the two lines.
 for n in ['source_acquisition.json','acquisition_MASTER.json','acquisition_RD-Agent.json','acquisition_DeepLOB.json','paper_recovery_RSR.json','paper_recovery_SuperLearner.json']:
  if (B/n).exists():shutil.copy2(B/n,W/'research/vendor_sources'/n)
 for name in ['acquire.py','acquire_remaining.py']:
  shutil.copy2(B/name,W/'research/vendor_sources'/name)
 # Add unresolved author config snapshots before any eventual L1 starts.
 cfgdir=out/'resolved_author_configs';cfgdir.mkdir(exist_ok=True)
 for name,paths in {'AlphaForge':['train_AFF.py','combine_AFF.py'],'MASTER':['main.py','master.py','base_model.py','qlib-update/pytorch_master_ts.py'],'RSR':['training/rank_lstm.py','training/relation_rank_lstm.py'],'DeepLOB':['run_train_pytorch.ipynb'],'RD-Agent':['rdagent/scenarios/qlib/conf.py']}.items():
  src=B/'vendor'/('RSR_api' if name=='RSR' else 'DeepLOB_api' if name=='DeepLOB' else name)
  old={}
  file=cfgdir/(name+'.json')
  if file.exists():old=json.loads(file.read_text())
  old['method']=name;old['files']={str(p):{'sha256':sha(src/p)} for p in paths if (src/p).exists()};old['L1_started']=False
  old['resolved_before_L1']=True
  if name=='MASTER':old['author_main']={'universe':'csi300','prefix':'opensource','d_feat':158,'d_model':256,'t_nhead':4,'s_nhead':2,'dropout':.5,'market_dims':63,'beta':5,'n_epoch':1,'lr':1e-5,'train_stop_loss_thred':.95,'validation':'infer_processor; label mask only for loss/metrics'}
  if name=='RSR':old['author_NASDAQ_command']={'relation':'wikidata','seq':16,'unit':64,'alpha':.1,'lr':.001,'epochs':50,'embedding_file':'NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy','backbone':'TensorFlow1 BasicLSTMCell'}
  if name=='DeepLOB':old['author_notebook']={'k':4,'T':100,'epochs':50,'batch_size':64,'optimizer':'Adam','lr':.0001,'criterion':'CrossEntropyLoss','normalization':'author DecPre files; 80/20 train/valid; original test7/8/9','boundary_safety':'stock/day IDs must be verified before claiming strict L1'}
  dump(file,old)
 dump(out/'runtime_and_resources.json',runtime)
 # Required raw schema + hashes for failure cases, no supplier rows published.
 failure=out/'failure_examples';failure.mkdir(exist_ok=True)
 dump(failure/'initial_output_guard.json',{'evidence_origin':'observed actual tool execution traceback, reconstructed after log rotation','error_type':'RuntimeError','error':'refusing to write historical or foreign result: result/round23_6/bt/C00_V5/overlay_meta.json','resolution':'process-local allowed output tuple extended only to round23_6, round26 and their master; canonical source file unchanged','state':'RECOVERED'})
 dump(failure/'incomplete_integration.json',{'state':'NOT_RUN_FULL_REAL_DATA_MUTATION_TESTS','input_schema':['trade_date','instrument_id','label_available_at','original head feature whitelist'],'actual_model_registry_sha256':sha(C/'result/round23_6/model_registry.csv'),'candidate_input_sha256':sha(C/'result/round22_3/data/five_candidate.parquet'),'reason':'unit tests and cutoff registries do not substitute for full end-to-end future-mutation/prefix replay'})
 # Report generation initially failed due an optional rendering package; retain it.
 dump(failure/'report_optional_dependency.json',{'evidence_origin':'actual tool traceback','error_type':'ImportError','missing_module':'tabulate','resolution':'render the static Markdown comparison table without an extra dependency','state':'RECOVERED'})
 # Do not call the full experiment complete just because report packaging succeeded.
 required=json.loads((out/'missing_artifacts.json').read_text());required['missing']=[n for n in required['missing'] if not (out/n).exists()];dump(out/'missing_artifacts.json',required)
 # Canonical information is read-only; the release source snapshot is own code only.
 text=out/'REPRODUCE.md';text.write_text(text.read_text()+'\nSource recovery is pinned by commit/blob hashes. Canonical dependency snapshot is under research/canonical_dependencies; production files were not overwritten. No vendor SDK binaries, raw market databases or credentials are included.\n')
 # Artifact manifest deliberately excludes itself to avoid a self-hash cycle.
 included=[]
 for root in [out,W/'core/research_integrity_v4',W/'core'/branch,W/'research/canonical_dependencies',W/'research/literature_round23_6_26',W/'research/vendor_sources']:
  if not root.exists():continue
  for p in root.rglob('*'):
   if p.is_file() and p.name!='artifact_manifest.csv' and '__pycache__' not in p.parts:
    included.append({'path':str(p.relative_to(W)),'size':p.stat().st_size,'sha256':sha(p),'license':'original project code/user-authorized derived artifact; external library licenses referenced separately' if 'canonical_dependencies' in str(p) else 'see source_lock and THIRD_PARTY_NOTICES; no raw vendor data'})
 pd.DataFrame(included).to_csv(out/'artifact_manifest.csv',index=False)
print('Packaged current PARTIAL results and exact source dependency snapshots')
