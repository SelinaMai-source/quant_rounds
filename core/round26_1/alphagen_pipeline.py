"""D3 seed17 full end-to-end before42/73; at most two true CPU learners."""
from pathlib import Path
import subprocess,time,json,os,sys,shutil,gzip
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd
from core.round26_1.market import C,B,R
from core.research_integrity_v5.cache import identity,sha
PYTHON=C/'research/round23_6_26_execution/env/bin/python'

def archive_attempt_metadata(arm,year,seed):
 """A resumed attempt must not erase the preceding measured resource segment."""
 runbase=B/'alphagen_runs'/arm/str(year)/str(seed)
 files=list((R/'worker_profiles').glob(f'{arm}_{year}_{seed}_*.json'))
 for name in ['RESOURCES.json','STATUS.json','RESUME.json','PROGRESS.json','HEARTBEAT.json','COMPLETE.json']:
  files.extend(runbase.glob('*/'+name))
 if not files:return
 dest=R/'attempt_history'/arm/str(seed)/str(year)/str(time.time_ns())
 dest.mkdir(parents=True,exist_ok=True);manifest=[]
 for source in files:
  relative=source.relative_to(C)
  target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
  manifest.append({'original':str(source),'snapshot':str(target.relative_to(dest)),'bytes':target.stat().st_size,'sha256':sha(target)})
 (dest/'manifest.json').write_text(json.dumps({'note':'actual prior-attempt profiles/statuses before launching another process; original checkpoint directories retained server-side','artifacts':manifest},ensure_ascii=False,indent=2))

def copy_completed(row):
 directory=Path(row['directory']);dest=R/'completed_factors'/row['arm']/str(row['seed'])/str(row['year']);dest.mkdir(parents=True,exist_ok=True)
 for name in ['CONTRACT.json','FROZEN_FACTOR.json','SELECTED_CHECKPOINT.json','factor_predictions.parquet','RESOURCES.json','RESOLVED_PPO.json','COMPLETE.json']:
  if (directory/name).exists():shutil.copy2(directory/name,dest/name)
 source=directory/'all_expression_evaluations.jsonl'
 if source.exists():
  with source.open('rb') as src,gzip.open(dest/'all_expression_evaluations.jsonl.gz','wb') as out:shutil.copyfileobj(src,out)
 artifacts=[]
 for p in sorted((directory/'resume_slots').rglob('*')):
  if p.is_file():artifacts.append({'schema':'original SB3 policy zip / own AST+RNG state; raw StockData excluded','path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
 (dest/'large_checkpoint_manifest.json').write_text(json.dumps(artifacts,ensure_ascii=False,indent=2))

def fold_job(arm,year,seed):
 archive_attempt_metadata(arm,year,seed)
 log=R/'logs'/f'{arm}_{year}_{seed}.log';log.parent.mkdir(parents=True,exist_ok=True);runbase=B/'alphagen_runs'/arm/str(year)/str(seed);prior={str(p):p.stat().st_mtime_ns for p in runbase.glob('*/RESOLVED_PPO.json')};started=time.time();env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
 with log.open('a') as stream:
  process=subprocess.Popen([str(PYTHON),'-u','-m','core.round26_1.alphagen_worker','--arm',arm,'--year',str(year),'--seed',str(seed),'--resume'],cwd=C,stdout=stream,stderr=subprocess.STDOUT,env=env)
  # The launcher serializes only shared-data preparation, not PPO training.
  # Mark when source tensors/labels/fingerprints have all been frozen.
  while process.poll() is None:
   fresh=any(str(p) not in prior or p.stat().st_mtime_ns>prior[str(p)] for p in runbase.glob('*/RESOLVED_PPO.json'))
   if fresh:break
   time.sleep(1)
  (R/'logs'/f'{arm}_{year}_{seed}.PREPARED.json').write_text(json.dumps({'pid':process.pid,'data_prepared_or_process_finished':True,'start_wall':started}))
  returncode=process.wait()
 result=R/'worker_profiles'/f'{arm}_{year}_{seed}_result.json'
 if returncode or not result.exists():return {'arm':arm,'year':year,'seed':seed,'directory':'','status':'FAILED_EXECUTION','returncode':returncode,'log':str(log)}
 row=json.loads(result.read_text());row['returncode']=returncode;row['log']=str(log)
 if row['status']=='FACTOR_OOF_YEAR_COMPLETE':copy_completed(row)
 return row

def assert_no_other_controller():
 import psutil
 for proc in psutil.process_iter(['pid','uids','cmdline']):
  if proc.pid==os.getpid():continue
  try:
   if proc.uids().real!=os.getuid():continue
   args=proc.cmdline()
   if any(Path(a).name=='continue_round23_7_round26_1.py' for a in args) and '--deliverable' in args and args[args.index('--deliverable')+1] in ['D3','all']:
    raise RuntimeError(f'Existing D3 controller PID {proc.pid}; do not launch duplicate CPU learners. Resume only after that process has exited.')
  except (psutil.NoSuchProcess,psutil.AccessDenied):continue

def run(jobs=1,publish=False):
 assert_no_other_controller()
 if jobs not in [1,2]:raise ValueError('at most2 CPU training processes')
 R.mkdir(parents=True,exist_ok=True);registry=[]
 if (R/'factor_run_registry.csv').exists():registry=pd.read_csv(R/'factor_run_registry.csv').to_dict('records')
 for seed in [17,42,73]:
  for year in range(2020,2027):
   # At most two tasks, with shared preprocessing initialized one at a time.
   with ThreadPoolExecutor(max_workers=jobs) as pool:
    futures=[]
    for arm in ['AG_RAW','AG_RES']:
     marker=R/'logs'/f'{arm}_{year}_{seed}.PREPARED.json'
     if marker.exists():marker.unlink()
     futures.append(pool.submit(fold_job,arm,year,seed))
     # Do not overlap two writers of identical immutable calibration inputs.
     while not marker.exists() and not futures[-1].done():time.sleep(1)
    for future in as_completed(futures):
     row=future.result();registry=[v for v in registry if not(v['arm']==row['arm'] and int(v['year'])==row['year'] and int(v['seed'])==row['seed'])];registry.append(row);pd.DataFrame(registry).to_csv(R/'factor_run_registry.csv',index=False);print('D3 fold status',row,flush=True)
  required=[v for v in registry if int(v['seed'])==seed];all_complete=len(required)==14 and all(v['status']=='FACTOR_OOF_YEAR_COMPLETE' for v in required)
  if not all_complete:
   (R/'STATUS.json').write_text(json.dumps({'status':'PARTIAL_GENERATION','IMPLEMENTED':True,'UNIT_TESTED':True,'TARGET_OOF_COMPLETE':False,'BACKTEST_COMPLETE':False,'EVIDENCE_COMPLETE':False,'unfinished':required,'later_seeds_not_started':'seed17/previous seed must finish its complete path first'},ensure_ascii=False,indent=2));return
  from core.round26_1.factor_models import build_factor_banks,run_models
  banks=build_factor_banks(pd.DataFrame(registry),seeds=(seed,),include_composite=False);run_models(banks,jobs=jobs)
  from core.round26_1.factor_evidence import report
  report(registry,stage_seed=seed)
  if publish:
   from core.research_integrity_v5.publish import publish as push
   push('round26_1',f'D3 checkpoint: complete real AlphaGen seed{seed} OOF, Add and evidence')
 # Three-seed combination uses prior-generation training scales, never winner selection.
 from core.round26_1.factor_models import build_factor_banks,run_models
 banks=build_factor_banks(pd.DataFrame(registry),seeds=(17,42,73),include_composite=True);run_models({key:path for key,path in banks.items() if key[1]=='EQ3'},jobs=jobs)
 from core.round26_1.factor_evidence import report
 report(registry,stage_seed='EQ3')
 if publish:
  from core.research_integrity_v5.publish import publish as push
  push('round26_1','D3: complete two-arm original PPO A-share nested OOF and all frozen Add trials')
