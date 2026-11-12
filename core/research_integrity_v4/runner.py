"""Resumable, serial DAG for implemented research stages; no synthetic success states."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2]
CANON=Path(os.environ.get('QRESEARCH_CANONICAL','/data1/stock-mxy/课题/一进二策略'))
EXEC=CANON/'research/round23_6_26_execution'

def main(default_round=None):
 p=argparse.ArgumentParser()
 p.add_argument('--phase',choices=['audit','reference','adapt','backtest','stats','report','publish','all'],default='all')
 p.add_argument('--method');p.add_argument('--outer-year',type=int);p.add_argument('--seed',type=int)
 p.add_argument('--resume',action='store_true');p.add_argument('--jobs',type=int,default=1);p.add_argument('--max-gpu-hours',type=float,default=24)
 a=p.parse_args();rounds=[default_round] if default_round else ['round23_6','round26']
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',QRESEARCH_EXECUTION_ROOT=str(EXEC))
 python=EXEC/'env/bin/python';python=python if python.exists() else Path(sys.executable)
 stages=['audit','reference','adapt','backtest','stats','report','publish'] if a.phase=='all' else [a.phase]
 for round_ in rounds:
  result=CANON/'result'/round_;result.mkdir(parents=True,exist_ok=True)
  statefile=result/'dag_execution.json';state=json.loads(statefile.read_text()) if statefile.exists() else {}
  def task(name,script,deps=(),interpreter=python,args=()):
   if any(state.get(d,{}).get('status')!='EXECUTED' for d in deps):
    state[name]={'status':'WAITING_DEPENDENCY','dependencies':list(deps)};statefile.write_text(json.dumps(state,indent=2));return
   path=ROOT/'core/research_integrity_v4/tasks'/script
   fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()
   if a.resume and state.get(name,{}).get('status')=='EXECUTED' and state[name].get('source_hash')==fingerprint:return
   started=time.time();log=result/(name+'.log')
   with log.open('w') as out:
    proc=subprocess.run([str(interpreter),str(path),*args],stdout=out,stderr=subprocess.STDOUT,env=env)
   state[name]={'status':'EXECUTED' if proc.returncode==0 else 'FAILED','returncode':proc.returncode,'source_hash':fingerprint,'elapsed_seconds':time.time()-started,'log':str(log),'scope':'only this task; not a statement that the full paper or stage is complete'}
   statefile.write_text(json.dumps(state,indent=2))
  for stage in stages:
   if stage=='audit':task('data_audit','audit_data.py')
   elif stage=='reference':
    methods=['TRA','DoubleAdapt','AlphaGen','AlphaForge','MASTER','DeepLOB']
    if a.method in ['SuperLearner','BOA','opera'] or (round_=='round23_6' and not a.method):task('official_R_reference','r_reference.py')
    for method in methods:
     if a.method and a.method!=method:continue
     if not a.method and ((round_=='round23_6')!=(method in ['TRA','DoubleAdapt'])):continue
     task('L0_'+method,'smoke.py',args=[method])
    if (not a.method and round_=='round26') or a.method=='RSR':task('L0_RSR','rsr_smoke.py',interpreter=EXEC/'rsr-env/bin/python')
   elif stage=='adapt':
    if round_=='round23_6' and (not a.method or a.method in ['SL','BOA','SuperLearner']):
     if a.outer_year is not None or a.seed not in [None,0]:
      state['meta_filter']={'status':'UNSUPPORTED_FILTER','reason':'deterministic aggregation bank is built jointly with all historical OOF years; this implementation does not silently ignore year/seed filters'}
     else:task('meta_bank_and_aggregation','meta.py',deps=['official_R_reference'])
    else:state['adapt_'+str(a.method or 'remaining_methods')]={'status':'NOT_IMPLEMENTED','reason':'No substituted algorithm is executed. Consult execution_status_by_paper.csv for actual pending and blocked work.'}
   elif stage=='backtest':
    if round_=='round23_6':task('aggregation_backtests','meta_backtest.py',deps=['meta_bank_and_aggregation'])
    else:state['round26_incrementality_backtests']={'status':'NOT_RUN','reason':'No qualified Round26 factor package has completed the evidence/OOF gates.'}
   elif stage=='stats':
    if round_=='round23_6':task('paired_statistics','statistics.py',deps=['aggregation_backtests'])
    else:state['round26_statistics']={'status':'NOT_APPLICABLE_NO_NEW_CANDIDATE','reason':'No invented candidate returns or p-values.'}
   elif stage=='report':task('report','report.py')
   elif stage=='publish':
    from core.research_integrity_v4.release import publish
    state['publish']=publish(EXEC/round_,round_,commit=True)
   statefile.write_text(json.dumps(state,indent=2))
  print(json.dumps({'round':round_,'state':state},ensure_ascii=False))
