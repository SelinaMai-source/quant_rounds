"""Own-process TRA jobs, immutable results and retained failed attempts."""
from pathlib import Path
import json,os,subprocess,sys,time,threading
import pandas as pd
from core.round23_7.tra_reference import C,R
_REGISTRY_LOCK=threading.Lock()

def run_job(head,year,seed,phase,data,*,init_state=None,fit_label_override=None,namespace='original'):
    token=str(time.time_ns());folder=R/'logs'/namespace;folder.mkdir(parents=True,exist_ok=True)
    log=folder/f'{seed}_{year}_{head}_{phase}_{token}.log'
    command=[sys.executable,'-B','-u','-m','core.round23_7.tra_fit','--head',head,'--year',str(year),'--seed',str(seed),'--phase',phase,'--data',str(data),'--namespace',namespace]
    if init_state is not None:command+=['--init-state',str(init_state)]
    if fit_label_override is not None:command+=['--fit-label-override',str(fit_label_override)]
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    with log.open('w') as stream:
        process=subprocess.Popen(command,cwd=C,env=env,stdout=stream,stderr=subprocess.STDOUT)
        (log.with_suffix('.LAUNCHED.json')).write_text(json.dumps({'pid':process.pid,'command':command,'UTC_epoch':time.time()},indent=2))
        code=process.wait()
    result=None
    for line in log.read_text(errors='replace').splitlines():
        if 'TRA_TARGET_RESULT ' in line:
            result=json.loads(line.split('TRA_TARGET_RESULT ',1)[1])
    if code!=0 or result is None:
        result={'status':'FAILED_TARGET_EXECUTION','head':head,'year':int(year),'seed':int(seed),'phase':phase,'namespace':namespace,'path':'','fingerprint':'','cache_hit':False,'external_blocker':False}
    result.update(returncode=code,log=str(log),attempt=token)
    attempt=R/'worker_attempts'/namespace/f'{seed}_{year}_{head}_{phase}_{token}.json'
    attempt.parent.mkdir(parents=True,exist_ok=True);attempt.write_text(json.dumps(result,indent=2))
    print('TRA_JOB_STATUS',json.dumps(result),flush=True)
    return result

def save_registry(rows):
    with _REGISTRY_LOCK:return _save_registry(rows)

def _save_registry(rows):
    path=R/'target_run_registry.csv';old=pd.read_csv(path).to_dict('records') if path.exists() else []
    merged={(v['namespace'],int(v['seed']),int(v['year']),v['head'],v['phase']):v for v in old+list(rows)}
    frame=pd.DataFrame(list(merged.values())).sort_values(['namespace','seed','year','head','phase'])
    temporary=path.with_name(path.name+'.partial.registry');frame.to_csv(temporary,index=False);os.replace(temporary,path)
    return frame
