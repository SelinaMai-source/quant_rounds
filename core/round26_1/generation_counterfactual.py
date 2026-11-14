"""Full original PPO unavailable-target counterfactuals, never extra alpha search."""
from pathlib import Path
import json,subprocess,os,sys,time,threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from core.round26_1.market import C,R,B
from core.research_integrity_v5.cache import identity


def ensure(jobs=1):
    if jobs not in [1,2]:raise ValueError('At most2 CPU learners')
    registry=pd.read_csv(R/'factor_run_registry.csv')
    required=[]
    for year in [2020,2023,2026]:
        for arm in ['AG_RAW','AG_RES']:
            found=registry.loc[registry.arm.eq(arm)&registry.seed.eq(17)&registry.year.eq(year)]
            if len(found)!=1 or found.iloc[0].status!='FACTOR_OOF_YEAR_COMPLETE':raise RuntimeError(f'Counterfactual requires complete original {arm}/{year}/17')
            required.append((arm,year,str(found.iloc[0].directory)))
    root=R/'generation_counterfactual_tests';root.mkdir(parents=True,exist_ok=True);preparation_lock=threading.Lock()
    def execute(args):
        arm,year,baseline=args;dest=root/arm/str(year);dest.mkdir(parents=True,exist_ok=True)
        # Existing incomplete/resumed segments remain auditable before any overwrite.
        for p in (B/'counterfactual_generators'/'alphagen_runs'/arm/str(year)/'17').glob('*/RESOURCES.json'):
            history=dest/'attempt_history'/str(time.time_ns());history.mkdir(parents=True,exist_ok=True)
            (history/'prior_RESOURCES.json').write_bytes(p.read_bytes())
            (history/'source.json').write_text(json.dumps(identity(p)))
        log=dest/'worker.log';env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
        with log.open('a') as out:
            with preparation_lock:
                runbase=B/'counterfactual_generators'/'alphagen_runs'/arm/str(year)/'17'
                previous={str(p):p.stat().st_mtime_ns for p in runbase.glob('*/RESOLVED_PPO.json')}
                command=[sys.executable,'-u','-m','core.round26_1.generation_counterfactual_worker','--arm',arm,'--year',str(year),'--baseline',baseline]
                process=subprocess.Popen(command,cwd=C,stdout=out,stderr=subprocess.STDOUT,env=env)
                while process.poll() is None:
                    if any(str(p) not in previous or p.stat().st_mtime_ns>previous[str(p)] for p in runbase.glob('*/RESOLVED_PPO.json')):break
                    time.sleep(1)
            returncode=process.wait()
            if returncode:raise subprocess.CalledProcessError(returncode,command)
        result=json.loads((dest/'RESULT.json').read_text())
        assert result['status']=='PASS_FULL_ORIGINAL_PPO_UNAVAILABLE_TARGET_COUNTERFACTUAL'
        return result
    with ThreadPoolExecutor(max_workers=jobs) as pool:results=list(pool.map(execute,required))
    (root/'STATUS.json').write_text(json.dumps({'status':'PASS_SIX_COMPLETE_ORIGINAL_PPO_COUNTERFACTUALS','cases':results,'diagnostic_only':True,'economic_backtests':0,'market_revision_retraining':'NOT_RUN; historical input prefix invariance audited separately'},ensure_ascii=False,indent=2))
    return results
