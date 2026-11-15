"""D4 full public reference, fixed seed paths and actual original-engine runs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os,subprocess,sys,time
import pandas as pd
import psutil
from core.round23_7.tra_reference import C,R,B
from core.round23_7.tra_runtime import run_job,save_registry
from core.research_integrity_v5.cache import identity,sha

def controllers():
    found=[]
    for process in psutil.process_iter(['pid','uids','cmdline']):
        try:
            if process.pid==os.getpid() or process.uids().real!=os.getuid():continue
            args=process.cmdline()
            if 'core.round23_7.tra_pipeline' in args:found.append(process.pid)
            elif any(Path(v).name=='continue_round23_7_round26_1.py' for v in args) and '--deliverable' in args and args[args.index('--deliverable')+1] in ['D4','all']:found.append(process.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    return found

def reference_processes():
    found=[]
    for process in psutil.process_iter(['pid','uids','cmdline']):
        try:
            if process.uids().real==os.getuid() and 'core.round23_7.tra_reference' in process.cmdline():found.append(process.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    return found

def complete_reference():
    active=reference_processes()
    if active:
        assert len(active)==1,'Do not run duplicate full reference learners'
        while reference_processes():
            live=sorted((B/'TRA_reference').glob('*/*/LIVE.json'),key=lambda p:p.stat().st_mtime)
            try:state=json.loads(live[-1].read_text()) if live else {'state':'reference data preparation'}
            except (OSError,json.JSONDecodeError):state={'state':'atomic stage transition; next observation pending'}
            print('D4 attached existing complete-budget author reference',active,state,flush=True);time.sleep(55)
        status=json.loads((R/'REFERENCE_STATUS.json').read_text()) if (R/'REFERENCE_STATUS.json').exists() else {}
        assert status.get('status')=='REFERENCE_RUN_COMPLETE','Existing reference failed; retain actual logs and fix the concrete failure before dependent runs'
    else:
        from core.round23_7.tra_reference import run_reference
        run_reference()
    from core.round23_7.tra_reference_report import run as benchmark
    metrics=benchmark()
    status=json.loads((R/'STATUS.json').read_text())
    status.update(status='FULL_REFERENCE_COMPLETE_TARGET_RUNNING',REFERENCE_RUN_COMPLETE=True,completed_full_reference_phases=2)
    (R/'STATUS.json').write_text(json.dumps(status,indent=2))
    report=R/'REPORT.md';text=report.read_text();marker='\n<!-- FULL_PUBLIC_REFERENCE_COMPLETE -->'
    report.write_text(text.split(marker)[0]+marker+'\n作者公开两阶段均已按原默认停止完成，原基准函数实际结果见reference/original_benchmark_results.csv。目标研究仍以STATUS分项为准。\n\n```json\n'+json.dumps(metrics,indent=2)+'\n```\n')

def fold_models(data,year,seed,jobs):
    rows=[]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures=[pool.submit(run_job,head,year,seed,'init',data) for head in ['five','ten']]
        for future in as_completed(futures):rows.append(future.result())
    registry=save_registry(rows)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures=[]
        for head in ['five','ten']:
            start=next(row for row in rows if row['head']==head)
            if start['status']!='FULL_TARGET_FIT_COMPLETE':continue
            for phase in ['T0','T1']:futures.append(pool.submit(run_job,head,year,seed,phase,data,init_state=Path(start['path'])/'model.bin'))
        for future in as_completed(futures):rows.append(future.result())
    return save_registry(rows)

def merge_runs(new):
    p=R/'run_manifest.csv';old=pd.read_csv(p).to_dict('records') if p.exists() else []
    rows={x['code']:x for x in old+new};pd.DataFrame(list(rows.values())).to_csv(p,index=False)

def native_trials(seed):
    from core.round23_7.tra_backtest import baseline_sources,materialize,execute
    _,base,formal=baseline_sources();rows=[];formal_rows=[]
    for name,scores,dest in [('TRA_A0',base,rows),('TRA_FORMAL_V5',formal,formal_rows)]:
        for bps in [0,10,20]:
            scenario='S'+str(bps//10);code=name+('' if bps==0 else '_'+scenario)
            path,fp,hit=execute(code,scores,bps=bps)
            dest.append({'method':name,'scenario':scenario,'code':code,'path':str(path),'fingerprint':fp,'cache_hit':hit})
    folder=R/'formal_baseline';folder.mkdir(parents=True,exist_ok=True);pd.DataFrame(formal_rows).to_csv(folder/'run_manifest.csv',index=False)
    registry=pd.read_csv(R/'target_run_registry.csv')
    selected=registry.loc[registry.namespace.eq('original')&registry.seed.eq(seed)]
    for model in ['T0','T1']:
        head_frames={};upstream=[]
        for head in ['five','ten']:
            paths=selected.loc[selected['head'].eq(head)&selected.phase.eq(model)].sort_values('year')
            assert len(paths)==9 and paths.status.eq('FULL_TARGET_FIT_COMPLETE').all(),'No fallback disguising a failed full model/year'
            parts=[]
            for row in paths.itertuples(index=False):
                p=Path(row.path)/'forecast.parquet';parts.append(pd.read_parquet(p)[['trade_date','instrument_id','score']]);upstream.append(identity(p))
            head_frames[head]=pd.concat(parts,ignore_index=True).rename(columns={'trade_date':'signal_date'})
        scores,fallback=materialize(head_frames['five'],head_frames['ten'],base)
        folder=R/'scores';folder.mkdir(parents=True,exist_ok=True);name=f'TRA_{model}_{seed}'
        scores.to_parquet(folder/f'{name}.parquet',index=False);fallback.to_csv(folder/f'{name}_whole_day_fallback.csv',index=False)
        for bps in [0,10,20]:
            scenario='S'+str(bps//10);code=name+('' if bps==0 else '_'+scenario)
            path,fp,hit=execute(code,scores,bps=bps,upstream=upstream)
            rows.append({'method':name,'scenario':scenario,'code':code,'path':str(path),'fingerprint':fp,'cache_hit':hit});print('TRA actual native trial',code,flush=True)
    merge_runs(rows)

def run(jobs=2,publish=False):
    assert jobs in [1,2]
    duplicates=controllers();assert not duplicates,f'Existing D4 controller PIDs{duplicates}; do not duplicate learners'
    complete_reference()
    if publish:
        from core.research_integrity_v5.publish import publish as push
        push('round23_7','D4: complete full locked author public reference and original benchmark')
    from core.round23_7.tra_data import prepare
    data=prepare()
    critical=[C/'core/round23_7'/name for name in ['tra_reference.py','tra_data.py','tra_causal.py','tra_fit.py']]+[C/'core/research_integrity_v5/cache.py',C/'core/research_integrity_v5/resources.py',C/'core/local_executor.py',C/'core/round7/contract.py']
    lock={str(p):sha(p) for p in critical};(R/'ACTIVE_LEARNING_CODE_LOCK.json').write_text(json.dumps(lock,indent=2))
    for seed in [17,42,73]:
        for year in range(2018,2027):
            assert all(sha(name)==value for name,value in lock.items()),'Learning code changed while fixed experiment active; do not mix versions'
            registry=fold_models(data,year,seed,jobs)
        current=registry.loc[registry.namespace.eq('original')&registry.seed.eq(seed)]
        if len(current)!=54 or not current.status.eq('FULL_TARGET_FIT_COMPLETE').all():
            (R/'STATUS.json').write_text(json.dumps({'status':'PARTIAL_REAL_TARGET_FAILURE','IMPLEMENTED':True,'UNIT_TESTED':True,'REFERENCE_RUN_COMPLETE':True,'TARGET_OOF_COMPLETE':False,'BACKTEST_COMPLETE':False,'EVIDENCE_COMPLETE':False,'external_blocker':False,'failed':current.loc[current.status.ne('FULL_TARGET_FIT_COMPLETE')].to_dict('records'),'later_seeds_pending':'fix actual implementation/numerical failure; never call NOT_IMPLEMENTED an external blocker'},indent=2))
            raise RuntimeError('Concrete target failures retained; independent head/year attempts completed, root must resolve failure')
        if seed==17:
            from core.round23_7.tra_counterfactual import run as counterfactual
            counterfactual(data,jobs=jobs)
            cf=json.loads((R/'counterfactual_tests/STATUS.json').read_text())
            assert cf['status']=='PASS_SIX_REAL_FULL_BUDGET_CASES','Actual time counterfactuals failed; retain evidence and repair before economic attribution'
        native_trials(seed)
        # Fresh report subprocess avoids retaining obsolete audit/report modules.
        subprocess.run([sys.executable,'-B','-c',f'from core.round23_7.tra_evidence import report; report({seed})'],cwd=C,check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        if publish:
            from core.research_integrity_v5.publish import publish as push
            push('round23_7',f'D4: complete real TRA seed{seed} full target fits, causal tests and native T0/T1 risk evidence')
    return json.loads((R/'STATUS.json').read_text())

if __name__=='__main__':run()
