"""Current D5 delivery only: fixed seed annual OOF, then original backend trials."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
import json,shutil
import pandas as pd
from core.round26_1.master_provider import R

def worker(task):
    from core.round26_1.master_fit import configure,run
    configure()
    store,arm,seed,year=task
    try:
        path=run(store,arm,seed,year)
    except BaseException:
        __import__('traceback').print_exc()
        raise
    return {'arm':arm,'seed':seed,'year':year,'path':str(path)}

def run(store,seed=17,jobs=2,fit_only=False):
    assert seed in [17,42,73] and 1<=jobs<=2
    # Each worker is a one-task lifetime process: interop settings and RSS isolated.
    tasks=[(str(store),arm,seed,year) for year in range(2015,2027) for arm in ['M0','M1']]
    records=[]
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=jobs,mp_context=context,max_tasks_per_child=1) as pool:
        futures={pool.submit(worker,t):t for t in tasks}
        for future in as_completed(futures):
            record=future.result();records.append(record)
            directory=R/'fits'/record['arm']/str(seed)/str(record['year']);directory.mkdir(parents=True,exist_ok=True)
            source=Path(record['path'])
            for name in ['MODEL_REGISTRY.json','NORMALIZER_AUDIT.json','PERMUTATION_REAL.json','TRAINING_DAYS.csv','RESOURCES.json','contract.json']:
                shutil.copy2(source/name,directory/name)
            shutil.copy2(source/'COMPLETE.json',directory/'SERVER_COMPLETE_MANIFEST.json')
            from core.research_integrity_v5.cache import identity
            (directory/'SERVER_ARTIFACTS.json').write_text(json.dumps({'directory':str(source),'private_training_panels_and_checkpoints_not_copied_to_repo':True,'artifacts':[identity(p) for p in sorted(source.iterdir()) if p.is_file()]},ensure_ascii=False,indent=2))
            pd.DataFrame(records).sort_values(['year','arm']).to_csv(R/f'FIT_PROGRESS_{seed}.csv',index=False)
            print('MASTER completed true OOF folds',seed,len(records),'/',len(tasks),flush=True)
    frame=pd.DataFrame(records).sort_values(['year','arm']);frame.to_csv(R/f'FIT_MANIFEST_{seed}.csv',index=False)
    from core.round26_1.master_backend import banks
    paths=banks(frame,seed)
    # Export derived scores only. Label annotations remain in private server fit artifacts.
    for path in paths.values():
        values=pd.read_parquet(path)
        assert values.phase.eq('OOF').all()
        columns=['trade_date','instrument_id','z_MASTER','factor_fit_cutoff','phase']
        temp=path.with_suffix('.partial.parquet');values[columns].to_parquet(temp,index=False);temp.replace(path)
    if fit_only:return paths
    from core.round26_1.master_trials import run as trials
    return trials(seed,paths)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--store',required=True);p.add_argument('--seed',type=int,default=17);p.add_argument('--jobs',type=int,default=2);p.add_argument('--fit-only',action='store_true');args=p.parse_args()
    run(args.store,args.seed,args.jobs,args.fit_only)
