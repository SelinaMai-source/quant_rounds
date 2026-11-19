"""Actual full-cap cold fits on unavailable targets; no fixture-only causality claim."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from core.round26_1.master_provider import B,R
from core.research_integrity_v5.cache import sha,identity

def nn_worker(task):
    from core.round26_1.master_fit import configure,run
    configure()
    store,arm,year,reference=task
    dest=run(store,arm,17,year,'unavailable_plus1e6',namespace=B/'MASTER_cold_fits'/arm/str(year))
    reference=Path(reference)
    first=torch.load(reference/'model.pt',map_location='cpu',weights_only=True)
    second=torch.load(dest/'model.pt',map_location='cpu',weights_only=True)
    assert first.keys()==second.keys()
    assert all(torch.equal(first[k],second[k]) for k in first),'Cold NN weights changed under unavailable target mutation'
    a=pd.read_parquet(reference/'factor_predictions.parquet');b=pd.read_parquet(dest/'factor_predictions.parquet')
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    registry=json.loads((dest/'MODEL_REGISTRY.json').read_text())
    assert registry['unavailable_training_target_interventions']>0
    assert sha(reference/'NORMALIZER.npz')==sha(dest/'NORMALIZER.npz')
    result={'status':'PASS_FULL_CAP_COLD_FIT','arm':arm,'year':year,'seed':17,'unavailable_training_target_interventions':registry['unavailable_training_target_interventions'],'all_state_tensors_bitexact':True,'OOF_predictions_bitexact':True,'same_persisted_normalizer_bytes':True,'original':str(reference),'cold':str(dest),'original_registry':identity(reference/'MODEL_REGISTRY.json'),'cold_registry':identity(dest/'MODEL_REGISTRY.json'),'resources':json.loads((dest/'RESOURCES.json').read_text())}
    out=R/'counterfactuals';out.mkdir(exist_ok=True);(out/f'NN_{arm}_{year}.json').write_text(json.dumps(result,indent=2));return result

def run_nn(store,jobs=2):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    rows=pd.read_csv(R/'FIT_MANIFEST_17.csv')
    tasks=[]
    for year in [2015,2021,2026]:
        for arm in ['M0','M1']:
            matches=rows.loc[rows.arm.eq(arm)&rows.year.eq(year)];assert len(matches)==1
            tasks.append((str(store),arm,year,matches.iloc[0].path))
    results=[]
    with ProcessPoolExecutor(max_workers=jobs,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
        for result in pool.map(nn_worker,tasks):
            results.append(result);print('MASTER actual full cold fit',result['arm'],result['year'],'PASS',flush=True)
    (R/'NN_COLD_COUNTERFACTUALS.json').write_text(json.dumps({'status':'PASS','cases':results},indent=2));return results

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--store',required=True);args=p.parse_args();run_nn(args.store)
