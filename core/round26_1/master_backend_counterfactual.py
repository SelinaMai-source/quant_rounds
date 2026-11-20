"""Full original XGB cold fits after real unavailable-target source interventions."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from core.round26_1.master_provider import C,B,R,KEY
from core.research_integrity_v5.cache import identity,sha

def fit_worker(task):
    arm,year,head_name,variant,original=task
    from core.round7.contract import FIVE_HEAD,TEN_HEAD
    from core.round26_1.master_backend import fit_head
    head=FIVE_HEAD if head_name=='five' else TEN_HEAD
    bank=R/'factor_banks'/arm/'17/OOF_factor.parquet'
    training=C/f'result/round22_3/data/{head.name}_training.parquet'
    reader=pd.read_parquet;interventions=[]
    def changed_reader(path,*args,**kwargs):
        frame=reader(path,*args,**kwargs)
        if isinstance(path,(str,Path)) and Path(path).resolve()==training.resolve():
            frame=frame.copy();dates=pd.to_datetime(frame.trade_date);available=pd.to_datetime(frame[head.label_end_column])+pd.Timedelta(hours=15)
            window=dates.ge(f'{year-3}-01-01')&dates.lt(f'{year}-01-01')
            unknown=window&~available.lt(pd.Timestamp(f'{year}-01-01'))
            assert unknown.any(),'Diagnostic intervention must be nonempty'
            interventions.append({'rows':int(unknown.sum()),'dates':int(frame.loc[unknown,'trade_date'].nunique()),'earliest_signal':str(frame.loc[unknown,'trade_date'].min()),'latest_signal':str(frame.loc[unknown,'trade_date'].max()),'stage':'actual source frame before maturity filtering'})
            if variant=='perturb':frame.loc[unknown,head.target]=np.nan_to_num(frame.loc[unknown,head.target].to_numpy(),nan=0.)+1e6
            elif variant=='delete':frame=frame.loc[~unknown].copy()
            else:raise ValueError(variant)
        return frame
    driver=sha(__file__)
    pd.read_parquet=changed_reader
    try:
        directory=fit_head(f'CF_{arm}_{variant}_{driver[:12]}',17,head,year,bank,'ADD',namespace=B/'MASTER_backend_CF'/arm/head_name/str(year)/variant)
    finally:pd.read_parquet=reader
    assert interventions,'No actual source intervention executed'
    original=Path(original)
    old=json.loads((original/'MODEL_REGISTRY.json').read_text());new=json.loads((directory/'MODEL_REGISTRY.json').read_text())
    for field in ['training_keys','prediction_keys','training_original','prediction_original','training_actual','prediction_actual','feature_order','tree_seeds','XGB_params','eval_metric','rank_bins','fit_cutoff','max_training_label_available_at']:
        assert old['parameters'][field]==new['parameters'][field],field
    pd.testing.assert_frame_equal(pd.read_parquet(original/'predictions.parquet'),pd.read_parquet(directory/'predictions.parquet'),check_exact=True)
    assert old['models']==new['models'],'Full original XGB model bytes differ after unavailable label intervention'
    result={'status':'PASS_FULL_ORIGINAL_XGB_COLD_FIT','arm':arm,'seed':17,'head':head_name,'year':year,'variant':variant,'source_interventions':interventions,'full_booster_fits':new['actual_booster_fits'],'all_training_input_and_prediction_hashes_exact':True,'all_booster_model_hashes_exact':True,'predictions_bitexact':True,'original':str(original),'cold':str(directory),'counterfactual_driver':identity(__file__),'original_registry':identity(original/'MODEL_REGISTRY.json'),'cold_registry':identity(directory/'MODEL_REGISTRY.json'),'resources':json.loads((directory/'RESOURCES.json').read_text())}
    dest=R/'backend_counterfactuals';dest.mkdir(exist_ok=True);(dest/f'{arm}_{head_name}_{year}_{variant}.json').write_text(json.dumps(result,indent=2))
    return result

def native_replays(cases):
    from core.round26_1.factor_models import canonical_fuse
    from core.round26_1.master_native import execute
    models=pd.read_csv(R/'head_model_manifest.csv');runs=pd.read_csv(R/'run_manifest.csv')
    records=[];tables=['overlayed','trades','rejections','positions','sleeve_states','nav','cashflows','costs']
    for arm in ['M0','M1']:
        method=f'{arm}_17_BOTH_ADD';source=runs.loc[runs.code.eq(method)];assert len(source)==1
        original=Path(source.iloc[0].path)
        expected=pd.read_parquet(R/'scores'/(method+'.parquet'))
        for year in [2018,2022,2026]:
            for variant in ['perturb','delete']:
                heads={}
                for head in ['five','ten']:
                    rows=models.loc[models.arm.eq(arm)&models.seed.eq(17)&models['head'].eq(head)&models['mode'].eq('ADD')].sort_values('year')
                    assert set(rows.year)==set(range(2018,2027));parts=[]
                    for row in rows.itertuples(index=False):
                        path=Path(row.path)
                        if row.year==year:
                            match=[r for r in cases if r['arm']==arm and r['year']==year and r['head']==head and r['variant']==variant];assert len(match)==1
                            path=Path(match[0]['cold'])
                        parts.append(pd.read_parquet(path/'predictions.parquet')[KEY+['score']])
                    heads[head]=pd.concat(parts,ignore_index=True).rename(columns={'trade_date':'signal_date'})
                scores=canonical_fuse(heads['five'],heads['ten']);pd.testing.assert_frame_equal(scores,expected,check_exact=True)
                code=f'CF_{method}_{year}_{variant}';path,fp,hit=execute(code,scores,upstream=[identity(__file__),identity(R/'DIAGNOSTIC_INTERVENTIONS_FREEZE.json')])
                for name in tables:
                    pd.testing.assert_frame_equal(pd.read_parquet(original/(name+'.parquet')),pd.read_parquet(path/(name+'.parquet')),check_exact=True)
                records.append({'status':'PASS_ACTUAL_NATIVE_REPLAY_ALL8_TABLES_EXACT','arm':arm,'year':year,'variant':variant,'code':code,'path':str(path),'original':str(original),'fingerprint':fp,'cache_hit':hit,'exact_tables':tables})
                print('MASTER actual native causal replay',code,'PASS',flush=True)
    (R/'NATIVE_COUNTERFACTUAL_REPLAYS.json').write_text(json.dumps({'status':'PASS','cases':records},indent=2));return records

def run(jobs=2):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    models=pd.read_csv(R/'head_model_manifest.csv');tasks=[]
    for arm in ['M0','M1']:
        for year in [2018,2022,2026]:
            for head in ['five','ten']:
                source=models.loc[models.arm.eq(arm)&models.seed.eq(17)&models['head'].eq(head)&models.year.eq(year)&models['mode'].eq('ADD')];assert len(source)==1
                for variant in ['perturb','delete']:tasks.append((arm,year,head,variant,source.iloc[0].path))
    cases=[]
    with ProcessPoolExecutor(max_workers=jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(fit_worker,tasks):
            cases.append(result);print('MASTER full backend cold fit',result['arm'],result['head'],result['year'],result['variant'],'PASS',flush=True)
    assert len(cases)==24
    native=native_replays(cases)
    (R/'BACKEND_COLD_COUNTERFACTUALS.json').write_text(json.dumps({'status':'PASS','full_configurations':len(cases),'full_booster_fits':sum(r['full_booster_fits'] for r in cases),'cases':cases,'actual_native_replays':len(native)},indent=2));return cases
if __name__=='__main__':run()
