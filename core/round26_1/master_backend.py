"""Original baseline head factory with genuine annual MASTER OOF Add/Remove/Replace."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from xgboost import XGBRanker
from core.round26_1.master_provider import C,B,R,KEY
from core.research_integrity_v5.cache import identity,fingerprint,transaction,sha
from core.research_integrity_v5.resources import measured

def prepare(head,year,factor=None):
    from core.round26_1.factor_models import prepare as original_prepare
    train,predict=original_prepare(head,year,None)
    if factor is not None:
        bank=pd.read_parquet(factor,columns=KEY+['z_MASTER','factor_fit_cutoff'])
        assert not bank.duplicated(KEY).any()
        for frame in [train,predict]:
            frame['original_row']=np.arange(len(frame))
        train=train.merge(bank,on=KEY,how='left',validate='one_to_one').sort_values('original_row')
        predict=predict.merge(bank,on=KEY,how='left',validate='one_to_one').sort_values('original_row')
        for frame in [train,predict]:
            assert np.isfinite(frame.z_MASTER).all(),'Missing real annual OOF; never replace with fitted values'
            assert (pd.to_datetime(frame.factor_fit_cutoff)<=pd.to_datetime(frame.trade_date.dt.year.astype(str)+'-01-01')).all()
    return train,predict

def fit_head(arm,seed,head,year,factor=None,mode='BASELINE',remove=None,namespace=None):
    from core.round7.contract import SEEDS,model_parameters
    from core.stock_readonly import ensure_stock_src_on_path
    ensure_stock_src_on_path()
    from first_board_hermes.baseline import daily_relevance
    import first_board_hermes.baseline as baseline_module
    import importlib.metadata,inspect
    if mode not in ['BASELINE','ADD','REMOVE','REPLACE']:raise ValueError(mode)
    if mode!='BASELINE' and factor is None:raise ValueError('Real OOF bank required')
    a,p=prepare(head,year,factor)
    features=[f for f in head.features if mode not in ['REMOVE','REPLACE'] or f!=remove]
    if mode in ['ADD','REPLACE']:features.append('z_MASTER')
    params={'arm':arm,'factor_seed':seed,'head':head.name,'year':year,'mode':mode,'removed':remove,'feature_order':features,'tree_seeds':list(SEEDS),'XGB_params':model_parameters(17),'eval_metric':head.eval_metric,'rank_bins':head.rank_n_bins,'fit_cutoff':f'{year}-01-01','max_training_label_available_at':str(a.label_available_at.max())}
    assert pd.Timestamp(params['max_training_label_available_at'])<pd.Timestamp(params['fit_cutoff'])
    def digest(frame,cols):return hashlib.sha256(pd.util.hash_pandas_object(frame[cols],index=False).values.tobytes()).hexdigest()
    params.update(training_keys=digest(a,KEY),prediction_keys=digest(p,KEY),training_original=digest(a,KEY+list(head.features)+[head.target,'label_available_at']),prediction_original=digest(p,KEY+list(head.features)),training_actual=digest(a,KEY+features+[head.target,'label_available_at']),prediction_actual=digest(p,KEY+features))
    sources=[Path(__file__),C/'core/local_executor.py',C/'core/round26_1/factor_models.py',C/'core/round26_1/market.py',C/'core/round7/contract.py',C/'core/stock_readonly.py',C/'core/research_integrity_v5/cache.py',C/'core/research_integrity_v5/resources.py',Path(inspect.getfile(baseline_module))]
    c=dict(code=[identity(s) for s in sources],third_party={'XGBoost':importlib.metadata.version('xgboost')},dirty_diff='full imported source identities',environment={n:importlib.metadata.version(n) for n in ['xgboost','numpy','pandas','zyquant']},parameters=params,seed=17,features=features,universe='unchanged baseline feature-complete candidate/training rows',calibration='original daily_relevance and five-seed daily percentile ensemble',data=[identity(C/f'result/round22_3/data/{head.name}_{kind}.parquet') for kind in ['training','candidate']]+([identity(factor)] if factor else []),label=head.target,maturity=head.label_end_column+' at15:00',windows=f'prior3y to{year}-01-01, heldout{year}',execution='baseline90Ten10Five, original native account downstream, no own-fit NN values',upstream=[identity(R/'CONTRACT.yaml'),identity(R/'TARGET_COMPATIBILITY_FREEZE.json')])
    fp=fingerprint(c);root=Path(namespace) if namespace else B/'MASTER_backend'/arm/str(seed)/head.name/mode/str(year)
    with transaction(root,c) as (dest,cached):
        if not cached:
            with measured(dest/'RESOURCES.json'):
                y=daily_relevance(a,target=head.target,n_bins=head.rank_n_bins).to_numpy('int32')
                ranks=[];models=[];out=p[KEY].copy()
                for tree_seed in SEEDS:
                    cfg=model_parameters(tree_seed);cfg['eval_metric']=head.eval_metric
                    assert cfg['subsample']==cfg['colsample_bytree']==1
                    model=XGBRanker(**cfg);model.fit(a[features].to_numpy('float32'),y,qid=a.trade_date.factorize(sort=True)[0],verbose=False)
                    target=dest/f'xgb_{tree_seed}.json';model.save_model(target)
                    raw=pd.Series(model.predict(p[features].to_numpy('float32')),index=p.index)
                    ranks.append(raw.groupby(p.trade_date).rank(pct=True));models.append({'seed':tree_seed,'sha256':sha(target)})
                out['score']=pd.concat(ranks,axis=1).mean(axis=1)
                out['fit_cutoff']=pd.Timestamp(f'{year}-01-01');out['max_training_label_available_at']=a.label_available_at.max()
                out.to_parquet(dest/'predictions.parquet',index=False)
                (dest/'MODEL_REGISTRY.json').write_text(json.dumps({'parameters':params,'models':models,'actual_booster_fits':len(models)},indent=2))
    return root/fp

def banks(registry,seed):
    paths={}
    for arm in ['M0','M1']:
        rows=registry.loc[registry.arm.eq(arm)&registry.seed.eq(seed)]
        assert set(rows.year)==set(range(2015,2027))
        parts=[];models=[]
        for row in rows.sort_values('year').itertuples(index=False):
            path=Path(row.path);state=json.loads((path/'MODEL_REGISTRY.json').read_text())
            assert state['status']=='FACTOR_OOF_YEAR_COMPLETE'
            p=pd.read_parquet(path/'factor_predictions.parquet');assert p.trade_date.dt.year.eq(row.year).all()
            assert pd.to_datetime(p.factor_fit_cutoff).eq(pd.Timestamp(f'{row.year}-01-01')).all()
            parts.append(p);models.append({'year':row.year,'checkpoint_sha256':state['checkpoint_sha256'],'prediction':identity(path/'factor_predictions.parquet')})
        bank=pd.concat(parts,ignore_index=True).sort_values(KEY)
        assert not bank.duplicated(KEY).any() and np.isfinite(bank.z_MASTER).all()
        dest=R/'factor_banks'/arm/str(seed);dest.mkdir(parents=True,exist_ok=True)
        path=dest/'OOF_factor.parquet';bank.to_parquet(path,index=False)
        (dest/'PROVENANCE.json').write_text(json.dumps({'models':models,'status':'TRUE_ANNUAL_OOF_ONLY','in_sample_NN_values_used':False},indent=2))
        paths[arm]=path
    return paths

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--baseline',action='store_true');args=p.parse_args()
    if args.baseline:
        from core.round7.contract import FIVE_HEAD,TEN_HEAD
        records=[]
        for year in range(2018,2027):
            for head in [FIVE_HEAD,TEN_HEAD]:
                path=fit_head('NO_FACTOR','CONTROL',head,year)
                records.append({'head':head.name,'year':year,'path':str(path)})
                out=R/'baseline_control';out.mkdir(exist_ok=True);pd.DataFrame(records).to_csv(out/'head_models.csv',index=False)
                print('MASTER genuine no-factor control',head.name,year,path,flush=True)
