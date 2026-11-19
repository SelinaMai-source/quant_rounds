"""One original full-epoch MASTER fit per annual genuinely held-out factor fold."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import torch
from core.round26_1.master_provider import C,B,R
from core.round26_1.master_model import create,legal_loss,EXPECTED,VENDOR
from core.round26_1.master_store import GraphStore
from core.research_integrity_v5.cache import identity,transaction,fingerprint,sha
from core.research_integrity_v5.resources import measured

def configure():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.deterministic=True

def guard():
    import psutil,os
    if psutil.Process(os.getpid()).memory_info().rss>16*1024**3:
        raise MemoryError('Own MASTER learner RSS exceeded frozen16GiB')

def _compute_normalizer(store,years,dest):
    from qlib.data.dataset.processor import RobustZScoreNorm
    indices=store.indices(f'{years[0]}-01-01',f'{years[1]}-01-01')
    values=[];dates=[]
    for i in indices:
        x=store.current(i);values.append(x);dates.extend([store.dates[i]]*len(x))
    frame=pd.DataFrame(np.concatenate(values),index=pd.MultiIndex.from_arrays([pd.DatetimeIndex(dates),np.arange(len(dates))],names=['datetime','instrument']))
    del values
    processor=RobustZScoreNorm(f'{years[0]}-01-01',f'{years[1]-1}-12-31',clip_outlier=True)
    processor.fit(frame)
    np.savez(dest/'NORMALIZER.npz',mean=processor.mean_train,std=processor.std_train)
    # Optimized tensor path must match original processor on actual training rows.
    sample=frame.iloc[::max(1,len(frame)//1024)].copy()
    expected=processor(sample.copy()).fillna(0).to_numpy('float32')
    actual=np.clip((sample.to_numpy()-processor.mean_train)/processor.std_train,-3,3)
    actual[np.isnan(actual)]=0
    assert np.array_equal(actual.astype('float32'),expected,equal_nan=True)
    audit={'fit_start':str(frame.index.get_level_values('datetime').min()),'fit_end':str(frame.index.get_level_values('datetime').max()),'rows':len(frame),'features':221,'missing_stats':int(np.isnan(processor.mean_train).sum()),'original_processor_actual_row_bitexact':True}
    (dest/'NORMALIZER_AUDIT.json').write_text(json.dumps(audit,indent=2))
    return processor.mean_train,processor.std_train

def fit_normalizer(store,years,dest):
    import fcntl,shutil
    c=dict(code=[identity(__file__)],third_party='original Qlib RobustZScoreNorm',dirty_diff='content identity',environment={'numpy':np.__version__,'pandas':pd.__version__},parameters={'training_years':years,'same_persisted_processor_for_all_arms_seeds':True},seed=None,features='221 original inputs',universe='all train graph nodes, no label filtering',calibration='median/MAD+EPS times1.4826 clip3 Fillna0',data=__import__('core.round26_1.master_bindings',fromlist=['validate_store']).validate_store(store.path),label='not read',maturity='not read',windows=years,execution='normalizer only',upstream=[identity(R/'TARGET_COMPATIBILITY_FREEZE.json')])
    fp=fingerprint(c);root=B/'MASTER_normalizers'/str(years[0])
    root.mkdir(parents=True,exist_ok=True)
    with (root/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with transaction(root,c) as (work,cached):
            if not cached:_compute_normalizer(store,years,work)
        source=root/fp
        for name in ['NORMALIZER.npz','NORMALIZER_AUDIT.json']:
            shutil.copy2(source/name,dest/name)
        fitted=np.load(source/'NORMALIZER.npz')
        return fitted['mean'],fitted['std']

def contract(store_path,arm,seed,year,mutation):
    import importlib.metadata,inspect
    from qlib.data.dataset.processor import RobustZScoreNorm
    own=['master_bindings.py','master_fit.py','master_model.py','master_store.py','master_graphs.py','master_provider.py','master_features.py','master_market.py']
    return dict(code=[identity(C/'core/round26_1'/n) for n in own]+[identity(C/'core/research_integrity_v5'/n) for n in ['cache.py','resources.py']],third_party={'author':[identity(VENDOR/n) for n in EXPECTED],'processor':identity(inspect.getfile(RobustZScoreNorm))},dirty_diff='complete imported source content identities',environment={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','pyqlib','zyquant']},parameters={'arm':arm,'year':year,'seed':seed,'epochs':1,'lr':1e-5,'threshold':.95,'gradient_clip_value':3.,'hidden':256,'temporal_heads':4,'spatial_heads':2,'dropout':.5,'beta':5,'CPUthreads':2,'interop':2,'mutation':mutation},seed=seed,features='original158+actual market63, train-only original RobustZScoreNorm',universe='frozen D-known all target+lagged top1000 context graph',calibration='one scalar canonicalTen target, no calibration',data=__import__('core.round26_1.master_bindings',fromlist=['validate_store']).validate_store(store_path),label='canonical label_exit_rule_net, mask only after all-node forward; drop2.5percent tails, p0 keepall',maturity=f'train before{year-1}-01-01, valid before{year}-01-01',windows={'train':[year-3,year-2],'valid':year-1,'heldout':year,'sequence':8},execution={'baseline':'v5 immutable; backend/native execution separate','costs':[12.3,1.,.1,5.],'capital':300000,'source':identity(C/'core/local_executor.py')},upstream=[identity(R/'CONTRACT.yaml'),identity(R/'TARGET_COMPATIBILITY_FREEZE.json')])

def run(store_path,arm,seed,year,mutation='none',namespace=None):
    c=contract(store_path,arm,seed,year,mutation);fp=fingerprint(c)
    root=Path(namespace) if namespace else B/'MASTER_fits'/arm/str(seed)/str(year)
    with transaction(root,c) as (dest,cached):
        if not cached:
            with measured(dest/'RESOURCES.json'):
                store=GraphStore(store_path)
                mean,std=fit_normalizer(store,(year-3,year-1),dest);guard()
                model=create(arm,seed);optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
                q=store.q;targets=q.label_exit_rule_net.to_numpy('float32').copy()
                mature=(pd.to_datetime(q.label_available_at)<pd.Timestamp(f'{year-1}-01-01')).to_numpy()
                unknown=~mature
                if mutation=='unavailable_plus1e6':targets[unknown]=np.nan_to_num(targets[unknown],nan=0.)+1e6
                elif mutation=='unavailable_delete':targets[unknown]=np.nan
                elif mutation!='none':raise ValueError(mutation)
                train=store.indices(f'{year-3}-01-01',f'{year-1}-01-01')
                order=np.arange(len(train));np.random.shuffle(order)
                model.train();logs=[]
                for step,local in enumerate(order):
                    index=int(train[local]);values,qi=store.batch(index,mean,std)
                    labels=np.full(len(qi),np.nan,dtype='float32');legal=np.zeros(len(qi),dtype=bool)
                    target=qi>=0;labels[target]=targets[qi[target]];legal[target]=mature[qi[target]]
                    prediction=model(torch.from_numpy(values))
                    loss,audit=legal_loss(prediction,torch.from_numpy(labels),torch.from_numpy(legal))
                    optimizer.zero_grad()
                    if loss is not None:
                        loss.backward();torch.nn.utils.clip_grad_value_(model.parameters(),3.);optimizer.step()
                    logs.append({'date':str(store.dates[index].date()),'step':step,'nodes':len(qi),'loss':None if loss is None else float(loss.detach()),**audit})
                    if step%25==0:
                        guard();print('MASTER',arm,seed,year,'full-epoch day',step,'/',len(order),'loss',logs[-1]['loss'],flush=True)
                pd.DataFrame(logs).to_csv(dest/'TRAINING_DAYS.csv',index=False)
                meanloss=float(pd.DataFrame(logs).loss.mean())
                torch.save(model.state_dict(),dest/'model.pt')
                model.eval();frames=[];permutations=[]
                for phase,start,end in [('valid',f'{year-1}-01-01',f'{year}-01-01'),('OOF',f'{year}-01-01',f'{year+1}-01-01')]:
                    indices=store.indices(start,end)
                    for j,index in enumerate(indices):
                        values,qi=store.batch(index,mean,std);tensor=torch.from_numpy(values)
                        with torch.no_grad():pred=model(tensor).numpy()
                        if phase=='OOF' and j in {0,len(indices)//2,len(indices)-1}:
                            perm=torch.randperm(len(tensor))
                            with torch.no_grad():other=model(tensor[perm])[torch.argsort(perm)].numpy()
                            assert np.allclose(pred,other,atol=1e-5,rtol=1e-5)
                            permutations.append({'date':str(store.dates[index].date()),'nodes':len(qi),'max_abs_error':float(np.abs(pred-other).max()),'atol':1e-5,'rtol':1e-5})
                        select=qi>=0
                        rows=q.iloc[qi[select]][['trade_date','instrument_id','label_exit_rule_net','label_available_at']].copy()
                        rows['z_MASTER']=pred[select];rows['phase']=phase
                        rows['factor_fit_cutoff']=pd.Timestamp(f'{year}-01-01')
                        if phase=='valid':rows['metric_label_legal']=pd.to_datetime(rows.label_available_at)<pd.Timestamp(f'{year}-01-01')
                        frames.append(rows)
                predictions=pd.concat(frames,ignore_index=True)
                predictions.loc[predictions.phase.eq('OOF')].to_parquet(dest/'factor_predictions.parquet',index=False)
                predictions.loc[predictions.phase.eq('valid')].to_parquet(dest/'validation_diagnostic.parquet',index=False)
                (dest/'PERMUTATION_REAL.json').write_text(json.dumps(permutations,indent=2))
                trainq=q.loc[q.trade_date.ge(f'{year-3}-01-01')&q.trade_date.lt(f'{year-1}-01-01')]
                registry={'status':'FACTOR_OOF_YEAR_COMPLETE','arm':arm,'seed':seed,'year':year,'epochs':1,'all_training_dates':len(train),'gradient_dates':int(pd.DataFrame(logs).used.gt(0).sum()),'mean_train_loss':meanloss,'checkpoint_reason':'THRESHOLD_REACHED' if meanloss<=.95 else 'CAP_REACHED','checkpoint_sha256':sha(dest/'model.pt'),'normalizer_sha256':sha(dest/'NORMALIZER.npz'),'fit_cutoff':f'{year}-01-01','max_training_label_available_at':str(trainq.loc[pd.to_datetime(trainq.label_available_at).lt(f'{year-1}-01-01'),'label_available_at'].max()),'unavailable_training_target_interventions':int((~(pd.to_datetime(trainq.label_available_at)<pd.Timestamp(f'{year-1}-01-01'))).sum()) if mutation!='none' else 0,'mutation':mutation,'model_shape_parameters':775041,'train_only_fit':True,'historical_OOF_not_training_fitted_values':True}
                (dest/'MODEL_REGISTRY.json').write_text(json.dumps(registry,indent=2))
    print('MASTER fold complete',arm,seed,year,fp,'cache',cached,flush=True)
    return root/fp
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--store',required=True);p.add_argument('--arm',choices=['M0','M1'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--year',type=int,required=True);args=p.parse_args()
    configure();run(args.store,args.arm,args.seed,args.year)
