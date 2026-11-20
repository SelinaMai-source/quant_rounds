"""One actual locked-author public CSI300 full-epoch reference and checkpoint diagnostic."""
from pathlib import Path
import json,inspect
import numpy as np
import pandas as pd
import torch
from core.round26_1.master_provider import C,B,R
from core.round26_1.master_model import upstream,VENDOR,EXPECTED
from core.round26_1.master_reference_data import ROOT,load,audit
from core.research_integrity_v5.cache import identity,transaction,fingerprint,sha
from core.research_integrity_v5.resources import measured

def finite_json(value):
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_json(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value

def original_model(seed,folder):
    return upstream().MASTERModel(d_feat=158,d_model=256,t_nhead=4,s_nhead=2,
        gate_input_start_index=158,gate_input_end_index=221,T_dropout_rate=.5,S_dropout_rate=.5,beta=5,
        n_epochs=1,lr=1e-5,GPU=None,seed=seed,train_stop_loss_thred=.95,save_path=str(folder),save_prefix='public_core')

def predict(model,view,dest,name):
    predictions,raw=model.predict(view)
    assert len(predictions)==len(view) and predictions.index.equals(view.get_index())
    assert np.isfinite(predictions.to_numpy()).all()
    frame=predictions.rename('score').to_frame();frame.to_parquet(dest/(name+'_predictions.parquet'))
    labels=view.labels();work=frame.copy();work['label']=labels
    daily=[]
    for day,g in work.groupby(level='datetime'):
        known=g.dropna(subset=['label'])
        ic=known.score.corr(known.label) if len(known)>=2 else np.nan
        ric=known.score.corr(known.label,method='spearman') if len(known)>=2 else np.nan
        daily.append({'date':str(day.date()),'all_inference_nodes':len(g),'finite_metric_labels':len(known),'score_std':float(known.score.std()) if len(known)>=2 else None,'label_std':float(known.label.std()) if len(known)>=2 else None,'IC':ic,'RIC':ric})
    daily=pd.DataFrame(daily);daily.to_csv(dest/(name+'_daily_IC.csv'),index=False)
    corrected={}
    for metric in ['IC','RIC']:
        finite=daily[metric].dropna();corrected[metric]=float(finite.mean())
        corrected[metric+'IR']=float(finite.mean()/finite.std(ddof=0)) if finite.std(ddof=0)>0 else None
    record={'original_source_metrics':finite_json(raw),'finite_label_metric_aggregation':finite_json(corrected),'scored_dates':int(daily.IC.notna().sum()),'all_forecast_dates':len(daily),'undefined_IC_dates':int(daily.IC.isna().sum()),'fewer_than_two_metric_labels_dates':int(daily.finite_metric_labels.lt(2).sum()),'constant_prediction_dates':int(daily.score_std.eq(0).sum()),'constant_label_dates':int(daily.label_std.eq(0).sum()),'prediction_nodes':len(view),'unknown_labels_retained_at_inference':int(np.isnan(labels).sum()),'metric_correction':'Only unscorable-label dates excluded from means; all forecast nodes/dates retained. Raw original np.mean metrics saved even if NaN.'}
    (dest/(name+'_METRICS.json')).write_text(json.dumps(record,indent=2));return record

def graph_tests(model,view,dest,name):
    dates=view.get_index().get_level_values('datetime');unique=pd.DatetimeIndex(dates.unique());records=[]
    model.model.eval()
    for day in unique[[0,len(unique)//2,len(unique)-1]]:
        select=np.flatnonzero(dates==day);batch=view[select].copy();x=torch.from_numpy(batch[:,:,:-1]).float()
        with torch.no_grad():first=model.model(x);perm=torch.randperm(len(x));other=model.model(x[perm])[torch.argsort(perm)]
        assert torch.allclose(first,other,atol=1e-5,rtol=1e-5)
        for kind in ['perturb_labels','delete_labels']:
            changed=batch.copy()
            if kind=='perturb_labels':changed[:,:,-1]=np.nan_to_num(changed[:,:,-1],nan=0.)+1e6
            else:changed[:,:,-1]=np.nan
            with torch.no_grad():actual=model.model(torch.from_numpy(changed[:,:,:-1]).float())
            assert torch.equal(first,actual)
        records.append({'date':str(day.date()),'nodes':len(x),'permutation_max_abs':float((first-other).abs().max()),'atol':1e-5,'rtol':1e-5,'actual_label_perturb_delete_predictions_bitexact':True})
    (dest/(name+'_GRAPH_TESTS.json')).write_text(json.dumps({'status':'PASS','cases':records},indent=2));return records

def run():
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.backends.mkldnn.deterministic=True
    import importlib.metadata
    checkpoint=B/'MASTER_author_assets/model/csi300_opensource_0.pkl'
    c=dict(code=[identity(Path(__file__)),identity(C/'core/round26_1/master_reference_data.py'),identity(C/'core/round26_1/master_model.py'),identity(C/'core/research_integrity_v5/cache.py'),identity(C/'core/research_integrity_v5/resources.py')],third_party=[identity(VENDOR/n) for n in EXPECTED]+[identity(inspect.getfile(__import__('qlib.data.dataset',fromlist=['TSDataSampler']).TSDataSampler))],dirty_diff='actual source content identities',environment={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','pyqlib']},parameters={'seed':17,'d_feat':158,'d_model':256,'t_nhead':4,'s_nhead':2,'dropout':.5,'market':63,'beta':5,'epochs':1,'lr':1e-5,'threshold':.95,'CPUthreads':2,'interop':2,'terminal_checkpoint_if_cap':True},seed=17,features='actual author processed158+market63, no synthetic inputs',universe='all available original CSI300 dump nodes; never label-filter inference',calibration='author frozen processed features; original per-date training label tail filter and zscore',data=[identity(ROOT/f'csi300_dl_{phase}.pkl') for phase in ['train','valid','test']]+[identity(checkpoint)],label='original author public benchmark label, not private A-share policy',maturity='author benchmark split; no actual-exit timestamp supplied, no source-vintage/as-of proof claimed',windows={'train':'2008-01-01..2020-03-31','valid':'2020-04-01..2020-06-30','test':'2020-07-01..2022-12-31','history':'original8 slots, original ffill+bfill within past window'},execution='public code/IC reference only, not A-share capital backtest',upstream=[identity(R/'CONTRACT.yaml'),identity(R/'PUBLIC_REFERENCE_COMPATIBILITY_FREEZE.json')])
    fp=fingerprint(c);root=B/'MASTER_reference'
    with transaction(root,c) as (dest,cached):
        if not cached:
            with measured(dest/'RESOURCES.json'):
                audit();train,valid,test=(load(p) for p in ['train','valid','test'])
                model=original_model(17,dest);expected_dates=int(train.get_index().get_level_values('datetime').nunique());steps=[]
                original_step=model.train_optimizer.step
                def observed_step(*args,**kwargs):
                    result=original_step(*args,**kwargs);steps.append(len(steps)+1)
                    if len(steps)%100==0:
                        import psutil,os
                        rss=psutil.Process(os.getpid()).memory_info().rss
                        if rss>16*1024**3:raise MemoryError('Reference own RSS exceeded16GiB')
                        print('MASTER_PUBLIC_ORIGINAL_FULL_EPOCH',len(steps),'/',expected_dates,'RSS',rss,flush=True)
                    return result
                model.train_optimizer.step=observed_step
                train_losses=[];original_epoch=model.train_epoch
                def observed_epoch(loader):
                    loss=original_epoch(loader);train_losses.append(float(loss));return loss
                model.train_epoch=observed_epoch
                model.fit(train,valid)
                assert len(steps)==expected_dates,'Original full training epoch did not visit all dates'
                torch.save(model.model.state_dict(),dest/'reference_terminal.pt')
                fresh_valid=predict(model,valid,dest,'fresh17_valid');fresh_test=predict(model,test,dest,'fresh17_test')
                graph_tests(model,test,dest,'fresh17_test')
                diagnostic=original_model(0,dest);diagnostic.load_param(str(checkpoint));prior_fitted=diagnostic.fitted
                # Old source uses str on load but compares fitted<0 in predict. IO gate only.
                diagnostic.fitted=0
                old_valid=predict(diagnostic,valid,dest,'author0_valid');old_test=predict(diagnostic,test,dest,'author0_test')
                graph_tests(diagnostic,test,dest,'author0_test')
                record={'status':'REFERENCE_PUBLIC_CORE_FULL_LOCKED_EPOCH_COMPLETE','scope':'original author public implementation/example-budget reference; not full40epoch paper reproduction or confidential business system','training_dates':expected_dates,'actual_optimizer_steps':len(steps),'train_losses':train_losses,'checkpoint_reason':'THRESHOLD_REACHED' if (dest/'public_core_17.pkl').exists() else 'CAP_REACHED','epochs':1,'fresh_seed':17,'terminal_checkpoint_sha256':sha(dest/'reference_terminal.pt'),'original_author_seed0_checkpoint':identity(checkpoint),'author_checkpoint_loader_original_fitted_type':type(prior_fitted).__name__,'author_loader_IO_gate_cast_only':True,'fresh_valid':fresh_valid,'fresh_test':fresh_test,'author0_valid':old_valid,'author0_test':old_test,'all_available_dump_nodes_retained':True,'prediction_segments_disjoint':True,'private_fits_use_author_weights':False,'dump_raw_universe_and_first_publication_vintage_proven':False,'train_label_available_at_metadata_supplied':False}
                (dest/'REFERENCE_STATUS.json').write_text(json.dumps(record,indent=2))
    path=root/fp;out=R/'reference';out.mkdir(exist_ok=True)
    for name in ['contract.json','RESOURCES.json','REFERENCE_STATUS.json','fresh17_valid_METRICS.json','fresh17_test_METRICS.json','fresh17_valid_daily_IC.csv','fresh17_test_daily_IC.csv','fresh17_test_GRAPH_TESTS.json','author0_valid_METRICS.json','author0_test_METRICS.json','author0_valid_daily_IC.csv','author0_test_daily_IC.csv','author0_test_GRAPH_TESTS.json']:
        __import__('shutil').copy2(path/name,out/name)
    __import__('shutil').copy2(path/'COMPLETE.json',out/'SERVER_COMPLETE_MANIFEST.json')
    (out/'SERVER_ARTIFACTS.json').write_text(json.dumps({'directory':str(path),'artifacts':[identity(p) for p in sorted(path.iterdir()) if p.is_file()]},ensure_ascii=False,indent=2))
    print('MASTER_PUBLIC_REFERENCE_COMPLETE',path,flush=True);return path
if __name__=='__main__':run()
