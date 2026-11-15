"""Full-budget target TRA folds; no target training before author reference."""
from pathlib import Path
import argparse,copy,datetime,hashlib,importlib.metadata,json,os,shutil,signal,subprocess,threading
import numpy as np
import pandas as pd
import torch
import psutil
from core.research_integrity_v5.cache import identity,source_manifest,fingerprint,transaction,hit
from core.research_integrity_v5.resources import measured
from core.round23_7.tra_reference import C,B,R,A,config,verify_frozen
from core.round23_7.tra_causal import Bundle,CausalDataset,causal_model_class

def value_hash(values):
    h=hashlib.sha256()
    for name,tensor in sorted(values.items()):
        h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def dataset_identity(ds):
    b=ds.bundle;i=ds.indices
    return {'rows':len(i),'dates':int(b.keys.iloc[i].trade_date.nunique()),'start':str(b.keys.iloc[i].trade_date.min()),'end':str(b.keys.iloc[i].trade_date.max()),'keys_sha256':hashlib.sha256(pd.util.hash_pandas_object(b.keys.iloc[i],index=False).values.tobytes()).hexdigest(),'features_sha256':hashlib.sha256(np.asarray(b.x[i]).tobytes()).hexdigest(),'history_mapping_sha256':hashlib.sha256(np.asarray(b.history[i]).tobytes()).hexdigest(),'targets_sha256':None if ds.segment=='test' else hashlib.sha256(b.y[i].tobytes()).hexdigest(),'maturity_sha256':None if ds.segment=='test' else hashlib.sha256(b.available[i].tobytes()).hexdigest(),'max_label_available_at':None if ds.segment=='test' else str(pd.Timestamp(b.available[i].max())),'prediction_membership_uses_target':False if ds.segment=='test' else 'training/validation loss requires a mature label'}

def observed_class(base):
    class Observed(base):
        def event(self,kind,**extra):
            obj={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'event':kind,'epoch':getattr(self,'observed_epoch',-1),'global_step':self.global_step,**extra}
            p=Path(self.logdir)/'LIVE.json';p.write_text(json.dumps(obj,default=str,indent=2))
            print('TRA_TARGET',kind,'epoch',obj['epoch'],'step',self.global_step,extra,flush=True)
        def train_epoch(self,dataset):
            self.observed_epoch=getattr(self,'observed_epoch',-1)+1;self.event('TRAIN_BEGIN')
            result=super().train_epoch(dataset)
            if not np.isfinite(result):raise FloatingPointError('Actual original MSE/OT training loss nonfinite; retain failed attempt, never call it an external blocker')
            self.event('TRAIN_END',loss=float(result));return result
        def test_epoch(self,dataset,return_pred=False):
            self.event('EVALUATE_BEGIN',segment=dataset.segment,return_pred=return_pred)
            if return_pred and dataset.segment=='test':
                # Exact initial state of the real final Outer causal forecast.
                np.savez_compressed(Path(self.logdir)/'OUTER_INITIAL_MEMORY.npz',predictions=dataset.predictions,recorded=dataset.recorded,record_kind=dataset.record_kind)
                torch.save(torch.get_rng_state(),Path(self.logdir)/'OUTER_INITIAL_TORCH_RNG.pt')
            result=super().test_epoch(dataset,return_pred=return_pred)
            self.event('EVALUATE_END',segment=dataset.segment,metrics=result[0]);return result
    return Observed

def fit(head,year,seed,phase,data,*,init_state=None,fit_label_override=None,namespace='original'):
    status=json.loads((R/'REFERENCE_STATUS.json').read_text()) if (R/'REFERENCE_STATUS.json').exists() else {}
    assert status.get('status')=='REFERENCE_RUN_COMPLETE','Full author reference still pending; do not start A-share learning'
    assert phase in ['init','T0','T1'] and head in ['five','ten'] and seed in [17,42,73]
    assert int(year) in range(2018,2027)
    verify_frozen();torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=2:torch.set_num_interop_threads(2)
    data=Path(data);marker=json.loads((data/'COMPLETE.json').read_text());assert hit(data,marker['experiment_fingerprint'])
    original=Bundle(data,head)
    override=pd.read_parquet(fit_label_override) if fit_label_override else None
    bundle=Bundle(data,head,target_override=override) if override is not None else original
    if override is not None:
        assert np.array_equal(bundle.available,original.available),'Fit counterfactual must preserve actual maturity metadata'
        allowed=(original.available==np.iinfo(np.int64).min)|(original.available>=pd.Timestamp(f'{year}-01-01').value)
        assert np.array_equal(bundle.y[~allowed],original.y[~allowed],equal_nan=True),'Do not perturb a legitimately mature fit target'
    if phase!='init':
        init_audit=json.loads((Path(init_state).parent/'FIT_AUDIT.json').read_text())
        assert (init_audit['phase'],init_audit['head'],init_audit['year'],init_audit['seed'])==('init',head,year,seed),'Wrong shared pretrained fold/head/seed'
    cfg,_=config('init' if phase=='init' else 'route',B/'placeholder',init_state)
    cfg['seed']=int(seed);cfg['model_config']['input_size']=6
    if phase=='T0':cfg['tra_config']['num_states']=1
    batch=512 if phase=='init' else 1024
    dataset=CausalDataset(bundle,year,cfg['tra_config']['num_states'],batch,inference_bundle=original)
    train,valid,test=dataset.prepare(['train','valid','test'])
    identities={name:dataset_identity(ds) for name,ds in [('train',train),('valid',valid),('test',test)]}
    for name,cut in [('train',f'{year-1}-01-01'),('valid',f'{year}-01-01')]:assert pd.Timestamp(identities[name]['max_label_available_at'])<pd.Timestamp(cut)
    base,patch=causal_model_class()
    parameters={'head':head,'year':int(year),'seed':int(seed),'phase':phase,'model':copy.deepcopy(cfg),'batch_size':batch,'dataset_identities':identities,'adapter_train_sha256':patch['train_adapted_sha256'],'adapter_test_sha256':patch['test_adapted_sha256'],'CPU_threads':2,'RSS_limit_GiB':16,'fit_label_override':identity(fit_label_override) if fit_label_override else None,'inference_feedback':'original observed targets released only at their actual event time; fit-only counterfactual label overrides never replace legitimately observed Outer feedback'}
    parameters['model'].pop('logdir',None)
    if init_state:parameters['model']['model_init_state']=identity(init_state)
    own=[Path(__file__),C/'core/round23_7/tra_causal.py',C/'core/round23_7/tra_reference.py',C/'core/research_integrity_v5/cache.py',C/'core/research_integrity_v5/resources.py']
    inputs=[data/n for n in ['contract.json','COMPLETE.json','candidate_keys.parquet','features.npy','history_candidate_indices.npy','valid.npy',f'{head}_targets.parquet']]+[R/'CONTRACT.yaml',R/'DATA_OBSERVATION_CLARIFICATION.json',R/'MATCHED_COVERAGE_PROTOCOL.json']
    if fit_label_override:inputs.append(Path(fit_label_override))
    inputs.append(R/'SINGLETON_SHAPE_COMPATIBILITY.json')
    c={'code':source_manifest(own+[A/'src/model.py',A/'src/dataset.py']),'third_party':{'locked_legacy_TRA_commit':verify_frozen()['source_commit']},'dirty_diff':{n:hashlib.sha256(subprocess.check_output(['git','-C',str(B/n),'diff','HEAD'])).hexdigest() for n in ['round23_7','round26_1']},'environment':{d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']},'parameters':parameters,'seed':int(seed),'features':['open','close','high','low','volume','vwap'],'universe':'all D-known first-board nodes; maturity filters training/validation loss only','calibration':'original author regression loss/OT; observed price/volume window normalization','data':[identity(p) for p in inputs],'label':head+' original net policy target','maturity':'actual per-head exit at local exchange15:00; variable-delay queue masks every historical feedback read','windows':{'outer_year':int(year),'train':[f'{year-3}-01-01',f'{year-1}-01-01'],'valid':[f'{year-1}-01-01',f'{year}-01-01']},'execution':'forecast learner only; unchanged original Stock executor downstream','upstream':[identity(init_state)] if init_state else []}
    fp=fingerprint(c);root=B/'TRA_target_models'/namespace/str(seed)/str(year)/head/phase
    stop=threading.Event();prior=signal.getsignal(signal.SIGUSR1)
    def exceeded(*_):raise MemoryError('Actual self RSS exceeded16GiB; frozen model/budget not reduced')
    signal.signal(signal.SIGUSR1,exceeded)
    def guard():
        while not stop.wait(.5):
            if psutil.Process().memory_info().rss>16*1024**3:os.kill(os.getpid(),signal.SIGUSR1);return
    thread=threading.Thread(target=guard,daemon=True);thread.start()
    try:
        with transaction(root,c) as (dest,cached):
            if not cached:
                cfg['logdir']=str(dest)
                with measured(dest/'RESOURCES.json'):
                    model=observed_class(base)(**cfg)
                    initial_backbone=value_hash(model.model.state_dict())
                    history={};model.fit(dataset,evals_result=history)
                    v=np.array([z['IC'] for z in history['valid']],dtype=float);epochs=len(v)
                    assert epochs>1 and epochs<=500 and np.isfinite(v).all()
                    best=int(np.argmax(v));assert epochs==500 or epochs-1-best>=20
                    for group in [model.model,model.tra]:assert all(torch.isfinite(x).all() for x in group.state_dict().values())
                    final_backbone=value_hash(model.model.state_dict())
                    if phase!='init':assert initial_backbone==final_backbone,'Frozen backbone changed'
                    predictions=pd.read_pickle(dest/'pred.pkl').drop(columns=['label']).reset_index().rename(columns={'datetime':'trade_date','instrument':'instrument_id'})
                    predictions=predictions.sort_values(['trade_date','instrument_id']).reset_index(drop=True)
                    expected=original.keys.iloc[test.indices].sort_values(['trade_date','instrument_id']).reset_index(drop=True)
                    pd.testing.assert_frame_equal(predictions[['trade_date','instrument_id']],expected,check_exact=True)
                    valid_inputs=np.asarray(original.valid[test.indices]);assert np.isfinite(predictions.loc[valid_inputs,'score']).all()
                    predictions.to_parquet(dest/'forecast.parquet',index=False)
                    np.savez_compressed(dest/'FINAL_SAVED_EXPERT_OUTPUTS.npz',predictions=dataset.predictions,recorded=dataset.recorded,record_kind=dataset.record_kind)
                    audit={'status':'PASS_FULL_AUTHOR_DEFAULT_TARGET_FIT','head':head,'seed':seed,'year':year,'phase':phase,'epochs_completed':epochs,'best_epoch':best,'stop_reason':'AUTHOR_MAX_EPOCHS' if epochs==500 else 'AUTHOR_EARLY_STOP20','global_step':model.global_step,'all_default_model_parameters_except_data_dimension_and_T0_ablation':True,'dataset':identities,'forecast_rows':len(predictions),'forecast_invalid_input_rows':int((~valid_inputs).sum()),'forecast_missing_on_valid_inputs':0,'initial_backbone_value_sha256':initial_backbone,'final_backbone_value_sha256':final_backbone,'frozen_backbone_unchanged':initial_backbone==final_backbone if phase!='init' else None,'queue_audit':dataset.feedback_audit,'training_prediction_records_are_not_OOF':True,'no_second_level_learner':True,'outer_labels_not_used_for_early_stop':True}
                    (dest/'FIT_AUDIT.json').write_text(json.dumps(audit,indent=2));(dest/'ADAPTER_SOURCE_PROVENANCE.json').write_text(json.dumps(patch,indent=2))
        path=root/fp;dest=R/'target_models'/namespace/str(seed)/str(year)/head/phase;dest.mkdir(parents=True,exist_ok=True)
        for name in ['contract.json','COMPLETE.json','model.bin','forecast.parquet','logs.csv','info.json','RESOURCES.json','LIVE.json','FIT_AUDIT.json','ADAPTER_SOURCE_PROVENANCE.json','OUTER_INITIAL_MEMORY.npz','OUTER_INITIAL_TORCH_RNG.pt','FINAL_SAVED_EXPERT_OUTPUTS.npz']:
            if (path/name).exists():shutil.copy2(path/name,dest/name)
        (dest/'server_artifact_manifest.json').write_text(json.dumps([identity(p) for p in sorted(path.iterdir()) if p.is_file()],indent=2))
        result={'status':'FULL_TARGET_FIT_COMPLETE','head':head,'year':int(year),'seed':int(seed),'phase':phase,'path':str(path),'fingerprint':fp,'cache_hit':cached,'namespace':namespace}
        print('TRA_TARGET_RESULT',json.dumps(result),flush=True);return result
    finally:stop.set();thread.join();signal.signal(signal.SIGUSR1,prior)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--head',choices=['five','ten'],required=True);p.add_argument('--year',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--phase',choices=['init','T0','T1'],required=True);p.add_argument('--data',required=True);p.add_argument('--init-state');p.add_argument('--fit-label-override');p.add_argument('--namespace',default='original');args=p.parse_args();fit(args.head,args.year,args.seed,args.phase,args.data,init_state=args.init_state,fit_label_override=args.fit_label_override,namespace=args.namespace)
