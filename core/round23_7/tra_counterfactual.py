"""Full author-budget fit and actual causal-forecast label counterfactuals."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,threading,traceback
import numpy as np
import pandas as pd
import torch
from core.research_integrity_v5.cache import identity,sha
from core.round23_7.tra_reference import B,R,config
from core.round23_7.tra_causal import Bundle,CausalDataset,causal_model_class,NAT
from core.round23_7.tra_runtime import run_job,save_registry
from core.round23_7.tra_fit import value_hash
_FORECAST_RNG_LOCK=threading.Lock()

def mutate_fit_inputs(data,head,year):
    data=Path(data);label=pd.read_parquet(data/f'{head}_targets.parquet')
    times=pd.to_datetime(label.trade_date).to_numpy('datetime64[ns]').view('i8')
    available=pd.to_datetime(label.label_available_at).to_numpy('datetime64[ns]').view('i8')
    start=pd.Timestamp(f'{year-3}-01-01').value;train_cut=pd.Timestamp(f'{year-1}-01-01').value;outer_cut=pd.Timestamp(f'{year}-01-01').value
    regions={'train':(times>=start)&(times<train_cut),'valid':(times>=train_cut)&(times<outer_cut)}
    masks={name:region&((available==NAT)|(available>=cut)) for (name,region),cut in zip(regions.items(),[train_cut,outer_cut])}
    assert all(mask.sum()>0 for mask in masks.values()),'Each real fit-stage intervention must be nonempty'
    before=label.target.to_numpy('float64',copy=True);after=before.copy()
    rng=np.random.default_rng(171717+int(year))
    counts={}
    for name,mask in masks.items():
        ids=np.flatnonzero(mask);values=before[ids];candidate=rng.permutation(values)
        unchanged=(candidate==values)|(np.isnan(candidate)&np.isnan(values))
        if unchanged.all():candidate=np.where(np.isfinite(values),values+1e6,1e6)
        after[ids]=candidate
        changed=~((candidate==values)|(np.isnan(candidate)&np.isnan(values)))
        counts[name]={'in_domain_unavailable_rows':len(ids),'actual_changed_rows':int(changed.sum()),'maturity_metadata_preserved':True}
        assert changed.any()
    label['target']=after
    folder=B/'TRA_counterfactual_inputs'/str(year)/head;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'fit_unavailable_targets.parquet';label.to_parquet(path,index=False)
    report={'head':head,'year':year,'input':identity(data/f'{head}_targets.parquet'),'mutated':identity(path),'stages':counts,'scope':'fit-only phase-specific unavailable labels; original legally observed Outer feedback source retained; separate forecast replay perturbs as-of-unavailable values'}
    (folder/'INTERVENTION.json').write_text(json.dumps(report,indent=2));return path,report

def compare_fits(original,mutant):
    original=Path(original);mutant=Path(mutant)
    a=torch.load(original/'model.bin',map_location='cpu',weights_only=True);b=torch.load(mutant/'model.bin',map_location='cpu',weights_only=True)
    for name in ['model','tra']:
        assert a[name].keys()==b[name].keys()
        assert all(torch.equal(a[name][k],b[name][k]) for k in a[name]),name
    pd.testing.assert_frame_equal(pd.read_parquet(original/'forecast.parquet'),pd.read_parquet(mutant/'forecast.parquet'),check_exact=True)
    x=json.loads((original/'FIT_AUDIT.json').read_text());y=json.loads((mutant/'FIT_AUDIT.json').read_text())
    for field in ['epochs_completed','best_epoch','global_step','stop_reason','source_initial_model_retained']:assert x[field]==y[field],field
    for segment in ['train','valid','test']:assert x['dataset'][segment]==y['dataset'][segment],segment
    return {'checkpoint_model_value_sha256':value_hash(a['model']),'checkpoint_TRA_value_sha256':value_hash(a['tra']),'full_forecast_rows_exact':len(pd.read_parquet(original/'forecast.parquet')),'default_stopping_and_all_eligible_input_identities_exact':True}

def replay_forecast(original,data,head,year):
    original=Path(original);data=Path(data)
    cfg=json.loads((original/'contract.json').read_text())['parameters']['model']
    cfg['logdir']=None
    # Restore the exact upstream init path, then the selected full checkpoint.
    cfg['model_init_state']=cfg['model_init_state']['path']
    cls,_=causal_model_class();model=cls(**cfg)
    selected=torch.load(original/'model.bin',map_location='cpu',weights_only=True)
    model.model.load_state_dict(selected['model']);model.tra.load_state_dict(selected['tra'])
    bundle=Bundle(data,head);events=[]
    def perturb(values,available,cutoff):
        changed=values.copy();mask=(available==NAT)|(available>cutoff)
        changed[mask]=np.where(np.isfinite(changed[mask]),changed[mask]+1e6,1e6)
        in_year=(bundle.times>=pd.Timestamp(f'{year}-01-01').value)&(bundle.times<pd.Timestamp(f'{year+1}-01-01').value)
        count=int((mask&in_year).sum());assert count>0
        events.append({'asof':str(pd.Timestamp(cutoff)),'actual_changed_unavailable_targets':int(mask.sum()),'in_forecast_year_changed_unavailable_targets':count})
        return changed
    state=np.load(original/'OUTER_INITIAL_MEMORY.npz')
    assert not state['recorded'].any(),'Outer must start from a fresh model-specific queue'
    results=[]
    for name,provider in [('original',None),('future_and_unmature_labels',perturb)]:
        dataset=CausalDataset(bundle,year,cfg['tra_config']['num_states'],1024,feedback_provider=provider).prepare('test')
        for field in ['predictions','recorded','record_kind']:getattr(dataset,field)[:]=state[field]
        torch.set_rng_state(torch.load(original/'OUTER_INITIAL_TORCH_RNG.pt',map_location='cpu',weights_only=True))
        _,forecast=model.test_epoch(dataset,return_pred=True)
        frame=forecast.drop(columns=['label']).reset_index().rename(columns={'datetime':'trade_date','instrument':'instrument_id'}).sort_values(['trade_date','instrument_id']).reset_index(drop=True)
        pd.testing.assert_frame_equal(frame,pd.read_parquet(original/'forecast.parquet'),check_exact=True)
        results.append({'case':name,'rows':len(frame),'queue_audit':dataset.feedback_audit})
    assert events and min(e['in_forecast_year_changed_unavailable_targets'] for e in events)>0
    return {'status':'PASS_REAL_FULL_OUTER_FORECAST_REPLAY','cases':results,'actual_nonempty_interventions':events,'no_in_sample_history_transferred':True}

def one_case(data,head,year,registry):
    rows=registry.loc[registry.namespace.eq('original')&registry.seed.eq(17)&registry.year.eq(year)&registry['head'].eq(head)]
    expected={phase:Path(rows.loc[rows.phase.eq(phase)].iloc[0].path) for phase in ['init','T1']}
    path,intervention=mutate_fit_inputs(data,head,year)
    folder=R/'counterfactual_tests'/str(year)/head;folder.mkdir(parents=True,exist_ok=True)
    (folder/'FIT_LABEL_INTERVENTION.json').write_text(json.dumps(intervention,indent=2))
    init=run_job(head,year,17,'init',data,fit_label_override=path,namespace='cold_unavailable_targets')
    save_registry([init]);assert init['status']=='FULL_TARGET_FIT_COMPLETE',init
    initial=compare_fits(expected['init'],init['path'])
    route=run_job(head,year,17,'T1',data,init_state=Path(init['path'])/'model.bin',fit_label_override=path,namespace='cold_unavailable_targets')
    save_registry([route]);assert route['status']=='FULL_TARGET_FIT_COMPLETE',route
    trained=compare_fits(expected['T1'],route['path'])
    # Source Gumbel inference uses process-global torch RNG. Case learners are
    # isolated children, while diagnostic replays must be serialized here.
    with _FORECAST_RNG_LOCK:forecast=replay_forecast(expected['T1'],data,head,year)
    result={'status':'PASS_FULL_DEFAULT_FIT_AND_CAUSAL_FORECAST_COUNTERFACTUAL','head':head,'year':year,'init':init,'T1':route,'init_exact':initial,'T1_exact':trained,'forecast_replay':forecast,'namespace_is_diagnostic_not_extra_alpha':True}
    (folder/'RESULT.json').write_text(json.dumps(result,indent=2));return result

def run(data,jobs=2):
    assert jobs in [1,2];torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=2:torch.set_num_interop_threads(2)
    registry=pd.read_csv(R/'target_run_registry.csv');results=[]
    # run_job launches one child learner at a time in each case; never exceed2.
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures={pool.submit(one_case,data,head,year,registry):(head,year) for year in [2018,2022,2026] for head in ['five','ten']}
        for task in as_completed(futures):
            head,year=futures[task]
            try:results.append(task.result())
            except Exception as error:
                failed={'status':'FAILED_COUNTERFACTUAL_NOT_EXTERNAL_BLOCKER','head':head,'year':year,'error':repr(error),'traceback':traceback.format_exc()}
                folder=R/'counterfactual_tests'/str(year)/head;folder.mkdir(parents=True,exist_ok=True);(folder/'FAILED.json').write_text(json.dumps(failed,indent=2));results.append(failed)
    passed=sum(v['status']=='PASS_FULL_DEFAULT_FIT_AND_CAUSAL_FORECAST_COUNTERFACTUAL' for v in results)
    status={'status':'PASS_SIX_REAL_FULL_BUDGET_CASES' if passed==6 else 'FAILED_OR_PARTIAL_REAL_COUNTERFACTUALS','cases':len(results),'passed':passed,'full_cold_target_models_required':12,'scope':'each case independently reruns complete original-default init and T1, plus actual full Outer pointwise unavailable-label prediction replay','alpha_candidates_added':0}
    (R/'counterfactual_tests/STATUS.json').write_text(json.dumps(status,indent=2));return results
