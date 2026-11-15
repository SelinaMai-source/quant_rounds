"""Queue, real inference membership and exact original OT-step integration."""
from types import SimpleNamespace
from pathlib import Path
import json,copy,tempfile
import numpy as np
import pandas as pd
import torch
from core.round23_7.tra_reference import R,config,load_author
from core.round23_7.tra_causal import CausalDataset,Bundle,causal_model_class,NAT
from core.round23_7.tra_data import traded_history
from core.round23_7.tra_fit import observed_class

def fixture():
    dates=pd.bdate_range('2015-01-05',periods=70)
    keys=pd.DataFrame({'trade_date':dates,'instrument_id':['600000.SH']*70})
    history=np.full((70,60),-1,dtype='int32')
    for i in range(70):
        start=max(0,i-59);history[i,-(i-start+1):]=np.arange(start,i+1)
    return SimpleNamespace(keys=keys,x=np.zeros((70,60,6),dtype='float32'),history=history,valid=np.ones(70,dtype=bool),y=np.linspace(-.03,.03,70,dtype='float32'),available=(dates+pd.Timedelta(days=2,hours=15)).to_numpy().view('i8'),times=dates.to_numpy().view('i8'),index=pd.MultiIndex.from_frame(keys[['instrument_id','trade_date']]).set_names(['instrument','datetime']))

def run():
    torch.set_num_threads(2)
    quote=pd.DataFrame({'trade_date':pd.date_range('2020-01-01',periods=4),'volume':[1,0,1,0],'amount':[2,0,2,1]})
    traded=traded_history(quote)
    assert traded.index.tolist()==[0,2,3] and traded_history(quote.iloc[:2]).equals(traded.iloc[:1])
    b=fixture();b.available[10]=(pd.Timestamp(b.keys.trade_date.iloc[45])+pd.Timedelta(hours=15)).value
    b.available[12]=NAT;b.y[13]=np.nan
    ds=CausalDataset(b,2018,3,512);ds.segment='test'
    saved=np.array([[.1,.3,.5],[.4,.2,.0],[.2,.4,.6]],dtype='float32')
    ds.record_predictions(np.array([10,12,13]),saved)
    before=ds.batch_data([40]);assert not before[:,:,6:].any()
    old=b.y.copy();b.y[10]=1e6;b.y[12]=-1e6
    assert np.array_equal(before,ds.batch_data([40]))
    b.y[:]=old
    after=ds.batch_data([50]);slot=np.flatnonzero(b.history[50]==10)[0]
    expected=(saved[0]-b.y[10])**2;expected-=expected.min()
    assert np.array_equal(after[0,slot,6:],expected)
    assert ds.feedback_audit['unmature_label_reads']==0
    assert ds.feedback_audit['mature_missing_target_rows']>0
    periods=fixture()
    period_dates=pd.to_datetime(['2016-12-01','2016-12-29','2017-01-03','2017-12-29','2018-01-03'])
    periods.keys=pd.DataFrame({'trade_date':period_dates,'instrument_id':['A']*5})
    periods.x=np.zeros((5,60,6),dtype='float32');periods.history=np.full((5,60),-1,dtype='int32');periods.valid=np.ones(5,dtype=bool);periods.y=np.ones(5,dtype='float32')*.01
    periods.times=period_dates.to_numpy().view('i8');periods.available=pd.to_datetime(['2016-12-05 15:00','2017-01-03 15:00','2017-01-05 15:00','2018-01-03 15:00','2018-01-05 15:00']).to_numpy().view('i8')
    periods.index=pd.MultiIndex.from_frame(periods.keys[['instrument_id','trade_date']]).set_names(['instrument','datetime'])
    root=CausalDataset(periods,2018,3,512);tr,va,ou=root.prepare(['train','valid','test'])
    assert tr.indices.tolist()==[0] and va.indices.tolist()==[2] and ou.indices.tolist()==[4]
    tr.record_predictions([0],np.ones((1,3),dtype='float32'))
    assert not va.recorded.any() and not ou.recorded.any()
    tr.eval();va.eval();assert tr.indices.tolist()==[0,1] and va.indices.tolist()==[2,3]
    assert np.isnan(list(tr)[-1]['label'].numpy()).all() and np.isnan(list(va)[-1]['label'].numpy()).all()
    va.record_predictions([2],np.ones((1,3),dtype='float32'));va.eval();assert not va.recorded.any()
    assert root.segment_states['train'][1][0] and not ou.recorded.any()
    interventions=[]
    def future_feedback(y,available,cutoff):
        out=y.copy();mask=(available==NAT)|(available>cutoff);out[mask]=1e6
        interventions.append(int(mask.sum()));return out
    ds.feedback_provider=future_feedback
    assert np.array_equal(after,ds.batch_data([50])) and min(interventions)>0
    model_class,provenance=causal_model_class();_,author=load_author()
    model_cfg,_=config('init',R/'not_written_fixture')
    model_cfg.update(logdir=None);model_cfg['model_config']['input_size']=6
    original=author.TRAModel(**model_cfg);adapted=model_class(**model_cfg)
    left=CausalDataset(fixture(),2018,3,512).prepare('train')
    right=CausalDataset(fixture(),2018,3,512).prepare('train')
    captured={}
    left.assign_data=lambda index,loss:captured.update(index=index.detach().cpu().numpy(),loss=loss.detach().cpu().numpy())
    def capture(_module,_args,out):captured['experts']=out[1].detach().cpu().numpy().copy()
    hook=original.tra.register_forward_hook(capture)
    torch.manual_seed(1729);np.random.seed(1729);loss_original=original.train_epoch(left)
    torch.manual_seed(1729);np.random.seed(1729);loss_adapted=adapted.train_epoch(right)
    hook.remove()
    assert loss_original==loss_adapted and np.isfinite(loss_original)
    assert np.array_equal(right.predictions[captured['index']],captured['experts'])
    old_loss={int(i):row for i,row in zip(captured['index'],captured['loss'])}
    routed=right.batch_data([69])
    for slot,i in enumerate(right.bundle.history[69,:-21]):
        if i>=0:assert np.array_equal(routed[0,slot,6:],old_loss[int(i)])
    for name in ['model','tra']:
        assert all(torch.equal(v,getattr(adapted,name).state_dict()[k]) for k,v in getattr(original,name).state_dict().items())
    # Same frozen backbone in actual T0/T1 constructors; only head/router differ.
    work=R/'fixture_work';work.mkdir(parents=True,exist_ok=True)
    checkpoint=work/'six_channel_init_fixture.bin'
    torch.save({'model':adapted.model.state_dict()},checkpoint)
    route_cfg,_=config('route',R/'not_written_fixture',checkpoint)
    route_cfg['model_config']['input_size']=6;route_cfg['logdir']=None
    t1=model_class(**route_cfg);t0_cfg=copy.deepcopy(route_cfg);t0_cfg['tra_config']['num_states']=1;t0=model_class(**t0_cfg)
    assert all(torch.equal(v,t0.model.state_dict()[k]) for k,v in t1.model.state_dict().items())
    assert all(not p.requires_grad for m in [t0,t1] for p in m.model.parameters())
    assert t0.lr==t1.lr and t0.n_epochs==t1.n_epochs==500 and t0.early_stop==t1.early_stop==20
    # Actual real input forecast entry: erase every future label/maturity field,
    # retaining every D-known node and exactly the same author model predictions.
    manifest=json.loads((R/'target_data/server_only_inputs_manifest.json').read_text())
    root=Path(next(x['path'] for x in manifest if x['path'].endswith('/candidate_keys.parquet'))).parent
    base=Bundle(root,'ten');unknown=pd.read_parquet(root/'ten_targets.parquet');unknown['target']=np.nan;unknown['label_available_at']=pd.NaT
    rr,ss=np.nonzero(base.history>=0);old=base.history[rr,ss]
    instruments=base.keys.instrument_id.to_numpy()
    assert (instruments[rr]==instruments[old]).all() and (base.times[old]<=base.times[rr]).all()
    mutant=Bundle(root,'ten',target_override=unknown)
    a=CausalDataset(base,2026,10,1024).prepare('test');z=CausalDataset(mutant,2026,10,1024).prepare('test')
    assert np.array_equal(a.indices,z.indices)
    valid=a.indices[base.valid[a.indices]][:32]
    da=a.batch_data(valid);dz=z.batch_data(valid);assert np.array_equal(da,dz)
    t1.model.eval();t1.tra.eval()
    with torch.no_grad():
        torch.manual_seed(7317);pa=t1.tra(t1.model(torch.from_numpy(da[:,:,:6])),torch.from_numpy(da[:,:-21,6:]))[0]
        torch.manual_seed(7317);pz=t1.tra(t1.model(torch.from_numpy(dz[:,:,:6])),torch.from_numpy(dz[:,:-21,6:]))[0]
    assert torch.equal(pa,pz) and torch.isfinite(pa).all()
    assert CausalDataset(mutant,2026,10,1024,inference_bundle=base).prepare('test').bundle is base
    # Run the complete small synthetic causal forecast loop, then restore the
    # actual initial queue/RNG and perturb as-of-unavailable labels at each day.
    # This is integration only; the real full-fit diagnostics are separate.
    with tempfile.TemporaryDirectory(prefix='causal_forecast_fixture_',dir=work) as tmp:
        cfg=copy.deepcopy(model_cfg);cfg['logdir']=tmp
        observed=observed_class(model_class)(**cfg)
        first=CausalDataset(fixture(),2015,3,512).prepare('test')
        _,forecast_original=observed.test_epoch(first,return_pred=True)
        snapshot=np.load(Path(tmp)/'OUTER_INITIAL_MEMORY.npz')
        replay=CausalDataset(fixture(),2015,3,512,feedback_provider=future_feedback).prepare('test')
        for name in ['predictions','recorded','record_kind']:getattr(replay,name)[:]=snapshot[name]
        plain=model_class(**{**model_cfg,'logdir':None})
        plain.model.load_state_dict(observed.model.state_dict());plain.tra.load_state_dict(observed.tra.state_dict())
        torch.set_rng_state(torch.load(Path(tmp)/'OUTER_INITIAL_TORCH_RNG.pt',weights_only=True))
        _,forecast_mutant=plain.test_epoch(replay,return_pred=True)
        pd.testing.assert_frame_equal(forecast_original,forecast_mutant,check_exact=True)
        assert replay.feedback_audit['mature_label_reads']>0
    result={'status':'PASS_CAUSAL_ADAPTER_INTEGRATION','variable_delay_exceeding_author21_exclusion_masked':True,'NaT_never_released':True,'saved_expert_outputs_exact':True,'release_after_actual_maturity_matches_original_normalized_squared_error':True,'original_vs_adapted_actual_one_step_MSE_OT_parameters_exact':True,'T0_T1_same_frozen_backbone_inputs_and_default_budget':True,'real2026_forecast_nodes_after_all_labels_erased':len(a.indices),'real_model_forecast_entry_exact_after_label_erasure_rows':len(valid),'fixture_is_not_full_reference_or_target_run':True,'full_fit_counterfactuals_still_pending':True,'queue_fixture_audit':ds.feedback_audit}
    (R/'CAUSAL_INTEGRATION.json').write_text(json.dumps(result,indent=2))
    result['full_fixture_causal_forecast_replay_rows']=len(forecast_original)
    result['actual_asof_unavailable_label_interventions']=len(interventions)
    result['asof_interventions_all_nonempty']=min(interventions)>0
    result['all_evaluation_nodes_retained_mature_metric_labels_only']=True
    result['no_train_fitted_state_in_validation_or_outer']=True
    result['original_Torch_expert_loss_vs_released_queue_exact']=True
    (R/'CAUSAL_INTEGRATION.json').write_text(json.dumps(result,indent=2))
    (R/'ADAPTER_SOURCE_PROVENANCE.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':run()
