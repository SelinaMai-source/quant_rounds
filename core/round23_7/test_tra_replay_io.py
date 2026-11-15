"""Synthetic checkpoint/RNG replay wiring; no fitted research model or account."""
from pathlib import Path
import copy,json,tempfile
import numpy as np
import pandas as pd
import torch
from core.round23_7.tra_reference import R,config
from core.round23_7.tra_causal import Bundle,CausalDataset,causal_model_class,configure_target_runtime
from core.round23_7.tra_counterfactual import replay_forecast
from core.research_integrity_v5.cache import identity

def run():
    runtime=configure_target_runtime()
    root=R/'fixture_work';root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='checkpoint_replay_fixture_',dir=root) as tmp:
        tmp=Path(tmp);data=tmp/'data';data.mkdir()
        keys=pd.DataFrame([(d,f'S{i}') for day,d in enumerate(pd.bdate_range('2018-01-02',periods=70)) for i in range(1+day%3)],columns=['trade_date','instrument_id'])
        n=len(keys);rng=np.random.default_rng(17)
        x=rng.normal(0,.01,(n,60,6)).astype('float32');history=np.full((n,60),-1,dtype='int32')
        for i,row in keys.iterrows():
            ids=np.flatnonzero(keys.instrument_id.eq(row.instrument_id)&keys.trade_date.le(row.trade_date))[-60:]
            history[i,-len(ids):]=ids
        np.save(data/'features.npy',x);np.save(data/'history_candidate_indices.npy',history);np.save(data/'valid.npy',np.ones(n,dtype=bool))
        keys.to_parquet(data/'candidate_keys.parquet',index=False)
        labels=keys.assign(target=rng.normal(0,.01,n),label_available_at=keys.trade_date+pd.Timedelta(days=3,hours=15))
        labels.to_parquet(data/'five_targets.parquet',index=False)
        cls,_=causal_model_class();init_cfg,_=config('init',tmp/'random_init');init_cfg['model_config']['input_size']=6
        init=cls(**init_cfg);init_path=tmp/'random_init/model.bin'
        torch.save({'model':init.model.state_dict(),'tra':init.tra.state_dict()},init_path)
        cfg,_=config('route',tmp/'random_selected',init_path);cfg['model_config']['input_size']=6
        model=cls(**cfg);selected=tmp/'random_selected';bundle=Bundle(data,'five')
        test=CausalDataset(bundle,2018,10,1024).prepare('test')
        np.savez_compressed(selected/'OUTER_INITIAL_MEMORY.npz',predictions=test.predictions,recorded=test.recorded,record_kind=test.record_kind)
        torch.save(torch.get_rng_state(),selected/'OUTER_INITIAL_TORCH_RNG.pt')
        _,forecast=model.test_epoch(test,return_pred=True)
        frame=forecast.drop(columns=['label']).reset_index().rename(columns={'datetime':'trade_date','instrument':'instrument_id'}).sort_values(['trade_date','instrument_id']).reset_index(drop=True)
        frame.to_parquet(selected/'forecast.parquet',index=False)
        torch.save({'model':model.model.state_dict(),'tra':model.tra.state_dict()},selected/'model.bin')
        parameters=copy.deepcopy(cfg);parameters['model_init_state']=identity(init_path)
        (selected/'contract.json').write_text(json.dumps({'parameters':{'model':parameters,'target_runtime':runtime}},default=str))
        result=replay_forecast(selected,data,'five',2018)
        assert len(result['cases'])==3 and all(c['rows']==n for c in result['cases'])
        assert result['actual_nonempty_interventions']
    report={'status':'PASS_SYNTHETIC_CHECKPOINT_RNG_AND_FULL_CAUSAL_REPLAY_IO','scope':'random unfitted checkpoints,70 synthetic dates, mixed1/2/3-node days; no author reference completion, no A-share fit or account replay','forecast_rows':n,'comparison_replay_passes':3,'baseline_forecast_creation_passes':1,'total_full_forward_passes':4,'future_mutation_and_deletion_nonempty_exact':True,'result':result,'actual_research_models_trained':0,'actual_native_accounts_run':0}
    (R/'FORECAST_REPLAY_IO_INTEGRATION.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='result'}),flush=True)

if __name__=='__main__':run()
