"""Full default stopping on tiny synthetic fixtures, not a research/reference fit."""
from types import SimpleNamespace
from pathlib import Path
import json,tempfile
import numpy as np
import pandas as pd
import torch
from core.round23_7.tra_reference import R,config
from core.round23_7.tra_causal import CausalDataset,causal_model_class,configure_target_runtime
from core.round23_7.tra_fit import observed_class,stopping_audit,value_hash

def fixture():
    dates=pd.DatetimeIndex([d for y in range(2015,2019) for d in pd.bdate_range(f'{y}-02-02',periods=3)])
    keys=pd.MultiIndex.from_product([dates,['A','B','C']],names=['trade_date','instrument_id']).to_frame(index=False)
    n=len(keys);stock=np.tile(np.arange(3),len(dates));x=np.zeros((n,60,6),dtype='float32')
    x[:,:,0]=(stock[:,None]-1)*.02;x[:,:,2]=.03;x[:,:,3]=-.03;x[:,:,5]=x[:,:,0]/2
    history=np.full((n,60),-1,dtype='int32')
    times=keys.trade_date.to_numpy('datetime64[ns]').view('i8')
    for i in range(n):
        allowed=np.flatnonzero((keys.instrument_id.to_numpy()==keys.instrument_id.iloc[i])&(times<=times[i])&(times>=times[i]-pd.Timedelta(days=60).value))
        history[i,-len(allowed):]=allowed
    return SimpleNamespace(keys=keys,x=x,history=history,valid=np.ones(n,dtype=bool),y=((stock-1)*.01).astype('float32'),available=(keys.trade_date+pd.Timedelta(days=2,hours=15)).to_numpy().view('i8'),times=times,index=pd.MultiIndex.from_frame(keys[['instrument_id','trade_date']]).set_names(['instrument','datetime']))

def run():
    configure_target_runtime()
    base,_=causal_model_class();cls=observed_class(base)
    work=R/'fixture_work';work.mkdir(parents=True,exist_ok=True);records=[]
    with tempfile.TemporaryDirectory(prefix='default_stop_io_fixture_',dir=work) as tmp:
        tmp=Path(tmp);init_path=None;shared=None
        for phase in ['init','T0','T1']:
            cfg,_=config('init' if phase=='init' else 'route',tmp/phase,init_path)
            cfg['model_config']['input_size']=6
            if phase=='T0':cfg['tra_config']['num_states']=1
            model=cls(**cfg);initial=value_hash(model.model.state_dict())
            data=CausalDataset(fixture(),2018,cfg['tra_config']['num_states'],512 if phase=='init' else 1024)
            history={};model.fit(data,evals_result=history)
            stop=stopping_audit(history);assert stop['epochs_completed']>=20
            assert model.global_step+1==stop['epochs_completed']
            out=pd.read_pickle(tmp/phase/'pred.pkl')
            assert len(out)==9 and out.score.notna().all() and out.label.isna().all()
            assert (tmp/phase/'OUTER_INITIAL_MEMORY.npz').is_file() and (tmp/phase/'OUTER_INITIAL_TORCH_RNG.pt').is_file()
            checkpoint=torch.load(tmp/phase/'model.bin',weights_only=True)
            assert value_hash(checkpoint['model'])==value_hash(model.model.state_dict())
            if phase=='init':init_path=tmp/phase/'model.bin';shared=value_hash(checkpoint['model'])
            else:assert initial==shared==value_hash(checkpoint['model'])
            records.append({'phase':phase,**stop,'actual_optimizer_steps':model.global_step+1,'forecast_rows':len(out),'all_future_test_labels_unavailable_to_model':True,'checkpoint_and_outer_snapshot_verified':True})
    report={'status':'PASS_SYNTHETIC_FULL_FIT_IO_AND_DEFAULT_STOPPING','scope':'tiny synthetic integration fixtures with original500epoch/20patience stopping; NOT full author public reference, NOT actual A-share research fits','phases':records,'T0_T1_same_shared_backbone_frozen':True,'no_research_model_count_increase':True}
    (R/'FIT_IO_INTEGRATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':run()
