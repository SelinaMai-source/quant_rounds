"""Integration checks; the one-batch parity fixture is not a full reference run."""
import copy,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from core.round23_7.tra_reference import load_author,verify_frozen,config,observed_model_class,D,R

def run():
    torch.set_num_threads(2)
    frozen=verify_frozen();dsmod,modelmod=load_author()
    init,ds=config('init',R/'fixture_unused')
    route,_=config('route',R/'fixture_unused',R/'not_a_real_checkpoint.bin')
    assert init['n_epochs']==route['n_epochs']==500
    assert init['early_stop']==route['early_stop']==20
    assert init['max_steps_per_epoch']==route['max_steps_per_epoch']==100
    assert init['tra_config']['num_states']==3 and route['tra_config']['num_states']==10
    assert route['freeze_model'] and not init['freeze_model']
    from qlib.data.dataset.loader import StaticDataLoader
    full=StaticDataLoader({'feature':str(D/'feature.pkl'),'label':str(D/'label.pkl')}).load()
    assert full.shape==(2512790,17) and full.notna().all().all()
    assert list(full['feature'].columns)==['MKTCAP','PB_MRQ','MOM_12M','PE_TTM','PS_TTM','PC_TTM','ASSET_GR','EQUITY_GR','EPS_GR','NET_MARGIN','RECEIVA_TO','ASSET_TO','REV_1M','LOTTERY','P_12MHI','P_12MLO']
    del full
    index=pd.MultiIndex.from_product([pd.date_range('2010-01-04',periods=8),['SH600000','SZ000001']],names=['datetime','instrument'])
    rng=np.random.default_rng(32317)
    feature=pd.DataFrame(rng.normal(0,.05,(16,16)),index=index)
    label=pd.Series(rng.normal(0,.01,16),index=index,name='label')
    def dataset():
        cfg=copy.deepcopy(ds)
        cfg['segments']={key:['2010-01-01','2010-02-01'] for key in ['train','valid','test']}
        cfg['handler']['kwargs']['data_loader']['kwargs']['config']={'feature':feature.copy(),'label':label.to_frame().copy()}
        return dsmod.MTSDatasetH(**cfg)
    left,right=dataset(),dataset()
    sl=left.batch_slices[8]
    assert sl.start==0 and left._index[sl.start][0]!=left._index[sl.stop-1][0]
    original=modelmod.TRAModel(**{**init,'logdir':None})
    fixtures=R/'fixture_work';fixtures.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='TRA_observer_fixture_',dir=fixtures) as tmp:
        observed=observed_model_class(modelmod)(**{**init,'logdir':tmp})
        for attr in ['model','tra']:
            assert all(torch.equal(v,getattr(observed,attr).state_dict()[k]) for k,v in getattr(original,attr).state_dict().items())
        torch.manual_seed(1729);np.random.seed(1729)
        a=original.train_epoch(left.prepare('train'))
        torch.manual_seed(1729);np.random.seed(1729)
        b=observed.train_epoch(right.prepare('train'))
        assert a==b and np.isfinite(a)
        assert np.array_equal(left._data,right._data)
        for attr in ['model','tra']:
            assert all(torch.equal(v,getattr(observed,attr).state_dict()[k]) for k,v in getattr(original,attr).state_dict().items())
        assert json.loads((Path(tmp)/'LIVE.json').read_text())['event']=='TRAIN_END'
    result={'status':'PASS_REFERENCE_INTEGRATION','public_loader_rows':2512790,'full_original_default_configs_unchanged':True,'public_data_and_locked_source_hashes_pass':True,'observer_vs_original_one_batch_loss_and_all_parameters_exact':True,'original_cross_stock_boundary_issue_reproduced':True,'fixture_only_not_reference_completion':True,'target_adapter_not_tested_here':True}
    R.mkdir(parents=True,exist_ok=True);(R/'REFERENCE_INTEGRATION.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':run()
