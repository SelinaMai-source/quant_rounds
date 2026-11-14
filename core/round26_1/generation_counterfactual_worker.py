"""Cold full-budget original run_fold with stage-aware unavailable target corruption."""
from pathlib import Path
from unittest.mock import patch
import argparse,json,time
import numpy as np
import pandas as pd
from core.round26_1 import alphagen_train as training
from core.round26_1.market import C,R,B
from core.research_integrity_v5.cache import identity
from core.research_integrity_v5.resources import measured
from core.round26_1.completed_factor_audit import audit


def run(arm,year,baseline):
    baseline=Path(baseline).resolve();dest=R/'generation_counterfactual_tests'/arm/str(year);dest.mkdir(parents=True,exist_ok=True)
    original_init=training.MatureCalculator.__init__;original_contract=training.full_contract;interventions=[]
    contract_metadata={'diagnostic_only':'unavailable target intervention; not additional alpha search','seed':17,'year':year,'arm':arm,'source':identity(__file__),'orchestrator_source':identity(C/'core/round26_1/generation_counterfactual.py'),'baseline_model':identity(baseline/'FROZEN_FACTOR.json'),'baseline_contract':identity(baseline/'CONTRACT.json'),'protocol':identity(R/'PPO_COUNTERFACTUAL_TEST_CONTRACT.json'),'budget':'same100000 environment steps/20000 source-evaluated expressions; original model/optimizer/seed'}
    def init(self,domain,targets,column,cutoff):
        changed=targets.copy();date=pd.to_datetime(changed.trade_date);available=pd.to_datetime(changed.label_available_at);cut=pd.Timestamp(cutoff)
        future=date.ge(cut);immature=date.lt(cut)&available.ge(cut)
        before=changed[column].copy();changed.loc[future,column]=np.nan
        rng=np.random.default_rng(171717)
        values=changed.loc[immature,column].to_numpy();corrupt=rng.permutation(values)
        if len(values) and np.array_equal(values,corrupt,equal_nan=True):corrupt=np.nan_to_num(values,nan=0.)+1e6
        changed.loc[immature,column]=corrupt
        actual=(~(before.eq(changed[column])|(before.isna()&changed[column].isna()))).sum()
        original_init(self,domain,changed,column,cutoff)
        interventions.append({'domain_start':domain.identity['start'],'domain_end':domain.identity['end'],'cutoff':str(cutoff),'target_column':column,'future_target_values_deleted':int((future&before.notna()).sum()),'immature_rows_selected':int(immature.sum()),'actual_changed_target_rows':int(actual),'post_mask_identity':self.identity})
    def contract(parameters,inputs,*args,**kwargs):
        parameters=dict(parameters);parameters['full_counterfactual_test']=contract_metadata
        return original_contract(parameters,inputs,*args,**kwargs)
    previous_result=json.loads((dest/'RESULT.json').read_text()) if (dest/'RESULT.json').exists() else None
    completed_before={p.parent.resolve() for p in (B/'counterfactual_generators'/'alphagen_runs'/arm/str(year)/'17').glob('*/COMPLETE.json')}
    stamp=str(time.time_ns())
    with measured(dest/f'whole_worker_{stamp}_RESOURCES.json'):
        with patch.object(training.MatureCalculator,'__init__',init),patch.object(training,'full_contract',contract),patch.object(training,'B',B/'counterfactual_generators'):
            directory=training.run_fold(arm,year,17,resume=True)
        cache_hit=directory.resolve() in completed_before
        if cache_hit and previous_result and previous_result['diagnostic']==str(directory):interventions=previous_result['interventions']
        assert (directory/'COMPLETE.json').exists(),'Full diagnostic PPO generation did not complete'
        before=json.loads((baseline/'FROZEN_FACTOR.json').read_text());after=json.loads((directory/'FROZEN_FACTOR.json').read_text())
        assert before['ASTs']==after['ASTs'],'ASTs changed under unavailable targets'
        np.testing.assert_array_equal(before['weights'],after['weights'])
        for name in ['generation_env_steps','valid_evaluations','unique_valid_expressions','selected_generation_step','selected_validation_pool_IC','prior_training_scale_mean','prior_training_scale_std']:
            assert before[name]==after[name],name
        pd.testing.assert_frame_equal(pd.read_parquet(baseline/'factor_predictions.parquet'),pd.read_parquet(directory/'factor_predictions.parquet'),check_exact=True)
        basecontract=json.loads((baseline/'CONTRACT.json').read_text())['parameters'];newcontract=json.loads((directory/'CONTRACT.json').read_text())['parameters']
        for name in ['train','validation']:assert basecontract[name]==newcontract[name],name
        # Output roots differ so diagnostics can never replace original factor artifacts.
        audit_root=dest/'completed_artifact_audit';audit(directory,output_root=audit_root)
        result={'status':'PASS_FULL_ORIGINAL_PPO_UNAVAILABLE_TARGET_COUNTERFACTUAL','arm':arm,'year':year,'seed':17,'baseline':str(baseline),'diagnostic':str(directory),'interventions':interventions,'ASTs_weights_predictions_exact':True,'actual_complete_environment_steps':after['generation_env_steps'],'diagnostic_only':True,'cache_hit_existing_complete_counterfactual':cache_hit,'actual_training_this_invocation_env_steps':0 if cache_hit else after['generation_env_steps'],'economic_backtests':0,'contract':contract_metadata,'scope':'Actual run_fold and original LSTM PPO, validation selection, pool refit and complete inference. Future/immature targets changed at each stage cutoff. Market data fixed; not a vendor-revision reconstruction proof.'}
        training.atomic_json(dest/'RESULT.json',result)
    print('FULL_PPO_COUNTERFACTUAL_RESULT',json.dumps({'arm':arm,'year':year,'status':result['status'],'directory':str(directory)}),flush=True)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['AG_RAW','AG_RES'],required=True);p.add_argument('--year',type=int,required=True);p.add_argument('--baseline',required=True);a=p.parse_args();run(a.arm,a.year,a.baseline)
