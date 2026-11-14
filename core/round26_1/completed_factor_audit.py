"""Audit actual completed PPO artifacts; never manufactures a factor/result."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from core.round26_1.market import R,KEY,factor_candidates
from core.round26_1.alphagen_data import ast_load,guard
from core.research_integrity_v5.cache import sha


def audit(directory,*,output_root=None):
    directory=Path(directory).resolve()
    s=json.loads((directory/'FROZEN_FACTOR.json').read_text())
    complete=json.loads((directory/'COMPLETE.json').read_text())
    for name,key in [('factor_predictions.parquet','factor_prediction_sha256'),('FROZEN_FACTOR.json','factor_model_sha256'),('CONTRACT.json','contract_sha256')]:
        assert sha(directory/name)==complete[key],name
    assert 0<s['generation_env_steps']<=100000 and 0<s['valid_evaluations']<=20000
    assert s['generation_env_steps']==100000 or s['valid_evaluations']>=20000
    assert 0<len(s['ASTs'])<=20 and len(s['ASTs'])==len(s['weights'])
    for node in s['ASTs']:guard(ast_load(node))
    assert pd.Timestamp(s['max_training_label_available_at'])<pd.Timestamp(s['fit_cutoff'])
    assert np.isfinite(s['prior_training_scale_std']) and s['prior_training_scale_std']>1e-12
    checkpoints=[json.loads(p.read_text()) for p in (directory/'selection_checkpoints').glob('*.json')]
    candidates=[p for p in checkpoints if p['validation_pool_IC'] is not None and p['training_pool']['size']>0]
    selected=max(candidates,key=lambda p:(p['validation_pool_IC'],-p['env_steps']))
    assert selected==json.loads((directory/'SELECTED_CHECKPOINT.json').read_text())
    assert selected['env_steps']==s['selected_generation_step']
    p=pd.read_parquet(directory/'factor_predictions.parquet')
    year=int(p.factor_outer_year.iloc[0]);arm=p.arm.iloc[0];seed=int(p.seed.iloc[0])
    expected=factor_candidates();expected=expected.loc[expected.trade_date.dt.year.eq(year),KEY].reset_index(drop=True)
    pd.testing.assert_frame_equal(p[KEY].reset_index(drop=True),expected,check_exact=True)
    assert not any('label' in col or col.startswith('next') for col in p.columns)
    assert (p.factor_fit_cutoff==pd.Timestamp(s['fit_cutoff'])).all()
    log=directory/'all_expression_evaluations.jsonl';attempts=source_evaluated=invalid=0
    last_source_pool_eval=0
    with log.open() as handle:
        for line in handle:
            row=json.loads(line);attempts+=1
            if row['status']=='SOURCE_EVALUATED':source_evaluated+=1;last_source_pool_eval=row['pool_eval_cnt']
            else:invalid+=1
    assert (attempts,source_evaluated,invalid)==(s['attempted_expressions'],s['valid_evaluations'],s['invalid_expressions'])
    result={'status':'PASS_ACTUAL_COMPLETED_GENERATION_ARTIFACTS','arm':arm,'seed':seed,'outer_year':year,'source_directory':str(directory),'env_steps':s['generation_env_steps'],'source_evaluation_calls_including_duplicates_and_source_rejections':source_evaluated,'unique_source_evaluated_ASTs':s['unique_valid_expressions'],'invalid_attempts_retained':invalid,'source_pool_admissible_eval_counter':last_source_pool_eval,'pool_size':len(s['ASTs']),'selected_validation_step':selected['env_steps'],'selected_validation_IC_not_strategy_performance':selected['validation_pool_IC'],'prediction_rows':len(p),'prediction_dates':p.trade_date.nunique(),'missing_factor_rows':int(p.z_AG.isna().sum()),'prediction_keys_exact_D_known_domain':True,'maturity_and_AST_scope_pass':True,'all_selection_checkpoints_compared':len(checkpoints),'last_fully_optimized_rollout':s['last_fully_optimized_rollout'],'unoptimized_environment_tail_steps':s['generation_env_steps']-s['last_fully_optimized_rollout'],'target_outer_return_used_for_selection':False,'candidate_backtest_complete':None,'candidate_backtest_assessment':'NOT_ASSESSED_BY_GENERATION_ARTIFACT_AUDIT','scope':'Frozen generation artifacts only. Aggregate research and full PPO counterfactual status are assessed separately in D3/STATUS.json and generation_counterfactual_tests/STATUS.json.'}
    dest=(Path(output_root) if output_root is not None else R/'generation_audits'/arm/str(seed));dest.mkdir(parents=True,exist_ok=True)
    (dest/f'{year}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result
if __name__=='__main__':
    import sys
    print(json.dumps(audit(sys.argv[1]),ensure_ascii=False,indent=2))
