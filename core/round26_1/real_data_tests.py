"""Real AlphaGen data entry tests; no small synthetic run counted as reproduction."""
from pathlib import Path
import json,importlib.util,time
import numpy as np,pandas as pd,torch
from core.round26_1.market import C,R,bars,calibrated_residual
from core.round26_1.alphagen_data import CandidateData,MatureCalculator,guard,ast_load
from core.research_integrity_v5.resources import measured
from alphagen.data.calculator import TensorAlphaCalculator
from alphagen.data.expression import Feature,Mean,Div,Corr
from alphagen_qlib.stock_data import FeatureType

def run():
 torch.set_num_threads(2);out=[];old=R/'superseded_label_column_dependency';spec=importlib.util.spec_from_file_location('old_market_label_reader',old/'market.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);read=pd.read_parquet
 def no_prediction_labels(path,*args,**kwargs):
  if str(path).endswith('ten_candidate.parquet'):
   requested=kwargs.get('columns',[])
   if any('label' in str(c) or c in ['actual_exit_date','next2_date'] for c in requested):raise KeyError('future label columns absent in predictor integration test')
  return read(path,*args,**kwargs)
 with measured(R/'real_data_tests_RESOURCES.json'):
  pd.read_parquet=no_prediction_labels
  try:
   failed=False
   try:module.factor_candidates()
   except KeyError:failed=True
   assert failed
   domain=CandidateData('2017-01-01','2018-12-31')
  finally:pd.read_parquet=read
  previous=json.loads((R/'real_data_preflight.json').read_text());assert domain.identity['market_tensor_sha256']==previous['market_domain']['market_tensor_sha256'];assert domain.identity['candidate_keys_sha256']==previous['market_domain']['candidate_keys_sha256'];out.append({'test':'real predictor label column deletion','old_reader_failed':True,'new_reader_PASS':True,'real_candidate_keys':len(domain.keys),'feature_and_key_hashes_unchanged':True})
  target=calibrated_residual();calc=MatureCalculator(domain,target,'label_exit_rule_net','2019-01-01');exprs=[Feature(FeatureType.CLOSE),Mean(Feature(FeatureType.VOLUME),5),Div(Feature(FeatureType.CLOSE),Mean(Feature(FeatureType.CLOSE),5)),Corr(Feature(FeatureType.CLOSE),Feature(FeatureType.VOLUME),5)]
  # Original exact metric implementations versus memoized identical values on real inputs.
  for a in exprs:
   reference=TensorAlphaCalculator.calc_single_IC_ret(calc,a);memo=calc.calc_single_IC_ret(a);assert reference==memo or np.isnan(reference) and np.isnan(memo)
   for b in exprs:
    reference=TensorAlphaCalculator.calc_mutual_IC(calc,a,b);memo=calc.calc_mutual_IC(a,b);assert reference==memo or np.isnan(reference) and np.isnan(memo)
  t=time.monotonic()
  for _ in range(25):
   for a in exprs:
    TensorAlphaCalculator.calc_single_IC_ret(calc,a)
    for b in exprs:TensorAlphaCalculator.calc_mutual_IC(calc,a,b)
  uncached=time.monotonic()-t;t=time.monotonic()
  for _ in range(25):
   for a in exprs:
    calc.calc_single_IC_ret(a)
    for b in exprs:calc.calc_mutual_IC(a,b)
  cached=time.monotonic()-t;out.append({'test':'source exact metric memoization','pairs':len(exprs)**2,'PASS_BIT_EXACT':True,'repeated_source_seconds':uncached,'repeated_cached_seconds':cached,'cache_scope':'one immutable domain/target/maturity instance, AST and tensor-version keys; no persistent cross-fold cache'})
  a=exprs[0];original=calc.calc_single_IC_ret(a);calc.target.mul_(-1);got=calc.calc_single_IC_ret(a);reference=TensorAlphaCalculator.calc_single_IC_ret(calc,a);assert got==reference;assert np.isclose(got,-original,atol=1e-7);calc.target.mul_(-1);out.append({'test':'in-memory target mutation invalidates metric cache','PASS':True})
  # The actual rolling entry must be insensitive to future observations and adjustment fields.
  full=bars();dates=full.index.get_level_values('trade_date');fields=['open','close','high','low','volume','vwap'];saved=full.loc[dates>pd.Timestamp('2018-12-31'),fields].copy();full.loc[dates>pd.Timestamp('2018-12-31'),fields]=1e9;full['future_adjustment_factor_test']=np.where(dates>pd.Timestamp('2018-12-31'),1234.,1.);future=CandidateData('2017-01-01','2018-12-31');assert future.identity['market_tensor_sha256']==domain.identity['market_tensor_sha256'];pd.testing.assert_frame_equal(domain.frame(exprs,[.25]*4),future.frame(exprs,[.25]*4),check_exact=True);full.loc[saved.index,fields]=saved;full.drop(columns='future_adjustment_factor_test',inplace=True);out.append({'test':'real full-sequence future data / company adjustment perturbation','PASS_EXACT_FEATURE_AND_FACTOR':True,'note':'raw observed prices consumed, no post-adjustment columns; not a proof against vendor revisions of past raw observations'})
  # Actual fit data constructor, both targets, three real early/middle/late outer folds.
  fit=[]
  for year in [2020,2023,2026]:
   data=CandidateData(f'{year-3}-01-01',f'{year-2}-12-31');cut=pd.Timestamp(f'{year-1}-01-01');changed=target.copy();future_or_immature=changed.trade_date.ge(cut)|changed.label_available_at.ge(cut)
   for col in ['label_exit_rule_net','residual']:changed.loc[future_or_immature,col]=np.arange(future_or_immature.sum())*1e7
   for col in ['label_exit_rule_net','residual']:
    x=MatureCalculator(data,target,col,cut);y=MatureCalculator(data,changed,col,cut);assert x.identity['target_tensor_sha256']==y.identity['target_tensor_sha256'];torch.testing.assert_close(x.target,y.target,rtol=0,atol=0,equal_nan=True);fit.append({'outer_year':year,'target':col,'PASS_EXACT_REAL_FIT_INPUT':True,'mature_rows':x.identity['mature_target_rows'],'max_label_available':x.identity['max_training_label_available_at'],'full_counterfactual_PPO_retrain':'NOT_RUN; actual full-input tensors/qualification identical, frozen-seed PPO accesses only these tensors'})
  out.append({'test':'generation fit cutoff future/immature mutation','coverage':fit})
 (R/'real_entry_tests.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
