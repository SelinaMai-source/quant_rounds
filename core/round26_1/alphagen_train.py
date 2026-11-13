"""Real A-share original LSTM MaskablePPO/MseAlphaPool generation, exact budgets."""
from pathlib import Path
import json,os,sys,time,hashlib,pickle,random,copy,resource
import numpy as np,pandas as pd,torch
from core.round26_1.alphagen_data import CandidateData,MatureCalculator,ast_dump,ast_load,guard,V
from core.round26_1.market import C,B,R,calibrated_residual,KEY
from core.research_integrity_v5.cache import identity,sha,fingerprint
from core.research_integrity_v5.resources import measured
from core.research_integrity_v5.experiment import full_contract
from core.research_integrity_v5.cache import source_manifest
sys.path.insert(0,str(V))
import alphagen.config as author_config
author_config.MAX_EXPR_LENGTH=20
from alphagen.rl.env.core import AlphaEnvCore
from alphagen.rl.env.wrapper import AlphaEnvWrapper
from alphagen.rl.policy import LSTMSharedNet
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.utils import reseed_everything
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback


def atomic_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);data=json.dumps(value,ensure_ascii=False,sort_keys=True,default=str,allow_nan=False).encode();temp=path.with_suffix(path.suffix+'.tmp');temp.write_bytes(data)
 if path.exists() and path.read_bytes()!=data:path.rename(path.with_name(path.name+'.superseded.'+str(time.time_ns())))
 os.replace(temp,path)
def pool_state(pool):
 return {'size':pool.size,'ASTs':[ast_dump(e) for e in pool.exprs[:pool.size]],'weights':pool.weights.tolist(),'single_ics':pool.single_ics.tolist(),'mutual_ics':pool._mutual_ics.tolist(),'best_obj':float(pool.best_obj),'best_ic_ret':float(getattr(pool,'best_ic_ret',0)),'failure_cache':sorted(pool._failure_cache),'eval_cnt':pool.eval_cnt}
def restore_pool(pool,state):
 pool.size=state['size'];pool.exprs=[ast_load(x) for x in state['ASTs']]+[None]*(pool.capacity+1-pool.size);pool._weights[:pool.size]=state['weights'];pool.single_ics[:]=state['single_ics'];pool._mutual_ics[:]=state['mutual_ics'];pool.best_obj=state['best_obj'];pool.best_ic_ret=state['best_ic_ret'];pool._failure_cache=set(state['failure_cache']);pool.eval_cnt=state['eval_cnt']

class AuditPool(MseAlphaPool):
 def __init__(self,*args,directory,**kwargs):
  super().__init__(*args,**kwargs);self.directory=Path(directory);self.attempts=0;self.valid_evaluations=0;self.unique_valid=set();self.invalid=0
 def try_new_expr(self,expr):
  self.attempts+=1;row={'attempt':self.attempts,'expression':str(expr)}
  try:
   row['AST']=guard(expr);result=super().try_new_expr(expr);self.valid_evaluations+=1;self.unique_valid.add(json.dumps(row['AST'],sort_keys=True));row.update(status='SOURCE_EVALUATED',reward=float(result),pool_size=self.size,pool_eval_cnt=self.eval_cnt)
  except Exception as e:
   row.update(status='INVALID_OR_FAILED',exception=type(e).__name__,message=str(e)[:300]);self.invalid+=1
   with (self.directory/'all_expression_evaluations.jsonl').open('a') as out:out.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
   raise
  with (self.directory/'all_expression_evaluations.jsonl').open('a') as out:out.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
  return result

def save_resume(model,pool,env,directory):
 directory=Path(directory);state={'pool':pool_state(pool),'attempts':pool.attempts,'valid_evaluations':pool.valid_evaluations,'unique_valid':sorted(pool.unique_valid),'invalid':pool.invalid,'wrapper_state':env.state.copy(),'counter':env.counter,'tokens':env.env._tokens,'builder':env.env._builder,'env_eval_cnt':env.env.eval_cnt,'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),'num_timesteps':model.num_timesteps,'n_updates':model._n_updates}
 # Two complete state slots alternate; CURRENT only switches after all bytes exist.
 slot=directory/'resume_slots'/str(model.num_timesteps);slot.mkdir(parents=True,exist_ok=True);tmp=slot/'policy.partial.zip';model.save(tmp);os.replace(tmp,slot/'policy.zip');payload=slot/'state.partial.pkl';payload.write_bytes(pickle.dumps(state,protocol=5));os.replace(payload,slot/'state.pkl');atomic_json(slot/'READY.json',{'step':model.num_timesteps,'policy_sha256':sha(slot/'policy.zip'),'state_sha256':sha(slot/'state.pkl')});atomic_json(directory/'RESUME.json',{'slot':str(slot),'step':model.num_timesteps});return slot

def load_resume(pool,env,directory):
 directory=Path(directory);p=directory/'RESUME.json'
 if not p.exists():return None
 slot=Path(json.loads(p.read_text())['slot']);ready=json.loads((slot/'READY.json').read_text());assert sha(slot/'policy.zip')==ready['policy_sha256'] and sha(slot/'state.pkl')==ready['state_sha256'];state=pickle.loads((slot/'state.pkl').read_bytes());restore_pool(pool,state['pool']);pool.attempts=state['attempts'];pool.valid_evaluations=state['valid_evaluations'];pool.unique_valid=set(state['unique_valid']);pool.invalid=state['invalid'];env.state=state['wrapper_state'];env.counter=state['counter'];env.env._tokens=state['tokens'];env.env._builder=state['builder'];env.env.eval_cnt=state['env_eval_cnt']
 model=AuditPPO.load(slot/'policy.zip',env=env,device='cpu',force_reset=False);random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy']);torch.set_rng_state(state['rng_torch']);model.audit_pool=pool;model.audit_env=env;model.audit_directory=directory;return model
class AuditPPO(MaskablePPO):
 def _excluded_save_params(self):
  return super()._excluded_save_params()+['audit_pool','audit_env','audit_directory']
 def train(self):
  super().train();save_resume(self,self.audit_pool,self.audit_env,self.audit_directory)
class BudgetCallback(BaseCallback):
 def __init__(self,pool,validation,directory,seed,year,arm):
  super().__init__();self.pool=pool;self.validation=validation;self.directory=Path(directory);self.seed=seed;self.year=year;self.arm=arm;self.last_checkpoint=-1;self.stop_reason=None;self.last_heartbeat=time.monotonic()
 def checkpoint(self,terminal=False):
  step=int(self.model.num_timesteps)
  if self.last_checkpoint==step:return
  if not self.pool.size:ic=None
  else:
   ic=float(self.pool.test_ensemble(self.validation)[0]);ic=ic if np.isfinite(ic) else None
  state={'env_steps':step,'seed':self.seed,'outer_year':self.year,'arm':self.arm,'validation_pool_IC':ic,'training_pool':pool_state(self.pool),'effective_expressions':self.pool.valid_evaluations,'unique_valid_expressions':len(self.pool.unique_valid),'expression_attempts':self.pool.attempts,'invalid_expressions':self.pool.invalid,'terminal':terminal,'PPO_update_epochs_counter':self.model._n_updates,'PPO_optimizer_minibatch_steps':self.model._n_updates*16};atomic_json(self.directory/'selection_checkpoints'/f'{step:06d}.json',state);self.last_checkpoint=step;atomic_json(self.directory/'PROGRESS.json',state);print('AlphaGen progress',self.arm,self.year,self.seed,step,'expr',self.pool.valid_evaluations,'pool',self.pool.size,'valIC',ic,flush=True)
 def _on_step(self):
  step=int(self.model.num_timesteps)
  if time.monotonic()-self.last_heartbeat>=60:
   atomic_json(self.directory/'HEARTBEAT.json',{'env_steps':step,'valid_expressions':self.pool.valid_evaluations,'attempts':self.pool.attempts,'pool_size':self.pool.size,'budget':100000});print('AlphaGen heartbeat',self.arm,self.year,self.seed,step,'pool',self.pool.size,flush=True);self.last_heartbeat=time.monotonic()
  if step%10000==0:self.checkpoint()
  if step>=100000 or self.pool.valid_evaluations>=20000:
   self.stop_reason='ENV_STEP_BUDGET' if step>=100000 else 'VALID_EXPRESSION_BUDGET';self.checkpoint(terminal=True);return False
  import psutil
  if psutil.Process().memory_info().rss>16*1024**3:
   self.stop_reason='PAUSED_RESOURCE_RSS_LIMIT';return False
  return True

def run_fold(arm,year,seed,resume=True):
 torch.set_num_threads(2);reseed_everything(seed)
 training=CandidateData(f'{year-3}-01-01',f'{year-2}-12-31');validation=CandidateData(f'{year-1}-01-01',f'{year-1}-12-31');targets=calibrated_residual();column='label_exit_rule_net' if arm=='AG_RAW' else 'residual'
 # Same eligible historical target rows for both arms; all factor inference keys remain D-known.
 train=MatureCalculator(training,targets,column,f'{year-1}-01-01');val=MatureCalculator(validation,targets,column,f'{year}-01-01');source=json.loads((R/'CONTRACT.yaml').read_text());contract={'protocol':source,'seed':seed,'arm':arm,'outer_year':year,'train':train.identity,'validation':val.identity,'code':[identity(C/'core/round26_1'/n) for n in ['alphagen_data.py','alphagen_train.py','market.py']],'source_author':{'commit':source['source_commit'],'files':[identity(V/p) for p in ['alphagen/data/expression.py','alphagen/models/linear_alpha_pool.py','alphagen/rl/policy.py','alphagen/rl/env/core.py','alphagen/rl/env/wrapper.py']]},'labels':identity(R/'residual_true_OOF.parquet'),'runtime':{'torch':torch.__version__,'CPU_threads':2,'CUDA_visible':torch.cuda.is_available()}}
 inputs=[R/'residual_true_OOF.parquet',R/'baseline_OOF_source_manifest.json',C/'result/round23_6/frozen_components.parquet']+[C/f'result/round22_3/data/{h}_{k}.parquet' for h in ['five','ten'] for k in ['training','candidate']]
 from core.round26_1.market import SNAP
 inputs+=list((SNAP/'daily_raw').rglob('*.parquet'))+[SNAP/'manifest.json']
 contract=full_contract(contract,inputs,upstream=[identity(R/'residual_calibrator_models.json')],seed=seed,features=['open','close','high','low','volume','vwap'],label=column+'; Ten original D+1 modeled buy to actual first executable close net; RES=y-temporalOOF_Ridge10_mu',maturity='actual_exit_date exchange15:00, each generation/validation/refit split cutoff')
 contract['code']=source_manifest([C/'core/round26_1/alphagen_train.py',C/'core/round26_1/alphagen_data.py',C/'core/round26_1/market.py',C/'core/research_integrity_v5/cache.py',C/'core/research_integrity_v5/resources.py',C/'core/research_integrity_v5/experiment.py',C/'core/stock_readonly.py',C/'core/paths.py',C/'core/round7/contract.py'])
 contract['author_code_closure']=source_manifest([V/'alphagen',V/'alphagen_qlib'])
 fp=fingerprint(contract);directory=B/'alphagen_runs'/arm/str(year)/str(seed)/fp;directory.mkdir(parents=True,exist_ok=True);atomic_json(directory/'CONTRACT.json',contract)
 if (directory/'COMPLETE.json').exists():
  complete=json.loads((directory/'COMPLETE.json').read_text());assert sha(directory/'factor_predictions.parquet')==complete['factor_prediction_sha256'] and sha(directory/'FROZEN_FACTOR.json')==complete['factor_model_sha256'] and sha(directory/'CONTRACT.json')==complete['contract_sha256'];print('AlphaGen COMPLETE cache hit',arm,year,seed,fp,flush=True);return directory
 with measured(directory/'RESOURCES.json'):
  pool=AuditPool(capacity=20,calculator=train,l1_alpha=.005,device=torch.device('cpu'),directory=directory);env=AlphaEnvWrapper(AlphaEnvCore(pool,device=torch.device('cpu'),print_expr=False));model=load_resume(pool,env,directory) if resume else None
  if model is None:
   model=AuditPPO('MlpPolicy',env,policy_kwargs={'features_extractor_class':LSTMSharedNet,'features_extractor_kwargs':{'n_layers':2,'d_model':128,'dropout':.1,'device':torch.device('cpu')}},gamma=1.,ent_coef=.01,batch_size=128,n_steps=2048,n_epochs=10,learning_rate=.0003,clip_range=.2,gae_lambda=.95,device='cpu',seed=seed,verbose=0)
  model.audit_pool=pool;model.audit_env=env;model.audit_directory=directory
  atomic_json(directory/'RESOLVED_PPO.json',{'architecture':str(model.policy),'parameters':sum(p.numel() for p in model.policy.parameters()),'source_MAX_EXPR_LENGTH':author_config.MAX_EXPR_LENGTH,'mapping':'20 author state slots includes BEG internal counter; at most19 appended expression tokens before automatic termination, SEP separate','model_class':'original MaskablePPO with audit-only train hook','LSTM_class':'original LSTMSharedNet','optimizer_defaults':{k:v for k,v in model.policy.optimizer.defaults.items() if k!='params'},'n_steps':model.n_steps,'batch_size':model.batch_size,'n_epochs':model.n_epochs})
  callback=BudgetCallback(pool,val,directory,seed,year,arm);model.learn(total_timesteps=max(0,100000-model.num_timesteps),reset_num_timesteps=False,callback=callback)
  if callback.stop_reason=='PAUSED_RESOURCE_RSS_LIMIT':
   atomic_json(directory/'STATUS.json',{'status':'PAUSED_RESOURCE','reason':callback.stop_reason,'resume_from_last_fully_optimized_rollout':json.loads((directory/'RESUME.json').read_text()) if (directory/'RESUME.json').exists() else None,'full_env_step_budget':100000,'architecture_unchanged':True});return directory
  callback.checkpoint(terminal=True);records=[json.loads(p.read_text()) for p in (directory/'selection_checkpoints').glob('*.json')];valid=[r for r in records if r['validation_pool_IC'] is not None and r['training_pool']['size']>0]
  if not valid:atomic_json(directory/'STATUS.json',{'status':'FAILED_NO_VALID_AST','steps':model.num_timesteps});return directory
  selected=max(valid,key=lambda r:(r['validation_pool_IC'],-r['env_steps']));atomic_json(directory/'SELECTED_CHECKPOINT.json',selected);exprs=[ast_load(a) for a in selected['training_pool']['ASTs']]
  # AST frozen; only original pool weights refitted on the mature prior3year domain.
  combined=CandidateData(f'{year-3}-01-01',f'{year-1}-12-31');calc=MatureCalculator(combined,targets,column,f'{year}-01-01');refit=MseAlphaPool(capacity=20,calculator=calc,l1_alpha=.005,device=torch.device('cpu'));refit.force_load_exprs(exprs);assert [ast_dump(x) for x in refit.exprs[:refit.size]]==[ast_dump(x) for x in exprs]
  scale_frame=combined.frame(exprs,refit.weights);scale_mean=float(scale_frame.z_AG.mean());scale_std=float(scale_frame.z_AG.std(ddof=0))
  if not np.isfinite(scale_std) or scale_std<=1e-12:
   atomic_json(directory/'STATUS.json',{'status':'FAILED_DEGENERATE_FACTOR','generation_steps':model.num_timesteps,'training_factor_std':scale_std});return directory
  predicted=CandidateData(f'{year}-01-01',f'{year}-12-31' if year<2026 else '2026-07-24');frame=predicted.frame(exprs,refit.weights);frame['arm']=arm;frame['seed']=seed;frame['factor_outer_year']=year;frame['factor_fit_cutoff']=pd.Timestamp(f'{year}-01-01');frame['factor_model_hash']=hashlib.sha256(json.dumps({'ASTs':[ast_dump(e) for e in exprs],'weights':refit.weights.tolist()},sort_keys=True).encode()).hexdigest();frame.to_parquet(directory/'factor_predictions.parquet',index=False)
  atomic_json(directory/'FROZEN_FACTOR.json',{'ASTs':[ast_dump(e) for e in exprs],'weights':refit.weights.tolist(),'prior_training_scale_mean':scale_mean,'prior_training_scale_std':scale_std,'selected_generation_step':selected['env_steps'],'selected_validation_pool_IC':selected['validation_pool_IC'],'max_training_label_available_at':calc.identity['max_training_label_available_at'],'fit_cutoff':f'{year}-01-01','generation_env_steps':model.num_timesteps,'attempted_expressions':pool.attempts,'valid_evaluations':pool.valid_evaluations,'unique_valid_expressions':len(pool.unique_valid),'invalid_expressions':pool.invalid,'PPO_update_epochs_counter':model._n_updates,'PPO_optimizer_minibatch_steps':model._n_updates*16,'last_fully_optimized_rollout':json.loads((directory/'RESUME.json').read_text())['step'],'original_source_and_input_fingerprint':fp,'reference_scope':'budgeted A-share core adaptation, not full paper CSI300 reproduction'})
  atomic_json(directory/'COMPLETE.json',{'status':'FACTOR_OOF_YEAR_COMPLETE','factor_prediction_sha256':sha(directory/'factor_predictions.parquet'),'factor_model_sha256':sha(directory/'FROZEN_FACTOR.json'),'contract_sha256':sha(directory/'CONTRACT.json')})
 return directory
