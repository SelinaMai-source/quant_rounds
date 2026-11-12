from pathlib import Path
import sys,ast,json,hashlib,traceback,importlib.util,time
import torch,numpy as np
from torch import nn
import torch.nn.functional as F
import math,collections
B=Path('/data1/stock-mxy/课题/一进二策略/research/round23_6_26_execution');C=B.parents[1]
torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
V=B/'vendor';S=C/'result/round23_6_round26_master/l0';S.mkdir(parents=True,exist_ok=True)
def selected(path,names,namespace=None):
 t=path.read_text();tree=ast.parse(t)
 ns={'torch':torch,'nn':nn,'F':F,'math':math,'np':np,'collections':collections}
 if namespace:ns.update(namespace)
 nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
 if len(nodes)!=len(names):raise RuntimeError('Missing official core definitions')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
 return ns

def step(model,x,y,lr):
 opt=torch.optim.Adam(model.parameters(),lr=lr)
 before=[v.detach().clone() for v in model.parameters()]
 opt.zero_grad();p=model(x);loss=F.mse_loss(p,y);loss.backward();opt.step()
 changed=sum(not torch.equal(a,b) for a,b in zip(before,model.parameters()))
 if changed==0 or not torch.isfinite(loss):raise RuntimeError('No finite model update')
 return {'loss':float(loss.detach()),'parameters_changed':changed,'parameter_count':sum(v.numel() for v in model.parameters()),'prediction_shape':list(p.shape)}

def run(method):
 r={'method':method,'seed':0,'level':'L0_ALGORITHM','data':'SYNTHETIC_ALGORITHM_TEST_NOT_REFERENCE','started_at':time.time()}
 try:
  if method=='TRA':
   p=V/'TRA/examples/benchmarks/TRA/src/model.py';n=selected(p,['LSTM','TRA','shoot_infs','sinkhorn'])
   m=n['LSTM'](input_size=16,hidden_size=256,num_layers=2,use_attn=True,dropout=.1)
   tra=n['TRA'](512,num_states=10,hidden_size=16,tau=1.,src_info='LR_TPE')
   opt=torch.optim.Adam(list(m.parameters())+list(tra.parameters()),lr=.0001)
   x=torch.randn(8,60,16);hist=torch.zeros(8,39,10);y=torch.randn(8)
   pred,allpred,prob=tra(m(x),hist);L=(allpred-y[:,None]).pow(2);L=L-L.min(dim=-1,keepdim=True).values
   P=n['sinkhorn'](-L,epsilon=.01);loss=(pred-y).pow(2).mean()-2.*prob.log().mul(P).sum(-1).mean()
   before=[v.detach().clone() for v in list(m.parameters())+list(tra.parameters())]
   loss.backward();opt.step();after=list(m.parameters())+list(tra.parameters())
   r.update(loss=float(loss.detach()),parameters_changed=sum(not torch.equal(a,b) for a,b in zip(before,after)),parameter_count=sum(v.numel() for v in after),prediction_shape=list(pred.shape),source_core='legacy src/model.py LSTM+TRA+author Sinkhorn and OT objective')
   r['status']='ALGORITHM_CORE_UPDATE_PASSED';r['limitations']='Full author TRAModel wrapper, dataset and L1 not covered by this finite core test'
  elif method=='MASTER':
   p=V/'MASTER';sys.path.insert(0,str(p));import master
   m=master.MASTER(158,256,4,2,.5,.5,158,221,5)
   r.update(step(m,torch.randn(40,8,221),torch.randn(40),1e-5));r['status']='ALGORITHM_UPDATE_PASSED'
   m.eval();x=torch.randn(40,8,221)
   with torch.no_grad():
    p1=m(x);perm=torch.randperm(40);p2=m(x[perm]);err=(p2-p1[perm]).abs().max().item()
   r['node_permutation_max_error']=err;r['all_prediction_nodes']=len(p1)
  elif method=='DoubleAdapt':
   src=V/'DoubleAdapt/src';sys.path.insert(0,str(src))
   # CPU-only compatibility patch. No architecture or update equation changes.
   text=(src/'net.py').read_text();patched=text.replace('self.device = torch.device("cuda")','self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")')
   import types
   net=types.ModuleType('net');exec(compile(patched,str(src/'net.py'),'exec'),net.__dict__);sys.modules['net']=net
   spec=importlib.util.spec_from_file_location('doubleadapt_manager',src/'model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
   # GRU forecast follows official qlib GRU core; no Ridge substitution.
   qpath=V/'TRA/qlib/contrib/model/pytorch_gru.py'
   q=selected(qpath,['GRUModel']);gru=q['GRUModel'](d_feat=6,hidden_size=64,num_layers=2,dropout=0.)
   manager=mod.DoubleAdaptManager(gru,lr_model=.01,lr_da=.01,lr_ma=.001,factor_num=6,x_dim=360,num_head=8,temperature=10,need_permute=False)
   x=torch.randn(8,60,6);task={'X_train':x,'y_train':torch.randn(8),'X_test':torch.randn(8,60,6),'y_test':torch.randn(8)}
   before=[v.detach().clone() for v in manager.framework.parameters()]
   pred=manager._run_task(task,'train');after=list(manager.framework.parameters())
   r.update(parameters_changed=sum(not torch.equal(a,b) for a,b in zip(before,after)),prediction_shape=list(pred.shape),prediction_finite=bool(np.isfinite(pred).all()),status='ALGORITHM_UPDATE_PASSED',compatibility_patch='cuda device -> available cuda else cpu; saved diff required',source='author DoubleAdaptManager._run_task with data and model adapters')
  elif method=='AlphaGen':
   sys.path.insert(0,str(V/'AlphaGen'))
   from alphagen.data.expression import Feature,FeatureType,Add,Mean
   from alphagen.data.calculator import TensorAlphaCalculator
   from alphagen.models.linear_alpha_pool import MseAlphaPool
   from alphagen_qlib.stock_data import StockData
   from alphagen_qlib.calculator import QLibStockDataCalculator
   import pandas as pd
   class Preloaded(StockData):
    @classmethod
    def _init_qlib(cls):
     # The author evaluator consumes an explicitly supplied synthetic tensor.
     # Bypass only the unused provider startup; expression code is unchanged.
     pass
   data=Preloaded('synthetic','2000-01-01','2000-01-30',max_backtrack_days=100,max_future_days=0,device=torch.device('cpu'),preloaded_data=(torch.rand(130,6,20)+1,pd.date_range('2000-01-01',periods=130),pd.Index(range(20))))
   calc=QLibStockDataCalculator(data,Mean(Feature(FeatureType.CLOSE),5));pool=MseAlphaPool(capacity=5,calculator=calc,l1_alpha=.005,device=torch.device('cpu'))
   for f in FeatureType:pool.try_new_expr(Feature(f))
   r['pool_size']=pool.size;r['pool_weights']=pool.weights.tolist();r['pool_evaluations']=pool.eval_cnt
   from alphagen.rl.env.wrapper import AlphaEnv
   from alphagen.rl.env.core import AlphaEnvCore
   from sb3_contrib import MaskablePPO
   env=AlphaEnv(pool,device=torch.device('cpu'))
   from alphagen.rl.policy import LSTMSharedNet
   model=MaskablePPO('MlpPolicy',env,policy_kwargs={'features_extractor_class':LSTMSharedNet,'features_extractor_kwargs':dict(n_layers=2,d_model=128,dropout=.1,device=torch.device('cpu'))},gamma=1.,ent_coef=.01,n_steps=32,batch_size=16,n_epochs=1,device='cpu',seed=0,verbose=0)
   before=[v.detach().clone() for v in model.policy.parameters()];model.learn(total_timesteps=64)
   r['policy_parameters_changed']=sum(not torch.equal(a,b) for a,b in zip(before,model.policy.parameters()))
   r['status']='ALGORITHM_CORE_UPDATE_PASSED';r['limitations']='author expression evaluation, synergistic pool and original LSTM policy on synthetic OHLCV tensors; finite 64-token rollout test, not L1'
  elif method=='DeepLOB':
   notebook=V/'DeepLOB_api/run_train_pytorch.ipynb';nb=json.loads(notebook.read_text())
   code=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and 'class deeplob(' in ''.join(c['source']))
   ns={'torch':torch,'nn':nn,'device':torch.device('cpu')};exec(compile(code,str(notebook)+'::author_class','exec'),ns)
   m=ns['deeplob'](3);opt=torch.optim.Adam(m.parameters(),lr=.0001)
   before=[v.detach().clone() for v in m.parameters()];x=torch.randn(4,1,100,40);y=torch.tensor([0,1,2,1])
   p=m(x);loss=nn.CrossEntropyLoss()(p,y);loss.backward();opt.step()
   r.update(status='ALGORITHM_UPDATE_PASSED',loss=float(loss.detach()),parameters_changed=sum(not torch.equal(a,b) for a,b in zip(before,m.parameters())),prediction_shape=list(p.shape),author_cell_sha256=hashlib.sha256(code.encode()).hexdigest(),notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest(),reference_config={'k':4,'T':100,'epochs':50,'batch_size':64,'optimizer':'Adam','lr':.0001,'loss':'CrossEntropyLoss applied to author softmax output, unchanged'},limitations='Synthetic tensors for L0; not FI-2010 L1 or A-share LOB')
  elif method=='AlphaForge':
   p=V/'AlphaForge';sys.path.insert(0,str(p))
   # Load unchanged official neural core definitions without data-loader side effects.
   ng=selected(p/'gan/network/generater.py',['NetG_DCGAN'])['NetG_DCGAN'](48,100,20,128)
   npred=selected(p/'gan/network/predictor.py',['NetP'])['NetP'](48,128,20)
   z=torch.randn(8,100);out=ng(z);scores=npred(out)
   opt=torch.optim.Adam(list(ng.parameters())+list(npred.parameters()),lr=.001)
   before=[v.detach().clone() for v in list(ng.parameters())+list(npred.parameters())]
   loss=(scores.squeeze()-torch.randn(8)).square().mean();loss.backward();opt.step()
   r.update(parameters_changed=sum(not torch.equal(a,b) for a,b in zip(before,list(ng.parameters())+list(npred.parameters()))),loss=float(loss.detach()),generated_tensor_shape=list(out.shape),status='ALGORITHM_COMPONENT_UPDATE_PASSED',limitations='Official generator and predictor updated; full adversarial mining + dynamic combination is NOT run')
  else:raise ValueError('unknown method')
 except Exception as e:
  r.update(status='FAILED_RUNTIME',error_type=type(e).__name__,error=str(e));(S/(method+'.traceback.txt')).write_text(traceback.format_exc())
 r['elapsed_seconds']=time.time()-r['started_at'];(S/(method+'.json')).write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
if __name__=='__main__':run(sys.argv[1])
