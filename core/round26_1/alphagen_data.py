"""Locked AlphaGen StockData tensors: full stock history first, candidates second."""
from pathlib import Path
from collections import OrderedDict
import sys,json,hashlib
import numpy as np,pandas as pd,torch
from core.round26_1.market import bars,factor_candidates,calibrated_residual,KEY,C,B,R
V=C/'research/round23_6_26_execution/vendor/AlphaGen';sys.path.insert(0,str(V))
from alphagen_qlib.stock_data import StockData,FeatureType
from alphagen.data.calculator import TensorAlphaCalculator
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen.utils.correlation import batch_pearsonr
class PreloadedMarket(StockData):
 @classmethod
 def _init_qlib(cls):pass

def ast_dump(expr):
 from alphagen.data.expression import Feature,Constant
 if isinstance(expr,Feature):return {'type':'Feature','feature':expr._feature.name}
 if isinstance(expr,Constant):return {'type':'Constant','value':expr.value}
 if hasattr(expr,'_lhs'):
  result={'type':type(expr).__name__,'args':[ast_dump(expr._lhs),ast_dump(expr._rhs)]}
  if hasattr(expr,'_delta_time'):result['delta']=expr._delta_time
  return result
 if hasattr(expr,'_operand'):
  result={'type':type(expr).__name__,'args':[ast_dump(expr._operand)]}
  if hasattr(expr,'_delta_time'):result['delta']=expr._delta_time
  return result
 raise ValueError('Unsupported original AST '+type(expr).__name__)
def ast_load(node):
 from alphagen import config
 from alphagen.data.expression import Feature,Constant
 if node['type']=='Feature':return Feature(FeatureType[node['feature']])
 if node['type']=='Constant':return Constant(float(node['value']))
 cls=next((c for c in config.OPERATORS if c.__name__==node['type']),None)
 if cls is None:raise ValueError('AST class outside frozen whitelist')
 args=[ast_load(n) for n in node['args']]
 if 'delta' in node:
  if int(node['delta'])<=0 or int(node['delta']) not in config.DELTA_TIMES:raise ValueError('Future or nonwhitelist feature time offset')
  args.append(int(node['delta']))
 return cls(*args)
def guard(expr):
 node=ast_dump(expr);ast_load(node);return node

class CandidateData:
 def __init__(self,start,end):
  full=bars();q=factor_candidates();q=q.loc[q.trade_date.between(pd.Timestamp(start),pd.Timestamp(end))].copy();self.keys=q[KEY].copy();calendar=pd.DatetimeIndex(full.index.get_level_values('trade_date').unique().sort_values());a=int(calendar.searchsorted(pd.Timestamp(start)));z=int(calendar.searchsorted(pd.Timestamp(end),side='right'));assert a>=100 and z>a
  dates=calendar[a-100:z];ids=pd.Index(sorted(q.instrument_id.unique()));fields=['open','close','high','low','volume','vwap'];idx=pd.MultiIndex.from_product([dates,ids],names=KEY);frame=full[fields].reindex(idx);value=frame.to_numpy('float32').reshape(len(dates),len(ids),6).transpose(0,2,1).copy();tensor=torch.from_numpy(value)
  self.data=PreloadedMarket(list(ids),str(calendar[a].date()),str(calendar[z-1].date()),max_backtrack_days=100,max_future_days=0,device=torch.device('cpu'),preloaded_data=(tensor,dates,ids));self.dates=calendar[a:z];self.ids=ids;self.row_day=q.trade_date.map(pd.Series(np.arange(len(self.dates)),index=self.dates)).to_numpy('int64');self.row_stock=q.instrument_id.map(pd.Series(np.arange(len(ids)),index=ids)).to_numpy('int64');self.row_slot=q.groupby('trade_date').cumcount().to_numpy('int64');assert len(q)==len(self.row_day) and self.row_slot.max()<1024
  self.mask=torch.zeros((len(self.dates),1024),dtype=torch.bool);self.mask[self.row_day,self.row_slot]=True;self.cache=OrderedDict()
  self.identity={'start':start,'end':end,'market_tensor_shape':list(tensor.shape),'market_tensor_sha256':hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),'candidate_keys_sha256':hashlib.sha256(pd.util.hash_pandas_object(self.keys,index=False).values.tobytes()).hexdigest(),'candidate_rows':len(q),'calendar_days':len(self.dates),'backtrack':100,'future_days':0,'features':fields,'prediction_target_mask':False}
 def values(self,expr):
  key=(self.data.data._version,self.mask._version,json.dumps(guard(expr),sort_keys=True))
  if key in self.cache:
   v=self.cache.pop(key);self.cache[key]=v;return v
  with torch.no_grad():
   complete=expr.evaluate(self.data);assert complete.shape==(len(self.dates),len(self.ids));mapped=torch.full(self.mask.shape,torch.nan);mapped[self.row_day,self.row_slot]=complete[self.row_day,self.row_stock];result=normalize_by_day(mapped);result[~self.mask]=torch.nan
  self.cache[key]=result
  while len(self.cache)>64:self.cache.popitem(last=False)
  return result
 def frame(self,exprs,weights):
  with torch.no_grad():v=sum(self.values(e)*float(w) for e,w in zip(exprs,weights))
  out=self.keys.copy();out['z_AG']=v[self.row_day,self.row_slot].numpy();return out

class MatureCalculator(TensorAlphaCalculator):
 def __init__(self,domain,targets,column,cutoff):
  self.domain=domain;t=domain.keys.merge(targets[KEY+[column,'label_available_at']],on=KEY,how='left',validate='one_to_one');valid=pd.to_datetime(t.label_available_at).lt(pd.Timestamp(cutoff))&np.isfinite(t[column]);raw=torch.full(domain.mask.shape,torch.nan);y=t[column].to_numpy('float32');y[~valid.to_numpy()]=np.nan;raw[domain.row_day,domain.row_slot]=torch.from_numpy(y);known=torch.isfinite(raw);self.active=known.sum(1).ge(2);self.target_mask=known[self.active];value=normalize_by_day(raw);value[~known]=torch.nan;super().__init__(value[self.active]);self.cache=OrderedDict();self.single_cache=OrderedDict();self.mutual_cache=OrderedDict()
  if not self.active.any():raise RuntimeError('No mature target dates for generation/validation')
  self.identity={'target_column':column,'cutoff':str(cutoff),'active_dates':int(self.active.sum()),'mature_target_rows':int(known.sum()),'max_training_label_available_at':str(pd.to_datetime(t.loc[valid,'label_available_at']).max()),'target_tensor_sha256':hashlib.sha256(raw.numpy().tobytes()).hexdigest(),'feature_domain':domain.identity}
 @property
 def n_days(self):return int(self.active.sum())
 def evaluate_alpha(self,expr):
  value=self.domain.values(expr)[self.active].clone();value[~self.target_mask]=torch.nan;return value

 def _metric_stamp(self):
  return (self.domain.data.data._version,self.domain.mask._version,self.target._version,self.active._version,self.target_mask._version)
 def calc_single_IC_ret(self,expr):
  key=(self._metric_stamp(),json.dumps(guard(expr),sort_keys=True))
  if key in self.single_cache:
   value=self.single_cache.pop(key);self.single_cache[key]=value;return value
  result=super().calc_single_IC_ret(expr);self.single_cache[key]=result
  while len(self.single_cache)>4096:self.single_cache.popitem(last=False)
  return result
 def calc_mutual_IC(self,left,right):
  key=(self._metric_stamp(),json.dumps(guard(left),sort_keys=True),json.dumps(guard(right),sort_keys=True))
  if key in self.mutual_cache:
   value=self.mutual_cache.pop(key);self.mutual_cache[key]=value;return value
  result=super().calc_mutual_IC(left,right);self.mutual_cache[key]=result
  while len(self.mutual_cache)>16384:self.mutual_cache.popitem(last=False)
  return result
