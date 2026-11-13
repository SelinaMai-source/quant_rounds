"""Observed complete stock daily sequences, then known-candidate mapping."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from core.research_integrity_v5.cache import identity,sha
C=Path('/data1/stock-mxy/课题/一进二策略');B=C/'research/round23_7_26_1_execution';R=C/'result/round26_1/D3';SNAP=Path('/data1/stock-mxy/data/datasets/hermes-cn-a-2010-20260724-v7-all-features')
KEY=['trade_date','instrument_id'];_bars=None;_qualified=None;_residual_cache=None

def bars():
 global _bars
 if _bars is not None:return _bars
 from zyquant.data import DataSnapshot
 snapshot=DataSnapshot(SNAP);f=snapshot.table('daily_raw',start=pd.Timestamp('2016-01-01').date(),end=pd.Timestamp('2026-07-24').date(),cutoff=snapshot.metadata.as_of_date,fields=['trade_date','instrument_id','open','close','high','low','volume','amount'])
 f['trade_date']=pd.to_datetime(f.trade_date);f['instrument_id']=f.instrument_id.astype(str);f=f.sort_values(KEY);assert not f.duplicated(KEY).any();f['vwap']=f.amount/f.volume.where(f.volume.gt(0));valid=f.volume.gt(0)&f.amount.gt(0);outside=valid&((f.vwap<f.low-.02)|(f.vwap>f.high+.02));inside=valid&~outside
 # Unknown inconsistent source rows are missing observations, not a price substitute.
 f.loc[outside,'vwap']=np.nan;R.mkdir(parents=True,exist_ok=True)
 meta={'snapshot_manifest':identity(SNAP/'manifest.json'),'rows':len(f),'start':str(f.trade_date.min()),'end':str(f.trade_date.max()),'stock_ids':f.instrument_id.nunique(),'field_units':{'prices':'CNY/share raw observed','amount':'CNY','volume':'shares','vwap':'amount/volume'},'positive_amount_volume_rows':int(valid.sum()),'VWAP_inside_daily_range_with2cent_tolerance':int(inside.sum()),'VWAP_inconsistent_rows_masked':int(outside.sum()),'features':['open','close','high','low','volume','vwap'],'post_adjustment_columns_consumed':[],'event_time':'exchange daily close reconstructed; decisions in frozen downstream legacy next-open lane','vendor_source_updated_at':'not treated as proof of historical first-publication; market observations reconstructed, source revision limitation retained','current_listing_list_used':False}
 (R/'market_units_and_time_audit.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));_bars=f.set_index(KEY).sort_index();return _bars

def qualified():
 global _qualified
 if _qualified is not None:return _qualified
 from core.round7.contract import FIVE_HEAD,TEN_HEAD
 parts=[];full=[]
 for h in [FIVE_HEAD,TEN_HEAD]:
  p=pd.read_parquet(C/f'result/round22_3/data/{h.name}_candidate.parquet',columns=list(dict.fromkeys(KEY+list(h.features)+['next1_date','actual_exit_date','label_exit_rule_net'])))
  p['trade_date']=pd.to_datetime(p.trade_date);p['instrument_id']=p.instrument_id.astype(str);base=p[list(h.features)].replace([np.inf,-np.inf],np.nan).notna().all(axis=1);p=p.loc[base&np.isfinite(p.d1_open_gap)&p.d1_open_gap.ge(-.01)];parts.append(p[KEY]);full.append(p)
 q=full[1].merge(parts[0],on=KEY,how='inner',validate='one_to_one');q['label_available_at']=pd.to_datetime(q.actual_exit_date)+pd.Timedelta(hours=15);q=q.sort_values(KEY).reset_index(drop=True);assert q.groupby('trade_date').size().max()<=1024
 _qualified=q;return q

def factor_candidates():
 # The factor domain is D-known first-board eligibility, NOT next-open gap or labels.
 p=pd.read_parquet(C/'result/round22_3/data/ten_candidate.parquet',columns=KEY+['is_first_board_hard','universe_eligible','label_exit_rule_net','actual_exit_date'])
 p['trade_date']=pd.to_datetime(p.trade_date);p['instrument_id']=p.instrument_id.astype(str)
 p=p.loc[p.is_first_board_hard.fillna(False).astype(bool)&p.universe_eligible.fillna(False).astype(bool)].sort_values(KEY).reset_index(drop=True)
 p['label_available_at']=pd.to_datetime(p.actual_exit_date)+pd.Timedelta(hours=15)
 assert not p.duplicated(KEY).any() and p.groupby('trade_date').size().max()<=1024
 return p

def baseline_oof():
 """Only existing XGB baseline-bank predictions, no current-line winner."""
 q=qualified();allparts=[];reg=[]
 for year in range(2016,2027):
  frame=q.loc[q.trade_date.dt.year.eq(year),KEY].copy()
  for head in ['five','ten']:
   p=C/f'result/round23_6/expert_bank/{head}/{year}/predictions.parquet';z=pd.read_parquet(p,columns=KEY+['xgb']).rename(columns={'xgb':head});frame=frame.merge(z,on=KEY,how='left',validate='one_to_one');reg.append(identity(p))
  for col in ['five','ten']:frame[col]=frame.groupby('trade_date')[col].rank(pct=True)
  frame['v5_score']=.9*frame.ten+.1*frame.five;allparts.append(frame)
 out=pd.concat(allparts,ignore_index=True).merge(q[KEY+['label_exit_rule_net','label_available_at']],on=KEY,validate='one_to_one');out.to_parquet(R/'baseline_true_OOF_scores.parquet',index=False)
 (R/'baseline_OOF_source_manifest.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2));return out

def calibrated_residual():
 global _residual_cache
 if _residual_cache is not None:return _residual_cache
 from sklearn.linear_model import Ridge
 data=baseline_oof();out=[];registry=[]
 for year in range(2017,2027):
  cutoff=pd.Timestamp(f'{year}-01-01');train=data.loc[data.trade_date.ge(f'{year-3}-01-01')&data.trade_date.lt(cutoff)&data.label_available_at.lt(cutoff)].dropna(subset=['v5_score','label_exit_rule_net']);pred=data.loc[data.trade_date.dt.year.eq(year)].copy();assert len(train)>=200
  model=Ridge(alpha=10);model.fit(train[['v5_score']],train.label_exit_rule_net);pred['mu_v5']=model.predict(pred[['v5_score']]);pred['residual']=pred.label_exit_rule_net-pred.mu_v5;pred['calibration_fit_cutoff']=cutoff;out.append(pred)
  registry.append({'year':year,'fit_cutoff':str(cutoff),'train_start':str(train.trade_date.min()),'train_end':str(train.trade_date.max()),'max_training_label_available_at':str(train.label_available_at.max()),'alpha':10,'slope':float(model.coef_[0]),'intercept':float(model.intercept_),'rows':len(train),'training_matrix_sha256':hashlib.sha256(pd.util.hash_pandas_object(train,index=False).values.tobytes()).hexdigest()})
 result=pd.concat(out,ignore_index=True);result.to_parquet(R/'residual_true_OOF.parquet',index=False);(R/'residual_calibrator_models.json').write_text(json.dumps(registry,indent=2));_residual_cache=result;return result
