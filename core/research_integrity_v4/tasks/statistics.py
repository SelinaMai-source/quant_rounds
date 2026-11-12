from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from arch.bootstrap import SPA
C=Path('/data1/stock-mxy/课题/一进二策略');R=C/'result/round23_6'
INITIAL=300000.

def sharpe(a,axis=0):
 s=np.std(a,axis=axis,ddof=1);return np.divide(np.mean(a,axis=axis),s,out=np.zeros_like(s),where=s>0)*np.sqrt(252)
def risk(nav):
 a=np.asarray(nav,float);r=a[1:]/a[:-1]-1
 path=np.r_[INITIAL,a];dd=path/np.maximum.accumulate(path)-1
 es=-np.mean(r[r<=np.quantile(r,.05)])
 duration=0;longest=0
 for v in dd:
  duration=duration+1 if v<0 else 0;longest=max(longest,duration)
 worst63=float(np.min(np.prod(1+np.lib.stride_tricks.sliding_window_view(r,63),axis=1)-1))
 return dict(UI=float(np.sqrt(np.mean(dd**2))),MaxDD=float(dd.min()),ES95=float(es),worst63=worst63,recovery_time_max_sessions=longest)

folders=sorted(p for p in (R/'bt').iterdir() if (p/'summary.json').exists())
base=pd.read_parquet(R/'bt/C00_V5/nav.parquet');dates=pd.to_datetime(base.trade_date)
years=(dates.iloc[-1]-dates.iloc[0]).days/365.25
rows=[];curves={};yr=[];attribution=[];sens=[];bridges=[];checks=[]
for folder in folders:
 code=folder.name;s=json.loads((folder/'summary.json').read_text());n=pd.read_parquet(folder/'nav.parquet')
 if not pd.to_datetime(n.trade_date).equals(dates):raise ValueError('Calendar mismatch '+code)
 a=n.total_asset.to_numpy(float)
 if not np.isfinite(a).all() or (a<=0).any():raise ValueError('Invalid NAV '+code)
 ret=np.r_[a[0]/INITIAL-1,a[1:]/a[:-1]-1]
 curves[code]=ret
 z={'experiment_id':code,'scope':'LEGACY_V5_COMPARISON','CAGR':float((a[-1]/INITIAL)**(1/years)-1),'Sharpe':float(sharpe(ret[1:])),'TerminalNAV':float(a[-1]),'CompletedTrades':s['completed_trades'],**risk(a)}
 z['status']='EXECUTED';z['data_time_chain']='PARTIALLY_VERIFIED';rows.append(z)
 checks.append({'code':code,'cash_plus_market_equals_nav_max_error':float(np.abs(n.cash+n.market_value-n.total_asset).max())})
 rr=pd.Series(ret,index=dates)
 for y,b in rr.groupby(rr.index.year):yr.append({'experiment_id':code,'year':y,'return':float(np.prod(1+b)-1),'Sharpe':float(sharpe(b.to_numpy())),'days':len(b)})
 trades=pd.read_parquet(folder/'trades.parquet');ov=pd.read_parquet(folder/'overlayed.parquet')
 ov=ov.sort_values(['signal_date','score','instrument_id'],ascending=[True,False,True]);ov['selection_rank']=ov.groupby('signal_date').cumcount()+1
 identity=ov.loc[ov.selection_rank.le(2),['signal_date','instrument_id','selection_rank']]
 trades=trades.merge(identity,on=['signal_date','instrument_id'],how='left',validate='many_to_one')
 if trades.selection_rank.isna().any():raise ValueError('Executed trade outside original planned top two')
 trades['actual_net_RMB_PnL']=trades.sell_proceeds-trades.buy_cost+trades.cash_dividend
 fee_cols=['buy_commission','buy_transfer_fee','sell_commission','sell_transfer_fee','stamp_tax']
 trades['fees_RMB']=trades[fee_cols].sum(axis=1)
 for rank,b in trades.groupby('selection_rank'):
  attribution.append({'experiment_id':code,'selection_rank':int(rank),'trades':len(b),'net_RMB_PnL':float(b.actual_net_RMB_PnL.sum()),'bought_quantity':int(b.initial_quantity.sum()),'sold_quantity':int(b.exit_quantity.sum()),'fees_RMB':float(b.fees_RMB.sum())})
 # Realized PnL plus residual equals total wealth change; residual separately disclosed.
 realized=float(trades.actual_net_RMB_PnL.sum());wealth=float(a[-1]-INITIAL)
 bridges.append({'experiment_id':code,'realized_net_RMB_PnL':realized,'terminal_wealth_change':wealth,'remaining_unrealized_and_corporate_action_residual':wealth-realized,'status':'RESIDUAL_DISCLOSED_NOT_ASSUMED_ZERO'})
 rej=pd.read_parquet(folder/'rejections.parquet')
 trace=rej.loc[rej.instrument_id.notna()].merge(identity,on=['signal_date','instrument_id'],how='left')
 if trace.selection_rank.isna().any():raise ValueError('Refused order outside planned top two')
 sens.append({'experiment_id':code,'scenario':'S3','status':'ALREADY_ENFORCED','open_limit_up_rejections':int(rej.reason.eq('open_limit_up').sum()),'plan_identities_preserved':True,'replacement_count':0,'trace_evidence':'all trades and nonnull rejected instruments belong to original per-date top-two plan; canonical engine loops only scheduled picks'})
 identity.to_csv(folder/'planned_order_identities.csv.gz',index=False)
 trades.to_csv(folder/'actual_trade_attribution.csv.gz',index=False)

allr=pd.DataFrame(rows);b=allr.set_index('experiment_id').loc['C00_V5']
allr['delta_CAGR']=allr.CAGR-b.CAGR;allr['delta_Sharpe']=allr.Sharpe-b.Sharpe;allr['nav_improvement']=allr.TerminalNAV/b.TerminalNAV-1
allr['delta_MaxDD']=allr.MaxDD-b.MaxDD
allr.to_csv(R/'all_results.csv',index=False);allr.to_csv(R/'risk_metrics.csv',index=False)
pd.DataFrame(curves,index=dates).rename_axis('trade_date').to_csv(R/'daily_returns.csv.gz')
pd.DataFrame(yr).to_csv(R/'yearly_metrics.csv',index=False)
pd.DataFrame(attribution).to_csv(R/'top1_top2_actual_pnl.csv',index=False)
pd.DataFrame(bridges).to_csv(R/'capital_compounding_bridge.csv',index=False)
(R/'ledger_checks.json').write_text(json.dumps(checks,indent=2))
# Paired execution stresses measure the candidate-versus-baseline delta in every scenario.
primary=[n for n in curves if not n.endswith(('_S1','_S2')) and n!='C00_V5']
for code in primary:
 deltas=[]
 for suffix,scenario in [('', 'S0'),('_S1','S1'),('_S2','S2')]:
  cand=allr.set_index('experiment_id').loc[code+suffix];base_=allr.set_index('experiment_id').loc['C00_V5'+suffix]
  delta=float(cand.TerminalNAV/base_.TerminalNAV-1);deltas.append(delta)
  sens.append({'experiment_id':code,'scenario':scenario,'candidate_CAGR':cand.CAGR,'baseline_CAGR':base_.CAGR,'candidate_Sharpe':cand.Sharpe,'baseline_Sharpe':base_.Sharpe,'nav_improvement':delta,'delta_CAGR':cand.CAGR-base_.CAGR,'delta_Sharpe':cand.Sharpe-base_.Sharpe})
 for x in sens:
  if x['experiment_id']==code:x['fee_direction_ok']=bool(np.sign(deltas[0])==np.sign(deltas[1])==np.sign(deltas[2]))
pd.DataFrame(sens).to_csv(R/'execution_sensitivity.csv',index=False)
# Deduplicate identical daily paths before all-candidate SPA.
unique=[];seen=set();aliases={}
for code in primary:
 h=hashlib.sha256(curves[code].tobytes()).hexdigest()
 if np.allclose(curves[code],curves['C00_V5'],rtol=0,atol=1e-12):aliases[code]='NUMERICALLY_IDENTICAL_TO_BASELINE_ATOL_1e-12';continue
 equivalent=next((other for other in unique if np.allclose(curves[code],curves[other],rtol=0,atol=1e-12)),None)
 if equivalent:aliases[code]='NUMERICALLY_IDENTICAL_TO_'+equivalent;continue
 unique.append(code);seen.add(h)
boot=[];base_ret=curves['C00_V5'][1:];rng=np.random.default_rng(0)
# The same sampled index is used for every candidate and baseline. Calendar-year CAGR convention preserved.
for block,nsamples in [(20,5000),(10,1000),(40,1000)]:
 arrays={code:[] for code in unique}
 for start in range(0,nsamples,100):
  k=min(100,nsamples-start);starts=rng.integers(0,len(base_ret),size=(k,int(np.ceil(len(base_ret)/block))))
  idx=((starts[:,:,None]+np.arange(block))%len(base_ret)).reshape(k,-1)[:,:len(base_ret)]
  br=base_ret[idx];bs=sharpe(br,axis=1);bc=np.exp(np.log1p(br).sum(1)/years)-1
  for code in unique:
   cr=curves[code][1:][idx];ds=sharpe(cr,axis=1)-bs;dc=np.exp(np.log1p(cr).sum(1)/years)-1-bc
   arrays[code].append(np.column_stack([ds,dc]))
 for code,parts in arrays.items():
  a=np.concatenate(parts)
  boot.append({'experiment_id':code,'block':block,'n':nsamples,'delta_Sharpe_q05':float(np.quantile(a[:,0],.05)),'delta_Sharpe_q50':float(np.quantile(a[:,0],.5)),'delta_Sharpe_q95':float(np.quantile(a[:,0],.95)),'delta_CAGR_q05':float(np.quantile(a[:,1],.05)),'delta_CAGR_q95':float(np.quantile(a[:,1],.95)),'paired':True,'cagr_years':years})
pd.DataFrame(boot).to_csv(R/'bootstrap_results.csv',index=False)
spa=SPA(-base_ret,np.column_stack([-curves[c][1:] for c in unique]),block_size=20,reps=5000,bootstrap='circular',seed=0)
spa.compute();(R/'spa_results.json').write_text(json.dumps({'status':'COMPUTED','pvalues':spa.pvalues.to_dict(),'candidate_family':unique,'deduplicated':aliases,'scope':'same legacy-v5 economic scope; all executed new aggregation candidates and their controls; no paper-reference metrics','repeated_historical_development':True},indent=2))
print(allr[['experiment_id','CAGR','Sharpe','MaxDD','CompletedTrades']].to_string(index=False))
