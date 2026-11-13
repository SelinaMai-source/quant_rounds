"""Actual P1 portfolio ledger attribution and preregistered paired inference."""
from pathlib import Path
import json,shutil,hashlib
import numpy as np,pandas as pd
from arch.bootstrap import SPA
from core.research_integrity_v5.evaluation import evaluate,sharpe
from core.research_integrity_v5.cache import identity
C=Path('/data1/stock-mxy/课题/一进二策略');R=C/'result/round23_7/D2'
def run():
 result=evaluate(R,baseline='D2_N0');result.to_csv(R/'all_results.csv',index=False);reg=pd.read_csv(R/'run_manifest.csv');friction=[];bridges=[];breadth=[]
 for z in reg.loc[reg.method.isin(['P_EQUAL','P_FIXED'])].itertuples(index=False):
  p=Path(z.path);targets=pd.read_parquet(p/'target_projection.parquet');orders=pd.read_parquet(p/'net_orders.parquet');nav=pd.read_parquet(p/'nav.parquet');pos=pd.read_parquet(p/'positions.parquet');tr=pd.read_parquet(p/'trades.parquet');c=pd.read_parquet(p/'costs.parquet');timeline=[]
  # Reconstruct phase cash from actual executions, separating opening and closing flows.
  buy=tr.groupby('entry_date').buy_cost.sum();sold_open=tr.loc[tr.exit_phase.eq('open')].groupby('actual_exit_date').sell_proceeds.sum();div=tr.groupby('actual_exit_date').cash_dividend.sum()
  previous_cash=300000.
  for row in nav.itertuples(index=False):
   day=row.trade_date;open_cash=previous_cash+sold_open.get(day,0)-buy.get(day,0)
   for phase,cash in [('open',open_cash),('close',row.cash)]:
    g=targets.loc[targets.trade_date.eq(day)&targets.phase.eq(phase)];raw=float(g.raw_target_weight.sum());mv=float((g.actual_quantity_after*g.reference_price).sum());value=cash+mv
    timeline.append({'trade_date':day,'phase':phase,'model_target_asset_capital_weight':raw,'model_target_cash_weight':1-raw,'projected_asset_weight_before_execution':float((g.projected_quantity*g.reference_price).sum()/g.pre_NAV.iloc[0]) if len(g) else 0,'actual_asset_value_RMB':mv,'actual_cash_RMB':cash,'actual_event_equity_RMB':value,'actual_cash_weight':cash/value if value else 1,'tracking_error_abs_RMB':float(g.tracking_error_RMB.abs().sum()),'cashflow_note':'original prepared market has zero corporate-action events; no hidden dividend residual'})
   previous_cash=row.cash
  pd.DataFrame(timeline).to_csv(R/(z.code+'_event_asset_cash_targets.csv.gz'),index=False)
  group=targets.groupby(['trade_date','phase']);friction.append({'experiment_id':z.code,'mean_abs_tracking_error_RMB_per_event':float(group.tracking_error_RMB.apply(lambda x:x.abs().sum()).mean()),'max_abs_tracking_error_RMB_per_event':float(group.tracking_error_RMB.apply(lambda x:x.abs().sum()).max()),'T1_locks_encountered':int(targets.T1_locked_quantity.gt(0).sum()),'unfilled_buy_intents':int(orders.side.eq('buy').sum()-c.side.eq('buy').sum()),'fees_RMB':float(c[['commission','transfer_fee','stamp_tax']].sum().sum()),'original_shadow_fees_not_added_to_real_NAV':True})
  b=pos.groupby('trade_date').instrument_id.nunique().reindex(nav.trade_date,fill_value=0);breadth.append({'experiment_id':z.code,'max_names':int(b.max()),'mean_names_all_calendar':float(b.mean()),'mean_names_when_invested':float(b[b>0].mean()),'min_order_initial_notional_RMB':float((tr.buy_cost-tr.buy_commission-tr.buy_transfer_fee).min()),'no_new_Top2_truncation':True})
 pd.DataFrame(friction).to_csv(R/'round_lot_cash_tracking_and_fees.csv',index=False);pd.DataFrame(breadth).to_csv(R/'breadth_capacity_small_capital.csv',index=False)
 # Correct standard two-sided turnover: actual daily traded notional normalized by NAV.
 for i,row in result.iterrows():
  p=Path(reg.loc[reg.code.eq(row.experiment_id),'path'].iloc[0]);nav=pd.read_parquet(p/'nav.parquet');tr=pd.read_parquet(p/'trades.parquet');buy=(tr.buy_cost-tr.buy_commission-tr.buy_transfer_fee).groupby(tr.entry_date).sum();sell=(tr.sell_proceeds+tr.sell_commission+tr.sell_transfer_fee+tr.stamp_tax).groupby(tr.actual_exit_date).sum();gross=buy.add(sell,fill_value=0).reindex(nav.trade_date,fill_value=0).to_numpy();result.loc[i,'turnover_annualized_two_sided_daily_normalized']=float(np.mean(gross/nav.total_asset.to_numpy())*252);result.loc[i,'gross_traded_notional_RMB']=float(gross.sum())
 result.to_csv(R/'all_results.csv',index=False)
 # P-FIXED versus P-EQUAL isolates annual weight choice within the same adapter.
 d=pd.read_csv(R/'daily_returns.csv.gz',parse_dates=['trade_date']);years=(d.trade_date.iloc[-1]-pd.Timestamp('2018-01-02')).days/365.25;a=d.P_FIXED.to_numpy();b=d.P_EQUAL.to_numpy();rng=np.random.default_rng(17);out=[]
 for block,count in [(20,5000),(10,1000),(40,1000)]:
  parts=[]
  for start in range(0,count,100):
   k=min(100,count-start);beg=rng.integers(0,len(a),size=(k,int(np.ceil(len(a)/block))));idx=((beg[:,:,None]+np.arange(block))%len(a)).reshape(k,-1)[:,:len(a)];aa=a[idx];bb=b[idx];parts.append(np.column_stack([sharpe(aa,axis=1)-sharpe(bb,axis=1),np.expm1(np.log1p(aa).sum(1)/years)-np.expm1(np.log1p(bb).sum(1)/years)]))
  z=np.concatenate(parts);out.append({'comparison':'P_FIXED minus P_EQUAL','block':block,'draws':count,'delta_Sharpe_q05':float(np.quantile(z[:,0],.05)),'delta_Sharpe_q95':float(np.quantile(z[:,0],.95)),'delta_CAGR_q05':float(np.quantile(z[:,1],.05)),'delta_CAGR_q95':float(np.quantile(z[:,1],.95)),'paired':True,'calendar_years':years})
 pd.DataFrame(out).to_csv(R/'fixed_vs_equal_paired_bootstrap.csv',index=False)
 old=pd.read_csv(C/'result/round23_7/D1/daily_returns.csv.gz',parse_dates=['trade_date']);assert old.trade_date.equals(d.trade_date);np.testing.assert_allclose(old.N0,d.D2_N0,atol=1e-12,rtol=0);family={n:old[n].to_numpy() for n in ['N1','N2']};family.update({n:d[n].to_numpy() for n in ['P_SCORE','P_EQUAL','P_FIXED']});base=d.D2_N0.to_numpy();unique=[];aliases={}
 for name,r in family.items():
  match=next((n for n in unique if np.allclose(r,family[n],atol=1e-12,rtol=0)),None)
  if np.allclose(r,base,atol=1e-12,rtol=0):aliases[name]='BASELINE'
  elif match:aliases[name]=match
  else:unique.append(name)
 spa=SPA(-base,np.column_stack([-family[n] for n in unique]),block_size=20,reps=5000,bootstrap='circular',seed=17);spa.compute();(R/'ROUND23_7_ALL_CURRENT_TRIALS_SPA.json').write_text(json.dumps({'scope':'all frozen current Round23.7 S0 D1+D2 economic alternatives, same calendar/cost/300000 original baseline; diagnostic errors/vertex tests excluded from alpha family','loss':'negative actual daily portfolio net returns','candidate_family':unique,'deduplicated':aliases,'pvalues':spa.pvalues.to_dict(),'repeated_development_history':True},indent=2))
 # Full frozen expert input scores, actual calendar decision clocks and identities.
 from core.round23_5.builders import rule_frame
 f=rule_frame(pd.read_parquet(C/'result/round23_6/frozen_components.parquet'));calendar=pd.read_parquet(C/'result/round22_3/data/ten_candidate.parquet',columns=['trade_date','instrument_id','next1_date']).rename(columns={'trade_date':'signal_date'});calendar['prediction_at']=pd.to_datetime(calendar.next1_date).dt.tz_localize('Asia/Shanghai')+pd.Timedelta(hours=9,minutes=30);pred=[]
 for name in ['v5','five','ten','lgbm']:
  x=f[['signal_date','instrument_id',name]].rename(columns={name:'score'}).merge(calendar[['signal_date','instrument_id','prediction_at']],on=['signal_date','instrument_id'],validate='one_to_one');p=R/(name+'_frozen_expert_predictions.parquet');x.to_parquet(p,index=False);pred.append(identity(p))
 (R/'prediction_manifest.json').write_text(json.dumps(pred,ensure_ascii=False,indent=2));(R/'input_manifest.json').write_text(json.dumps([identity(C/'result/round23_6/frozen_components.parquet'),identity(C/'result/round22_3/data/ten_candidate.parquet'),identity(C/'result/round23_5/selected_parameters_by_fold.csv')],ensure_ascii=False,indent=2));(R/'model_manifest.json').write_text(json.dumps({'predictive_model_training':'none in D2; frozen source expert predictions','annual_utility_weight_models':8,'policy_contract':identity(R/'CONTRACT.yaml'),'annual_weights':identity(R/'weights_validation.json'),'original_native_model_and_feature_contracts':'parent baseline_v5 audit and D0 read audit; source experts independent of D1','optimizer':'.99 alpha, gamma3, original frozen utility, no search'},ensure_ascii=False,indent=2))
 s=json.loads((R/'STATUS.json').read_text());s['EVIDENCE_COMPLETE']=True;s['current_four_vertex_full_gate_pass']=True;s['policy_execution_separation_financial_parity']=True;s['real_preview_mutation_portfolio_runs']=2;s['total_adapter_economic_runs_including_separation_replays']=12;s['unique_adapter_economic_configurations']=6;s['native_shadow_runs_including_initial_diagnostic']=16;s['vertex_runs_including_diagnostic_and_separation_replay']=12;s['decision']='REJECT_CURRENT_CONFIGURATION';s['remaining']=[];(R/'STATUS.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
 for p in C.glob('result/round23_7_D2*.log'):shutil.copy2(p,R/p.name)
 print(result.loc[result.scenario.eq('S0'),['experiment_id','CAGR','Sharpe','MaxDD','trades']].to_string(index=False),flush=True)
