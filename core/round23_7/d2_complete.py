"""D2 immutable native shadow captures, extended four-vertex gate and fixed arms."""
from pathlib import Path
import json,hashlib,shutil,time
import numpy as np,pandas as pd
from core.research_integrity_v5.cache import identity,sha,transaction,fingerprint
from core.research_integrity_v5.experiment import full_contract,backtest
from core.round23_7.shadow import generate
from core.round23_7.portfolio import simulate
C=Path('/data1/stock-mxy/课题/一进二策略');R=C/'result/round23_7/D2';NAMES=['v5','five','ten','lgbm']
def contract(kind,parameters,upstream=()):
 from core.r225241_sleeve import market
 m=Path(market()['snapshot_path']);inputs=list((m/'daily_raw').rglob('*.parquet'))+[m/'manifest.json',R/'CONTRACT.yaml',C/'result/round23_6/frozen_components.parquet']+[C/f'result/round22_3/data/{h}_{k}.parquet' for h in ['five','ten'] for k in ['training','candidate']]
 return full_contract({'deliverable':'D2','kind':kind,**parameters},inputs,upstream)
def shadow_run(name,scores,bps):
 params={'name':name,'extra_buy_bps':bps,'score_hash':hashlib.sha256(pd.util.hash_pandas_object(scores,index=False).values.tobytes()).hexdigest()};c=contract('causal_original_shadow',params);cache=R/'immutable_shadow'/f'{name}_S{bps//10}';fp=fingerprint(c)
 with transaction(cache,c) as (p,cached):
  if not cached:generate(name,scores,destination=p,extra_buy_bps=bps)
 dest=cache/fp;return json.loads((dest/'causal_policy_events.json').read_text()),dest,fp
def vertex(shadows,refs,market):
 rows=[]
 for j,name in enumerate(NAMES):
  weights={y:[float(i==j) for i in range(4)] for y in range(2018,2027)};c=contract('vertex_projection',{'expert':name,'weights':weights},[sha(refs[n]/'causal_policy_events.json') for n in NAMES]);cache=R/'immutable_vertices'/name;fp=fingerprint(c)
  with transaction(cache,c) as (p,cached):
   if not cached:
    result=simulate(shadows,market,weights)
    for k,v in result.items():v.to_parquet(p/(k+'.parquet'),index=False)
  p=cache/fp;checks={}
  specs=[('nav',['cash','market_value','total_asset'],['trade_date']),('trades',['initial_quantity','exit_quantity','buy_cost','sell_proceeds','cash_dividend','entry_reference_price','entry_price','exit_price','buy_commission','buy_transfer_fee','sell_commission','sell_transfer_fee','stamp_tax','entry_portfolio_nav','target_name_budget','allocation_multiplier','score'],['signal_date','instrument_id','entry_date','actual_exit_date','sleeve_id']),('positions',['quantity','last_close','market_value'],['trade_date','cohort_id','instrument_id']),('costs',['commission','transfer_fee','stamp_tax','slippage_cost'],['trade_date','cohort_id','instrument_id','side'])]
  for table,cols,keys in specs:
   a=pd.read_parquet(p/(table+'.parquet'));b=pd.read_parquet(refs[name]/(table+'.parquet'));pd.testing.assert_frame_equal(a[keys+cols].sort_values(keys).reset_index(drop=True),b[keys+cols].sort_values(keys).reset_index(drop=True),check_exact=False,atol=1e-6,rtol=1e-10);checks[table]={'PASS':True,'rows':len(a)}
  # Before-order identities, intended capital, slot, multiplier and final lot quantity.
  original=[]
  for e in shadows[name]:
   for i in e['intents']:original.append({'trade_date':pd.Timestamp(e['day']),'signal_date':pd.Timestamp(i['signal_date']),'instrument_id':i['instrument_id'],'sleeve_id':i['sleeve_id'],'selection_rank':i['rank'],'planned_capital':i['target_name_budget'],'allocation_multiplier':i['allocation_multiplier'],'quantity':int(np.floor(i['desired_fractional_quantity']/100+1e-9))*100})
  a=pd.read_parquet(p/'net_orders.parquet');a=a.loc[a.side.eq('buy')];b=pd.DataFrame(original);keys=['trade_date','signal_date','instrument_id'];cols=keys+['sleeve_id','selection_rank','planned_capital','allocation_multiplier','quantity'];pd.testing.assert_frame_equal(a[cols].sort_values(keys).reset_index(drop=True),b[cols].sort_values(keys).reset_index(drop=True),check_exact=False,atol=1e-6,rtol=1e-10,check_dtype=False);checks['all_planned_buy_intents']={'PASS':True,'rows':len(a)}
  a=pd.read_parquet(p/'rejections.parquet');b=pd.read_parquet(refs[name]/'rejections.parquet');cols=['event','reason','signal_date','execution_date','instrument_id','sleeve_id','cohort_id']
  for f in [a,b]:
   for col in ['signal_date','execution_date']:f[col]=pd.to_datetime(f[col])
  keys=['execution_date','signal_date','instrument_id','event','reason','cohort_id'];pd.testing.assert_frame_equal(a[cols].sort_values(keys).reset_index(drop=True),b[cols].sort_values(keys).reset_index(drop=True),check_dtype=False);checks['rejected_order_lot_lineage']={'PASS':True,'rows':len(a)}
  rows.append({'expert':name,'PASS':True,'checks':checks,'path':str(p),'fingerprint':fp});print('D2 EXTENDED VERTEX',name,'PASS',flush=True)
 (R/'extended_four_vertex_parity.json').write_text(json.dumps(rows,indent=2));return rows

def run():
 from core.r225241_sleeve import market
 from core.round23_5.builders import rule_frame,monthly_rules,weights_by_year
 from core.round23_5.nested_validation import outer_years
 rules=rule_frame(pd.read_parquet(C/'result/round23_6/frozen_components.parquet'));m=market()['prepared'];shadows={};refs={};fps=[]
 for n in NAMES:shadows[n],refs[n],fp=shadow_run(n,rules[['signal_date','instrument_id',n]].rename(columns={n:'score'}),0);fps.append(fp)
 vertex(shadows,refs,m)
 # No economic combination runs until every extended vertex has passed.
 monthly=monthly_rules({n:refs[n]/'nav.parquet' for n in NAMES});weights,rows=weights_by_year(monthly,.99);pd.DataFrame(rows).to_csv(R/'recomputed_P1_A099_annual_weights.csv',index=False);monthly.to_csv(R/'prior_expert_monthly_gross.csv')
 old=pd.read_csv(C/'result/round23_5/selected_parameters_by_fold.csv');old=old.loc[old.experiment_id.eq('P1_BASE_A099')];new=pd.DataFrame(rows)
 for year in range(2019,2027):
  a=new.loc[new.year.eq(year),NAMES].to_numpy(float);b=old.loc[old.year.eq(year),NAMES].to_numpy(float);np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-8)
 fixed={y:([float(weights[y][n]) for n in NAMES] if np.isfinite(list(weights[y].values())[0]) else [1.,0.,0.,0.]) for y in range(2018,2027)};equal={y:[.25]*4 for y in range(2018,2027)}
 (R/'weights_validation.json').write_text(json.dumps({'PASS':True,'max_tolerance':1e-8,'source':'recomputed original alpha.99 utility from only pre-year original expert NAV','annual_weights':fixed},indent=2))
 registry=[];summaries=[]
 for bps in [0,10,20]:
  if bps:
   sh={};rf={};sf=[]
   for n in NAMES:sh[n],rf[n],fp=shadow_run(n,rules[['signal_date','instrument_id',n]].rename(columns={n:'score'}),bps);sf.append(fp)
  else:sh,rf,sf=shadows,refs,fps
  suffix='' if bps==0 else '_S'+str(bps//10)
  # Matched original baseline and exact old P1 score replay remain native engine results.
  for method in ['D2_N0','P_SCORE']:
   x=rules[['signal_date','instrument_id','v5']].rename(columns={'v5':'score'}) if method=='D2_N0' else pd.read_parquet(C/'result/round23_6/C01_frozen_scores.parquet')[['signal_date','instrument_id','score']]
   summary,path,proof=backtest('round23_7',method+suffix,x,bps=bps,upstream=sf);registry.append({'method':method,'scenario':'S'+str(bps//10),'code':method+suffix,**proof});summaries.append(summary)
  for method,annual in [('P_EQUAL',equal),('P_FIXED',fixed)]:
   c=contract('single_real_account',{'method':method,'annual_weights':annual,'extra_buy_bps':bps},sf);cache=R/'immutable_portfolios'/(method+suffix);fp=fingerprint(c)
   with transaction(cache,c) as (p,cached):
    if not cached:
     result=simulate(sh,m,annual,extra_buy_bps=bps)
     for k,v in result.items():v.to_parquet(p/(k+'.parquet'),index=False)
   path=cache/fp;registry.append({'method':method,'scenario':'S'+str(bps//10),'code':method+suffix,'path':str(path),'experiment_fingerprint':fp,'cache_hit':cached});print('D2 ECONOMIC COMBINATION',method,suffix,'ran',flush=True)
 pd.DataFrame(registry).to_csv(R/'run_manifest.csv',index=False);(R/'STATUS.json').write_text(json.dumps({'IMPLEMENTED':True,'UNIT_TESTED':True,'REFERENCE_RUN_COMPLETE':'NOT_APPLICABLE_PROJECT_ADAPTER','TARGET_OOF_COMPLETE':'NOT_APPLICABLE_PORTFOLIO_LAYER','BACKTEST_COMPLETE':True,'EVIDENCE_COMPLETE':False,'four_vertices_all_pass':True,'current_native_shadow_runs':12,'current_vertex_runs':4,'economic_configuration_runs':12,'new_combinations_runs':6,'weights':'annual frozen .99, no alpha search','remaining':'portfolio attribution, paired statistics and reporting'},indent=2))
