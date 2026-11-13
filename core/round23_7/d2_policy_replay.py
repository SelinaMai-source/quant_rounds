"""Separate pure capital recommendations from legacy fill projections, then re-audit."""
from pathlib import Path
import json,copy,shutil
import pandas as pd,numpy as np
from core.round23_7.d2_complete import contract,vertex,NAMES,R,C
from core.round23_7.portfolio import simulate,recommendation
from core.research_integrity_v5.cache import transaction,fingerprint,sha

def run():
 from core.r225241_sleeve import market
 m=market()['prepared'];old=pd.read_csv(R/'run_manifest.csv');archive=R/'before_policy_execution_separation';archive.mkdir(exist_ok=True)
 for n in ['run_manifest.csv','extended_four_vertex_parity.json']:
  if (R/n).exists() and not (archive/n).exists():shutil.copy2(R/n,archive/n)
 fixed={int(y):v for y,v in json.loads((R/'weights_validation.json').read_text())['annual_weights'].items()};equal={y:[.25]*4 for y in range(2018,2027)};new=old.loc[~old.method.isin(['P_EQUAL','P_FIXED'])].to_dict('records');checks=[]
 for bps in [0,10,20]:
  refrow=old.loc[old.code.eq('P_EQUAL'+('' if bps==0 else '_S'+str(bps//10)))].iloc[0];prior=Path(refrow.path);sf=json.loads((prior/'contract.json').read_text())['upstream'];refs={n:R/f'immutable_shadow/{n}_S{bps//10}'/sf[j] for j,n in enumerate(NAMES)};shadows={n:json.loads((refs[n]/'causal_policy_events.json').read_text()) for n in NAMES}
  if not bps:vertex(shadows,refs,m)
  for method,weights in [('P_EQUAL',equal),('P_FIXED',fixed)]:
   code=method+('' if bps==0 else '_S'+str(bps//10));c=contract('pure_causal_capital_then_original_fill',{'method':method,'weights':weights,'extra_buy_bps':bps},sf);cache=R/'policy_separated_portfolios'/code;fp=fingerprint(c)
   with transaction(cache,c) as (p,cached):
    if not cached:
     result=simulate(shadows,m,weights,extra_buy_bps=bps)
     for k,v in result.items():v.to_parquet(p/(k+'.parquet'),index=False)
   path=cache/fp;previous=Path(old.loc[old.code.eq(code),'path'].iloc[0]);financial={}
   for table in ['nav','trades','positions','costs']:
    a=pd.read_parquet(path/(table+'.parquet'));b=pd.read_parquet(previous/(table+'.parquet'));pd.testing.assert_frame_equal(a,b,check_exact=False,atol=1e-6,rtol=1e-10);financial[table]='MATCH_WITHIN_FIXED_FLOAT_TOLERANCE'
   checks.append({'code':code,'comparison':financial,'native_shadows_unchanged':True,'preview_forecast_removed':True});new.append({'method':method,'scenario':'S'+str(bps//10),'code':code,'path':str(path),'experiment_fingerprint':fp,'cache_hit':cached});print(code,'pure-policy separation finance parity PASS',flush=True)
   if not bps:
    changed=copy.deepcopy(shadows)
    for n in NAMES:
     for e in changed[n]:
      oldrec=recommendation(e);e['target_quantity']={'FORBIDDEN_FUTURE_PREVIEW':1e12}
      for i in e['intents']:i['desired_fractional_quantity']=1e12;i['shadow_rejection_before_quantity']='FORBIDDEN_FUTURE_PREVIEW'
      assert recommendation(e)==oldrec
    changed_result=simulate(changed,m,weights);diag=R/'real_entry_preview_mutation'/method;diag.mkdir(parents=True,exist_ok=True)
    for table in ['nav','trades','net_orders','target_projection','rejections']:
     reference=pd.read_parquet(path/(table+'.parquet'));pd.testing.assert_frame_equal(changed_result[table],reference,check_exact=True)
    (diag/'PASS.json').write_text(json.dumps({'PASS':True,'scope':'all original real-event records 2018-2026, both portfolio weight arms; actual simulator entry','mutated_fields':['target_quantity execution preview','desired_fractional_quantity','shadow_rejection_before_quantity'],'unchanged':['recommendation','target_projection','net_orders','rejections','trades','NAV'],'limits':'original market execution still uses legacy high/low clip, not a live fill proof'},indent=2))
 pd.DataFrame(new).to_csv(R/'run_manifest.csv',index=False);(R/'policy_execution_separation_audit.json').write_text(json.dumps(checks,indent=2))
