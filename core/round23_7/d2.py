"""D2 four real expert executions, causal targets and mandatory one-hot gate."""
from pathlib import Path
import json,time
import numpy as np,pandas as pd
from core.round23_7.shadow import generate
from core.round23_7.portfolio import simulate
from core.research_integrity_v5.cache import identity
C=Path('/data1/stock-mxy/课题/一进二策略');R=C/'result/round23_7/D2'
def run_vertices():
 from core.round23_5.builders import rule_frame
 from core.r225241_sleeve import market
 r=rule_frame(pd.read_parquet(C/'result/round23_6/frozen_components.parquet'));names=['v5','five','ten','lgbm'];shadows={};native=market()['prepared'];start=time.time()
 for name in names:shadows[name]=generate(name,r[['signal_date','instrument_id',name]].rename(columns={name:'score'}))
 checks=[]
 for j,name in enumerate(names):
  weights={y:[float(i==j) for i in range(4)] for y in range(2018,2027)};got=simulate(shadows,native,weights);dest=R/'vertices'/name;dest.mkdir(parents=True,exist_ok=True)
  for key,value in got.items():value.to_parquet(dest/(key+'.parquet'),index=False)
  proof={'expert':name,'PASS':True,'checks':{},'adapter':'same target projection routine for all weights; no vertex-specialized execution bypass'}
  for table,cols,keys in [('nav',['cash','market_value','total_asset'],['trade_date']),('trades',['initial_quantity','exit_quantity','buy_cost','sell_proceeds','cash_dividend','entry_reference_price','entry_price','exit_price','buy_commission','buy_transfer_fee','sell_commission','sell_transfer_fee','stamp_tax','entry_portfolio_nav','target_name_budget','allocation_multiplier','score'],['signal_date','instrument_id','entry_date','actual_exit_date','sleeve_id']),('positions',['quantity','last_close','market_value'],['trade_date','cohort_id','instrument_id']),('costs',['commission','transfer_fee','stamp_tax','slippage_cost'],['trade_date','cohort_id','instrument_id','side'])]:
   ref=pd.read_parquet(R/f'shadow/{name}/{table}.parquet');a=got[table]
   try:
    aa=a[keys+cols].sort_values(keys).reset_index(drop=True);bb=ref[keys+cols].sort_values(keys).reset_index(drop=True);pd.testing.assert_frame_equal(aa,bb,check_exact=False,atol=1e-6,rtol=1e-10);proof['checks'][table]={'PASS':True,'rows':len(a)}
   except (AssertionError,KeyError) as e:
    proof['PASS']=False;proof['checks'][table]={'PASS':False,'detail':str(e)[:4000],'actual_rows':len(a),'expected_rows':len(ref)}
  (dest/'parity.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2));checks.append(proof);print('D2 VERTEX',name,json.dumps(proof,ensure_ascii=False),flush=True)
 (R/'four_vertex_parity.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2));status={'IMPLEMENTED':True,'UNIT_TESTED':False,'REFERENCE_RUN_COMPLETE':'NOT_APPLICABLE_NEW_EXECUTION_ADAPTER','TARGET_OOF_COMPLETE':'NOT_APPLICABLE_PORTFOLIO_LAYER','BACKTEST_COMPLETE':False,'EVIDENCE_COMPLETE':False,'shadow_original_backtests':4,'vertex_adapter_backtests':4,'four_vertices_all_pass':all(c['PASS'] for c in checks),'economic_combinations_run':0,'wall_seconds':time.time()-start};(R/'STATUS.json').write_text(json.dumps(status,indent=2));return shadows,checks
if __name__=='__main__':run_vertices()
