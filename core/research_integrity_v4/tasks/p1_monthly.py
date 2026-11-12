from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
C=Path('/data1/stock-mxy/课题/一进二策略');sys.path.insert(0,str(C));sys.dont_write_bytecode=True
from core.round23_5.builders import monthly_rules,RULES
from core.round23_5.paper_adapters.nardari import combination_weights
R=C/'result/round23_6';old=C/'result/round23_5/bt'
monthly=monthly_rules({k:old/v/'nav.parquet' for k,v in [('v5','B0_V5'),('five','RULE_FIVE'),('ten','RULE_TEN'),('lgbm','RULE_LGBM')]})
rows=[]
for month in pd.date_range('2018-01-01','2026-07-01',freq='MS'):
 hist=monthly.loc[monthly.index<month,RULES]
 for method,alpha in [('C02_PORT_EQUAL',None),('C04_PORT_STATPOOL_MONTHLY',1.),('C05_PORT_FLEXPOOL_MONTHLY',1.),('C06_PORT_A099_MONTHLY',.99)]:
  state='WEIGHTS_ONLY_NOT_EXECUTED'
  if len(hist)<12:w=np.array([1.,0,0,0]);state='FALLBACK_V5'
  elif method=='C02_PORT_EQUAL':w=np.ones(4)/4
  else:
   if method=='C05_PORT_FLEXPOOL_MONTHLY':
    if len(hist)<120:state='WARMUP_ALPHA_ONE_NO_FLEXPOOL_ACTIVE'
    else:
     # Fixed PDF equation (9): current w*_{t-1}(alpha) is evaluated
     # against R_s over s=tau+tau0 .. t-1; tau0=60 and 60 initial tuning months.
     grid=[round(.90+i*.01,2) for i in range(11)];scores=[];solutions=[]
     for candidate in grid:
      wc,ok=combination_weights(hist.to_numpy(),candidate,3);solutions.append(wc)
      gross=np.maximum(hist.iloc[60:].to_numpy()@wc,1e-8)
      scores.append(np.sum((gross**-2-1)/-2))
     i=int(np.argmax(scores));alpha=grid[i];w=solutions[i]
   if method!='C05_PORT_FLEXPOOL_MONTHLY' or len(hist)<120:w,_=combination_weights(hist.to_numpy(),alpha,3)
  rows.append({'method':method,'decision_month':str(month.date()),'max_expert_return_month':str(hist.index.max()) if len(hist) else None,'n_months':len(hist),'alpha':alpha,'state':state,**dict(zip(RULES,w))})
pd.DataFrame(rows).to_csv(R/'monthly_portfolio_weight_research.csv',index=False)
pdf=C/'research/round23_6_26_execution/vendor/P1.pdf'
(R/'equation_to_code.md').write_text('''# Fixed-PDF P1 mapping

PDF SHA256: '''+hashlib.sha256(pdf.read_bytes()).hexdigest()+'''

Equation (5): nonnegative expert weights sum to one.
Equation (6): asset target q = Omega @ w; score convex fusion and subsequent Top2 selection are not this operation.
Equation (7): maximize discounted power utility, gamma=3, discount alpha^(t-1-s). The historical nardari.py solver is a project implementation, not author code.
Equation (9): the PDF displays w*_(t-1)(alpha) inside the sum over s=tau*..t-1, with tau*=tau+tau0. The current fitted vector is used exactly as displayed; it is not replaced by a made-up held-out-return alpha selector. The empirical design reserves 60 expert OOS months for weights and another 60 for initial alpha tuning. Our available frozen expert ledgers start in 2018, so this mathematical weight study never declares adaptive FLEXPOOL active through July 2026. Recovering exact pre-2018 expert strategy ledgers remains incomplete.

Monthly optimization uses only completed earlier months. All monthly outputs are weights only: no shared-cash, netted-order, T+1-constrained account has been implemented. No economic result is attributed to C02-C06. C03 would reuse the exact C01 annual weights, but its asset-space execution is also unfinished. The component-NAV cost model is not claimed equal to the required netted external-order cost model.
''')
print('Actual monthly weight optimizations',len(rows),'tradable portfolio backtests',0)
