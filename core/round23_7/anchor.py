"""Two-dimensional bounded convex RankNet-style project hypothesis, no grid."""
from itertools import combinations
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
KEY=['signal_date','instrument_id']

def scores(frame,beta):
 b=np.asarray(beta)
 if np.array_equal(b,[0.,0.]):return frame.v5.copy()
 return .1*((1-b[0])*frame.xF+b[0]*frame.lF)+.9*((1-b[1])*frame.xT+b[1]*frame.lT)
def build_pairs(frame):
 rows=[]
 for day,g in frame.groupby('signal_date',sort=True):
  g=g.sort_values(['v5','instrument_id'],ascending=[False,True],kind='mergesort').head(5)
  records=list(g.itertuples(index=False))
  for i,j in combinations(range(len(records)),2):
   if i>=2:continue
   a,b=records[i],records[j]
   if not np.isfinite([a.y,b.y,a.lF,a.lT,b.lF,b.lT]).all() or pd.isna(a.label_available_at) or pd.isna(b.label_available_at):continue
   dy=a.y-b.y
   if dy==0:continue
   rows.append({'signal_date':day,'i':a.instrument_id,'j':b.instrument_id,'q':np.sign(dy),'gap':abs(dy),'base':a.v5-b.v5,'dF':.1*((a.lF-a.xF)-(b.lF-b.xF)),'dT':.9*((a.lT-a.xT)-(b.lT-b.xT)),'pair_available_at':max(a.label_available_at,b.label_available_at),'i_action':a.action_status,'j_action':b.action_status})
 return pd.DataFrame(rows)
def objective(beta,pairs,weighted):
 if pairs.empty:return .1*float(np.dot(beta,beta)),.2*np.asarray(beta)
 cap=float(pairs.gap.quantile(.95));a=np.minimum(pairs.gap.to_numpy(),cap) if weighted else np.ones(len(pairs));dates=pairs.signal_date.to_numpy();weights=pd.Series(a,index=pairs.index);denom=weights.groupby(pairs.signal_date).transform('sum').to_numpy();w=a/denom/pairs.signal_date.nunique()
 d=pairs[['dF','dT']].to_numpy();q=pairs.q.to_numpy();margin=q*(pairs.base.to_numpy()+d@beta)/.1
 loss=float(np.dot(w,np.logaddexp(0,-margin))+.1*np.dot(beta,beta));grad=d.T@(w*(-q/.1)*expit(-margin))+.2*np.asarray(beta)
 return loss,grad

def fit(pairs,weighted):
 n=pairs.signal_date.nunique() if not pairs.empty else 0
 if n<200 or len(pairs)<500:return np.zeros(2),{'status':'FALLBACK_INSUFFICIENT_MATURE_PAIRS','dates':n,'pairs':len(pairs)}
 result=minimize(lambda b:objective(b,pairs,weighted),np.zeros(2),method='L-BFGS-B',jac=True,bounds=[(0,.25),(0,.25)],options={'ftol':1e-14,'gtol':1e-10,'maxiter':1000})
 result.x=np.round(result.x,8)
 grad=objective(result.x,pairs,weighted)[1];projected=grad.copy();projected[(result.x<=1e-8)&(grad>0)]=0;projected[(result.x>=.25-1e-8)&(grad<0)]=0;kkt=float(np.abs(projected).max());okay=bool(result.success and kkt<1e-7 and np.all((result.x>=0)&(result.x<=.25)))
 return result.x if okay else np.zeros(2),{'status':'CONVERGED' if okay else 'FALLBACK_NONCONVERGENCE','success':bool(result.success),'message':str(result.message),'iterations':int(result.nit),'objective':float(result.fun),'baseline_objective':objective(np.zeros(2),pairs,weighted)[0],'gradient':grad.tolist(),'projected_gradient_max':kkt,'cap95':float(pairs.gap.quantile(.95)),'dates':n,'pairs':len(pairs)}
