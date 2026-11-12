"""Ports of locked SuperLearner method.NNLS and opera BOA; R parity required.

Source algorithms remain external; see source_lock.json and numerical reference tests.
"""
import numpy as np
from scipy.optimize import nnls


def superlearner_nnls(z,y,observation_weights=None):
    z=np.asarray(z,float);y=np.asarray(y,float)
    ow=np.ones(len(y)) if observation_weights is None else np.asarray(observation_weights,float)
    if not np.isfinite(z).all() or not np.isfinite(y).all() or (ow<0).any():
        raise ValueError('NNLS requires finite inputs and nonnegative observation weights')
    raw,_=nnls(np.sqrt(ow)[:,None]*z,np.sqrt(ow)*y)
    weights=raw/raw.sum() if raw.sum()>0 else raw.copy()
    return {'raw':raw,'weights':weights,'cvRisk':np.mean(ow[:,None]*(z-y[:,None])**2,axis=0)}


class BernsteinOnlineAggregation:
    """opera/R/BOA.R: square gradient, uniform prior, 2.2 second-order term.

    The delayed adapter passes regrets from STORED prediction-time predictions;
    the reference mode uses a single immediate feedback item per update.
    """
    def __init__(self,n_experts):
        self.w0=np.ones(n_experts);self.eta_inv2=np.zeros(n_experts)
        self.R=np.zeros(n_experts);self.Rreg=np.zeros(n_experts)
    def weights(self):
        w=self.w0.copy();idx=self.eta_inv2>0
        if idx.any():
            aux=-np.log(self.eta_inv2[idx])/2+np.log(self.w0[idx])+self.Rreg[idx]/np.sqrt(self.eta_inv2[idx])
            raw=np.exp(aux-aux.max());w[idx]=self.w0[idx].sum()*raw/raw.sum()
        return w/w.sum()
    def predict(self,experts):
        return np.asarray(experts,float)@self.weights()
    def update_regret(self,regret):
        r=np.asarray(regret,float)
        if not np.isfinite(r).all():raise ValueError('nonfinite feedback')
        self.eta_inv2+=2.2*r*r
        idx=self.eta_inv2>0
        rr=np.zeros_like(r);rr[idx]=r[idx]-r[idx]**2/np.sqrt(self.eta_inv2[idx])
        self.R+=r;self.Rreg+=rr
    def update(self,experts,y,prediction):
        # Author loss gradient: 2*(prediction-y)*expert. Regret uses l(pred)-l(expert).
        self.update_regret(2*(float(prediction)-float(y))*(float(prediction)-np.asarray(experts,float)))


def immediate_stream(experts,y):
    b=BernsteinOnlineAggregation(np.shape(experts)[1]);weights=[];pred=[]
    for x,t in zip(experts,y):
        weights.append(b.weights());p=b.predict(x);pred.append(p);b.update(x,t,p)
    return np.array(pred),np.array(weights),b.weights()
