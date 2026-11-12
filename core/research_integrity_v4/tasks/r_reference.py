from pathlib import Path
import numpy as np,subprocess,json,sys
B=Path('/data1/stock-mxy/课题/一进二策略/research/round23_6_26_execution');R=B.parents[1]/'result/round23_6';D=R/'r_reference';D.mkdir(exist_ok=True)
sys.path.insert(0,str(B/'round23_6'))
from core.research_integrity_v4.aggregation import superlearner_nnls,immediate_stream
x=np.arange(40.)
z=np.column_stack([np.sin(x*.2)*.1,np.cos(x*.3)*.1,np.sin(x*.11)*.2,np.repeat(.01,40)])
y=z@np.array([.1,.6,.2,.1])+np.sin(x*.47)*.01
np.savetxt(D/'experts.csv',z,delimiter=',');np.savetxt(D/'target.csv',y,delimiter=',')
script='''args=commandArgs(TRUE)
v=args[1];d=args[2]
library(SuperLearner);library(nnls)
source(file.path(v,"SuperLearner/R/method.R"))
Z=as.matrix(read.csv(file.path(d,"experts.csv"),header=FALSE));Y=scan(file.path(d,"target.csv"),quiet=TRUE)
a=method.NNLS()$computeCoef(Z=Z,Y=Y,libraryNames=colnames(Z),verbose=FALSE,obsWeights=rep(1,length(Y)))
write.table(cbind(coef(a$optimizer),a$coef),file.path(d,"sl_coefficients.csv"),sep=",",row.names=FALSE,col.names=FALSE)
for (f in list.files(file.path(v,"opera/R"),pattern="[.]R$",full.names=TRUE)) source(f)
b=mixture(Y=Y,experts=Z,model="BOA",loss.type="square",quiet=TRUE)
write.table(b$weights,file.path(d,"boa_weights.csv"),sep=",",row.names=FALSE,col.names=FALSE)
write.table(b$prediction,file.path(d,"boa_prediction.csv"),sep=",",row.names=FALSE,col.names=FALSE)
write.table(b$coefficients,file.path(d,"boa_final.csv"),sep=",",row.names=FALSE,col.names=FALSE)
set.seed(0)
X=data.frame(x1=sin(seq_len(200)/5),x2=cos(seq_len(200)/13))
Y=0.2*X$x1+0.7*X$x2+rnorm(200,sd=.03)
s=SuperLearner(Y=Y,X=X,family=gaussian(),SL.library=c("SL.mean","SL.glm"),method="method.NNLS",cvControl=list(V=5))
write.table(cbind(s$SL.predict,Y),file.path(d,"sl_official_example.csv"),sep=",",row.names=FALSE,col.names=FALSE)
saveRDS(s,file.path(d,"sl_official_example.rds"))
'''
(D/'run.R').write_text(script)
p=subprocess.run([str(B/'r-env/bin/Rscript'),str(D/'run.R'),str(B/'vendor'),str(D)],capture_output=True,text=True)
(D/'R.log').write_text(p.stdout+p.stderr)
if p.returncode:print(p.stderr[-2000:]);raise SystemExit(p.returncode)
a=superlearner_nnls(z,y);expected=np.loadtxt(D/'sl_coefficients.csv',delimiter=',')
pred,w,final=immediate_stream(z,y)
errs={'SL_raw':float(np.max(np.abs(a['raw']-expected[:,0]))),'SL_normalized':float(np.max(np.abs(a['weights']-expected[:,1]))),'BOA_prediction':float(np.max(np.abs(pred-np.loadtxt(D/'boa_prediction.csv',delimiter=',')))),'BOA_weights':float(np.max(np.abs(w-np.loadtxt(D/'boa_weights.csv',delimiter=',')))),'BOA_final':float(np.max(np.abs(final-np.loadtxt(D/'boa_final.csv',delimiter=','))))}
status='PASS' if max(errs.values())<1e-10 else 'FAIL'
(D/'numerical_parity.json').write_text(json.dumps({'status':status,'max_absolute_errors':errs,'source':'locked official R source with installed dependency package; no-delay BOA gradient defaults','L1_status':'REFERENCE_PARTIAL official synthetic example only, not full paper original numerical study'},indent=2));print(status,errs)
