from pathlib import Path
import subprocess,json,urllib.request,hashlib
B=Path(__file__).resolve().parent
items=[('MASTER','SJTU-DMTai/MASTER','2312.15235',['qlib-update']),('RD-Agent','microsoft/RD-Agent','2505.15155',['rdagent','docs','test']),('DeepLOB','zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books','1808.03668',[])]
for n,repo,paper,dirs in items:
 p=B/'vendor'/n;r={'method':n,'repo':'https://github.com/'+repo,'paper':'https://arxiv.org/pdf/'+paper}
 try:
  if not (p/'.git').exists():
   a=subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse',r['repo'],str(p)],capture_output=True,text=True,timeout=600)
   r['clone_returncode']=a.returncode
   if a.returncode:r['error']=a.stderr[-1000:]
  if (p/'.git').exists():
   r['commit']=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
   if dirs:subprocess.run(['git','-C',str(p),'sparse-checkout','set',*dirs],check=True,capture_output=True,timeout=600)
  data=urllib.request.urlopen(r['paper'],timeout=60).read();(B/'vendor'/(n+'.pdf')).write_bytes(data)
  r['paper_sha256']=hashlib.sha256(data).hexdigest();r['paper_is_pdf']=data.startswith(b'%PDF');r['paper_bytes']=len(data)
 except Exception as e:r['error_type']=type(e).__name__;r['error']=str(e)[:800]
 (B/('acquisition_'+n+'.json')).write_text(json.dumps(r,indent=2));print(n,r.get('commit'),r.get('error_type','acquired'),flush=True)
