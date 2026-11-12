import subprocess,json,pathlib,urllib.request,hashlib,time
ROOT=pathlib.Path(__file__).resolve().parent
sources=[('SuperLearner','ecpolley/SuperLearner','https://doi.org/10.2202/1544-6115.1309'),('opera','Dralliag/opera','https://arxiv.org/pdf/1404.1356'),('TRA','microsoft/qlib','https://arxiv.org/pdf/2106.12950'),('DoubleAdapt','SJTU-DMTai/DoubleAdapt','https://arxiv.org/pdf/2306.09862'),('AlphaGen','ICT-FinD-Lab/alphagen','https://arxiv.org/pdf/2306.12964'),('AlphaForge','DulyHao/AlphaForge','https://arxiv.org/pdf/2406.18394'),('RSR','fulifeng/Temporal_Relational_Stock_Ranking','https://arxiv.org/pdf/1809.09441'),('MASTER','SJTU-DMTai/MASTER','https://arxiv.org/pdf/2312.15235'),('RD-Agent','microsoft/RD-Agent','https://arxiv.org/pdf/2505.15155'),('DeepLOB','zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books','https://arxiv.org/pdf/1808.03668')]
records=[]
for name,repo,paper in sources:
 p=ROOT/'vendor'/name
 r={'method':name,'repo':'https://github.com/'+repo,'paper':paper}
 if not (p/'.git').exists():
  proc=subprocess.run(['git','clone','--depth','1','--filter=blob:none',r['repo'],str(p)],capture_output=True,text=True,timeout=1200)
  r['clone_returncode']=proc.returncode;r['clone_error']=proc.stderr[-1000:] if proc.returncode else ''
 if (p/'.git').exists():
  r['commit']=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
  r['licenses']={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if 'license' in f.name.lower() or 'copying' in f.name.lower()}
 try:
  req=urllib.request.Request(paper,headers={'User-Agent':'Mozilla/5.0'})
  data=urllib.request.urlopen(req,timeout=60).read(); dest=ROOT/'vendor'/(name+'.pdf');dest.write_bytes(data)
  r['paper_sha256']=hashlib.sha256(data).hexdigest();r['paper_bytes']=len(data);r['paper_is_pdf']=data.startswith(b'%PDF')
 except Exception as e:r['paper_error']=str(e)
 records.append(r);(ROOT/'source_acquisition.json').write_text(json.dumps(records,indent=2));print(name,r.get('commit'),r.get('paper_error','downloaded'),flush=True)
