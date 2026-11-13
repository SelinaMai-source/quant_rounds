"""Explicit-whitelist normal publication and fetched artifact verification."""
from pathlib import Path
import subprocess,json,hashlib,re,shutil,csv,os
from .cache import sha
C=Path('/data1/stock-mxy/课题/一进二策略');B=C/'research/round23_7_26_1_execution'
def git(w,*args):
 p=subprocess.run(['git','-C',str(w),*args],capture_output=True)
 if p.returncode:raise RuntimeError((p.stdout+p.stderr).decode(errors='replace')[-3000:])
 return p.stdout
def prepare(branch):
 w=B/branch
 for n in ['research_integrity_v5',branch]:
  src=C/'core'/n
  if src.exists():shutil.copytree(src,w/'core'/n,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 src=C/'result'/branch
 if src.exists():shutil.copytree(src,w/'result'/branch,dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.partial.*','residual_true_OOF.parquet','baseline_true_OOF_scores.parquet'))
 tools=C/'tools/continue_round23_7_round26_1.py'
 if tools.exists():shutil.copy2(tools,w/'tools'/tools.name)
 for script in ['finalize_d0.py','finalize_d1.py','finalize_d2.py']:
  src=B/script
  if src.exists():shutil.copy2(src,w/'research/round23_7_round26_1'/script)
 # All self-owned source, contracts and derived output files bind to exact bytes.
 roots=[w/'core/research_integrity_v5',w/'core'/branch,w/'result'/branch,w/'research/round23_7_round26_1']
 rows=[]
 for root in roots:
  if not root.exists():continue
  for p in sorted(root.rglob('*')):
   if p.is_file() and p.name!='artifact_manifest.csv' and '__pycache__' not in p.parts:rows.append({'path':str(p.relative_to(w)),'bytes':p.stat().st_size,'sha256':sha(p)})
 if tools.exists():p=w/'tools'/tools.name;rows.append({'path':str(p.relative_to(w)),'bytes':p.stat().st_size,'sha256':sha(p)})
 with (w/'result'/branch/'artifact_manifest.csv').open('w') as f:
  out=csv.DictWriter(f,fieldnames=['path','bytes','sha256'],lineterminator='\n');out.writeheader();out.writerows(rows)
 return w
def publish(branch,message):
 w=prepare(branch);assert git(w,'branch','--show-current').decode().strip()==branch
 prefixes=['core/research_integrity_v5/','core/'+branch+'/','result/'+branch+'/','research/round23_7_round26_1/'];allowed=['tools/continue_round23_7_round26_1.py']
 changed=set(git(w,'diff','--name-only','HEAD','-z').decode().split('\0'))|set(git(w,'ls-files','--others','--exclude-standard','-z').decode().split('\0'));changed.discard('')
 assert all(p in allowed or any(p.startswith(x) for x in prefixes) for p in changed),changed
 for name in changed:
  p=w/name
  if not p.exists():continue
  assert not p.is_symlink() and p.stat().st_size<90*1024*1024,name
  assert p.name!='.env' and p.suffix not in ['.key','.pem'] and '__pycache__' not in p.parts,name
  data=p.read_bytes()
  for pattern in [rb'sk-(?:proj-)?[A-Za-z0-9_-]{24,}',rb'gh[pousr]_[A-Za-z0-9_]{25,}',rb'AKIA[0-9A-Z]{16}',rb'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----']:
   assert not re.search(pattern,data),'credential-shaped material in '+name
 paths=[p for p in prefixes+allowed if (w/p).exists()];git(w,'add','--',*paths);git(w,'diff','--cached','--check')
 if git(w,'diff','--cached','--name-only').strip():git(w,'commit','-m',message)
 git(w,'push','-u','origin','HEAD:'+branch);head=git(w,'rev-parse','HEAD').decode().strip();remote=git(w,'ls-remote','origin','refs/heads/'+branch).decode().split()[0];assert remote==head
 git(w,'fetch','origin','refs/heads/'+branch);assert git(w,'rev-parse','FETCH_HEAD').decode().strip()==head
 rows=list(csv.DictReader((w/'result'/branch/'artifact_manifest.csv').open()))
 # Entire manifest verified against fetched commit objects, including source/tests/results.
 batch=''.join('FETCH_HEAD:'+r['path']+'\n' for r in rows).encode();blobdata=subprocess.run(['git','-C',str(w),'cat-file','--batch'],input=batch,capture_output=True,check=True).stdout;pos=0
 for r in rows:
  end=blobdata.index(b'\n',pos);header=blobdata[pos:end].split();assert header[1]==b'blob',r['path'];size=int(header[2]);start=end+1;data=blobdata[start:start+size];pos=start+size+1;assert hashlib.sha256(data).hexdigest()==r['sha256'],r['path']
 assert git(w,'show','FETCH_HEAD:result/'+branch+'/artifact_manifest.csv')==(w/'result'/branch/'artifact_manifest.csv').read_bytes()
 result={'branch':branch,'commit':head,'remote_sha':remote,'manifest_artifacts_verified':len(rows),'clean':not bool(git(w,'status','--porcelain')),'source_tests_results_permissions_manifest_verified':True}
 m=C/'result/round23_7_round26_1_master';m.mkdir(parents=True,exist_ok=True);(m/(branch+'_remote_verification.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return result
