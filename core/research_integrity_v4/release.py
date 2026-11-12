"""Whitelist publication with secret/large-file checks and remote content verification."""
from pathlib import Path
import hashlib,json,os,re,subprocess


def _git(root,*args):
 p=subprocess.run(['git','-C',str(root),*args],capture_output=True)
 if p.returncode:raise RuntimeError((p.stdout+p.stderr).decode(errors='replace')[-2000:])
 return p.stdout


def publish(root,branch,commit=True,dry_run=False):
 root=Path(root)
 actual=_git(root,'branch','--show-current').decode().strip()
 if actual!=branch:raise RuntimeError('Refusing publication from a different branch')
 prefixes=['core/research_integrity_v4/','core/'+branch+'/','research/canonical_dependencies/','research/literature_round23_6_26/','research/vendor_sources/','result/'+branch+'/']
 if branch=='round26':prefixes.append('result/round23_6_round26_master/')
 files_allowed=['.gitattributes','tools/run_round23_6.py','tools/run_round26.py','tools/run_round23_6_round26.py']
 changed=set(_git(root,'diff','--name-only','-z','HEAD').decode().split('\0'))|set(_git(root,'ls-files','--others','--exclude-standard','-z').decode().split('\0'))
 changed.discard('')
 foreign=[p for p in changed if p not in files_allowed and not any(p.startswith(x) for x in prefixes)]
 if foreign:raise RuntimeError('Unrelated changes retained; refuse staging: '+repr(foreign[:10]))
 # Current-user configured provider secrets are checked internally and never printed.
 needles=[]
 config=Path('/home/mxy/AmazingData/.env')
 if config.exists() and config.stat().st_uid==os.getuid():
  from dotenv import dotenv_values
  for k,v in dotenv_values(config).items():
   if v and any(x in k.upper() for x in ['PASSWORD','TOKEN','SECRET','USERNAME']) and len(v)>=8:needles.append(v.encode())
 problems=[];total=0
 committed=set(_git(root,'diff','--name-only','-z','2c28c2083097a28cda921c59c28b254c30cd0976','HEAD').decode().split('\0'));committed.discard('')
 scanfiles=changed|committed
 for name in sorted(scanfiles):
  p=root/name
  if not p.is_file():continue
  if p.is_symlink():problems.append({'path':name,'reason':'symlink requires explicit source review'});continue
  if p.name=='.env' or p.suffix in ['.pem','.key'] or any(x in p.parts for x in ['site-packages','__pycache__']):problems.append({'path':name,'reason':'forbidden private/runtime artifact'})
  if p.stat().st_size>90*1024*1024:problems.append({'path':name,'reason':'large file exceeds 90MiB'})
  data=p.read_bytes();total+=len(data)
  if any(s in data for s in needles):problems.append({'path':name,'reason':'configured credential value found; value redacted'})
  for pattern in [rb'sk-(?:proj-)?[A-Za-z0-9_-]{24,}',rb'gh[pousr]_[A-Za-z0-9_]{25,}',rb'AKIA[0-9A-Z]{16}']:
   if re.search(pattern,data):problems.append({'path':name,'reason':'credential-shaped literal; value redacted'})
  if re.search(rb'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----',data):problems.append({'path':name,'reason':'private key material'})
 if problems:raise RuntimeError('Release scan failed: '+json.dumps(problems,ensure_ascii=False))
 if dry_run:return {'status':'SCAN_PASSED','files_checked':len(scanfiles),'bytes_checked':total,'credential_findings':0,'large_file_findings':0}
 if commit and changed:
  paths=[str(root/x) for x in prefixes if (root/x).exists()]+[str(root/x) for x in files_allowed if (root/x).exists()]
  _git(root,'add','--',*paths)
  _git(root,'diff','--cached','--check')
  _git(root,'commit','-m','results: publish executed research, explicit partial states and reproducibility evidence')
 elif changed:raise RuntimeError('Working tree is uncommitted')
 head=_git(root,'rev-parse','HEAD').decode().strip()
 _git(root,'push','-u','origin','HEAD:'+branch)
 remote=_git(root,'ls-remote','origin','refs/heads/'+branch).decode().split()[0]
 if head!=remote:raise RuntimeError('Remote SHA mismatch')
 _git(root,'fetch','origin','refs/heads/'+branch)
 report='ROUND23_6_ENSEMBLE_REPORT.md' if branch=='round23_6' else 'ROUND26_ALPHA_REPORT.md'
 selected=['result/'+branch+'/'+report,'tools/run_'+branch+'.py','core/research_integrity_v4/test_time_safety.py','result/'+branch+'/all_results.csv','result/'+branch+'/无权限数据.md','result/'+branch+'/source_dependency_manifest.csv','result/'+branch+'/artifact_manifest.csv']
 if branch=='round23_6':selected.append('core/round23_6/meta.py')
 verified=[]
 for path in selected:
  data=_git(root,'show','FETCH_HEAD:'+path);local=(root/path).read_bytes()
  if data!=local:raise RuntimeError('Fetched remote contents differ: '+path)
  verified.append({'path':path,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
 return {'branch':branch,'commit':head,'remote_sha':remote,'REMOTE_SHA_VERIFIED':True,'remote_content_verified':verified,'release_scan':{'files_checked_since_frozen_anchor':len(scanfiles),'bytes_checked':total,'credential_findings':0,'large_file_findings':0},'status':'PARTIAL','url':'https://github.com/SelinaMai-source/quant_rounds/tree/'+branch}
