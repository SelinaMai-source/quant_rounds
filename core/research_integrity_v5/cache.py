"""Content identity, explicit contracts, atomic immutable completed artifacts."""
from pathlib import Path
from contextlib import contextmanager
import hashlib,json,os,tempfile,shutil,subprocess,importlib.metadata

REQUIRED=('code','third_party','dirty_diff','environment','parameters','seed','features','universe','calibration','data','label','maturity','windows','execution','upstream')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),default=str).encode()
def identity(path):
 p=Path(path);a=p.stat();h=sha(p);z=p.stat()
 if (a.st_size,a.st_mtime_ns,a.st_ino)!=(z.st_size,z.st_mtime_ns,z.st_ino):raise RuntimeError('Input changed while hashing: '+str(p))
 result={'path':str(p.resolve()),'bytes':z.st_size,'sha256':h}
 if p.suffix=='.parquet':
  import pyarrow.parquet as pq
  result['schema']=str(pq.read_schema(p));result['rows']=pq.read_metadata(p).num_rows
 return result
def fingerprint(contract):
 missing=set(REQUIRED)-set(contract)
 if missing:raise ValueError('Incomplete fingerprint: '+repr(sorted(missing)))
 return hashlib.sha256(canonical(contract)).hexdigest()
def source_manifest(roots):
 return [identity(p) for root in roots for p in (sorted(Path(root).rglob('*.py')) if Path(root).is_dir() else [Path(root)])]
def hit(path,fp):
 p=Path(path)
 try:
  marker=json.loads((p/'COMPLETE.json').read_text())
  return marker['experiment_fingerprint']==fp and bool(marker['artifacts']) and all((p/r['path']).is_file() and (p/r['path']).stat().st_size==r['bytes'] and sha(p/r['path'])==r['sha256'] for r in marker['artifacts'])
 except (OSError,ValueError,KeyError):return False
@contextmanager
def transaction(root,contract):
 root=Path(root);fp=fingerprint(contract);root.mkdir(parents=True,exist_ok=True);dest=root/fp
 if hit(dest,fp):yield dest,True;return
 if dest.exists():dest.rename(root/(fp+'.invalid.'+str(__import__('time').time_ns())))
 temp=Path(tempfile.mkdtemp(prefix=fp+'.partial.',dir=root))
 try:
  (temp/'contract.json').write_bytes(canonical(contract));yield temp,False
  artifacts=[{'path':str(p.relative_to(temp)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(temp.rglob('*')) if p.is_file() and p.name!='COMPLETE.json']
  if len(artifacts)<2:raise RuntimeError('No actual output written')
  marker=temp/'COMPLETE.json';marker.write_bytes(canonical({'experiment_fingerprint':fp,'artifacts':artifacts}))
  with marker.open('rb') as f:os.fsync(f.fileno())
  os.rename(temp,dest)
 except BaseException:
  (temp/'FAILED.json').write_text('{"status":"FAILED_NO_CACHE_HIT"}');raise
