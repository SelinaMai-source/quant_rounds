from pathlib import Path
import hashlib,csv,sys
B=Path('/data1/stock-mxy/课题/一进二策略/research/round23_6_26_execution')
for branch in sys.argv[1:] or ['round23_6','round26']:
 W=B/branch;R=W/'result'/branch
 roots=[R,W/'core/research_integrity_v4',W/'core'/branch,W/'research/canonical_dependencies',W/'research/literature_round23_6_26',W/'research/vendor_sources']
 if branch=='round26':roots.append(W/'result/round23_6_round26_master')
 files=set()
 for root in roots:
  if root.exists():files.update(p for p in root.rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.name!='artifact_manifest.csv')
 files.update(p for p in [W/'.gitattributes',*[W/'tools'/n for n in ['run_round23_6.py','run_round26.py','run_round23_6_round26.py']]] if p.exists())
 with (R/'artifact_manifest.csv').open('w') as out:
  writer=csv.DictWriter(out,fieldnames=['path','size','sha256','license'],lineterminator='\n');writer.writeheader()
  for p in sorted(files):
   data=p.read_bytes();writer.writerow({'path':str(p.relative_to(W)),'size':len(data),'sha256':hashlib.sha256(data).hexdigest(),'license':'See source_lock.json and third-party notices; canonical project code and user-authorized derived outputs; no supplier raw rows'})
 print(branch,'manifest entries',len(files))
