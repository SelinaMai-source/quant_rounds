from pathlib import Path
import json,re,pyarrow.parquet as pq
B=Path('/data1/stock-mxy/课题/一进二策略/research/round23_6_26_execution');R=B.parents[1]/'result/round23_6_round26_master'
roots=[Path('/data1/stock-mxy/data'),Path('/data1/Stock/data/acquisitions'),Path('/data1/Stock/data_live'),Path('/home/mxy/AmazingData/data')]
seen=set();matches=[]
for root in roots:
 for p in root.rglob('*.parquet'):
  path=str(p)
  if path in seen:continue
  seen.add(path)
  if not re.search('tick|orderbook|lob|snapshot|auction',p.name,re.I):continue
  try:
   f=pq.ParquetFile(p);cols=f.schema_arrow.names
   matches.append({'path':path,'rows':f.metadata.num_rows,'columns':cols,'has_ask10':any(re.search('ask.*10|ask10|a10',c,re.I) for c in cols),'has_bid10':any(re.search('bid.*10|bid10|b10',c,re.I) for c in cols)})
  except (OSError,ValueError):continue
(R/'LOB_local_schema_search.json').write_text(json.dumps({'parquet_paths_considered':len(seen),'filename_matches_inspected':len(matches),'ten_level_candidates':[m for m in matches if m['has_ask10'] and m['has_bid10']],'matches':matches},indent=2));print('Schema search',len(seen),len(matches))
