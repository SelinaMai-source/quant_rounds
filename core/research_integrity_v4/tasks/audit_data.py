from pathlib import Path
import os,json,hashlib,urllib.request,urllib.error,datetime,stat,sys
import pyarrow.parquet as pq
C=Path('/data1/stock-mxy/课题/一进二策略');R=C/'result/round23_6_round26_master';R.mkdir(exist_ok=True)
paths=['/data1/stock-mxy/data/datasets/hermes-cn-a-2010-20260724-v7-all-features','/data1/Stock/data/acquisitions/hermes-cn-a-2010-20260724-v7-rebuild/raw','/data1/stock-mxy/data/source_snapshots','/data1/Stock/data_live/inbox/batches','/data1/Stock/data_live/inbox/server_relay','/data1/Stock/.jq_stage','/home/mxy/AmazingData/data/amazingdata_raw','/data1/stock-mxy/data/factor_cache','/data1/Stock/.zyquant/cache/factors-pit-ex-st-v2']
records=[]
for path in paths:
 p=Path(path);r={'source_uri':path,'action':'list/read parquet metadata','retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows_recovered':0}
 try:
  children=list(p.iterdir());r['entries']=[x.name for x in children[:50]]
  r['status']='READABLE';files=[]
  for f in p.rglob('*.parquet'):
   files.append(f)
   if len(files)>=3:break
  r['schema_samples']=[]
  for f in files:
   try:
    q=pq.ParquetFile(f);r['schema_samples'].append({'path':str(f),'rows':q.metadata.num_rows,'columns':q.schema_arrow.names})
   except Exception as e:r['schema_samples'].append({'path':str(f),'error':type(e).__name__})
 except PermissionError:r['status']='PERMISSION_DENIED'
 except FileNotFoundError:r['status']='DATA_NOT_FOUND'
 r['pit']='UNVERIFIED_PIT';records.append(r)
# Existing configuration keys only; never echo values; exclude root-owned 0600 files.
configs=[Path('/home/mxy/AmazingData/.env'),Path('/data1/Stock/.env'),Path('/data1/stock-mxy/.env')]
for p in configs:
 if not p.exists():continue
 st=p.stat()
 if st.st_uid==0 and not(st.st_mode & stat.S_IROTH):
  records.append({'source_uri':str(p),'status':'CREDENTIAL_FILE_EXCLUDED','reason':'root-owned private credential file; not opened'});continue
 try:
  text=p.read_text();keys=[line.split('=',1)[0].strip() for line in text.splitlines() if '=' in line and not line.lstrip().startswith('#')]
  records.append({'source_uri':str(p),'status':'CONFIG_KEYS_INSPECTED','keys':keys})
 except Exception as e:records.append({'source_uri':str(p),'status':type(e).__name__})
# Actual read-only ClickHouse query to named required sources, one authentication denial stops repeated calls.
sql='SELECT * FROM hermes.fdmt_ef_new LIMIT 1 FORMAT JSONEachRow'
try:
 req=urllib.request.Request('http://127.0.0.1:18123/',data=sql.encode(),method='POST')
 data=urllib.request.urlopen(req,timeout=15).read()
 records.append({'source_uri':'clickhouse://127.0.0.1:18123/hermes/fdmt_ef_new','status':'READ_OK','response_bytes':len(data),'rows_recovered':len(data.splitlines()),'pit':'UNVERIFIED_PIT'})
except urllib.error.HTTPError as e:
 body=e.read().decode(errors='replace')
 code='AUTH_REQUIRED' if ('AUTHENTICATION_FAILED' in body or e.code==401) else 'QUERY_ERROR'
 records.append({'source_uri':'clickhouse://127.0.0.1:18123/hermes/fdmt_ef_new','status':code,'http_status':e.code,'error_code':'AUTHENTICATION_FAILED' if code=='AUTH_REQUIRED' else 'HTTP_ERROR','rows_recovered':0,'pit':'UNVERIFIED_PIT'})
except Exception as e:records.append({'source_uri':'clickhouse://127.0.0.1:18123/hermes/fdmt_ef_new','status':'CONNECTION_ERROR','error_type':type(e).__name__,'rows_recovered':0})
# Snapshot identity and per-table schema hashes
snap=Path(paths[0]);tables=[]
for p in snap.rglob('*.parquet'):
 try:
  f=pq.ParquetFile(p);tables.append({'path':str(p),'rows':f.metadata.num_rows,'columns':f.schema_arrow.names,'size':p.stat().st_size})
 except Exception as e:tables.append({'path':str(p),'error_type':type(e).__name__})
(R/'snapshot_table_schema.json').write_text(json.dumps(tables,ensure_ascii=False,indent=2))
(R/'data_recovery_jobs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
p=Path('/data1/stock-mxy/数据清单.md');(R/'data_inventory_input.md').write_bytes(p.read_bytes())
(R/'data_inventory_input.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest())
print(json.dumps([{'source':r['source_uri'],'status':r['status']} for r in records],ensure_ascii=False))
