import os,json,contextlib,inspect
from pathlib import Path
from dotenv import dotenv_values
R=Path('/data1/stock-mxy/课题/一进二策略/result/round23_6_round26_master')
# SDK output is suppressed at file-descriptor level so credentials cannot enter logs.
out=os.dup(1);err=os.dup(2)
with open(os.devnull,'w') as null:os.dup2(null.fileno(),1);os.dup2(null.fileno(),2)
record={'source_uri':'amazingdata://query_snapshot/000001.SZ/20260723','rows_recovered':0}
try:
 import AmazingData as a
 cfg=dotenv_values('/home/mxy/AmazingData/.env')
 result=a.login(cfg['AMAZINGDATA_USERNAME'],cfg['AMAZINGDATA_PASSWORD'],cfg['AMAZINGDATA_HOST'],int(cfg['AMAZINGDATA_PORT']))
 record['login_result_type']=type(result).__name__
 # Keep only numerical error codes; no account values.
 if isinstance(result,(int,bool)):record['login_code']=int(result)
 base=a.BaseData();cal=base.get_calendar()
 market=a.MarketData(cal)
 data=market.query_snapshot(['000001.SZ'],20260723,20260723)
 record['response_type']=type(data).__name__
 if isinstance(data,dict):
  record['response_keys']=[str(k) for k in data.keys()][:10]
  for key,value in data.items():
   if hasattr(value,'columns'):record['columns']=list(value.columns);record['rows_recovered']+=len(value)
   if isinstance(value,dict):
    for frame in value.values():
     if hasattr(frame,'columns'):record['columns']=list(frame.columns);record['rows_recovered']+=len(frame)
 if isinstance(data,(list,tuple)):
  payload,code=data[0],data[1]
  record['api_error_code']=code if isinstance(code,(int,float,bool,type(None))) else str(code)[:200]
  if isinstance(payload,dict):
   record['date_count']=len(payload)
   for date,stocks in payload.items():
    if isinstance(stocks,dict):
     for symbol,frame in stocks.items():
      if hasattr(frame,'columns'):record['columns']=list(frame.columns);record['rows_recovered']+=len(frame)
 record['status']='READ_OK' if record['rows_recovered'] else 'API_RESPONSE_REQUIRES_REVIEW'
 record['product_semantics']='SDK query_snapshot explicitly supports LEVEL1 only; not ten-level LOB'
 # SDK logout omitted: process teardown closes the session.
except Exception as e:
 record['status']='SDK_ERROR';record['error_type']=type(e).__name__
finally:
 os.dup2(out,1);os.dup2(err,2)
 (R/'amazing_probe.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
 print(json.dumps(record,ensure_ascii=False))
