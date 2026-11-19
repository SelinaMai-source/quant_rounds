import requests,re,json,time
from pathlib import Path
R=Path('/data1/stock-mxy/课题/一进二策略/result/round26_1/D5/public_data')
url='https://pan.baidu.com/s/1qmDIepmGY1DVBTGGiipxfA?pwd=pm49'
s=requests.Session();s.headers.update({'User-Agent':'Mozilla/5.0','Referer':url,'Origin':'https://pan.baidu.com'})
page=s.get(url,timeout=45);text=page.text
report=[]
def number(key):
 m=re.search(r'[\x22\x27]'+key+r'[\x22\x27]\s*:\s*[\x22\x27]?(\d+)',text)
 return m[1] if m else None
shareid=number('shareid');owner=number('share_uk')
report.append({'step':'owner_fields','shareid_present':bool(shareid),'share_uk_present':bool(owner),'visitor_uk_differs_from_owner':owner!=number('uk')})
params={'surl':'qmDIepmGY1DVBTGGiipxfA','channel':'chunlei','web':1,'app_id':250528,'clienttype':0,'t':int(time.time()*1000)}
v=s.post('https://pan.baidu.com/share/verify',params=params,data={'pwd':'pm49','vcode':'','vcode_str':''},timeout=45).json()
report.append({'step':'published_password','errno':v.get('errno')})
if v.get('errno')==0 and shareid and owner:
 params={'uk':owner,'shareid':shareid,'root':1,'page':1,'num':100,'order':'name','desc':0,'web':1,'clienttype':0,'channel':'chunlei','app_id':250528}
 response=s.get('https://pan.baidu.com/share/list',params=params,timeout=45)
 data=response.json();files=[{'name':f.get('server_filename'),'size':f.get('size'),'isdir':f.get('isdir')} for f in data.get('list',[])]
 report.append({'step':'correct_owner_file_list','status':response.status_code,'errno':data.get('errno'),'files':files})
 # Public owner identifiers and file IDs can be retained locally, never cookies/tokens.
 if data.get('errno')==0:
  Path('/data1/stock-mxy/课题/一进二策略/research/round23_7_26_1_execution/MASTER_public/baidu_listing.json').write_text(json.dumps(data))
(R/'BAIDU_OWNER_FIELD_CORRECTION_PROBE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False))
if v.get('errno')==0:
 params['root']=0;params['dir']='/MASTER-data'
 response=s.get('https://pan.baidu.com/share/list',params=params,timeout=45)
 data=response.json();files=[{'name':f.get('server_filename'),'size':f.get('size'),'isdir':f.get('isdir')} for f in data.get('list',[])]
 report.append({'step':'author_folder_files','status':response.status_code,'errno':data.get('errno'),'files':files})
 cfg=s.get('https://pan.baidu.com/share/tplconfig',params={'shareid':shareid,'uk':owner,'fields':'sign,timestamp'},timeout=45)
 try:
  cf=cfg.json();report.append({'step':'public_download_configuration','status':cfg.status_code,'errno':cf.get('errno'),'data_fields':list(cf.get('data',{}))})
 except ValueError:report.append({'step':'public_download_configuration','status':cfg.status_code,'json':False})
 (R/'BAIDU_PUBLIC_FOLDER_PROBE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print(json.dumps(report[-2:],ensure_ascii=False))
if data.get('errno')==0 and cf.get('errno')==0:
 from urllib.parse import unquote
 chosen=[f for f in data['list'] if f.get('server_filename')=='opensource.zip'];assert len(chosen)==1
 chosen=chosen[0]
 params={'sign':cf['data']['sign'],'timestamp':cf['data']['timestamp'],'channel':'chunlei','web':1,'app_id':250528,'clienttype':0}
 cookie=s.cookies.get('BDCLND')
 body={'encrypt':0,'product':'share','uk':owner,'primaryid':shareid,'fid_list':json.dumps([chosen['fs_id']]),'extra':json.dumps({'sekey':unquote(cookie or '')})}
 result=s.post('https://pan.baidu.com/api/sharedownload',params=params,data=body,timeout=45).json()
 report.append({'step':'public_file_download','errno':result.get('errno'),'download_link_present':bool(result.get('list') and result['list'][0].get('dlink'))})
 (R/'BAIDU_PUBLIC_DOWNLOAD_PROBE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report[-1]))
 if result.get('errno')==0 and result.get('list'):
  link=result['list'][0]['dlink']
  target=Path('/data1/stock-mxy/课题/一进二策略/research/round23_7_26_1_execution/MASTER_public/files/baidu_opensource.zip.partial')
  target.parent.mkdir(parents=True,exist_ok=True)
  assert not target.exists(),'Preserve existing partial; add verified range resume before retry'
  response=s.get(link,stream=True,timeout=(15,60))
  (R/'BAIDU_PUBLIC_DOWNLOAD_HTTP.json').write_text(json.dumps({'status':response.status_code,'expected_bytes':chosen['size']}))
  if response.status_code==200:
   size=0;last=time.monotonic()
   with target.open('wb') as out:
    for block in response.iter_content(1024*1024):
     if not block:continue
     out.write(block);size+=len(block);assert size<=chosen['size']
     if time.monotonic()-last>15:
      print('Author Baidu public mirror downloaded',size,'/',chosen['size'],flush=True);last=time.monotonic()
   assert size==chosen['size']
   from core.research_integrity_v5.cache import identity
   target=target.rename(target.with_suffix(''))
   proof=identity(target);proof['source']='author README Baidu public share; published password; no account authentication'
   (R/'BAIDU_PUBLIC_ARCHIVE_VERIFIED.json').write_text(json.dumps(proof,indent=2))
   from core.round26_1.master_public import extract_author_archive
   extract_author_archive({'path':str(target)})
