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
 page=s.get(url.split('?')[0],timeout=45)
 root=Path('/data1/stock-mxy/课题/一进二策略/research/round23_7_26_1_execution/MASTER_public')
 (root/'baidu_verified.html').write_text(page.text)
 print('Verified share page',page.status_code,len(page.content),re.findall(r'<title>(.*?)</title>',page.text))
 print('\n'.join(re.findall(r'<script[^>]+src=[\x22\x27]([^\x22\x27]+)',page.text)))
