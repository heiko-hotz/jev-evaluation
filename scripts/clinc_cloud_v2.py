"""Explicit schema adaptation; preserves all first-run Jev responses without repeats."""
import argparse,json,math,os,time,urllib.request,urllib.error
from pathlib import Path
import clinc_eval as base
ROOT=base.ROOT
PARENT=ROOT/'local/runs/clinc150-v1-2026-09-21T053002.912317Z'
AMENDMENT=ROOT/'docs/clinc150-cloud-v2-amendment.md'
def payload(model,case,criteria):
 body=base.payload(model,case,criteria)
 if model!='jev':
  body['systemInstruction']['parts'][0]['text']=base.INSTRUCTION+' Return its integer index as route.\nCategories:\n'+json.dumps([{'index':i,'label':k,'description':v} for i,(k,v) in enumerate(criteria.items())])
  body['generationConfig']['responseJsonSchema']={'type':'object','properties':{'route':{'type':'integer','minimum':0,'maximum':len(criteria)-1}},'required':['route'],'additionalProperties':False}
 return body

def normalize(model,raw,criteria):
 if model=='jev':
  assert raw['model']==base.MODELS[model];a=raw['answers']['route'];p=a['probabilities'];choice=a['choice'];assert a['type']=='choice' and choice in criteria and set(p)==set(criteria)
  assert all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for x in p.values()) and sum(p.values())>0
  assert p[choice]==max(p.values()) and type(a['confidence']) in (int,float) and 0<=a['confidence']<=1
  assert all(type(raw['usage'][k]) is int and raw['usage'][k]>=0 for k in ['input_tokens','output_tokens'])
  return dict(choice=choice,probabilities=p,probability_mass=sum(p.values()),confidence=a['confidence'],model=raw['model'],usage=raw['usage'])
 version=raw['modelVersion'];assert version==base.MODELS[model] or version.startswith(base.MODELS[model]+'-')
 cs=raw['candidates'];assert len(cs)==1 and cs[0]['finishReason']=='STOP';parts=cs[0]['content']['parts'];assert parts and all(isinstance(p.get('text'),str) and not p.get('thought') for p in parts)
 obj=json.loads(''.join(p['text'] for p in parts));assert isinstance(obj,dict) and set(obj)=={'route'} and type(obj['route']) is int and 0<=obj['route']<len(criteria)
 usage=raw['usageMetadata'];assert all(type(usage[k]) is int and usage[k]>=0 for k in ['promptTokenCount','candidatesTokenCount','totalTokenCount'])
 return dict(choice=list(criteria)[obj['route']],route_index=obj['route'],model=version,usage=usage)

def init():
 cases,criteria,meta=base.verify(PARENT);folder=PARENT/'cloud-v2';folder.mkdir(exist_ok=False);os.umask(0o077)
 for m in ['jev','gemini','gemma']:
  (folder/m).mkdir();base.dump(folder/m/'requests.json',[dict(id=c['id'],payload=payload(m,c,criteria)) for c in cases])
 extra={str(p.relative_to(ROOT)):base.digest(p) for p in [Path(__file__).resolve(),AMENDMENT]}
 base.dump(folder/'manifest.json',dict(parent=str(PARENT),timestamp=base.now(),file_sha256=meta['file_sha256']|extra,request_sha256={m:base.digest(folder/m/'requests.json') for m in ['jev','gemini','gemma']},spending_ceiling_usd=None,retries=0,requests_per_model=1834))
 print(folder)

def run(model):
 folder=PARENT/'cloud-v2';cases,criteria,meta=base.verify(folder);f=folder/model;assert base.digest(f/'requests.json')==meta['request_sha256'][model]
 requests=json.loads((f/'requests.json').read_text());assert requests==[dict(id=c['id'],payload=payload(model,c,criteria)) for c in cases]
 with (f/'started.json').open('x') as out:json.dump({'utc':base.now()},out)
 rows=[];wall=time.monotonic()
 if model=='jev':
  saved=json.loads((PARENT/model/'results.json').read_text());assert len(saved)==3
  for c,old in zip(cases,saved):
   assert c['id']==old['id'];source=PARENT/model/(c['id']+'.json');raw=json.loads(source.read_text());row={k:v for k,v in old.items() if k not in ['error','http_status']};row.update(normalize(model,raw,criteria),reused_from=str(source));(f/source.name).write_bytes(source.read_bytes());rows.append(row)
  base.dump(f/'results.json',rows)
 key=base.smoke.key_from_environment() if model=='jev' else base.gemini.read_key(ROOT.parent/'ace-economic-sim/.env.local')
 endpoint='https://api.typesafe.ai/v1/systemone' if model=='jev' else 'https://generativelanguage.googleapis.com/v1beta/models/'+base.MODELS[model]+':generateContent'
 headers=({'Authorization':'Bearer '+key} if model=='jev' else {'x-goog-api-key':key})|{'Content-Type':'application/json'};op=urllib.request.build_opener(urllib.request.ProxyHandler({}),base.smoke.NoRedirect())
 for c,item in list(zip(cases,requests))[len(rows):]:
  row=dict(id=c['id'],expected=c['expected'],started_utc=base.now());start=time.monotonic()
  with (f/'attempts.jsonl').open('a') as out:out.write(json.dumps(row)+'\n');out.flush();os.fsync(out.fileno())
  try:
   req=urllib.request.Request(endpoint,data=json.dumps(item['payload']).encode(),headers=headers,method='POST')
   with op.open(req,timeout=120 if model=='gemma' else 60) as r:b=r.read()
   row['latency_seconds']=time.monotonic()-start;(f/(c['id']+'.json')).write_bytes(b);row.update(normalize(model,json.loads(b),criteria))
  except Exception as e:
   row.update(error=type(e).__name__,http_status=getattr(e,'code',None),latency_seconds=time.monotonic()-start)
   if isinstance(e,urllib.error.HTTPError):(f/(c['id']+'-error.json')).write_bytes(e.read())
  rows.append(row);base.dump(f/'results.json',rows)
  if 'error' in row:print(model,'STOP',row,flush=True);break
  if len(rows)%50==0 or len(rows)==1:print(model,len(rows),'/',len(cases),flush=True)
 summary=dict(attempted=len(rows),valid=sum('error' not in r for r in rows),complete=len(rows)==len(cases) and all('error' not in r for r in rows),wall_seconds=time.monotonic()-wall,finished_utc=base.now())
 base.dump(f/'summary.json',summary);print(model,json.dumps(summary),flush=True)
 if not summary['complete']:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['init','run']);p.add_argument('--model',choices=['jev','gemini','gemma']);a=p.parse_args()
 if a.command=='init':init()
 else:run(a.model)
