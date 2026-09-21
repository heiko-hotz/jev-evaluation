"""SNIPS seven-intent benchmark. Frozen requests, strict scoring, bounded dispatch."""
import argparse,copy,json,math,os,platform,subprocess,time,urllib.request,urllib.error
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import clinc_eval as prior
import clinc_cloud_v2 as jev_parser
ROOT=prior.ROOT;DATA=ROOT/'data/snips-v1.json';PROTOCOL=ROOT/'docs/snips-v1-protocol.md'
MODELS=prior.MODELS
CRITERIA={
 'AddToPlaylist':'Add a song, album, or artist to a music playlist.',
 'BookRestaurant':'Reserve a table at a restaurant.',
 'GetWeather':'Ask about weather conditions or forecasts.',
 'PlayMusic':'Play or listen to music, a song, an album, or an artist.',
 'RateBook':'Give a rating or review to a book.',
 'SearchCreativeWork':'Find a creative work such as a book, film, television show, song, or game; not a screening time or a request to play music.',
 'SearchScreeningEvent':'Find movie screenings, cinema showtimes, or movie tickets.'}
INSTRUCTION='Which intent best matches this user request? Select the single best category.'
CAP=700
SOURCES=['scripts/snips_eval.py','scripts/snips_deberta_remote.py','scripts/clinc_eval.py','scripts/clinc_cloud_v2.py','scripts/smoke.py','scripts/gemini_challenge.py','scripts/deberta.py','scripts/deberta_remote.py','scripts/deberta_challenge.py','docs/snips-v1-protocol.md','data/snips-v1.json','data/snips-provenance-v1.json']
def dump(path,obj):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def payload(model,case):
 if model=='jev':return {'model':MODELS[model],'state':{'request':case['text']},'questions':{'route':{'type':'choice','instructions':INSTRUCTION,'criteria':CRITERIA}}}
 if model=='deberta':return {'inputs':case['text'],'parameters':{'candidate_labels':[f'{k}: {v}' for k,v in CRITERIA.items()],'multi_label':False,'hypothesis_template':prior.deberta.TEMPLATE}}
 body=copy.deepcopy(prior.gemini.payload(case));body['systemInstruction']['parts'][0]['text']=INSTRUCTION+'\nCategories:\n'+json.dumps(CRITERIA)
 body['generationConfig']['responseJsonSchema']={'type':'object','properties':{'route':{'type':'string','enum':list(CRITERIA)}},'required':['route'],'additionalProperties':False}
 if model=='gemma':body['generationConfig'].update(maxOutputTokens=8192,thinkingConfig={'thinkingLevel':'HIGH','includeThoughts':False})
 return body

def normalize(model,raw):
 if model=='jev':return jev_parser.normalize(model,raw,CRITERIA)
 if model=='deberta':return prior.normalize(model,raw,CRITERIA)
 version=raw.get('modelVersion','');assert version==MODELS[model] or version.startswith(MODELS[model]+'-'),'Unexpected model'
 usage=raw['usageMetadata'];assert all(type(usage[k]) is int and usage[k]>=0 for k in ['promptTokenCount','candidatesTokenCount','totalTokenCount'])
 assert all(type(usage.get(k,0)) is int and usage.get(k,0)>=0 for k in ['thoughtsTokenCount','cachedContentTokenCount'])
 result=dict(model=version,usage=usage)
 try:
  cs=raw['candidates'];assert len(cs)==1 and cs[0]['finishReason']=='STOP'
  parts=cs[0]['content']['parts'];assert parts and all(isinstance(p.get('text'),str) and not p.get('thought') for p in parts)
  value=json.loads(''.join(p['text'] for p in parts));assert isinstance(value,dict) and set(value)=={'route'} and value['route'] in CRITERIA
  result['choice']=value['route']
 except (AssertionError,KeyError,ValueError,TypeError) as e:result.update(choice=None,invalid_output=True,validation_error=type(e).__name__)
 return result

def verify(folder):
 meta=json.loads((folder/'manifest.json').read_text())
 for name,sha in meta['file_sha256'].items():assert prior.digest(ROOT/name)==sha,name
 cases=json.loads(DATA.read_text());assert len(cases)==CAP and Counter(c['expected'] for c in cases)==dict.fromkeys(CRITERIA,100)
 return cases,meta

def init():
 cases=json.loads(DATA.read_text());assert len(cases)==CAP and len({c['id'] for c in cases})==CAP
 assert Counter(c['expected'] for c in cases)==dict.fromkeys(CRITERIA,100)
 os.umask(0o077);folder=ROOT/'local/runs'/datetime.now(timezone.utc).strftime('snips-v1-%Y-%m-%dT%H%M%S.%fZ')
 with (ROOT/'local/snips-v1-dispatch.json').open('x') as f:json.dump({'bundle':str(folder)},f)
 folder.mkdir(parents=True)
 for m in MODELS:
  (folder/m).mkdir();dump(folder/m/'requests.json',[dict(id=c['id'],payload=payload(m,c)) for c in cases])
 dump(folder/'manifest.json',dict(timestamp=prior.now(),models=MODELS,criteria=CRITERIA,requests_per_model=CAP,spending_ceiling_usd=None,concurrency_per_model=1,max_429_retries_per_model=10,gemma_min_interval_seconds=2.5,
  file_sha256={name:prior.digest(ROOT/name) for name in SOURCES},request_sha256={m:prior.digest(folder/m/'requests.json') for m in MODELS},python=platform.python_version(),platform=platform.platform(),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty=True))
 print(folder)

def run(folder,model):
 cases,meta=verify(folder);dest=folder/model;assert prior.digest(dest/'requests.json')==meta['request_sha256'][model]
 requests=json.loads((dest/'requests.json').read_text());assert requests==[dict(id=c['id'],payload=payload(model,c)) for c in cases]
 with (dest/'started.json').open('x') as f:json.dump({'utc':prior.now()},f)
 rows=[];wall=time.monotonic();attempts=0;retry_rows=[]
 if model=='deberta':
  prior.snapshot(dest,'before');runner='deberta-'+folder.name+'-runner.py';remote_input='deberta-'+folder.name+'-requests.json';wrappers=prior.deberta.WRAPPERS
  subprocess.run([str(wrappers/'copy-to-tmp.sh'),str(ROOT/'scripts/snips_deberta_remote.py'),runner],check=True)
  subprocess.run([str(wrappers/'copy-to-tmp.sh'),str(dest/'requests.json'),remote_input],check=True)
  proc=subprocess.Popen([str(wrappers/'ssh.sh'),'/usr/bin/python3','-u','/tmp/'+runner,'/tmp/'+remote_input],stdout=subprocess.PIPE,text=True)
  with (dest/'remote-responses.jsonl').open('x') as log:
   for line in proc.stdout:
    log.write(line);log.flush();row=json.loads(line);c=cases[len(rows)];assert row['id']==c['id'];attempts+=1
    if 'response' in row:
     raw=row.pop('response');dump(dest/(c['id']+'.json'),raw);row.update(normalize(model,raw))
    row['expected']=c['expected'];rows.append(row);dump(dest/'results.json',rows)
    if len(rows)%50==0 or len(rows)==1:print(model,len(rows),'/',CAP,'latest_seconds',row['latency_seconds'],flush=True)
  rc=proc.wait();prior.snapshot(dest,'after');subprocess.run([str(wrappers/'ssh.sh'),'rm','--','/tmp/'+runner,'/tmp/'+remote_input],check=True)
 else:
  key=prior.smoke.key_from_environment() if model=='jev' else prior.gemini.read_key(ROOT.parent/'ace-economic-sim/.env.local')
  url='https://api.typesafe.ai/v1/systemone' if model=='jev' else 'https://generativelanguage.googleapis.com/v1beta/models/'+MODELS[model]+':generateContent'
  headers=({'Authorization':'Bearer '+key} if model=='jev' else {'x-goog-api-key':key})|{'Content-Type':'application/json'};op=urllib.request.build_opener(urllib.request.ProxyHandler({}),prior.smoke.NoRedirect());last_start=0;recent=[]
  for c,item in zip(cases,requests):
   for attempt in range(2):
    if model=='gemma':
     time.sleep(max(0,2.5-(time.monotonic()-last_start)))
     # Conservatively reserve 650 tokens/request, then debit reported actual usage.
     while True:
      recent=[(t,n) for t,n in recent if time.monotonic()-t<61]
      if sum(n for t,n in recent)+650<=15000:break
      time.sleep(min(10,max(.1,61-(time.monotonic()-recent[0][0]))))
    start=time.monotonic();last_start=start;attempts+=1;row=dict(id=c['id'],expected=c['expected'],started_utc=prior.now(),attempt=attempt+1)
    with (dest/'attempts.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    try:
     req=urllib.request.Request(url,data=json.dumps(item['payload']).encode(),headers=headers,method='POST')
     with op.open(req,timeout=120 if model=='gemma' else 60) as r:raw_bytes=r.read()
     row['latency_seconds']=time.monotonic()-start;(dest/(c['id']+'.json')).write_bytes(raw_bytes);row.update(normalize(model,json.loads(raw_bytes)))
    except Exception as e:
     row.update(error=type(e).__name__,http_status=getattr(e,'code',None),latency_seconds=time.monotonic()-start)
     if isinstance(e,urllib.error.HTTPError):(dest/(c['id']+f'-attempt-{attempt+1}-error.json')).write_bytes(e.read())
    if model=='gemma':recent.append((start,row.get('usage',{}).get('promptTokenCount',650)))
    if row.get('http_status')==429 and attempt==0 and len(retry_rows)<10:
     retry_rows.append(row);dump(dest/'retry-records.json',retry_rows);print(model,'quota wait',len(retry_rows),flush=True);time.sleep(65);continue
    break
   rows.append(row);dump(dest/'results.json',rows)
   if 'error' in row:print(model,'STOP',row,flush=True);break
   if len(rows)%50==0 or len(rows)==1:print(model,len(rows),'/',CAP,'invalid',sum(bool(r.get('invalid_output')) for r in rows),flush=True)
 summary=dict(attempted_examples=len(rows),http_attempts=attempts,valid=sum('error' not in r and not r.get('invalid_output') for r in rows),invalid_outputs=sum(bool(r.get('invalid_output')) for r in rows),complete=len(rows)==CAP and all('error' not in r for r in rows),wall_seconds=time.monotonic()-wall,finished_utc=prior.now())
 dump(dest/'summary.json',summary);print(model,json.dumps(summary),flush=True)
 if not summary['complete']:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['init','run']);p.add_argument('--model',choices=MODELS);p.add_argument('--bundle',type=Path);a=p.parse_args()
 if a.command=='init':init()
 else:run(a.bundle,a.model)
