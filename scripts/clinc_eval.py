"""Frozen CLINC150 runner: sequential, durable records, no duplicate dispatch."""
import argparse,hashlib,json,math,os,platform,subprocess,time,urllib.request,urllib.error
from datetime import datetime,timezone
from pathlib import Path
import smoke,gemini_challenge as gemini,deberta,deberta_remote
ROOT=smoke.ROOT
DATA=ROOT/'data/clinc150-v1.json'; LABELS=ROOT/'data/clinc150-labels-v1.json'
PROTOCOL=ROOT/'docs/clinc150-v1-protocol.md'
MODELS={'jev':smoke.MODEL,'gemini':gemini.MODEL,'gemma':'gemma-4-31b-it','deberta':deberta_remote.MODEL}
INSTRUCTION='Which intent best matches this user request? Select the single best category. Select oos only if none of the listed intents fits.'
SERVER_HASH='8289db577c690ebd225f64b966af67df84c04c6c2f0b50951dc310e5881fec9d'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def payload(model,case,criteria):
 if model=='jev':return {'model':MODELS[model],'state':{'request':case['text']},'questions':{'route':{'type':'choice','instructions':INSTRUCTION,'criteria':criteria}}}
 if model=='deberta':return {'inputs':case['text'],'parameters':{'candidate_labels':[f'{k}: {v}' for k,v in criteria.items()],'multi_label':False,'hypothesis_template':deberta.TEMPLATE}}
 body=gemini.payload(case)
 body['systemInstruction']['parts'][0]['text']=INSTRUCTION+'\nCategories:\n'+json.dumps(criteria)
 body['generationConfig']['responseJsonSchema']['properties']['route']['enum']=list(criteria)
 if model=='gemma':body['generationConfig'].update(maxOutputTokens=8192,thinkingConfig={'thinkingLevel':'HIGH','includeThoughts':False})
 return body

def normalize(model,raw,criteria):
 if model=='jev':
  assert raw['model']==MODELS[model];a=raw['answers']['route'];p=a['probabilities'];choice=a['choice']
  assert a['type']=='choice' and set(p)==set(criteria) and choice in criteria
  assert all(isinstance(x,(int,float)) and math.isfinite(x) and 0<=x<=1 for x in p.values()) and math.isclose(sum(p.values()),1,abs_tol=.001)
  assert p[choice]>=max(p.values()) and 0<=a['confidence']<=1
  assert all(type(raw['usage'][k]) is int and raw['usage'][k]>=0 for k in ('input_tokens','output_tokens'))
  return dict(choice=choice,probabilities=p,confidence=a['confidence'],model=raw['model'],usage=raw['usage'])
 if model=='deberta':
  labels=[f'{k}: {v}' for k,v in criteria.items()];deberta_remote.validate(raw,labels);inverse=dict(zip(labels,criteria))
  return dict(choice=inverse[raw['labels'][0]],probabilities={inverse[l]:p for l,p in zip(raw['labels'],raw['scores'])},model=raw['model'],revision=raw['revision'])
 version=raw['modelVersion'];assert version==MODELS[model] or version.startswith(MODELS[model]+'-')
 cs=raw['candidates'];assert len(cs)==1 and cs[0]['finishReason']=='STOP'
 parts=cs[0]['content']['parts'];assert parts and all(isinstance(p.get('text'),str) and not p.get('thought') for p in parts)
 obj=json.loads(''.join(p['text'] for p in parts));assert isinstance(obj,dict) and set(obj)=={'route'} and obj['route'] in criteria
 usage=raw['usageMetadata'];assert all(type(usage[k]) is int and usage[k]>=0 for k in ('promptTokenCount','candidatesTokenCount','totalTokenCount'))
 assert usage.get('thoughtsTokenCount',0)>=0
 return dict(choice=obj['route'],model=version,usage=usage)

def verify(bundle):
 meta=json.loads((bundle/'manifest.json').read_text())
 for name,sha in meta['file_sha256'].items():assert digest(ROOT/name)==sha,name
 cases=json.loads(DATA.read_text());criteria=json.loads(LABELS.read_text());assert len(cases)==1834 and len(criteria)==151
 return cases,criteria,meta

def init():
 os.umask(0o077);cases=json.loads(DATA.read_text());criteria=json.loads(LABELS.read_text())
 folder=ROOT/'local/runs'/datetime.now(timezone.utc).strftime('clinc150-v1-%Y-%m-%dT%H%M%S.%fZ')
 ledger=ROOT/'local/clinc150-v1-dispatch.json'
 with ledger.open('x') as f:json.dump({'bundle':str(folder)},f)
 folder.mkdir(parents=True)
 for m in MODELS:
  (folder/m).mkdir();dump(folder/m/'requests.json',[dict(id=c['id'],payload=payload(m,c,criteria)) for c in cases])
 files=['data/clinc150-v1.json','data/clinc150-labels-v1.json','data/clinc150-provenance-v1.json','docs/clinc150-v1-protocol.md','scripts/clinc_eval.py','scripts/clinc_deberta_remote.py','scripts/clinc_prepare.py','scripts/gemini_challenge.py','scripts/smoke.py','scripts/deberta.py','scripts/deberta_remote.py']
 dump(folder/'manifest.json',dict(timestamp=now(),models=MODELS,requests_per_model=len(cases),spending_ceiling_usd=None,retries=0,concurrency_per_model=1,file_sha256={p:digest(ROOT/p) for p in files},request_sha256={m:digest(folder/m/'requests.json') for m in MODELS},python=platform.python_version(),platform=platform.platform(),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty=True))
 print(folder)

def snapshot(folder,stage):
 from deberta_challenge import HASHES
 hashes=HASHES|{'/opt/orthanc-deberta/server.py':SERVER_HASH}
 def call(wrapper,*args):return subprocess.check_output([str(deberta.WRAPPERS/wrapper),*args],text=True)
 observed={line.split()[1]:line.split()[0] for line in call('sudo.sh','/usr/bin/sha256sum',*hashes).splitlines()}
 state=call('ssh.sh','systemctl','show','orthanc-deberta.service','--property=ActiveState,SubState,NRestarts,MemoryCurrent')
 runtime=json.loads(call('sudo.sh','/opt/orthanc-deberta/.venv/bin/python','-c','import json,platform,importlib.metadata as m;print(json.dumps(dict(python=platform.python_version(),torch=m.version("torch"),transformers=m.version("transformers"))))'))
 dump(folder/(stage+'-service.json'),dict(hashes=observed,service=state,runtime=runtime,loadavg=call('ssh.sh','cat','/proc/loadavg')))
 assert observed==hashes and 'ActiveState=active' in state and 'SubState=running' in state
 assert runtime==dict(python='3.12.14',torch='2.14.0+cpu',transformers='4.57.6')

def run(bundle,model):
 cases,criteria,meta=verify(bundle);folder=bundle/model
 assert digest(folder/'requests.json')==meta['request_sha256'][model]
 requests=json.loads((folder/'requests.json').read_text());assert requests==[dict(id=c['id'],payload=payload(model,c,criteria)) for c in cases]
 with (folder/'started.json').open('x') as f:json.dump({'utc':now()},f)
 wall=time.monotonic();rows=[]
 if model=='deberta':
  snapshot(folder,'before');remote_runner='deberta-'+bundle.name+'-runner.py'
  subprocess.run([str(deberta.WRAPPERS/'copy-to-tmp.sh'),str(ROOT/'scripts/clinc_deberta_remote.py'),remote_runner],check=True)
  remote_input='deberta-'+bundle.name+'-requests.json'
  subprocess.run([str(deberta.WRAPPERS/'copy-to-tmp.sh'),str(folder/'requests.json'),remote_input],check=True)
  proc=subprocess.Popen([str(deberta.WRAPPERS/'ssh.sh'),'/usr/bin/python3','-u','/tmp/'+remote_runner,'/tmp/'+remote_input],stdout=subprocess.PIPE,text=True)
  with (folder/'remote-responses.jsonl').open('x') as log:
   for line in proc.stdout:
    log.write(line);log.flush();row=json.loads(line);c=cases[len(rows)];assert row['id']==c['id']
    if 'response' in row:
     dump(folder/(c['id']+'.json'),row['response']);row.update(normalize(model,row.pop('response'),criteria))
    row['expected']=c['expected'];rows.append(row);dump(folder/'results.json',rows)
    if len(rows)%20==0 or len(rows)==1:print(model,len(rows),'/',len(cases),'latest_seconds',row['latency_seconds'],flush=True)
  rc=proc.wait();snapshot(folder,'after')
  subprocess.run([str(deberta.WRAPPERS/'ssh.sh'),'rm','--','/tmp/'+remote_runner,'/tmp/'+remote_input],check=True)
 else:
  key=smoke.key_from_environment() if model=='jev' else gemini.read_key(ROOT.parent/'ace-economic-sim/.env.local')
  endpoint='https://api.typesafe.ai/v1/systemone' if model=='jev' else 'https://generativelanguage.googleapis.com/v1beta/models/'+MODELS[model]+':generateContent'
  headers=({'Authorization':'Bearer '+key} if model=='jev' else {'x-goog-api-key':key})|{'Content-Type':'application/json'}
  op=urllib.request.build_opener(urllib.request.ProxyHandler({}),smoke.NoRedirect())
  for c,item in zip(cases,requests):
   row=dict(id=c['id'],expected=c['expected'],started_utc=now());start=time.monotonic()
   # Durable attempt record precedes dispatch, including uncertain transport outcomes.
   with (folder/'attempts.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
   try:
    req=urllib.request.Request(endpoint,data=json.dumps(item['payload']).encode(),headers=headers,method='POST')
    with op.open(req,timeout=120 if model=='gemma' else 60) as r:b=r.read()
    row['latency_seconds']=time.monotonic()-start;(folder/(c['id']+'.json')).write_bytes(b)
    row.update(normalize(model,json.loads(b),criteria))
   except Exception as e:
    row.update(error=type(e).__name__,http_status=getattr(e,'code',None),latency_seconds=time.monotonic()-start)
    if isinstance(e,urllib.error.HTTPError):(folder/(c['id']+'-error.json')).write_bytes(e.read())
   rows.append(row);dump(folder/'results.json',rows)
   if 'error' in row:print(model,'STOP',row,flush=True);break
   if len(rows)%50==0 or len(rows)==1:print(model,len(rows),'/',len(cases),flush=True)
 good=[r for r in rows if 'error' not in r];summary=dict(attempted=len(rows),valid=len(good),complete=len(good)==len(cases),wall_seconds=time.monotonic()-wall,finished_utc=now())
 dump(folder/'summary.json',summary);print(model,json.dumps(summary),flush=True)
 if not summary['complete']:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['init','run']);p.add_argument('--model',choices=MODELS);p.add_argument('--bundle',type=Path);a=p.parse_args()
 if a.command=='init':init()
 else:run(a.bundle,a.model)
