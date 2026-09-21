"""Portable SNIPS dispatch; frozen payloads/validators, no workspace dependencies.

No network access without run --live. Never invoke the historical runner for a
new run. This adapter deliberately leaves the original evidence source unchanged.
"""
import argparse
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
import snips_eval as frozen
from snips_eval import ROOT, MODELS, CAP, payload, normalize, verify, dump, prior

def run(folder,model):
 cases,meta=verify(folder);dest=folder/model;assert prior.digest(dest/'requests.json')==meta['request_sha256'][model]
 requests=json.loads((dest/'requests.json').read_text());assert requests==[dict(id=c['id'],payload=payload(model,c)) for c in cases]
 with (dest/'started.json').open('x') as f:json.dump({'utc':prior.now()},f)
 rows=[];wall=time.monotonic();attempts=0;retry_rows=[]
 key = os.environ['TYPESAFE_API_KEY'] if model=='jev' else (os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')) if model!='deberta' else ''
 url='http://127.0.0.1:8091/classify' if model=='deberta' else 'https://api.typesafe.ai/v1/systemone' if model=='jev' else 'https://generativelanguage.googleapis.com/v1beta/models/'+MODELS[model]+':generateContent'
 headers=({} if model=='deberta' else {'Authorization':'Bearer '+key} if model=='jev' else {'x-goog-api-key':key})|{'Content-Type':'application/json'};op=urllib.request.build_opener(urllib.request.ProxyHandler({}),prior.smoke.NoRedirect());last_start=0;recent=[]
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
   if model!='deberta' and row.get('http_status')==429 and attempt==0 and len(retry_rows)<10:
    retry_rows.append(row);dump(dest/'retry-records.json',retry_rows);print(model,'quota wait',len(retry_rows),flush=True);time.sleep(65);continue
   break
  rows.append(row);dump(dest/'results.json',rows)
  if 'error' in row:print(model,'STOP',row,flush=True);break
  if len(rows)%50==0 or len(rows)==1:print(model,len(rows),'/',CAP,'invalid',sum(bool(r.get('invalid_output')) for r in rows),flush=True)
 summary=dict(attempted_examples=len(rows),http_attempts=attempts,valid=sum('error' not in r and not r.get('invalid_output') for r in rows),invalid_outputs=sum(bool(r.get('invalid_output')) for r in rows),complete=len(rows)==CAP and all('error' not in r for r in rows),wall_seconds=time.monotonic()-wall,finished_utc=prior.now())
 dump(dest/'summary.json',summary);print(model,json.dumps(summary),flush=True)
 if not summary['complete']:raise SystemExit(1)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'init', 'run'])
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    if args.command == 'check':
        cases = json.loads(frozen.DATA.read_text())
        assert len(cases) == CAP
        for model in MODELS:
            assert len([payload(model, case) for case in cases]) == CAP
        print('Ready: 700 examples/model. No network access or inference.')
    elif args.command == 'init':
        (ROOT / 'local').mkdir(exist_ok=True)
        frozen.SOURCES = frozen.SOURCES + [
            'scripts/replicate_snips.py', 'scripts/analyze_routing.py',
            'baseline/deberta/server.py', 'baseline/deberta/uv.lock',
            'baseline/deberta/model-manifest.json',
        ]
        frozen.init()
    else:
        if not args.model or not args.bundle:
            parser.error('run requires --model and --bundle')
        if not args.live:
            verify(args.bundle)
            print('Dry run. Add --live to dispatch up to 700 examples (710 cloud attempts).')
            return
        if args.model == 'jev' and not os.environ.get('TYPESAFE_API_KEY'):
            parser.error('Set TYPESAFE_API_KEY in your environment')
        if args.model in ('gemini', 'gemma') and not (os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')):
            parser.error('Set GEMINI_API_KEY in your environment')
        run(args.bundle, args.model)

if __name__ == '__main__':
    main()
