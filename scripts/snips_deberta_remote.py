"""Sequential loopback requests on the existing Orthanc DeBERTa service."""
import json,sys,time,urllib.request
from pathlib import Path
requests=json.loads(Path(sys.argv[1]).read_text());assert len(requests)==700
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for i,item in enumerate(requests):
 assert len(item['payload']['parameters']['candidate_labels'])==7
 row=dict(id=item['id'],index=i);start=time.monotonic()
 try:
  req=urllib.request.Request('http://127.0.0.1:8091/classify',data=json.dumps(item['payload']).encode(),headers={'Content-Type':'application/json'},method='POST')
  with op.open(req,timeout=60) as response:raw=json.load(response)
  row.update(response=raw,latency_seconds=time.monotonic()-start)
 except Exception as e:row.update(error=type(e).__name__,http_status=getattr(e,'code',None),latency_seconds=time.monotonic()-start)
 print(json.dumps(row),flush=True)
 if 'error' in row:raise SystemExit(1)
