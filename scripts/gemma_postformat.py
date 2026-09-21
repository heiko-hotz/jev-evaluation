"""Conservative formatting recovery: Markdown fences only; never infer a label."""
import json

def _unique_object(pairs):
 result={}
 for key,value in pairs:
  if key in result:raise ValueError('Duplicate JSON key')
  result[key]=value
 return result

def recover_route(raw,labels):
 candidates=raw.get('candidates',[])
 if len(candidates)!=1 or candidates[0].get('finishReason')!='STOP':return None
 parts=candidates[0].get('content',{}).get('parts',[])
 if not parts or any(not isinstance(p.get('text'),str) or p.get('thought') for p in parts):return None
 text=''.join(p['text'] for p in parts).strip()
 if text.startswith('```json'):text=text[7:].strip()
 elif text.startswith('```'):text=text[3:].strip()
 if text.endswith('```'):text=text[:-3].strip()
 try:
  obj=json.loads(text,object_pairs_hook=_unique_object)
  if not isinstance(obj,dict) or set(obj)!={'route'} or not isinstance(obj['route'],str) or obj['route'] not in labels:return None
  return obj['route']
 except (ValueError,TypeError):return None
