"""Verify saved evidence and calculate SNIPS results without provider access."""
import argparse,csv,json,math,random,statistics
from collections import Counter,defaultdict
from pathlib import Path
import snips_eval as run
ROOT=run.ROOT

def classification(gold,pred,labels):
 per={};matrix={label:dict.fromkeys(list(labels)+['INVALID'],0) for label in labels}
 for g,p in zip(gold,pred):matrix[g][p if p in labels else 'INVALID']+=1
 for label in labels:
  tp=matrix[label][label];fp=sum(matrix[g][label] for g in labels if g!=label);fn=sum(matrix[label].values())-tp
  per[label]=dict(n=tp+fn,correct=tp,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
 correct=sum(g==p for g,p in zip(gold,pred))
 return dict(n=len(gold),correct=correct,accuracy=correct/len(gold),macro_f1=statistics.mean(x['f1'] for x in per.values()),per_class=per,confusion=matrix)

def wilson(k,n):
 z=1.95996398454;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [mid-half,mid+half]

def presentation_parse(raw):
 from gemma_postformat import recover_route
 return recover_route(raw,run.CRITERIA)


def read_model(bundle,model,cases,meta):
 folder=bundle/model
 if not (folder/'results.json').exists():return [],{},[]
 assert run.prior.digest(folder/'requests.json')==meta['request_sha256'][model]
 requests=json.loads((folder/'requests.json').read_text());assert requests==[dict(id=c['id'],payload=run.payload(model,c)) for c in cases]
 rows=json.loads((folder/'results.json').read_text());assert len(rows)<=700
 raws=[]
 for case,row in zip(cases,rows):
  assert row['id']==case['id'] and row['expected']==case['expected'];assert math.isfinite(row['latency_seconds']) and row['latency_seconds']>0
  if 'error' in row:raws.append(None);continue
  raw=json.loads((folder/(case['id']+'.json')).read_text());normalized=run.normalize(model,raw)
  assert all(row.get(k)==v for k,v in normalized.items()),(model,case['id']);raws.append(raw)
 summary=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else {}
 if summary:
  assert summary['attempted_examples']==len(rows) and summary['valid']==sum('error' not in r and not r.get('invalid_output') for r in rows)
  assert summary['complete']==(len(rows)==700 and all('error' not in r for r in rows))
 return rows,summary,raws

def analyze(bundle):
 cases,meta=run.verify(bundle);labels=list(run.CRITERIA);by_id={c['id']:c for c in cases};all_rows={};formatted_predictions={};report=dict(dataset_n=700,unique_texts=len({c['text'] for c in cases}),models={},pairwise={},billing_verified=False)
 # Verify the derivative against preserved upstream originals, including their hashes.
 provenance=json.loads((ROOT/'data/snips-provenance-v1.json').read_text())
 source={}
 for spec in provenance['sources']:
  p=ROOT/'local/sources/snips'/spec['url'].rsplit('/',1)[1]
  if not p.exists():p=ROOT/'data/snips-source'/p.name
  assert run.prior.digest(p)==spec['sha256'];source[p.name]=json.loads(p.read_text(encoding='utf-8-sig'))
 for c in cases:assert c['text']==''.join(p['text'] for p in source[c['source_file']][c['expected']][c['source_index']]['data'])
 for model in run.MODELS:
  rows,summary,raws=read_model(bundle,model,cases,meta)
  if not rows:continue
  all_rows[model]=rows;good=[r for r in rows if 'error' not in r];n=len(rows)
  metrics=classification([r['expected'] for r in rows],[r.get('choice') for r in rows],labels)
  lat=sorted(r['latency_seconds'] for r in rows);valid=sum('error' not in r and not r.get('invalid_output') for r in rows)
  out=dict(complete=bool(summary.get('complete')),attempted=n,valid=valid,invalid_outputs=sum(bool(r.get('invalid_output')) for r in rows),transport_or_contract_errors=sum('error' in r for r in rows),valid_output_rate=valid/n,metrics=metrics,accuracy_wilson95_descriptive=wilson(metrics['correct'],n),latency_p50_seconds=statistics.median(lat),latency_p95_seconds=lat[math.ceil(.95*n)-1],latency_mean_seconds=statistics.mean(lat),latency_min_seconds=min(lat),latency_max_seconds=max(lat),total_request_seconds=sum(lat),requests_per_second_http=n/sum(lat),wall_seconds=summary.get('wall_seconds'),model_versions=sorted({r['model'] for r in good}),classes={})
  if summary.get('wall_seconds'):out['requests_per_second_wall']=n/summary['wall_seconds']
  retries_path=bundle/model/'retry-records.json';retries=json.loads(retries_path.read_text()) if retries_path.exists() else [];out['http_429_retries']=len(retries);out['http_attempts']=summary.get('http_attempts',n+len(retries));out['retry_http_seconds']=sum(r['latency_seconds'] for r in retries);out['error_billing_unknown']=bool(retries or out['transport_or_contract_errors'])
  by_text=defaultdict(list)
  for r in rows:by_text[by_id[r['id']]['text']].append(r)
  # Give each unique text total weight one, averaging repeated stochastic predictions.
  out['unique_text_accuracy']=statistics.mean(statistics.mean(r.get('choice')==r['expected'] for r in group) for group in by_text.values())
  out['unique_text_n']=len(by_text)
  out['duplicate_text_conflicting_gold']=sum(len({r['expected'] for r in group})>1 for group in by_text.values())
  if model in ['gemini','gemma']:
   input_tokens=sum(r['usage']['promptTokenCount'] for r in good);output_tokens=sum(r['usage']['candidatesTokenCount'] for r in good);thinking=sum(r['usage'].get('thoughtsTokenCount',0) for r in good);cached=sum(r['usage'].get('cachedContentTokenCount',0) for r in good)
   cost=((input_tokens-cached)*.30+cached*.03+(output_tokens+thinking)*2.50)/1e6 if model=='gemini' else 0
   tolerant=[presentation_parse(raw) if raw is not None else None for raw in raws];out['secondary_fence_normalized']=classification([r['expected'] for r in rows],tolerant,labels)
   assert all(r.get('choice')==p for r,p in zip(rows,tolerant) if r.get('choice') is not None), 'Post-formatting changed a valid answer'
   formatted_predictions[model]=dict(zip([r['id'] for r in rows],tolerant))
   out['format_recovery']=dict(recovered_outputs=sum(r.get('invalid_output',False) and p is not None for r,p in zip(rows,tolerant)),unrecoverable_outputs=sum(p is None for p in tolerant),valid_answers_changed=0,procedure_sha256=run.prior.digest(ROOT/'scripts/gemma_postformat.py'))
   out.update(thinking_usage_present_count=sum('thoughtsTokenCount' in r['usage'] for r in good),cached_input_tokens=cached,thinking_tokens_reported=thinking,undiscounted_standard_cost_usd=(input_tokens*.30+(output_tokens+thinking)*2.50)/1e6 if model=='gemini' else 0)
  elif model=='jev':
   input_tokens=sum(r['usage']['input_tokens'] for r in good);output_tokens=sum(r['usage']['output_tokens'] for r in good);cost=input_tokens*.042/1e6
  else:input_tokens=output_tokens=None;cost=0
  out.update(input_tokens=input_tokens,output_tokens=output_tokens,estimated_api_cost_usd=cost,estimated_api_cost_per_1000=cost/n*1000,estimated_api_cost_per_correct=cost/metrics['correct'] if metrics['correct'] else None)
  if model in ['jev','deberta']:
   masses=[];brier=[];loss=[];calibration=[]
   for r in good:
    mass=sum(r['probabilities'].values());masses.append(mass);p={k:v/mass for k,v in r['probabilities'].items()};brier.append(sum((v-(k==r['expected']))**2 for k,v in p.items()));loss.append(-math.log(max(p[r['expected']],1e-15)));calibration.append((max(p.values()),r['choice']==r['expected']))
   bins=[]
   for i in range(10):
    vals=[(p,c) for p,c in calibration if min(int(p*10),9)==i]
    if vals:bins.append(dict(lower=i/10,upper=(i+1)/10,n=len(vals),mean_top_probability=statistics.mean(p for p,c in vals),accuracy=statistics.mean(c for p,c in vals)))
   out.update(multiclass_brier=statistics.mean(brier),log_loss_epsilon_1e15=statistics.mean(loss),probability_mass=dict(min=min(masses),max=max(masses),nonunit_count=sum(not math.isclose(m,1,rel_tol=0,abs_tol=1e-6) for m in masses),nonunit_absolute_tolerance=1e-6),calibration_bins_exploratory=bins,ece_10_bins_exploratory=sum(b['n']*abs(b['mean_top_probability']-b['accuracy']) for b in bins)/len(good))
  out['errors']=[dict(id=r['id'],expected=r['expected'],predicted=r.get('choice'),invalid_output=bool(r.get('invalid_output'))) for r in rows if r.get('choice')!=r['expected']]
  report['models'][model]=out
 complete=[m for m in all_rows if report['models'][m]['complete']];comparison_rows=dict(all_rows)
 if 'gemma' in complete:
  comparison_rows['gemma_postformatted']=[r|{'choice':formatted_predictions['gemma'][r['id']]} for r in all_rows['gemma']]
  complete.append('gemma_postformatted')
 rng=random.Random(20260921)
 for i,a in enumerate(complete):
  for b in complete[i+1:]:
   if {a,b}=={'gemma','gemma_postformatted'}:continue
   ar=comparison_rows[a];br=comparison_rows[b];assert [r['id'] for r in ar]==[r['id'] for r in br];diffs=[int(x.get('choice')==x['expected'])-int(y.get('choice')==y['expected']) for x,y in zip(ar,br)]
   groups=defaultdict(list)
   for c,d in zip(cases,diffs):groups[c['text']].append(d)
   values=list(groups.values());boot=[]
   for _ in range(5000):
    selected=rng.choices(values,k=len(values));boot.append(sum(sum(x) for x in selected)/sum(len(x) for x in selected))
   boot.sort();wins=sum(d==1 for d in diffs);losses=sum(d==-1 for d in diffs);discordant=wins+losses
   exact_p=min(1,2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))/2**discordant) if discordant else 1
   report['pairwise'][a+'_minus_'+b]=dict(accuracy_difference=statistics.mean(diffs),a_only_correct=wins,b_only_correct=losses,paired_text_cluster_bootstrap95=[boot[124],boot[4874]],mcnemar_exact_two_sided_unadjusted=exact_p)
 report['analysis_source_sha256']=run.prior.digest(Path(__file__).resolve());report['analysis_utc']=run.prior.now();dest=ROOT/'docs/results/2026-09-21-snips-metrics.json';run.dump(dest,report)
 with (ROOT/'docs/results/2026-09-21-snips-predictions.csv').open('w',newline='') as f:
  fields=['id','text','expected']+list(run.MODELS)+['gemma_postformatted','gemma_format_recovered'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();maps={m:{r['id']:r.get('choice') or 'INVALID' for r in rows} for m,rows in all_rows.items()}
  for c in cases:
   row={k:c[k] for k in ['id','text','expected']}|{m:maps.get(m,{}).get(c['id'],'') for m in run.MODELS}
   formatted=formatted_predictions.get('gemma',{}).get(c['id']);row['gemma_postformatted']=formatted or ('INVALID' if c['id'] in formatted_predictions.get('gemma',{}) else '')
   row['gemma_format_recovered']=row['gemma']=='INVALID' and formatted is not None
   writer.writerow(row)
 print(json.dumps({m:dict(n=x['attempted'],complete=x['complete'],accuracy=x['metrics']['accuracy'],macro_f1=x['metrics']['macro_f1'],invalid=x['invalid_outputs'],p50=x['latency_p50_seconds'],cost=x['estimated_api_cost_usd']) for m,x in report['models'].items()},indent=2))
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);args=p.parse_args();analyze(args.bundle)
