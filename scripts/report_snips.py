"""Render the verified offline metrics into a concise Markdown report."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'docs/results/2026-09-21-snips-metrics.json';r=json.loads(p.read_text());models=r['models']
names={'jev':'Jev 1.13.0','gemini':'Gemini 3.5 Flash-Lite','gemma':'Gemma 4 31B','deberta':'DeBERTa-large (CPU)'}
percent=lambda x:f'{x*100:.2f}%'
complete=all(m in models and models[m]['complete'] for m in names)
lines=['# SNIPS seven-intent evaluation — '+('complete' if complete else 'in progress'),'',
'700 official evaluation examples, 100 per intent; 697 unique texts. Original SNIPS',
'validation/evaluation files at revision `b86ac7f1577868c42158d0dec77db50956046696`.',
'All four models receive the same frozen examples and category definitions. No',
'few-shot examples, model training, prompt tuning, or out-of-scope class.','',
'**Status:** '+('All four runs completed and saved evidence verified.' if complete else 'Partial snapshot. Do not compare models on unequal completed subsets.'),'',
'## Quality and output validity','',
'| Model | Completed | Strict correct | Accuracy | Macro F1 | Valid outputs | Unique-text accuracy |',
'|---|---:|---:|---:|---:|---:|---:|']
for m,x in models.items():
 q=x['metrics'];lines.append(f"| {names[m]} | {x['attempted']}/700 | {q['correct']}/{q['n']} | {percent(q['accuracy'])} | {q['macro_f1']:.4f} | {x['valid']}/{x['attempted']} | {percent(x['unique_text_accuracy'])} |")
lines+=['','Strict scoring requires a valid schema-conforming answer with the correct label.',
'Malformed model output counts as incorrect. No response is silently repaired or',
'dropped from the primary score. Unique-text accuracy gives each of the 697 texts',
'equal total weight, averaging predictions for duplicates.','',
'### Post-formatting score (owner requested)','',
'| Model | Correct after removing only Markdown fences | Accuracy | Macro F1 | Recovered outputs |',
'|---|---:|---:|---:|---:|']
for m in ['gemma']:
 if m in models:
  q=models[m]['secondary_fence_normalized'];lines.append(f"| {names[m]} | {q['correct']}/{q['n']} | {percent(q['accuracy'])} | {q['macro_f1']:.4f} | {models[m]['format_recovery']['recovered_outputs']} |")
lines+=['','This second score removes only enclosing/trailing Markdown fences, then repeats',
'strict JSON and label validation, rejecting duplicate keys. Valid answers are verified unchanged. It does not alter primary results or excuse',
'invalid structured output.','',
'All 130 malformed Gemma outputs were recovered; 128 recovered labels were correct',
'and two were incorrect. None of the 570 originally valid answers changed. This',
'establishes recovery for the observed fence errors, not arbitrary malformed output.',
'The formatter was introduced after observing these errors; see the',
'[post-formatting addendum](../snips-gemma-postformat-addendum.md).','',
'## Inference and cost','',
'| Model | HTTP p50 | HTTP p95 | HTTP requests/s | Wall time | API cost estimate | Cost / 1,000 |',
'|---|---:|---:|---:|---:|---:|---:|']
for m,x in models.items():
 wall=f"{x['wall_seconds']/60:.2f} min" if x['wall_seconds'] is not None else 'running'
 lines.append(f"| {names[m]} | {x['latency_p50_seconds']:.3f} s | {x['latency_p95_seconds']:.3f} s | {x['requests_per_second_http']:.3f} | {wall} | ${x['estimated_api_cost_usd']:.6f} | ${x['estimated_api_cost_per_1000']:.6f} |")
lines+=['','HTTP throughput is completed examples divided by accumulated request latency,',
'not a concurrency benchmark. Wall time includes client overhead and Gemma pacing.',
'Cloud timings include internet round trips; DeBERTa timings are Orthanc loopback',
'HTTP, excluding SSH/staging. Invalid-output responses remain in latency and cost.',
'No streaming TTFT or decode speed was measured.','',
'Costs use returned token usage and published rates, not verified invoices. Jev:',
'$0.042/M input tokens, free output. Gemini: $0.30/M uncached input, $0.03/M',
'cached input, $2.50/M output including thinking. Gemma API token price is free.',
'DeBERTa has no API fee; electricity, hardware and operations costs are unmeasured.',
'Gemma HIGH thinking was requested, Gemini MINIMAL. Missing reported thinking',
'usage is not proof of whether internal reasoning occurred.','',
'| Model | Input tokens | Cached input | Output tokens | Reported thinking tokens | HTTP attempts | Invalid outputs |',
'|---|---:|---:|---:|---:|---:|---:|']
for m,x in models.items():
 lines.append(f"| {names[m]} | {x['input_tokens'] if x['input_tokens'] is not None else 'N/A'} | {x.get('cached_input_tokens','N/A')} | {x['output_tokens'] if x['output_tokens'] is not None else 'N/A'} | {x.get('thinking_tokens_reported','N/A')} | {x['http_attempts']} | {x['invalid_outputs']} |")
lines+=['','## Per-intent recall','',
'| Intent | '+ ' | '.join(names.values())+' | Gemma after formatting |','|---|'+'---:|'*(len(names)+1)]
for label in next(iter(models.values()))['metrics']['per_class']:
 vals=[percent(models[m]['metrics']['per_class'][label]['recall']) if m in models else 'pending' for m in names]
 vals.append(percent(models['gemma']['secondary_fence_normalized']['per_class'][label]['recall']) if 'gemma' in models else 'pending')
 lines.append('| '+label+' | '+' | '.join(vals)+' |')
lines+=['','## Paired uncertainty','',
'Pairs below include only complete runs over the identical 700 examples. Confidence',
'intervals resample unique-text groups together (5,000 paired cluster bootstrap',
'replicates). McNemar p-values are exact and unadjusted for multiple comparisons.',
'With easy, familiar benchmark data, small gaps are not a general model ranking.','',
'| Pair (first minus second) | Accuracy difference | Paired 95% interval | Only first correct | Only second correct | Exact p |',
'|---|---:|---:|---:|---:|---:|']
for pair,x in r['pairwise'].items():
 lo,hi=x['paired_text_cluster_bootstrap95'];a,b=pair.split('_minus_');pair_names=names|{'gemma_postformatted':'Gemma 4 31B after formatting'};lines.append(f"| {pair_names[a]} − {pair_names[b]} | {x['accuracy_difference']*100:+.2f} pp | [{lo*100:+.2f}, {hi*100:+.2f}] pp | {x['a_only_correct']} | {x['b_only_correct']} | {x['mcnemar_exact_two_sided_unadjusted']:.4g} |")
lines+=['','## Probability diagnostics','',
'| Model | Multiclass Brier | Log loss | Raw probability mass range | Mass differs from 1 by >1e-6 |','|---|---:|---:|---:|---:|']
for m in ['jev','deberta']:
 if m in models:
  x=models[m];mass=x['probability_mass'];lines.append(f"| {names[m]} | {x['multiclass_brier']:.4f} | {x['log_loss_epsilon_1e15']:.4f} | {mass['min']:.6f}–{mass['max']:.6f} | {mass['nonunit_count']} |")
lines+=['','For these metrics only, saved probabilities are divided by their total mass;',
'gold probability is clipped to 1e-15 for log loss. Raw probabilities are retained.',
'Jev and DeBERTa scores have different definitions. Calibration bins in the JSON',
'are exploratory; no review threshold was selected or evaluated.','',
'## Reproduction, evidence and limitations','',
'- Protocol: [SNIPS v1](../snips-v1-protocol.md).',
'- [Machine-readable metrics](2026-09-21-snips-metrics.json) and [all predictions](2026-09-21-snips-predictions.csv).',
'- Raw responses and immutable manifests: `local/runs/snips-v1-2026-09-21T054639.162986Z/`.',
'- Recompute offline: `.venv/bin/python scripts/analyze_snips.py local/runs/snips-v1-2026-09-21T054639.162986Z`, then `.venv/bin/python scripts/report_snips.py`.',
'- Analysis verified original-file hashes, exact text reconstruction, frozen source/request hashes, IDs/order, raw output normalization, models, usage, and latency values.',
'- DeBERTa retains its model revision, FP32, four CPU threads, batch size one and original NLI template. No hardware or model changes for this experiment.',
'- The seven assistant intents are fairly distinct; strong accuracy may reflect an easy benchmark. This does not measure out-of-scope rejection, production traffic or broad reasoning.',
'- The original June 2017 benchmark evaluated slot filling separately within each intent. This experiment combines its evaluation files for seven-way intent classification; original published slot F1 scores are not comparable.',
'- Some source labels imply missing context: “I want to see Outcast.” is labelled SearchScreeningEvent despite not explicitly mentioning cinema times. Another utterance, “Where is Belgium located”, is labelled GetWeather although it asks about geography. Scores measure agreement with retained source labels; no post-result relabelling or exclusions were made.',
'- Dataset/model training exposure is unknown. SNIPS was absent from the inspected DeBERTa task-training inventory; this is not a contamination guarantee.',
'- Gemma remains on AI Studio: the current [Groq model list](https://console.groq.com/docs/models) did not list Gemma 4 31B. No Groq credential was accessed.',
'- Dataset licence CC0 1.0; attribution: Coucke et al. (2018), *Snips Voice Platform*. [Official dataset](https://github.com/sonos/nlu-benchmark/tree/master/2017-06-custom-intent-engines).',
'- Price sources: [TypeSafe](https://docs.typesafe.ai/models), [Gemini API](https://ai.google.dev/gemini-api/docs/pricing).','']
(ROOT/'docs/results/2026-09-21-snips.md').write_text('\n'.join(lines))
print('Report written; complete:',complete)
