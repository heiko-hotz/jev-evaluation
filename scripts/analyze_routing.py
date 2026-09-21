"""Offline analysis: never calls Jev. Baseline frozen before predictions."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

try:
    from smoke import ROOT, CRITERIA, RATE, validate
except ModuleNotFoundError:
    from scripts.smoke import ROOT, CRITERIA, RATE, validate

LABELS = list(CRITERIA)
KEYWORDS = {
    'account_access': r'password|sign[ -]?in|log[ -]?in|authenticat|locked|lockout|two.step|verification|backup code|single sign.on|saml|magic link',
    'billing': r'bill|charg|payment|paid|pay\b|invoice|receipt|refund|debit|credit|tax|balance|bank transfer|card|instalment',
    'technical': r'crash|bug|error|freez|broken|blank|api|upload|export|import|webhook|sync|render|notification|dashboard|disappear|overlap|malformed|save|undo|screen reader',
}

def baseline(text):
    for label, pattern in KEYWORDS.items():
        if re.search(pattern, text, flags=re.I):
            return label
    return 'other'


def metrics(expected, predicted):
    if not expected or len(expected) != len(predicted):
        raise ValueError('Expected nonempty paired predictions')
    matrix = {a: {b: 0 for b in LABELS} for a in LABELS}
    for a, b in zip(expected, predicted):
        matrix[a][b] += 1
    scores = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted_n = sum(matrix[a][label] for a in LABELS)
        scores[label] = {'support': support, 'precision': tp / predicted_n if predicted_n else 0,
                         'recall': tp / support if support else 0,
                         'f1': 2 * tp / (support + predicted_n) if support + predicted_n else 0}
    return {'n': len(expected), 'accuracy': sum(a == b for a,b in zip(expected,predicted)) / len(expected),
            'macro_f1': statistics.mean(v['f1'] for v in scores.values()), 'per_class': scores, 'confusion': matrix}



def verify_saved_row(row, case, raw):
    answer = validate(raw)
    if row['expected'] != case['expected']:
        raise ValueError('Saved label differs from frozen dataset')
    if any(row[field] != answer[field] for field in ('choice', 'probabilities', 'confidence', 'type')):
        raise ValueError('Saved prediction differs from raw response')
    if row['usage'] != raw['usage'] or row['model'] != raw['model']:
        raise ValueError('Saved metadata differs from raw response')


def analyze(folder):
    source = (ROOT / 'data/routing.json').read_bytes()
    cases = json.loads(source)
    manifest = json.loads((folder / 'manifest.json').read_text())
    rows = json.loads((folder / 'results.json').read_text())
    assert hashlib.sha256(source).hexdigest() == manifest['dataset_sha256']
    assert len(cases) == 100 and Counter(c['split'] for c in cases) == {'dev':20,'test':80}
    assert len({c['text'].lower() for c in cases}) == 100
    assert {c['family'] for c in cases if c['split']=='dev'}.isdisjoint(c['family'] for c in cases if c['split']=='test')
    assert len(rows) == 100 and {r['id'] for r in rows} == {c['id'] for c in cases}
    canonical = {c['id']: c for c in cases}
    for row in rows:
        assert 'error' not in row
        verify_saved_row(row, canonical[row['id']], json.loads((folder / (row['id'] + '.json')).read_text()))
    by_id = {r['id']:r for r in rows}
    results = {}
    for split in ('dev','test','all'):
        subset = [c for c in cases if split=='all' or c['split']==split]
        actual = [c['expected'] for c in subset]
        predictions = [by_id[c['id']]['choice'] for c in subset]
        results[split] = {'jev': metrics(actual,predictions), 'keywords': metrics(actual,[baseline(c['text']) for c in subset])}
        results[split]['brier_multiclass'] = statistics.mean(sum((by_id[c['id']]['probabilities'][l] - int(c['expected']==l))**2 for l in LABELS) for c in subset)
        results[split]['log_loss_clipped_1e-15'] = statistics.mean(-math.log(max(1e-15,by_id[c['id']]['probabilities'][c['expected']])) for c in subset)
    # Predeclared selection: lowest threshold with zero dev errors and >=50% dev coverage.
    dev = [by_id[c['id']] for c in cases if c['split']=='dev']
    selected = None
    for threshold in (0,0.25,0.5,0.75,0.9,0.95,0.99):
        accepted = [r for r in dev if r['confidence'] >= threshold]
        if len(accepted)>=10 and all(r['choice']==r['expected'] for r in accepted):
            selected=threshold
            break
    test = [by_id[c['id']] for c in cases if c['split']=='test']
    curve=[]
    for threshold in (0,0.25,0.5,0.75,0.9,0.95,0.99):
        accepted=[r for r in test if r['confidence']>=threshold]
        curve.append({'threshold':threshold,'accepted':len(accepted),'coverage':len(accepted)/len(test),
                      'error_rate':sum(r['choice']!=r['expected'] for r in accepted)/len(accepted) if accepted else None})
    latencies=sorted(r['latency_seconds'] for r in rows)
    results.update(selected_dev_threshold=selected, test_threshold_curve=curve,
        latency_p50_seconds=statistics.median(latencies), latency_p95_seconds=latencies[math.ceil(.95*len(latencies))-1],
        throughput_requests_per_network_second=len(rows)/sum(latencies),
        input_tokens=sum(r['usage']['input_tokens'] for r in rows), output_tokens=sum(r['usage']['output_tokens'] for r in rows),
        failures=[{'id':c['id'],'split':c['split'],'text':c['text'],'expected':c['expected'],'observed':by_id[c['id']]['choice'],
                   'confidence':by_id[c['id']]['confidence']} for c in cases if c['expected']!=by_id[c['id']]['choice']])
    results['estimated_cost_usd']=results['input_tokens']*RATE
    return results


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run_folder',type=Path)
    args=parser.parse_args()
    results=analyze(args.run_folder)
    destination=ROOT/'output/routing-metrics.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({k:v for k,v in results.items() if k not in ('dev','all','test_threshold_curve')},indent=2))
