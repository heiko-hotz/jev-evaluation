"""Verify published SNIPS evidence offline; never reads keys or calls a provider."""
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from analyze_snips import classification
from snips_eval import CRITERIA, MODELS, ROOT, payload


def verify():
    checksums = json.loads((ROOT / 'data/snips-publication-sha256.json').read_text())
    for name, expected in checksums.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    cases = json.loads((ROOT / 'data/snips-v1.json').read_text())
    assert len(cases) == len({c['id'] for c in cases}) == 700
    assert len({c['text'] for c in cases}) == 697
    assert Counter(c['expected'] for c in cases) == dict.fromkeys(CRITERIA, 100)
    provenance = json.loads((ROOT / 'data/snips-provenance-v1.json').read_text())
    sources = {}
    for item in provenance['sources']:
        path = ROOT / 'data/snips-source' / item['url'].rsplit('/', 1)[1]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        sources[path.name] = json.loads(path.read_text(encoding='utf-8-sig'))
    original = json.loads((ROOT / 'data/snips-original-manifest.json').read_text())
    for model in MODELS:
        requests = [dict(id=c['id'], payload=payload(model, c)) for c in cases]
        encoded = (json.dumps(requests, ensure_ascii=False, indent=2) + '\n').encode()
        assert hashlib.sha256(encoded).hexdigest() == original['request_sha256'][model]
    for name, digest in original['file_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    for c in cases:
        assert c['text'] == ''.join(p['text'] for p in sources[c['source_file']][c['expected']][c['source_index']]['data'])
    result = ROOT / 'docs/results'
    report = json.loads((result / '2026-09-21-snips-metrics.json').read_text())
    observations = json.loads((result / '2026-09-21-snips-observations.json').read_text())
    with (result / '2026-09-21-snips-predictions.csv').open(newline='') as f:
        predictions = list(csv.DictReader(f))
    assert len(predictions) == 700
    for c, row in zip(cases, predictions):
        assert all(c[k] == row[k] for k in ('id', 'text', 'expected'))
    for model in MODELS:
        rows = observations[model]
        assert len(rows) == 700
        for case, row, published in zip(cases, rows, predictions):
            assert row['id'] == case['id'] and row['expected'] == case['expected']
            assert (row.get('choice') or 'INVALID') == published[model]
            assert row.get('invalid_output', False) == (row.get('choice') is None)
        expected = report['models'][model]
        actual = classification([c['expected'] for c in cases], [r.get('choice') for r in rows], CRITERIA)
        assert actual == expected['metrics'], model
        assert sum(r.get('choice') is not None for r in rows) == expected['valid']
        latency = sorted(r['latency_seconds'] for r in rows)
        assert statistics.median(latency) == expected['latency_p50_seconds']
        assert latency[math.ceil(.95 * len(rows)) - 1] == expected['latency_p95_seconds']
        groups = defaultdict(list)
        for c, r in zip(cases, rows):
            groups[c['text']].append(c['expected'] == r.get('choice'))
        assert statistics.mean(statistics.mean(v) for v in groups.values()) == expected['unique_text_accuracy']
        if model in ('jev', 'deberta'):
            brier, loss = [], []
            for r in rows:
                p = r['probabilities']
                p = {k: v / sum(p.values()) for k, v in p.items()}
                brier.append(sum((v - (k == r['expected'])) ** 2 for k, v in p.items()))
                loss.append(-math.log(max(p[r['expected']], 1e-15)))
            assert statistics.mean(brier) == expected['multiclass_brier']
            assert statistics.mean(loss) == expected['log_loss_epsilon_1e15']
        if model in ('gemini', 'gemma'):
            formatted = [r['postformatted_choice'] for r in rows]
            assert classification([c['expected'] for c in cases], formatted, CRITERIA) == expected['secondary_fence_normalized']
            assert all(r['choice'] == p for r, p in zip(rows, formatted) if r.get('choice'))
            recovered = sum(r.get('invalid_output', False) and p is not None for r, p in zip(rows, formatted))
            assert recovered == expected['format_recovery']['recovered_outputs']
            if model == 'gemma':
                for r, p, csv_row in zip(rows, formatted, predictions):
                    assert (p or 'INVALID') == csv_row['gemma_postformatted']
                    assert str(bool(r.get('invalid_output') and p)) == csv_row['gemma_format_recovered']
            inputs = sum(r['usage']['promptTokenCount'] for r in rows)
            outputs = sum(r['usage']['candidatesTokenCount'] for r in rows)
            thoughts = sum(r['usage'].get('thoughtsTokenCount', 0) for r in rows)
            cached = sum(r['usage'].get('cachedContentTokenCount', 0) for r in rows)
            cost = ((inputs-cached)*.30 + cached*.03 + (outputs+thoughts)*2.50)/1e6 if model == 'gemini' else 0
        elif model == 'jev':
            inputs = sum(r['usage']['input_tokens'] for r in rows)
            outputs = sum(r['usage']['output_tokens'] for r in rows)
            cost = inputs * .042 / 1e6
        else:
            inputs = outputs = None
            cost = 0
        assert inputs == expected['input_tokens'] and outputs == expected['output_tokens']
        assert cost == expected['estimated_api_cost_usd']
        print(f'{model}: {actual["correct"]}/700 strict; published metrics verified')
    print('Source hashes, 700 predictions, sanitized observations and publication checksums verified.')


if __name__ == '__main__':
    verify()
