"""Gemini forced-JSON classification. Dry run by default; offline analysis supported."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import urllib.request

try:
    from smoke import ROOT, CRITERIA, NoRedirect
    from analyze_routing import metrics
except ModuleNotFoundError:
    from scripts.smoke import ROOT, CRITERIA, NoRedirect
    from scripts.analyze_routing import metrics

MODEL = 'gemini-3.5-flash-lite'
BASE = 'https://generativelanguage.googleapis.com/v1beta/models/' + MODEL
DATA_HASH = 'd97970cf8d719d49d11e9fcc4159ccd72e5e6402484864d2b8579bdc369d81db'
CAP, CEILING = 80, 0.50
INPUT_RATE, OUTPUT_RATE = .30 / 1_000_000, 2.50 / 1_000_000
INPUT_RESERVE, OUTPUT_CAP = 8192, 1024
RESERVE = INPUT_RESERVE * INPUT_RATE + OUTPUT_CAP * OUTPUT_RATE
SCHEMA = {'type': 'object', 'properties': {'route': {'type': 'string', 'enum': list(CRITERIA)}},
          'required': ['route'], 'additionalProperties': False}


def payload(case):
    instructions = 'Which team should handle this customer request? Select the single best category.'
    return {
        'systemInstruction': {'parts': [{'text': instructions + '\nCategories:\n' + json.dumps(CRITERIA)}]},
        'contents': [{'role': 'user', 'parts': [{'text': json.dumps({'request': case['text']}, ensure_ascii=False)}]}],
        'generationConfig': {'responseMimeType': 'application/json', 'responseJsonSchema': SCHEMA,
                             'candidateCount': 1, 'temperature': 1.0, 'maxOutputTokens': OUTPUT_CAP,
                             'thinkingConfig': {'thinkingLevel': 'MINIMAL', 'includeThoughts': False}},
    }


def dataset():
    source = (ROOT / 'data/routing-challenge.json').read_bytes()
    assert hashlib.sha256(source).hexdigest() == DATA_HASH
    cases = json.loads(source)
    assert len(cases) == CAP and len({c['id'] for c in cases}) == CAP
    assert Counter(c['expected'] for c in cases) == dict.fromkeys(CRITERIA, 16) | {None: 16}
    # Short text-only payloads; reserve far more input tokens than their byte size.
    assert all(len(json.dumps(payload(c)).encode()) < INPUT_RESERVE // 2 for c in cases)
    assert CAP * RESERVE <= CEILING
    return cases


def read_key(env_file):
    for name in ('GEMINI_API_KEY', 'GOOGLE_API_KEY'):
        if os.environ.get(name):
            return os.environ[name]
    values = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.strip().removeprefix('export ').partition('=')
            if sep and key.strip() in ('GEMINI_API_KEY', 'GOOGLE_API_KEY'):
                values[key.strip()] = value.strip().strip('\"\'')
    for name in ('GEMINI_API_KEY', 'GOOGLE_API_KEY'):
        if values.get(name):
            return values[name]
    raise SystemExit('Missing Gemini API key; configure GEMINI_API_KEY locally or supply --env-file. No API calls made.')


def validate(raw):
    version = raw.get('modelVersion', '')
    if version != MODEL and not version.startswith(MODEL + '-'):
        raise ValueError('Unexpected model version')
    candidates = raw.get('candidates', [])
    if len(candidates) != 1 or candidates[0].get('finishReason') != 'STOP':
        raise ValueError('Missing or incomplete candidate')
    parts = candidates[0]['content']['parts']
    if not parts or any(p.get('thought') or not isinstance(p.get('text'), str) for p in parts):
        raise ValueError('Unexpected non-text output')
    value = json.loads(''.join(p['text'] for p in parts))
    if not isinstance(value, dict) or set(value) != {'route'} or value['route'] not in CRITERIA:
        raise ValueError('Output violates required category schema')
    usage = raw['usageMetadata']
    for field in ('promptTokenCount', 'candidatesTokenCount', 'totalTokenCount'):
        if type(usage.get(field)) is not int or usage[field] < 0:
            raise ValueError('Invalid token usage')
    thoughts = usage.get('thoughtsTokenCount', 0)
    if type(thoughts) is not int or thoughts < 0:
        raise ValueError('Invalid thinking usage')
    if usage['promptTokenCount'] > INPUT_RESERVE or usage['candidatesTokenCount'] + thoughts > OUTPUT_CAP:
        raise ValueError('Token usage exceeded budget reservation')
    return value['route']


def cost(raw):
    usage = raw['usageMetadata']
    return usage['promptTokenCount'] * INPUT_RATE + (usage['candidatesTokenCount'] + usage.get('thoughtsTokenCount', 0)) * OUTPUT_RATE


def analyze(folder):
    cases = dataset()
    manifest = json.loads((folder / 'manifest.json').read_text())
    requests = [payload(c) for c in cases]
    assert manifest['dataset_sha256'] == DATA_HASH and manifest['requests'] == requests
    assert manifest['requests_sha256'] == hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest()
    assert manifest['source_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows = json.loads((folder / 'results.json').read_text())
    assert len(rows) == CAP
    for case, row in zip(cases, rows):
        assert case['id'] == row['id'] and 'error' not in row
        raw = json.loads((folder / (case['id'] + '.json')).read_text())
        assert row['choice'] == validate(raw) and row['usage'] == raw['usageMetadata']
        assert row['expected'] == case['expected'] and row['model'] == raw['modelVersion']
        assert row['correct'] == (row['choice'] == case['expected'] if case['expected'] is not None else None)
        assert math.isfinite(row['latency_seconds']) and row['latency_seconds'] > 0
    pairs = [(c, r) for c, r in zip(cases, rows) if c['expected'] is not None]
    latencies = sorted(r['latency_seconds'] for r in rows)
    result = dict(gemini=metrics([c['expected'] for c, r in pairs], [r['choice'] for c, r in pairs]),
        model_versions=sorted({r['model'] for r in rows}), families={},
        failures=[dict(id=c['id'], text=c['text'], expected=c['expected'], observed=r['choice'])
                  for c, r in pairs if c['expected'] != r['choice']],
        ambiguity=[dict(id=c['id'], text=c['text'], choice=r['choice'], plausible_labels=c['plausible_labels'])
                   for c, r in zip(cases, rows) if c['expected'] is None],
        valid=CAP, latency_p50_seconds=statistics.median(latencies),
        latency_p95_seconds=latencies[math.ceil(.95 * CAP) - 1],
        total_http_seconds=sum(latencies), input_tokens=sum(r['usage']['promptTokenCount'] for r in rows),
        output_tokens=sum(r['usage']['candidatesTokenCount'] for r in rows),
        thinking_tokens=sum(r['usage'].get('thoughtsTokenCount', 0) for r in rows),
        estimated_cost_usd=sum(cost({'usageMetadata': r['usage']}) for r in rows))
    for family in sorted({c['family'] for c, r in pairs}):
        subset = [(c, r) for c, r in pairs if c['family'] == family]
        result['families'][family] = dict(n=len(subset), correct=sum(c['expected'] == r['choice'] for c, r in subset))
    destination = ROOT / 'output' / (folder.name + '-metrics.json')
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--live', action='store_true')
    mode.add_argument('--analyze', type=Path)
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    args = parser.parse_args()
    if args.analyze:
        print(json.dumps(analyze(args.analyze), indent=2))
        return
    cases = dataset()
    requests = [payload(c) for c in cases]
    print(f'{MODEL}: 80 calls maximum, US${CEILING:.2f} ceiling; forced JSON enum; thinking=MINIMAL; no retries.', flush=True)
    if not args.live:
        print('Dry run passed; no API access or credential reads.')
        return
    key = read_key(args.env_file)
    os.umask(0o077)
    stamp = datetime.now(timezone.utc).strftime('gemini-challenge-%Y-%m-%dT%H%M%S.%fZ')
    folder = ROOT / 'local/runs' / stamp
    folder.mkdir(parents=True)
    meta = dict(timestamp_utc=stamp, model=MODEL, dataset='routing-challenge', dataset_sha256=DATA_HASH,
        request_cap=CAP, ceiling_usd=CEILING, timeout_seconds=60, retries=0, concurrency=1,
        input_price_per_million=.30, output_price_per_million=2.50, billing_verified=False,
        input_token_reserve=INPUT_RESERVE, output_token_cap=OUTPUT_CAP, requests=requests,
        requests_sha256=hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((ROOT / 'docs/gemini-challenge-protocol.md').read_bytes()).hexdigest(),
        python=platform.python_version(), platform=platform.platform(),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)))
    (folder / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    print(f'Run directory: {folder}', flush=True)
    rows, spent = [], 0.0
    for case, body in zip(cases, requests):
        if spent + RESERVE > CEILING:
            break
        row = dict(id=case['id'], expected=case['expected'])
        start = time.perf_counter()
        try:
            request = urllib.request.Request(BASE + ':generateContent', data=json.dumps(body).encode(),
                headers={'x-goog-api-key': key, 'Content-Type': 'application/json'}, method='POST')
            with opener.open(request, timeout=60) as response:
                raw_bytes = response.read()
            row['latency_seconds'] = time.perf_counter() - start
            (folder / (case['id'] + '.json')).write_bytes(raw_bytes)
            raw = json.loads(raw_bytes)
            choice = validate(raw)
            spent += cost(raw)
            row.update(choice=choice, model=raw['modelVersion'], usage=raw['usageMetadata'],
                       correct=choice == case['expected'] if case['expected'] is not None else None)
        except Exception as error:
            row.update(error=type(error).__name__, http_status=getattr(error, 'code', None),
                       latency_seconds=time.perf_counter() - start)
        rows.append(row)
        (folder / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(f"{case['id']}: " + ('ERROR; stopping' if 'error' in row else f"{row['choice']}, correct={row['correct']}, {row['latency_seconds']:.3f}s"), flush=True)
        if 'error' in row:
            break
    summary = dict(attempted=len(rows), valid=sum('error' not in r for r in rows),
        scored=sum('correct' in r and r['correct'] is not None for r in rows), correct=sum(r.get('correct') is True for r in rows),
        estimated_cost_usd_known_usage=spent, billing_verified=False,
        complete=len(rows) == CAP and all('error' not in r for r in rows))
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if not summary['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
