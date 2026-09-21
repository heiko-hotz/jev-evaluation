"""Run exactly the frozen 80-case stress suite on the existing Orthanc service."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess

from deberta import OPTIONS, TEMPLATE, WRAPPERS, payload, run_bundle
from deberta_remote import MODEL, REVISION
from smoke import ROOT

HASHES = {
    '/opt/orthanc-deberta/server.py': 'f0ed4bbc6e738761ac49dc5099368b25730630ece3151c05ebdd930b742a28e7',
    '/opt/orthanc-deberta/uv.lock': '922740e8b9158e325be9c19862e31c7ca19bc6d3ac99af6b20e5dc3cc03058ce',
    '/etc/systemd/system/orthanc-deberta.service': '44ac4b75e0da0ffbbb927f03122e8e94764d59f8845b7bab52e047397f5838d5',
}


def snapshot(folder, stage):
    def capture(wrapper, *args):
        return subprocess.check_output([str(WRAPPERS / wrapper), *args], text=True)
    state = capture('ssh.sh', 'systemctl', 'show', 'orthanc-deberta.service',
                    '--property=ActiveState,SubState,NRestarts,MemoryCurrent')
    hashes = capture('sudo.sh', '/usr/bin/sha256sum', *HASHES)
    observed = {line.split()[1]: line.split()[0] for line in hashes.splitlines()}
    data = dict(service=state, hashes=observed,
                loadavg=capture('ssh.sh', 'cat', '/proc/loadavg'),
                runtime=json.loads(capture('sudo.sh', '/opt/orthanc-deberta/.venv/bin/python', '-c',
                    'import json,platform,importlib.metadata as m; print(json.dumps(dict(python=platform.python_version(),torch=m.version("torch"),transformers=m.version("transformers"))))')))
    (folder / f'{stage}-service.json').write_text(json.dumps(data, indent=2) + '\n')
    assert observed == HASHES, 'Deployment changed; do not continue without review'
    assert 'ActiveState=active' in state and 'SubState=running' in state
    assert data['runtime'] == dict(python='3.12.14', torch='2.14.0+cpu', transformers='4.57.6')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    source = (ROOT / 'data/routing-challenge.json').read_bytes()
    cases = json.loads(source)
    assert hashlib.sha256(source).hexdigest() == 'd97970cf8d719d49d11e9fcc4159ccd72e5e6402484864d2b8579bdc369d81db'
    assert len(cases) == 80 and len({c['id'] for c in cases}) == 80
    assert Counter(c['expected'] for c in cases) == dict(billing=16, technical=16, account_access=16, other=16) | {None: 16}
    requests = [dict(id=c['id'], payload=payload(c['text'])) for c in cases]
    if not args.live:
        print('Dry run passed: 80 cases, 64 scored + 16 ambiguity; unchanged template; no network requests.')
        return
    os.umask(0o077)
    stamp = datetime.now(timezone.utc).strftime('deberta-challenge-%Y-%m-%dT%H%M%S.%fZ')
    folder = ROOT / 'local/runs' / stamp
    folder.mkdir(parents=True)
    manifest = dict(timestamp_utc=stamp, dataset='routing-challenge', request_cap=80, retries=0,
        timeout_seconds=60, concurrency=1, model=MODEL, revision=REVISION,
        options=OPTIONS, hypothesis_template=TEMPLATE, cpu_threads=4, pipeline_batch_size=1,
        precision='float32', api_fee_usd=0, electricity_cost_measured=False,
        latency_boundary='Orthanc loopback HTTP round trip; excludes SSH/staging; no warm-up exclusion',
        dataset_sha256=hashlib.sha256(source).hexdigest(), requests=requests,
        requests_sha256=hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest(),
        source_sha256={name: hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest()
            for name in ('deberta_challenge.py', 'deberta.py', 'deberta_remote.py', 'smoke.py', 'analyze_deberta_challenge.py')},
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)))
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Run directory: {folder}', flush=True)
    snapshot(folder, 'before')
    remote_runner = f'deberta-{stamp}-runner.py'
    subprocess.run([str(WRAPPERS / 'copy-to-tmp.sh'), str(ROOT / 'scripts/deberta_remote.py'), remote_runner], check=True)
    run_bundle(folder, 'challenge', requests)
    snapshot(folder, 'after')
    subprocess.run([str(WRAPPERS / 'ssh.sh'), 'rm', '--', f'/tmp/{remote_runner}',
                    f'/tmp/deberta-{stamp}-challenge.json'], check=True)
    print('Complete: exactly 80 classification requests, no retries or Jev calls; staged files removed.', flush=True)


if __name__ == '__main__':
    main()
