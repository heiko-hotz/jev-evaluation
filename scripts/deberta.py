"""One smoke request, then 100 frozen routing cases on Orthanc CPU; no Jev calls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from smoke import ROOT, CRITERIA

OPS = ROOT.parent / "orthanc-ops"
WRAPPERS = OPS / ".agents/skills/manage-orthanc/scripts"
OPTIONS = {key: f"{key}: {description}" for key, description in CRITERIA.items()}
TEMPLATE = "The appropriate routing category for this customer request is: {}"


def payload(text):
    return {"inputs": text, "parameters": {"candidate_labels": list(OPTIONS.values()),
             "multi_label": False, "hypothesis_template": TEMPLATE}}


def run_bundle(folder, kind, requests):
    bundle = folder / f"{kind}-requests.json"
    bundle.write_text(json.dumps(requests, indent=2) + "\n")
    remote_name = f"deberta-{folder.name}-{kind}.json"
    subprocess.run([str(WRAPPERS / "copy-to-tmp.sh"), str(bundle), remote_name], check=True)
    proc = subprocess.Popen([str(WRAPPERS / "ssh.sh"), "/usr/bin/python3", "-u",
        f"/tmp/deberta-{folder.name}-runner.py", f"/tmp/{remote_name}"], stdout=subprocess.PIPE, text=True)
    rows = []
    with (folder / f"{kind}-responses.jsonl").open("w") as saved:
        for line in proc.stdout:
            saved.write(line)
            saved.flush()
            row = json.loads(line)
            rows.append(row)
            if "error" in row:
                print(f"{kind}: stopped on {row['id']}: {row['error']}", flush=True)
            elif len(rows) == 1 or len(rows) % 20 == 0:
                print(f"{kind}: {len(rows)}/{len(requests)} completed; latest {row['latency_seconds']:.3f}s", flush=True)
    if proc.wait() != 0 or len(rows) != len(requests) or any("error" in r for r in rows):
        raise RuntimeError("Incomplete run; saved responses retained; no retries")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    source = (ROOT / "data/routing.json").read_bytes()
    cases = json.loads(source)
    assert len(cases) == 100 and len({c['id'] for c in cases}) == 100
    requests = [{"id": c["id"], "payload": payload(c["text"])} for c in cases]
    smoke = [{"id": "smoke-account-access", "payload": payload("I cannot sign in because my account is locked. Please help me regain access.")}]
    if not args.live:
        print("Ready: 1 smoke request followed by 100 routing requests; no network requests made.")
        return
    os.umask(0o077)
    stamp = datetime.now(timezone.utc).strftime("deberta-%Y-%m-%dT%H%M%S.%fZ")
    folder = ROOT / "local/runs" / stamp
    folder.mkdir(parents=True)
    manifest = {"timestamp_utc": stamp, "request_cap": 101, "retries": 0, "timeout_seconds": 60,
        "dataset_sha256": hashlib.sha256(source).hexdigest(), "options": OPTIONS,
        "hypothesis_template": TEMPLATE, "model": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        "revision": "cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2", "concurrency": 1,
        "cpu_threads": 4, "pipeline_batch_size": 1, "precision": "float32",
        "latency_boundary": "Orthanc loopback HTTP round trip; excludes SSH, staging, startup and smoke",
        "torch": "2.14.0+cpu", "transformers": "4.57.6", "service_python": "3.12.14",
        "server_sha256": "f0ed4bbc6e738761ac49dc5099368b25730630ece3151c05ebdd930b742a28e7",
        "runtime_lock_sha256": "922740e8b9158e325be9c19862e31c7ca19bc6d3ac99af6b20e5dc3cc03058ce",
        "requests": requests, "smoke": smoke,
        "requests_sha256": hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest(),
        "source_sha256": {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
            for name in ("deberta.py", "deberta_remote.py", "smoke.py")},
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Run directory: {folder}", flush=True)
    subprocess.run([str(WRAPPERS / "copy-to-tmp.sh"), str(ROOT / "scripts/deberta_remote.py"),
        f"deberta-{stamp}-runner.py"], check=True)
    rows = run_bundle(folder, "smoke", smoke)
    if rows[0]["response"]["labels"][0] != OPTIONS["account_access"]:
        raise RuntimeError("Smoke prediction incorrect; benchmark not dispatched")
    print("Smoke inference valid and correct; dispatching the 100-case dataset.", flush=True)
    run_bundle(folder, "routing", requests)
    print("Complete: 1 smoke + 100 benchmark requests; no retries or Jev calls.", flush=True)


if __name__ == "__main__":
    main()
