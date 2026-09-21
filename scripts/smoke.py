"""Synthetic smoke test; default dry run. Standard library only, no retries."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL = "jev-1.13.0"
CAP = 10
CEILING = 0.03
RATE = 0.042 / 1_000_000
RESERVE = 65_536 * RATE  # Conservatively reserve full documented context per call.
CRITERIA = {
    "billing": "Charges, payments, invoices, receipts, or refunds.",
    "technical": "Software bugs, crashes, API failures, or broken features; excludes login and authentication problems.",
    "account_access": "Signing in, password resets, authentication, or account lockouts.",
    "other": "None of the above, including hiring, sponsorship, and unrelated requests.",
}


def payload(case):
    return {"model": MODEL, "state": {"request": case["text"]}, "questions": {
        "route": {"type": "choice", "instructions": "Which team should handle this customer request? Select the single best category.", "criteria": CRITERIA}}}


def validate(response):
    if response.get("model") != MODEL:
        raise ValueError("Unexpected model version")
    a = response["answers"]["route"]
    p = a["probabilities"]
    numeric = lambda v: type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
    if a["type"] != "choice" or a["choice"] not in CRITERIA or set(p) != set(CRITERIA):
        raise ValueError("Invalid choice schema")
    if not all(numeric(v) for v in p.values()) or not math.isclose(sum(p.values()), 1, abs_tol=0.001):
        raise ValueError("Invalid probability distribution")
    if not numeric(a["confidence"]) or p[a["choice"]] < max(p.values()):
        raise ValueError("Invalid confidence or selected option")
    for k in ("input_tokens", "output_tokens"):
        if type(response["usage"][k]) is not int or response["usage"][k] < 0:
            raise ValueError("Invalid usage")
    if response["usage"]["input_tokens"] > 65_536:
        raise ValueError("Usage exceeds documented context")
    return a


def key_from_environment():
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            name, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and name.strip() == "TYPESAFE_API_KEY":
                key = value.strip().strip("\"'")
    if not key:
        raise SystemExit("Missing TYPESAFE_API_KEY; configure it locally in .env.")
    return key


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def main(dataset="smoke", cap=CAP, ceiling=CEILING, allow_ambiguous=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help=f"Make at most {cap} paid API calls")
    args = parser.parse_args()
    source = (ROOT / f"data/{dataset}.json").read_bytes()
    cases = json.loads(source)
    assert len(cases) == cap and len({c["id"] for c in cases}) == cap
    assert all((c["expected"] in CRITERIA or (allow_ambiguous and c["expected"] is None))
               and len(c["text"]) < 1000 for c in cases)
    assert cap * RESERVE <= ceiling
    requests = [payload(c) for c in cases]
    print(f"{len(cases)} synthetic cases; {MODEL}; cap={cap}; ceiling=US${ceiling:.2f}; retries=0", flush=True)
    if not args.live:
        print("Dry run passed. No network requests made.")
        return
    key = key_from_environment()
    os.umask(0o077)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S.%fZ")
    folder = ROOT / "local/runs" / stamp
    folder.mkdir(parents=True)
    meta = {"timestamp": stamp, "model": MODEL, "dataset": dataset, "request_cap": cap, "ceiling_usd": ceiling,
            "price_usd_per_million_input_tokens": 0.042, "concurrency": 1, "retries": 0,
            "timeout_seconds": 30, "dataset_sha256": hashlib.sha256(source).hexdigest(),
            "requests_sha256": hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest(),
            "python": platform.python_version(), "platform": platform.platform(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((ROOT / "scripts").glob("*.py"))},
            "requests": requests}
    (folder / "manifest.json").write_text(json.dumps(meta, indent=2))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    rows = []
    spent = 0.0
    for case, body in zip(cases, requests):
        if spent + RESERVE > ceiling:
            break
        row = {"id": case["id"], "expected": case["expected"]}
        start = time.perf_counter()
        try:
            request = urllib.request.Request("https://api.typesafe.ai/v1/systemone", data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
            with opener.open(request, timeout=30) as response:
                raw = response.read()
            row["latency_seconds"] = time.perf_counter() - start
            (folder / f'{case["id"]}.json').write_bytes(raw)
            result = json.loads(raw)
            answer = validate(result)
            spent += result["usage"]["input_tokens"] * RATE
            row.update(answer)
            row.update(usage=result["usage"],
                       correct=answer["choice"] == case["expected"] if case["expected"] is not None else None,
                       model=result["model"])
        except Exception as error:
            # Never print exception bodies, headers, credentials, or account responses.
            row.update(error=type(error).__name__, http_status=getattr(error, "code", None),
                       latency_seconds=time.perf_counter() - start)
        rows.append(row)
        (folder / "results.json").write_text(json.dumps(rows, indent=2))
        print(f'{case["id"]}: ' + ("ERROR; stopping" if "error" in row else f'{row["choice"]}, correct={row["correct"]}, {row["latency_seconds"]:.3f}s'), flush=True)
        if "error" in row:
            break
    good = [r for r in rows if "error" not in r]
    summary = {"attempted": len(rows), "valid": len(good), "correct": sum(r["correct"] is True for r in good),
               "scored": sum(r["correct"] is not None for r in good),
               "estimated_cost_usd_known_usage": spent, "billing_verified": False,
               "complete": len(good) == cap, "run_folder": str(folder)}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
