"""Bounded loopback HTTP runner, executed on Orthanc through its SSH wrapper."""
import json
import math
from pathlib import Path
import sys
import time
import urllib.request

MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
REVISION = "cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2"


def validate(raw, labels):
    if raw.get("model") != MODEL or raw.get("revision") != REVISION:
        raise ValueError("Unexpected model/revision")
    returned = raw.get("labels", [])
    scores = raw.get("scores", [])
    if len(returned) != len(labels) or set(returned) != set(labels) or len(scores) != len(labels):
        raise ValueError("Unexpected response labels")
    if not all(type(x) in (float, int) and math.isfinite(x) and 0 <= x <= 1 for x in scores):
        raise ValueError("Invalid score")
    if not math.isclose(sum(scores), 1, abs_tol=0.001) or scores != sorted(scores, reverse=True):
        raise ValueError("Invalid single-label distribution")


def main():
    requests = json.loads(Path(sys.argv[1]).read_text())
    if not 1 <= len(requests) <= 100:
        raise ValueError("Request cap exceeded")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for index, item in enumerate(requests):
        payload = item["payload"]
        request = urllib.request.Request("http://127.0.0.1:8091/classify",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        start = time.perf_counter()
        row = {"id": item["id"], "index": index}
        try:
            with opener.open(request, timeout=60) as response:
                raw = json.load(response)
            row["latency_seconds"] = time.perf_counter() - start
            row["response"] = raw
            validate(raw, payload["parameters"]["candidate_labels"])
        except Exception as error:
            row["error"] = type(error).__name__ + ": " + str(error)
            print(json.dumps(row), flush=True)
            raise SystemExit(1)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
