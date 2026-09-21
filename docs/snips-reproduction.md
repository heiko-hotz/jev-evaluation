# Reproducing the SNIPS evaluation

## Two kinds of reproduction

1. **Offline score verification:** use the published predictions and sanitized
   observations to recompute classification, validity, probability, latency,
   and token-cost metrics. This requires only Python 3.11+ and no network.
2. **Fresh inference:** submit the frozen 700 cases to your own provider accounts
   and/or a local CPU DeBERTa service. Responses, latency, cost and model aliases
   can change. Exact historical cloud outputs cannot be guaranteed.

The published report was checked against all original responses locally. Those
raw account responses remain private. Sanitized observations retain only scored
labels, probabilities, usage counts, latency, model/revision, and the separate
postformatted label. They cannot independently prove the original response syntax
or billing. Formatter behavior is covered by offline tests; its published recovery
counts were verified against private originals before export.

## Offline verification

```sh
python3 scripts/verify_snips_publication.py
python3 -m unittest discover -s tests -p 'test_snips*.py' -v
python3 scripts/replicate_snips.py check
```

`data/snips-source/` contains the seven unchanged source files and upstream
README/LICENSE (CC0-1.0). Provenance pins repository revision and source hashes.
The check verifies exact text reconstruction, order/IDs, 100 cases per intent,
697 distinct texts, the publication checksums, and saved scores.

## Fresh inference

The standard-library runner uses Python 3.11+; optional `uv sync --frozen` creates
an environment from the root lockfile. Set `TYPESAFE_API_KEY` for Jev and
`GEMINI_API_KEY` (or `GOOGLE_API_KEY`) for Gemini/Gemma in your shell or secret
manager. The portable runner reads environment variables only; it does not read
another project, execute dotenv files, or publish credentials.

Read the [protocol](snips-v1-protocol.md) and
[formatting addendum](snips-gemma-postformat-addendum.md) first. Check current
provider model availability, contracts, quotas, and pricing before dispatch:
[TypeSafe API](https://docs.typesafe.ai/api),
[Google API](https://ai.google.dev/api/generate-content),
[Google pricing](https://ai.google.dev/gemini-api/docs/pricing).
Live TypeSafe documentation could not be fetched during packaging, so this release
preserves the previously verified experiment contract without claiming a new check.

```sh
python3 scripts/replicate_snips.py init
# Copy the printed bundle path into BUNDLE, for example:
BUNDLE=local/runs/snips-v1-<timestamp>
python3 scripts/replicate_snips.py run --bundle "$BUNDLE" --model jev
# The command above is a dry run. Each command below performs live inference:
python3 scripts/replicate_snips.py run --bundle "$BUNDLE" --model jev --live
python3 scripts/replicate_snips.py run --bundle "$BUNDLE" --model gemini --live
python3 scripts/replicate_snips.py run --bundle "$BUNDLE" --model gemma --live
python3 scripts/replicate_snips.py run --bundle "$BUNDLE" --model deberta --live
```

Each model is sequential; separate models may run in separate terminals. There
are exactly 700 examples per model, at most ten HTTP 429 retries per cloud model
(one per example), and no other retries. Gemma retains the original 2.5-second
minimum spacing and rolling token pacing. Malformed answers remain scored wrong.
Failures preserve partial evidence and stop that stream. A started model cannot
be redispatched in the same bundle. Init is also exclusive per checkout; use a
fresh clone for an intentional repeat, rather than deleting guards.

The original owner removed the spending ceiling: **the runner has a request cap
but no dollar cap**. Set account-level spending controls before running. The
historical $0.06982041 API estimate is not a promised price; hardware cost was
unmeasured. No fresh live inference was performed to package this repository.

## DeBERTa CPU baseline

Use Linux x86_64, Python 3.12, and `uv` for the supplied original dependency lock.
The observed runtime was Python 3.12.14, Torch 2.14.0+cpu, Transformers 4.57.6.
The model is pinned to `cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2`.
Download is roughly 900 MB. Model files remain ignored and retain their upstream
licence; they are not redistributed in this repository.

```sh
uv sync --project baseline/deberta --frozen --python 3.12.14
python3 baseline/deberta/download_model.py
uv run --project baseline/deberta --frozen python baseline/deberta/server.py
```

Run the DeBERTa evaluation command on that same machine in another terminal.
The original service code is included unchanged: loopback port 8091, FP32 CPU,
four intra-op threads, one inter-op thread, batch size one. The portable adapter
replaces the original private SSH/staging wrappers with direct loopback HTTP.
It uses the same payload, model revision, validator, and latency boundary.
Different hardware/load will change timing. Model loading/downloads are outside
the measured interval; no warm-up example is excluded.

The download helper verifies upstream file hashes. `model-manifest.json` records
the original downloaded-file hashes for comparison with `model/manifest.json`.
The service and locked runtime were inspected during packaging; installation and
model inference were not repeated on a clean Linux machine.

## Analyze your new run

```sh
python3 scripts/analyze_snips.py "$BUNDLE"
python3 scripts/report_snips.py
```

Analysis rechecks source/request hashes and your raw responses. It overwrites
`docs/results/2026-09-21-snips-{metrics.json,predictions.csv}` and the report in
your clone, retaining historical filenames for compatibility. Copy the original
published files before doing so if needed. Partial runs are identified explicitly.
The report's historical run-path/provenance prose describes the original experiment;
for your run, update that prose/date and record your machine, run path, and changes.
Do not present a new run as the original result. The offline publication verifier
intentionally fails after published artifacts change.

The historical `snips_eval.py` runner and imported helper modules are preserved
for source-hash verification. Use `replicate_snips.py` for new runs; historical
runners refer to private infrastructure and are not portable entry points.
