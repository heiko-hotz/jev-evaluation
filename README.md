# SNIPS intent classification evaluation

A reproducible comparison of **Jev, Gemini, Gemma, and DeBERTa** on 700 English
assistant requests across seven intents. Same inputs and label definitions;
no demonstrations, fine-tuning, retrieval, or prompt optimization.

| Model | Strict accuracy | After formatting recovery |
|---|---:|---:|
| Jev 1.13.0 | 97.14% | — |
| Gemini 3.5 Flash-Lite | 99.00% | 99.00% |
| Gemma 4 31B | 80.57% | 98.86% |
| DeBERTa-v3-large zero-shot, CPU | 88.86% | — |

Results from **21 September 2026**, 700 requests per model. Strict scoring counts
malformed answers as wrong. Gemma's second score removes only Markdown fences:
130 outputs recovered, zero valid answers changed. This procedure was introduced
after observing formatting errors.

**What this is:** a small, auditable comparison of specific model configurations,
including accuracy, output validity, latency, and estimated API cost.

**What this isn't:** a general model leaderboard, a production-readiness test,
an out-of-scope detection test, or a reproduction of SNIPS's original slot-filling
benchmark. Some labels are ambiguous, training exposure is unknown, and timing
boundaries differ. The one-example Gemini/Gemma gap does not establish a winner.

## Reproduce

Python 3.11+; offline verification needs no dependencies, accounts, or API calls:

```sh
git clone https://github.com/heiko-hotz/jev-evaluation.git
cd jev-evaluation
python3 scripts/verify_snips_publication.py
python3 -m unittest discover -s tests -p 'test_snips*.py' -v
```

- **[Replication guide](docs/snips-reproduction.md):** setup, keys, CPU baseline, and fresh runs.
- **[Full results](docs/results/2026-09-21-snips.md):** metrics, limitations, and paired comparisons.
- **[Protocol](docs/snips-v1-protocol.md)** · **[Predictions](docs/results/2026-09-21-snips-predictions.csv)** · **[Data provenance](data/snips-provenance-v1.json)**

The frozen data, upstream CC0 sources, prompts, inference/analysis code, baseline
lockfile, and sanitized observations are included. Raw account responses and
credentials are excluded. Historical helper modules retain their original names;
the replication guide is the entry point for SNIPS.
