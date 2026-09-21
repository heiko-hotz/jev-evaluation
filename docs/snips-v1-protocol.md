# SNIPS seven-intent four-model evaluation v1

Authorized by the owner on 2026-09-21: run the proposed complete 700-example
SNIPS evaluation on Jev, DeBERTa, Gemini and Gemma. Prior no-spending-ceiling
instruction persists. This is a new experiment; cancelled CLINC remains stopped.

## Frozen data and task

SNIPS official repository sonos/nlu-benchmark, commit
b86ac7f1577868c42158d0dec77db50956046696, 2017-06-custom-intent-engines.
All seven validate_<intent>.json files: 100 examples per intent, 700 total.
These are the original repository's validation/evaluation files, not a claim
of identity with later third-party SNIPS train/dev/test partitions. No tuning
on these examples. Concatenate each data[].text exactly; ignore slot annotations.
Originals and CC0-1.0 licence retained at local/sources/snips; source hashes in
data/snips-provenance-v1.json. Cite Coucke et al. (2018),
https://arxiv.org/abs/1805.10190. Dataset source:
https://github.com/sonos/nlu-benchmark/tree/master/2017-06-custom-intent-engines.

Retain all official examples, including three repeated texts (697 unique texts).
Report a unique-text sensitivity check; duplicate groups are the resampling unit
for paired confidence intervals. Randomized shared order, seed 20260921.
Seven-way forced classification using original labels, no OOS, no demonstrations,
retrieval, fine-tuning, thresholds, prompt optimization or selective resampling.
Shared concise label definitions are in scripts/snips_eval.py and run manifest.

## Models and run conditions

- Jev jev-1.13.0, prior Choice endpoint and probability output.
- Gemini gemini-3.5-flash-lite, prior temperature 1, MINIMAL thinking,
  maxOutputTokens 1024, one candidate, JSON route string enum of seven labels.
- Gemma gemma-4-31b-it through the same Gemini key, temperature 1,
  HIGH thinking requested, maxOutputTokens 8192, same seven-label JSON schema.
  Prior live evidence shows Gemma may accept a schema without enforcing valid
  JSON on every response. Strict invalid outputs count as incorrect; no repair.
  Thinking-token absence does not prove whether internal thinking occurred.
- DeBERTa MoritzLaurer/deberta-v3-large-zeroshot-v2.0, pinned revision
  cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2, existing Orthanc CPU service,
  FP32, four intra-op threads, one inter-op, batch size one, multi_label=false.
  Original template: The appropriate routing category for this customer request is: {}
  Candidate labels use label: description. The authorized 151-label service
  validator remains installed; this task uses seven labels and changes no service.

700 examples per model, one sequential stream each; models may run concurrently.
No warm-up exclusion or extra classification probes: the first actual example
serves as the integration check and remains scored. At most ten HTTP 429 retries
per cloud model, at most one per example, after a 65-second wait. Never retry
successful/malformed-output responses or uncertain transport outcomes. Maximum
cloud HTTP attempts 710/model; DeBERTa 700. Stop on other HTTP/transport failure,
unexpected model or invalid usage/probability contract. Score validly transported
but malformed Gemini/Gemma output as incorrect and continue unchanged.
Gemma requests start at least 2.5 seconds apart, plus a 61-second rolling input
budget of 15,000 tokens with a 650-token request reserve, then actual usage.
Prior observed quota was 16,000 input tokens/minute. Exact other quotas may vary.

## Metrics and cost

Report strict accuracy, macro F1, per-class precision/recall/F1, confusion matrix,
valid-output rate, paired correctness and bootstrap intervals, unique-text
accuracy. Jev/DeBERTa probability metrics use positive-mass normalization of the
saved probabilities (raw mass and values retained); Brier and log loss (epsilon
1e-15) have distinct semantics from correctness. No invented LLM probabilities.

Report p50/p95 nonstreaming HTTP latency, HTTP and wall throughput, token usage,
retry/error counts and wall time including pacing. Cloud latency includes the
internet round trip; DeBERTa is Orthanc loopback HTTP and excludes staging/SSH.
No TTFT or decode-speed claims. Invalid-output latency/usage remains included.

Prices verified earlier in this same session: Jev $0.042/M input, free output;
Gemini $0.30/M uncached input, $0.03/M cached input, $2.50/M output including
thinking; Gemma free token pricing. No explicit caching is created; apply cache
discount only to returned cachedContentTokenCount. Account bills unverified.
DeBERTa has zero API fees; electricity/hardware cost unmeasured. API estimates
include returned retry usage if any; errors without usage have unknown charges.
https://docs.typesafe.ai/models; https://ai.google.dev/gemini-api/docs/pricing.

This is an English crowdsourced assistant benchmark with fairly distinct labels,
possible accuracy ceiling, no OOS test, and unknown training exposure. SNIPS was
absent from DeBERTa's inspected published task-training inventory, which does not
prove no pretraining/synthetic overlap. Findings apply to these configurations.
Raw data stays local/runs; sanitized report, metrics and predictions go in docs/results.
