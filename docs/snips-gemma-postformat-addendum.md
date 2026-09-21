# Gemma post-formatting score

The owner explicitly requested two Gemma scores during the SNIPS run: raw and
post-formatting, provided recovery is reliable. Both use all 700 gold labels
unchanged. The inference requests and settings remain frozen; no new inference
is performed for recovery.

The deterministic rule removes only enclosing/trailing Markdown code fences,
then parses exactly one JSON object with unique keys and exactly one `route`
property whose string is an allowed label. It rejects prose, extra fields,
unknown labels, truncation and duplicate keys. It never guesses a corrected
label or uses gold labels. Code: `scripts/gemma_postformat.py`.

Analysis verifies that all initially valid answers remain unchanged, records
recovered/unrecoverable counts, and exports raw plus postformatted predictions.
Primary raw scores remain available. Syntax recovery does not establish semantic
correctness; recovered labels are scored normally against original gold.
The procedure was introduced after observing formatting errors, before the run
completed; it is an observed pipeline correction, not a fresh held-out study of
a previously frozen formatter.
