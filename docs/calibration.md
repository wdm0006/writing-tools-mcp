# AI-detection calibration

Published operating characteristics for the two AI-detection signals, `stylometric_analysis` and `perplexity_analysis`, derived entirely from the committed benchmark scores - no new data collection, no model inference. Positive class is **machine-generated text**: a true positive is a machine-written document the detector flagged, a false positive is a human-written document it flagged.

These are statistical measurements of writing style, not evidence of authorship. A flagged document is a document whose measured features differ from a baseline; it is not a document proven to be machine-written.

This file is generated; do not edit it by hand. Regenerate with `uv run benchmarks/calibrate_detection.py` and commit the result - CI fails when this file drifts from the committed scores, so a threshold change must re-derive the published numbers.

## Provenance

- Scores: `benchmarks/reports/raid_wiki_v1/scores.jsonl` - 220 documents (110 human / 110 machine)
- Corpus: `benchmarks/corpora/raid_wiki_v1/corpus.jsonl`, baseline `brown_corpus` (from `run_metadata.json` beside the scores)
- Scores were produced by `benchmarks/run_benchmark.py` running the repository's own detector code; `benchmarks/reports/raid_wiki_v1/report.md` records the thresholds in effect for the boolean flags below.

## Shipped boolean decisions, measured

The recorded `flags.high_ai_probability` decisions, scored as-is on the labeled corpus.

### `stylometric_analysis` → `flags.high_ai_probability`

Documents: 220 total, 220 scored (110 human / 110 machine), 0 failed (0 human / 0 machine). Failed analyses are excluded from the matrix below rather than counted as negatives.

| | predicted machine | predicted human |
| --- | ---: | ---: |
| **actual machine** | 10 (TP) | 100 (FN) |
| **actual human** | 1 (FP) | 109 (TN) |

| metric | value |
| --- | ---: |
| accuracy | 0.5409 |
| precision | 0.9091 |
| recall (TPR) | 0.0909 |
| F1 | 0.1653 |
| false positive rate | 0.0091 |
| specificity | 0.9909 |

### `perplexity_analysis` → `flags.high_ai_probability`

Documents: 220 total, 220 scored (110 human / 110 machine), 0 failed (0 human / 0 machine). Failed analyses are excluded from the matrix below rather than counted as negatives.

| | predicted machine | predicted human |
| --- | ---: | ---: |
| **actual machine** | 0 (TP) | 110 (FN) |
| **actual human** | 0 (FP) | 110 (TN) |

| metric | value |
| --- | ---: |
| accuracy | 0.5000 |
| precision | n/a |
| recall (TPR) | 0.0000 |
| F1 | n/a |
| false positive rate | 0.0000 |
| specificity | 1.0000 |

## Single-measurement sweeps at target true-positive rates

The shipped booleans each combine several conditions, so no single knob maps onto them. The tables below are deliberately simpler classifiers - one measurement, one cut - built from the same recorded scores. They answer the question a threshold change actually asks (what recall costs at a given false-positive rate) and are expected to disagree with the shipped decisions above. Sweeps run over the recorded measurements, never over the recorded boolean flags: the flags were decided at the thresholds in effect when the harness ran, so sweeping them would reproduce those thresholds and nothing else.

### Stylometry: `confidence_score` against a cut

`confidence_score` is the 0-1 number the stylometry flag thresholds on - the sweep cuts it with `confidence_score >= t`, the direction the machine class scores higher.

Each row is the highest cut point whose measured recall still reaches the target - the cheapest threshold for that recall, since a higher cut can only lower the FPR. Scores are coarse and discrete, so the achieved TPR is reported next to the target rather than assumed equal to it.

Cut points that reach no target are omitted; `-` means no threshold in this sample reaches the target at all.

| target TPR | cut | achieved TPR | precision | recall (TPR) | FPR | TP | FN | FP | TN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50.00% | >= 0.200 | 0.6727 | 0.4302 | 0.6727 | 0.8909 | 74 | 36 | 98 | 12 |
| 75.00% | >= 0.100 | 0.9091 | 0.4808 | 0.9091 | 0.9818 | 100 | 10 | 108 | 2 |
| 90.00% | >= 0.100 | 0.9091 | 0.4808 | 0.9091 | 0.9818 | 100 | 10 | 108 | 2 |

#### Full `confidence_score` sweep

Every distinct `confidence_score` value as a cut point, ascending.

| cut | precision | recall (TPR) | FPR | TP | FN | FP | TN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| >= 0.000 | 0.5000 | 1.0000 | 1.0000 | 110 | 0 | 110 | 0 |
| >= 0.100 | 0.4808 | 0.9091 | 0.9818 | 100 | 10 | 108 | 2 |
| >= 0.200 | 0.4302 | 0.6727 | 0.8909 | 74 | 36 | 98 | 12 |
| >= 0.300 | 0.4274 | 0.4545 | 0.6091 | 50 | 60 | 67 | 43 |
| >= 0.350 | 0.4583 | 0.3000 | 0.3545 | 33 | 77 | 39 | 71 |
| >= 0.400 | 0.4638 | 0.2909 | 0.3364 | 32 | 78 | 37 | 73 |
| >= 0.450 | 0.6250 | 0.2273 | 0.1364 | 25 | 85 | 15 | 95 |
| >= 0.500 | 0.6154 | 0.2182 | 0.1364 | 24 | 86 | 15 | 95 |
| >= 0.550 | 0.7600 | 0.1727 | 0.0545 | 19 | 91 | 6 | 104 |
| >= 0.600 | 0.7500 | 0.1636 | 0.0545 | 18 | 92 | 6 | 104 |
| >= 0.650 | 0.9167 | 0.1000 | 0.0091 | 11 | 99 | 1 | 109 |
| >= 0.700 | 0.9091 | 0.0909 | 0.0091 | 10 | 100 | 1 | 109 |
| >= 0.800 | 1.0000 | 0.0727 | 0.0000 | 8 | 102 | 0 | 110 |
| >= 0.900 | 1.0000 | 0.0455 | 0.0000 | 5 | 105 | 0 | 110 |
| >= 0.950 | 1.0000 | 0.0273 | 0.0000 | 3 | 107 | 0 | 110 |
| >= 1.000 | 1.0000 | 0.0182 | 0.0000 | 2 | 108 | 0 | 110 |

### Perplexity: `doc_ppl` against a cut

Machine text measures *lower* GPT-2 perplexity, so the cut renders as `doc_ppl <= t`; internally the measurement is negated and the same `score >= threshold` helpers run unchanged.

Each row is the highest cut point whose measured recall still reaches the target - the cheapest threshold for that recall, since a higher cut can only lower the FPR. Scores are coarse and discrete, so the achieved TPR is reported next to the target rather than assumed equal to it.

Cut points that reach no target are omitted; `-` means no threshold in this sample reaches the target at all.

| target TPR | cut | achieved TPR | precision | recall (TPR) | FPR | TP | FN | FP | TN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50.00% | <= 42.21 | 0.5000 | 0.7857 | 0.5000 | 0.1364 | 55 | 55 | 15 | 95 |
| 75.00% | <= 57.78 | 0.7545 | 0.7155 | 0.7545 | 0.3000 | 83 | 27 | 33 | 77 |
| 90.00% | <= 118.42 | 0.9000 | 0.5380 | 0.9000 | 0.7727 | 99 | 11 | 85 | 25 |

`doc_ppl` takes 218 distinct values across this corpus, more than the 25-row limit for a published full sweep, so only the target rows above are reproduced here. The full sweep is computed by the same `threshold_sweep` helper and available from the generator.
