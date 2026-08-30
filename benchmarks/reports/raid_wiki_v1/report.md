# Detector benchmark - raid_wiki_v1

Positive class is **machine-generated text**: a true positive is a machine-written document the detector flagged, a false positive is a human-written document it flagged.

These are statistical measurements of writing style, not evidence of authorship. A flagged document is a document whose measured features differ from a baseline; it is not a document proven to be machine-written.

This report is generated; do not edit it by hand. Regenerate with the command in `benchmarks/README.md`. Run-varying values (timestamps, library versions) are recorded in `run_metadata.json` beside this file, not here.

## Corpus

- Corpus: `raid_wiki_v1` (220 labeled documents)
- Documents: 220 (110 human, 110 machine)
- Manifest: `benchmarks/corpora/raid_wiki_v1/MANIFEST.md`
- Analyses run: `stylometry`, `perplexity`

## Configuration under test

Thresholds are read from the server's own configuration and are reported here only so a reader can tell which constants produced these numbers. This benchmark changes none of them.

| setting | value |
| --- | ---: |
| `perplexity.model_name` | gpt2 |
| `perplexity.thresholds.burstiness_min` | 2.5 |
| `perplexity.thresholds.ppl_max` | 25.0 |
| `stylometry.thresholds.ai_confidence_threshold` | 0.7 |
| `stylometry.thresholds.error_z` | 3.0 |
| `stylometry.thresholds.warning_z` | 2.0 |

## Sample counts

| method | scored | scored human | scored machine | failed |
| --- | ---: | ---: | ---: | ---: |
| `stylometry` | 220 | 110 | 110 | 0 |
| `perplexity` | 220 | 110 | 110 | 0 |

## Failed analyses

- `stylometry`: 0 failures.
- `perplexity`: 0 failures.

Every document produced an analyzer result, so no scores are missing.

## Shipped boolean decisions

### `stylometric_analysis` -> `flags.high_ai_probability`

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
| true positive rate | 0.0909 |
| specificity | 0.9909 |

### `perplexity_analysis` -> `flags.high_ai_probability`

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
| true positive rate | 0.0000 |
| specificity | 1.0000 |

### Unmeasured perplexity statistics

A successful analysis can still leave a statistic undefined - the analyzer reports `null` rather than substituting a value, and a `null` can never set the flag.

- `doc_ppl` null on 0 of 220 scored documents.
- `doc_burstiness` null on 0 of 220 scored documents.

## Stylometry FPR at declared TPR targets

This sweeps `flags.confidence_score` alone with a `score >= threshold` rule. That is a *different* classifier from the shipped boolean above, which additionally requires at least two named AI indicators - so the two sections are expected to disagree.

`confidence_score` takes a small number of discrete values, so an exact target TPR is generally unreachable. Each row reports the achieved TPR next to the target; the threshold chosen is the highest one still reaching the target.

| target TPR | threshold | achieved TPR | FPR | TP | FN | FP | TN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50.00% | 0.200 | 0.6727 | 0.8909 | 74 | 36 | 98 | 12 |
| 75.00% | 0.100 | 0.9091 | 0.9818 | 100 | 10 | 108 | 2 |
| 90.00% | 0.100 | 0.9091 | 0.9818 | 100 | 10 | 108 | 2 |

### Full confidence-score sweep

| threshold | TPR | FPR | TP | FN | FP | TN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 1.0000 | 1.0000 | 110 | 0 | 110 | 0 |
| 0.100 | 0.9091 | 0.9818 | 100 | 10 | 108 | 2 |
| 0.200 | 0.6727 | 0.8909 | 74 | 36 | 98 | 12 |
| 0.300 | 0.4545 | 0.6091 | 50 | 60 | 67 | 43 |
| 0.350 | 0.3000 | 0.3545 | 33 | 77 | 39 | 71 |
| 0.400 | 0.2909 | 0.3364 | 32 | 78 | 37 | 73 |
| 0.450 | 0.2273 | 0.1364 | 25 | 85 | 15 | 95 |
| 0.500 | 0.2182 | 0.1364 | 24 | 86 | 15 | 95 |
| 0.550 | 0.1727 | 0.0545 | 19 | 91 | 6 | 104 |
| 0.600 | 0.1636 | 0.0545 | 18 | 92 | 6 | 104 |
| 0.650 | 0.1000 | 0.0091 | 11 | 99 | 1 | 109 |
| 0.700 | 0.0909 | 0.0091 | 10 | 100 | 1 | 109 |
| 0.800 | 0.0727 | 0.0000 | 8 | 102 | 0 | 110 |
| 0.900 | 0.0455 | 0.0000 | 5 | 105 | 0 | 110 |
| 0.950 | 0.0273 | 0.0000 | 3 | 107 | 0 | 110 |
| 1.000 | 0.0182 | 0.0000 | 2 | 108 | 0 | 110 |

## Per-feature class summaries

### Stylometric features (raw values)

Direction is the sign of the machine-minus-human mean difference. Cohen's d is the pooled-standard-deviation effect size; values near zero mean the feature does not separate the two classes in this corpus.

| feature | human n | human mean | human std | machine n | machine mean | machine std | mean diff (machine - human) | Cohen's d | direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ari` | 110 | 13.2573 | 2.8142 | 110 | 13.8491 | 5.9667 | 0.5918 | 0.1269 | higher in machine |
| `avg_sentence_len` | 110 | 22.3358 | 4.3081 | 110 | 21.9537 | 9.3976 | -0.3821 | -0.0523 | lower in machine |
| `avg_word_len` | 110 | 4.9395 | 0.3459 | 110 | 4.9863 | 0.6058 | 0.0468 | 0.0949 | higher in machine |
| `booster_rate` | 110 | 0.0007 | 0.0028 | 110 | 0.0004 | 0.0014 | -0.0003 | -0.1582 | lower in machine |
| `coleman_liau` | 110 | 11.8075 | 2.1295 | 110 | 12.0540 | 3.6506 | 0.2465 | 0.0825 | higher in machine |
| `comma_ratio` | 110 | 0.3615 | 0.1030 | 110 | 0.3733 | 0.1363 | 0.0118 | 0.0978 | higher in machine |
| `dale_chall` | 110 | 9.9195 | 0.9084 | 110 | 9.4131 | 2.2110 | -0.5065 | -0.2996 | lower in machine |
| `ellipsis_ratio` | 110 | 0.0002 | 0.0025 | 110 | 0.0017 | 0.0093 | 0.0015 | 0.2170 | higher in machine |
| `em_dash_ratio` | 110 | 0.0019 | 0.0106 | 110 | 0.0019 | 0.0099 | 0.0001 | 0.0062 | higher in machine |
| `exclamation_ratio` | 110 | 0.0000 | 0.0000 | 110 | 0.0021 | 0.0105 | 0.0021 | 0.2762 | higher in machine |
| `fog` | 110 | 12.9971 | 2.4734 | 110 | 13.2960 | 4.7778 | 0.2989 | 0.0786 | higher in machine |
| `fourgram_repetition_rate` | 110 | 0.0108 | 0.0190 | 110 | 0.0789 | 0.1829 | 0.0681 | 0.5236 | higher in machine |
| `function_word_ratio` | 110 | 0.3875 | 0.0382 | 110 | 0.3984 | 0.0669 | 0.0109 | 0.1997 | higher in machine |
| `hapax_legomena_rate` | 110 | 0.7276 | 0.0564 | 110 | 0.6948 | 0.1561 | -0.0328 | -0.2793 | lower in machine |
| `hedge_rate` | 110 | 0.0022 | 0.0043 | 110 | 0.0038 | 0.0053 | 0.0015 | 0.3211 | higher in machine |
| `kincaid` | 110 | 12.0582 | 2.5037 | 110 | 12.6236 | 4.5119 | 0.5655 | 0.1550 | higher in machine |
| `lexical_density` | 110 | 0.5713 | 0.0409 | 110 | 0.5598 | 0.0667 | -0.0115 | -0.2071 | lower in machine |
| `mattr` | 110 | 0.7796 | 0.0431 | 110 | 0.7669 | 0.1188 | -0.0127 | -0.1420 | lower in machine |
| `mean_dependency_distance` | 110 | 2.5130 | 0.3247 | 110 | 2.5372 | 1.6284 | 0.0241 | 0.0206 | higher in machine |
| `mean_word_frequency` | 110 | 5.4253 | 0.2232 | 110 | 5.5039 | 0.3177 | 0.0786 | 0.2863 | higher in machine |
| `mtld` | 110 | 74.6848 | 24.5181 | 110 | 213.7973 | 860.2212 | 139.1126 | 0.2286 | higher in machine |
| `mtld_lemma` | 110 | 68.2736 | 22.0338 | 110 | 110.7145 | 238.9708 | 42.4409 | 0.2501 | higher in machine |
| `parenthetical_rate` | 110 | 0.2382 | 0.2252 | 110 | 0.1948 | 0.4874 | -0.0434 | -0.1143 | lower in machine |
| `pos_ratios.ADJ` | 110 | 0.0656 | 0.0316 | 110 | 0.0688 | 0.0330 | 0.0031 | 0.0973 | higher in machine |
| `pos_ratios.ADP` | 110 | 0.1427 | 0.0265 | 110 | 0.1389 | 0.0309 | -0.0037 | -0.1300 | lower in machine |
| `pos_ratios.ADV` | 108 | 0.0265 | 0.0137 | 105 | 0.0276 | 0.0199 | 0.0011 | 0.0624 | higher in machine |
| `pos_ratios.AUX` | 110 | 0.0387 | 0.0173 | 110 | 0.0458 | 0.0197 | 0.0071 | 0.3843 | higher in machine |
| `pos_ratios.CCONJ` | 110 | 0.0386 | 0.0160 | 110 | 0.0389 | 0.0167 | 0.0003 | 0.0167 | higher in machine |
| `pos_ratios.DET` | 110 | 0.0952 | 0.0259 | 110 | 0.0996 | 0.0377 | 0.0045 | 0.1378 | higher in machine |
| `pos_ratios.INTJ` | 2 | 0.0055 | 0.0007 | 7 | 0.0036 | 0.0011 | -0.0019 | -1.7803 | lower in machine |
| `pos_ratios.NOUN` | 110 | 0.1789 | 0.0555 | 110 | 0.2039 | 0.0710 | 0.0250 | 0.3916 | higher in machine |
| `pos_ratios.NUM` | 109 | 0.0431 | 0.0274 | 105 | 0.0362 | 0.0310 | -0.0069 | -0.2363 | lower in machine |
| `pos_ratios.PART` | 97 | 0.0165 | 0.0106 | 98 | 0.0198 | 0.0130 | 0.0033 | 0.2791 | higher in machine |
| `pos_ratios.PRON` | 110 | 0.0471 | 0.0194 | 110 | 0.0531 | 0.0232 | 0.0060 | 0.2793 | higher in machine |
| `pos_ratios.PROPN` | 110 | 0.2129 | 0.0875 | 108 | 0.1650 | 0.0901 | -0.0479 | -0.5390 | lower in machine |
| `pos_ratios.PUNCT` | 1 | 0.0040 | n/a | 3 | 0.0057 | 0.0023 | 0.0017 | n/a | higher in machine |
| `pos_ratios.SCONJ` | 85 | 0.0108 | 0.0064 | 91 | 0.0111 | 0.0091 | 0.0003 | 0.0360 | higher in machine |
| `pos_ratios.SYM` | 3 | 0.0093 | 0.0040 | 5 | 0.0052 | 0.0015 | -0.0041 | -1.5723 | lower in machine |
| `pos_ratios.VERB` | 110 | 0.0878 | 0.0243 | 110 | 0.0989 | 0.0316 | 0.0111 | 0.3919 | higher in machine |
| `pos_ratios.X` | 9 | 0.0059 | 0.0025 | 20 | 0.0105 | 0.0142 | 0.0046 | 0.3852 | higher in machine |
| `punct_density` | 110 | 0.0268 | 0.0069 | 110 | 0.0289 | 0.0214 | 0.0021 | 0.1330 | higher in machine |
| `semicolon_ratio` | 110 | 0.0097 | 0.0191 | 110 | 0.0040 | 0.0121 | -0.0057 | -0.3556 | lower in machine |
| `sentence_len_std` | 110 | 9.8993 | 3.9359 | 110 | 10.2359 | 14.0200 | 0.3366 | 0.0327 | higher in machine |
| `smog` | 110 | 13.7582 | 2.1523 | 110 | 14.0927 | 3.0775 | 0.3345 | 0.1260 | higher in machine |
| `subordinate_clause_ratio` | 110 | 0.8666 | 0.4499 | 110 | 0.9819 | 1.0623 | 0.1153 | 0.1413 | higher in machine |
| `ttr` | 110 | 0.5788 | 0.0561 | 110 | 0.5210 | 0.1650 | -0.0577 | -0.4684 | lower in machine |
| `word_len_std` | 110 | 2.6561 | 0.2908 | 110 | 2.8186 | 1.0174 | 0.1625 | 0.2171 | higher in machine |
| `zipf_slope` | 110 | -0.5740 | 0.0629 | 110 | -0.6272 | 0.2042 | -0.0532 | -0.3518 | lower in machine |

### Stylometric z-scores against the baseline

| feature | human n | human mean | human std | machine n | machine mean | machine std | mean diff (machine - human) | Cohen's d | direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `avg_sentence_len` | 110 | 0.5530 | 0.5252 | 110 | 0.5063 | 1.1459 | -0.0467 | -0.0524 | lower in machine |
| `avg_word_len` | 110 | 0.2323 | 0.5764 | 110 | 0.3109 | 1.0101 | 0.0786 | 0.0956 | higher in machine |
| `comma_ratio` | 110 | -0.7295 | 1.2866 | 110 | -0.5827 | 1.7036 | 0.1468 | 0.0973 | higher in machine |
| `function_word_ratio` | 110 | -1.2489 | 0.7615 | 110 | -1.0325 | 1.3388 | 0.2165 | 0.1987 | higher in machine |
| `hapax_legomena_rate` | 110 | 4.2945 | 0.9387 | 110 | 3.7475 | 2.6007 | -0.5470 | -0.2798 | lower in machine |
| `pos_adj` | 110 | -0.7185 | 1.5795 | 110 | -0.5615 | 1.6530 | 0.1571 | 0.0972 | higher in machine |
| `pos_adp` | 110 | 1.1305 | 1.3233 | 110 | 0.9454 | 1.5458 | -0.1851 | -0.1286 | lower in machine |
| `pos_adv` | 108 | -1.6730 | 0.6850 | 105 | -1.6222 | 0.9922 | 0.0508 | 0.0597 | higher in machine |
| `pos_det` | 110 | -0.7425 | 1.2933 | 110 | -0.5204 | 1.8833 | 0.2222 | 0.1375 | higher in machine |
| `pos_noun` | 110 | -1.2772 | 1.3873 | 110 | -0.6537 | 1.7754 | 0.6235 | 0.3913 | higher in machine |
| `pos_num` | 109 | 2.3171 | 2.7382 | 105 | 1.6266 | 3.1043 | -0.6905 | -0.2362 | lower in machine |
| `pos_part` | 97 | -0.3545 | 1.0678 | 98 | -0.0252 | 1.3001 | 0.3293 | 0.2767 | higher in machine |
| `pos_pron` | 110 | -1.1455 | 0.9694 | 110 | -0.8489 | 1.1595 | 0.2966 | 0.2776 | higher in machine |
| `pos_verb` | 110 | -2.4066 | 0.8104 | 110 | -2.0386 | 1.0523 | 0.3680 | 0.3919 | higher in machine |
| `punct_density` | 110 | -3.7750 | 0.2285 | 110 | -3.7021 | 0.7137 | 0.0729 | 0.1376 | higher in machine |
| `sentence_len_std` | 110 | 1.1664 | 1.6401 | 110 | 1.3066 | 5.8414 | 0.1403 | 0.0327 | higher in machine |
| `ttr` | 110 | 0.7343 | 0.7020 | 110 | 0.0128 | 2.0627 | -0.7215 | -0.4683 | lower in machine |

### Stylometric per-document measurements

`confidence_score` is the number the shipped boolean thresholds on; `char_ngram_similarity` is the whole-profile cosine similarity to the baseline.

| feature | human n | human mean | human std | machine n | machine mean | machine std | mean diff (machine - human) | Cohen's d | direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `confidence_score` | 110 | 0.3027 | 0.1383 | 110 | 0.2959 | 0.2438 | -0.0068 | -0.0344 | lower in machine |

Reported as null on some scored documents, and excluded from those rows rather than counted as zero: `char_ngram_similarity` (220 of 220).

### Perplexity measurements

| feature | human n | human mean | human std | machine n | machine mean | machine std | mean diff (machine - human) | Cohen's d | direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `doc_burstiness` | 110 | 99.3877 | 155.9039 | 110 | 837.1460 | 6315.2944 | 737.7583 | 0.1652 | higher in machine |
| `doc_ppl` | 110 | 96.1480 | 68.8230 | 110 | 333.8663 | 2464.6174 | 237.7183 | 0.1364 | higher in machine |
