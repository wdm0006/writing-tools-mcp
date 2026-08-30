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

Direction is the sign of the machine-minus-human difference, reported separately for the mean and the median. Several of these features are heavy-tailed, so the two can disagree - where they do, the median describes the typical document and the mean is being carried by a few extreme ones. Cohen's d is the pooled-standard-deviation effect size and inherits the mean's sensitivity to those outliers; values near zero mean the feature does not separate the two classes in this corpus.

| feature | human n | human mean | human median | human std | machine n | machine mean | machine median | machine std | mean diff | median diff | Cohen's d | direction (mean) | direction (median) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `ari` | 110 | 13.2573 | 13.0000 | 2.8142 | 110 | 13.8491 | 12.6000 | 5.9667 | 0.5918 | -0.4000 | 0.1269 | higher in machine | lower in machine |
| `avg_sentence_len` | 110 | 22.3358 | 21.7950 | 4.3081 | 110 | 21.9537 | 20.1220 | 9.3976 | -0.3821 | -1.6730 | -0.0523 | lower in machine | lower in machine |
| `avg_word_len` | 110 | 4.9395 | 4.9220 | 0.3459 | 110 | 4.9863 | 4.9060 | 0.6058 | 0.0468 | -0.0160 | 0.0949 | higher in machine | lower in machine |
| `booster_rate` | 110 | 0.0007 | 0.0000 | 0.0028 | 110 | 0.0004 | 0.0000 | 0.0014 | -0.0003 | 0.0000 | -0.1582 | lower in machine | no difference |
| `coleman_liau` | 110 | 11.8075 | 11.6050 | 2.1295 | 110 | 12.0540 | 11.4900 | 3.6506 | 0.2465 | -0.1150 | 0.0825 | higher in machine | lower in machine |
| `comma_ratio` | 110 | 0.3615 | 0.3590 | 0.1030 | 110 | 0.3733 | 0.3940 | 0.1363 | 0.0118 | 0.0350 | 0.0978 | higher in machine | higher in machine |
| `dale_chall` | 110 | 9.9195 | 9.9300 | 0.9084 | 110 | 9.4131 | 9.3750 | 2.2110 | -0.5065 | -0.5550 | -0.2996 | lower in machine | lower in machine |
| `ellipsis_ratio` | 110 | 0.0002 | 0.0000 | 0.0025 | 110 | 0.0017 | 0.0000 | 0.0093 | 0.0015 | 0.0000 | 0.2170 | higher in machine | no difference |
| `em_dash_ratio` | 110 | 0.0019 | 0.0000 | 0.0106 | 110 | 0.0019 | 0.0000 | 0.0099 | 0.0001 | 0.0000 | 0.0062 | higher in machine | no difference |
| `exclamation_ratio` | 110 | 0.0000 | 0.0000 | 0.0000 | 110 | 0.0021 | 0.0000 | 0.0105 | 0.0021 | 0.0000 | 0.2762 | higher in machine | no difference |
| `fog` | 110 | 12.9971 | 12.7450 | 2.4734 | 110 | 13.2960 | 12.5750 | 4.7778 | 0.2989 | -0.1700 | 0.0786 | higher in machine | lower in machine |
| `fourgram_repetition_rate` | 110 | 0.0108 | 0.0040 | 0.0190 | 110 | 0.0789 | 0.0165 | 0.1829 | 0.0681 | 0.0125 | 0.5236 | higher in machine | higher in machine |
| `function_word_ratio` | 110 | 0.3875 | 0.3885 | 0.0382 | 110 | 0.3984 | 0.4080 | 0.0669 | 0.0109 | 0.0195 | 0.1997 | higher in machine | higher in machine |
| `hapax_legomena_rate` | 110 | 0.7276 | 0.7330 | 0.0564 | 110 | 0.6948 | 0.7110 | 0.1561 | -0.0328 | -0.0220 | -0.2793 | lower in machine | lower in machine |
| `hedge_rate` | 110 | 0.0022 | 0.0000 | 0.0043 | 110 | 0.0038 | 0.0000 | 0.0053 | 0.0015 | 0.0000 | 0.3211 | higher in machine | no difference |
| `kincaid` | 110 | 12.0582 | 12.0500 | 2.5037 | 110 | 12.6236 | 11.7500 | 4.5119 | 0.5655 | -0.3000 | 0.1550 | higher in machine | lower in machine |
| `lexical_density` | 110 | 0.5713 | 0.5750 | 0.0409 | 110 | 0.5598 | 0.5585 | 0.0667 | -0.0115 | -0.0165 | -0.2071 | lower in machine | lower in machine |
| `mattr` | 110 | 0.7796 | 0.7810 | 0.0431 | 110 | 0.7669 | 0.7850 | 0.1188 | -0.0127 | 0.0040 | -0.1420 | lower in machine | higher in machine |
| `mean_dependency_distance` | 110 | 2.5130 | 2.4845 | 0.3247 | 110 | 2.5372 | 2.2805 | 1.6284 | 0.0241 | -0.2040 | 0.0206 | higher in machine | lower in machine |
| `mean_word_frequency` | 110 | 5.4253 | 5.4790 | 0.2232 | 110 | 5.5039 | 5.5570 | 0.3177 | 0.0786 | 0.0780 | 0.2863 | higher in machine | higher in machine |
| `mtld` | 110 | 74.6848 | 69.0990 | 24.5181 | 110 | 213.7973 | 72.5595 | 860.2212 | 139.1126 | 3.4605 | 0.2286 | higher in machine | higher in machine |
| `mtld_lemma` | 110 | 68.2736 | 65.7120 | 22.0338 | 110 | 110.7145 | 63.9620 | 238.9708 | 42.4409 | -1.7500 | 0.2501 | higher in machine | lower in machine |
| `parenthetical_rate` | 110 | 0.2382 | 0.1670 | 0.2252 | 110 | 0.1948 | 0.0770 | 0.4874 | -0.0434 | -0.0900 | -0.1143 | lower in machine | lower in machine |
| `pos_ratios.ADJ` | 110 | 0.0656 | 0.0610 | 0.0316 | 110 | 0.0688 | 0.0615 | 0.0330 | 0.0031 | 0.0005 | 0.0973 | higher in machine | higher in machine |
| `pos_ratios.ADP` | 110 | 0.1427 | 0.1415 | 0.0265 | 110 | 0.1389 | 0.1370 | 0.0309 | -0.0037 | -0.0045 | -0.1300 | lower in machine | lower in machine |
| `pos_ratios.ADV` | 108 | 0.0265 | 0.0250 | 0.0137 | 105 | 0.0276 | 0.0230 | 0.0199 | 0.0011 | -0.0020 | 0.0624 | higher in machine | lower in machine |
| `pos_ratios.AUX` | 110 | 0.0387 | 0.0360 | 0.0173 | 110 | 0.0458 | 0.0440 | 0.0197 | 0.0071 | 0.0080 | 0.3843 | higher in machine | higher in machine |
| `pos_ratios.CCONJ` | 110 | 0.0386 | 0.0375 | 0.0160 | 110 | 0.0389 | 0.0380 | 0.0167 | 0.0003 | 0.0005 | 0.0167 | higher in machine | higher in machine |
| `pos_ratios.DET` | 110 | 0.0952 | 0.0950 | 0.0259 | 110 | 0.0996 | 0.0965 | 0.0377 | 0.0045 | 0.0015 | 0.1378 | higher in machine | higher in machine |
| `pos_ratios.INTJ` | 2 | 0.0055 | 0.0055 | 0.0007 | 7 | 0.0036 | 0.0030 | 0.0011 | -0.0019 | -0.0025 | -1.7803 | lower in machine | lower in machine |
| `pos_ratios.NOUN` | 110 | 0.1789 | 0.1695 | 0.0555 | 110 | 0.2039 | 0.2080 | 0.0710 | 0.0250 | 0.0385 | 0.3916 | higher in machine | higher in machine |
| `pos_ratios.NUM` | 109 | 0.0431 | 0.0390 | 0.0274 | 105 | 0.0362 | 0.0280 | 0.0310 | -0.0069 | -0.0110 | -0.2363 | lower in machine | lower in machine |
| `pos_ratios.PART` | 97 | 0.0165 | 0.0150 | 0.0106 | 98 | 0.0198 | 0.0160 | 0.0130 | 0.0033 | 0.0010 | 0.2791 | higher in machine | higher in machine |
| `pos_ratios.PRON` | 110 | 0.0471 | 0.0455 | 0.0194 | 110 | 0.0531 | 0.0515 | 0.0232 | 0.0060 | 0.0060 | 0.2793 | higher in machine | higher in machine |
| `pos_ratios.PROPN` | 110 | 0.2129 | 0.2155 | 0.0875 | 108 | 0.1650 | 0.1540 | 0.0901 | -0.0479 | -0.0615 | -0.5390 | lower in machine | lower in machine |
| `pos_ratios.PUNCT` | 1 | 0.0040 | 0.0040 | n/a | 3 | 0.0057 | 0.0070 | 0.0023 | 0.0017 | 0.0030 | n/a | higher in machine | higher in machine |
| `pos_ratios.SCONJ` | 85 | 0.0108 | 0.0100 | 0.0064 | 91 | 0.0111 | 0.0080 | 0.0091 | 0.0003 | -0.0020 | 0.0360 | higher in machine | lower in machine |
| `pos_ratios.SYM` | 3 | 0.0093 | 0.0100 | 0.0040 | 5 | 0.0052 | 0.0050 | 0.0015 | -0.0041 | -0.0050 | -1.5723 | lower in machine | lower in machine |
| `pos_ratios.VERB` | 110 | 0.0878 | 0.0885 | 0.0243 | 110 | 0.0989 | 0.1025 | 0.0316 | 0.0111 | 0.0140 | 0.3919 | higher in machine | higher in machine |
| `pos_ratios.X` | 9 | 0.0059 | 0.0060 | 0.0025 | 20 | 0.0105 | 0.0045 | 0.0142 | 0.0046 | -0.0015 | 0.3852 | higher in machine | lower in machine |
| `punct_density` | 110 | 0.0268 | 0.0260 | 0.0069 | 110 | 0.0289 | 0.0250 | 0.0214 | 0.0021 | -0.0010 | 0.1330 | higher in machine | lower in machine |
| `semicolon_ratio` | 110 | 0.0097 | 0.0000 | 0.0191 | 110 | 0.0040 | 0.0000 | 0.0121 | -0.0057 | 0.0000 | -0.3556 | lower in machine | no difference |
| `sentence_len_std` | 110 | 9.8993 | 9.2270 | 3.9359 | 110 | 10.2359 | 6.9485 | 14.0200 | 0.3366 | -2.2785 | 0.0327 | higher in machine | lower in machine |
| `smog` | 110 | 13.7582 | 13.5500 | 2.1523 | 110 | 14.0927 | 13.8500 | 3.0775 | 0.3345 | 0.3000 | 0.1260 | higher in machine | higher in machine |
| `subordinate_clause_ratio` | 110 | 0.8666 | 0.7780 | 0.4499 | 110 | 0.9819 | 0.7350 | 1.0623 | 0.1153 | -0.0430 | 0.1413 | higher in machine | lower in machine |
| `ttr` | 110 | 0.5788 | 0.5780 | 0.0561 | 110 | 0.5210 | 0.5090 | 0.1650 | -0.0577 | -0.0690 | -0.4684 | lower in machine | lower in machine |
| `word_len_std` | 110 | 2.6561 | 2.6100 | 0.2908 | 110 | 2.8186 | 2.6810 | 1.0174 | 0.1625 | 0.0710 | 0.2171 | higher in machine | higher in machine |
| `zipf_slope` | 110 | -0.5740 | -0.5740 | 0.0629 | 110 | -0.6272 | -0.6460 | 0.2042 | -0.0532 | -0.0720 | -0.3518 | lower in machine | lower in machine |

### Stylometric z-scores against the baseline

| feature | human n | human mean | human median | human std | machine n | machine mean | machine median | machine std | mean diff | median diff | Cohen's d | direction (mean) | direction (median) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `avg_sentence_len` | 110 | 0.5530 | 0.4900 | 0.5252 | 110 | 0.5063 | 0.2800 | 1.1459 | -0.0467 | -0.2100 | -0.0524 | lower in machine | lower in machine |
| `avg_word_len` | 110 | 0.2323 | 0.2050 | 0.5764 | 110 | 0.3109 | 0.1750 | 1.0101 | 0.0786 | -0.0300 | 0.0956 | higher in machine | lower in machine |
| `comma_ratio` | 110 | -0.7295 | -0.7600 | 1.2866 | 110 | -0.5827 | -0.3300 | 1.7036 | 0.1468 | 0.4300 | 0.0973 | higher in machine | higher in machine |
| `function_word_ratio` | 110 | -1.2489 | -1.2350 | 0.7615 | 110 | -1.0325 | -0.8400 | 1.3388 | 0.2165 | 0.3950 | 0.1987 | higher in machine | higher in machine |
| `hapax_legomena_rate` | 110 | 4.2945 | 4.3850 | 0.9387 | 110 | 3.7475 | 4.0200 | 2.6007 | -0.5470 | -0.3650 | -0.2798 | lower in machine | lower in machine |
| `pos_adj` | 110 | -0.7185 | -0.9400 | 1.5795 | 110 | -0.5615 | -0.9250 | 1.6530 | 0.1571 | 0.0150 | 0.0972 | higher in machine | higher in machine |
| `pos_adp` | 110 | 1.1305 | 1.0850 | 1.3233 | 110 | 0.9454 | 0.8550 | 1.5458 | -0.1851 | -0.2300 | -0.1286 | lower in machine | lower in machine |
| `pos_adv` | 108 | -1.6730 | -1.7650 | 0.6850 | 105 | -1.6222 | -1.8600 | 0.9922 | 0.0508 | -0.0950 | 0.0597 | higher in machine | lower in machine |
| `pos_det` | 110 | -0.7425 | -0.7500 | 1.2933 | 110 | -0.5204 | -0.6800 | 1.8833 | 0.2222 | 0.0700 | 0.1375 | higher in machine | higher in machine |
| `pos_noun` | 110 | -1.2772 | -1.5250 | 1.3873 | 110 | -0.6537 | -0.5450 | 1.7754 | 0.6235 | 0.9800 | 0.3913 | higher in machine | higher in machine |
| `pos_num` | 109 | 2.3171 | 1.9000 | 2.7382 | 105 | 1.6266 | 0.8500 | 3.1043 | -0.6905 | -1.0500 | -0.2362 | lower in machine | lower in machine |
| `pos_part` | 97 | -0.3545 | -0.5400 | 1.0678 | 98 | -0.0252 | -0.3800 | 1.3001 | 0.3293 | 0.1600 | 0.2767 | higher in machine | higher in machine |
| `pos_pron` | 110 | -1.1455 | -1.2250 | 0.9694 | 110 | -0.8489 | -0.9350 | 1.1595 | 0.2966 | 0.2900 | 0.2776 | higher in machine | higher in machine |
| `pos_verb` | 110 | -2.4066 | -2.3700 | 0.8104 | 110 | -2.0386 | -1.9250 | 1.0523 | 0.3680 | 0.4450 | 0.3919 | higher in machine | higher in machine |
| `punct_density` | 110 | -3.7750 | -3.7950 | 0.2285 | 110 | -3.7021 | -3.8200 | 0.7137 | 0.0729 | -0.0250 | 0.1376 | higher in machine | lower in machine |
| `sentence_len_std` | 110 | 1.1664 | 0.8850 | 1.6401 | 110 | 1.3066 | -0.0650 | 5.8414 | 0.1403 | -0.9500 | 0.0327 | higher in machine | lower in machine |
| `ttr` | 110 | 0.7343 | 0.7200 | 0.7020 | 110 | 0.0128 | -0.1400 | 2.0627 | -0.7215 | -0.8600 | -0.4683 | lower in machine | lower in machine |

### Stylometric per-document measurements

`confidence_score` is the number the shipped boolean thresholds on; `char_ngram_similarity` is the whole-profile cosine similarity to the baseline.

| feature | human n | human mean | human median | human std | machine n | machine mean | machine median | machine std | mean diff | median diff | Cohen's d | direction (mean) | direction (median) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `confidence_score` | 110 | 0.3027 | 0.3000 | 0.1383 | 110 | 0.2959 | 0.2000 | 0.2438 | -0.0068 | -0.1000 | -0.0344 | lower in machine | lower in machine |

Reported as null on some scored documents, and excluded from those rows rather than counted as zero: `char_ngram_similarity` (220 of 220).

### Perplexity measurements

| feature | human n | human mean | human median | human std | machine n | machine mean | machine median | machine std | mean diff | median diff | Cohen's d | direction (mean) | direction (median) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `doc_burstiness` | 110 | 99.3877 | 58.3100 | 155.9039 | 110 | 837.1460 | 24.5450 | 6315.2944 | 737.7583 | -33.7650 | 0.1652 | higher in machine | lower in machine |
| `doc_ppl` | 110 | 96.1480 | 80.1100 | 68.8230 | 110 | 333.8663 | 42.5000 | 2464.6174 | 237.7183 | -37.6100 | 0.1364 | higher in machine | lower in machine |
