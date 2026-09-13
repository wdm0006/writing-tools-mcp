# Preference-Scoring Foundation — Evaluation Report (W9)

**Question.** The stylometric-delta tool (W4) reports, per statistic, whether a revised text moved
closer to a baseline's typical range. If editing reliably moves prose *toward* the baseline, that
direction could power a "writing quality" preference score. This evaluation measures how often the
delta direction actually agrees with the edited variant on real public-domain edit pairs.

**Answer.** It doesn't. The delta direction agrees with the edited variant at coin-flip rates
(pair level 41.7%, 95% CI [15%, 72%]; verdict level 47.6%, 95% CI [41%, 55%]). **Recommendation:
no-go** on building a production preference score from this signal in its current form. The
evidence, method, and what would reverse this call are below.

This work item shipped an **evaluation and a decision** — deliberately not a product tool. No
`writing_quality` MCP tool was added.

---

## 1. Method

1. **Corpus.** 13 excerpt pairs from four public-domain works where an earlier draft or edition can
   be paired with a later, revised one (§2). Each side is cut to the same sentence count (12) so
   text length does not confound the comparison.
2. **Delta machinery.** For each pair, `AIDetectionAnalyzer.stylometric_delta(unedited, edited,
   baseline="brown_corpus")` (the W4 implementation, unchanged) profiles both excerpts against the
   same committed baseline and classifies each of up to 17 statistics:
   - `improved` — the edited text's z-score moved closer to 0 (toward the baseline's typical range);
   - `regressed` — it moved further away;
   - `unchanged` — no movement at the reported precision.
3. **Pair preference.** Majority vote among *decided* statistics (improved/regressed): `edited`,
   `unedited`, or `split` on a tie. Unchanged statistics do not vote.
4. **Agreement views.**
   - *Pair level* — the fraction of decided pairs whose majority prefers the edited text. This is
     the headline number.
   - *Verdict level* — all decided (pair, statistic) observations pooled. This shows whether
     individual statistic movements track the edit direction even when pair-level votes are noisy,
     and which statistics carry any signal.
5. **Uncertainty.** Exact Clopper–Pearson 95% intervals, computed with a pure-`math` bisection
   cross-verified against `scipy.stats.beta.ppf` to 2.2e-16 on eight cases including the
   degenerate k=0 and k=n endpoints.

Everything is deterministic: committed corpus, committed baseline statistics, spaCy 3.8.5 with a
pinned model, no GPU, no network at evaluation or test time. `uv run python -m
evals.preference.evaluate` regenerates `evals/preference/results.json` byte-identically; a test
pins `run_evaluation()` to the committed file.

## 2. Data

| Work | Stratum | Pairs | Earlier → Later |
|---|---|---|---|
| Frankenstein | edition_revision | 4 | 1818 edition → 1831 revised edition |
| On the Origin of Species | edition_revision | 3 | 1st (1859) → 6th (1872) edition |
| Alice's Adventures | manuscript_to_publication | 3 | 1864 "Under Ground" fair copy → 1865 published edition |
| The Picture of Dorian Gray | edition_revision | 3 | 1890 Lippincott's Magazine → 1891 book edition |

All texts are public domain, fetched from Project Gutenberg and Wikisource (canonical hosts only).
Provenance is committed in three layers:

- `evals/preference/corpus.jsonl` — the excerpt pairs, each side carrying label, year, license,
  and source URL;
- `evals/preference/extraction.json` — per-excerpt extraction record: anchor phrase, character
  offset, sentence count, word count, and per-source sha256 digests of the downloaded documents;
- `evals/preference/corpus_sources.md` — human-readable source list and licenses.

`evals/preference/build_corpus.py` re-derives `corpus.jsonl` and `extraction.json` from those
sources deterministically; `--check` verifies byte-identical re-derivation against the committed
files (network required, so it is a maintainer tool, not a test).

**Known caveats in the data.** These are author-revised or editor-revised texts, not quality-graded
human judgments: "edited" is a *proxy* for "better writing". The excerpts are 19th-century literary
and scientific prose; the `brown_corpus` baseline is 20th-century expository American English —
a deliberate mismatch revisited in §5.

## 3. Results

219 (pair, statistic) observations were recorded: 206 decided (98 improved, 108 regressed), 13
unchanged, and 2 absent (`pos_num` is undefined for two excerpts containing no numerals).

**Pair level.** Of 12 decided pairs (1 split: Frankenstein "creation night", an 8–8 tie), **5
(41.7%) prefer the edited text**; 95% CI **[15.2%, 72.3%]**. A two-sided binomial test against a
fair coin gives p = 0.774 — this sample cannot even distinguish the signal from chance, and the
point estimate sits *below* 50%.

**Verdict level.** Of 206 decided statistic movements, **98 (47.6%) were improvements**; 95% CI
**[40.6%, 54.6%]**. This interval is tight enough to say the underlying rate is within about ±7
points of a coin flip for this operationalization.

**Statistic level** (agreement = improved share of decided observations):

| Statistic | improved | regressed | unchanged | agreement |
|---|---|---|---|---|
| pos_adp | 9 | 4 | 0 | 0.692 |
| comma_ratio | 8 | 5 | 0 | 0.615 |
| ttr | 8 | 5 | 0 | 0.615 |
| avg_word_len | 6 | 4 | 3 | 0.600 |
| punct_density | 7 | 6 | 0 | 0.538 |
| pos_det / pos_pron / pos_verb | 6 | 6 | 1 | 0.500 |
| avg_sentence_len | 6 | 7 | 0 | 0.462 |
| hapax_legomena_rate | 6 | 7 | 0 | 0.462 |
| pos_noun | 6 | 7 | 0 | 0.462 |
| pos_part | 5 | 6 | 2 | 0.455 |
| sentence_len_std | 5 | 8 | 0 | 0.385 |
| function_word_ratio | 4 | 7 | 2 | 0.364 |
| pos_adj | 4 | 8 | 1 | 0.333 |
| pos_num | 3 | 7 | 1 | 0.300 |
| pos_adv | 3 | 9 | 1 | 0.250 |

No statistic clears 0.7; every interval is wide at n ≤ 13. The most encouraging candidates
(preposition and comma behavior, lexical diversity) are directionally interesting but not
significant at this sample size, and selecting them *after* seeing results would be multiple
comparisons — they are noted as hypotheses, not findings.

A secondary observation: edited excerpts sit marginally closer to the baseline *on average*
(mean |z| 1.794 vs 1.821) — a slight global pull toward the baseline that does not determine the
per-statistic direction majorities the preference rule depends on.

Per stratum: edition_revision pairs prefer the edited text 4/10 (1 split); manuscript_to_publication
1/3. Both strata point the same way — nowhere.

## 4. Limitations

- **Small n.** 13 pairs (12 decided). The pair-level CI spans 57 points; the study has essentially
  no power to detect a modest true effect. This is a foundation-building probe, not a powered
  evaluation.
- **Provenance proxy.** "Edited" means "later author/editor revision", not a human quality
  judgment. Genuine editing quality varies and is not uniformly measurable by stylometry.
- **Baseline mismatch.** The brown corpus measures 20th-century expository prose against
  19th-century literary/scientific texts. Both sides of each pair share this mismatch, so the
  *direction* comparison is partially protected — but if neither side sits in a well-modeled
  region, z-scores are noisy coordinates.
- **Direction-only reduction.** The majority vote discards magnitude; a statistic that barely
  moves votes with the same weight as one that jumps.
- **One reduction rule, evaluated post hoc.** Majority voting was chosen before running, but the
  space of reasonable reductions is large and unexplored; the null here binds this rule on this
  corpus, not all preference signals.
- **Corpus composition.** Four works, all canonical published literature; drafting-stage text
  (true "before editing" drafts) is represented only by Alice's manuscript.

## 5. Recommendation and what would reverse it

**No-go** on building a production preference score from stylometric delta direction. The signal
that would justify the product — "revisions move prose toward a reference distribution" — measured
at chance on real edit pairs. A production score would be a confident interface over a coin flip,
and the verdict-level interval (the tightest bound we have) makes "it's a small-n artifact" hard
to sustain for anything but a weak effect: the true rate is unlikely to exceed ~55%.

Cheap probes that could reverse the call, in order of cost:

1. **Domain-matched baselines** (now available from W8: `essays`, `technical_docs`,
   `scientific_prose`). Re-running this pipeline against a matched baseline is nearly free with the
   committed harness. If agreement jumps, the brown mismatch was the problem; if it doesn't, the
   hypothesis itself is what failed.
2. **A larger, quality-graded corpus** (30+ pairs, ideally with human quality labels rather than
   edition provenance). Reversal bar: pair-level agreement ≥ 0.65 with a 95% CI lower bound above
   0.5.
3. **A pre-registered magnitude-aware reduction** (e.g., |delta|-weighted vote) declared before
   evaluating on a *fresh* pair set — post-hoc rule search on this corpus would be overfitting.
4. **Per-stratification analysis** at scale, to check whether the 0.692–0.615 candidates become
   reliable within strata rather than pooled.

Failure of probes 1–2 on adequate data closes the question permanently: the stylometric-proximity
hypothesis for writing quality would be falsified in this system, and the W4 delta machinery
remains valuable for what it was built for — measuring revision drift, not scoring quality.

## 6. Reproducing

```bash
uv sync --all-extras                              # locked environment (spaCy model included)
uv run python -m evals.preference.evaluate        # regenerate results.json (offline, deterministic)
uv run pytest tests/test_preference_corpus.py \
              tests/test_preference_evaluate.py -v  # 36 tests, incl. determinism pin
cd evals/preference && uv run python build_corpus.py --check   # maintainer: needs network
```
