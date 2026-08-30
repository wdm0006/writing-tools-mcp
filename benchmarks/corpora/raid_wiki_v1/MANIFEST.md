# `raid_wiki_v1` corpus manifest

An initial, deliberately narrow labeled corpus for measuring this repository's
AI-detection code. It is a **single-domain, non-adversarial sample**, not a
representative population of human or machine writing - see
[Limitations](#limitations) before quoting any number derived from it.

Licensing and attribution are in
[`LICENSE-AND-ATTRIBUTION.md`](./LICENSE-AND-ATTRIBUTION.md). This directory is
third-party text and is **not** covered by the repository's MIT license.

## Source

| field | value |
| --- | --- |
| dataset | RAID (`liamdugan/raid`) |
| split | `train` |
| revision | `865cac74188466cb0c3b7574a10204007b57a459` |
| file | `train.csv` (11,779,491,051 bytes) |
| download URL | `https://huggingface.co/datasets/liamdugan/raid/resolve/865cac74188466cb0c3b7574a10204007b57a459/train.csv` |

The revision is pinned rather than tracking `main`, so the extraction rule below
identifies exactly one input file.

## Extraction rule

Reproduced by `benchmarks/extract_raid_corpus.py`, which streams `train.csv` and
applies these steps in order. Every step is a pure function of the input file -
no sampling, no randomness, no seed.

1. **Filter rows.** Keep only rows with `domain == "wiki"` and
   `attack == "none"`. RAID's twelve adversarial attacks (homoglyph
   substitution, zero-width spaces, paraphrase, whitespace insertion, ...) are a
   separate and much harder question and are excluded here.
2. **Normalize text.** `generation` has its line endings normalized to `\n` and
   is stripped. Nothing else is changed: the analyzers see the source prose.
3. **Filter documents by length.** Keep documents whose whitespace-token count
   is between `120` and `500` inclusive. The floor keeps documents long enough
   for sentence-level statistics to mean anything; the ceiling bounds per-document
   analysis cost.
4. **Group by source article.** Each surviving row is grouped under its
   `source_id`. Where a source article has more than one human row, the
   lexicographically smallest `id` wins.
5. **Assign a generator model per article.** Source articles are walked in
   ascending `source_id` order. The article at position *i* is assigned
   `models[i % len(models)]`, where `models` is the sorted list of generator
   model names present after filtering. An article whose assigned model produced
   no eligible generation is **skipped**, not reassigned - so the assignment
   stays a pure function of position and every model contributes a near-equal
   share.
6. **Emit topic-matched pairs.** For each accepted article, emit the human
   document and the assigned model's generation with the lexicographically
   smallest `id`. Exact-duplicate text is rejected in both directions, so no
   string appears under two labels or twice under one.
7. **Stop** at 110 pairs, and sort the output by `doc_id`.

The realized counts, per-model breakdown, and a SHA-256 of the emitted file are
recorded in [`extraction.json`](./extraction.json).

To re-run it:

```
uv run benchmarks/extract_raid_corpus.py --out benchmarks/corpora/raid_wiki_v1
```

That streams the full 11.8 GB source; it is not needed to run the benchmark,
because `corpus.jsonl` is committed.

## Record format

One JSON object per line:

| field | meaning |
| --- | --- |
| `doc_id` | `raid-wiki-<pair>-<label>`. Stable across re-extractions of the same revision. |
| `label` | `human` or `machine`. |
| `text` | The document as analyzed. |
| `source.raid_id` | The `id` of the RAID row this came from - the key to trace it back. |
| `source.raid_source_id` | RAID's `source_id`, shared by a pair. |
| `source.model` | `human`, or the generator model name. |
| `source.decoding`, `source.repetition_penalty` | RAID's generation settings; empty for human rows. |
| `source.attack`, `source.domain` | Always `none` and `wiki` in this corpus. |
| `source.title` | The Wikipedia article title, which is also the attribution handle for human documents. |
| `source.dataset`, `source.revision`, `source.split` | Provenance repeated per record so a single line is self-describing. |

## Limitations

Read these before treating any measurement on this corpus as a property of the
detector in general.

- **One domain.** Every document is Wikipedia-style encyclopedic prose about a
  single named subject. Encyclopedic register is unusually uniform, which is
  exactly the kind of writing stylometric outlier detection finds hardest. A
  number measured here does not transfer to blog posts, fiction, email, or
  technical documentation.
- **One language and one genre of machine text.** English only, and only
  RAID's prompting setup for the `wiki` domain.
- **Generators are 2024-era.** RAID's generations predate 2025-2026 model
  releases. A detector that separates these may not separate current output,
  and this corpus cannot tell you which.
- **Non-adversarial only.** Nothing here has been paraphrased, homoglyphed, or
  otherwise perturbed to evade detection. Measured performance is therefore an
  upper bound on performance against a motivated evader.
- **Topic-matched, not independent.** Each machine document answers the same
  source article as its paired human document. That removes subject matter as a
  confound in the per-feature comparison, but it means the sample is 110
  articles rather than 220 independent draws, and the two classes are not
  statistically independent.
- **The two classes are not length-matched.** Within the 120-500 word window the
  filter allows, the machine documents run longer: median 314 whitespace tokens
  against 198 for the human documents. Several stylometric features respond to
  document length, so part of any measured separation is a length effect rather
  than an authorship effect. The per-feature table should be read with that in
  mind.
- **Length-filtered.** Documents outside 120-500 words are excluded, so the
  corpus says nothing about very short or very long inputs - and several of the
  detector's features are known to behave differently at those lengths.
- **Small.** 220 documents. Confidence intervals on every rate reported from it
  are wide; treat differences of a few percentage points as noise.
- **Not an authorship test.** A label here records how a document was produced
  in RAID. A detector agreeing or disagreeing with that label is evidence about
  the detector, never proof about any real document's authorship.
