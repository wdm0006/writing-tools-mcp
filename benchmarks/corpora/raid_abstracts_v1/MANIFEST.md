# `raid_abstracts_v1` corpus manifest

A second labeled corpus from a different domain (arXiv abstracts), added so
thresholds chosen on `raid_wiki_v1` can be reported on text they were not
swept on. It is a **single-domain, non-adversarial sample**, not a
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

1. **Filter rows.** Keep only rows with `domain == "abstracts"` and
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
uv run benchmarks/extract_raid_corpus.py --domain abstracts --out benchmarks/corpora/raid_abstracts_v1
```

RAID's `train.csv` is grouped by domain and `abstracts` comes first; every
`abstracts` row with `attack == "none"` lies in the first 1.6 GB, so
`curl -r 0-1599999999 <url> -o head.csv` plus `--source-csv head.csv` rebuilds the
same file. The full download is not needed, and neither is any of it to run the benchmark,
because `corpus.jsonl` is committed.

## Record format

One JSON object per line:

| field | meaning |
| --- | --- |
| `doc_id` | `raid-abstracts-<pair>-<label>`. Stable across re-extractions of the same revision. |
| `label` | `human` or `machine`. |
| `text` | The document as analyzed. |
| `source.raid_id` | The `id` of the RAID row this came from - the key to trace it back. |
| `source.raid_source_id` | RAID's `source_id`, shared by a pair. |
| `source.model` | `human`, or the generator model name. |
| `source.decoding`, `source.repetition_penalty` | RAID's generation settings; empty for human rows. |
| `source.attack`, `source.domain` | Always `none` and `abstracts` in this corpus. |
| `source.title` | The arXiv paper title (may contain a hard-wrapped newline), which is the attribution handle for human documents. |
| `source.dataset`, `source.revision`, `source.split` | Provenance repeated per record so a single line is self-describing. |

## Limitations

Read these before treating any measurement on this corpus as a property of the
detector in general.

- **One domain, and a very different one.** Every document is a scientific
  abstract: dense, formulaic, heavy in technical vocabulary. It is a
  distribution shift from `raid_wiki_v1`, not a representative population.
- **Short documents.** The 120-word floor keeps the same length window as
  `raid_wiki_v1`, but abstracts rarely exceed 300 words, so the window is
  narrower in practice: human median 169 words (range 121-292), machine median
  178.5 (range 120-432). The classes are close in length here, unlike
  `raid_wiki_v1`, so length confounds separation less.
- **Generators are 2024-era**, English only, non-adversarial only.
- **Per-model share is uneven.** Round-robin assignment skips articles whose
  assigned model has no eligible generation, so per-model counts range from 5
  (`mpt-chat`) to 13 (`gpt2`); see `extraction.json`.
- **Topic-matched, not independent.** 110 papers, each contributing one human
  and one machine document.
- **Small.** 220 documents; rates have wide intervals and differences of a few
  percentage points are noise.
- **Not an authorship test.** A label records how RAID produced a document;
  agreement or disagreement is evidence about the detector, never proof about
  any real document's authorship.
