# Detector benchmark

An offline evaluation surface for this repository's AI-detection code. It is
**not** part of the MCP server: nothing under `server/` imports it, no MCP tool
schema, default, or threshold changes because of it, and running it is a
maintainer activity rather than a product feature.

The point is to replace hand-picked constants with measured numbers. The shipped
thresholds (`perplexity.thresholds.ppl_max`, `perplexity.thresholds.burstiness_min`,
`stylometry.thresholds.ai_confidence_threshold`) have never been scored against
labeled text. This is that measurement, and only that: it retunes nothing.

Detector output is statistical evidence about writing style. It is not proof of
authorship, and no number produced here should be presented as such.

## Reproduce the committed report

```
uv run benchmarks/run_benchmark.py
```

That scores `benchmarks/corpora/raid_wiki_v1/corpus.jsonl` with the real
`AIDetectionAnalyzer` and rewrites `benchmarks/reports/raid_wiki_v1/`. It needs
no network access except the first-run GPT-2 download that
`perplexity_analysis` performs anyway.

Add `--analyses stylometry` to skip GPT-2 entirely, or `--limit N` for a smoke
run. `--help` lists the rest.

## What is reproducible, and how far

| artifact | reproducible |
| --- | --- |
| `scores.jsonl` stylometry block | Byte-identical for a given corpus, spaCy model, and `textstat`/`wordfreq` versions. |
| `scores.jsonl` perplexity block | Deterministic per machine, but GPT-2 floating-point output varies across `torch` builds and CPU architectures. A perplexity landing near `ppl_max` can therefore flip a boolean on a different machine. |
| `report.md` | A pure function of `scores.jsonl`. It contains no timestamp and no version string. |
| `run_metadata.json` | Deliberately **not** reproducible - it is where the timestamp and library versions live, so the other two files can be diffed without false positives. |

To check reproducibility, run the command twice into different directories and
diff `scores.jsonl` and `report.md`.

## Layout

| path | what it is |
| --- | --- |
| `metrics.py` | Pure metric functions - confusion matrices, rates, threshold sweeps, per-feature summaries. No analyzer dependency. |
| `report.py` | Deterministic Markdown rendering of a scored run. |
| `runner.py` | Drives the repository's real `AIDetectionAnalyzer` over a corpus and records results or explicit failures. |
| `run_benchmark.py` | Command-line entry point. |
| `extract_raid_corpus.py` | One-time provenance script that built the committed corpus from RAID-train. Not needed to run the benchmark. |
| `corpora/<name>/` | A committed labeled corpus, its manifest, and its licensing. |
| `reports/<name>/` | The committed scored output for that corpus. |

## Adding a corpus

A corpus is a JSONL file with one object per line and these fields:

```json
{
  "doc_id": "stable unique id",
  "label": "human" | "machine",
  "text": "the document",
  "source": { "...provenance sufficient to trace the original row..." }
}
```

Commit a `MANIFEST.md` beside it recording the source and version, the exact
selection rule, attribution and licensing, and the limitations of the sample.
Corpus licensing is tracked per corpus and is separate from this repository's
MIT license, which covers the code only.
