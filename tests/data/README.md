# Test fixtures

`tiny_benchmark_corpus.jsonl` is a six-document slice of
`benchmarks/corpora/raid_wiki_v1/corpus.jsonl`, copied unmodified: pairs 0001,
0002, and 0039. It exists so `tests/test_benchmark_runner.py` can drive the real
benchmark runner over genuine labeled prose in under a second.

Pair 0039 is included deliberately - its machine document is one the detector
flags. Without a flagged document in the fixture, a runner that hardcoded
`high_ai_probability: false` would satisfy every assertion in that test.

It carries the same third-party licensing as the corpus it came from - see
`benchmarks/corpora/raid_wiki_v1/LICENSE-AND-ATTRIBUTION.md`.
