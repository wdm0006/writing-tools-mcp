# writing-tools-mcp — Agent Guide

MCP (Model Context Protocol) server exposing 13 text-analysis tools: counts, spellcheck,
readability, keyword analysis, passive voice, GPT-2 perplexity, and stylometric AI-detection
against built-in or custom baselines. Speaks **stdio** — there is no port and no HTTP server;
stdout carries the JSON-RPC protocol stream, all logs go to stderr. Owner: wdm0006. License: MIT.

## Stack

| Layer | Choice |
| --- | --- |
| Language | Python >= 3.10 (CI matrix: 3.10 / 3.11 / 3.12 — 3.13 does not satisfy the lock, see Gotchas) |
| Package manager | uv, hatchling build backend; `uv.lock` is tracked in git |
| Server framework | FastMCP 3.4.5, stdio transport |
| NLP | spaCy 3.8.5 + `en_core_web_sm` 3.8.0 (direct GitHub-release wheel dependency) |
| AI detection | transformers 4.52.3 + torch 2.7.0 (GPT-2; weights download from HuggingFace on first `perplexity_analysis` call) |
| Other runtime deps | textstat, pyspellchecker, wordfreq, markdown-it-py, PyYAML, numpy |
| Dev tooling | ruff (lint + format), pytest 8.3.5, pre-commit |
| Services | None. No database, no Docker/Compose, no required env vars, no secrets. |

## Commands

Install (locked, reproducible — preferred):

```bash
uv sync --all-extras   # README's `uv sync` + the dev extra (ruff, pytest)
```

Install (Makefile / CI path — unlocked latest versions, fresh checkout only, see Gotchas):

```bash
make install           # uv venv .venv --seed && uv pip install -e ".[dev]"
```

Quality gates (exactly what CI runs):

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ -v
```

Run the server (stdio; logs on stderr; exits cleanly on stdin EOF):

```bash
uv run run_server.py       # canonical launcher; PEP-723 inline deps — see Gotchas
uv run writing-tools-mcp   # console script, equivalent entry point (server.app:main)
```

Extras:

```bash
make build-mcpb                             # build writing-tools-mcp.mcpb Claude Desktop bundle
uv run scripts/build_baseline.py NAME DIR/  # build a custom stylometry baseline from *.txt files
uv run benchmarks/run_benchmark.py          # offline detector benchmark; rewrites benchmarks/reports/
```

## Codebase Map

See `codebase-map.md` for the folder-level table.

## Configuration

`.mcp-config.yaml` (optional, read from the server's working directory): `perplexity`
(model, thresholds), `stylometry` (baseline, thresholds, features), `logging` (level, format).
Unknown keys and wrongly typed values warn on stderr and fall back to defaults — a bad config
never stops the server from starting.

## Local Verification Summary

Verified 2026-09-06 by the onboarding run (sandbox `cmp_zFBvSt5g`):

- `uv run ruff check .` — **All checks passed**
- `uv run ruff format --check .` — **60 files already formatted**
- `uv run pytest tests/ -v` — **401 passed**, 1 warning (textstat `pkg_resources` deprecation), 14s
- MCP end-to-end via fastmcp `Client` over stdio to `uv run run_server.py`:
  - handshake + `tools/list` → **13 tools** discovered
  - `word_count` → 17 (correct for the sample)
  - `spellcheck` → flagged all 3 planted misspellings
  - `readability_score` → flesch 77.7 / kincaid 4.33 / fog 3.07
  - `passive_voice_detection` → returned exactly the 2 passive sentences, excluded the active one
  - `stylometric_analysis` → features + z-scores against the `brown_corpus` baseline
  - `perplexity_analysis` → real GPT-2 inference: doc_ppl 610.0, burstiness 604.6, per-sentence scores, AI flags
  - server exited 0 on stdin EOF; spaCy/GPT-2 memory released after each tool call

Reproduce the smoke test with the recipe in `skills/local-dev/SKILL.md`.

## Sandbox Snapshot

- Template: `9ppe91fnmj806lonn8gs:default` (E2B), captured 2026-09-06T20:27:10.675Z
- Contents: repo @ `main` (fc5028e); uv 0.12.10 in `~/.local/bin`; `.venv` on Python 3.12.14 with
  the locked dependency set (`uv sync --all-extras`), `en_core_web_sm` installed, GPT-2 weights
  cached under `~/.cache/huggingface`.

## Gotchas

- **`make` targets need a fresh checkout.** Every Makefile target depends on `install`, which
  reruns `uv venv .venv --seed` and errors when `.venv` already exists. After the first install,
  run the `uv run …` commands directly (that is what CI does after its one-time `uv venv`).
- **Python 3.13 does not satisfy `uv.lock`.** The lock pins spacy 3.8.5, which has no cp313
  wheel; uv falls back to a source build that fails (Cython < 3 vs NumPy headers). Build the
  venv on 3.10–3.12 (`uv venv .venv --python 3.12 --seed`; uv downloads the managed interpreter).
- **`uv run run_server.py` resolves its own environment.** The launcher carries PEP-723 inline
  metadata, so uv builds an ephemeral unlocked env for it, separate from the project venv.
  `uv run writing-tools-mcp` and `uv run python run_server.py` use the locked project venv.
- **stdout is protocol.** Never print to stdout in server code — it corrupts the MCP JSON-RPC
  stream. Logs belong on stderr.
- `uv.lock` is tracked in git even though `.gitignore` lists it — leave it in place.
