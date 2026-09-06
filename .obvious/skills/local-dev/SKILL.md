---
name: local-dev
description: Bring up and verify the writing-tools-mcp dev stack — uv + Python 3.12 locked env, CI quality gates, and an MCP stdio smoke test of the analysis tools
---

# local-dev — writing-tools-mcp

Verified working 2026-09-06 by the onboarding run (sandbox `cmp_zFBvSt5g`, snapshot
`9ppe91fnmj806lonn8gs:default`). Everything below was executed, not inferred.

## What "healthy" means here

The repo is a stdio MCP server with no services and no env vars. Healthy = the locked
environment installs, the CI gates pass (ruff + 401 pytest tests), and the server answers
MCP tool calls over stdio.

## Bring-up (from a fresh sandbox)

```bash
# uv is not preinstalled — install once per machine
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Build the venv on 3.10-3.12, NOT the system 3.13 (the lock pins spacy 3.8.5,
# which has no cp313 wheel and fails to build from source).
uv venv .venv --python 3.12 --seed
uv sync --all-extras   # locked deps + dev extra (ruff, pytest); en_core_web_sm arrives from a GitHub-release wheel
```

Roughly 30 s with a warm uv cache; the wheel set is ~3 GB (torch + CUDA libs are in the lock).

## Quality gates (mirror of `.github/workflows/test.yml`)

```bash
uv run ruff check .            # -> All checks passed
uv run ruff format --check .   # -> 60 files already formatted
uv run pytest tests/ -v        # -> 401 passed in ~14 s (GPT-2 mocked in tests; spaCy model loads for real)
```

Do **not** use `make test` / `make lint-check` on a warm checkout — every Makefile target
re-runs `uv venv .venv --seed` and fails when `.venv` already exists. Run the `uv run …`
commands directly (that is what CI runs after its one-time `uv venv`).

## Primary-flow smoke test (MCP over stdio)

`uv run run_server.py` speaks newline-delimited JSON-RPC on stdin/stdout (logs → stderr, no
port). The easiest verified path is fastmcp's own client, from a script with **no** PEP-723
header so it runs in the project venv:

```python
# /tmp/mcp_smoke.py
import asyncio
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

async def main():
    t = StdioTransport(command="uv", args=["run", "run_server.py"], cwd="/home/user/work/writing-tools-mcp")
    async with Client(t) as client:
        tools = await client.list_tools()                      # expect 13 tools
        r = await client.call_tool("word_count", {"text": "The quick brown fox jumps over the lazy dog."})
        print(r.data)

asyncio.run(main())
```

```bash
uv run python /tmp/mcp_smoke.py
```

FastMCP 3.x note: `StdioTransport` takes `command` + `args` (not `server_command=`).

Results from the verified run: 13 tools listed; `word_count` 17; `spellcheck` caught 3/3
planted misspellings; `readability_score` flesch 77.7 / kincaid 4.33 / fog 3.07;
`passive_voice_detection` returned exactly the 2 passive sentences; `stylometric_analysis`
produced features + z-scores vs `brown_corpus`; `perplexity_analysis` ran real GPT-2 (the
first call downloads weights from HuggingFace to `~/.cache/huggingface`, then returned
doc_ppl 610 / burstiness 605 on sample text).

Boot-only check (no client): `timeout 120 uv run run_server.py </dev/null` → exits 0 and
logs `Starting MCP server … with transport 'stdio'` on stderr.

## Gotchas (all hit during onboarding)

1. Python 3.13 + `uv.lock` → spacy 3.8.5 source build fails (Cython/NumPy headers). Use 3.12.
2. `make *` targets assume a fresh checkout — they try to recreate `.venv` every time.
3. `uv run run_server.py` resolves its own unlocked ephemeral env from the script's PEP-723
   metadata, not the project venv. For the locked env use `uv run writing-tools-mcp`.
4. stdout is the protocol stream; anything printed there corrupts the client session.
5. The first `perplexity_analysis` call needs HuggingFace network access for GPT-2 (~550 MB).
6. `uv.lock` is tracked although `.gitignore` lists it — leave it alone.

## Timing (warm uv cache)

venv + sync ~30 s · ruff ~2 s · pytest ~14 s · server boot ~20 s (first ephemeral env
resolve) · full smoke test incl. GPT-2 ~17 s.
