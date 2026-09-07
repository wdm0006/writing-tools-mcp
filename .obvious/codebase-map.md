# Codebase Map — writing-tools-mcp

Folder-level map (depth 2). All 13 MCP tools are defined in `server/app.py`; each delegates to
an analyzer class under `server/analyzers/`.

| Path | What lives there |
| --- | --- |
| `server/` | The Python package that *is* the MCP server (shipped as `writing-tools-mcp`) |
| `server/app.py` | FastMCP instance, all 13 `@mcp.tool` definitions, `main()` entry point |
| `server/analyzers/` | Tool implementations: `basic_stats`, `readability`, `keyword_analysis`, `style_analysis`, `ai_detection` |
| `server/config/` | `.mcp-config.yaml` handling: `defaults.py`, `schema.py`, `loader.py` |
| `server/models/` | Lazy model managers: `spacy_manager.py` (en_core_web_sm), `gpt2_manager.py` (GPT-2) |
| `server/stylometry/` | Stylometric feature extraction, baselines (Brown Corpus + custom), z-score statistics |
| `server/text_processing/` | Markdown parsing/section split, preprocessor (lemmatize/stopwords), sentence splitter |
| `server/data/baselines/` | Shipped baselines: `brown_corpus.json` + `custom_baselines/` worked example |
| `tests/` | pytest suite (401 tests): per-analyzer unit tests + config/logging/version/registry tests; GPT-2 is mocked, spaCy loads for real |
| `benchmarks/` | Offline detector benchmark, separate from the server: `run_benchmark.py`, `runner.py`, `metrics.py`, `corpora/`, `reports/` |
| `scripts/` | `build_baseline.py` (custom stylometry baselines), `build_mcpb.sh`, `get_version.py` |
| `.github/workflows/` | CI: `test.yml` (ruff + pytest on Python 3.10–3.12), `build-mcpb.yml` (bundle build) |
| `.cursor/rules/` | Cursor editor rule describing the server (partially stale — references the old single-file `server.py` layout) |
| `images/` | Icon and screenshot assets for the MCP bundle |
| `run_server.py` (root) | Standalone PEP-723 launcher → `server.app.main`; the path the Claude Desktop bundle runs |
| `pyproject.toml`, `uv.lock` (root) | Hatchling project + locked deps (spacy 3.8.5, torch 2.7.0, fastmcp 3.4.5, transformers 4.52.3) |
| `Makefile` (root) | `install` / `lint` / `lint-check` / `format` / `format-check` / `test` / `build-mcpb` / `clean` |
| `manifest.json`, `build.sh`, `.mcp-config.yaml`, `cache_gpt2_model.py` (root) | MCP bundle manifest + build script, optional server config, GPT-2 pre-cache helper |
