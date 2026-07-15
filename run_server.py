#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pip",
#   "fastmcp>=2.14.0,<3.0.0",
#   "pyspellchecker",
#   "textstat",
#   "spacy",
#   "markdown-it-py",
#   "requests",
#   "transformers>=4.35.0",
#   "torch>=2.0.0",
#   "pyyaml>=6.0",
#   "numpy>=1.24.0",
#   "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
# ]
# ///
"""Standalone launcher for the Writing Tools MCP server.

Running this file with ``uv run run_server.py`` uses the inline PEP-723 metadata
above to resolve dependencies, then hands off to ``server.app.main``. This is the
path used by the Claude Desktop bundle (see manifest.json) and by anyone running
the server from a checkout without installing it. Pip installs use the
``writing-tools-mcp`` console script (``server.app:main``) instead.
"""

from server.app import main

if __name__ == "__main__":
    main()
