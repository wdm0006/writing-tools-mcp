# Makefile for Writing Tools MCP Server Development

.PHONY: install lint format test clean all build-mcpb run-server

# Variables
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
UV = uv
BUNDLE_NAME = writing-tools-mcp.mcpb

# Default target: Run common development tasks
all: install test lint

# Setup virtual environment and install dependencies
# This target creates the venv and installs packages only if the venv
# marker file ($(VENV_DIR)/bin/activate) doesn't exist.
install:
	@echo "--- Creating virtual environment in $(VENV_DIR)... ---"
	$(UV) venv $(VENV_DIR) --seed
	@echo "--- Installing dependencies from pyproject.toml using uv... ---"
	$(UV) pip install -e ".[dev]"

# Lint the code using Ruff within the virtual environment
lint: install
	@echo "--- Linting with Ruff... ---"
	$(UV) run ruff check --fix .

# Format the code using Ruff within the virtual environment
format: install
	@echo "--- Formatting with Ruff... ---"
	$(UV) run ruff format .

# Run tests using Pytest within the virtual environment
test: install
	@echo "--- Running tests with Pytest... ---"
	$(UV) run pytest tests/

# Run the MCP server
run-server:
	@echo "--- Running MCP server (server.py)... ---"
	$(UV) run server.py

# Build the MCPB bundle for Claude Desktop
build-mcpb:
	@echo "--- Building MCPB bundle for Claude Desktop... ---"
	@bash scripts/build_mcpb.sh

# Clean up the virtual environment and cache files
clean:
	@echo "--- Cleaning up project... ---"
	rm -rf $(VENV_DIR)
	rm -rf build/
	rm -f $(BUNDLE_NAME)
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	@echo "--- Clean complete. ---" 