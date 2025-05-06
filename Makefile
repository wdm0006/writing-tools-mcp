# Makefile for Writing Tools MCP Server Development

.PHONY: install lint format test clean all

# Variables
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
UV = uv

# Default target: Run common development tasks
all: install test lint

# Setup virtual environment and install dependencies
# This target creates the venv and installs packages only if the venv
# marker file ($(VENV_DIR)/bin/activate) doesn't exist.
install: 
	@echo "--- Creating virtual environment in $(VENV_DIR)... ---"
	$(UV) venv $(VENV_DIR) --seed
	@echo "--- Installing dependencies using uv... ---"
	# Install main dependencies from server.py requirements and dev tools
	$(UV) pip install "mcp[cli]" pyspellchecker textstat spacy ruff pytest

# Lint the code using Ruff within the virtual environment
lint: install
	@echo "--- Linting with Ruff... ---"
	$(UV) run ruff check --fix .

# Format the code using Ruff within the virtual environment
format: install
	@echo "--- Formatting with Ruff... ---"
	$(UV) run ruff format .

# Run tests using Pytest within the virtual environment
test:
	@echo "--- Running tests with Pytest... ---"
	$(UV) run pytest tests/

# Clean up the virtual environment and cache files
clean:
	@echo "--- Cleaning up project... ---"
	rm -rf $(VENV_DIR)
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	@echo "--- Clean complete. ---" 