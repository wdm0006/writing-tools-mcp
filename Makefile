.PHONY: install lint lint-check format format-check test build-mcpb clean

VENV_DIR = .venv
UV = uv
BUNDLE_NAME = writing-tools-mcp.mcpb

install:
	$(UV) venv $(VENV_DIR) --seed
	$(UV) pip install -e ".[dev]"

lint: install
	$(UV) run ruff check --fix .

lint-check: install
	$(UV) run ruff check .

format: install
	$(UV) run ruff format .

format-check: install
	$(UV) run ruff format --check .

test: install
	$(UV) run pytest tests/

build-mcpb:
	@bash scripts/build_mcpb.sh

clean:
	rm -rf $(VENV_DIR)
	rm -rf build/
	rm -f $(BUNDLE_NAME)
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
