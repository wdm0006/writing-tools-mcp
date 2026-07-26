"""Regression check for the portable version extraction used by the bundle build.

Exercises ``scripts/get_version.py`` without relying on GNU grep and asserts it
returns ``project.version`` from pyproject.toml.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
SCRIPT = ROOT / "scripts" / "get_version.py"


def _expected_version() -> str:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def _load_script():
    spec = importlib.util.spec_from_file_location("get_version", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_version_matches_pyproject():
    module = _load_script()
    assert module.read_version(PYPROJECT) == _expected_version()


def test_script_cli_matches_pyproject():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == _expected_version()
