"""Packaging metadata guarantees.

These tests pin two release-hygiene invariants: package metadata carries no
direct-URL dependencies (the spaCy model wheel is GitHub-hosted and a direct
URL in Requires-Dist blocks PyPI publication - it is bootstrapped at first use
instead, see ``server/models/spacy_manager.py``), and the versions stamped in
pyproject.toml and manifest.json agree.
"""

import json
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10 - the dev extra provides the backport
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parent.parent


def _all_requirements(pyproject: dict) -> list[str]:
    project = pyproject["project"]
    requirements = list(project.get("dependencies", []))
    for extra_requirements in project.get("optional-dependencies", {}).values():
        requirements.extend(extra_requirements)
    return requirements


def test_no_direct_url_dependencies():
    """Every dependency is a versioned PyPI requirement - no direct URLs."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)

    for requirement in _all_requirements(pyproject):
        assert " @ " not in requirement, f"direct-URL dependency found: {requirement}"
        assert "://" not in requirement, f"direct-URL dependency found: {requirement}"


def test_pyproject_and_manifest_versions_agree():
    """The MCPB bundle version must match the Python package version."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    manifest = json.loads((REPO_ROOT / "manifest.json").read_text(encoding="utf-8"))

    assert pyproject["project"]["version"] == manifest["version"]
