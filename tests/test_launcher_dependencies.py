"""Keep the standalone launcher's runtime dependencies aligned with the project."""

import re
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).parent.parent


def _load_toml(path):
    with path.open("rb") as file:
        return tomllib.load(file)


def _load_script_metadata(path):
    content = path.read_text(encoding="utf-8")
    match = re.search(r"(?ms)^# /// script\s*$\n(?P<metadata>(?:^#.*$\n)+)^# ///\s*$", content)
    assert match, f"PEP 723 script metadata not found in {path}"

    metadata = re.sub(r"(?m)^# ?", "", match.group("metadata"))
    return tomllib.loads(metadata)


def _normalize(requirement_string):
    requirement = Requirement(requirement_string)
    return (
        canonicalize_name(requirement.name),
        tuple(sorted(canonicalize_name(extra) for extra in requirement.extras)),
        str(requirement.specifier),
        requirement.url,
        str(requirement.marker) if requirement.marker else None,
    )


def test_standalone_launcher_dependencies_match_project_dependencies():
    project_dependencies = _load_toml(ROOT / "pyproject.toml")["project"]["dependencies"]
    launcher_dependencies = _load_script_metadata(ROOT / "run_server.py")["dependencies"]

    project = {_normalize(requirement): requirement for requirement in project_dependencies}
    launcher = {_normalize(requirement): requirement for requirement in launcher_dependencies}
    missing = sorted(project[key] for key in project.keys() - launcher.keys())
    extra = sorted(launcher[key] for key in launcher.keys() - project.keys())

    assert not missing and not extra, (
        "Standalone launcher dependencies differ from project.dependencies.\n"
        f"Missing from run_server.py: {missing}\n"
        f"Extra in run_server.py: {extra}"
    )
