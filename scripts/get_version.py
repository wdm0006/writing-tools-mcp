#!/usr/bin/env python3
"""Print ``project.version`` from pyproject.toml without relying on GNU grep."""

from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


def read_version(pyproject: Path) -> str:
    if tomllib is not None:
        with pyproject.open("rb") as fh:
            return tomllib.load(fh)["project"]["version"]

    # Fallback for Python 3.10 without the ``tomli`` backport installed.
    in_project = False
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_project = stripped == "[project]"
            continue
        if in_project:
            key, sep, value = stripped.partition("=")
            if sep and key.strip() == "version":
                return value.strip().strip("'\"")
    raise SystemExit("Could not find project.version in pyproject.toml")


def main() -> None:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    print(read_version(pyproject))


if __name__ == "__main__":
    main()
