#!/usr/bin/env python3
"""Score a committed labeled corpus with the repository's detector and report it.

Usage:
    uv run benchmarks/run_benchmark.py

That reproduces the committed artifacts under ``benchmarks/reports/``. See
``benchmarks/README.md`` for what is and is not reproducible byte-for-byte.
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import report as report_module  # noqa: E402
from benchmarks import runner  # noqa: E402
from server.config import load_config  # noqa: E402

DEFAULT_CORPUS = REPO_ROOT / "benchmarks" / "corpora" / "raid_wiki_v1" / "corpus.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "benchmarks" / "reports" / "raid_wiki_v1"


def _repo_relative(path: Path) -> str:
    """Repo-relative path where possible, so artifacts stay machine-independent."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _library_versions() -> dict:
    """Installed versions of every library whose output lands in the report.

    Read from distribution metadata rather than a ``__version__`` attribute:
    not every one of these exposes that attribute, and ``textstat`` exposes a
    tuple, so the attribute route silently reports "unknown" for some of them.
    """
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("spacy", "torch", "transformers", "textstat", "wordfreq", "numpy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="Corpus JSONL to score")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for the run's artifacts")
    parser.add_argument(
        "--analyses",
        default=",".join(runner.ANALYSES),
        help="Comma-separated subset of %s. Dropping 'perplexity' skips loading GPT-2." % (",".join(runner.ANALYSES),),
    )
    parser.add_argument("--baseline", default="brown_corpus", help="Stylometric baseline name")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N documents (for smoke runs)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-document progress output")
    args = parser.parse_args(argv)

    analyses = [name.strip() for name in args.analyses.split(",") if name.strip()]
    unknown = [name for name in analyses if name not in runner.ANALYSES]
    if unknown:
        parser.error(f"Unknown analyses: {', '.join(unknown)}")

    records = runner.load_corpus(args.corpus)
    if args.limit is not None:
        records = records[: args.limit]

    manifest = args.corpus.parent / "MANIFEST.md"
    corpus_info = {
        "name": args.corpus.parent.name,
        "description": f"{len(records)} labeled documents",
        "manifest": _repo_relative(manifest) if manifest.exists() else "n/a",
    }

    config = load_config()
    analyzer = runner.build_analyzer(config)

    def progress(index, doc_id):
        print(f"[{index}/{len(records)}] {doc_id}", file=sys.stderr)

    scored = runner.score_records(
        records,
        analyzer,
        analyses=analyses,
        baseline=args.baseline,
        progress=None if args.quiet else progress,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runner.write_scores(args.out_dir / "scores.jsonl", scored)

    markdown = report_module.build_report(
        scored,
        corpus_info=corpus_info,
        thresholds=runner.effective_thresholds(config),
        methods=analyses,
    )
    (args.out_dir / "report.md").write_text(markdown, encoding="utf-8")

    # Everything that legitimately varies between runs lives here, so report.md
    # and scores.jsonl can be diffed for reproducibility without false positives.
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": _repo_relative(args.corpus),
        "baseline": args.baseline,
        "analyses": analyses,
        "n_documents": len(scored),
        "library_versions": _library_versions(),
    }
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Wrote {args.out_dir / 'scores.jsonl'}, {args.out_dir / 'report.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
