#!/usr/bin/env python3
"""Build a custom stylometric baseline from a directory of your own writing.

Each ``*.txt`` file in the given directory is treated as one document in the
corpus (front matter, markdown markup, and code fences should already be
stripped - this script analyzes exactly the text it's given). The result is
saved via ``BaselineManager.save_baseline()``, so it's immediately usable as::

    stylometric_analysis(text, baseline="<name>")

Usage:
    uv run scripts/build_baseline.py <name> <texts_dir> [--description TEXT] [--all-features]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.config import load_config  # noqa: E402
from server.models import initialize_models  # noqa: E402
from server.stylometry import ALL_SIMPLE_FEATURES, BaselineManager, build_baseline_from_texts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="Baseline name to save as (letters, digits, '.', '_', '-' only)")
    parser.add_argument("texts_dir", help="Directory of *.txt files, one document per file")
    parser.add_argument("--description", default=None, help="Human-readable description stored in corpus_info")
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="Include every stylometric feature, including TTR and hapax rate. These are "
        "length-confounded (see server/stylometry/corpus_baseline.py); omit this flag to "
        "build the length-robust baseline used by default.",
    )
    parser.add_argument("--min-words", type=int, default=50, help="Drop documents shorter than this (default: 50)")
    args = parser.parse_args()

    texts_dir = Path(args.texts_dir)
    paths = sorted(texts_dir.glob("*.txt"))
    if not paths:
        parser.error(f"No .txt files found in {texts_dir}")

    texts = [path.read_text(encoding="utf-8") for path in paths]

    config = load_config()
    model_managers = initialize_models(config)
    nlp = model_managers["spacy"].get_model()

    baseline = build_baseline_from_texts(
        texts,
        nlp,
        corpus_info={
            "name": args.name,
            "description": args.description or f"Custom baseline built from {texts_dir}",
        },
        features=ALL_SIMPLE_FEATURES if args.all_features else None,
        min_words=args.min_words,
    )

    manager = BaselineManager()
    if not manager.save_baseline(args.name, baseline):
        raise SystemExit(f"Failed to save baseline '{args.name}'")

    sample_size = baseline["corpus_info"]["sample_size"]
    print(f"Saved baseline '{args.name}' ({sample_size} documents, out of {len(paths)} files found)")
    print(f'Use it with: stylometric_analysis(text, baseline="{args.name}")')


if __name__ == "__main__":
    main()
