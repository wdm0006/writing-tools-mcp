"""Pairwise preference evaluation (W9): data model and corpus loading.

The committed corpus (``corpus.jsonl``) holds short excerpt pairs from
public-domain works where an earlier draft or edition can be paired with a
later, revised one. Every pair carries its own provenance so the assembly is
auditable offline; ``MANIFEST.md`` documents the sources and licenses, and
``extraction.json`` records exactly how each excerpt was cut from its source.
"""

import json
from pathlib import Path
from typing import Any, TypedDict

__all__ = [
    "PAIR_ID_REQUIRED_FIELDS",
    "CorpusError",
    "load_corpus",
    "validate_corpus",
]


class CorpusError(ValueError):
    """Raised when the committed corpus violates its documented schema."""


class PairText(TypedDict):
    """One side of a pair: an excerpt plus its provenance."""

    label: str
    year: int
    source_url: str
    license: str
    text: str


class Pair(TypedDict):
    """A draft/unedited excerpt paired with its revised/edited counterpart.

    ``unedited`` is always the earlier text and ``edited`` the later,
    revised one; the delta pipeline measures whether the W4 verdict direction
    agrees with that ordering.
    """

    pair_id: str
    work: str
    scene: str
    stratum: str
    baseline: str
    unedited: PairText
    edited: PairText


#: Fields every side of every pair must carry. Kept as data (not inline asserts)
#: so the corpus contract has exactly one statement of itself.
PAIR_ID_REQUIRED_FIELDS = ("pair_id", "work", "scene", "stratum", "baseline", "unedited", "edited")
PAIR_TEXT_REQUIRED_FIELDS = ("label", "year", "source_url", "license", "text")

CORPUS_PATH = Path(__file__).parent / "corpus.jsonl"


def validate_corpus(pairs: list[dict[str, Any]]) -> None:
    """Check the corpus contract; raise :class:`CorpusError` on any violation.

    Checks shape (required provenance fields), uniqueness (pair ids), and
    plausibility (non-trivial excerpt lengths). Length thresholds are
    intentionally lower than the builder's guarantees so a hand-edit of a
    committed excerpt still loads, but a corrupted or empty record cannot.
    """
    seen: set[str] = set()
    for pair in pairs:
        missing = [field for field in PAIR_ID_REQUIRED_FIELDS if field not in pair]
        if missing:
            raise CorpusError(f"pair missing required fields {missing}: {pair.get('pair_id', '?')}")
        pair_id = pair["pair_id"]
        if pair_id in seen:
            raise CorpusError(f"duplicate pair_id: {pair_id!r}")
        seen.add(pair_id)

        for side_name in ("unedited", "edited"):
            side = pair[side_name]
            side_missing = [field for field in PAIR_TEXT_REQUIRED_FIELDS if field not in side]
            if side_missing:
                raise CorpusError(f"pair {pair_id!r} side {side_name!r} missing fields {side_missing}")
            text = side["text"]
            if not isinstance(text, str) or len(text.split()) < 60:
                raise CorpusError(f"pair {pair_id!r} side {side_name!r} excerpt too short or not text")
            if not side["source_url"].startswith(("http://", "https://")):
                raise CorpusError(f"pair {pair_id!r} side {side_name!r} has no documented source URL")


def load_corpus(path: Path | None = None) -> list[Pair]:
    """Load and validate the committed pairwise corpus.

    Args:
        path: Override for the corpus location (tests use it to exercise the
            loader against fixtures without touching the committed file).

    Returns:
        The validated pair list, in committed order (the builder's fixed
        scene order — the pipeline never re-sorts data).
    """
    corpus_path = path if path is not None else CORPUS_PATH
    try:
        with open(corpus_path, encoding="utf-8") as handle:
            pairs = [json.loads(line) for line in handle if line.strip()]
    except json.JSONDecodeError as exc:
        raise CorpusError(f"{corpus_path}: malformed JSON: {exc}") from exc
    validate_corpus(pairs)
    return pairs
