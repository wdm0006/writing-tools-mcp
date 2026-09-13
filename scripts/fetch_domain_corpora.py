#!/usr/bin/env python3
"""Assemble the public-domain corpora behind the shipped domain baselines.

Downloads each pinned Project Gutenberg source (cached locally), verifies it
against a pinned sha256, strips the Gutenberg boilerplate, splits the work into
documents on per-work heading patterns (skipping indexes and transcriber notes),
and writes one UTF-8 ``*.txt`` per document under ``<corpora-root>/<register>/``.

The output feeds ``scripts/build_baseline.py``::

    uv run scripts/fetch_domain_corpora.py --corpora-root data/corpora
    uv run scripts/build_baseline.py essays data/corpora/essays --description "..."
    mv server/data/baselines/custom_baselines/essays.json server/data/baselines/essays.json
    # (repeat the build/mv pair for technical_docs and scientific_prose)

Determinism: the same pinned sources always produce byte-identical corpora.
Document order follows the WORKS table, filenames are zero-padded, and every
download is content-verified - a source that no longer matches its pin fails
loudly instead of silently building a different baseline.
"""

import argparse
import hashlib
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REGISTERS = ("essays", "technical_docs", "scientific_prose")

GUTENBERG_START = re.compile(r"\*{3}\s*START OF THE PROJECT GUTENBERG EBOOK[^*]*\*{3}", re.IGNORECASE)
GUTENBERG_END = re.compile(r"\*{3}\s*END OF THE PROJECT GUTENBERG EBOOK[^*]*\*{3}", re.IGNORECASE)
# Old-format files open their trailing license block with this line before the
# formal END marker, so the content ends at whichever boundary comes first.
GUTENBERG_LICENSE_LEAD = re.compile(r"^End of (the )?Project Gutenberg.*$", re.IGNORECASE | re.MULTILINE)
ILLUSTRATION_START = re.compile(r"^\s*\[Illustration:")


@dataclass(frozen=True)
class WorkSource:
    """One pinned Project Gutenberg source text and how to segment it."""

    register: str
    gutenberg_id: int
    author: str
    title: str
    year: str
    sha256: str
    heading: str  # full-line regex that starts a new document
    stops: tuple[str, ...] = ()  # regexes that close the current document and skip to the next heading


# Split rules were validated per work against the actual files: heading patterns
# only match real section/essay title lines, and the stop patterns excise
# transcriber quotation indexes (Montaigne) and back-of-book indexes.
WORKS: tuple[WorkSource, ...] = (
    # essays - reflective, first-person argumentative prose
    WorkSource(
        register="essays",
        gutenberg_id=3600,
        author="Michel de Montaigne",
        title="Essays of Michel de Montaigne — Complete (Cotton translation, Hazlitt edition)",
        year="first published 1580; this translation published 1877",
        sha256="1b4c87312f0890e04cecee48e3a5fa263de65743230b819e16cb1bd72f7aee59",
        # Essay titles are bare all-caps lines (some also carry a "CHAPTER N" line);
        # volume title pages match too but split off as tiny front-matter fragments.
        heading=r"^[A-Z][A-Z ,'-]{7,60}\s*$",
        stops=(r"BOOKMARKS",),
    ),
    WorkSource(
        register="essays",
        gutenberg_id=2944,
        author="Ralph Waldo Emerson",
        title="Essays — First Series",
        year="1841",
        sha256="5d9d28c12da8bd9aaf3043e9a2d08fbd46116efd5a3736490a393599aef6d278",
        heading=r"^[IVX]+\.\s*$",
    ),
    WorkSource(
        register="essays",
        gutenberg_id=2945,
        author="Ralph Waldo Emerson",
        title="Essays — Second Series",
        year="1844",
        sha256="2e7eb7281ade2ab623b9e4a790f79ed8e8b25dc1d95d31f9eda4f7a7dc471e6e",
        heading=r"^[IVX]+\.\s+[A-Z]",
    ),
    WorkSource(
        register="essays",
        gutenberg_id=130,
        author="G. K. Chesterton",
        title="Orthodoxy",
        year="1908",
        sha256="ec525c93afddb05465bf6fa0af8f873078b8e42a2a24f9b999582994f054fdf2",
        heading=r"^[IVX]+\s+[A-Z][A-Z ,'-]*$",
    ),
    # technical_docs - instructional/technical prose
    WorkSource(
        register="technical_docs",
        gutenberg_id=15460,
        author="Archie Seldon Milton and Otto K. Wohlers",
        title="A Course in Wood Turning",
        year="1919",
        sha256="707edfd649a58335fb911fddd0a148af3abbd455bc63b88cb473ff3f811a9f36",
        # Body chapters have no trailing period; the table of contents entries do.
        heading=r"^CHAPTER [IVX]+\s*$",
    ),
    WorkSource(
        register="technical_docs",
        gutenberg_id=20846,
        author="William Noyes",
        title="Handwork in Wood",
        year="1910",
        sha256="9f2f62be4bbbd1c16557776eaf8fe81725b7ac9f4345a005fa2b313a03308652",
        heading=r"^CHAPTER [IVX]+[.,]",
        stops=(r"^INDEX\.",),
    ),
    WorkSource(
        register="technical_docs",
        gutenberg_id=27257,
        author="Frederick Irving Anderson",
        title="Electricity for the Farm",
        year="1915",
        sha256="7f5c3bdac70f37e0b22e121e62727fef811bb3c0cca711beef0421afb91d8e20",
        heading=r"^CHAPTER [IVX]+\s*$",
    ),
    WorkSource(
        register="technical_docs",
        gutenberg_id=12655,
        author="Popular Mechanics Co.",
        title="The Boy Mechanic, Volume 1: 700 Things for Boys to Do",
        year="1913",
        sha256="78e63335ced1fa63f6164fe6cc145eab26a559501094e2d9ec44574f96e4028a",
        heading=r"^\*\* .+\[\d+\]\s*$",
        stops=(r"^CONTENTS",),
    ),
    # scientific_prose - expository scientific writing
    WorkSource(
        register="scientific_prose",
        gutenberg_id=1228,
        author="Charles Darwin",
        title="On the Origin of Species (first edition)",
        year="1859",
        sha256="ededa9c0bf8761efed092c303b46c1c92de956838cba6249a33bedfd6d7363b4",
        heading=r"^CHAPTER [0-9IVX]+\.\s*$",
        stops=(r"^INDEX\.",),
    ),
    WorkSource(
        register="scientific_prose",
        gutenberg_id=14474,
        author="Michael Faraday",
        title="Experimental Researches in Electricity, Volume 1",
        year="papers first published 1831-1852; collected edition reissued 1889",
        sha256="c053ca4ae7880585a15001cd3ffdaf5f09cd08c3329b828ffd8c90ce335482b9",
        heading=r"^LECTURE [IVX]+\.$",
    ),
    WorkSource(
        register="scientific_prose",
        gutenberg_id=30155,
        author="Albert Einstein",
        title="Relativity: The Special and General Theory (Lawson translation)",
        year="German original 1916; English translation published 1920",
        sha256="86ed8156239455cbb6ed33e06097707d3ab7c40d46e9aa5ff32d3c3f94ef68fd",
        # Section titles are bare all-caps lines; wrapped titles split once more,
        # which only drops the continuation line into a discarded tiny fragment.
        heading=r"^[A-Z][A-Z ,()-]{9,70}\s*$",
    ),
)


def source_urls(gutenberg_id: int) -> tuple[str, ...]:
    """Download candidates for a Gutenberg id. Any candidate matching the pinned
    hash is accepted, so layout changes on the mirror stay harmless."""
    return (
        f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt.utf-8",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt",
    )


def strip_gutenberg_boilerplate(text: str) -> str:
    """Return the content between the Gutenberg START marker and the earliest
    end boundary (formal END marker or the license block's lead line)."""
    start = GUTENBERG_START.search(text)
    if start is None:
        raise ValueError("Gutenberg START marker not found")
    remainder = text[start.end() :]
    boundaries = [m.start() for pattern in (GUTENBERG_END, GUTENBERG_LICENSE_LEAD) if (m := pattern.search(remainder))]
    if not boundaries:
        raise ValueError("Gutenberg END marker not found")
    return remainder[: min(boundaries)]


def extract_fragments(text: str, heading: str, stops: tuple[str, ...] = ()) -> list[list[str]]:
    """Split boilerplate-free text into line fragments at heading lines.

    A stop line closes the current fragment and skips everything up to the next
    heading, so indexes and transcriber notes never enter the corpus. Lines
    before the first heading (front matter) never join a fragment, so fragment
    0 is the first heading's section, not front matter.
    """
    heading_re = re.compile(heading)
    stop_res = [re.compile(pattern) for pattern in stops]
    fragments: list[list[str]] = []
    current: list[str] | None = None
    skipping = False

    for line in text.replace("\r\n", "\n").split("\n"):
        if any(stop.search(line) for stop in stop_res):
            if current is not None:
                fragments.append(current)
            current, skipping = None, True
            continue
        if heading_re.match(line):
            if current is not None:
                fragments.append(current)
            current, skipping = [line], False
            continue
        if skipping or current is None:
            continue
        current.append(line)
    if current is not None:
        fragments.append(current)
    return fragments


def clean_fragment(fragment: list[str]) -> str:
    """Drop the heading line and [Illustration: ...] caption blocks, collapse
    blank-line runs, and trim — deterministically."""
    body: list[str] = []
    in_illustration = False
    for line in fragment[1:]:
        if in_illustration:
            if "]" in line:
                in_illustration = False
            continue
        if ILLUSTRATION_START.match(line):
            in_illustration = "]" not in line
            continue
        body.append(line)
    cleaned = "\n".join(body)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def split_into_documents(text: str, heading: str, stops: tuple[str, ...] = ()) -> list[str]:
    """Return the prose documents of a Gutenberg text in reading order.

    Front matter before the first heading, indexes/transcriber notes, tiny
    table-of-contents fragments (those are the builder's ``--min-words`` job,
    not ours), and illustration captions are all excluded.
    """
    documents = []
    for fragment in extract_fragments(strip_gutenberg_boilerplate(text), heading, stops):
        cleaned = clean_fragment(fragment)
        if cleaned:
            documents.append(cleaned)
    return documents


def document_filename(work: WorkSource, index: int) -> str:
    return f"pg{work.gutenberg_id}_{index:04d}.txt"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_work(work: WorkSource, raw_cache_dir: Path) -> Path:
    """Return the cached raw text for ``work``, downloading and verifying it if
    the cache misses. Raises on hash mismatch with re-pin instructions."""
    expected = f"pg{work.gutenberg_id}.txt"
    cached = raw_cache_dir / expected
    if cached.exists():
        actual = sha256_of(cached)
        if actual != work.sha256:
            raise SystemExit(
                f"cached {cached} sha256 mismatch\n  expected {work.sha256}\n  actual   {actual}\n"
                "Delete the file and re-run to re-download; if the mismatch persists, "
                "update the pin in WORKS and document the change."
            )
        return cached

    for url in source_urls(work.gutenberg_id):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
        except OSError as error:
            print(f"{url} failed: {error}", file=sys.stderr)
            continue
        actual = hashlib.sha256(payload).hexdigest()
        if actual == work.sha256:
            raw_cache_dir.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(payload)
            return cached
        # Never cache unverified bytes: a transient bad mirror must not
        # masquerade as a changed source on the next run.
        print(f"{url} content mismatch (sha256 {actual}), trying next candidate", file=sys.stderr)

    raise SystemExit(
        f"Could not fetch pg{work.gutenberg_id} matching the pinned sha256 {work.sha256}. "
        "Check your network connection, or re-pin WORKS if Project Gutenberg replaced the text."
    )


def assemble_work(work: WorkSource, raw_cache_dir: Path, corpora_root: Path) -> int:
    raw_path = fetch_work(work, raw_cache_dir)
    documents = split_into_documents(raw_path.read_text(encoding="utf-8"), work.heading, work.stops)

    out_dir = corpora_root / work.register
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, document in enumerate(documents, start=1):
        (out_dir / document_filename(work, index)).write_text(f"{document}\n", encoding="utf-8")
    return len(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpora-root", default="data/corpora", help="Output root, one subdirectory per register")
    parser.add_argument("--raw-cache-dir", default=None, help="Where to cache downloads (default: <corpora-root>/_raw)")
    parser.add_argument("--only", choices=REGISTERS, default=None, help="Assemble a single register")
    args = parser.parse_args()

    corpora_root = Path(args.corpora_root)
    raw_cache_dir = Path(args.raw_cache_dir) if args.raw_cache_dir else corpora_root / "_raw"

    for register in REGISTERS:
        if args.only and register != args.only:
            continue
        print(f"== {register} ==", file=sys.stderr)
        for work in (w for w in WORKS if w.register == register):
            count = assemble_work(work, raw_cache_dir, corpora_root)
            print(f"pg{work.gutenberg_id} {work.title}: {count} documents", file=sys.stderr)

    print(f"Corpora assembled under {corpora_root}", file=sys.stderr)


if __name__ == "__main__":
    main()
