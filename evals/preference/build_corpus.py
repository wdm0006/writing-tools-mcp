"""Pairwise preference evaluation (W9): deterministic corpus assembly.

Builds ``corpus.jsonl`` + ``extraction.json`` from public-domain sources:
for each documented scene, both editions of a work are fetched, an anchor
phrase unique in each edition locates the same story beat, and a fixed number
of sentences is extracted from each side (harmonized to the smaller count so
both excerpts cover the same span).

This module touches the network — but ONLY when run as a script
(``python -m evals.preference.build_corpus``). It is never imported by tests;
the committed corpus makes every test offline-deterministic. Re-running the
builder re-fetches the sources and re-derives the corpus byte-identically
(``--check``), and the recorded sha256 digests tie the committed excerpts to
the documented source documents.

Extraction is intentionally simple and fully documented rather than clever:
normalize whitespace, find the anchor exactly once, walk back to the
preceding sentence boundary, take N sentences. Any anchor that goes missing
or ambiguous in a future re-upload of a source fails the build loudly.
"""

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

#: Polite-fetch policy: sources throttle scripted downloads (Gutenberg in
#: particular), so every request waits out a small gap and retries 429/5xx
#: with exponential backoff before failing.
FETCH_DELAY_SECONDS = 2.0
FETCH_RETRIES = 6

__all__ = [
    "MIN_SENTENCES",
    "SENTENCES_PER_EXCERPT",
    "SCENES",
    "SOURCES",
    "CorpusBuildError",
    "extract_window",
    "normalize_gutenberg_text",
    "render_wikisource_page",
    "scrub_wikisource_text",
]

CORPUS_DIR = Path(__file__).parent
CORPUS_OUT = CORPUS_DIR / "corpus.jsonl"
EXTRACTION_OUT = CORPUS_DIR / "extraction.json"

#: Both sides of a pair are cut to the same sentence count, so per-sentence
#: statistics (mean sentence length, TTR, hapax rates) compare like with like.
SENTENCES_PER_EXCERPT = 12
#: A scene where either edition cannot supply this many sentences is an error,
#: not a silent shortening.
MIN_SENTENCES = 8

# Sentence splitter used ONLY for corpus assembly. The evaluation itself
# re-splits through spaCy (the analyzer pipeline); this one just has to cut
# the same windows out of the same sources deterministically. It is a
# zero-width lookahead between sentences, so splitting never deletes a
# character (quote marks after a full stop stay attached to their sentence).
# Guarded against the common period abbreviations in these sources.
_SENTENCE_START = re.compile(
    r"(?<=[.!?])(?<!\bMr\.)(?<!\bMrs\.)(?<!\bDr\.)(?<!\bSt\.)(?<!\bNo\.)"
    r"(?=[\"“”’]?\s+[\"“”’]?[A-Z0-9(\"“”’])"
)
_PG_BOILERPLATE_END = re.compile(r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*", re.S)
_ILLUSTRATION = re.compile(r"\s*\[Illustration[^\]]*\]")
# Invisible formatting characters that leak out of HTML renderings (zero-width
# spaces at page/span boundaries, soft hyphens, BOMs). Removed everywhere.
_FORMAT_CHARS = re.compile("[\u200b\u200c\u200d\u2060\u00ad\ufeff]")
# Wikisource page-render furniture: bare page numbers from the magazine scan,
# which exist in the scan but never in the prose.
_PAGE_NUMBER = re.compile(r"(?<=\s)\d{1,5}(?=\s)")
# Small-caps artifact of the 1890 magazine scan: the word "The" typeset as
# "T HE" (leading capital + spaced small caps) at a sentence start. Merged
# only for whitelisted words, so ordinary prose like "I AM glad" is never
# touched.
_SMALL_CAPS_WORDS = ("THE",)


def _merge_smallcaps(text: str) -> str:
    """Merge whitelisted spaced small-caps words ('T HE studio' -> 'THE studio')."""
    return re.sub(
        r"\b([A-Z])\s+([A-Z]{1,3})\s+(?=[a-z])",
        lambda m: (m.group(1) + m.group(2) + " ")
        if (m.group(1) + m.group(2)).upper() in _SMALL_CAPS_WORDS
        else m.group(0),
        text,
    )

WIKISOURCE_API = "https://en.wikisource.org/w/api.php"
WIKISOURCE_DORIAN_TITLE = "Lippincott's Monthly Magazine/Volume 46/July 1890/The Picture of Dorian Gray"
WIKISOURCE_DORIAN_URL = "https://en.wikisource.org/wiki/" + urllib.parse.quote(WIKISOURCE_DORIAN_TITLE)
WIKISOURCE_UA = "writing-tools-mcp-eval/0.1 (pairwise preference corpus; public-domain research)"

#: Fetched sources. Every excerpt in the corpus is cut from one of these.
#: ``source_url`` is the human-readable provenance link; gutenberg sources
#: fetch that URL directly, wikisource sources fetch chapter subpages of a
#: page-transcluded scan through the MediaWiki API.
SOURCES: dict[str, dict[str, Any]] = {
    "frankenstein_1818": {
        "source_url": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
        "work": "Frankenstein; Or, The Modern Prometheus",
        "label": "1818 first edition (anonymous)",
        "year": 1818,
        "kind": "gutenberg",
        "license": "Public domain (published 1818). Project Gutenberg transcription of a public-domain text.",
    },
    "frankenstein_1831": {
        "source_url": "https://www.gutenberg.org/cache/epub/42324/pg42324.txt",
        "work": "Frankenstein; Or, The Modern Prometheus",
        "label": "1831 revised edition (Colburn & Bentley, new introduction by the author)",
        "year": 1831,
        "kind": "gutenberg",
        "license": "Public domain (published 1831). Project Gutenberg transcription of a photo-reprint of the 1831 edition.",
    },
    "origin_1859": {
        "source_url": "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
        "work": "On the Origin of Species by Means of Natural Selection",
        "label": "1859 first edition (John Murray)",
        "year": 1859,
        "kind": "gutenberg",
        "license": "Public domain (published 1859). Project Gutenberg transcription of a public-domain text.",
    },
    "origin_1872": {
        "source_url": "https://www.gutenberg.org/cache/epub/2009/pg2009.txt",
        "work": "On the Origin of Species by Means of Natural Selection",
        "label": "1872 sixth edition, 'with all Additions and Corrections' (John Murray)",
        "year": 1872,
        "kind": "gutenberg",
        "license": "Public domain (published 1872). Project Gutenberg transcription of a public-domain text.",
    },
    "alice_under_ground_1864": {
        "source_url": "https://www.gutenberg.org/cache/epub/19002/pg19002.txt",
        "work": "Alice's Adventures Under Ground",
        "label": "1864 manuscript fair copy ('Under Ground')",
        "year": 1864,
        "kind": "gutenberg",
        "license": "Public domain (written 1862-1864). Project Gutenberg transcription of the 1886 facsimile of a public-domain manuscript.",
    },
    "alice_wonderland_1865": {
        "source_url": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
        "work": "Alice's Adventures in Wonderland",
        "label": "1865 published edition (Macmillan)",
        "year": 1865,
        "kind": "gutenberg",
        "license": "Public domain (published 1865). Project Gutenberg transcription of a public-domain text.",
    },
    "dorian_1890": {
        "source_url": WIKISOURCE_DORIAN_URL,
        "work": "The Picture of Dorian Gray",
        "label": "July 1890 Lippincott's Magazine text (13-chapter version)",
        "year": 1890,
        "kind": "wikisource",
        "chapters": 13,
        "license": "Public domain (published 1890; author died 1900). Wikisource transcription of Lippincott's Monthly Magazine, Vol. 46.",
    },
    "dorian_1891": {
        "source_url": "https://www.gutenberg.org/cache/epub/174/pg174.txt",
        "work": "The Picture of Dorian Gray",
        "label": "April 1891 revised book edition (Ward, Lock & Co., 20 chapters, new preface)",
        "year": 1891,
        "kind": "gutenberg",
        "license": "Public domain (published 1891). Project Gutenberg transcription of a public-domain text.",
    },
}

#: Scenes. Each side's ``anchor`` must occur exactly once in its normalized
#: source text; the window starts at the sentence containing the anchor.
#: ``edited`` names the later text.
SCENES: list[dict[str, Any]] = [
    {
        "pair_id": "frankenstein-creation-night",
        "work": "Frankenstein",
        "scene": "The creature's first waking (Ch. 5 in 1831 numbering)",
        "stratum": "edition_revision",
        "unedited": {"source": "frankenstein_1818", "anchor": "beheld the accomplishment of my toils"},
        "edited": {"source": "frankenstein_1831", "anchor": "beheld the accomplishment of my toils"},
    },
    {
        "pair_id": "frankenstein-two-years-toil",
        "work": "Frankenstein",
        "scene": "Victor recounts the two years of construction",
        "stratum": "edition_revision",
        "unedited": {"source": "frankenstein_1818", "anchor": "worked hard for nearly two years"},
        "edited": {"source": "frankenstein_1831", "anchor": "worked hard for nearly two years"},
    },
    {
        "pair_id": "frankenstein-candle-night",
        "work": "Frankenstein",
        "scene": "The creature appears at Victor's bed by candlelight",
        "stratum": "edition_revision",
        "unedited": {"source": "frankenstein_1818", "anchor": "my candle was nearly burnt out"},
        "edited": {"source": "frankenstein_1831", "anchor": "my candle was nearly burnt out"},
    },
    {
        "pair_id": "frankenstein-elizabeth",
        "work": "Frankenstein",
        "scene": "Elizabeth introduced into the family (spelling edit: 'Everyone' -> 'Every one')",
        "stratum": "edition_revision",
        "unedited": {"source": "frankenstein_1818", "anchor": "Everyone loved Elizabeth"},
        "edited": {"source": "frankenstein_1831", "anchor": "Every one loved Elizabeth"},
    },
    {
        "pair_id": "origin-eye",
        "work": "On the Origin of Species",
        "scene": "The eye as an organ of extreme perfection (punctuation and clause edits)",
        "stratum": "edition_revision",
        "unedited": {"source": "origin_1859", "anchor": "the eye, with all its inimitable contrivances"},
        "edited": {"source": "origin_1872", "anchor": "the eye with all its inimitable contrivances"},
    },
    {
        "pair_id": "origin-pigeon-court",
        "work": "On the Origin of Species",
        "scene": "Akber Khan's pigeon court under selection",
        "stratum": "edition_revision",
        "unedited": {"source": "origin_1859", "anchor": "Akber Khan"},
        "edited": {"source": "origin_1872", "anchor": "Akber Khan"},
    },
    {
        "pair_id": "origin-entangled-bank",
        "work": "On the Origin of Species",
        "scene": "Closing paragraph ('entangled bank' -> 'tangled bank')",
        "stratum": "edition_revision",
        "unedited": {"source": "origin_1859", "anchor": "It is interesting to contemplate"},
        "edited": {"source": "origin_1872", "anchor": "It is interesting to contemplate"},
    },
    {
        "pair_id": "alice-opening",
        "work": "Alice's Adventures in Wonderland",
        "scene": "Opening by the riverbank",
        "stratum": "manuscript_to_publication",
        "unedited": {"source": "alice_under_ground_1864", "anchor": "beginning to get very tired"},
        "edited": {"source": "alice_wonderland_1865", "anchor": "beginning to get very tired"},
    },
    {
        "pair_id": "alice-pool-of-tears",
        "work": "Alice's Adventures in Wonderland",
        "scene": "The pool of tears",
        "stratum": "manuscript_to_publication",
        "unedited": {"source": "alice_under_ground_1864", "anchor": "pool of tears"},
        "edited": {"source": "alice_wonderland_1865", "anchor": "pool of tears"},
    },
    {
        "pair_id": "alice-rabbit-hole",
        "work": "Alice's Adventures in Wonderland",
        "scene": "Down the rabbit-hole",
        "stratum": "manuscript_to_publication",
        "unedited": {"source": "alice_under_ground_1864", "anchor": "a large rabbit-hole"},
        "edited": {"source": "alice_wonderland_1865", "anchor": "a large rabbit-hole"},
    },
    {
        "pair_id": "dorian-studio",
        "work": "The Picture of Dorian Gray",
        "scene": "Opening at Basil's studio ('rich odor' -> 'odour' among other edits)",
        "stratum": "edition_revision",
        "unedited": {"source": "dorian_1890", "anchor": "studio was filled"},
        "edited": {"source": "dorian_1891", "anchor": "studio was filled"},
    },
    {
        "pair_id": "dorian-temptation",
        "work": "The Picture of Dorian Gray",
        "scene": "Lord Henry on temptation",
        "stratum": "edition_revision",
        "unedited": {"source": "dorian_1890", "anchor": "only way to get rid of a temptation"},
        "edited": {"source": "dorian_1891", "anchor": "only way to get rid of a temptation"},
    },
    {
        "pair_id": "dorian-adonis",
        "work": "The Picture of Dorian Gray",
        "scene": "Basil's studio banter ('young Adonis')",
        "stratum": "edition_revision",
        "unedited": {"source": "dorian_1890", "anchor": "young Adonis"},
        "edited": {"source": "dorian_1891", "anchor": "young Adonis"},
    },
]


class CorpusBuildError(RuntimeError):
    """Raised when a source or scene cannot be extracted as documented."""


_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def _normalize_quotes(text: str) -> str:
    """Map curly quotes to straight ASCII quotes.

    Applied identically to every source: the 1890 magazine transcription and
    the 1891 Gutenberg transcription use different quote conventions, which is
    a transcription artifact rather than an authorial signal, and would
    otherwise leak into punctuation-based stylometric features.
    """
    return text.translate(_QUOTES)


def normalize_gutenberg_text(raw: str) -> str:
    """Strip Project Gutenberg boilerplate and collapse whitespace."""
    marker = "START OF THE PROJECT GUTENBERG EBOOK"
    start = raw.find(marker)
    if start < 0:
        raise CorpusBuildError("source text has no Project Gutenberg start marker")
    start = raw.find("***", start + len(marker)) + 3
    end_match = _PG_BOILERPLATE_END.search(raw, start)
    if not end_match:
        raise CorpusBuildError("source text has no Project Gutenberg end marker")
    body = raw[start:end_match.start()]
    body = _ILLUSTRATION.sub(" ", body)
    body = _FORMAT_CHARS.sub("", body)
    return _normalize_quotes(re.sub(r"\s+", " ", body).strip())


def _open_with_retry(url: str) -> bytes:
    """Fetch a URL, waiting politely and retrying throttled responses."""
    last_error: Exception | None = None
    for attempt in range(FETCH_RETRIES):
        time.sleep(FETCH_DELAY_SECONDS * (2**attempt if attempt else 0))
        request = urllib.request.Request(url, headers={"User-Agent": WIKISOURCE_UA})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https URLs
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 429 or error.code >= 500:
                last_error = error
                # Honor the server's Retry-After when it is longer than our backoff.
                retry_after = error.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(int(retry_after))
                continue
            raise
    raise CorpusBuildError(f"fetch failed after {FETCH_RETRIES} attempts: {url}") from last_error


def render_wikisource_page(title: str, chapters: int) -> str:
    """Render a Wikisource page with ``chapters`` subpages to plain text.

    The 1890 Lippincott's transcription is a page scan rendered through the
    MediaWiki parser; raw wikitext carries only ``<pages>`` macros, so we ask
    the API for rendered HTML and strip it.
    """
    parts: list[str] = []
    for chapter in range(1, chapters + 1):
        url = WIKISOURCE_API + "?" + urllib.parse.urlencode(
            {
                "action": "parse",
                "page": f"{title}/Chapter {chapter}",
                "prop": "text",
                "format": "json",
                "formatversion": "2",
            }
        )
        payload = json.loads(_open_with_retry(url))
        parts.append(payload["parse"]["text"])
    return "\n".join(parts)


def scrub_wikisource_text(html: str) -> str:
    """Strip rendered-HTML furniture from a Wikisource page to running prose."""
    import html as html_module

    cleaned = re.sub(r"<(style|script)[^>]*>.*?</\1>", " ", html, flags=re.S)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html_module.unescape(cleaned)
    cleaned = _FORMAT_CHARS.sub("", cleaned)
    cleaned = _merge_smallcaps(cleaned)
    cleaned = _PAGE_NUMBER.sub(" ", cleaned)
    return _normalize_quotes(re.sub(r"\s+", " ", cleaned).strip())


def split_sentences(text: str) -> list[str]:
    """Split at zero-width sentence boundaries; no characters are dropped."""
    return [part.strip() for part in _SENTENCE_START.split(text) if part.strip()]


def extract_window(text: str, anchor: str, n_sentences: int = SENTENCES_PER_EXCERPT) -> str:
    """Extract a fixed-sentence window around ``anchor`` from normalized text.

    The anchor must occur exactly once. The window begins at the sentence
    containing the anchor and runs for ``n_sentences`` complete sentences, so
    both sides of a pair start at the same story beat.
    """
    if text.count(anchor) != 1:
        raise CorpusBuildError(f"anchor {anchor!r} occurs {text.count(anchor)} times (expected 1)")

    anchor_at = text.index(anchor)
    # Walk back to the start of the sentence containing the anchor.
    preceding = text[:anchor_at]
    boundary_positions = [match.start() for match in _SENTENCE_START.finditer(preceding)]
    window_start = boundary_positions[-1] if boundary_positions else 0

    sentences = split_sentences(text[window_start:])
    if len(sentences) < MIN_SENTENCES:
        raise CorpusBuildError(
            f"anchor {anchor!r}: only {len(sentences)} sentences available (minimum {MIN_SENTENCES})"
        )
    return " ".join(sentences[:n_sentences])


def fetch_text(url: str, cache_dir: Path) -> Path:
    """Fetch (or reuse a cached copy of) a source document."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".cache")
    if cache_path.exists():
        return cache_path
    request = urllib.request.Request(url, headers={"User-Agent": WIKISOURCE_UA})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https URLs
        body = response.read()
    cache_path.write_bytes(body)
    return cache_path


def load_source_text(source_id: str, cache_dir: Path) -> tuple[str, str]:
    """Return (normalized text, sha256 of the raw fetched bytes) for a source."""
    source = SOURCES[source_id]
    if source["kind"] == "gutenberg":
        raw_path = fetch_text(source["source_url"], cache_dir)
        raw_bytes = raw_path.read_bytes()
        raw = raw_bytes.decode("utf-8", errors="replace")
        return normalize_gutenberg_text(raw), hashlib.sha256(raw_bytes).hexdigest()

    # Wikisource: fetch every chapter subpage, then render+scrub the union.
    chapters_html = render_wikisource_page(WIKISOURCE_DORIAN_TITLE, source["chapters"])
    combined = "\n".join(chapters_html) if isinstance(chapters_html, list) else chapters_html
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return scrub_wikisource_text(combined), digest


def build_corpus(cache_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assemble the full pair list and extraction metadata from the sources.

    Returns:
        ``(pairs, extraction)`` — the corpus records (ready for JSONL) and
        the extraction manifest (anchors, digests, lengths, fetch notes).
    """
    texts: dict[str, str] = {}
    digests: dict[str, str] = {}
    for source_id in sorted(SOURCES):
        texts[source_id], digests[source_id] = load_source_text(source_id, cache_dir)

    pairs: list[dict[str, Any]] = []
    extractions: list[dict[str, Any]] = []
    for scene in SCENES:
        side_records: dict[str, Any] = {}
        side_extraction: dict[str, Any] = {}
        sentence_counts: list[int] = []
        for side_name in ("unedited", "edited"):
            spec = scene[side_name]
            source_id = spec["source"]
            source = SOURCES[source_id]
            text = texts[source_id]
            anchor = spec["anchor"]
            anchor_at = text.index(anchor)
            window = extract_window(text, anchor)
            sentence_counts.append(len(split_sentences(window)))
            side_records[side_name] = {
                "label": source["label"],
                "year": source["year"],
                "source_url": source["source_url"],
                "license": source["license"],
                "text": window,
            }
            side_extraction[side_name] = {
                "source": source_id,
                "anchor": anchor,
                "anchor_offset": anchor_at,
                "sentences": len(split_sentences(window)),
                "words": len(window.split()),
            }

        # Harmonize: both sides cover the same number of sentences, so
        # per-sentence statistics compare the same span of each edition.
        shared = min(sentence_counts)
        if shared < MIN_SENTENCES:
            raise CorpusBuildError(f"scene {scene['pair_id']!r}: shortest side has {shared} sentences")
        for side_name in ("unedited", "edited"):
            spec = scene[side_name]
            window = extract_window(texts[spec["source"]], spec["anchor"], n_sentences=shared)
            side_records[side_name]["text"] = window
            side_extraction[side_name]["sentences"] = len(split_sentences(window))
            side_extraction[side_name]["words"] = len(window.split())

        pairs.append(
            {
                "pair_id": scene["pair_id"],
                "work": scene["work"],
                "scene": scene["scene"],
                "stratum": scene["stratum"],
                "baseline": "brown_corpus",
                **side_records,
            }
        )
        extractions.append({"pair_id": scene["pair_id"], **side_extraction})

    extraction: dict[str, Any] = {
        "parameters": {
            "sentences_per_excerpt": SENTENCES_PER_EXCERPT,
            "min_sentences": MIN_SENTENCES,
            "sentence_splitter": "zero-width boundary after [.!?] where a quote/space/capital starts the next sentence; abbreviation-guarded (Mr./Mrs./Dr./St./No.)",
            "note": "Both sides of a pair are cut to the same sentence count (the shorter side's).",
        },
        "sources": {
            source_id: {
                "url": SOURCES[source_id]["source_url"],
                "work": SOURCES[source_id]["work"],
                "label": SOURCES[source_id]["label"],
                "year": SOURCES[source_id]["year"],
                "license": SOURCES[source_id]["license"],
                "sha256": digests[source_id],
            }
            for source_id in sorted(SOURCES)
        },
        "pairs": extractions,
    }
    return pairs, extraction


def derived_corpus_text(pairs: list[dict[str, Any]]) -> str:
    """Serialize pairs exactly as ``write_corpus`` would commit them."""
    import unicodedata

    normalized = []
    for pair in pairs:
        normalized.append(
            {
                **pair,
                "edited": {**pair["edited"], "text": unicodedata.normalize("NFC", pair["edited"]["text"])},
                "unedited": {**pair["unedited"], "text": unicodedata.normalize("NFC", pair["unedited"]["text"])},
            }
        )
    return "".join(json.dumps(pair, ensure_ascii=False, sort_keys=True) + "\n" for pair in normalized)


def write_corpus(pairs: list[dict[str, Any]], extraction: dict[str, Any]) -> None:
    """Write the committed corpus and extraction manifest deterministically."""
    with open(CORPUS_OUT, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(derived_corpus_text(pairs))
    with open(EXTRACTION_OUT, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(extraction, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild the pairwise preference corpus from documented sources.")
    parser.add_argument("--cache-dir", default=None, help="Directory for fetched source documents (default: temp dir)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Re-derive the corpus and compare byte-for-byte with the committed files",
    )
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(tempfile.mkdtemp(prefix="preference-corpus-"))
    pairs, extraction = build_corpus(cache_dir)

    if args.check:
        committed = CORPUS_OUT.read_text(encoding="utf-8")
        if committed != derived_corpus_text(pairs):
            print("CHECK FAILED: re-derived corpus differs from the committed corpus.jsonl", file=sys.stderr)
            return 1
        print("CHECK OK: re-derived corpus is byte-identical to the committed corpus.jsonl", file=sys.stderr)
        return 0

    write_corpus(pairs, extraction)
    print(f"wrote {len(pairs)} pairs to {CORPUS_OUT} (extraction metadata: {EXTRACTION_OUT})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
