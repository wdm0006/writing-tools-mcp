"""Unit tests for scripts/fetch_domain_corpora.py.

These run fully offline against synthetic Project-Gutenberg-shaped texts; they
pin the split semantics (front-matter drop, heading splits, stop-to-next-heading
skip, illustration-caption removal) and the content-verified caching that make
corpus assembly deterministic. End-to-end assembly happens when baselines are
rebuilt; the integration guarantees live in tests/test_domain_baselines.py.
"""

import hashlib
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "fetch_domain_corpora.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("fetch_domain_corpora", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fdc = _load_script()


def _pg_text(body: str, lead: str = "\n*** START OF THE PROJECT GUTENBERG EBOOK DUMMY ***\n") -> str:
    return f"{lead}{body}\n*** END OF THE PROJECT GUTENBERG EBOOK DUMMY ***\n"


def _work(**overrides):
    defaults = {
        "register": "essays",
        "gutenberg_id": 1,
        "author": "A. Author",
        "title": "A Work",
        "year": "1900",
        "sha256": "0" * 64,
        "heading": r"^CHAPTER [IVX]+\s*$",
        "stops": (),
    }
    defaults.update(overrides)
    return fdc.WorkSource(**defaults)


class TestBoilerplateStripping:
    def test_content_between_markers_is_kept(self):
        text = _pg_text("Title page\n\nChapter prose.")

        # The newline after the START marker is content-side; only the marker
        # lines and everything after the earliest end boundary are dropped.
        assert fdc.strip_gutenberg_boilerplate(text) == "\nTitle page\n\nChapter prose.\n"

    def test_old_format_license_lead_line_ends_content(self):
        # Old-format files open the trailing license block with this line
        # before the formal END marker; content must stop at the earlier one.
        text = _pg_text("Prose.")
        text = text.replace(
            "*** END OF THE PROJECT GUTENBERG EBOOK DUMMY ***",
            "End of Project Gutenberg's A Work, by A. Author\n\n*** END OF THE PROJECT GUTENBERG EBOOK DUMMY ***",
        )

        assert fdc.strip_gutenberg_boilerplate(text) == "\nProse.\n"

    @pytest.mark.parametrize("marker", ["START", "END"])
    def test_missing_marker_raises(self, marker):
        text = _pg_text("body")
        stripped = text.replace(f"*** {marker} OF THE PROJECT GUTENBERG EBOOK DUMMY ***", "")

        with pytest.raises(ValueError, match=marker):
            fdc.strip_gutenberg_boilerplate(stripped)


class TestSplitSemantics:
    def test_front_matter_and_heading_lines_are_excluded(self):
        text = _pg_text("Some front matter.\nCHAPTER I\nFirst prose.\nCHAPTER II\nSecond prose.")

        docs = fdc.split_into_documents(text, r"^CHAPTER [IVX]+\s*$")

        assert docs == ["First prose.", "Second prose."]

    def test_stop_line_skips_to_next_heading(self):
        # Index lines between two chapters vanish; both chapters survive.
        text = _pg_text("CHAPTER I\nFirst prose.\nINDEX.\nindexed entries\nCHAPTER II\nSecond prose.")

        docs = fdc.split_into_documents(text, r"^CHAPTER [IVX]+\s*$", (r"^INDEX\.",))

        assert docs == ["First prose.", "Second prose."]

    def test_stop_line_in_front_matter_keeps_first_heading(self):
        # A transcriber note before any heading must not eat the corpus.
        text = _pg_text("Transcriber notes.\nBOOKMARKS listed here.\nCHAPTER I\nProse.")

        docs = fdc.split_into_documents(text, r"^CHAPTER [IVX]+\s*$", (r"BOOKMARKS",))

        assert docs == ["Prose."]

    def test_crlf_and_illustration_blocks_are_normalized(self):
        text = _pg_text(
            "CHAPTER I\r\nBefore.\r\n[Illustration: one-line caption]\r\n"
            "Middle.\r\n[Illustration: caption that\r\nspans lines]\r\nAfter.\r\n\r\n\r\n\r\nTail."
        )

        docs = fdc.split_into_documents(text, r"^CHAPTER [IVX]+\s*$")

        assert docs == ["Before.\nMiddle.\nAfter.\n\nTail."]


class TestWorkTable:
    def test_every_work_is_well_formed(self):
        for work in fdc.WORKS:
            assert work.register in fdc.REGISTERS
            assert work.gutenberg_id > 0
            assert len(work.sha256) == 64
            int(work.sha256, 16)  # sha pins are hex
            fdc.re.compile(work.heading)  # headings compile
            for stop in work.stops:
                fdc.re.compile(stop)

    def test_registers_are_populated(self):
        for register in fdc.REGISTERS:
            assert any(work.register == register for work in fdc.WORKS)


class TestCachedFetch:
    def test_matching_cache_is_returned_without_download(self, tmp_path):
        payload = b"pinned bytes"
        work = _work(sha256=hashlib.sha256(payload).hexdigest())
        cache = tmp_path / "raw"
        cache.mkdir()
        (cache / "pg1.txt").write_bytes(payload)

        assert fdc.fetch_work(work, cache) == cache / "pg1.txt"

    def test_mismatched_cache_fails_loudly(self, tmp_path):
        work = _work()
        cache = tmp_path / "raw"
        cache.mkdir()
        (cache / "pg1.txt").write_bytes(b"tampered")

        with pytest.raises(SystemExit, match="mismatch"):
            fdc.fetch_work(work, cache)

    def test_download_is_verified_before_caching(self, tmp_path, monkeypatch):
        work = _work(sha256=hashlib.sha256(b"pinned bytes").hexdigest())
        cache = tmp_path / "raw"
        calls = []

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def read(self):
                return self.payload

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_urlopen(url, timeout):
            calls.append(url)
            # First candidate serves corrupted bytes, second the pinned ones.
            return FakeResponse(b"corrupted" if len(calls) == 1 else b"pinned bytes")

        monkeypatch.setattr(fdc.urllib.request, "urlopen", fake_urlopen)

        path = fdc.fetch_work(work, cache)

        assert path.read_bytes() == b"pinned bytes"
        assert len(calls) == 2  # the corrupted mirror was skipped, not cached


def test_document_filename_is_zero_padded():
    work = _work(gutenberg_id=42)
    assert fdc.document_filename(work, 7) == "pg42_0007.txt"
