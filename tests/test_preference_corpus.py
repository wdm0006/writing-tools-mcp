"""W9 preference-corpus tests: schema, provenance coherence, loader rejections.

Pure file work — no models, no network. The corpus is a committed artifact;
these tests pin its documented invariants. Per-excerpt extraction metadata
(anchor, offsets, word counts) lives in ``evals/preference/extraction.json``
and must stay consistent with the committed texts.
"""

import json
import re
from pathlib import Path

import pytest

from evals.preference.corpus import CORPUS_PATH, PAIR_ID_REQUIRED_FIELDS, CorpusError, Pair, load_corpus

EXTRACTION_PATH = CORPUS_PATH.parent / "extraction.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
CANONICAL_HOSTS = {"www.gutenberg.org", "en.wikisource.org"}


@pytest.fixture(scope="module")
def pairs() -> list[Pair]:
    return load_corpus()


@pytest.fixture(scope="module")
def extraction() -> dict:
    return json.loads(EXTRACTION_PATH.read_text(encoding="utf-8"))


def _by_id(extraction: dict) -> dict[str, dict]:
    return {entry["pair_id"]: entry for entry in extraction["pairs"]}


class TestCommittedCorpus:
    def test_shape(self, pairs: list[Pair]) -> None:
        assert 10 <= len(pairs) <= 30  # a small-n evaluation, not a benchmark suite
        for pair in pairs:
            assert set(pair) == set(PAIR_ID_REQUIRED_FIELDS)
            for side in ("unedited", "edited"):
                assert set(pair[side]) == {"label", "year", "source_url", "license", "text"}

    def test_unique_ids(self, pairs: list[Pair]) -> None:
        ids = [pair["pair_id"] for pair in pairs]
        assert len(ids) == len(set(ids))

    def test_strata_are_documented_kinds(self, pairs: list[Pair]) -> None:
        assert {pair["stratum"] for pair in pairs} == {"edition_revision", "manuscript_to_publication"}

    def test_baseline_is_uniform(self, pairs: list[Pair]) -> None:
        assert {pair["baseline"] for pair in pairs} == {"brown_corpus"}

    def test_side_provenance_fields_are_populated(self, pairs: list[Pair]) -> None:
        for pair in pairs:
            for side_name in ("unedited", "edited"):
                side = pair[side_name]
                assert side["label"], f"{pair['pair_id']}/{side_name}: empty label"
                assert isinstance(side["year"], int) and 1800 <= side["year"] <= 1930
                assert side["license"], f"{pair['pair_id']}/{side_name}: empty license"
                assert side["source_url"].startswith("https://"), f"{pair['pair_id']}/{side_name}"
                host = side["source_url"].split("/")[2]
                assert host in CANONICAL_HOSTS, f"{pair['pair_id']}/{side_name}: unexpected host {host}"

    def test_excerpts_differ(self, pairs: list[Pair]) -> None:
        for pair in pairs:
            assert pair["unedited"]["text"] != pair["edited"]["text"], pair["pair_id"]

    def test_extraction_metadata_agrees_with_texts(self, pairs: list[Pair], extraction: dict) -> None:
        # The committed texts and the extraction record must tell the same story:
        # same pair ids, same source ids, same word and sentence counts.
        by_id = _by_id(extraction)
        assert set(by_id) == {pair["pair_id"] for pair in pairs}
        for pair in pairs:
            meta = by_id[pair["pair_id"]]
            for side_name in ("unedited", "edited"):
                side_meta = meta[side_name]
                assert len(pair[side_name]["text"].split()) == side_meta["words"]
                assert side_meta["sentences"] == extraction["parameters"]["sentences_per_excerpt"]
                assert side_meta["source"] in extraction["sources"]

    def test_source_digests_are_sha256_hex(self, extraction: dict) -> None:
        for source_id, source in extraction["sources"].items():
            assert HEX64.match(source["sha256"]), source_id
            assert source["url"].startswith("https://"), source_id
            assert source["license"], source_id

    def test_extraction_parameters_are_documented(self, extraction: dict) -> None:
        params = extraction["parameters"]
        assert params["sentences_per_excerpt"] >= 8
        assert params["min_sentences"] >= 1
        assert params["note"]


class TestLoaderRejections:
    """Schema violations raise CorpusError naming the offending pair."""

    def _corpus_from_pairs(self, tmp_path, records: list[dict]) -> "Path":
        path = tmp_path / "corpus.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
        return path

    def test_malformed_json_line(self, tmp_path, pairs: list[Pair]) -> None:
        path = tmp_path / "corpus.jsonl"
        path.write_text("{not json\n", encoding="utf-8")
        with pytest.raises(CorpusError, match="malformed JSON"):
            load_corpus(path)

    def test_duplicate_pair_id(self, tmp_path, pairs: list[Pair]) -> None:
        path = self._corpus_from_pairs(tmp_path, [pairs[0], pairs[0]])
        with pytest.raises(CorpusError, match="duplicate pair_id"):
            load_corpus(path)

    def test_missing_required_fields(self, tmp_path, pairs: list[Pair]) -> None:
        broken = json.loads(json.dumps(pairs[0]))
        del broken["stratum"]
        path = self._corpus_from_pairs(tmp_path, [broken])
        with pytest.raises(CorpusError, match="missing required fields"):
            load_corpus(path)

    def test_missing_side_field(self, tmp_path, pairs: list[Pair]) -> None:
        broken = json.loads(json.dumps(pairs[0]))
        del broken["edited"]["license"]
        path = self._corpus_from_pairs(tmp_path, [broken])
        with pytest.raises(CorpusError, match="edited"):
            load_corpus(path)

    def test_excerpt_too_short(self, tmp_path, pairs: list[Pair]) -> None:
        broken = json.loads(json.dumps(pairs[0]))
        broken["edited"]["text"] = "too short"
        path = self._corpus_from_pairs(tmp_path, [broken])
        with pytest.raises(CorpusError, match="too short"):
            load_corpus(path)

    def test_source_url_missing(self, tmp_path, pairs: list[Pair]) -> None:
        broken = json.loads(json.dumps(pairs[0]))
        broken["edited"]["source_url"] = "not a url"
        path = self._corpus_from_pairs(tmp_path, [broken])
        with pytest.raises(CorpusError, match="source URL"):
            load_corpus(path)
