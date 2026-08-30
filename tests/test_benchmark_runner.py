"""End-to-end tests for the offline benchmark runner.

These drive the real runner over a tiny committed fixture. Only the stylometry
analysis is requested, so nothing downloads GPT-2; the perplexity path is
exercised against a stub analyzer instead.
"""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from benchmarks import run_benchmark, runner

FIXTURE_CORPUS = Path(__file__).parent / "data" / "tiny_benchmark_corpus.jsonl"


@pytest.fixture(scope="module")
def scored_run(tmp_path_factory):
    """Run the actual CLI over the tiny fixture, stylometry only."""
    out_dir = tmp_path_factory.mktemp("benchmark-run")
    exit_code = run_benchmark.main(
        ["--corpus", str(FIXTURE_CORPUS), "--out-dir", str(out_dir), "--analyses", "stylometry", "--quiet"]
    )
    assert exit_code == 0
    return out_dir


class TestRunnerOverTinyCorpus:
    def test_writes_all_three_artifacts(self, scored_run):
        assert (scored_run / "scores.jsonl").exists()
        assert (scored_run / "report.md").exists()
        assert (scored_run / "run_metadata.json").exists()

    def test_every_corpus_document_gets_a_record_with_provenance(self, scored_run):
        corpus = runner.load_corpus(FIXTURE_CORPUS)
        lines = (scored_run / "scores.jsonl").read_text(encoding="utf-8").strip().split("\n")
        records = [json.loads(line) for line in lines]

        assert [record["doc_id"] for record in records] == [entry["doc_id"] for entry in corpus]
        assert [record["label"] for record in records] == [entry["label"] for entry in corpus]
        for record, entry in zip(records, corpus, strict=True):
            assert record["source"] == entry["source"]
            assert record["source"]["raid_id"]

    def test_records_carry_real_analyzer_output(self, scored_run):
        records = [json.loads(line) for line in (scored_run / "scores.jsonl").read_text().strip().split("\n")]
        for record in records:
            stylometry = record["stylometry"]
            assert stylometry["ok"] is True, stylometry
            assert isinstance(stylometry["high_ai_probability"], bool)
            assert 0.0 <= stylometry["confidence_score"] <= 1.0
            # Real feature extraction, not a placeholder.
            assert stylometry["features"]["avg_sentence_len"] > 0
            assert "ttr" in stylometry["z_scores"]

    def test_recorded_values_equal_the_analyzer_s_own_output(self, scored_run, ai_detection_analyzer):
        """The runner must report what the analyzer said, not a value of its own.

        Shape assertions alone pass on a runner that hardcodes its answers, so
        this compares every recorded field against a direct analyzer call on the
        same text.
        """
        corpus = {entry["doc_id"]: entry for entry in runner.load_corpus(FIXTURE_CORPUS)}
        records = [json.loads(line) for line in (scored_run / "scores.jsonl").read_text().strip().split("\n")]

        distinct_flags = set()
        for record in records:
            expected = ai_detection_analyzer.stylometric_analysis(corpus[record["doc_id"]]["text"])
            flags = expected["flags"]
            stylometry = record["stylometry"]
            distinct_flags.add(stylometry["high_ai_probability"])

            assert stylometry["high_ai_probability"] == flags["high_ai_probability"]
            assert stylometry["confidence_score"] == flags["confidence_score"]
            assert stylometry["ai_detection_confidence"] == flags["ai_detection_confidence"]
            assert stylometry["ai_indicators"] == flags["ai_indicators"]
            assert stylometry["warnings"] == flags["warnings"]
            assert stylometry["errors"] == flags["errors"]
            assert stylometry["z_scores"] == expected["z_scores"]
            for feature in ("avg_sentence_len", "ttr", "fog", "mtld"):
                assert stylometry["features"][feature] == expected["features"][feature]
            for tag, value in expected["features"]["pos_ratios"].items():
                assert stylometry["features"][f"pos_ratios.{tag}"] == value

        # The fixture must contain both outcomes, or a runner hardcoding one of
        # them would satisfy every assertion above.
        assert distinct_flags == {True, False}

    def test_report_reports_the_fixture_sample_counts(self, scored_run):
        markdown = (scored_run / "report.md").read_text(encoding="utf-8")
        corpus = runner.load_corpus(FIXTURE_CORPUS)
        humans = sum(1 for entry in corpus if entry["label"] == "human")
        machines = len(corpus) - humans
        assert f"Documents: {len(corpus)} ({humans} human, {machines} machine)" in markdown
        assert "| `stylometry` |" in markdown
        assert "Every document produced an analyzer result" in markdown

    def test_rerunning_reproduces_both_artifacts(self, scored_run, tmp_path):
        second = tmp_path / "second"
        assert (
            run_benchmark.main(
                ["--corpus", str(FIXTURE_CORPUS), "--out-dir", str(second), "--analyses", "stylometry", "--quiet"]
            )
            == 0
        )
        assert (second / "scores.jsonl").read_bytes() == (scored_run / "scores.jsonl").read_bytes()
        assert (second / "report.md").read_bytes() == (scored_run / "report.md").read_bytes()

    def test_no_gpt2_model_was_loaded(self, scored_run):
        metadata = json.loads((scored_run / "run_metadata.json").read_text())
        assert metadata["analyses"] == ["stylometry"]


class TestRunnerSetsUpItsOwnState:
    def test_build_analyzer_initializes_text_processing_itself(self, monkeypatch):
        """The runner must work outside pytest, where conftest sets no globals."""
        from server.text_processing import preprocessor as preprocessor_module
        from server.text_processing import sentence_splitter as splitter_module

        monkeypatch.setattr(preprocessor_module, "_preprocessor", None)
        monkeypatch.setattr(splitter_module, "_nlp_model", None)

        analyzer = runner.build_analyzer()

        assert preprocessor_module._preprocessor is not None
        assert splitter_module._nlp_model is not None
        result = analyzer.stylometric_analysis("The cat sat. A dog ran away quickly. Birds sang all morning long.")
        assert "error" not in result


class TestFailuresAreRecordedNotDropped:
    def test_error_shaped_analyzer_result_becomes_an_explicit_failure(self):
        analyzer = Mock()
        analyzer.stylometric_analysis.return_value = {"error": "Empty text provided", "features": {}}
        analyzer.perplexity_analysis.return_value = {"error": "Analysis failed: 'max_length'"}

        scored = runner.score_records(
            [{"doc_id": "d1", "label": "human", "text": "x", "source": {"raid_id": "abc"}}], analyzer
        )
        assert scored[0]["stylometry"] == {"ok": False, "error": "Empty text provided"}
        assert scored[0]["perplexity"] == {"ok": False, "error": "Analysis failed: 'max_length'"}
        assert scored[0]["source"] == {"raid_id": "abc"}

    def test_a_raising_analyzer_does_not_end_the_run(self):
        analyzer = Mock()
        analyzer.stylometric_analysis.side_effect = [RuntimeError("kaboom"), {"error": "second"}]

        scored = runner.score_records(
            [
                {"doc_id": "d1", "label": "human", "text": "x"},
                {"doc_id": "d2", "label": "machine", "text": "y"},
            ],
            analyzer,
            analyses=["stylometry"],
        )
        assert len(scored) == 2
        assert scored[0]["stylometry"] == {"ok": False, "error": "RuntimeError: kaboom"}
        assert scored[1]["stylometry"] == {"ok": False, "error": "second"}

    def test_perplexity_results_are_recorded_without_loading_gpt2(self):
        analyzer = Mock()
        analyzer.perplexity_analysis.return_value = {
            "doc_ppl": 18.5,
            "doc_burstiness": None,
            "sentences": [{"text": "a", "ppl": 18.5}, {"text": "b", "ppl": None}],
            "flags": {"high_ai_probability": False, "reasons": ["Burstiness requires at least two scored sentences"]},
        }
        scored = runner.score_records(
            [{"doc_id": "d1", "label": "machine", "text": "x"}], analyzer, analyses=["perplexity"]
        )
        assert scored[0]["perplexity"]["ok"] is True
        assert scored[0]["perplexity"]["measurements"] == {"doc_ppl": 18.5, "doc_burstiness": None}
        assert scored[0]["perplexity"]["n_sentences"] == 2
        assert scored[0]["perplexity"]["n_scored_sentences"] == 1


class TestCorpusLoading:
    def test_committed_fixture_has_both_labels_and_unique_text(self):
        corpus = runner.load_corpus(FIXTURE_CORPUS)
        labels = {entry["label"] for entry in corpus}
        assert labels == {"human", "machine"}
        texts = [entry["text"] for entry in corpus]
        assert len(set(texts)) == len(texts)

    def test_duplicate_doc_id_is_rejected(self, tmp_path):
        path = tmp_path / "corpus.jsonl"
        entry = {"doc_id": "a", "label": "human", "text": "hello"}
        path.write_text(json.dumps(entry) + "\n" + json.dumps(entry) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="repeats doc_id"):
            runner.load_corpus(path)

    def test_unknown_label_is_rejected(self, tmp_path):
        path = tmp_path / "corpus.jsonl"
        path.write_text(json.dumps({"doc_id": "a", "label": "robot", "text": "hi"}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown label"):
            runner.load_corpus(path)

    def test_missing_field_is_rejected(self, tmp_path):
        path = tmp_path / "corpus.jsonl"
        path.write_text(json.dumps({"doc_id": "a", "label": "human"}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required field 'text'"):
            runner.load_corpus(path)
