"""Tests for building a stylometric baseline from a corpus of an author's own writing."""

import pytest

from server.stylometry import ALL_SIMPLE_FEATURES, build_baseline_from_texts

DOCS = [
    "The committee approved the revised budget yesterday. The council rejected the amended "
    "proposal quickly. Every member voted after a long and occasionally heated debate about "
    "the numbers involved, and the chair adjourned the session close to midnight.",
    "Rain fell across the valley for three straight days. Farmers watched the river rise past "
    "the old stone bridge, worried about the low fields near the mill. By the fourth morning "
    "the water had receded, leaving a fine silt over everything it touched.",
    "The new engine ran quieter than the last one, though it drew more current under load. "
    "Engineers traced the noise to a bearing that had been over-torqued at assembly, and the "
    "fix shipped within the week once the root cause was confirmed in testing.",
    "She spent the summer rebuilding the porch her grandfather had built decades earlier. Each "
    "board came off easier than she expected, and by August the new railing stood square and "
    "level, a small stubborn victory against the years of weather that had worn the old one down.",
]

# Every DOCS entry is well over 20 words but under the module's real-world default of 50,
# so tests build with a lower floor to keep the fixture corpus small and readable.
TEST_MIN_WORDS = 20

TOO_SHORT_DOC = "Way too short."


class TestBuildBaselineFromTexts:
    def test_raises_with_fewer_than_two_qualifying_documents(self, nlp):
        with pytest.raises(ValueError, match="at least 2 documents"):
            build_baseline_from_texts([DOCS[0], TOO_SHORT_DOC], nlp, min_words=TEST_MIN_WORDS)

    def test_default_features_exclude_length_confounded_ones(self, nlp):
        baseline = build_baseline_from_texts(DOCS, nlp, min_words=TEST_MIN_WORDS)
        stats = baseline["statistics"]

        assert "avg_sentence_len" in stats
        assert "sentence_len_std" in stats
        assert "fog" in stats
        assert "kincaid" in stats
        assert "ttr" not in stats
        assert "hapax_legomena_rate" not in stats

    def test_default_pos_tags_are_adp_and_det_only(self, nlp):
        baseline = build_baseline_from_texts(DOCS, nlp, min_words=TEST_MIN_WORDS)
        pos_ratios = baseline["statistics"]["pos_ratios"]

        assert set(pos_ratios.keys()) == {"ADP", "DET"}

    def test_all_simple_features_opts_in_to_full_set(self, nlp):
        baseline = build_baseline_from_texts(DOCS, nlp, features=ALL_SIMPLE_FEATURES, min_words=TEST_MIN_WORDS)
        stats = baseline["statistics"]

        assert "ttr" in stats
        assert "hapax_legomena_rate" in stats

    def test_statistics_have_mean_and_std(self, nlp):
        baseline = build_baseline_from_texts(DOCS, nlp, min_words=TEST_MIN_WORDS)

        for feature_stats in baseline["statistics"].values():
            if "mean" in feature_stats:
                assert "std" in feature_stats
                assert feature_stats["std"] >= 0

    def test_min_words_filters_short_documents(self, nlp):
        baseline = build_baseline_from_texts(DOCS + [TOO_SHORT_DOC], nlp, min_words=TEST_MIN_WORDS)
        assert baseline["corpus_info"]["sample_size"] == len(DOCS)

    def test_corpus_info_defaults_are_filled_in(self, nlp):
        baseline = build_baseline_from_texts(DOCS, nlp, corpus_info={"name": "test_corpus"}, min_words=TEST_MIN_WORDS)

        assert baseline["corpus_info"]["name"] == "test_corpus"
        assert baseline["corpus_info"]["sample_size"] == len(DOCS)
        assert baseline["corpus_info"]["language"] == "en"

    def test_built_baseline_scores_a_real_draft(self, nlp):
        """A baseline built by this module z-scores a real draft through the existing engine."""
        from server.stylometry import StylemetricAnalyzer, calculate_z_scores

        baseline = build_baseline_from_texts(DOCS, nlp, min_words=TEST_MIN_WORDS)
        features = StylemetricAnalyzer(nlp).extract_features(DOCS[0])

        z_scores = calculate_z_scores(features, baseline["statistics"])

        assert set(z_scores.keys()) >= {"avg_sentence_len", "sentence_len_std", "fog", "kincaid"}
        assert "ttr" not in z_scores
