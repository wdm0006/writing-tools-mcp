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

# mtld/mattr need >= 50 tokens per document (see StylemetricAnalyzer), longer than
# every DOCS entry above, so they get their own fixture rather than reusing DOCS.
LONG_DOCS = [
    "Rain fell across the valley for three straight days, and farmers watched the river "
    "rise past the old stone bridge, worried about the low fields near the mill. By the "
    "fourth morning the water had receded, leaving a fine silt over everything it touched, "
    "and the road crews spent the rest of the week clearing debris from the culverts.",
    "Engineers traced a strange noise in the new engine to a bearing that had been "
    "over-torqued at assembly, and the fix shipped within a week once the root cause was "
    "confirmed by testing. Nobody had expected a torque spec to be the culprit, least of "
    "all the technician who had signed off on the original build sheet.",
    "She spent the summer rebuilding a porch her grandfather had built decades earlier, and "
    "by August the railing stood square again, a small stubborn victory against years of "
    "weather that had worn the old one down. Neighbors stopped to admire the joinery, and "
    "she found herself explaining the same three cuts to nearly all of them.",
]


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

    def test_default_features_include_new_length_robust_additions(self, nlp):
        baseline = build_baseline_from_texts(DOCS, nlp, min_words=TEST_MIN_WORDS)
        stats = baseline["statistics"]

        for key in (
            "smog",
            "coleman_liau",
            "ari",
            "dale_chall",
            "mean_dependency_distance",
            "subordinate_clause_ratio",
        ):
            assert key in stats

    def test_default_features_include_mtld_and_mattr_given_long_enough_docs(self, nlp):
        baseline = build_baseline_from_texts(LONG_DOCS, nlp, min_words=TEST_MIN_WORDS)
        stats = baseline["statistics"]

        assert "mtld" in stats
        assert "mattr" in stats
        assert "mtld_lemma" in stats

    def test_default_features_include_second_round_additions(self, nlp):
        baseline = build_baseline_from_texts(DOCS, nlp, min_words=TEST_MIN_WORDS)
        stats = baseline["statistics"]

        for key in (
            "mean_word_frequency",
            "word_len_std",
            "lexical_density",
            "semicolon_ratio",
            "em_dash_ratio",
            "ellipsis_ratio",
            "exclamation_ratio",
            "parenthetical_rate",
            "hedge_rate",
            "booster_rate",
        ):
            assert key in stats

    def test_all_simple_features_includes_repetition_and_zipf(self, nlp):
        baseline = build_baseline_from_texts(DOCS, nlp, features=ALL_SIMPLE_FEATURES, min_words=TEST_MIN_WORDS)
        stats = baseline["statistics"]

        assert "fourgram_repetition_rate" in stats
        assert "zipf_slope" in stats

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


class TestFunctionWordBaseline:
    """Burrows'-Delta-style per-function-word baseline dimension."""

    def test_default_builds_function_word_freqs_for_every_tracked_word(self, nlp):
        from server.stylometry import StylemetricAnalyzer

        analyzer = StylemetricAnalyzer(nlp)
        baseline = build_baseline_from_texts(DOCS, nlp, min_words=TEST_MIN_WORDS)

        assert set(baseline["statistics"]["function_word_freqs"].keys()) == analyzer.function_words

    def test_empty_list_skips_the_dimension_entirely(self, nlp):
        baseline = build_baseline_from_texts(DOCS, nlp, function_words=[], min_words=TEST_MIN_WORDS)
        assert "function_word_freqs" not in baseline["statistics"]

    def test_custom_subset_is_respected(self, nlp):
        baseline = build_baseline_from_texts(DOCS, nlp, function_words=["the", "of"], min_words=TEST_MIN_WORDS)
        assert set(baseline["statistics"]["function_word_freqs"].keys()) == {"the", "of"}

    def test_built_baseline_scores_burrows_delta_on_a_real_draft(self, nlp):
        from server.stylometry import StylemetricAnalyzer, calculate_z_scores

        baseline = build_baseline_from_texts(DOCS, nlp, min_words=TEST_MIN_WORDS)
        features = StylemetricAnalyzer(nlp).extract_features(DOCS[0])

        z_scores = calculate_z_scores(features, baseline["statistics"])

        assert "burrows_delta" in z_scores
        assert z_scores["burrows_delta"] >= 0


class TestPOSBigramBaseline:
    def test_default_builds_ten_curated_bigrams(self, nlp):
        from server.stylometry.corpus_baseline import DEFAULT_ROBUST_POS_BIGRAMS

        baseline = build_baseline_from_texts(DOCS, nlp, min_words=TEST_MIN_WORDS)
        built = set(baseline["statistics"]["pos_bigram_ratios"].keys())

        assert built <= set(DEFAULT_ROBUST_POS_BIGRAMS)
        assert built  # at least some of the curated bigrams appear in this small corpus

    def test_empty_list_skips_the_dimension(self, nlp):
        baseline = build_baseline_from_texts(DOCS, nlp, pos_bigrams=[], min_words=TEST_MIN_WORDS)
        assert "pos_bigram_ratios" not in baseline["statistics"]

    def test_custom_subset_is_respected(self, nlp):
        baseline = build_baseline_from_texts(DOCS, nlp, pos_bigrams=["DET_NOUN"], min_words=TEST_MIN_WORDS)
        assert set(baseline["statistics"]["pos_bigram_ratios"].keys()) <= {"DET_NOUN"}


class TestCharNgramBaseline:
    def test_default_builds_a_profile(self, nlp):
        baseline = build_baseline_from_texts(DOCS, nlp, min_words=TEST_MIN_WORDS)
        assert "char_ngram_profile" in baseline["statistics"]
        assert len(baseline["statistics"]["char_ngram_profile"]) > 0

    def test_top_k_bounds_the_profile_size(self, nlp):
        baseline = build_baseline_from_texts(DOCS, nlp, char_ngram_top_k=5, min_words=TEST_MIN_WORDS)
        assert len(baseline["statistics"]["char_ngram_profile"]) <= 5

    def test_zero_top_k_skips_the_dimension(self, nlp):
        baseline = build_baseline_from_texts(DOCS, nlp, char_ngram_top_k=0, min_words=TEST_MIN_WORDS)
        assert "char_ngram_profile" not in baseline["statistics"]

    def test_built_profile_scores_similarity_on_a_real_draft(self, nlp):
        from server.stylometry import StylemetricAnalyzer, calculate_char_ngram_similarity

        baseline = build_baseline_from_texts(DOCS, nlp, min_words=TEST_MIN_WORDS)
        features = StylemetricAnalyzer(nlp).extract_features(DOCS[0])

        similarity = calculate_char_ngram_similarity(features, baseline["statistics"])

        assert similarity is not None
        assert 0.0 <= similarity <= 1.0
