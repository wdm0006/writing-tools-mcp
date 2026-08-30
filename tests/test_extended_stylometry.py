"""Tests for the length-robust lexical diversity, syntactic, repetition, and
Burrows'-Delta-style features added on top of the original stylometric feature set.
"""

from server.stylometry import (
    StylemetricAnalyzer,
    calculate_char_ngram_similarity,
    calculate_z_scores,
    generate_flags,
)

REPETITIVE_TEXT = " ".join(["the cat sat on the mat and the cat slept"] * 8)

DIVERSE_TEXT = (
    "Rain fell across the valley for three straight days, and farmers watched the river "
    "rise past the old stone bridge, worried about the low fields near the mill. Engineers "
    "traced a strange noise in the new engine to a bearing that had been over-torqued at "
    "assembly, and the fix shipped within a week once the root cause was confirmed by testing. "
    "She spent the summer rebuilding a porch her grandfather had built decades earlier, and by "
    "August the railing stood square again, a small stubborn victory against years of weather "
    "that had worn the old one down considerably over time."
)

SUBORDINATE_TEXT = (
    "Because the market moved before the close, the model looked better than it should have, "
    "although the underlying edge had not actually changed."
)

NO_SUBORDINATE_TEXT = "The market moved. The model looked good. The edge stayed the same."

REPEATED_PHRASE_TEXT = "run the backtest again and run the backtest again to be sure of the result"
UNIQUE_TEXT = "each of these seven words differs entirely from every other one nearby"


class TestLengthRobustLexicalDiversity:
    def test_mtld_none_below_fifty_tokens(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features("This is a short document, nowhere near fifty words long.")
        assert features["mtld"] is None

    def test_mtld_lower_for_repetitive_text(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        repetitive = analyzer.extract_features(REPETITIVE_TEXT)
        diverse = analyzer.extract_features(DIVERSE_TEXT)

        assert repetitive["mtld"] is not None
        assert diverse["mtld"] is not None
        assert repetitive["mtld"] < diverse["mtld"]

    def test_mattr_none_below_window(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features("Too few words here for a fifty token window.")
        assert features["mattr"] is None

    def test_mattr_lower_for_repetitive_text(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        repetitive = analyzer.extract_features(REPETITIVE_TEXT)
        diverse = analyzer.extract_features(DIVERSE_TEXT)

        assert repetitive["mattr"] is not None
        assert diverse["mattr"] is not None
        assert repetitive["mattr"] < diverse["mattr"]


class TestReadabilityAdditions:
    def test_new_readability_scores_present(self, ai_detection_analyzer):
        result = ai_detection_analyzer.stylometric_analysis(DIVERSE_TEXT)
        features = result["features"]

        for key in ("smog", "coleman_liau", "ari", "dale_chall"):
            assert features[key] is not None


class TestStylometricAnalysisCharNgramWiring:
    def test_char_ngram_similarity_none_without_baseline_profile(self, ai_detection_analyzer):
        """brown_corpus has no char_ngram_profile, so the tool-level result is None,
        not an error - actual similarity computation is covered in test_corpus_baseline.py."""
        result = ai_detection_analyzer.stylometric_analysis(DIVERSE_TEXT, baseline="brown_corpus")

        assert "error" not in result
        assert result["char_ngram_similarity"] is None

    def test_char_ngram_profile_excluded_from_returned_features(self, ai_detection_analyzer):
        """The raw several-hundred-entry profile is an internal intermediate, not
        something a caller needs to see key-by-key in the features dict."""
        result = ai_detection_analyzer.stylometric_analysis(DIVERSE_TEXT)

        assert "char_ngram_profile" not in result["features"]


class TestSyntacticComplexity:
    def test_mean_dependency_distance_is_positive(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features(DIVERSE_TEXT)
        assert features["mean_dependency_distance"] is not None
        assert features["mean_dependency_distance"] > 0

    def test_subordinate_clause_ratio_detects_subordination(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        with_sub = analyzer.extract_features(SUBORDINATE_TEXT)
        without_sub = analyzer.extract_features(NO_SUBORDINATE_TEXT)

        assert with_sub["subordinate_clause_ratio"] > without_sub["subordinate_clause_ratio"]
        assert without_sub["subordinate_clause_ratio"] == 0.0


class TestRepetitionAndZipf:
    def test_fourgram_repetition_rate_detects_repeats(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        repeated = analyzer.extract_features(REPEATED_PHRASE_TEXT)
        unique = analyzer.extract_features(UNIQUE_TEXT)

        assert repeated["fourgram_repetition_rate"] > 0
        assert unique["fourgram_repetition_rate"] == 0.0

    def test_fourgram_repetition_rate_none_for_short_text(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features("Two three.")
        assert features["fourgram_repetition_rate"] is None

    def test_zipf_slope_negative_for_natural_text(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features(DIVERSE_TEXT)
        assert features["zipf_slope"] is not None
        assert features["zipf_slope"] < 0

    def test_zipf_slope_none_for_short_text(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features("Only a few words appear right here.")
        assert features["zipf_slope"] is None


class TestFunctionWordFreqs:
    def test_every_tracked_word_gets_an_entry(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features(DIVERSE_TEXT)

        assert set(features["function_word_freqs"].keys()) == analyzer.function_words

    def test_absent_word_is_a_real_zero(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features("Colorless green ideas sleep furiously near quiet rivers.")
        # "day" is a tracked function word that does not appear in this sentence.
        assert features["function_word_freqs"]["day"] == 0.0


class TestBurrowsDeltaScoring:
    def test_calculate_z_scores_scores_each_word_and_reduces_to_delta(self):
        baseline = {
            "function_word_freqs": {
                "the": {"mean": 0.05, "std": 0.01},
                "of": {"mean": 0.02, "std": 0.01},
            }
        }
        features = {"function_word_freqs": {"the": 0.07, "of": 0.02}}  # z=2.0, z=0.0

        z_scores = calculate_z_scores(features, baseline)

        assert abs(z_scores["fw_the"] - 2.0) < 0.01
        assert abs(z_scores["fw_of"] - 0.0) < 0.01
        assert abs(z_scores["burrows_delta"] - 1.0) < 0.01  # mean(|2.0|, |0.0|)

    def test_burrows_delta_absent_without_baseline(self):
        features = {"function_word_freqs": {"the": 0.07}}
        z_scores = calculate_z_scores(features, {})
        assert "burrows_delta" not in z_scores

    def test_generate_flags_low_mtld(self):
        thresholds = {"warning_z": 2.0, "error_z": 3.0, "ai_confidence_threshold": 0.7}
        flags = generate_flags({"mtld": -2.5}, {"mtld": 40.0}, thresholds)

        assert "low_mtld" in flags["ai_indicators"]

    def test_generate_flags_distinct_function_word_profile(self):
        thresholds = {"warning_z": 2.0, "error_z": 3.0, "ai_confidence_threshold": 0.7}
        flags = generate_flags({"burrows_delta": 2.5}, {}, thresholds)

        assert "distinct_function_word_profile" in flags["ai_indicators"]

    def test_generate_flags_low_burrows_delta_not_flagged(self):
        """A small Burrows' Delta (author matches baseline) must not trigger the indicator."""
        thresholds = {"warning_z": 2.0, "error_z": 3.0, "ai_confidence_threshold": 0.7}
        flags = generate_flags({"burrows_delta": 0.5}, {}, thresholds)

        assert "distinct_function_word_profile" not in flags["ai_indicators"]


RARE_VOCAB_TEXT = (
    "Sesquipedalian loquaciousness pervaded the peroration, replete with abstruse "
    "circumlocutions that obfuscated any perspicuous meaning whatsoever, rendering the "
    "prolix disquisition nearly incomprehensible to the assembled cognoscenti."
)

COMMON_VOCAB_TEXT = (
    "The dog ran to the door and the cat sat by the window. It was a good day for a walk, "
    "and the kids went out to play in the yard with the ball."
)

BRITISH_STYLE_TEXT = (
    "We tend to organise our colour palette around a few favourites, and we recognise the "
    "risk that some readers may not realise how much analysing the data actually takes."
)


class TestVocabularySophistication:
    def test_mtld_lemma_present_for_long_text(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features(DIVERSE_TEXT * 2)
        assert features["mtld_lemma"] is not None

    def test_mtld_lemma_none_below_fifty_tokens(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features("This is a short document, nowhere near fifty words long.")
        assert features["mtld_lemma"] is None

    def test_mean_word_frequency_lower_for_rare_vocabulary(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        rare = analyzer.extract_features(RARE_VOCAB_TEXT)
        common = analyzer.extract_features(COMMON_VOCAB_TEXT)

        assert rare["mean_word_frequency"] is not None
        assert common["mean_word_frequency"] is not None
        assert rare["mean_word_frequency"] < common["mean_word_frequency"]

    def test_word_len_std_none_for_single_word(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features("Word.")
        assert features["word_len_std"] is None

    def test_lexical_density_in_unit_range(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features(DIVERSE_TEXT)
        assert 0.0 <= features["lexical_density"] <= 1.0


class TestPOSBigrams:
    def test_pos_bigram_ratios_sum_to_approximately_one(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features(DIVERSE_TEXT)
        assert abs(sum(features["pos_bigram_ratios"].values()) - 1.0) < 0.01

    def test_calculate_z_scores_handles_pos_bigrams(self):
        baseline = {"pos_bigram_ratios": {"DET_NOUN": {"mean": 0.1, "std": 0.02}}}
        features = {"pos_bigram_ratios": {"DET_NOUN": 0.14}}  # z = 2.0

        z_scores = calculate_z_scores(features, baseline)

        assert abs(z_scores["posbi_det_noun"] - 2.0) < 0.01

    def test_generate_flags_pos_anomalies_covers_bigrams(self):
        thresholds = {"warning_z": 2.0, "error_z": 3.0, "ai_confidence_threshold": 0.7}
        flags = generate_flags({"posbi_det_noun": 2.5}, {}, thresholds)

        assert "pos_anomalies" in flags["ai_indicators"]
        assert any("DET_NOUN" in reason for reason in flags["reasons"])


class TestPunctuationIdiosyncrasies:
    def test_semicolon_ratio_detects_semicolons(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        with_semi = analyzer.extract_features("This works; that also works. Fine, then.")
        without_semi = analyzer.extract_features("This works. That also works. Fine, then.")

        assert with_semi["semicolon_ratio"] > without_semi["semicolon_ratio"]
        assert without_semi["semicolon_ratio"] == 0.0

    def test_em_dash_ratio_detects_em_dashes(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features("This works — that also works. Fine, then.")
        assert features["em_dash_ratio"] > 0.0

    def test_ellipsis_ratio_detects_both_forms(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        literal = analyzer.extract_features("Well... that happened. Sure, fine.")
        unicode_form = analyzer.extract_features("Well… that happened. Sure, fine.")

        assert literal["ellipsis_ratio"] > 0.0
        assert unicode_form["ellipsis_ratio"] > 0.0

    def test_exclamation_ratio_detects_exclamations(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features("This is great! Really great. Truly.")
        assert features["exclamation_ratio"] > 0.0

    def test_parenthetical_rate_detects_parens(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        with_parens = analyzer.extract_features("This works (mostly). That also works. Fine.")
        without_parens = analyzer.extract_features("This works fine. That also works. Fine.")

        assert with_parens["parenthetical_rate"] > without_parens["parenthetical_rate"]
        assert without_parens["parenthetical_rate"] == 0.0


class TestHedgeAndBoosterRate:
    def test_hedge_rate_detects_hedge_words(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        hedged = analyzer.extract_features("This might possibly work, and it could perhaps help.")
        plain = analyzer.extract_features("This will work, and it will help everyone here.")

        assert hedged["hedge_rate"] > plain["hedge_rate"]
        assert plain["hedge_rate"] == 0.0

    def test_booster_rate_detects_booster_words(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        boosted = analyzer.extract_features("This definitely and certainly always works perfectly.")
        plain = analyzer.extract_features("This might possibly work in some cases here.")

        assert boosted["booster_rate"] > plain["booster_rate"]
        assert plain["booster_rate"] == 0.0


class TestCharNgramProfile:
    def test_profile_empty_for_very_short_text(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features("Hi.")
        assert features["char_ngram_profile"] == {}

    def test_profile_frequencies_sum_to_one(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features(DIVERSE_TEXT)
        assert abs(sum(features["char_ngram_profile"].values()) - 1.0) < 0.01

    def test_identical_profile_has_similarity_near_one(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features(DIVERSE_TEXT)
        baseline = {"char_ngram_profile": features["char_ngram_profile"]}

        similarity = calculate_char_ngram_similarity(features, baseline)

        assert similarity is not None
        assert similarity > 0.99

    def test_similarity_none_without_baseline_profile(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features(DIVERSE_TEXT)

        assert calculate_char_ngram_similarity(features, {}) is None

    def test_similarity_lower_for_different_profile(self, nlp):
        analyzer = StylemetricAnalyzer(nlp)
        features = analyzer.extract_features(DIVERSE_TEXT)
        other_profile = analyzer.extract_features(RARE_VOCAB_TEXT)["char_ngram_profile"]

        same = calculate_char_ngram_similarity(features, {"char_ngram_profile": features["char_ngram_profile"]})
        different = calculate_char_ngram_similarity(features, {"char_ngram_profile": other_profile})

        assert different < same
