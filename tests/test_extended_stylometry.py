"""Tests for the length-robust lexical diversity, syntactic, repetition, and
Burrows'-Delta-style features added on top of the original stylometric feature set.
"""

from server.stylometry import StylemetricAnalyzer, calculate_z_scores, generate_flags

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
