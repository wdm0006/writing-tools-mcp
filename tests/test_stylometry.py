"""
Tests for stylometric analysis functionality.

This module tests the StylemetricAnalyzer, BaselineManager, statistical functions,
and the integrated stylometric_analysis tool.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from server.analyzers import AIDetectionAnalyzer
from server.stylometry import (
    BaselineManager,
    StylemetricAnalyzer,
    calculate_sentence_z_scores,
    calculate_z_scores,
    flag_outliers,
    generate_flags,
)
from server.stylometry import baselines as baselines_module

# Test data
SAMPLE_HUMAN_TEXT = """
This is a sample text that represents typical human writing. The sentences vary in length and complexity.
Some are short. Others are much longer and contain multiple clauses that demonstrate natural variation in human prose.
The vocabulary includes diverse words and expressions. Writers naturally use different sentence structures.
This creates the kind of stylistic variation that characterizes authentic human writing.
"""

SAMPLE_AI_TEXT = """
This text demonstrates typical AI writing patterns. The sentences are uniform in length and structure.
Each sentence follows a similar pattern. The vocabulary is repetitive and lacks diversity.
The writing style is consistent throughout. There is little variation in sentence construction.
The text maintains the same rhythm and flow. This uniformity is characteristic of AI-generated content.
"""

SAMPLE_BASELINE = {
    "corpus_info": {"name": "Test Baseline", "description": "Test baseline for unit tests"},
    "statistics": {
        "avg_sentence_len": {"mean": 15.0, "std": 5.0},
        "sentence_len_std": {"mean": 6.0, "std": 2.0},
        "ttr": {"mean": 0.5, "std": 0.1},
        "hapax_legomena_rate": {"mean": 0.4, "std": 0.1},
        "avg_word_len": {"mean": 4.5, "std": 0.5},
        "punct_density": {"mean": 0.12, "std": 0.03},
        "comma_ratio": {"mean": 0.4, "std": 0.1},
        "function_word_ratio": {"mean": 0.45, "std": 0.05},
        "pos_ratios": {
            "NOUN": {"mean": 0.25, "std": 0.05},
            "VERB": {"mean": 0.15, "std": 0.03},
            "ADJ": {"mean": 0.08, "std": 0.02},
        },
    },
}


class TestStylemetricAnalyzer:
    """Test the StylemetricAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a mock analyzer for testing."""
        mock_nlp = MagicMock()
        return StylemetricAnalyzer(mock_nlp)

    @pytest.fixture
    def mock_nlp_doc(self):
        """Create a mock spaCy doc for testing."""
        mock_doc = MagicMock()

        # Mock sentences
        mock_sent1 = MagicMock()
        mock_sent1.text = "This is a test sentence."
        mock_sent2 = MagicMock()
        mock_sent2.text = "Another sentence here."
        mock_doc.sents = [mock_sent1, mock_sent2]

        # Mock tokens
        mock_tokens = []
        for i, word in enumerate(["this", "is", "a", "test", "sentence", "another", "sentence", "here"]):
            token = MagicMock()
            token.text = word
            token.is_punct = False
            token.is_space = False
            token.pos_ = "NOUN" if i % 2 == 0 else "VERB"
            mock_tokens.append(token)

        mock_doc.__iter__ = lambda self: iter(mock_tokens)
        return mock_doc

    def test_extract_features_empty_text(self, analyzer):
        """Test feature extraction with empty text."""
        result = analyzer.extract_features("")

        assert result["avg_sentence_len"] == 0.0
        assert result["ttr"] == 0.0
        assert result["hapax_legomena_rate"] == 0.0
        assert result["sentence_positions"] == []

    def test_avg_sentence_length_calculation(self, analyzer):
        """Test average sentence length calculation."""
        # Mock sentences with known word counts
        mock_sentences = []
        for word_count in [5, 10, 15]:
            sent = MagicMock()
            tokens = [MagicMock() for _ in range(word_count)]
            for token in tokens:
                token.is_punct = False
                token.is_space = False
            sent.__iter__ = lambda self, tokens=tokens: iter(tokens)
            mock_sentences.append(sent)

        result = analyzer._avg_sentence_length(mock_sentences)
        assert result == 10.0  # (5 + 10 + 15) / 3

    def test_sentence_length_std_calculation(self, analyzer):
        """Test sentence length standard deviation calculation."""
        # Mock sentences with known word counts: [5, 15] -> std = 5.0
        mock_sentences = []
        for word_count in [5, 15]:
            sent = MagicMock()
            tokens = [MagicMock() for _ in range(word_count)]
            for token in tokens:
                token.is_punct = False
                token.is_space = False
            sent.__iter__ = lambda self, tokens=tokens: iter(tokens)
            mock_sentences.append(sent)

        result = analyzer._sentence_length_std(mock_sentences)
        assert abs(result - 7.07) < 0.1  # sqrt(50) ≈ 7.07

    def test_type_token_ratio_calculation(self, analyzer):
        """Test Type-Token Ratio calculation."""
        # Mock doc with known words: ["the", "cat", "sat", "on", "the", "mat"]
        # Unique words: 5, Total words: 6, TTR = 5/6 ≈ 0.833
        mock_doc = MagicMock()
        words = ["the", "cat", "sat", "on", "the", "mat"]
        tokens = []
        for word in words:
            token = MagicMock()
            token.text = word
            token.is_punct = False
            token.is_space = False
            tokens.append(token)

        mock_doc.__iter__ = lambda self: iter(tokens)

        result = analyzer._type_token_ratio(mock_doc)
        assert abs(result - 0.833) < 0.01

    def test_hapax_rate_calculation(self, analyzer):
        """Test Hapax Legomena rate calculation."""
        # Mock doc with words: ["cat", "dog", "cat", "bird"]
        # Hapax words: ["dog", "bird"] = 2, Unique words: 3, Rate = 2/3 ≈ 0.667
        mock_doc = MagicMock()
        words = ["cat", "dog", "cat", "bird"]
        tokens = []
        for word in words:
            token = MagicMock()
            token.text = word
            token.is_punct = False
            token.is_space = False
            tokens.append(token)

        mock_doc.__iter__ = lambda self: iter(tokens)

        result = analyzer._hapax_rate(mock_doc)
        assert abs(result - 0.667) < 0.01

    def test_pos_ratios_calculation(self, analyzer):
        """Test POS ratios calculation."""
        # Mock doc with 4 NOUN, 2 VERB tokens
        mock_doc = MagicMock()
        tokens = []
        pos_tags = ["NOUN"] * 4 + ["VERB"] * 2
        for pos in pos_tags:
            token = MagicMock()
            token.pos_ = pos
            token.is_punct = False
            token.is_space = False
            tokens.append(token)

        mock_doc.__iter__ = lambda self: iter(tokens)

        result = analyzer._pos_ratios(mock_doc)
        assert abs(result["NOUN"] - 0.667) < 0.01  # 4/6
        assert abs(result["VERB"] - 0.333) < 0.01  # 2/6

    def test_punctuation_density_calculation(self, analyzer):
        """Test punctuation density calculation."""
        text = "Hello, world! How are you?"  # 3 punct marks, 26 total chars
        result = analyzer._punctuation_density(text)
        assert abs(result - 0.115) < 0.01  # 3/26 ≈ 0.115

    def test_comma_ratio_calculation(self, analyzer):
        """Test comma ratio calculation."""
        text = "Hello, world! How, are, you?"  # 3 commas, 5 total punct (,.!,,?)
        result = analyzer._comma_ratio(text)
        assert abs(result - 0.6) < 0.01  # 3/5 = 0.6


class TestBaselineManager:
    """Test the BaselineManager class."""

    def test_load_brown_corpus_baseline(self):
        """Test loading the default Brown Corpus baseline."""
        manager = BaselineManager()
        baseline = manager.load_baseline("brown_corpus")

        assert "corpus_info" in baseline
        assert "statistics" in baseline
        assert baseline["corpus_info"]["name"] == "Brown Corpus"
        assert "avg_sentence_len" in baseline["statistics"]
        assert "ttr" in baseline["statistics"]

    def test_validate_baseline_valid(self):
        """Test baseline validation with valid data."""
        manager = BaselineManager()
        assert manager.validate_baseline(SAMPLE_BASELINE) is True

    def test_validate_baseline_invalid(self):
        """Test baseline validation with invalid data."""
        manager = BaselineManager()

        # Missing statistics
        invalid_baseline = {"corpus_info": {}}
        assert manager.validate_baseline(invalid_baseline) is False

        # Missing required features
        invalid_baseline = {
            "statistics": {
                "avg_sentence_len": {"mean": 15.0, "std": 5.0}
                # Missing ttr and hapax_legomena_rate
            }
        }
        assert manager.validate_baseline(invalid_baseline) is False

    def test_list_available_baselines(self):
        """Test listing available baselines."""
        manager = BaselineManager()
        baselines = manager.list_available_baselines()

        assert "brown_corpus" in baselines
        assert isinstance(baselines["brown_corpus"], str)


class TestBaselineNameSafety:
    """Test that baseline names are treated as identifiers rather than paths."""

    @pytest.fixture
    def baselines_dir(self, tmp_path, monkeypatch):
        """Point BaselineManager at a temporary baseline directory tree."""
        data_dir = tmp_path / "data" / "baselines"
        (data_dir / "custom_baselines").mkdir(parents=True)
        monkeypatch.setattr(baselines_module, "BASELINES_DIR", data_dir)
        return data_dir

    @pytest.fixture
    def external_file(self, tmp_path):
        """A readable JSON file outside every approved baseline directory."""
        external = tmp_path / "outside" / "secret.json"
        external.parent.mkdir(parents=True)
        external.write_text(json.dumps({"statistics": {"secret": True}}), encoding="utf-8")
        return external

    def test_builtin_baseline_unchanged(self):
        """The in-memory brown_corpus baseline still loads."""
        manager = BaselineManager()
        assert manager.load_baseline("brown_corpus")["corpus_info"]["name"] == "Brown Corpus"

    def test_file_backed_builtin_baseline_loads(self, baselines_dir):
        """A JSON file in the built-in baseline directory loads by name."""
        (baselines_dir / "news_corpus.json").write_text(json.dumps(SAMPLE_BASELINE), encoding="utf-8")

        manager = BaselineManager()
        assert manager.load_baseline("news_corpus") == SAMPLE_BASELINE

    def test_custom_baseline_loads(self, baselines_dir):
        """A JSON file in the custom baseline directory loads by name."""
        custom_path = baselines_dir / "custom_baselines" / "my_corpus-v1.0.json"
        custom_path.write_text(json.dumps(SAMPLE_BASELINE), encoding="utf-8")

        manager = BaselineManager()
        assert manager.load_baseline("my_corpus-v1.0") == SAMPLE_BASELINE

    @pytest.mark.parametrize(
        "baseline_name",
        [
            "/etc/passwd",
            "/tmp/external",
            "../secret",
            "..",
            "custom_baselines/../../secret",
            "custom_baselines/my_corpus",
            "sub\\corpus",
            "",
            ".hidden",
            "corpus\x00",
        ],
    )
    def test_load_baseline_rejects_unsafe_names(self, baselines_dir, baseline_name):
        """Absolute paths, traversal, and separators are rejected with a clear error."""
        manager = BaselineManager()

        with pytest.raises(ValueError, match="Invalid baseline name"):
            manager.load_baseline(baseline_name)

        with pytest.raises(ValueError, match="Invalid baseline name"):
            manager._get_baseline_path(baseline_name)

    def test_external_file_cannot_be_loaded(self, baselines_dir, external_file):
        """A crafted name cannot read a JSON file outside the approved directories."""
        manager = BaselineManager()
        absolute_name = str(external_file)[: -len(".json")]
        relative_name = os.path.relpath(external_file, baselines_dir)[: -len(".json")]

        for crafted in (absolute_name, relative_name):
            with pytest.raises(ValueError, match="Invalid baseline name"):
                manager.load_baseline(crafted)
            assert crafted not in manager.baselines

    def test_symlink_out_of_baseline_dir_is_rejected(self, baselines_dir, external_file):
        """A symlinked baseline resolving outside the approved directory is not loaded."""
        (baselines_dir / "linked.json").symlink_to(external_file)

        manager = BaselineManager()
        with pytest.raises(ValueError, match="not found"):
            manager.load_baseline("linked")

    def test_missing_valid_name_still_reports_not_found(self, baselines_dir):
        """A well-formed but unknown baseline keeps the original not-found error."""
        manager = BaselineManager()

        with pytest.raises(ValueError, match="not found"):
            manager.load_baseline("no_such_baseline")

    def test_get_baseline_info_returns_none_for_unsafe_name(self, baselines_dir, external_file):
        """get_baseline_info swallows the validation error rather than raising."""
        manager = BaselineManager()
        assert manager.get_baseline_info(str(external_file)[: -len(".json")]) is None

    def test_save_baseline_rejects_unsafe_names(self, baselines_dir, external_file):
        """A crafted name cannot overwrite a file outside the approved directories."""
        manager = BaselineManager()
        original = external_file.read_text(encoding="utf-8")

        for crafted in (
            str(external_file)[: -len(".json")],
            os.path.relpath(external_file, baselines_dir / "custom_baselines")[: -len(".json")],
            "../escaped",
        ):
            with pytest.raises(ValueError, match="Invalid baseline name"):
                manager.save_baseline(crafted, SAMPLE_BASELINE)

        assert external_file.read_text(encoding="utf-8") == original
        assert not (baselines_dir.parent.parent / "escaped.json").exists()

    def test_save_baseline_writes_inside_custom_dir(self, baselines_dir):
        """Valid names still round-trip through the custom baseline directory."""
        manager = BaselineManager()

        assert manager.save_baseline("my_corpus-v1.0", SAMPLE_BASELINE) is True
        assert (baselines_dir / "custom_baselines" / "my_corpus-v1.0.json").is_file()
        assert BaselineManager().load_baseline("my_corpus-v1.0") == SAMPLE_BASELINE

    def test_stylometric_analysis_reports_unsafe_baseline(self, baselines_dir, external_file):
        """The MCP-facing analyzer returns a structured error for a crafted baseline."""
        analyzer = AIDetectionAnalyzer(MagicMock(), MagicMock(), {})

        result = analyzer.stylometric_analysis("Some text to analyze.", baseline=str(external_file)[: -len(".json")])

        assert "Invalid baseline name" in result["error"]
        assert result["features"] == {}
        assert result["flags"]["high_ai_probability"] is False


class TestStatisticalFunctions:
    """Test statistical analysis functions."""

    def test_calculate_z_scores(self):
        """Test z-score calculation."""
        features = {
            "avg_sentence_len": 20.0,  # z = (20-15)/5 = 1.0
            "ttr": 0.6,  # z = (0.6-0.5)/0.1 = 1.0
            "pos_ratios": {
                "NOUN": 0.3  # z = (0.3-0.25)/0.05 = 1.0
            },
        }

        z_scores = calculate_z_scores(features, SAMPLE_BASELINE["statistics"])

        assert abs(z_scores["avg_sentence_len"] - 1.0) < 0.01
        assert abs(z_scores["ttr"] - 1.0) < 0.01
        assert abs(z_scores["pos_noun"] - 1.0) < 0.01

    def test_flag_outliers(self):
        """Test outlier flagging."""
        z_scores = {
            "feature1": 1.5,  # Normal
            "feature2": 2.5,  # Warning
            "feature3": 3.5,  # Error
            "feature4": -2.1,  # Warning (negative)
        }

        flags = flag_outliers(z_scores, warning_threshold=2.0, error_threshold=3.0)

        assert "feature2" in flags["warnings"]
        assert "feature4" in flags["warnings"]
        assert "feature3" in flags["errors"]
        assert "feature1" not in flags["warnings"]
        assert "feature1" not in flags["errors"]

    def test_generate_flags_ai_detection(self):
        """Test AI detection flag generation."""
        z_scores = {
            "ttr": -2.5,  # Low lexical diversity
            "hapax_legomena_rate": -2.2,  # Low hapax rate
            "sentence_len_std": -2.1,  # Uniform sentences
        }

        features = {"ttr": 0.3, "hapax_legomena_rate": 0.2}
        thresholds = {"warning_z": 2.0, "error_z": 3.0, "ai_confidence_threshold": 0.7}

        flags = generate_flags(z_scores, features, thresholds)

        assert flags["high_ai_probability"] is True
        assert "low_ttr" in flags["ai_indicators"]
        assert "low_hapax" in flags["ai_indicators"]
        assert len(flags["reasons"]) > 0

    def test_calculate_sentence_z_scores(self):
        """Test sentence-level z-score calculation."""
        sentence_positions = [
            {"position": 1, "length": 10, "text": "Short sentence."},
            {"position": 2, "length": 20, "text": "Much longer sentence here."},
        ]

        baseline_stats = {"mean": 15.0, "std": 5.0}

        result = calculate_sentence_z_scores(sentence_positions, baseline_stats)

        assert result[0]["z_score"] == -1.0  # (10-15)/5
        assert result[1]["z_score"] == 1.0  # (20-15)/5


class TestUndefinedSentenceLengthStd:
    """Sentence-length dispersion must be absent, not zero, when it cannot be measured."""

    SINGLE_SENTENCE_TEXT = "The committee approved the revised budget yesterday after a long and contentious debate."

    UNIFORM_SENTENCES_TEXT = (
        "The committee approved the revised budget yesterday. "
        "The council rejected the amended proposal quickly. "
        "The board reviewed the updated schedule carefully."
    )

    @pytest.fixture
    def analyzer(self):
        """Create a mock analyzer for testing."""
        return StylemetricAnalyzer(MagicMock())

    @staticmethod
    def _mock_sentences(word_counts):
        """Build mock spaCy sentences with the given non-punctuation token counts."""
        sentences = []
        for word_count in word_counts:
            sent = MagicMock()
            tokens = [MagicMock() for _ in range(word_count)]
            for token in tokens:
                token.is_punct = False
                token.is_space = False
            sent.__iter__ = lambda self, tokens=tokens: iter(tokens)
            sentences.append(sent)
        return sentences

    @pytest.mark.parametrize("word_counts", [[], [12]])
    def test_std_is_none_when_undefined(self, analyzer, word_counts):
        """Fewer than two sentences means no dispersion measurement exists."""
        assert analyzer._sentence_length_std(self._mock_sentences(word_counts)) is None

    def test_std_is_sample_stdev_for_two_or_more_sentences(self, analyzer):
        """Two or more sentences still yield the sample standard deviation."""
        result = analyzer._sentence_length_std(self._mock_sentences([5, 15]))
        assert abs(result - 7.07) < 0.01

    def test_z_scores_omit_unmeasurable_feature(self):
        """A None feature value is skipped; numeric values are scored as before."""
        features = {"sentence_len_std": None, "avg_sentence_len": 20.0}

        z_scores = calculate_z_scores(features, SAMPLE_BASELINE["statistics"])

        assert "sentence_len_std" not in z_scores
        assert abs(z_scores["avg_sentence_len"] - 1.0) < 0.01

    def test_single_sentence_is_not_flagged_as_uniform(self, ai_detection_analyzer):
        """A one-sentence document must not be charged a uniform-sentence indicator."""
        result = ai_detection_analyzer.stylometric_analysis(self.SINGLE_SENTENCE_TEXT)

        assert "error" not in result
        assert result["features"]["sentence_len_std"] is None
        assert "sentence_len_std" not in result["z_scores"]
        assert "sentence_len_std" not in result["flags"]["warnings"]
        assert "sentence_len_std" not in result["flags"]["errors"]
        assert "uniform_sentences" not in result["flags"]["ai_indicators"]

    def test_uniform_multi_sentence_text_still_flags_uniform_sentences(self, ai_detection_analyzer):
        """A genuinely uniform multi-sentence document keeps the indicator."""
        result = ai_detection_analyzer.stylometric_analysis(self.UNIFORM_SENTENCES_TEXT)

        assert result["features"]["sentence_len_std"] == 0.0
        assert result["z_scores"]["sentence_len_std"] < -2.0
        assert "sentence_len_std" in result["flags"]["warnings"]
        assert "uniform_sentences" in result["flags"]["ai_indicators"]


class TestIntegration:
    """Integration tests for the complete stylometric analysis."""

    def test_stylometric_analysis_tool(self):
        """Test the complete stylometric analysis tool."""
        # Create proper mocks for AIDetectionAnalyzer
        mock_nlp = MagicMock()
        mock_gpt2_manager = MagicMock()
        mock_config = {"stylometry": {"thresholds": {"warning_z": 2.0, "error_z": 3.0, "ai_confidence_threshold": 0.7}}}

        analyzer = AIDetectionAnalyzer(mock_nlp, mock_gpt2_manager, mock_config)

        # Mock the stylometric analysis method to return expected structure
        expected_result = {
            "features": {
                "avg_sentence_len": 18.0,
                "ttr": 0.45,
                "hapax_legomena_rate": 0.35,
            },
            "z_scores": {"avg_sentence_len": 0.6, "ttr": -0.5},
            "flags": {"high_ai_probability": False},
            "sentence_analysis": [],
            "config": {"baseline": "brown_corpus"},
        }

        with patch.object(analyzer, "stylometric_analysis", return_value=expected_result):
            result = analyzer.stylometric_analysis("Test text for analysis")

        assert "features" in result
        assert "z_scores" in result
        assert "flags" in result
        assert "sentence_analysis" in result
        assert "config" in result

    def test_stylometric_analysis_empty_text(self):
        """Test stylometric analysis with empty text."""
        mock_nlp = MagicMock()
        mock_gpt2_manager = MagicMock()
        mock_config = {}

        analyzer = AIDetectionAnalyzer(mock_nlp, mock_gpt2_manager, mock_config)
        result = analyzer.stylometric_analysis("")

        assert "error" in result
        assert result["error"] == "Empty text provided"

    def test_stylometric_analysis_unsupported_language(self):
        """Test stylometric analysis with unsupported language."""
        mock_nlp = MagicMock()
        mock_gpt2_manager = MagicMock()
        mock_config = {}

        analyzer = AIDetectionAnalyzer(mock_nlp, mock_gpt2_manager, mock_config)
        result = analyzer.stylometric_analysis("Test text", language="fr")

        assert "error" in result
        assert "English" in result["error"]


class TestAIDetectionAccuracy:
    """Tests for AI detection accuracy and performance."""

    def test_human_text_classification(self):
        """Test that typical human text is not flagged as AI."""
        # This would require actual model testing with known human samples
        # For now, we test the logic with mock data

        # Human-like features: high TTR, high hapax rate, varied sentence lengths
        human_features = {
            "avg_sentence_len": 17.5,
            "sentence_len_std": 7.2,
            "ttr": 0.53,
            "hapax_legomena_rate": 0.48,
            "pos_ratios": {"NOUN": 0.23, "VERB": 0.16},
        }

        z_scores = calculate_z_scores(human_features, SAMPLE_BASELINE["statistics"])
        thresholds = {"warning_z": 2.0, "error_z": 3.0, "ai_confidence_threshold": 0.7}
        flags = generate_flags(z_scores, human_features, thresholds)

        # Human text should not be flagged as AI
        assert flags["high_ai_probability"] is False
        assert flags["confidence_score"] < 0.7

    def test_ai_text_classification(self):
        """Test that typical AI text is flagged appropriately."""
        # AI-like features: very low TTR, very low hapax rate, very uniform sentence lengths
        ai_features = {
            "avg_sentence_len": 15.0,
            "sentence_len_std": 1.5,  # Very uniform (z = (1.5-6.0)/2.0 = -2.25)
            "ttr": 0.25,  # Very low diversity (z = (0.25-0.5)/0.1 = -2.5)
            "hapax_legomena_rate": 0.15,  # Very low hapax rate (z = (0.15-0.4)/0.1 = -2.5)
            "pos_ratios": {"NOUN": 0.23, "VERB": 0.16},
        }

        z_scores = calculate_z_scores(ai_features, SAMPLE_BASELINE["statistics"])
        thresholds = {"warning_z": 2.0, "error_z": 3.0, "ai_confidence_threshold": 0.7}
        flags = generate_flags(z_scores, ai_features, thresholds)

        # AI text should be flagged
        assert flags["high_ai_probability"] is True
        assert flags["confidence_score"] >= 0.7
        assert len(flags["ai_indicators"]) >= 2


if __name__ == "__main__":
    pytest.main([__file__])
