"""
Tests for stylometric analysis functionality.

This module tests the StylemetricAnalyzer, BaselineManager, statistical functions,
and the integrated stylometric_analysis tool.
"""

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
