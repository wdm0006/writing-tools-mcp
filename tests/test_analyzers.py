"""Test module for all analyzer modules."""

from unittest.mock import MagicMock, patch

import pytest

from server.analyzers import (
    AIDetectionAnalyzer,
    BasicStatsAnalyzer,
    KeywordAnalyzer,
    ReadabilityAnalyzer,
    StyleAnalyzer,
)


class TestBasicStatsAnalyzer:
    """Test the BasicStatsAnalyzer class."""

    def setup_method(self):
        """Set up BasicStatsAnalyzer instance for testing."""
        self.analyzer = BasicStatsAnalyzer()

    def test_character_count(self):
        """Test character count functionality."""
        assert self.analyzer.character_count("hello") == 5
        assert self.analyzer.character_count("hello world") == 11
        assert self.analyzer.character_count("") == 0

    def test_word_count(self):
        """Test word count functionality."""
        assert self.analyzer.word_count("hello world") == 2
        assert self.analyzer.word_count("hello") == 1
        assert self.analyzer.word_count("") == 0
        assert self.analyzer.word_count("  ") == 0

    @patch("server.analyzers.basic_stats.SpellChecker")
    def test_spellcheck(self, mock_spell_checker):
        """Test spellcheck functionality."""
        # Mock the spell checker
        mock_checker = MagicMock()
        mock_checker.unknown.return_value = {"wrng", "speling"}
        mock_spell_checker.return_value = mock_checker

        result = self.analyzer.spellcheck("This is wrng speling")
        assert isinstance(result, list)


class TestReadabilityAnalyzer:
    """Test the ReadabilityAnalyzer class."""

    def setup_method(self):
        """Set up ReadabilityAnalyzer instance for testing."""
        self.analyzer = ReadabilityAnalyzer()

    def test_readability_score_full(self):
        """Test readability scoring for full text."""
        text = "This is a simple sentence. This is another simple sentence."
        result = self.analyzer.readability_score(text, level="full")

        assert isinstance(result, dict)
        assert "flesch" in result
        assert "kincaid" in result
        assert "fog" in result

    def test_readability_score_empty_text(self):
        """Test readability scoring with empty text."""
        result = self.analyzer.readability_score("", level="full")

        assert result["flesch"] is None
        assert result["kincaid"] is None
        assert result["fog"] is None

    def test_reading_time_full(self):
        """Test reading time estimation for full text."""
        text = "This is a simple sentence. This is another simple sentence."
        result = self.analyzer.reading_time(text, level="full")

        assert isinstance(result, dict)
        assert "full_text" in result
        assert isinstance(result["full_text"], (int, float))


class TestKeywordAnalyzer:
    """Test the KeywordAnalyzer class."""

    @pytest.fixture
    def mock_nlp(self):
        """Create mock spaCy model."""
        mock_nlp = MagicMock()

        # Mock document with tokens
        mock_doc = MagicMock()
        mock_token1 = MagicMock()
        mock_token1.text = "example"
        mock_token1.lemma_ = "example"
        mock_token1.is_stop = False
        mock_token1.is_alpha = True
        mock_token1.pos_ = "NOUN"

        mock_token2 = MagicMock()
        mock_token2.text = "text"
        mock_token2.lemma_ = "text"
        mock_token2.is_stop = False
        mock_token2.is_alpha = True
        mock_token2.pos_ = "NOUN"

        mock_doc.__iter__ = lambda self: iter([mock_token1, mock_token2])
        mock_nlp.return_value = mock_doc

        return mock_nlp

    def setup_method(self):
        """Set up KeywordAnalyzer instance for testing."""
        # Create mock nlp for initialization
        mock_nlp = MagicMock()
        self.analyzer = KeywordAnalyzer(mock_nlp)

    def test_keyword_density(self, mock_nlp):
        """Test keyword density calculation."""
        # Update the analyzer's nlp model
        self.analyzer.nlp = mock_nlp

        result = self.analyzer.keyword_density("example text", "example")
        assert isinstance(result, (int, float))
        assert result >= 0

    def test_keyword_frequency(self, mock_nlp):
        """Test keyword frequency calculation."""
        # Update the analyzer's nlp model
        self.analyzer.nlp = mock_nlp

        result = self.analyzer.keyword_frequency("example text")
        assert isinstance(result, dict)

    def test_top_keywords(self, mock_nlp):
        """Test top keywords extraction."""
        # Update the analyzer's nlp model
        self.analyzer.nlp = mock_nlp

        result = self.analyzer.top_keywords("example text", top_n=5)
        assert isinstance(result, list)


class TestStyleAnalyzer:
    """Test the StyleAnalyzer class."""

    @pytest.fixture
    def mock_nlp(self):
        """Create mock spaCy model."""
        mock_nlp = MagicMock()

        # Mock document with sentences
        mock_doc = MagicMock()
        mock_sent = MagicMock()
        mock_sent.text = "This is a sentence."

        # Mock tokens in sentence
        mock_token1 = MagicMock()
        mock_token1.pos_ = "AUX"  # auxiliary verb
        mock_token1.tag_ = "VBZ"
        mock_token1.dep_ = "aux"

        mock_token2 = MagicMock()
        mock_token2.pos_ = "VERB"
        mock_token2.tag_ = "VBN"  # past participle
        mock_token2.dep_ = "ROOT"

        mock_sent.__iter__ = lambda self: iter([mock_token1, mock_token2])
        mock_doc.sents = [mock_sent]
        mock_nlp.return_value = mock_doc

        return mock_nlp

    def setup_method(self):
        """Set up StyleAnalyzer instance for testing."""
        mock_nlp = MagicMock()
        self.analyzer = StyleAnalyzer(mock_nlp)

    def test_passive_voice_detection(self, mock_nlp):
        """Test passive voice detection."""
        # Update the analyzer's nlp model
        self.analyzer.nlp = mock_nlp

        result = self.analyzer.passive_voice_detection("This was written by someone.")
        assert isinstance(result, list)


class TestAIDetectionAnalyzer:
    """Test the AIDetectionAnalyzer class."""

    @pytest.fixture
    def mock_nlp(self):
        """Create mock spaCy model."""
        return MagicMock()

    @pytest.fixture
    def mock_gpt2_manager(self):
        """Create mock GPT2Manager."""
        mock_manager = MagicMock()
        mock_manager.get_model_and_tokenizer.return_value = (
            MagicMock(),  # model
            MagicMock(),  # tokenizer
            {
                "model_name": "gpt2",
                "max_length": 512,
                "overlap": 50,
                "thresholds": {"ppl_max": 50, "burstiness_min": 1.0},
            },
        )
        return mock_manager

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {"stylometry": {"thresholds": {"warning_z": 2.0, "error_z": 3.0, "ai_confidence_threshold": 0.7}}}

    def test_init(self, mock_nlp, mock_gpt2_manager, mock_config):
        """Test AIDetectionAnalyzer initialization."""
        analyzer = AIDetectionAnalyzer(mock_nlp, mock_gpt2_manager, mock_config)
        assert analyzer.nlp == mock_nlp
        assert analyzer.gpt2_manager == mock_gpt2_manager
        assert analyzer.config == mock_config

    def test_perplexity_analysis_unsupported_language(self, mock_nlp, mock_gpt2_manager, mock_config):
        """Test perplexity analysis with unsupported language."""
        analyzer = AIDetectionAnalyzer(mock_nlp, mock_gpt2_manager, mock_config)
        result = analyzer.perplexity_analysis("Test text", language="fr")

        assert "error" in result
        assert result["error"] == "Only English language ('en') is currently supported"

    def test_perplexity_analysis_empty_text(self, mock_nlp, mock_gpt2_manager, mock_config):
        """Test perplexity analysis with empty text."""
        analyzer = AIDetectionAnalyzer(mock_nlp, mock_gpt2_manager, mock_config)
        result = analyzer.perplexity_analysis("", language="en")

        assert "error" in result
        assert result["error"] == "Empty text provided"

    def test_stylometric_analysis_unsupported_language(self, mock_nlp, mock_gpt2_manager, mock_config):
        """Test stylometric analysis with unsupported language."""
        analyzer = AIDetectionAnalyzer(mock_nlp, mock_gpt2_manager, mock_config)
        result = analyzer.stylometric_analysis("Test text", language="fr")

        assert "error" in result
        assert result["error"] == "Only English language ('en') is currently supported"

    def test_stylometric_analysis_empty_text(self, mock_nlp, mock_gpt2_manager, mock_config):
        """Test stylometric analysis with empty text."""
        analyzer = AIDetectionAnalyzer(mock_nlp, mock_gpt2_manager, mock_config)
        result = analyzer.stylometric_analysis("", language="en")

        assert "error" in result
        assert result["error"] == "Empty text provided"
