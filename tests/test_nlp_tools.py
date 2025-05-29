"""Test module for NLP analysis tools."""

from unittest.mock import MagicMock

import pytest

from server.analyzers import KeywordAnalyzer, StyleAnalyzer


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
        mock_nlp = MagicMock()
        self.analyzer = KeywordAnalyzer(mock_nlp)

    def test_keyword_density(self, mock_nlp):
        """Test keyword density calculation."""
        self.analyzer.nlp = mock_nlp
        result = self.analyzer.keyword_density("example text", "example")
        assert isinstance(result, (int, float))
        assert result >= 0

    def test_keyword_frequency(self, mock_nlp):
        """Test keyword frequency calculation."""
        self.analyzer.nlp = mock_nlp
        result = self.analyzer.keyword_frequency("example text")
        assert isinstance(result, dict)

    def test_top_keywords(self, mock_nlp):
        """Test top keywords extraction."""
        self.analyzer.nlp = mock_nlp
        result = self.analyzer.top_keywords("example text", top_n=5)
        assert isinstance(result, list)

    def test_keyword_context(self, mock_nlp):
        """Test keyword context extraction."""
        self.analyzer.nlp = mock_nlp
        result = self.analyzer.keyword_context("example text", "example")
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
        self.analyzer.nlp = mock_nlp
        result = self.analyzer.passive_voice_detection("This was written by someone.")
        assert isinstance(result, list)
