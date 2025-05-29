"""
Tests for perplexity analysis functionality.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from server.server import (
    _calculate_burstiness,
    _calculate_perplexity,
    _chunk_text,
    _split_into_sentences,
    load_config,
    perplexity_analysis,
)


class TestPerplexityBasics:
    """Test basic perplexity analysis functionality"""

    @patch("server.server.nlp")
    def test_split_into_sentences_basic(self, mock_nlp):
        """Test sentence splitting functionality"""
        # Mock spaCy doc and sentences
        mock_doc = Mock()
        mock_sent1 = Mock()
        mock_sent1.text = "This is the first sentence."
        mock_sent2 = Mock()
        mock_sent2.text = "This is the second sentence!"
        mock_sent3 = Mock()
        mock_sent3.text = "And a third one?"
        mock_doc.sents = [mock_sent1, mock_sent2, mock_sent3]
        mock_nlp.return_value = mock_doc

        text = "This is the first sentence. This is the second sentence! And a third one?"
        sentences = _split_into_sentences(text)

        assert len(sentences) == 3
        assert "This is the first sentence." in sentences
        assert "This is the second sentence!" in sentences
        assert "And a third one?" in sentences

    @patch("server.server.nlp")
    def test_split_into_sentences_empty(self, mock_nlp):
        """Test sentence splitting with empty text"""
        # Mock spaCy doc with no sentences
        mock_doc = Mock()
        mock_doc.sents = []
        mock_nlp.return_value = mock_doc

        sentences = _split_into_sentences("")
        assert sentences == []

    def test_calculate_burstiness_basic(self):
        """Test burstiness calculation with valid data"""
        perplexities = [10.0, 15.0, 12.0, 18.0, 11.0]
        burstiness = _calculate_burstiness(perplexities)

        # Should be a positive number representing standard deviation
        assert burstiness > 0
        assert isinstance(burstiness, float)

    def test_calculate_burstiness_insufficient_data(self):
        """Test burstiness with insufficient data points"""
        # Single value
        assert _calculate_burstiness([10.0]) == 0.0

        # Empty list
        assert _calculate_burstiness([]) == 0.0

    def test_calculate_burstiness_with_infinities(self):
        """Test burstiness calculation filtering infinite values"""
        perplexities = [10.0, float("inf"), 15.0, float("inf"), 12.0]
        burstiness = _calculate_burstiness(perplexities)

        # Should calculate based on finite values only
        assert burstiness > 0
        assert not np.isinf(burstiness)


class TestConfigLoading:
    """Test configuration loading functionality"""

    def test_load_config_defaults(self):
        """Test loading default configuration when no config file exists"""
        with patch("pathlib.Path.exists", return_value=False):
            config = load_config()

            assert "perplexity" in config
            assert config["perplexity"]["model_name"] == "gpt2"
            assert config["perplexity"]["max_length"] == 512
            assert config["perplexity"]["thresholds"]["ppl_max"] == 25.0
            assert config["perplexity"]["thresholds"]["burstiness_min"] == 2.5

    @patch("builtins.open")
    @patch("pathlib.Path.exists")
    @patch("yaml.safe_load")
    def test_load_config_custom(self, mock_yaml, mock_exists, mock_open):
        """Test loading custom configuration"""
        mock_exists.return_value = True
        mock_yaml.return_value = {"perplexity": {"thresholds": {"ppl_max": 30.0, "burstiness_min": 3.0}}}

        config = load_config()

        # Should merge with defaults
        assert config["perplexity"]["model_name"] == "gpt2"  # Default
        assert config["perplexity"]["thresholds"]["ppl_max"] == 30.0  # Custom
        assert config["perplexity"]["thresholds"]["burstiness_min"] == 3.0  # Custom


class TestChunking:
    """Test text chunking functionality"""

    def test_chunk_text_short(self):
        """Test chunking with text shorter than max_length"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens

        chunks = _chunk_text("short text", mock_tokenizer, max_length=10, overlap=2)

        assert chunks == ["short text"]

    def test_chunk_text_long(self):
        """Test chunking with text longer than max_length"""
        mock_tokenizer = Mock()
        # Simulate 15 tokens
        mock_tokenizer.encode.return_value = list(range(15))
        mock_tokenizer.decode.side_effect = lambda tokens, **kwargs: f"chunk_{tokens[0]}_{tokens[-1]}"

        chunks = _chunk_text("long text", mock_tokenizer, max_length=10, overlap=2)

        assert len(chunks) > 1
        assert len(chunks) <= 5  # Safety check to ensure we don't get infinite chunks
        mock_tokenizer.encode.assert_called_once_with("long text", add_special_tokens=False)


class TestPerplexityCalculation:
    """Test perplexity calculation with mocked models"""

    @patch("torch.no_grad")
    def test_calculate_perplexity_success(self, mock_no_grad):
        """Test successful perplexity calculation"""
        # Mock tokenizer properly - it needs to return an object with input_ids attribute
        mock_tokenizer = Mock()
        mock_inputs = Mock()
        mock_inputs.input_ids = torch.tensor([[1, 2, 3]])
        mock_tokenizer.return_value = mock_inputs

        # Mock model
        mock_model = Mock()
        mock_outputs = Mock()
        mock_outputs.loss.item.return_value = 2.0
        mock_model.return_value = mock_outputs

        # Mock torch.exp
        with patch("torch.exp") as mock_exp:
            mock_exp.return_value.item.return_value = 7.389  # exp(2.0)

            perplexity = _calculate_perplexity("test text", mock_model, mock_tokenizer)

            assert perplexity == 7.389
            # Verify the mocks were called correctly
            mock_tokenizer.assert_called_once_with("test text", return_tensors="pt", truncation=True, max_length=1024)
            mock_model.assert_called_once_with(mock_inputs.input_ids, labels=mock_inputs.input_ids)

    def test_calculate_perplexity_empty_text(self):
        """Test perplexity calculation with empty text"""
        mock_model = Mock()
        mock_tokenizer = Mock()

        perplexity = _calculate_perplexity("", mock_model, mock_tokenizer)

        assert np.isinf(perplexity)

    def test_calculate_perplexity_error(self):
        """Test perplexity calculation with model error"""
        mock_tokenizer = Mock()
        mock_tokenizer.side_effect = Exception("Model error")
        mock_model = Mock()

        perplexity = _calculate_perplexity("test text", mock_model, mock_tokenizer)

        assert np.isinf(perplexity)


class TestPerplexityAnalysisIntegration:
    """Test the main perplexity_analysis function"""

    @patch("server.server.get_perplexity_model")
    @patch("server.server._split_into_sentences")
    @patch("server.server._calculate_perplexity")
    def test_perplexity_analysis_success(self, mock_calc_ppl, mock_split, mock_get_model):
        """Test successful perplexity analysis"""
        # Mock dependencies
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_config = {
            "model_name": "gpt2",
            "max_length": 512,
            "overlap": 50,
            "thresholds": {"ppl_max": 25.0, "burstiness_min": 2.5},
        }
        mock_get_model.return_value = (mock_model, mock_tokenizer, mock_config)

        mock_split.return_value = ["First sentence.", "Second sentence."]
        mock_calc_ppl.side_effect = [15.0, 20.0]  # Different perplexities for burstiness

        # Mock chunking to return single chunks
        with patch("server.server._chunk_text") as mock_chunk:
            mock_chunk.side_effect = [["First sentence."], ["Second sentence."]]

            result = perplexity_analysis("First sentence. Second sentence.")

        assert "error" not in result
        assert result["doc_ppl"] == 17.5  # Average of 15.0 and 20.0
        assert result["doc_burstiness"] > 0  # Should have some variance
        assert len(result["sentences"]) == 2
        assert result["sentences"][0]["text"] == "First sentence."
        assert result["sentences"][0]["ppl"] == 15.0

    def test_perplexity_analysis_non_english(self):
        """Test perplexity analysis with non-English language"""
        # This test doesn't need model loading since it returns early for non-English
        result = perplexity_analysis("Bonjour le monde", language="fr")

        assert "error" in result
        assert "Only English language" in result["error"]
        assert result["doc_ppl"] is None

    def test_perplexity_analysis_empty_text(self):
        """Test perplexity analysis with empty text"""
        # This test doesn't need model loading since it returns early for empty text
        result = perplexity_analysis("")

        assert "error" in result
        assert "Empty text provided" in result["error"]
        assert result["doc_ppl"] is None

    @patch("server.server.get_perplexity_model")
    @patch("server.server._split_into_sentences")
    def test_perplexity_analysis_no_sentences(self, mock_split, mock_get_model):
        """Test perplexity analysis when no sentences are found"""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_config = {"model_name": "gpt2", "thresholds": {"ppl_max": 25.0, "burstiness_min": 2.5}}
        mock_get_model.return_value = (mock_model, mock_tokenizer, mock_config)

        mock_split.return_value = []

        result = perplexity_analysis("...")

        assert "error" in result
        assert "No valid sentences found" in result["error"]

    @patch("server.server.get_perplexity_model")
    @patch("server.server._split_into_sentences")
    @patch("server.server._calculate_perplexity")
    def test_perplexity_analysis_ai_detection(self, mock_calc_ppl, mock_split, mock_get_model):
        """Test AI detection flags in perplexity analysis"""
        # Mock low perplexity and low burstiness (AI-like)
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_config = {
            "model_name": "gpt2",
            "max_length": 512,
            "overlap": 50,
            "thresholds": {"ppl_max": 25.0, "burstiness_min": 2.5},
        }
        mock_get_model.return_value = (mock_model, mock_tokenizer, mock_config)

        mock_split.return_value = ["Sentence one.", "Sentence two.", "Sentence three."]
        # Low perplexity values with low variance (AI-like)
        mock_calc_ppl.side_effect = [10.0, 10.1, 10.2]

        with patch("server.server._chunk_text") as mock_chunk:
            mock_chunk.side_effect = [["Sentence one."], ["Sentence two."], ["Sentence three."]]

            result = perplexity_analysis("Sentence one. Sentence two. Sentence three.")

        assert result["flags"]["high_ai_probability"] is True
        assert len(result["flags"]["reasons"]) > 0
        assert "Low perplexity" in result["flags"]["reasons"][0]
        assert "low burstiness" in result["flags"]["reasons"][0]


class TestEdgeCases:
    """Test edge cases and error handling"""

    @patch("server.server.get_perplexity_model")
    def test_perplexity_analysis_model_error(self, mock_get_model):
        """Test perplexity analysis when model loading fails"""
        mock_get_model.side_effect = Exception("Model loading failed")

        result = perplexity_analysis("Test text")

        assert "error" in result
        assert "Analysis failed" in result["error"]
        assert result["doc_ppl"] is None

    def test_burstiness_all_infinite_perplexities(self):
        """Test burstiness calculation with all infinite perplexities"""
        perplexities = [float("inf"), float("inf"), float("inf")]
        burstiness = _calculate_burstiness(perplexities)

        assert burstiness == 0.0


# Sample text fixtures for integration testing
@pytest.fixture
def sample_human_text():
    """Sample of human-written text (should have higher burstiness)"""
    return """
    The old lighthouse stood sentinel against the crashing waves, its weathered stone facade
    telling stories of countless storms weathered and ships guided safely to harbor.
    Tonight was different though. The keeper noticed something unusual in the pattern of
    the light as it swept across the dark waters - a rhythmic interruption that seemed
    almost intentional, as if someone was signaling from the depths below.
    """


@pytest.fixture
def sample_ai_text():
    """Sample of AI-generated text (should have lower burstiness)"""
    return """
    The company's quarterly report showed strong performance across all metrics.
    Revenue increased by fifteen percent compared to the previous quarter.
    Operating expenses remained stable throughout the reporting period.
    The management team expressed confidence in future growth prospects.
    Shareholders can expect continued strong performance in upcoming quarters.
    """
