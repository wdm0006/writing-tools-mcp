"""Test module for perplexity analysis functionality."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from server.analyzers import AIDetectionAnalyzer


class TestPerplexityAnalysis:
    """Test perplexity analysis functionality."""

    @pytest.fixture
    def mock_nlp(self):
        """Create mock spaCy model."""
        return Mock()

    @pytest.fixture
    def mock_gpt2_manager(self):
        """Create mock GPT2Manager."""
        mock_manager = Mock()

        # Mock model, tokenizer, and config
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_config = {
            "model_name": "gpt2",
            "max_length": 512,
            "overlap": 50,
            "thresholds": {"ppl_max": 50, "burstiness_min": 1.0},
        }

        mock_manager.get_model_and_tokenizer.return_value = (mock_model, mock_tokenizer, mock_config)
        return mock_manager

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {"gpt2": {"model_name": "gpt2", "cache_dir": "models/gpt2", "tokenizer": "gpt2"}}

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

    @patch("server.analyzers.ai_detection.split_into_sentences")
    def test_perplexity_analysis_no_sentences(self, mock_split, mock_nlp, mock_gpt2_manager, mock_config):
        """Test perplexity analysis when no sentences are found."""
        mock_split.return_value = []

        analyzer = AIDetectionAnalyzer(mock_nlp, mock_gpt2_manager, mock_config)
        result = analyzer.perplexity_analysis("Some text", language="en")

        assert "error" in result
        assert result["error"] == "No valid sentences found in text"

    def test_burstiness_calculation(self, mock_nlp, mock_gpt2_manager, mock_config):
        """Test burstiness calculation."""
        analyzer = AIDetectionAnalyzer(mock_nlp, mock_gpt2_manager, mock_config)

        # Test with valid perplexities
        perplexities = [10.0, 15.0, 12.0, 18.0, 11.0]
        burstiness = analyzer._calculate_burstiness(perplexities)
        assert burstiness > 0
        assert isinstance(burstiness, float)

        # Test with insufficient data
        assert analyzer._calculate_burstiness([10.0]) == 0.0
        assert analyzer._calculate_burstiness([]) == 0.0

        # Test with infinities
        perplexities_with_inf = [10.0, float("inf"), 15.0, float("inf"), 12.0]
        burstiness_filtered = analyzer._calculate_burstiness(perplexities_with_inf)
        assert burstiness_filtered > 0
        assert not np.isinf(burstiness_filtered)

    def test_chunk_text(self, mock_nlp, mock_gpt2_manager, mock_config):
        """Test text chunking functionality."""
        analyzer = AIDetectionAnalyzer(mock_nlp, mock_gpt2_manager, mock_config)

        # Test short text
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens

        chunks = analyzer._chunk_text("short text", mock_tokenizer, max_length=10, overlap=2)
        assert chunks == ["short text"]

        # Test long text
        mock_tokenizer.encode.return_value = list(range(15))  # 15 tokens
        mock_tokenizer.decode.side_effect = lambda tokens, **kwargs: f"chunk_{tokens[0]}_{tokens[-1]}"

        chunks = analyzer._chunk_text("long text", mock_tokenizer, max_length=10, overlap=2)
        assert len(chunks) > 1
        assert len(chunks) <= 5  # Safety check

    def test_calculate_perplexity(self, mock_nlp, mock_gpt2_manager, mock_config):
        """Test perplexity calculation."""
        analyzer = AIDetectionAnalyzer(mock_nlp, mock_gpt2_manager, mock_config)

        # Test empty text
        mock_model = Mock()
        mock_tokenizer = Mock()
        perplexity = analyzer._calculate_perplexity("", mock_model, mock_tokenizer)
        assert np.isinf(perplexity)

        # Test with error
        mock_tokenizer.side_effect = Exception("Model error")
        perplexity = analyzer._calculate_perplexity("test text", mock_model, mock_tokenizer)
        assert np.isinf(perplexity)


class TestPerplexityNaNHandling:
    """NaN perplexities must never reach the aggregations.

    A sentence that tokenizes to a single token gives GPT-2 nothing to predict, so
    the model returns a NaN loss and `exp(loss)` is NaN. NaN passes an `isinf`
    check, so it used to flow into the document average (making it NaN) and into
    `statistics.stdev`, which raises "cannot convert NaN to integer ratio" and
    failed the entire analysis.
    """

    @pytest.fixture
    def analyzer(self):
        mock_config = {"gpt2": {"model_name": "gpt2", "cache_dir": "models/gpt2", "tokenizer": "gpt2"}}
        return AIDetectionAnalyzer(Mock(), Mock(), mock_config)

    def test_burstiness_filters_nan(self, analyzer):
        """stdev must not see a NaN, and the result stays finite."""
        with_nan = [10.0, float("nan"), 15.0, 12.0, float("nan"), 18.0]
        burstiness = analyzer._calculate_burstiness(with_nan)
        assert np.isfinite(burstiness)
        assert burstiness > 0

    def test_burstiness_filters_nan_and_inf_together(self, analyzer):
        mixed = [10.0, float("nan"), float("inf"), 15.0, float("-inf"), 12.0]
        assert np.isfinite(analyzer._calculate_burstiness(mixed))

    def test_burstiness_all_nan_returns_zero(self, analyzer):
        assert analyzer._calculate_burstiness([float("nan"), float("nan")]) == 0.0

    def test_calculate_perplexity_converts_nan_to_inf(self, analyzer):
        """A NaN loss from the model is reported as a failed calculation."""
        mock_model = Mock()
        mock_tokenizer = Mock()

        mock_inputs = Mock()
        mock_inputs.input_ids = torch.tensor([[42]])  # single token
        mock_tokenizer.return_value = mock_inputs

        mock_outputs = Mock()
        mock_outputs.loss = torch.tensor(float("nan"))
        mock_model.return_value = mock_outputs

        perplexity = analyzer._calculate_perplexity("Hi", mock_model, mock_tokenizer)
        assert np.isinf(perplexity), "NaN loss must be reported as inf, never as NaN"

    def test_analysis_survives_a_nan_sentence(self, analyzer):
        """End to end: one NaN sentence must not fail the whole analysis."""
        mock_model, mock_tokenizer = Mock(), Mock()
        config = {
            "model_name": "gpt2",
            "max_length": 512,
            "overlap": 50,
            "thresholds": {"ppl_max": 50, "burstiness_min": 1.0},
        }
        analyzer.gpt2_manager.get_model_and_tokenizer.return_value = (mock_model, mock_tokenizer, config)

        sentences = ["A real sentence here.", "Hi", "Another real sentence here."]
        values = iter([20.0, float("nan"), 40.0])

        with (
            patch("server.analyzers.ai_detection.split_into_sentences", return_value=sentences),
            patch.object(analyzer, "_chunk_text", side_effect=lambda s, *a, **k: [s]),
            patch.object(analyzer, "_calculate_perplexity", side_effect=lambda *a, **k: next(values)),
        ):
            result = analyzer.perplexity_analysis("irrelevant, sentences are patched")

        assert result.get("error") is None, f"analysis failed: {result.get('error')}"
        assert result["doc_ppl"] == 30.0, "the NaN sentence must be excluded from the average"
        assert np.isfinite(result["doc_burstiness"])
        # the NaN sentence is still reported, with a null score
        assert [s["ppl"] for s in result["sentences"]] == [20.0, None, 40.0]
