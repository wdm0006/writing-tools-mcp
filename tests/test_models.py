"""Test module for model management."""

from unittest.mock import MagicMock, Mock, patch

from server.analyzers import AIDetectionAnalyzer
from server.config import load_config
from server.models import GPT2Manager, SpacyManager, initialize_models


class TestSpacyManager:
    """Test the SpacyManager class."""

    def test_init_with_default_model(self):
        """Test initialization with default model name."""
        manager = SpacyManager()
        assert manager.model_name == "en_core_web_sm"
        assert manager._model is None

    def test_init_with_custom_model(self):
        """Test initialization with custom model name."""
        manager = SpacyManager(model_name="en_core_web_lg")
        assert manager.model_name == "en_core_web_lg"
        assert manager._model is None

    @patch("server.models.spacy_manager.spacy.load")
    def test_get_model_success(self, mock_spacy_load):
        """Test successful model loading."""
        mock_nlp = MagicMock()
        mock_spacy_load.return_value = mock_nlp

        manager = SpacyManager()
        result = manager.get_model()

        assert result == mock_nlp
        assert manager._model == mock_nlp
        mock_spacy_load.assert_called_once_with("en_core_web_sm")

    @patch("server.models.spacy_manager.spacy.load")
    def test_get_model_cached(self, mock_spacy_load):
        """Test that subsequent calls return cached model."""
        mock_nlp = MagicMock()
        mock_spacy_load.return_value = mock_nlp

        manager = SpacyManager()
        result1 = manager.get_model()
        result2 = manager.get_model()

        assert result1 == result2
        mock_spacy_load.assert_called_once()  # Only called once


class TestGPT2Manager:
    """Test the GPT2Manager class."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        config = {"model_name": "gpt2", "cache_dir": "models/gpt2", "tokenizer": "gpt2"}
        manager = GPT2Manager(config)
        assert manager.config == config
        assert manager._model is None
        assert manager._tokenizer is None

    @patch("server.models.gpt2_manager.GPT2Tokenizer.from_pretrained")
    @patch("server.models.gpt2_manager.GPT2LMHeadModel.from_pretrained")
    def test_get_model_and_tokenizer_success(self, mock_model, mock_tokenizer):
        """Test successful model and tokenizer loading."""
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = mock_tokenizer_instance

        config = {"model_name": "gpt2"}
        manager = GPT2Manager(config)
        model, tokenizer, returned_config = manager.get_model_and_tokenizer()

        assert model == mock_model_instance
        assert tokenizer == mock_tokenizer_instance
        assert returned_config == config


DISTINCTIVE_PERPLEXITY_CONFIG = {
    "model_name": "distilgpt2",
    "max_length": 128,
    "overlap": 7,
    "thresholds": {"ppl_max": 12.5, "burstiness_min": 4.0},
    "device": "cpu",
    "language": "en",
}


class TestInitializeModels:
    """Test that initialize_models wires configuration through to the managers."""

    def test_gpt2_manager_receives_perplexity_section(self):
        """The GPT-2 manager is configured from the documented 'perplexity' section."""
        config = {"perplexity": DISTINCTIVE_PERPLEXITY_CONFIG, "stylometry": {}, "logging": {}}

        gpt2_manager = initialize_models(config)["gpt2"]

        assert gpt2_manager.config == DISTINCTIVE_PERPLEXITY_CONFIG
        assert gpt2_manager.config["model_name"] == "distilgpt2"
        assert gpt2_manager.config["max_length"] == 128
        assert gpt2_manager.config["overlap"] == 7
        assert gpt2_manager.config["thresholds"] == {"ppl_max": 12.5, "burstiness_min": 4.0}

    def test_loaded_config_supplies_keys_the_analysis_path_indexes(self):
        """Defaults loaded from disk reach the manager with every key perplexity analysis indexes."""
        gpt2_manager = initialize_models(load_config())["gpt2"]

        for key in ("model_name", "max_length", "overlap", "thresholds"):
            assert key in gpt2_manager.config
        for threshold in ("ppl_max", "burstiness_min"):
            assert threshold in gpt2_manager.config["thresholds"]

    @patch("server.analyzers.ai_detection.split_into_sentences")
    def test_perplexity_analysis_uses_the_wired_configuration(self, mock_split):
        """A full perplexity analysis succeeds and reports the configured model and thresholds."""
        mock_split.return_value = ["A configured sentence.", "Another configured sentence."]

        config = {"perplexity": DISTINCTIVE_PERPLEXITY_CONFIG, "stylometry": {}, "logging": {}}
        gpt2_manager = initialize_models(config)["gpt2"]

        # Populate the lazy cache directly so no real GPT-2 weights are fetched.
        gpt2_manager._model = Mock()
        gpt2_manager._tokenizer = Mock()
        gpt2_manager._tokenizer.encode.return_value = [1, 2, 3]

        analyzer = AIDetectionAnalyzer(Mock(), gpt2_manager, config)
        with patch.object(AIDetectionAnalyzer, "_calculate_perplexity", side_effect=[8.0, 9.0]):
            result = analyzer.perplexity_analysis("A configured sentence. Another configured sentence.")

        assert "error" not in result
        assert result["config"] == {
            "model": "distilgpt2",
            "thresholds": {"ppl_max": 12.5, "burstiness_min": 4.0},
        }
        assert result["doc_ppl"] == 8.5
        # doc_ppl 8.5 < ppl_max 12.5 and burstiness 0.71 < burstiness_min 4.0
        assert result["flags"]["high_ai_probability"] is True
