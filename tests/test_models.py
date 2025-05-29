"""Test module for model management."""

from unittest.mock import MagicMock, patch

from server.models import GPT2Manager, SpacyManager


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
