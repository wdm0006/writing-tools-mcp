"""Test module for lazy analyzer loading and automatic model cleanup."""

from unittest.mock import Mock, patch

import pytest

from server import app
from server.models.gpt2_manager import GPT2Manager
from server.models.spacy_manager import SpacyManager


@pytest.fixture
def mock_managers():
    """Replace the module-level model managers so no real model is ever loaded."""
    spacy_manager = Mock()
    spacy_manager.get_model.return_value = Mock(name="nlp")
    gpt2_manager = Mock()

    with (
        patch.object(app, "spacy_manager", spacy_manager),
        patch.object(app, "gpt2_manager", gpt2_manager),
    ):
        yield {"spacy": spacy_manager, "gpt2": gpt2_manager}


@pytest.fixture
def mock_analyzer_factory():
    """Replace the analyzer factory and text-processing initializers."""
    factory = Mock(side_effect=lambda *a, **kw: {"basic_stats": Mock()})

    with (
        patch.object(app, "initialize_analyzers", factory),
        patch.object(app, "initialize_preprocessor", Mock()),
        patch.object(app, "initialize_sentence_splitter", Mock()),
    ):
        yield factory


@pytest.fixture(autouse=True)
def reset_analyzer_cache():
    """Keep the ``_analyzers`` global from leaking between tests."""
    app._analyzers = None
    yield
    app._analyzers = None


class TestGetAnalyzers:
    """Test lazy initialization of the analyzer set."""

    def test_initializes_once_and_reuses_cache(self, mock_managers, mock_analyzer_factory):
        first = app.get_analyzers()
        second = app.get_analyzers()

        assert first is second
        assert mock_analyzer_factory.call_count == 1
        assert mock_managers["spacy"].get_model.call_count == 1

    def test_reinitializes_after_cleanup(self, mock_managers, mock_analyzer_factory):
        first = app.get_analyzers()
        app.cleanup_models("spacy")
        second = app.get_analyzers()

        assert first is not second
        assert mock_analyzer_factory.call_count == 2

    def test_builds_analyzers_from_loaded_model(self, mock_managers, mock_analyzer_factory):
        app.get_analyzers()

        nlp = mock_managers["spacy"].get_model.return_value
        mock_analyzer_factory.assert_called_once_with(nlp, mock_managers["gpt2"], app.config)


class TestCleanupModels:
    """Test explicit model release."""

    def test_unloads_only_named_managers(self, mock_managers, mock_analyzer_factory):
        app.get_analyzers()
        app.cleanup_models("spacy")

        mock_managers["spacy"].unload_model.assert_called_once_with()
        mock_managers["gpt2"].unload_model.assert_not_called()
        assert app._analyzers is None

    def test_unloads_both_managers(self, mock_managers, mock_analyzer_factory):
        app.get_analyzers()
        app.cleanup_models("spacy", "gpt2")

        mock_managers["spacy"].unload_model.assert_called_once_with()
        mock_managers["gpt2"].unload_model.assert_called_once_with()
        assert app._analyzers is None


class TestAutoCleanup:
    """Test the ``auto_cleanup`` decorator wrapping tool functions."""

    def test_spacy_only_tool_leaves_gpt2_loaded(self, mock_managers, mock_analyzer_factory):
        @app.auto_cleanup("spacy")
        def tool():
            app.get_analyzers()
            return "result"

        assert tool() == "result"
        mock_managers["spacy"].unload_model.assert_called_once_with()
        mock_managers["gpt2"].unload_model.assert_not_called()
        assert app._analyzers is None

    def test_ai_detection_tool_unloads_both(self, mock_managers, mock_analyzer_factory):
        @app.auto_cleanup("spacy", "gpt2")
        def tool():
            app.get_analyzers()
            return "result"

        tool()

        mock_managers["spacy"].unload_model.assert_called_once_with()
        mock_managers["gpt2"].unload_model.assert_called_once_with()
        assert app._analyzers is None

    def test_cleanup_runs_when_tool_raises(self, mock_managers, mock_analyzer_factory):
        @app.auto_cleanup("spacy", "gpt2")
        def tool():
            app.get_analyzers()
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            tool()

        mock_managers["spacy"].unload_model.assert_called_once_with()
        mock_managers["gpt2"].unload_model.assert_called_once_with()
        assert app._analyzers is None

    def test_preserves_wrapped_function_metadata_and_arguments(self, mock_managers, mock_analyzer_factory):
        @app.auto_cleanup("spacy")
        def tool(text, level="full"):
            """Docstring."""
            return f"{text}:{level}"

        assert tool.__name__ == "tool"
        assert tool.__doc__ == "Docstring."
        assert tool("hello", level="section") == "hello:section"


class TestManagerUnload:
    """Test that the managers actually drop their model references."""

    def test_spacy_manager_unload_resets_state(self):
        manager = SpacyManager()
        manager._model = Mock(name="nlp")

        manager.unload_model()

        assert manager._model is None

    def test_spacy_manager_unload_is_safe_when_not_loaded(self):
        manager = SpacyManager()

        manager.unload_model()

        assert manager._model is None

    def test_gpt2_manager_unload_resets_state(self):
        manager = GPT2Manager({"model_name": "gpt2"})
        manager._model = Mock(name="model")
        manager._tokenizer = Mock(name="tokenizer")

        manager.unload_model()

        assert manager._model is None
        assert manager._tokenizer is None

    def test_gpt2_manager_unload_is_safe_when_not_loaded(self):
        manager = GPT2Manager({"model_name": "gpt2"})

        manager.unload_model()

        assert manager._model is None
        assert manager._tokenizer is None
