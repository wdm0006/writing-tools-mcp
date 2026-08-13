"""Test module for the tool wrappers that must never touch an NLP model.

``character_count``, ``word_count``, ``readability_score`` and ``reading_time``
delegate to analyzers built without spaCy, so they must not go through
``get_analyzers()`` (which loads the model) or declare spaCy cleanup.
"""

from unittest.mock import Mock, patch

import pytest

from server import app
from server.analyzers import BasicStatsAnalyzer, ReadabilityAnalyzer

MODEL_INDEPENDENT_TOOLS = [
    (app.character_count, ("Some sample text here.",)),
    (app.word_count, ("Some sample text here.",)),
    (app.readability_score, ("Some sample text here.", "full")),
    (app.reading_time, ("Some sample text here.", "full")),
]
MODEL_INDEPENDENT_TOOL_IDS = [tool.__name__ for tool, _ in MODEL_INDEPENDENT_TOOLS]


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


@pytest.fixture(autouse=True)
def reset_analyzer_caches():
    """Keep the module-level analyzer caches from leaking between tests."""
    app._analyzers = None
    app._model_independent_analyzers = None
    yield
    app._analyzers = None
    app._model_independent_analyzers = None


class TestNoModelInitialization:
    """The four wrappers must not request, initialize or unload spaCy."""

    @pytest.mark.parametrize("tool, args", MODEL_INDEPENDENT_TOOLS, ids=MODEL_INDEPENDENT_TOOL_IDS)
    def test_tool_never_loads_or_unloads_spacy(self, tool, args, mock_managers):
        factory = Mock(side_effect=AssertionError("get_analyzers() must not be used by this tool"))

        with (
            patch.object(app, "initialize_analyzers", factory),
            patch.object(app, "initialize_preprocessor", Mock()) as preprocessor,
            patch.object(app, "initialize_sentence_splitter", Mock()) as splitter,
        ):
            tool(*args)

        mock_managers["spacy"].get_model.assert_not_called()
        mock_managers["spacy"].unload_model.assert_not_called()
        mock_managers["gpt2"].unload_model.assert_not_called()
        factory.assert_not_called()
        preprocessor.assert_not_called()
        splitter.assert_not_called()
        assert app._analyzers is None

    def test_analyzer_set_holds_only_model_independent_analyzers(self, mock_managers):
        analyzers = app.get_model_independent_analyzers()

        assert set(analyzers) == {"basic_stats", "readability"}
        assert isinstance(analyzers["basic_stats"], BasicStatsAnalyzer)
        assert isinstance(analyzers["readability"], ReadabilityAnalyzer)
        mock_managers["spacy"].get_model.assert_not_called()

    def test_analyzer_set_is_cached(self, mock_managers):
        assert app.get_model_independent_analyzers() is app.get_model_independent_analyzers()

    def test_cached_set_does_not_retain_a_spell_checker(self, mock_managers):
        """The cache outlives cleanup, so it must not hold the spellchecker dictionary."""
        with patch("server.analyzers.basic_stats.SpellChecker") as spell_checker_cls:
            app.character_count("hello world")
            app.word_count("hello world")

            spell_checker_cls.assert_not_called()

            app.get_model_independent_analyzers()["basic_stats"].spellcheck("hello world")

            spell_checker_cls.assert_called_once_with()

    def test_cleanup_leaves_model_independent_analyzers_usable(self, mock_managers):
        app.cleanup_models("spacy")

        assert app.character_count("hello world") == 11
        mock_managers["spacy"].get_model.assert_not_called()


class TestDelegation:
    """The wrappers must stay thin — the project analyzers do the work."""

    def test_character_count_delegates(self, mock_managers):
        with patch.object(BasicStatsAnalyzer, "character_count", return_value=1234) as method:
            assert app.character_count("hello world") == 1234

        method.assert_called_once_with("hello world")

    def test_word_count_delegates(self, mock_managers):
        with patch.object(BasicStatsAnalyzer, "word_count", return_value=99) as method:
            assert app.word_count("hello world") == 99

        method.assert_called_once_with("hello world")

    def test_readability_score_delegates(self, mock_managers):
        sentinel = {"flesch": 1.0, "kincaid": 2.0, "fog": 3.0}

        with patch.object(ReadabilityAnalyzer, "readability_score", return_value=sentinel) as method:
            assert app.readability_score("hello world", "paragraph") is sentinel

        method.assert_called_once_with("hello world", "paragraph")

    def test_reading_time_delegates(self, mock_managers):
        sentinel = {"full_text": 4.2}

        with patch.object(ReadabilityAnalyzer, "reading_time", return_value=sentinel) as method:
            assert app.reading_time("hello world", "section") is sentinel

        method.assert_called_once_with("hello world", "section")


class TestResultsUnchanged:
    """Public result shapes and values must match the analyzers' own output."""

    TEXT = "# Heading\n\nThe quick brown fox jumps over the lazy dog. It ran away.\n\nA second paragraph here."

    @pytest.mark.parametrize("level", ["full", "section", "paragraph", "bogus"])
    def test_readability_score_matches_analyzer(self, level, mock_managers):
        assert app.readability_score(self.TEXT, level) == ReadabilityAnalyzer().readability_score(self.TEXT, level)

    @pytest.mark.parametrize("level", ["full", "section", "paragraph", "bogus"])
    def test_reading_time_matches_analyzer(self, level, mock_managers):
        assert app.reading_time(self.TEXT, level) == ReadabilityAnalyzer().reading_time(self.TEXT, level)

    def test_counts_match_analyzer(self, mock_managers):
        analyzer = BasicStatsAnalyzer()

        assert app.character_count(self.TEXT) == analyzer.character_count(self.TEXT) == len(self.TEXT)
        assert app.word_count(self.TEXT) == analyzer.word_count(self.TEXT) == len(self.TEXT.split())

    def test_default_level_is_full(self, mock_managers):
        assert app.readability_score(self.TEXT) == app.readability_score(self.TEXT, "full")
        assert app.reading_time(self.TEXT) == app.reading_time(self.TEXT, "full")


class TestNlpToolsStillLoadAndCleanUp:
    """The NLP-dependent tools keep their lazy-load and cleanup-on-exit behavior."""

    @pytest.fixture
    def mock_analyzers(self):
        analyzers = {name: Mock() for name in ("basic_stats", "keyword", "style")}

        with (
            patch.object(app, "initialize_analyzers", Mock(return_value=analyzers)),
            patch.object(app, "initialize_preprocessor", Mock()),
            patch.object(app, "initialize_sentence_splitter", Mock()),
        ):
            yield analyzers

    @pytest.mark.parametrize(
        "tool, args, key, method, expected_call",
        [
            (app.spellcheck, ("some text",), "basic_stats", "spellcheck", ("some text",)),
            (app.keyword_frequency, ("some text",), "keyword", "keyword_frequency", ("some text", True)),
            (app.passive_voice_detection, ("some text",), "style", "passive_voice_detection", ("some text",)),
        ],
        ids=["spellcheck", "keyword_frequency", "passive_voice_detection"],
    )
    def test_nlp_tool_loads_then_unloads_spacy(
        self, tool, args, key, method, expected_call, mock_managers, mock_analyzers
    ):
        tool(*args)

        mock_managers["spacy"].get_model.assert_called_once_with()
        mock_managers["spacy"].unload_model.assert_called_once_with()
        mock_managers["gpt2"].unload_model.assert_not_called()
        assert app._analyzers is None
        getattr(mock_analyzers[key], method).assert_called_once_with(*expected_call)
