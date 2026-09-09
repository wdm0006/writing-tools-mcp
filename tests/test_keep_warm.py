"""Tests for the ``model.keep_warm_seconds`` TTL-based model eviction.

The clock is fully mocked (``app._now``) — no test here ever sleeps, so the whole
module runs comfortably inside the 60s pytest timeout. The default-TTL contract is
owned by ``tests/test_cleanup.py`` (which must pass unmodified); this module proves
the TTL behavior on top of it.
"""

from unittest.mock import Mock, patch

import pytest

from server import app

TTL = 300.0


class FakeClock:
    """Deterministic stand-in for ``app._now``; advance it in seconds."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def advance(self, seconds: float) -> None:
        self._now += seconds

    def __call__(self) -> float:
        return self._now


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


@pytest.fixture
def clock(mock_managers, mock_analyzer_factory, monkeypatch):
    """Isolate keep-warm state and hand the test a mocked clock.

    ``monkeypatch`` restores ``_keep_warm_seconds`` (0 by default) and
    ``_last_model_use`` after every test, so neighboring test modules see the
    module exactly as the shipped default config leaves it.
    """
    monkeypatch.setattr(app, "_last_model_use", None)
    fake_clock = FakeClock()
    monkeypatch.setattr(app, "_now", fake_clock)
    return fake_clock


def use_ttl(monkeypatch, seconds: float) -> None:
    """Set the module TTL the way a positive ``keep_warm_seconds`` config would."""
    monkeypatch.setattr(app, "_keep_warm_seconds", float(seconds))


def make_tool():
    """A model-backed tool shaped like the decorated MCP tools."""

    @app.auto_cleanup("spacy", "gpt2")
    def tool():
        app.get_analyzers()
        return "result"

    return tool


class TestKeepWarmWindow:
    """TTL behavior of the ``auto_cleanup`` end-of-call release."""

    def test_second_call_inside_window_does_not_reload(self, clock, monkeypatch):
        """Within the TTL the managers stay resident: no reload, no unload."""
        use_ttl(monkeypatch, TTL)
        tool = make_tool()

        assert tool() == "result"
        clock.advance(TTL - 1)
        assert tool() == "result"

        assert app._analyzers is not None
        app.spacy_manager.get_model.assert_called_once()
        assert app.initialize_analyzers.call_count == 1
        app.spacy_manager.unload_model.assert_not_called()
        app.gpt2_manager.unload_model.assert_not_called()

    def test_eviction_fires_after_expiry(self, clock, monkeypatch):
        """The first call to finish after the window lapses evicts via cleanup_models."""
        use_ttl(monkeypatch, TTL)
        tool = make_tool()

        tool()
        clock.advance(TTL + 1)
        tool()

        app.spacy_manager.unload_model.assert_called_once_with()
        app.gpt2_manager.unload_model.assert_called_once_with()
        assert app._analyzers is None

        # The eviction really dropped the cache: the next call reloads.
        clock.advance(1)
        tool()
        assert app.initialize_analyzers.call_count == 2
        assert app.spacy_manager.get_model.call_count == 2

    def test_window_refreshes_on_each_call(self, clock, monkeypatch):
        """Each in-window call restarts the TTL from zero."""
        use_ttl(monkeypatch, TTL)
        tool = make_tool()

        tool()
        clock.advance(TTL - 1)
        tool()
        clock.advance(TTL - 1)
        tool()

        assert app.initialize_analyzers.call_count == 1
        app.spacy_manager.unload_model.assert_not_called()

        # Only the first call to finish more than TTL after the LAST use evicts.
        clock.advance(TTL)
        tool()
        app.spacy_manager.unload_model.assert_called_once_with()
        assert app._analyzers is None

    def test_default_ttl_zero_evicts_after_every_call(self, clock, monkeypatch):
        """With the default TTL 0, back-to-back calls unload every time — the old contract."""
        use_ttl(monkeypatch, 0)
        tool = make_tool()

        tool()
        tool()

        assert app.spacy_manager.unload_model.call_count == 2
        assert app.gpt2_manager.unload_model.call_count == 2
        assert app.initialize_analyzers.call_count == 2
        assert app._analyzers is None

    def test_first_call_after_startup_keeps_models_warm(self, clock, monkeypatch):
        """A positive TTL's very first call starts the window instead of unloading."""
        use_ttl(monkeypatch, TTL)
        tool = make_tool()

        tool()

        app.spacy_manager.unload_model.assert_not_called()
        app.gpt2_manager.unload_model.assert_not_called()
        assert app._analyzers is not None
        assert app._last_model_use == 0.0


class TestExplicitCleanupOverridesTTL:
    """Explicit cleanup unloads immediately, whatever the window says."""

    def test_cleanup_mid_window_unloads_and_resets(self, clock, monkeypatch):
        use_ttl(monkeypatch, TTL)
        tool = make_tool()

        tool()
        clock.advance(100)
        app.cleanup_models("spacy", "gpt2")

        app.spacy_manager.unload_model.assert_called_once_with()
        app.gpt2_manager.unload_model.assert_called_once_with()
        assert app._analyzers is None
        assert app._last_model_use is None

        # The next call reloads even though the original window had time left.
        tool()
        assert app.initialize_analyzers.call_count == 2

    def test_cleanup_models_respects_no_ttl(self, clock, monkeypatch):
        """cleanup_models keeps its unconditional semantics with the default TTL."""
        app.cleanup_models("spacy")

        app.spacy_manager.unload_model.assert_called_once_with()
        app.gpt2_manager.unload_model.assert_not_called()
        assert app._analyzers is None
        assert app._last_model_use is None


class TestKeepWarmExpiryDecision:
    """Unit tests for the pure TTL predicate."""

    def test_zero_ttl_is_always_expired(self):
        assert app._keep_warm_expired(now=1000.0, last_use=999.0, ttl=0) is True
        assert app._keep_warm_expired(now=0.0, last_use=None, ttl=0) is True

    def test_negative_ttl_is_always_expired(self):
        assert app._keep_warm_expired(now=0.0, last_use=0.0, ttl=-5.0) is True

    def test_first_use_starts_the_window(self):
        assert app._keep_warm_expired(now=42.0, last_use=None, ttl=300.0) is False

    def test_call_exactly_on_ttl_is_expired(self):
        assert app._keep_warm_expired(now=300.0, last_use=0.0, ttl=300.0) is True
        assert app._keep_warm_expired(now=299.999, last_use=0.0, ttl=300.0) is False
