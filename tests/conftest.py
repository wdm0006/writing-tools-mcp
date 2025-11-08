"""Pytest configuration and fixtures for testing."""

import pytest

from server.analyzers import initialize_analyzers
from server.config import load_config
from server.models import initialize_models
from server.text_processing import initialize_preprocessor
from server.text_processing.sentence_splitter import initialize_sentence_splitter


@pytest.fixture(scope="session")
def config():
    """Load configuration for tests."""
    return load_config()


@pytest.fixture(scope="session")
def models(config):
    """Initialize models for tests."""
    return initialize_models(config)


@pytest.fixture(scope="session")
def nlp(models):
    """Get spaCy NLP model."""
    return models["spacy"]


@pytest.fixture(scope="session")
def gpt2_manager(models):
    """Get GPT-2 manager."""
    return models["gpt2"]


@pytest.fixture(scope="session", autouse=True)
def initialize_text_processing(nlp):
    """Initialize text processing modules."""
    initialize_preprocessor(nlp)
    initialize_sentence_splitter(nlp)


@pytest.fixture(scope="session")
def analyzers(nlp, gpt2_manager, config):
    """Initialize all analyzers for tests."""
    return initialize_analyzers(nlp, gpt2_manager, config)


@pytest.fixture
def basic_stats_analyzer(analyzers):
    """Get basic stats analyzer."""
    return analyzers["basic_stats"]


@pytest.fixture
def readability_analyzer(analyzers):
    """Get readability analyzer."""
    return analyzers["readability"]


@pytest.fixture
def keyword_analyzer(analyzers):
    """Get keyword analyzer."""
    return analyzers["keyword"]


@pytest.fixture
def style_analyzer(analyzers):
    """Get style analyzer."""
    return analyzers["style"]


@pytest.fixture
def ai_detection_analyzer(analyzers):
    """Get AI detection analyzer."""
    return analyzers["ai_detection"]
