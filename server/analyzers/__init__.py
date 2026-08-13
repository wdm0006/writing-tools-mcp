"""Analysis modules for different text analysis domains."""

from server.analyzers.ai_detection import AIDetectionAnalyzer
from server.analyzers.basic_stats import BasicStatsAnalyzer
from server.analyzers.keyword_analysis import KeywordAnalyzer
from server.analyzers.readability import ReadabilityAnalyzer
from server.analyzers.style_analysis import StyleAnalyzer

__all__ = [
    # Classes
    "BasicStatsAnalyzer",
    "ReadabilityAnalyzer",
    "KeywordAnalyzer",
    "StyleAnalyzer",
    "AIDetectionAnalyzer",
]


def initialize_model_independent_analyzers():
    """Initialize the analyzers that need no NLP model.

    These are safe to construct without loading spaCy. ``BasicStatsAnalyzer.spellcheck``
    still depends on the shared preprocessor, so only the count and readability tools
    may use this set on its own.
    """
    return {
        "basic_stats": BasicStatsAnalyzer(),
        "readability": ReadabilityAnalyzer(),
    }


def initialize_analyzers(nlp_model, gpt2_manager, config):
    """Initialize all analyzers with required dependencies."""
    return {
        **initialize_model_independent_analyzers(),
        "keyword": KeywordAnalyzer(nlp_model),
        "style": StyleAnalyzer(nlp_model),
        "ai_detection": AIDetectionAnalyzer(nlp_model, gpt2_manager, config),
    }
