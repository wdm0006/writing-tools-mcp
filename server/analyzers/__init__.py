"""Analysis modules for different text analysis domains."""

from .ai_detection import AIDetectionAnalyzer
from .basic_stats import BasicStatsAnalyzer
from .keyword_analysis import KeywordAnalyzer
from .readability import ReadabilityAnalyzer
from .style_analysis import StyleAnalyzer

__all__ = [
    # Classes
    "BasicStatsAnalyzer",
    "ReadabilityAnalyzer",
    "KeywordAnalyzer",
    "StyleAnalyzer",
    "AIDetectionAnalyzer",
]


def initialize_analyzers(nlp_model, gpt2_manager, config):
    """Initialize all analyzers with required dependencies."""
    return {
        "basic_stats": BasicStatsAnalyzer(),
        "readability": ReadabilityAnalyzer(),
        "keyword": KeywordAnalyzer(nlp_model),
        "style": StyleAnalyzer(nlp_model),
        "ai_detection": AIDetectionAnalyzer(nlp_model, gpt2_manager, config),
    }
