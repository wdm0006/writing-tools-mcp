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


def initialize_analyzers(nlp_model, gpt2_manager, config):
    """Initialize all analyzers with required dependencies."""
    return {
        "basic_stats": BasicStatsAnalyzer(),
        "readability": ReadabilityAnalyzer(),
        "keyword": KeywordAnalyzer(nlp_model),
        "style": StyleAnalyzer(nlp_model),
        "ai_detection": AIDetectionAnalyzer(nlp_model, gpt2_manager, config),
    }
