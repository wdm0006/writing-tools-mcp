"""
Stylometric analysis package for detecting AI-generated text.

This package provides tools for analyzing writing style through statistical
features like sentence length, lexical diversity, and part-of-speech patterns.
"""

from server.stylometry.analyzer import StylemetricAnalyzer
from server.stylometry.baselines import BaselineManager
from server.stylometry.corpus_baseline import (
    ALL_SIMPLE_FEATURES,
    DEFAULT_ROBUST_FEATURES,
    DEFAULT_ROBUST_POS_TAGS,
    build_baseline_from_texts,
)
from server.stylometry.statistical import (
    calculate_char_ngram_similarity,
    calculate_sentence_z_scores,
    calculate_z_scores,
    flag_outliers,
    generate_flags,
)

__all__ = [
    "StylemetricAnalyzer",
    "BaselineManager",
    "calculate_z_scores",
    "calculate_char_ngram_similarity",
    "flag_outliers",
    "generate_flags",
    "calculate_sentence_z_scores",
    "build_baseline_from_texts",
    "DEFAULT_ROBUST_FEATURES",
    "DEFAULT_ROBUST_POS_TAGS",
    "ALL_SIMPLE_FEATURES",
]
