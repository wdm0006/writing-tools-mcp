"""
Stylometric analysis package for detecting AI-generated text.

This package provides tools for analyzing writing style through statistical
features like sentence length, lexical diversity, and part-of-speech patterns.
"""

from .analyzer import StylemetricAnalyzer
from .baselines import BaselineManager
from .statistical import calculate_sentence_z_scores, calculate_z_scores, flag_outliers, generate_flags

__all__ = [
    "StylemetricAnalyzer",
    "BaselineManager",
    "calculate_z_scores",
    "flag_outliers",
    "generate_flags",
    "calculate_sentence_z_scores",
]
