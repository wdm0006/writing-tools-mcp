"""
Core stylometric feature extraction module.

This module provides the StylemetricAnalyzer class for extracting
various stylometric features from text that can be used to identify
AI-generated content.
"""

import statistics
from collections import Counter
from typing import Any, Dict, List


class StylemetricAnalyzer:
    """Analyzer for extracting stylometric features from text."""

    def __init__(self, nlp_model):
        """Initialize with a spaCy NLP model."""
        self.nlp = nlp_model

        # Common function words for analysis
        self.function_words = {
            "the",
            "of",
            "and",
            "a",
            "to",
            "in",
            "is",
            "you",
            "that",
            "it",
            "he",
            "was",
            "for",
            "on",
            "are",
            "as",
            "with",
            "his",
            "they",
            "i",
            "at",
            "be",
            "this",
            "have",
            "from",
            "or",
            "one",
            "had",
            "by",
            "word",
            "but",
            "not",
            "what",
            "all",
            "were",
            "we",
            "when",
            "your",
            "can",
            "said",
            "there",
            "each",
            "which",
            "she",
            "do",
            "how",
            "their",
            "if",
            "will",
            "up",
            "other",
            "about",
            "out",
            "many",
            "then",
            "them",
            "these",
            "so",
            "some",
            "her",
            "would",
            "make",
            "like",
            "into",
            "him",
            "has",
            "two",
            "more",
            "very",
            "after",
            "my",
            "than",
            "first",
            "been",
            "who",
            "its",
            "now",
            "people",
            "may",
            "down",
            "day",
            "get",
            "use",
            "man",
            "new",
            "way",
            "could",
            "does",
            "only",
            "where",
            "most",
            "over",
            "think",
            "also",
            "back",
            "work",
            "life",
            "why",
            "go",
            "should",
            "even",
        }

    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract all stylometric features from text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary containing all extracted stylometric features
        """
        if not text.strip():
            return self._empty_features()

        doc = self.nlp(text)
        sentences = list(doc.sents)

        if not sentences:
            return self._empty_features()

        return {
            # Sentence-level features
            "avg_sentence_len": self._avg_sentence_length(sentences),
            "sentence_len_std": self._sentence_length_std(sentences),
            "sentence_positions": self._sentence_positions(sentences),
            # Lexical diversity
            "ttr": self._type_token_ratio(doc),
            "hapax_legomena_rate": self._hapax_rate(doc),
            "avg_word_len": self._avg_word_length(doc),
            # POS ratios
            "pos_ratios": self._pos_ratios(doc),
            # Punctuation patterns
            "punct_density": self._punctuation_density(text),
            "comma_ratio": self._comma_ratio(text),
            # Additional features
            "function_word_ratio": self._function_word_ratio(doc),
        }

    def _empty_features(self) -> Dict[str, Any]:
        """Return empty/default feature values for empty text."""
        return {
            "avg_sentence_len": 0.0,
            "sentence_len_std": 0.0,
            "sentence_positions": [],
            "ttr": 0.0,
            "hapax_legomena_rate": 0.0,
            "avg_word_len": 0.0,
            "pos_ratios": {},
            "punct_density": 0.0,
            "comma_ratio": 0.0,
            "function_word_ratio": 0.0,
        }

    def _avg_sentence_length(self, sentences) -> float:
        """Calculate average sentence length in words."""
        if not sentences:
            return 0.0

        lengths = []
        for sent in sentences:
            # Count non-punctuation, non-space tokens
            word_count = sum(1 for token in sent if not token.is_punct and not token.is_space)
            lengths.append(word_count)

        return statistics.mean(lengths) if lengths else 0.0

    def _sentence_length_std(self, sentences) -> float:
        """Calculate standard deviation of sentence lengths."""
        if len(sentences) < 2:
            return 0.0

        lengths = []
        for sent in sentences:
            word_count = sum(1 for token in sent if not token.is_punct and not token.is_space)
            lengths.append(word_count)

        return statistics.stdev(lengths) if len(lengths) > 1 else 0.0

    def _sentence_positions(self, sentences) -> List[Dict[str, Any]]:
        """Extract detailed information about each sentence."""
        positions = []

        for i, sent in enumerate(sentences):
            word_count = sum(1 for token in sent if not token.is_punct and not token.is_space)
            positions.append(
                {
                    "position": i + 1,
                    "length": word_count,
                    "text": sent.text.strip()[:100] + "..." if len(sent.text) > 100 else sent.text.strip(),
                }
            )

        return positions

    def _type_token_ratio(self, doc) -> float:
        """Calculate Type-Token Ratio (lexical diversity)."""
        # Get all words (excluding punctuation and spaces)
        words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

        if not words:
            return 0.0

        unique_words = set(words)
        return len(unique_words) / len(words)

    def _hapax_rate(self, doc) -> float:
        """Calculate Hapax Legomena rate (words appearing only once)."""
        # Get all words (excluding punctuation and spaces)
        words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

        if not words:
            return 0.0

        word_counts = Counter(words)
        hapax_words = sum(1 for count in word_counts.values() if count == 1)

        return hapax_words / len(word_counts) if word_counts else 0.0

    def _avg_word_length(self, doc) -> float:
        """Calculate average word length in characters."""
        words = [token.text for token in doc if not token.is_punct and not token.is_space]

        if not words:
            return 0.0

        return statistics.mean(len(word) for word in words)

    def _pos_ratios(self, doc) -> Dict[str, float]:
        """Calculate part-of-speech tag ratios."""
        # Count POS tags for non-punctuation, non-space tokens
        pos_counts = Counter(token.pos_ for token in doc if not token.is_punct and not token.is_space)
        total_tokens = sum(pos_counts.values())

        if total_tokens == 0:
            return {}

        # Calculate ratios for major POS categories
        pos_ratios = {}
        for pos_tag, count in pos_counts.items():
            pos_ratios[pos_tag] = count / total_tokens

        return pos_ratios

    def _punctuation_density(self, text: str) -> float:
        """Calculate punctuation density (punctuation marks / total characters)."""
        if not text:
            return 0.0

        punct_count = sum(1 for char in text if char in ".,;:!?()[]{}\"'-")
        return punct_count / len(text)

    def _comma_ratio(self, text: str) -> float:
        """Calculate comma usage ratio (commas / total punctuation)."""
        if not text:
            return 0.0

        comma_count = text.count(",")
        total_punct = sum(1 for char in text if char in ".,;:!?()[]{}\"'-")

        return comma_count / total_punct if total_punct > 0 else 0.0

    def _function_word_ratio(self, doc) -> float:
        """Calculate ratio of function words to total words."""
        words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

        if not words:
            return 0.0

        function_word_count = sum(1 for word in words if word in self.function_words)
        return function_word_count / len(words)
