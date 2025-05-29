"""Text preprocessing functionality."""

from typing import List


class TextPreprocessor:
    """Handles text preprocessing operations."""

    def __init__(self, nlp_model):
        self.nlp = nlp_model

    def preprocess(self, text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> List[str]:
        """
        Preprocess text using spaCy.

        Args:
            text: Input text to process
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize tokens

        Returns:
            List of processed tokens
        """
        doc = self.nlp(text.lower())

        if remove_stopwords and lemmatize:
            return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
        elif remove_stopwords:
            return [token.text for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
        elif lemmatize:
            return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        else:
            return [token.text for token in doc if not token.is_punct and not token.is_space]


# Global instance - initialized by main.py
_preprocessor = None


def preprocess_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> List[str]:
    """Global function for backward compatibility."""
    if _preprocessor is None:
        raise RuntimeError("Text preprocessor not initialized")
    return _preprocessor.preprocess(text, remove_stopwords, lemmatize)


def initialize_preprocessor(nlp_model):
    """Initialize the global preprocessor instance."""
    global _preprocessor
    _preprocessor = TextPreprocessor(nlp_model)
