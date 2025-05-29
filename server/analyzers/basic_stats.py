"""Basic text statistics analysis."""

from spellchecker import SpellChecker

from server.text_processing import preprocess_text


class BasicStatsAnalyzer:
    """Handles basic text statistics like character count, word count, and spellchecking."""

    def __init__(self):
        self.spell_checker = SpellChecker()

    def character_count(self, text: str) -> int:
        """Calculate the total number of characters in the text."""
        return len(text)

    def word_count(self, text: str) -> int:
        """Calculate the total number of words in the text."""
        return len(text.split())

    def spellcheck(self, text: str) -> list[str]:
        """Find potentially misspelled words in the text."""
        # Use preprocess_text to get clean words, without lemmatization or stopword removal
        words = preprocess_text(text, remove_stopwords=False, lemmatize=False)
        misspelled = self.spell_checker.unknown(words)
        return list(misspelled)
