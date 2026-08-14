"""Keyword analysis functionality."""

from collections import Counter

from server.text_processing import preprocess_text


class KeywordAnalyzer:
    """Handles keyword-related text analysis."""

    def __init__(self, nlp_model):
        self.nlp = nlp_model

    @staticmethod
    def _count_sequence(tokens: list[str], keyword_tokens: list[str]) -> int:
        """Count contiguous occurrences of a token sequence."""
        keyword_length = len(keyword_tokens)
        if keyword_length == 0:
            return 0
        return sum(
            tokens[index : index + keyword_length] == keyword_tokens
            for index in range(len(tokens) - keyword_length + 1)
        )

    def keyword_density(self, text: str, keyword: str) -> float:
        """Calculate the density of a specific keyword within the text."""
        processed_text = preprocess_text(text)
        processed_keyword = preprocess_text(keyword)

        keyword_count = self._count_sequence(processed_text, processed_keyword)
        return (keyword_count / len(processed_text)) * 100 if processed_text else 0

    def keyword_frequency(self, text: str, remove_stopwords: bool = True) -> dict:
        """Count the frequency of each word (or lemma) in the provided text."""
        processed_text = preprocess_text(text, remove_stopwords=remove_stopwords)
        return dict(Counter(processed_text))

    def top_keywords(self, text: str, top_n: int = 10, remove_stopwords: bool = True) -> list:
        """Identify the most frequently occurring keywords (words or lemmas) in the text."""
        processed_text = preprocess_text(text, remove_stopwords=remove_stopwords)
        frequency = Counter(processed_text)
        return frequency.most_common(top_n)

    def keyword_context(self, text: str, keyword: str) -> list[str]:
        """Extract sentences from the text that contain a specific keyword or its lemma."""
        doc = self.nlp(text)

        keyword_doc = self.nlp(keyword.lower())
        keyword_tokens = [token.lemma_.lower() for token in keyword_doc if not token.is_punct and not token.is_space]
        if not keyword_tokens:
            return []

        contexts = []
        for sent in doc.sents:
            sentence_tokens = [token.lemma_.lower() for token in sent if not token.is_punct and not token.is_space]
            if self._count_sequence(sentence_tokens, keyword_tokens):
                contexts.append(sent.text)

        return contexts
