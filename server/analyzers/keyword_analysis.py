"""Keyword analysis functionality."""

from collections import Counter

from ..text_processing import preprocess_text


class KeywordAnalyzer:
    """Handles keyword-related text analysis."""

    def __init__(self, nlp_model):
        self.nlp = nlp_model

    def keyword_density(self, text: str, keyword: str) -> float:
        """Calculate the density of a specific keyword within the text."""
        processed_text = preprocess_text(text)
        processed_keyword = preprocess_text(keyword)[0] if preprocess_text(keyword) else keyword.lower()

        keyword_count = processed_text.count(processed_keyword)
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

        # Process the keyword to get its lemma
        keyword_doc = self.nlp(keyword.lower())
        keyword_lemma = keyword_doc[0].lemma_ if len(keyword_doc) > 0 else keyword.lower()

        # Extract sentences containing the keyword or its lemmatized form
        contexts = []
        for sent in doc.sents:
            if any(token.text.lower() == keyword.lower() or token.lemma_ == keyword_lemma for token in sent):
                contexts.append(sent.text)

        return contexts
