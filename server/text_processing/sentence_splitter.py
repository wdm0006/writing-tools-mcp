"""Sentence and paragraph splitting utilities."""

import re

# Global nlp model reference - initialized by main.py
_nlp_model = None


def split_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs.
    Returns a list of paragraphs.
    """
    # Split on empty lines (two or more newlines)
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using spaCy"""
    if _nlp_model is None:
        raise RuntimeError("NLP model not initialized for sentence splitting")

    doc = _nlp_model(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


def initialize_sentence_splitter(nlp_model):
    """Initialize the global NLP model for sentence splitting."""
    global _nlp_model
    _nlp_model = nlp_model
