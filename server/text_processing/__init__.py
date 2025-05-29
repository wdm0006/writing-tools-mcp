"""Text processing utilities for preprocessing, markdown parsing, and sentence splitting."""

from .markdown_parser import parse_markdown_sections, strip_markdown_markup
from .preprocessor import TextPreprocessor, initialize_preprocessor, preprocess_text
from .sentence_splitter import split_into_sentences, split_paragraphs

__all__ = [
    # Classes
    "TextPreprocessor",
    # Preprocessing functions
    "preprocess_text",
    "initialize_preprocessor",
    # Markdown parsing functions
    "parse_markdown_sections",
    "strip_markdown_markup",
    # Sentence splitting functions
    "split_into_sentences",
    "split_paragraphs",
]

# Backward compatibility aliases
split_sentences = split_into_sentences


def initialize_sentence_splitter(nlp_model):
    """Initialize sentence splitter with nlp model."""
    # This is a placeholder - sentence splitting is handled by spaCy directly
    pass
