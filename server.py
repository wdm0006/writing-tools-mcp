#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pip",
#   "mcp[cli]",
#   "pyspellchecker",
#   "textstat",
#   "spacy",
#   "markdown-it-py",
#   "requests",
#   "transformers>=4.35.0",
#   "torch>=2.0.0",
#   "pyyaml>=6.0",
#   "numpy>=1.24.0"
# ]
# ///

import logging
import statistics
import sys

import numpy as np
import torch
from mcp.server.fastmcp import FastMCP

# Analysis imports
from server.analyzers import initialize_analyzers

# Configuration imports
from server.config import load_config

# Model imports
from server.models import initialize_models

# Text processing imports
from server.text_processing import initialize_preprocessor
from server.text_processing.sentence_splitter import initialize_sentence_splitter

logger = logging.getLogger(__name__)
# Explicitly configure logging to use stderr to avoid breaking MCP JSON-RPC protocol on stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger.debug("Starting server.py initialization")

mcp = FastMCP("Writing Tools MCP Server")

# Initialize configuration and models
config = load_config()
models = initialize_models(config)
nlp = models["spacy"]
gpt2_manager = models["gpt2"]

# Initialize text processing modules
initialize_preprocessor(nlp)
initialize_sentence_splitter(nlp)

# Initialize analyzers
analyzers = initialize_analyzers(nlp, gpt2_manager, config)


def get_perplexity_model():
    """Load and cache GPT-2 model and tokenizer for perplexity analysis"""
    return gpt2_manager.get_model_and_tokenizer()


@mcp.tool()
def list_tools():
    """Lists the names of all available tools provided by this server.

    Returns:
        list[str]: A list containing the names of all registered MCP tools.
    """
    return [
        "list-tools",
        "character-count",
        "word-count",
        "spellcheck",
        "readability-score",
        "reading-time",
        "keyword-density",
        "keyword-frequency",
        "top-keywords",
        "keyword-context",
        "passive-voice-detection",
        "perplexity-analysis",
        "stylometric-analysis",
    ]


@mcp.tool()
def character_count(text: str) -> int:
    """Calculates the total number of characters in the provided text.

    Args:
        text: The input string to count characters from.

    Returns:
        int: The total character count of the input text.
    """
    return analyzers["basic_stats"].character_count(text)


@mcp.tool()
def word_count(text: str) -> int:
    """Calculates the total number of words in the provided text, splitting by whitespace.

    Args:
        text: The input string to count words from.

    Returns:
        int: The total word count of the input text.
    """
    return analyzers["basic_stats"].word_count(text)


@mcp.tool()
def spellcheck(text: str):
    """Identifies potentially misspelled words in the input text using pyspellchecker.

    Note: This function preprocesses the text to check individual words, excluding stopwords
    and punctuation by default (based on the current `preprocess_text` settings used).

    Args:
        text: The input string to perform spellchecking on.

    Returns:
        list[str]: A list of words from the input text identified as potentially misspelled.
    """
    return analyzers["basic_stats"].spellcheck(text)


@mcp.tool()
def readability_score(text: str, level: str = "full") -> dict:
    """
    Calculates various readability scores (Flesch Reading Ease, Flesch-Kincaid Grade Level,
    Gunning Fog Index) for the input text. The analysis can be performed on the full text,
    individual markdown sections, or individual paragraphs.

    Args:
        text: The input string to analyze.
        level: The granularity level for the analysis. Accepts:
               - "full" (default): Analyze the entire text as one segment.
               - "section": Analyze each markdown section (identified by headings) separately.
               - "paragraph": Analyze each paragraph (separated by double newlines) separately.

    Returns:
        dict: A dictionary containing the readability scores. The structure depends on the `level`:
              - If `level` is "full": `{"flesch": float, "kincaid": float, "fog": float}` or `{"flesch": None, ...}` if text is too short.
              - If `level` is "section": `{"full_text": {...}, "sections": {"section_heading": {...}, ...}}`
              - If `level` is "paragraph": `{"full_text": {...}, "paragraphs": [{"paragraph_number": int, "text": str, "scores": {...}}, ...]}`
              - If `level` is invalid: `{"error": str}`
    """
    return analyzers["readability"].readability_score(text, level)


@mcp.tool()
def reading_time(text: str, level: str = "full") -> dict:
    """
    Estimates the reading time for the input text using textstat. The estimation
    can be performed on the full text, individual markdown sections, or individual paragraphs.

    Args:
        text: The input string to estimate reading time for.
        level: The granularity level for the estimation. Accepts:
               - "full" (default): Estimate for the entire text.
               - "section": Estimate for each markdown section separately.
               - "paragraph": Estimate for each paragraph separately.

    Returns:
        dict: A dictionary containing the estimated reading time in minutes. The structure depends on the `level`:
              - If `level` is "full": `{"full_text": float}` (time in minutes)
              - If `level` is "section": `{"full_text": float, "sections": {"section_heading": float, ...}}`
              - If `level` is "paragraph": `{"full_text": float, "paragraphs": [{"paragraph_number": int, "text": str, "reading_time_minutes": float}, ...]}`
              - If `level` is invalid: `{"error": str}`
    """
    return analyzers["readability"].reading_time(text, level)


@mcp.tool()
def keyword_density(text: str, keyword: str) -> float:
    """Calculates the density of a specific keyword within the text.

    Density is defined as (keyword count / total word count) * 100.
    The text and keyword are preprocessed (lowercased, lemmatized, stopwords potentially removed)
    before counting.

    Args:
        text: The input string to analyze.
        keyword: The keyword to calculate the density for.

    Returns:
        float: The density of the keyword as a percentage. Returns 0 if the text is empty.
    """
    return analyzers["keyword"].keyword_density(text, keyword)


@mcp.tool()
def keyword_frequency(text: str, remove_stopwords: bool = True) -> dict:
    """
    Counts the frequency of each word (or lemma) in the provided text.

    Args:
        text: The input string to analyze.
        remove_stopwords: If True (default), common English stopwords are removed before counting.
                           Uses spaCy's preprocessing.

    Returns:
        dict: A dictionary where keys are the words (or lemmas if lemmatization is enabled
              in `preprocess_text`) and values are their corresponding frequency counts.
    """
    return analyzers["keyword"].keyword_frequency(text, remove_stopwords)


@mcp.tool()
def top_keywords(text: str, top_n: int = 10, remove_stopwords: bool = True) -> list:
    """
    Identifies the most frequently occurring keywords (words or lemmas) in the text.

    Args:
        text: The input string to analyze.
        top_n: The maximum number of top keywords to return (default is 10).
        remove_stopwords: If True (default), common English stopwords are removed before counting.

    Returns:
        list[tuple[str, int]]: A list of tuples, where each tuple contains a keyword (str)
                               and its frequency count (int), sorted in descending order of frequency.
    """
    return analyzers["keyword"].top_keywords(text, top_n, remove_stopwords)


@mcp.tool()
def keyword_context(text: str, keyword: str) -> list:
    """Extracts sentences from the text that contain a specific keyword or its lemma.

    Uses spaCy for sentence boundary detection and lemmatization to match variations of the keyword.
    The search is case-insensitive.

    Args:
        text: The input string to search within.
        keyword: The keyword to find the context for.

    Returns:
        list[str]: A list of sentences from the text that contain the specified keyword or its lemma.
    """
    return analyzers["keyword"].keyword_context(text, keyword)


@mcp.tool()
def passive_voice_detection(text: str) -> list:
    """
    Detects sentences potentially written in passive voice using a simplified rule-based approach with spaCy.

    Looks for patterns like auxiliary verb + past participle (e.g., "was written").
    Note: This is a basic detection and might not catch all passive constructions or might have false positives.

    Args:
        text: The input string to analyze for passive voice.

    Returns:
        list[str]: A list of sentences from the text identified as potentially containing passive voice.
    """
    return analyzers["style"].passive_voice_detection(text)


def _chunk_text(text, tokenizer, max_length=512, overlap=50):
    """
    Split text into overlapping chunks for processing long texts.

    Args:
        text (str): Input text to chunk
        tokenizer: GPT-2 tokenizer
        max_length (int): Maximum tokens per chunk
        overlap (int): Number of overlapping tokens between chunks

    Returns:
        list: List of text chunks
    """
    # Tokenize the full text
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= max_length:
        return [text]

    # Ensure overlap is not larger than max_length to prevent infinite loops
    overlap = min(overlap, max_length - 1)

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_length, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)

        # Move to next chunk with overlap
        next_start = end - overlap

        # Ensure we always advance to prevent infinite loops
        if next_start <= start:
            next_start = start + 1

        start = next_start

        # Safety check to prevent infinite loops
        if len(chunks) > 100:  # Reasonable upper limit
            break

    return chunks


def _calculate_perplexity(text, model, tokenizer):
    """
    Calculate perplexity for a given text using GPT-2.

    Args:
        text (str): Input text
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer

    Returns:
        float: Perplexity score
    """
    if not text.strip():
        return float("inf")

    try:
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs.input_ids

        # Calculate loss
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()

        # Calculate perplexity from loss
        perplexity = torch.exp(torch.tensor(loss)).item()

        return perplexity

    except Exception as e:
        logger.warning(f"Error calculating perplexity for text: {e}")
        return float("inf")


def _calculate_burstiness(sentence_perplexities):
    """
    Calculate burstiness as the standard deviation of sentence perplexities.

    Args:
        sentence_perplexities (list): List of perplexity scores for sentences

    Returns:
        float: Burstiness score (standard deviation)
    """
    if len(sentence_perplexities) < 2:
        return 0.0

    # Filter out infinite values
    valid_perplexities = [p for p in sentence_perplexities if not np.isinf(p)]

    if len(valid_perplexities) < 2:
        return 0.0

    return statistics.stdev(valid_perplexities)


@mcp.tool()
def perplexity_analysis(text: str, language: str = "en") -> dict:
    """
    Analyze text for perplexity and burstiness to detect AI-generated content.

    This function computes document-level and sentence-level perplexity using GPT-2,
    along with burstiness (variance of perplexity across sentences). Low perplexity
    combined with low burstiness is a statistical signal used by AI detectors.

    Args:
        text (str): The text to analyze
        language (str): Language code (only "en" supported currently)

    Returns:
        dict: Analysis results including document perplexity, burstiness,
              sentence-level scores, and AI detection flags
    """
    return analyzers["ai_detection"].perplexity_analysis(text, language)


@mcp.tool()
def stylometric_analysis(text: str, baseline: str = "brown_corpus", language: str = "en") -> dict:
    """
    Analyze text for stylometric features and detect AI-generated content.

    Computes sentence length distribution, lexical diversity metrics (TTR, Hapax Legomena),
    POS ratios, and other stylometric features. Flags outliers relative to human writing
    baselines using z-score analysis.

    Args:
        text: Input text to analyze
        baseline: Baseline corpus name ("brown_corpus" or custom baseline name)
        language: Language code (only "en" supported currently)

    Returns:
        dict: Stylometric analysis with features, z-scores, and AI detection flags
    """
    return analyzers["ai_detection"].stylometric_analysis(text, baseline, language)


if __name__ == "__main__":
    logging.info("Starting MCP server...")
    mcp.run()
