"""Writing Tools MCP server — FastMCP tool layer and entry point.

This module owns the FastMCP instance, the ``@mcp.tool`` wrappers, and ``main()``.
It lives inside the ``server`` package so it no longer collides with the package
name, which is what makes the console script (``server.app:main``) and the built
wheel work.
"""

import functools
import logging
import sys

from fastmcp import FastMCP

# Analysis imports
from server.analyzers import initialize_analyzers, initialize_model_independent_analyzers
from server.analyzers.findings import (
    from_keyword_context,
    from_keyword_density,
    from_keyword_frequency,
    from_perplexity,
    from_readability_result,
    from_stylometry,
    from_top_keywords,
    order_by_impact,
)
from server.analyzers.section_analysis import (
    analyze_document_sections,
    models_for_selection,
    resolve_selection,
)

# Configuration imports
from server.config import load_config
from server.config.defaults import DEFAULT_CONFIG

# Model imports
from server.models import initialize_models
from server.prompts import render_guided_revision, render_writing_checklist

# Text processing imports
from server.text_processing import initialize_preprocessor
from server.text_processing.sentence_splitter import initialize_sentence_splitter

logger = logging.getLogger(__name__)

DEFAULT_LOGGING = DEFAULT_CONFIG["logging"]

# Bootstrap logging before the configuration is read, so `load_config`'s own validation
# warnings are visible. Logs go to stderr because the server speaks JSON-RPC on stdout.
logging.basicConfig(level=DEFAULT_LOGGING["level"], format=DEFAULT_LOGGING["format"], stream=sys.stderr)
logger.debug("Starting server.app initialization")


def configure_logging(config):
    """Apply the configured logging level and format to the stderr handler.

    An unrecognized level falls back to the default rather than raising, so a typo in
    `.mcp-config.yaml` cannot stop the server from starting. The stream stays pinned to
    stderr: a log record on stdout would corrupt the MCP protocol stream.
    """
    logging_config = config.get("logging", {})
    level = logging_config.get("level", DEFAULT_LOGGING["level"])
    log_format = logging_config.get("format", DEFAULT_LOGGING["format"])

    resolved_level = logging.getLevelName(str(level).upper())
    unknown_level = not isinstance(resolved_level, int)
    if unknown_level:
        resolved_level = logging.getLevelName(DEFAULT_LOGGING["level"])

    logging.basicConfig(level=resolved_level, format=log_format, stream=sys.stderr, force=True)

    if unknown_level:
        logger.warning("Unknown logging level %r in configuration. Using %s.", level, DEFAULT_LOGGING["level"])


mcp = FastMCP("Writing Tools MCP Server")

# Initialize configuration and model managers (lazy loading)
config = load_config()
configure_logging(config)
model_managers = initialize_models(config)
spacy_manager = model_managers["spacy"]
gpt2_manager = model_managers["gpt2"]

# Analyzers will be initialized lazily on first use
_analyzers = None
_model_independent_analyzers = None


def get_model_independent_analyzers():
    """Lazily initialize and return the analyzers that need no NLP model.

    Deliberately separate from :func:`get_analyzers` so the count and readability
    tools never trigger a spaCy load they cannot use.
    """
    global _model_independent_analyzers
    if _model_independent_analyzers is None:
        _model_independent_analyzers = initialize_model_independent_analyzers()
        logger.info("Model-independent analyzers initialized on first use (no NLP model loaded)")
    return _model_independent_analyzers


def get_analyzers():
    """Lazily initialize and return analyzers."""
    global _analyzers
    if _analyzers is None:
        # Load spaCy model only when first needed
        nlp = spacy_manager.get_model()
        # Initialize text processing modules
        initialize_preprocessor(nlp)
        initialize_sentence_splitter(nlp)
        # Initialize analyzers
        _analyzers = initialize_analyzers(nlp, gpt2_manager, config)
        logger.info("Analyzers initialized on first use (lazy loading)")
    return _analyzers


def cleanup_models(*model_names):
    """Release specified model memory immediately after use."""
    global _analyzers
    if "spacy" in model_names:
        spacy_manager.unload_model()
    if "gpt2" in model_names:
        gpt2_manager.unload_model()
    # Clear analyzers to force re-initialization on next use
    _analyzers = None
    logger.info("Released models: %s", ", ".join(model_names))


def auto_cleanup(*model_names):
    """Decorator to automatically cleanup specified models after tool execution.

    Args:
        model_names: Names of models to unload ("spacy", "gpt2").
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                cleanup_models(*model_names)

        return wrapper

    return decorator


@mcp.tool()
async def list_tools() -> list[str]:
    """Lists the names of all available tools provided by this server.

    Names are derived from the live FastMCP registry, so registering or removing
    a tool is reflected here without a second place to edit.

    Returns:
        list[str]: A sorted list containing the names of all registered MCP tools.
    """
    registered = await mcp.list_tools()
    return sorted(tool.name for tool in registered)


@mcp.tool()
def character_count(text: str) -> int:
    """Calculates the total number of characters in the provided text.

    Args:
        text: The input string to count characters from.

    Returns:
        int: The total character count of the input text.
    """
    return get_model_independent_analyzers()["basic_stats"].character_count(text)


@mcp.tool()
def word_count(text: str) -> int:
    """Calculates the total number of words in the provided text, splitting by whitespace.

    Args:
        text: The input string to count words from.

    Returns:
        int: The total word count of the input text.
    """
    return get_model_independent_analyzers()["basic_stats"].word_count(text)


@mcp.tool()
@auto_cleanup("spacy")
def spellcheck(text: str):
    """Identifies potentially misspelled words in the input text using pyspellchecker.

    Note: This function preprocesses the text to check individual words, excluding stopwords
    and punctuation by default (based on the current `preprocess_text` settings used).

    Args:
        text: The input string to perform spellchecking on.

    Returns:
        list[str]: A list of words from the input text identified as potentially misspelled.
    """
    return get_analyzers()["basic_stats"].spellcheck(text)


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

              On non-error paths the response also carries `findings`: a list of
              actionable, located observations built from the scores (see
              `server/analyzers/findings.py`).
    """
    result = get_model_independent_analyzers()["readability"].readability_score(text, level)
    if "error" not in result:
        result["findings"] = order_by_impact(from_readability_result(result))
    return result


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
    return get_model_independent_analyzers()["readability"].reading_time(text, level)


@mcp.tool()
@auto_cleanup("spacy")
def keyword_density(text: str, keyword: str) -> dict:
    """Calculates the density of a specific keyword within the text.

    Density is defined as (keyword count / total word count) * 100.
    The text and keyword are preprocessed (lowercased, lemmatized, stopwords potentially removed)
    before counting.

    Args:
        text: The input string to analyze.
        keyword: The keyword to calculate the density for.

    Returns:
        dict: `{"keyword": str, "density": float, "findings": list}` — the density
              percentage (0 if the text is empty) plus actionable findings
              (e.g. keyword stuffing above 5%, or the keyword being absent).
    """
    density = get_analyzers()["keyword"].keyword_density(text, keyword)
    result: dict = {"keyword": keyword, "density": density}
    result["findings"] = order_by_impact(from_keyword_density(keyword, density)) if text.strip() else []
    return result


@mcp.tool()
@auto_cleanup("spacy")
def keyword_frequency(text: str, remove_stopwords: bool = True) -> dict:
    """
    Counts the frequency of each word (or lemma) in the provided text.

    Args:
        text: The input string to analyze.
        remove_stopwords: If True (default), common English stopwords are removed before counting.
                           Uses spaCy's preprocessing.

    Returns:
        dict: `{"frequencies": {word: count, ...}, "findings": list}` — the word (or
              lemma) frequency map, plus actionable findings (overused terms). The
              counts live under `frequencies` so a word literally spelled
              "findings" can never collide with the findings array itself.
    """
    frequencies = get_analyzers()["keyword"].keyword_frequency(text, remove_stopwords)
    return {
        "frequencies": frequencies,
        "findings": order_by_impact(from_keyword_frequency(frequencies, stopwords_removed=remove_stopwords)),
    }


@mcp.tool()
@auto_cleanup("spacy")
def top_keywords(text: str, top_n: int = 10, remove_stopwords: bool = True) -> dict:
    """
    Identifies the most frequently occurring keywords (words or lemmas) in the text.

    Args:
        text: The input string to analyze.
        top_n: The maximum number of top keywords to return (default is 10).
        remove_stopwords: If True (default), common English stopwords are removed before counting.

    Returns:
        dict: `{"keywords": [[keyword, count], ...], "findings": list}` — up to `top_n`
              keyword/count pairs sorted by descending frequency, plus actionable
              findings (e.g. a single keyword dominating the distribution).
    """
    keywords = get_analyzers()["keyword"].top_keywords(text, top_n, remove_stopwords)
    return {"keywords": keywords, "findings": order_by_impact(from_top_keywords(keywords))}


@mcp.tool()
@auto_cleanup("spacy")
def keyword_context(text: str, keyword: str) -> dict:
    """Extracts sentences from the text that contain a specific keyword or its lemma.

    Uses spaCy for sentence boundary detection and lemmatization to match variations of the keyword.
    The search is case-insensitive.

    Args:
        text: The input string to search within.
        keyword: The keyword to find the context for.

    Returns:
        dict: `{"keyword": str, "sentences": list[str], "findings": list}` — the matching
              sentences, plus actionable findings (e.g. the keyword appearing nowhere).
    """
    sentences = get_analyzers()["keyword"].keyword_context(text, keyword)
    return {
        "keyword": keyword,
        "sentences": sentences,
        "findings": order_by_impact(from_keyword_context(sentences)),
    }


@mcp.tool()
@auto_cleanup("spacy")
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
    return get_analyzers()["style"].passive_voice_detection(text)


@mcp.tool()
@auto_cleanup("spacy", "gpt2")
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
              sentence-level scores, and AI detection flags. On success the
              response also carries `findings`: actionable, located observations
              (headline verdict plus the most predictable sentences) built by
              `server/analyzers/findings.py`. Error responses are unchanged.
    """
    result = get_analyzers()["ai_detection"].perplexity_analysis(text, language)
    if "error" not in result:
        result["findings"] = order_by_impact(from_perplexity(result))
    return result


@mcp.tool()
@auto_cleanup("spacy", "gpt2")
def stylometric_analysis(text: str, baseline: str | None = None, language: str = "en") -> dict:
    """
    Analyze text for stylometric features and detect AI-generated content.

    Computes sentence length distribution, lexical diversity metrics (TTR, Hapax Legomena),
    POS ratios, and other stylometric features. Flags outliers relative to human writing
    baselines using z-score analysis.

    Args:
        text: Input text to analyze
        baseline: Baseline corpus name; when omitted, the configured
            stylometry.default_baseline (or "brown_corpus") is used. The response's
            baseline_used field names the baseline actually measured against.
        language: Language code (only "en" supported currently)

    Returns:
        dict: Stylometric analysis with features, z-scores, and AI detection flags.
              On success the response also carries `findings`: actionable, located
              observations (AI indicators, baseline outliers, the headline verdict)
              built by `server/analyzers/findings.py` — honestly scoped to what the
              chosen baseline can actually measure. Error responses are unchanged.
    """
    result = get_analyzers()["ai_detection"].stylometric_analysis(text, baseline, language)
    if "error" not in result:
        result["findings"] = order_by_impact(from_stylometry(result))
    return result


@mcp.tool()
def analyze_sections(text: str, tools: list[str] | None = None, baseline: str | None = None) -> dict:
    """
    Run a selected subset of the analysis tools on every markdown section of a
    document, plus a whole-document rollup.

    Sections come from parse_markdown_sections: heading-keyed and hierarchy-aware
    (a subsection's body folds into its parent), with pre-first-heading content
    preserved as its own "_leading_content" section (heading level 0). Each
    section entry carries `key`, `heading_level`, the rendered section `text`,
    an impact-ordered `findings` array (the shared findings shape, located at
    "section:<key>"), and per-tool `results` — each exactly the response that
    tool returns for the section text at its default settings. The `rollup`
    carries each selected tool's whole-document response (findings included),
    identical to the standalone tool's output. An empty document yields no
    sections with the rollup still computed; a document with no headings yields
    the single "_leading_content" section.

    Args:
        text: The markdown document to analyze.
        tools: The analysis tools to run per section. Choose from: readability,
               word_count, character_count, reading_time, spellcheck,
               passive_voice, perplexity, stylometry. Defaults to everything
               except the GPT-2 tools (perplexity and stylometry stay opt-in).
               Unknown names yield {"error": ...}.
        baseline: Baseline name passed through to the per-section and rollup
                  stylometry runs (ignored unless "stylometry" is selected).

    Returns:
        dict: {"sections": [...], "rollup": {tool: whole-document response},
               "tools_used": [...], "section_count": int}.
    """
    try:
        selection = resolve_selection(tools)
    except ValueError as exc:
        return {"error": str(exc)}

    needed_models = models_for_selection(selection)
    analyzers = get_analyzers() if needed_models else get_model_independent_analyzers()

    try:
        return analyze_document_sections(
            text,
            selection,
            basic_stats=analyzers["basic_stats"],
            readability=analyzers["readability"],
            style=analyzers.get("style"),
            ai_detection=analyzers.get("ai_detection"),
            baseline=baseline,
        )
    finally:
        # Release only the tiers this selection used: a counts-and-readability
        # scan never triggers a model load, and a spaCy-only scan must not
        # unload the GPT-2 weights the way a blanket @auto_cleanup would.
        if needed_models:
            cleanup_models(*needed_models)


@mcp.prompt()
def guided_revision(document: str, findings: str | None = None) -> str:
    """Build an impact-ordered revision brief for a document.

    Args:
        document: The text to revise.
        findings: Optional JSON array — the `findings` value returned by an
                  analysis tool (readability_score, stylometric_analysis,
                  perplexity_analysis, keyword_density, keyword_frequency,
                  top_keywords, keyword_context). Each finding's rule, location,
                  message, and fix hint are rendered in impact order. When omitted,
                  the brief tells you which tools to run first.

    Returns:
        str: The rendered revision brief.
    """
    return render_guided_revision(document, findings)


@mcp.prompt()
def writing_checklist() -> str:
    """Render a pre-flight drafting checklist (structure, sentence variety,
    hedging, readability, keywords, voice) to apply while writing, before any
    analysis exists.

    Returns:
        str: The rendered checklist.
    """
    return render_writing_checklist()


def main():
    """Entry point for the Writing Tools MCP server."""
    logging.info("Starting MCP server...")
    mcp.run()


if __name__ == "__main__":
    main()
