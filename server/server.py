#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pip",
#   "mcp[cli]",
#   "pyspellchecker",
#   "textstat",
#   "spacy",
#   "markdown-it-py"
# ]
# ///

import logging
import re
from collections import Counter
import sys

def patch_nuitka_resource_reader_hashable():
    for name, mod in list(sys.modules.items()):
        if name and name.startswith("nuitka_resource_reader"):
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if isinstance(obj, type) and not hasattr(obj, "__hash__"):
                    setattr(obj, "__hash__", lambda self: id(self))

patch_nuitka_resource_reader_hashable()

# Only patch if running in a Nuitka-built, frozen executable
if getattr(sys, 'frozen', False) and globals().get('__compiled__', False):
    import pyphen
    import os

    orig_load = pyphen.Pyphen._load
    def safe_load(self, lang):
        try:
            return orig_load(self, lang)
        except TypeError:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            dict_path = os.path.join(base_dir, "pyphen", "dictionaries", f"{lang}.dic")
            with open(dict_path, "r", encoding="utf-8") as f:
                return f.read()
    pyphen.Pyphen._load = safe_load

    # Patch __init__ to always use the file path for dictionaries
    orig_init = pyphen.Pyphen.__init__
    def safe_init(self, *args, **kwargs):
        if 'filename' not in kwargs and 'lang' in kwargs:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            dict_path = os.path.join(base_dir, "pyphen", "dictionaries", f"{kwargs['lang']}.dic")
            kwargs['filename'] = dict_path
        orig_init(self, *args, **kwargs)
    pyphen.Pyphen.__init__ = safe_init

import spacy
from markdown_it import MarkdownIt
from markdown_it.token import Token
from mcp.server.fastmcp import FastMCP
from spellchecker import SpellChecker
from textstat import flesch_kincaid_grade, flesch_reading_ease, gunning_fog, textstat

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


mcp = FastMCP("Writing Tools MCP Server")

# Load spaCy model
try:
    try:
        nlp = spacy.load("en_core_web_sm")
    except TypeError as e:
        # This can happen in Nuitka builds due to unhashable resource reader
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "en_core_web_sm")
        nlp = spacy.load(model_path)
except OSError:
    # If the model isn't installed, show a helpful message
    logging.info("Downloading spaCy English model, this may take a while...")
    from spacy.cli import download

    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """
    Preprocess text using spaCy:
    - Tokenize
    - Remove stopwords (optional)
    - Lemmatize/stem words (optional)
    """
    doc = nlp(text.lower())

    if remove_stopwords and lemmatize:
        # Return lemmatized tokens that aren't stopwords, punctuation, or whitespace
        return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    elif remove_stopwords:
        # Return tokens that aren't stopwords, punctuation, or whitespace
        return [token.text for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    elif lemmatize:
        # Return lemmatized tokens that aren't punctuation or whitespace
        return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    else:
        # Return all tokens that aren't punctuation or whitespace
        return [token.text for token in doc if not token.is_punct and not token.is_space]


def _render_tokens_to_text(tokens: list[Token]) -> str:
    """Helper to render markdown tokens to plain text, excluding headings."""
    text_content = []
    in_heading = False
    for token in tokens:
        # Simplified heading skipping logic
        if token.type == "heading_open":
            in_heading = True
            continue
        if token.type == "heading_close":
            in_heading = False
            continue
        if in_heading:
            continue  # Skip the inline content within the heading

        # Original rendering logic for non-heading tokens
        if token.type == "text":
            text_content.append(token.content)
        elif token.type == "paragraph_open":
            if text_content and not text_content[-1].endswith("\n\n"):
                text_content.append("\n\n")
        elif token.type == "paragraph_close":
            pass  # Don't add extra newlines after paragraph
        elif token.type == "inline" and token.children:
            # Keep track if the last added element was text to avoid double space/newlines
            last_was_text = text_content and not text_content[-1].endswith("\n")
            for i, child in enumerate(token.children):
                if child.type == "text":
                    # Add space between consecutive text elements unless previous ended with newline
                    if i > 0 and child.content and not text_content[-1].endswith("\n"):
                        # Heuristic: Add space if no space exists
                        if not text_content[-1].endswith(" "):
                            text_content.append(" ")
                    text_content.append(child.content)
                    last_was_text = True
                elif child.type == "softbreak":
                    text_content.append("\n")
                    last_was_text = False
                elif child.type == "hardbreak":
                    text_content.append("\n\n")  # Treat hardbreak as paragraph break
                    last_was_text = False
                # Add handling for other inline elements (code, strong, em, etc.)
                elif child.type == "code_inline":
                    if last_was_text and not text_content[-1].endswith(" "):
                        text_content.append(" ")
                    text_content.append(f"`{child.content}`")
                    if not child.content.endswith(" "):
                        text_content.append(" ")
                    last_was_text = True  # Assume code ends with space conceptually
        # Add handling for other block elements (fence, lists, blockquote)
        elif token.type == "fence":
            if text_content and not text_content[-1].endswith("\n\n"):
                text_content.append("\n\n")
            text_content.append(f"```\n{token.content.strip()}\n```\n\n")
        elif token.type == "bullet_list_open" or token.type == "ordered_list_open":
            if text_content and not text_content[-1].endswith("\n\n"):
                text_content.append("\n\n")
        elif token.type == "list_item_open":
            prefix = (
                "* "
                if tokens[tokens.index(token) - 1].type == "bullet_list_open"
                else f"{tokens[tokens.index(token) - 1].meta.get('start', 1)}. "
            )
            text_content.append(prefix)
        elif token.type == "list_item_close":
            text_content.append("\n")

    # Join and clean up extra whitespace/newlines
    result = "".join(text_content).strip()
    result = re.sub(r"\n{3,}", "\n\n", result)  # Collapse excess newlines
    return result


def parse_markdown_sections(text):
    """
    Parse markdown text into sections based on headings (h1-h6).
    Uses markdown-it-py for robust parsing.
    Returns a dictionary with heading text as keys and content as values.
    Top level has key "full_text". Includes paragraphs for full text and sections.
    """
    md = MarkdownIt()
    tokens = md.parse(text)

    sections = {"full_text": text}
    current_section_key = None
    current_section_content_tokens = []
    current_section_level = 0
    section_data = []  # Store (level, key, content_tokens)
    in_heading = False  # Track if we are inside heading tokens

    # First pass: identify headings and group content tokens under them
    content_buffer_tokens = []
    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            in_heading = True
            # Store content of the previous section/buffer
            if current_section_key:
                section_data.append(
                    (
                        current_section_level,
                        current_section_key,
                        current_section_content_tokens,
                    )
                )
            elif content_buffer_tokens:  # Content before the first heading
                section_data.append((0, "_leading_content", content_buffer_tokens))

            current_section_level = int(token.tag[1])  # h1 -> 1
            current_section_content_tokens = []  # Reset content tokens
            content_buffer_tokens = []  # Reset buffer

            # Extract heading key from the NEXT inline token
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                current_section_key = f"{'#' * current_section_level} {tokens[i + 1].content.strip()}"
            else:
                current_section_key = f"{'#' * current_section_level} Untitled Section"
            # Skip the heading_open token itself
            continue

        if token.type == "heading_close":
            in_heading = False
            # Skip the heading_close token
            continue

        if in_heading:
            # Skip the inline token within the heading
            continue

        # Add content tokens to the current section or buffer
        if current_section_key:
            current_section_content_tokens.append(token)
        else:
            content_buffer_tokens.append(token)

    # Add the last section or remaining buffer content
    if current_section_key:
        section_data.append((current_section_level, current_section_key, current_section_content_tokens))
    elif content_buffer_tokens:
        section_data.append((0, "_leading_content", content_buffer_tokens))

    # Second pass: reconstruct text content for each section, handling hierarchy
    temp_sections_tokens = {}  # Stores tokens for each section key

    # Create a lookup for faster access
    section_data_lookup = {key: (level, tokens) for level, key, tokens in section_data}

    # Process leading content first (if any)
    if "_leading_content" in section_data_lookup:
        temp_sections_tokens["_leading_content"] = section_data_lookup["_leading_content"][1]

    # Process actual sections
    processed_keys = {"_leading_content"}  # Keep track of processed keys
    active_hierarchy = []  # Stack of (level, key, tokens)

    for level, key, content_tokens in section_data:
        if key == "_leading_content":
            continue

        # Pop sections from hierarchy stack that are at the same or higher level
        while active_hierarchy and active_hierarchy[-1][0] >= level:
            processed_level, processed_key, processed_tokens = active_hierarchy.pop()
            # Assign collected tokens to the key being popped
            if processed_key not in processed_keys:
                temp_sections_tokens[processed_key] = processed_tokens
                processed_keys.add(processed_key)

        # Add current section's direct content to the tokens of the parent in the hierarchy
        if active_hierarchy:
            active_hierarchy[-1][2].extend(content_tokens)

        # Push current section onto the hierarchy stack with its *direct* content
        active_hierarchy.append([level, key, list(content_tokens)])  # Use list() for mutable copy

    # Process any remaining sections left in the hierarchy stack
    while active_hierarchy:
        processed_level, processed_key, processed_tokens = active_hierarchy.pop()
        if processed_key not in processed_keys:
            temp_sections_tokens[processed_key] = processed_tokens
            processed_keys.add(processed_key)

    # Final assembly into the desired output format: Render tokens to text
    # Add top-level paragraphs (split the original text)
    sections["paragraphs"] = split_paragraphs(text)

    for key, tokens_list in temp_sections_tokens.items():
        if key == "_leading_content":
            # Maybe handle leading content separately if needed
            pass
        else:
            section_text = _render_tokens_to_text(tokens_list)
            sections[key] = section_text
            sections[f"{key}_paragraphs"] = split_paragraphs(section_text)

    # Ensure no list values accidentally assigned to section keys
    for key, value in sections.items():
        if isinstance(value, list) and not key.endswith("_paragraphs") and key != "paragraphs":
            logging.warning(f"Section key '{key}' has list value: {value}")  # Debug print
            # Decide how to handle this - maybe convert list to string?
            # For now, let's try joining if it's a list of strings
            if all(isinstance(item, str) for item in value):
                sections[key] = "\n\n".join(value)
            else:
                sections[key] = str(value)  # Fallback

    return sections


def split_paragraphs(text):
    """
    Split text into paragraphs.
    Returns a list of paragraphs.
    """
    # Split on empty lines (two or more newlines)
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def strip_markdown_markup(text: str) -> str:
    """
    Strips markdown markup from text, extracting a plain text representation.
    Handles links (uses link text) and images (uses alt text).
    Aims for a simple and reliable extraction, not exhaustive.
    """
    md = MarkdownIt()
    tokens = md.parse(text)

    content_parts = []

    for token in tokens:
        if token.type == "text":
            content_parts.append(token.content)
        elif token.type == "image":
            # token.content is the alt text for an image
            content_parts.append(token.content)
        elif token.type == "code_inline":
            content_parts.append(token.content)
        elif token.type == "fence":  # Block code
            content_parts.append(token.content)
            if content_parts and not content_parts[-1].endswith("\n"):
                content_parts.append("\n")
        elif token.type == "softbreak":
            content_parts.append("\n")
        elif token.type == "hardbreak":
            content_parts.append("\n\n")
        elif token.type == "html_inline" or token.type == "html_block":
            # For simplicity, ignoring HTML content/tags as per "simple and reliable" for MD.
            pass
        elif token.type == "inline" and token.children:
            # Process children of inline tokens (e.g., text within emphasis, strong, links)
            for child in token.children:
                if child.type == "text":
                    content_parts.append(child.content)
                elif child.type == "image":
                    content_parts.append(child.content)  # alt text
                elif child.type == "code_inline":
                    content_parts.append(child.content)
                elif child.type == "softbreak":
                    content_parts.append("\n")
                elif child.type == "hardbreak":
                    content_parts.append("\n\n")
        elif token.type == "paragraph_close":
            current_text_so_far = "".join(content_parts)
            if content_parts and not current_text_so_far.endswith("\n\n"):
                if current_text_so_far.endswith("\n"):
                    content_parts.append("\n")
                else:
                    content_parts.append("\n\n")
            elif not content_parts:  # Handle case of empty paragraph if it results in tokens
                content_parts.append("\n\n")
        elif token.type == "list_item_close":
            current_text_so_far = "".join(content_parts)
            if (
                current_text_so_far
                and not current_text_so_far.endswith("\n")
                and not current_text_so_far.endswith("\n\n")
            ):
                content_parts.append("\n")

    result = "".join(content_parts)

    # Normalize spacing
    result = re.sub(r" +", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = result.strip()

    return result


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
    ]


@mcp.tool()
def character_count(text: str) -> int:
    """Calculates the total number of characters in the provided text.

    Args:
        text: The input string to count characters from.

    Returns:
        int: The total character count of the input text.
    """
    return len(text)


@mcp.tool()
def word_count(text: str) -> int:
    """Calculates the total number of words in the provided text, splitting by whitespace.

    Args:
        text: The input string to count words from.

    Returns:
        int: The total word count of the input text.
    """
    return len(text.split())


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
    spell = SpellChecker()
    # Use preprocess_text to get clean words, without lemmatization or stopword removal
    words = preprocess_text(text, remove_stopwords=False, lemmatize=False)
    misspelled = spell.unknown(words)
    return list(misspelled)


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

    def get_scores(text_segment):
        text_segment = strip_markdown_markup(text_segment)
        if not text_segment or len(text_segment.split()) < 3:
            return {"flesch": None, "kincaid": None, "fog": None}
        return {
            "flesch": flesch_reading_ease(text_segment),
            "kincaid": flesch_kincaid_grade(text_segment),
            "fog": gunning_fog(text_segment),
        }

    if level == "full":
        return get_scores(text)

    elif level == "section":
        sections_data = parse_markdown_sections(text)
        # Match structure of reading_time result
        result = {"full_text": get_scores(text), "sections": {}}

        for section_name, section_content in sections_data.items():
            # Extra check: ensure section_content is a string before processing
            if (
                isinstance(section_content, str)
                and section_name != "full_text"
                and not section_name.endswith("_paragraphs")
                and section_name != "paragraphs"
            ):
                result["sections"][section_name] = get_scores(section_content)
            elif (
                not isinstance(section_content, str)
                and section_name != "full_text"
                and not section_name.endswith("_paragraphs")
                and section_name != "paragraphs"
            ):
                # This case should not happen based on parse_markdown_sections, but adding safety print
                logging.warning(
                    f"[Readability Score] Warning: Skipping non-string content for section '{section_name}': {type(section_content)}"
                )

        return result

    elif level == "paragraph":
        result = {"full_text": get_scores(text), "paragraphs": []}

        # Get paragraphs and score each one
        paragraphs = split_paragraphs(text)
        for i, paragraph in enumerate(paragraphs):
            result["paragraphs"].append(
                {
                    "paragraph_number": i + 1,
                    "text": paragraph[:50] + "..." if len(paragraph) > 50 else paragraph,
                    "scores": get_scores(paragraph),
                }
            )

        return result

    else:
        return {"error": f"Invalid level: {level}. Choose from 'full', 'section', or 'paragraph'."}


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

    ms_per_character = 28  # default is ~14

    def get_reading_time(text_segment):
        if not text_segment:
            return 0
        # Convert milliseconds to minutes
        return textstat.reading_time(text_segment, ms_per_char=ms_per_character) / 60

    if level == "full":
        return {"full_text": get_reading_time(text)}

    elif level == "section":
        sections_data = parse_markdown_sections(text)
        result = {"full_text": get_reading_time(text), "sections": {}}

        for section_name, section_content in sections_data.items():
            # Extra check: ensure section_content is a string before processing
            if (
                isinstance(section_content, str)
                and section_name != "full_text"
                and not section_name.endswith("_paragraphs")
                and section_name != "paragraphs"
            ):
                result["sections"][section_name] = get_reading_time(section_content)
            elif (
                not isinstance(section_content, str)
                and section_name != "full_text"
                and not section_name.endswith("_paragraphs")
                and section_name != "paragraphs"
            ):
                # Safety print
                logging.warning(
                    f"[Reading Time] Warning: Skipping non-string content for section '{section_name}': {type(section_content)}"
                )

        return result

    elif level == "paragraph":
        paragraphs = split_paragraphs(text)
        result = {"full_text": get_reading_time(text), "paragraphs": []}

        # Calculate reading time for each paragraph
        for i, paragraph in enumerate(paragraphs):
            result["paragraphs"].append(
                {
                    "paragraph_number": i + 1,
                    "text": paragraph[:50] + "..." if len(paragraph) > 50 else paragraph,
                    "reading_time_minutes": get_reading_time(paragraph),
                }
            )

        return result

    else:
        return {"error": f"Invalid level: {level}. Choose from 'full', 'section', or 'paragraph'."}


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
    processed_text = preprocess_text(text)
    processed_keyword = preprocess_text(keyword)[0] if preprocess_text(keyword) else keyword.lower()

    keyword_count = processed_text.count(processed_keyword)
    return (keyword_count / len(processed_text)) * 100 if processed_text else 0


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
    processed_text = preprocess_text(text, remove_stopwords=remove_stopwords)
    return dict(Counter(processed_text))


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
    processed_text = preprocess_text(text, remove_stopwords=remove_stopwords)
    frequency = Counter(processed_text)
    return frequency.most_common(top_n)


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
    doc = nlp(text)

    # Process the keyword to get its lemma
    keyword_doc = nlp(keyword.lower())
    keyword_lemma = keyword_doc[0].lemma_ if len(keyword_doc) > 0 else keyword.lower()

    # Extract sentences containing the keyword or its lemmatized form
    contexts = []
    for sent in doc.sents:
        if any(token.text.lower() == keyword.lower() or token.lemma_ == keyword_lemma for token in sent):
            contexts.append(sent.text)

    return contexts


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
    doc = nlp(text)
    passive_sentences = []

    for sent in doc.sents:
        # Simple passive voice detection: aux verb + past participle
        # This is a simplified approach and might miss some complex passive constructions
        for token in sent:
            if token.dep_ == "auxpass" or (
                token.pos_ == "AUX" and any(child.tag_ == "VBN" for child in token.children)
            ):
                passive_sentences.append(sent.text)
                break

    return passive_sentences


if __name__ == "__main__":
    logging.info("Starting MCP server...")
    mcp.run()
