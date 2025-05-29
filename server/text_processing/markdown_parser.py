"""Markdown parsing utilities."""

import re
from typing import Any, Dict, List

from markdown_it import MarkdownIt
from markdown_it.token import Token


def _render_tokens_to_text(tokens: List[Token]) -> str:
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


def parse_markdown_sections(text: str) -> Dict[str, Any]:
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
    from .sentence_splitter import split_paragraphs

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
            # Decide how to handle this - maybe convert list to string?
            # For now, let's try joining if it's a list of strings
            if all(isinstance(item, str) for item in value):
                sections[key] = "\n\n".join(value)
            else:
                sections[key] = str(value)  # Fallback

    return sections


def strip_markdown_markup(text: str) -> str:
    """Strip markdown markup from text, returning only the content.

    This is a simple and reliable implementation that extracts text content from markdown
    while preserving basic structure (paragraphs, line breaks) but removing all markup.
    """
    if not text or text.isspace():
        return ""

    md = MarkdownIt()
    tokens = md.parse(text.strip())
    content_parts = []

    for token in tokens:
        if token.type == "text":
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
            # For simplicity, ignoring HTML content/tags
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
            elif not content_parts:  # Handle case of empty paragraph
                content_parts.append("\n\n")
        elif token.type == "list_item_close":
            current_text_so_far = "".join(content_parts).strip()
            # Only add newlines for non-empty list items
            if current_text_so_far:
                if not current_text_so_far.endswith("\n\n"):
                    content_parts.append("\n\n")

    result = "".join(content_parts).strip()
    # Collapse multiple newlines into two
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result
