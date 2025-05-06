from server import (
    readability_score,
    reading_time,
    split_paragraphs,
    parse_markdown_sections,
)

# --- Test Helper Functions ---

MARKDOWN_EXAMPLE = """
# Section 1

This is the first paragraph.
It has two sentences.

This is the second paragraph.

## Subsection 1.1

Content here.

# Section 2

Another top-level section.

With a paragraph spanning
multiple lines.


"""

EXPECTED_PARAGRAPHS_FULL = [
    "# Section 1",
    """This is the first paragraph.
It has two sentences.""",
    "This is the second paragraph.",
    "## Subsection 1.1",
    "Content here.",
    "# Section 2",
    "Another top-level section.",
    """With a paragraph spanning
multiple lines.""",
]

EXPECTED_SECTIONS = {
    "full_text": MARKDOWN_EXAMPLE.strip(),
    "# Section 1": """This is the first paragraph.
It has two sentences.

This is the second paragraph.

Content here.""",
    "## Subsection 1.1": "Content here.",
    "# Section 2": """Another top-level section.

With a paragraph spanning
multiple lines.""",
    # Paragraphs within sections (added by parse_markdown_sections)
    "# Section 1_paragraphs": [
        """This is the first paragraph.
It has two sentences.""",
        "This is the second paragraph.",
        "Content here.",
    ],
    "## Subsection 1.1_paragraphs": ["Content here."],
    "# Section 2_paragraphs": [
        "Another top-level section.",
        """With a paragraph spanning
multiple lines.""",
    ],
}
# Add full text paragraphs as well
EXPECTED_SECTIONS["paragraphs"] = EXPECTED_PARAGRAPHS_FULL


def test_split_paragraphs():
    """Test splitting text into paragraphs."""
    text = "Para 1\n\nPara 2\n\n\nPara 3"
    expected = ["Para 1", "Para 2", "Para 3"]
    assert split_paragraphs(text) == expected

    text_md = MARKDOWN_EXAMPLE.strip()
    assert split_paragraphs(text_md) == EXPECTED_PARAGRAPHS_FULL

    assert split_paragraphs("") == []
    assert split_paragraphs("Single paragraph") == ["Single paragraph"]


def test_parse_markdown_sections():
    """Test parsing markdown text into sections."""
    text = MARKDOWN_EXAMPLE.strip()
    result = parse_markdown_sections(text)

    # Check section keys and content
    assert set(result.keys()) == set(EXPECTED_SECTIONS.keys())
    assert result["# Section 1"] == EXPECTED_SECTIONS["# Section 1"]
    assert result["## Subsection 1.1"] == EXPECTED_SECTIONS["## Subsection 1.1"]
    assert result["# Section 2"] == EXPECTED_SECTIONS["# Section 2"]
    assert result["full_text"] == EXPECTED_SECTIONS["full_text"]

    # Check paragraphs within sections
    assert (
        result["# Section 1_paragraphs"] == EXPECTED_SECTIONS["# Section 1_paragraphs"]
    )
    assert (
        result["## Subsection 1.1_paragraphs"]
        == EXPECTED_SECTIONS["## Subsection 1.1_paragraphs"]
    )
    assert (
        result["# Section 2_paragraphs"] == EXPECTED_SECTIONS["# Section 2_paragraphs"]
    )
    assert (
        result["paragraphs"] == EXPECTED_SECTIONS["paragraphs"]
    )  # paragraphs for full text


def test_parse_markdown_no_headings():
    """Test parsing markdown with no headings."""
    text = "Just some text.\n\nAnother paragraph."
    expected = {
        "full_text": text,
        "paragraphs": ["Just some text.", "Another paragraph."],
    }
    assert parse_markdown_sections(text) == expected


# --- Test Readability Score ---

SIMPLE_TEXT = "This is a simple sentence. It should be easy to read."
# Flesch Reading Ease: ~87.7
# Flesch-Kincaid Grade: ~2.4
# Gunning Fog: ~2.2


def test_readability_score_full():
    """Test readability score for the full text."""
    scores = readability_score(SIMPLE_TEXT, level="full")
    assert isinstance(scores, dict)
    assert "flesch" in scores and isinstance(scores["flesch"], (int, float))
    assert "kincaid" in scores and isinstance(scores["kincaid"], (int, float))
    assert "fog" in scores and isinstance(scores["fog"], (int, float))
    # Check approximate values (allow some tolerance)
    assert abs(scores["flesch"] - 87.7) < 5
    assert abs(scores["kincaid"] - 2.4) < 1
    assert abs(scores["fog"] - 2.2) < 0.1


def test_readability_score_sections():
    """Test readability score by markdown sections."""
    text = MARKDOWN_EXAMPLE.strip()
    scores = readability_score(text, level="section")
    assert isinstance(scores, dict)
    assert "full_text" in scores
    assert "sections" in scores and isinstance(scores["sections"], dict)
    assert "# Section 1" in scores["sections"]
    assert "## Subsection 1.1" in scores["sections"]
    assert "# Section 2" in scores["sections"]
    assert isinstance(scores["full_text"]["flesch"], (int, float))
    assert isinstance(scores["sections"]["# Section 1"]["flesch"], (int, float))


def test_readability_score_paragraphs():
    """Test readability score by paragraphs."""
    text = MARKDOWN_EXAMPLE.strip()
    scores = readability_score(text, level="paragraph")
    assert isinstance(scores, dict)
    assert "full_text" in scores
    assert "paragraphs" in scores and isinstance(scores["paragraphs"], list)
    assert len(scores["paragraphs"]) == len(EXPECTED_PARAGRAPHS_FULL)
    assert "paragraph_number" in scores["paragraphs"][0]
    assert "text" in scores["paragraphs"][0]
    assert "scores" in scores["paragraphs"][0]
    assert isinstance(scores["paragraphs"][0]["scores"]["flesch"], (int, float))


def test_readability_score_short_text():
    """Test readability score returns None for very short text."""
    short_text = "One two."
    scores = readability_score(short_text, level="full")
    assert scores["flesch"] is None
    assert scores["kincaid"] is None
    assert scores["fog"] is None


def test_readability_score_invalid_level():
    """Test readability score with an invalid level."""
    scores = readability_score(SIMPLE_TEXT, level="invalid")
    assert "error" in scores


# --- Test Reading Time ---


def test_reading_time_full():
    """Test reading time for the full text."""
    time_info = reading_time(SIMPLE_TEXT, level="full")
    assert isinstance(time_info, dict)
    assert "full_text" in time_info
    assert isinstance(time_info["full_text"], float)
    assert time_info["full_text"] > 0  # Should take some time


def test_reading_time_sections():
    """Test reading time by markdown sections."""
    text = MARKDOWN_EXAMPLE.strip()
    time_info = reading_time(text, level="section")
    assert isinstance(time_info, dict)
    assert "full_text" in time_info
    assert "sections" in time_info and isinstance(time_info["sections"], dict)
    assert "# Section 1" in time_info["sections"]
    assert "## Subsection 1.1" in time_info["sections"]
    assert "# Section 2" in time_info["sections"]
    assert isinstance(time_info["sections"]["# Section 1"], float)


def test_reading_time_paragraphs():
    """Test reading time by paragraphs."""
    text = MARKDOWN_EXAMPLE.strip()
    time_info = reading_time(text, level="paragraph")
    assert isinstance(time_info, dict)
    assert "full_text" in time_info
    assert "paragraphs" in time_info and isinstance(time_info["paragraphs"], list)
    assert len(time_info["paragraphs"]) == len(EXPECTED_PARAGRAPHS_FULL)
    assert "paragraph_number" in time_info["paragraphs"][0]
    assert "text" in time_info["paragraphs"][0]
    assert "reading_time_minutes" in time_info["paragraphs"][0]
    assert isinstance(time_info["paragraphs"][0]["reading_time_minutes"], float)


def test_reading_time_empty_text():
    """Test reading time for empty text."""
    time_info = reading_time("", level="full")
    assert time_info["full_text"] == 0


def test_reading_time_invalid_level():
    """Test reading time with an invalid level."""
    time_info = reading_time(SIMPLE_TEXT, level="invalid")
    assert "error" in time_info
