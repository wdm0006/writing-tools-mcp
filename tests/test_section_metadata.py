"""Parser gap regressions fixed for section-level batch analysis (W5).

Two gaps in ``parse_markdown_sections`` are closed here:

1. Content before the first heading used to be collected and then silently
   dropped at final assembly — it now becomes its own ``_leading_content``
   section (heading level 0).
2. Heading level was not exposed in the parsed output — it now appears in the
   ``_section_levels`` metadata mapping, whose insertion order is document
   order.
"""

from server.text_processing import parse_markdown_sections

LEADING_CONTENT_DOC = (
    "Leading paragraph before any heading.\n\n"
    "## First\n\n"
    "Alpha content.\n\n"
    "### Nested\n\n"
    "Nested content.\n\n"
    "## Second\n\n"
    "Beta content.\n"
)


class TestLeadingContentSection:
    """Pre-first-heading content is a real section — the silent drop cannot recur."""

    def test_leading_content_becomes_its_own_section(self):
        result = parse_markdown_sections(LEADING_CONTENT_DOC)

        assert result["_leading_content"] == "Leading paragraph before any heading."

    def test_leading_content_gets_a_paragraphs_sibling(self):
        result = parse_markdown_sections(LEADING_CONTENT_DOC)

        assert result["_leading_content_paragraphs"] == ["Leading paragraph before any heading."]

    def test_leading_content_is_not_absorbed_by_the_first_heading(self):
        """The first heading's section carries only its own body."""
        result = parse_markdown_sections(LEADING_CONTENT_DOC)

        assert result["## First"] == "Alpha content.\n\nNested content."

    def test_leading_content_markdown_is_rendered(self):
        # Removed emphasis markers leave double spaces, same as headed sections
        # (see TestSectionTextRendering for the documented quirk).
        result = parse_markdown_sections("A **bold** lead.\n\n## Heading\n\nBody.\n")

        assert result["_leading_content"] == "A bold  lead."


class TestSectionLevels:
    """Heading level is exposed per section key, in document order."""

    def test_every_section_gets_its_level(self):
        result = parse_markdown_sections(LEADING_CONTENT_DOC)

        assert result["_section_levels"] == {
            "_leading_content": 0,
            "## First": 2,
            "### Nested": 3,
            "## Second": 2,
        }

    def test_levels_are_in_document_order(self):
        result = parse_markdown_sections(LEADING_CONTENT_DOC)

        assert list(result["_section_levels"]) == ["_leading_content", "## First", "### Nested", "## Second"]

    def test_disambiguated_repeats_share_the_heading_level(self):
        text = "## Notes\n\nAlpha.\n\n## Notes\n\nBeta.\n"
        result = parse_markdown_sections(text)

        assert result["_section_levels"] == {"## Notes": 2, "## Notes (2)": 2}

    def test_nested_headings_resolve_to_their_owning_section(self):
        """Existing hierarchy behavior is unchanged: the h3 body folds into the h2."""
        result = parse_markdown_sections(LEADING_CONTENT_DOC)

        assert result["_section_levels"]["### Nested"] == 3
        assert "Nested content." in result["## First"]
        assert result["### Nested"] == "Nested content."


class TestDegenerateDocuments:
    """Empty and heading-less documents have defined parser behavior."""

    def test_empty_document_has_no_sections(self):
        assert parse_markdown_sections("") == {"full_text": "", "paragraphs": [], "_section_levels": {}}

    def test_whitespace_only_document_has_no_sections(self):
        assert parse_markdown_sections("   \n  ") == {"full_text": "   \n  ", "paragraphs": [], "_section_levels": {}}
