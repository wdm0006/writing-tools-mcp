"""Test module for section-level analysis of documents with repeated headings."""

from server.analyzers import ReadabilityAnalyzer
from server.text_processing import parse_markdown_sections

# Three sibling headings sharing the same text, with bodies of different lengths so
# that a result which reuses one body for every entry cannot satisfy the assertions.
FLAT_SIBLINGS = (
    "## Notes\n\n"
    "Alpha content one here.\n\n"
    "## Notes\n\n"
    "Beta content two here and rather more words besides.\n\n"
    "## Notes\n\n"
    "Gamma three words.\n"
)

FLAT_SIBLING_BODIES = {
    "## Notes": "Alpha content one here.",
    "## Notes (2)": "Beta content two here and rather more words besides.",
    "## Notes (3)": "Gamma three words.",
}

UNIQUE_HEADINGS = "# Section 1\n\nContent for section one.\n\n## Subsection 1.1\n\nSubsection content.\n\n# Section 2\n\nContent for section two.\n"


def section_keys(sections_data):
    """The heading keys of a parse result, excluding paragraph siblings."""
    return [key for key in sections_data if key not in ("full_text", "paragraphs") and not key.endswith("_paragraphs")]


class TestParseDuplicateHeadings:
    """Repeated headings each get their own entry in parse_markdown_sections."""

    def test_repeated_siblings_each_get_their_own_entry(self):
        result = parse_markdown_sections(FLAT_SIBLINGS)

        assert sorted(section_keys(result)) == sorted(FLAT_SIBLING_BODIES)
        for key, body in FLAT_SIBLING_BODIES.items():
            assert result[key] == body

    def test_every_section_key_has_a_paragraphs_sibling(self):
        result = parse_markdown_sections(FLAT_SIBLINGS)

        for key, body in FLAT_SIBLING_BODIES.items():
            assert result[f"{key}_paragraphs"] == [body]

    def test_unique_headings_keep_their_keys(self):
        result = parse_markdown_sections(UNIQUE_HEADINGS)

        assert sorted(section_keys(result)) == ["# Section 1", "# Section 2", "## Subsection 1.1"]

    def test_suffix_walks_past_a_literal_collision(self):
        """A heading that already spells the suffix does not steal a later key."""
        text = "## Notes\n\nAlpha one.\n\n## Notes (2)\n\nBeta two.\n\n## Notes\n\nGamma three.\n"
        result = parse_markdown_sections(text)

        assert sorted(section_keys(result)) == ["## Notes", "## Notes (2)", "## Notes (3)"]
        assert result["## Notes"] == "Alpha one."
        assert result["## Notes (2)"] == "Beta two."
        assert result["## Notes (3)"] == "Gamma three."


class TestDuplicateHeadingReadability:
    """Both section-level readability tools report every heading occurrence."""

    def setup_method(self):
        self.analyzer = ReadabilityAnalyzer()

    def test_reading_time_scores_each_occurrence_separately(self):
        sections = self.analyzer.reading_time(FLAT_SIBLINGS, level="section")["sections"]

        assert sorted(sections) == sorted(FLAT_SIBLING_BODIES)
        for key, body in FLAT_SIBLING_BODIES.items():
            assert sections[key] == self.analyzer.reading_time(body)["full_text"]

    def test_readability_score_scores_each_occurrence_separately(self):
        sections = self.analyzer.readability_score(FLAT_SIBLINGS, level="section")["sections"]

        assert sorted(sections) == sorted(FLAT_SIBLING_BODIES)
        for key, body in FLAT_SIBLING_BODIES.items():
            assert sections[key] == self.analyzer.readability_score(body, level="full")

    def test_no_paragraph_entries_leak_into_sections(self):
        for result in (
            self.analyzer.reading_time(FLAT_SIBLINGS, level="section"),
            self.analyzer.readability_score(FLAT_SIBLINGS, level="section"),
        ):
            assert not any(key.endswith("_paragraphs") for key in result["sections"])
            assert "paragraphs" not in result["sections"]

    def test_repeated_subsections_under_different_parents(self):
        """A README-shaped document reports all four sections, not three."""
        readme = (
            "## Setup\n\nInstall the package first.\n\n"
            "### Example\n\nRun the setup command here.\n\n"
            "## Usage\n\nCall the tool from a client.\n\n"
            "### Example\n\nRun the usage command.\n"
        )
        sections = self.analyzer.reading_time(readme, level="section")["sections"]

        assert sorted(sections) == ["## Setup", "## Usage", "### Example", "### Example (2)"]
        assert sections["### Example"] == self.analyzer.reading_time("Run the setup command here.")["full_text"]
        assert sections["### Example (2)"] == self.analyzer.reading_time("Run the usage command.")["full_text"]
