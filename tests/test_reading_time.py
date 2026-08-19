"""Test module for reading time functionality."""

from server.analyzers import ReadabilityAnalyzer
from server.text_processing import strip_markdown_markup


class TestReadingTime:
    """Test reading time functionality."""

    def setup_method(self):
        """Set up ReadabilityAnalyzer instance for testing."""
        self.analyzer = ReadabilityAnalyzer()

    def test_reading_time_basic(self):
        """Test basic reading time calculation."""
        text = "This is a test sentence. It should take some time to read."
        result = self.analyzer.reading_time(text)

        assert isinstance(result, dict)
        assert "full_text" in result
        assert isinstance(result["full_text"], (int, float))
        assert result["full_text"] >= 0

    def test_reading_time_empty(self):
        """Test reading time with empty text."""
        result = self.analyzer.reading_time("")
        assert result["full_text"] == 0

    def test_reading_time_sections(self):
        """Test reading time by sections."""
        markdown_text = """
# Section 1

This is content for section one.

# Section 2

This is content for section two.
"""
        result = self.analyzer.reading_time(markdown_text.strip(), level="section")

        assert isinstance(result, dict)
        assert "full_text" in result
        assert "sections" in result

    def test_reading_time_paragraphs(self):
        """Test reading time by paragraphs."""
        text = """First paragraph with some content.

Second paragraph with more content."""

        result = self.analyzer.reading_time(text, level="paragraph")

        assert isinstance(result, dict)
        assert "full_text" in result
        assert "paragraphs" in result

    def test_reading_time_ignores_link_urls(self):
        """A link's target URL contributes nothing to the estimate."""
        markdown = "See the [installation guide](https://example.com/docs/installation/getting-started) for details."
        plain = "See the installation guide for details."

        assert self.analyzer.reading_time(markdown)["full_text"] == self.analyzer.reading_time(plain)["full_text"]

    def test_reading_time_matches_stripped_equivalent(self):
        """Markdown syntax is stripped before timing, so markup costs nothing."""
        markdown = (
            "# Getting started\n\n"
            "See the [installation guide](https://example.com/docs/installation/getting-started) "
            "for **details**.\n\n"
            "![Diagram of the pipeline](https://example.com/assets/images/pipeline-diagram.png)\n\n"
            "```python\n"
            'print("hello")\n'
            "```\n"
        )
        stripped = strip_markdown_markup(markdown)

        assert markdown != stripped
        assert self.analyzer.reading_time(markdown)["full_text"] == self.analyzer.reading_time(stripped)["full_text"]

    def test_reading_time_sections_ignore_link_urls(self):
        """Section-level results also exclude link targets, keeping their existing shape."""
        markdown = "# Section 1\n\nRead the [guide](https://example.com/a/very/long/target/url/indeed) now."
        plain = "# Section 1\n\nRead the guide now."

        result = self.analyzer.reading_time(markdown, level="section")
        expected = self.analyzer.reading_time(plain, level="section")

        assert result["full_text"] == expected["full_text"]
        assert result["sections"] == expected["sections"]
        assert "# Section 1" in result["sections"]

    def test_reading_time_whitespace_only(self):
        """Whitespace-only input has nothing to read."""
        assert self.analyzer.reading_time("   \n\n  ")["full_text"] == 0
