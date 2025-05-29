"""Test module for reading time functionality."""

from server.analyzers import ReadabilityAnalyzer


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
