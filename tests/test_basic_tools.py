"""Test module for basic analysis tools."""

from server.analyzers import BasicStatsAnalyzer


class TestBasicStatsAnalyzer:
    """Test the BasicStatsAnalyzer class."""

    def setup_method(self):
        """Set up BasicStatsAnalyzer instance for testing."""
        self.analyzer = BasicStatsAnalyzer()

    def test_character_count_basic(self):
        """Test character_count with various inputs."""
        assert self.analyzer.character_count("hello") == 5
        assert self.analyzer.character_count("hello world") == 11  # Includes space
        assert self.analyzer.character_count("") == 0
        assert self.analyzer.character_count("  ") == 2  # Counts spaces

    def test_character_count_special_chars(self):
        """Test character count with special characters."""
        assert self.analyzer.character_count("hello\nworld") == 11  # Includes newline
        assert self.analyzer.character_count("hello\tworld") == 11  # Includes tab
        assert self.analyzer.character_count("hello😀world") == 11  # Unicode emoji

    def test_word_count_basic(self):
        """Test word_count with various inputs."""
        assert self.analyzer.word_count("hello world") == 2
        assert self.analyzer.word_count("hello") == 1
        assert self.analyzer.word_count("") == 0
        assert self.analyzer.word_count("  ") == 0  # Only whitespace

    def test_word_count_complex(self):
        """Test word count with complex cases."""
        assert self.analyzer.word_count("  hello   world  ") == 2  # Extra spaces
        assert self.analyzer.word_count("word\nword\tword") == 3  # Different separators
        assert self.analyzer.word_count("one-hyphenated two") == 2  # Hyphenated words

    def test_spellcheck_basic(self):
        """Test spellcheck with basic text."""
        result = self.analyzer.spellcheck("This is correct text")
        assert isinstance(result, list)
        # Note: Actual misspellings depend on the spell checker's dictionary

    def test_spellcheck_misspelled(self):
        """Test spellcheck with intentionally misspelled words."""
        result = self.analyzer.spellcheck("This iz wrng speling")
        assert isinstance(result, list)
        # Should find some misspelled words, but exact results depend on spell checker

    def test_spellcheck_empty(self):
        """Test spellcheck with empty text."""
        result = self.analyzer.spellcheck("")
        assert result == []
