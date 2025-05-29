"""Test module for spellcheck functionality."""

from unittest.mock import MagicMock, patch

from server.analyzers import BasicStatsAnalyzer


class TestSpellcheck:
    """Test spellcheck functionality."""

    def setup_method(self):
        """Set up BasicStatsAnalyzer instance for testing."""
        self.analyzer = BasicStatsAnalyzer()

    @patch("server.analyzers.basic_stats.SpellChecker")
    def test_spellcheck_basic(self, mock_spell_checker):
        """Test basic spellcheck functionality."""
        # Mock the spell checker
        mock_checker = MagicMock()
        mock_checker.unknown.return_value = {"wrng", "speling"}
        mock_spell_checker.return_value = mock_checker

        result = self.analyzer.spellcheck("This is wrng speling")
        assert isinstance(result, list)

    @patch("server.analyzers.basic_stats.SpellChecker")
    def test_spellcheck_empty(self, mock_spell_checker):
        """Test spellcheck with empty text."""
        mock_checker = MagicMock()
        mock_checker.unknown.return_value = set()
        mock_spell_checker.return_value = mock_checker

        result = self.analyzer.spellcheck("")
        assert result == []

    @patch("server.analyzers.basic_stats.SpellChecker")
    def test_spellcheck_correct_text(self, mock_spell_checker):
        """Test spellcheck with correctly spelled text."""
        mock_checker = MagicMock()
        mock_checker.unknown.return_value = set()
        mock_spell_checker.return_value = mock_checker

        result = self.analyzer.spellcheck("This is correct text")
        assert result == []
