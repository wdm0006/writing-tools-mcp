"""Test module for text processing utilities."""

from unittest.mock import MagicMock

from server.text_processing import (
    TextPreprocessor,
    parse_markdown_sections,
    preprocess_text,
    split_into_sentences,
    split_paragraphs,
    strip_markdown_markup,
)


class TestTextPreprocessor:
    """Test the TextPreprocessor class."""

    def setup_method(self):
        """Set up TextPreprocessor instance for testing."""
        # Create mock nlp model
        mock_nlp = MagicMock()

        def mock_nlp_side_effect(text):
            """Mock nlp side effect that returns different docs based on input."""
            mock_doc = MagicMock()

            if not text:  # Empty text
                mock_doc.__iter__ = lambda self: iter([])
                return mock_doc

            # Mock tokens for non-empty text
            mock_token1 = MagicMock()
            mock_token1.text = "example"
            mock_token1.lemma_ = "example"
            mock_token1.is_stop = False
            mock_token1.is_punct = False
            mock_token1.is_space = False

            mock_token2 = MagicMock()
            mock_token2.text = "text"
            mock_token2.lemma_ = "text"
            mock_token2.is_stop = False
            mock_token2.is_punct = False
            mock_token2.is_space = False

            mock_doc.__iter__ = lambda self: iter([mock_token1, mock_token2])
            return mock_doc

        mock_nlp.side_effect = mock_nlp_side_effect
        self.preprocessor = TextPreprocessor(mock_nlp)

    def test_preprocess_defaults(self):
        """Test preprocessing with default parameters."""
        result = self.preprocessor.preprocess("Example text")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result == ["example", "text"]

    def test_preprocess_no_stopwords(self):
        """Test preprocessing without removing stopwords."""
        result = self.preprocessor.preprocess("Example text", remove_stopwords=False)
        assert isinstance(result, list)

    def test_preprocess_no_lemmatize(self):
        """Test preprocessing without lemmatization."""
        result = self.preprocessor.preprocess("Example text", lemmatize=False)
        assert isinstance(result, list)

    def test_preprocess_no_processing(self):
        """Test preprocessing with no stopword removal or lemmatization."""
        result = self.preprocessor.preprocess("Example text", remove_stopwords=False, lemmatize=False)
        assert isinstance(result, list)

    def test_preprocess_empty(self):
        """Test preprocessing with empty text."""
        result = self.preprocessor.preprocess("")
        assert result == []

    def test_preprocess_text_function(self):
        """Test the global preprocess_text function."""
        # This would need the global preprocessor to be initialized
        # For now, just test that it exists
        assert callable(preprocess_text)


class TestMarkdownParser:
    """Test markdown parsing functionality."""

    def test_parse_markdown_sections_basic(self):
        """Test basic markdown section parsing."""
        markdown = """
# Section 1

Content for section 1.

## Subsection 1.1

Subsection content.

# Section 2

Content for section 2.
"""
        result = parse_markdown_sections(markdown.strip())

        assert isinstance(result, dict)
        assert "full_text" in result
        assert "# Section 1" in result
        assert "## Subsection 1.1" in result
        assert "# Section 2" in result

    def test_parse_markdown_no_sections(self):
        """Test parsing markdown with no headings."""
        text = "Just some text without headings."
        result = parse_markdown_sections(text)

        assert result["full_text"] == text
        assert len(result) == 2  # full_text and paragraphs

    def test_strip_markdown_markup(self):
        """Test stripping markdown markup."""
        markdown = "# Heading\n\n**Bold** text with *italic* words."
        result = strip_markdown_markup(markdown)

        # Should remove markdown formatting
        assert "#" not in result
        assert "**" not in result
        assert "*" not in result
        assert "Heading" in result
        assert "Bold" in result
        assert "italic" in result


class TestSentenceSplitter:
    """Test sentence splitting functionality."""

    def test_split_into_sentences_basic(self):
        """Test basic sentence splitting."""
        text = "This is sentence one. This is sentence two! And a third?"
        result = split_into_sentences(text)

        assert isinstance(result, list)
        assert len(result) == 3

    def test_split_into_sentences_empty(self):
        """Test sentence splitting with empty text."""
        result = split_into_sentences("")
        assert result == []

    def test_split_paragraphs_basic(self):
        """Test basic paragraph splitting."""
        text = "Paragraph one.\n\nParagraph two.\n\n\nParagraph three."
        result = split_paragraphs(text)

        expected = ["Paragraph one.", "Paragraph two.", "Paragraph three."]
        assert result == expected

    def test_split_paragraphs_empty(self):
        """Test paragraph splitting with empty text."""
        result = split_paragraphs("")
        assert result == []

    def test_split_paragraphs_single(self):
        """Test paragraph splitting with single paragraph."""
        text = "Single paragraph without line breaks."
        result = split_paragraphs(text)
        assert result == [text]


class TestSentenceFunctions:
    """Test the sentence splitting functions."""

    def test_split_into_sentences_basic(self):
        """Test basic sentence splitting."""
        text = "This is the first sentence. This is the second sentence."
        expected = ["This is the first sentence.", "This is the second sentence."]
        result = split_into_sentences(text)
        assert result == expected

    def test_split_into_sentences_complex(self):
        """Test sentence splitting with complex punctuation."""
        text = "Hello world! How are you? I'm fine. What about you?"
        expected = ["Hello world!", "How are you?", "I'm fine.", "What about you?"]
        result = split_into_sentences(text)
        assert result == expected

    def test_split_into_sentences_empty(self):
        """Test splitting empty text."""
        result = split_into_sentences("")
        assert result == []

    def test_split_into_sentences_single(self):
        """Test splitting single sentence."""
        text = "This is a single sentence."
        result = split_into_sentences(text)
        assert result == [text]


class TestStripMarkdown:
    """Test the strip_markdown_markup function."""

    def test_plain_text(self):
        """Test stripping plain text (no markdown)."""
        assert strip_markdown_markup("Hello world") == "Hello world"

    def test_empty_string(self):
        """Test stripping empty string."""
        assert strip_markdown_markup("") == ""

    def test_markdown_link(self):
        """Test stripping markdown links."""
        assert strip_markdown_markup("This is a [link text](http://example.com).") == "This is a link text."

    def test_bold_text(self):
        """Test stripping bold markdown."""
        assert strip_markdown_markup("This is **bold** text.") == "This is bold text."

    def test_italic_text(self):
        """Test stripping italic markdown."""
        assert strip_markdown_markup("This is *italic* text.") == "This is italic text."
