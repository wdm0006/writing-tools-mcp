from server.server import spellcheck

# Basic spellcheck tests


def test_spellcheck_correct():
    """Test spellcheck with correctly spelled words."""
    text = "this is a sentence with correct spelling"
    assert spellcheck(text) == []


def test_spellcheck_incorrect():
    """Test spellcheck with misspelled words."""
    text = "this is a sentense with incorrectt speling"
    # The exact output might depend on the dictionary used by pyspellchecker
    # We expect it to find the misspelled words
    result = spellcheck(text)
    assert "sentense" in result
    assert "incorrectt" in result
    assert "speling" in result


def test_spellcheck_mixed():
    """Test spellcheck with a mix of correct and incorrect words."""
    text = "correct speling hapens sometims"
    result = spellcheck(text)
    assert "speling" in result
    assert "hapens" in result
    assert "sometims" in result
    assert "correct" not in result


def test_spellcheck_empty():
    """Test spellcheck with empty input."""
    text = ""
    assert spellcheck(text) == []


def test_spellcheck_punctuation():
    """Test spellcheck ignores punctuation."""
    text = "hello, world! this. is; correctt?"
    result = spellcheck(text)
    assert "correctt" in result
    assert len(result) == 1  # Should only find 'correctt'


# Note: pyspellchecker might treat numbers or specific symbols differently.
# More advanced tests could cover those if needed.
