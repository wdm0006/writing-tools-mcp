from server.server import character_count, list_tools, word_count


def test_character_count():
    """Test character_count with various inputs."""
    assert character_count("hello") == 5
    assert character_count("hello world") == 11  # Includes space
    assert character_count("") == 0
    assert character_count("  ") == 2  # Counts spaces


def test_word_count():
    """Test word_count based on default string split behavior (splits on whitespace, ignores empty strings)."""
    assert word_count("hello world") == 2
    assert word_count("  hello   world  ") == 2  # Default split handles extra whitespace
    assert word_count("oneword") == 1
    assert word_count("") == 0  # Splitting an empty string results in an empty list
    assert word_count("   ") == 0  # Splitting a string with only spaces results in an empty list
    assert word_count("word\nword\tword") == 3  # Splits on various whitespace
