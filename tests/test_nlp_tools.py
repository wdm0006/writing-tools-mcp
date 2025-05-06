from server import (
    preprocess_text,
    keyword_density,
    keyword_frequency,
    top_keywords,
    keyword_context,
    passive_voice_detection,
)

# --- Test preprocess_text Helper ---

TEXT_FOR_PREPROCESSING = (
    "The quick brown foxes jumped over the lazy dogs. Running fast!"
)


def test_preprocess_text_defaults():
    """Test preprocess_text with default settings (remove stopwords, lemmatize)."""
    expected = ["quick", "brown", "fox", "jump", "lazy", "dog", "run", "fast"]
    assert preprocess_text(TEXT_FOR_PREPROCESSING) == expected


def test_preprocess_text_no_stopwords():
    """Test preprocess_text without removing stopwords (lemmatize=True)."""
    expected = [
        "the",
        "quick",
        "brown",
        "fox",
        "jump",
        "over",
        "the",
        "lazy",
        "dog",
        "run",
        "fast",
    ]
    assert (
        preprocess_text(TEXT_FOR_PREPROCESSING, remove_stopwords=False, lemmatize=True)
        == expected
    )


def test_preprocess_text_no_lemmatize():
    """Test preprocess_text without lemmatizing (remove_stopwords=True)."""
    # Note: 'jumped' might still be lemmatized depending on spaCy model details, focusing on 'running'
    expected = ["quick", "brown", "foxes", "jumped", "lazy", "dogs", "running", "fast"]
    result = preprocess_text(
        TEXT_FOR_PREPROCESSING, remove_stopwords=True, lemmatize=False
    )
    assert result == expected


def test_preprocess_text_no_processing():
    """Test preprocess_text with no stopword removal or lemmatization."""
    expected = [
        "the",
        "quick",
        "brown",
        "foxes",
        "jumped",
        "over",
        "the",
        "lazy",
        "dogs",
        "running",
        "fast",
    ]
    assert (
        preprocess_text(TEXT_FOR_PREPROCESSING, remove_stopwords=False, lemmatize=False)
        == expected
    )


def test_preprocess_text_empty():
    """Test preprocess_text with empty string."""
    assert preprocess_text("") == []


# --- Test Keyword Tools ---

TEXT_FOR_KEYWORDS = "Analyze this text. This text has keywords. Analyze keywords."
# Default preprocessed (lemmatized, stopwords removed):
# ['analyze', 'text', 'text', 'keyword', 'analyze', 'keyword']


def test_keyword_density():
    """Test keyword_density calculation."""
    # Density of 'text' (lemma: 'text'): 2 / 6 = 33.33%
    assert abs(keyword_density(TEXT_FOR_KEYWORDS, "text") - 33.33) < 0.01
    # Density of 'analyze' (lemma: 'analyze'): 2 / 6 = 33.33%
    assert abs(keyword_density(TEXT_FOR_KEYWORDS, "Analyze") - 33.33) < 0.01
    # Density of 'keywords' (lemma: 'keyword'): 2 / 6 = 33.33%
    assert abs(keyword_density(TEXT_FOR_KEYWORDS, "keywords") - 33.33) < 0.01
    # Density of non-existent word
    assert keyword_density(TEXT_FOR_KEYWORDS, "missing") == 0
    # Density in empty text
    assert keyword_density("", "text") == 0


def test_keyword_frequency_defaults():
    """Test keyword_frequency with default settings."""
    expected = {"analyze": 2, "text": 2, "keyword": 2}
    assert keyword_frequency(TEXT_FOR_KEYWORDS) == expected


def test_keyword_frequency_no_stopwords():
    """Test keyword_frequency keeping stopwords."""
    # Preprocessed: ['analyze', 'this', 'text', 'this', 'text', 'have', 'keyword', 'analyze', 'keyword']
    expected = {"analyze": 2, "this": 2, "text": 2, "have": 1, "keyword": 2}
    assert keyword_frequency(TEXT_FOR_KEYWORDS, remove_stopwords=False) == expected


def test_keyword_frequency_empty():
    """Test keyword_frequency with empty text."""
    assert keyword_frequency("") == {}


def test_top_keywords_defaults():
    """Test top_keywords with default settings."""
    # All have frequency 2, order might vary
    expected_keywords = {("analyze", 2), ("text", 2), ("keyword", 2)}
    result = top_keywords(TEXT_FOR_KEYWORDS, top_n=3)
    assert isinstance(result, list)
    assert len(result) == 3
    assert set(result) == expected_keywords


def test_top_keywords_more_complex():
    """Test top_keywords with more varied frequencies."""
    text = "one two two three three three four four four four"
    # Processed: ['one', 'two', 'two', 'three', 'three', 'three', 'four', 'four', 'four', 'four']
    expected = [("four", 4), ("three", 3), ("two", 2)]
    assert top_keywords(text, top_n=3, remove_stopwords=False) == expected
    # Test top_n limit
    assert top_keywords(text, top_n=1, remove_stopwords=False) == [("four", 4)]
    # Test requesting more than available
    result_all = top_keywords(text, top_n=5, remove_stopwords=False)
    assert len(result_all) == 4
    assert set(result_all) == set([("four", 4), ("three", 3), ("two", 2), ("one", 1)])


def test_top_keywords_no_stopwords():
    """Test top_keywords keeping stopwords."""
    text = "the test is the test and the test should pass"
    # Processed (no lem/no stop): ['the', 'test', 'is', 'the', 'test', 'and', 'the', 'test', 'should', 'pass']
    # Freq: the:3, test:3, is:1, and:1, should:1, pass:1
    result = top_keywords(text, top_n=2, remove_stopwords=False)
    expected_keywords = {("the", 3), ("test", 3)}
    assert set(result) == expected_keywords
    assert len(result) == 2


def test_keyword_context():
    """Test keyword_context extraction."""
    text = "The first sentence mentions the keyword. The second sentence does not. The third sentence also mentions the keyword."
    keyword = "keyword"
    expected = [
        "The first sentence mentions the keyword.",
        "The third sentence also mentions the keyword.",
    ]
    assert keyword_context(text, keyword) == expected


def test_keyword_context_case_insensitive():
    """Test keyword_context is case-insensitive and uses lemmas."""
    text = "Testing sentence. We are tests. This test is final."
    keyword = "Test"
    expected = [
        "Testing sentence.",  # Matches lemma 'test'
        "We are tests.",  # Matches lemma 'test'
        "This test is final.",  # Matches lemma 'test'
    ]
    assert keyword_context(text, keyword) == expected


def test_keyword_context_no_match():
    """Test keyword_context when the keyword is not found."""
    text = "This sentence is irrelevant."
    keyword = "missing"
    assert keyword_context(text, keyword) == []


def test_keyword_context_empty():
    """Test keyword_context with empty text or keyword."""
    assert keyword_context("", "keyword") == []
    assert keyword_context("Some text.", "") == []  # Empty keyword likely won't match


# --- Test Passive Voice Detection ---

ACTIVE_SENTENCE_1 = "The dog chased the ball."
PASSIVE_SENTENCE_1 = "The ball was chased by the dog."
ACTIVE_SENTENCE_2 = "The team completed the project."
PASSIVE_SENTENCE_2 = "The project was completed by the team."
MIXED_TEXT = (
    f"{ACTIVE_SENTENCE_1} {PASSIVE_SENTENCE_1} {ACTIVE_SENTENCE_2} {PASSIVE_SENTENCE_2}"
)


def test_passive_voice_detection_passive():
    """Test passive_voice_detection identifies simple passive sentences."""
    assert passive_voice_detection(PASSIVE_SENTENCE_1) == [PASSIVE_SENTENCE_1]
    assert passive_voice_detection(PASSIVE_SENTENCE_2) == [PASSIVE_SENTENCE_2]


def test_passive_voice_detection_active():
    """Test passive_voice_detection does not flag active sentences."""
    assert passive_voice_detection(ACTIVE_SENTENCE_1) == []
    assert passive_voice_detection(ACTIVE_SENTENCE_2) == []


def test_passive_voice_detection_mixed():
    """Test passive_voice_detection with mixed active and passive sentences."""
    expected = [PASSIVE_SENTENCE_1, PASSIVE_SENTENCE_2]
    result = passive_voice_detection(MIXED_TEXT)
    # Order might vary depending on sentence splitting
    assert len(result) == 2
    assert set(result) == set(expected)


def test_passive_voice_detection_empty():
    """Test passive_voice_detection with empty text."""
    assert passive_voice_detection("") == []


# Note: The passive voice detection is simple and might have false positives/negatives.
# More complex linguistic structures could be added for testing if needed.
