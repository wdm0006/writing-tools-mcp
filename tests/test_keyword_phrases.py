"""Value-based tests for keyword and phrase matching."""

import pytest


@pytest.mark.parametrize(
    ("text", "keyword", "expected_density"),
    [
        (
            "Artificial intelligence changes work. Artificial methods differ. "
            "Artificial intelligence improves systems.",
            "artificial intelligence",
            200 / 11,
        ),
        ("Artificial methods improve intelligence. Artificial intelligence works.", "artificial intelligence", 100 / 7),
        ("Running teams run daily.", "RUN", 50.0),
        ("Models run quickly. A model runs daily.", "model run", 100 / 3),
        ("Words remain.", "", 0),
        ("Words remain.", "...", 0),
    ],
)
def test_keyword_density_exact_values(keyword_analyzer, text, keyword, expected_density):
    assert keyword_analyzer.keyword_density(text, keyword) == pytest.approx(expected_density)


@pytest.mark.parametrize("keyword", ["", "..."])
def test_empty_keyword_has_no_context(keyword_analyzer, keyword):
    assert keyword_analyzer.keyword_context("Words remain. More words follow.", keyword) == []


def test_keyword_frequency_counts_lemmas_without_stopwords(keyword_analyzer):
    """Exact lemma counts on a mixed stopword/non-stopword document."""
    text = "The cat sat. The cat ran home."

    assert keyword_analyzer.keyword_frequency(text) == {"cat": 2, "sit": 1, "run": 1, "home": 1}


def test_keyword_frequency_keeps_stopwords_when_asked(keyword_analyzer):
    """remove_stopwords=False retains function words and their counts."""
    text = "The cat sat. The cat ran home."

    assert keyword_analyzer.keyword_frequency(text, remove_stopwords=False) == {
        "the": 2,
        "cat": 2,
        "sit": 1,
        "run": 1,
        "home": 1,
    }


def test_empty_text_yields_no_frequencies(keyword_analyzer):
    assert keyword_analyzer.keyword_frequency("") == {}


def test_keyword_at_the_very_end_counts(keyword_analyzer):
    """The counting window must include a phrase ending at the last token."""
    assert keyword_analyzer.keyword_density("Run fast then run", "run") == pytest.approx(200 / 3)


def test_keyword_context_matches_only_complete_phrase(keyword_analyzer):
    text = "Artificial methods differ. Artificial methods improve intelligence. Artificial intelligence changes work."

    assert keyword_analyzer.keyword_context(text, "artificial intelligence") == [
        "Artificial intelligence changes work."
    ]


def test_keyword_context_is_case_insensitive_and_lemmatized(keyword_analyzer):
    text = "Models run quickly. A model rests. Different systems RUN daily."

    assert keyword_analyzer.keyword_context(text, "MODEL RUN") == ["Models run quickly."]
