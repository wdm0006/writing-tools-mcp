"""
Core stylometric feature extraction module.

This module provides the StylemetricAnalyzer class for extracting
various stylometric features from text that can be used to identify
AI-generated content.
"""

import math
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional

import textstat
from wordfreq import zipf_frequency

#: Universal Dependencies labels that mark a token as heading a subordinate clause.
#: Used as a cheap, dependency-parse-only proxy for T-unit-style clause density.
SUBORDINATE_CLAUSE_DEPS = {"advcl", "ccomp", "acl", "relcl", "xcomp", "csubj", "csubjpass"}

#: spaCy POS tags conventionally treated as "content words" for lexical density
#: (content words / total words), as opposed to function words.
CONTENT_POS_TAGS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}

#: MTLD's standard TTR-drop threshold (McCarthy & Jarvis 2010).
MTLD_THRESHOLD = 0.72

#: MATTR's sliding window size in tokens (Covington & McFall 2010). Below this many
#: tokens there's no window to slide, so MATTR falls back to None.
MATTR_WINDOW = 50

#: Character n-gram size used for the PAN/CLEF-style orthographic profile.
CHAR_NGRAM_SIZE = 4

#: Hand-built hedge/booster wordlists (no open, non-LIWC dictionary exists for these
#: categories the way Empath substitutes for LIWC generally - see Pennebaker's work on
#: function words and epistemic-marker research on hedges/boosters). Deliberately small
#: and conservative rather than exhaustive.
HEDGE_WORDS = {
    "might",
    "may",
    "could",
    "perhaps",
    "possibly",
    "seem",
    "seems",
    "seemed",
    "appears",
    "appeared",
    "likely",
    "somewhat",
    "generally",
    "often",
    "usually",
    "suggest",
    "suggests",
    "tend",
    "tends",
    "arguably",
    "presumably",
    "apparently",
    "probably",
    "roughly",
    "relatively",
}

BOOSTER_WORDS = {
    "definitely",
    "certainly",
    "always",
    "never",
    "clearly",
    "obviously",
    "undoubtedly",
    "absolutely",
    "surely",
    "indeed",
    "must",
    "will",
    "guaranteed",
    "unquestionably",
    "inevitably",
    "entirely",
    "completely",
    "totally",
}


class StylemetricAnalyzer:
    """Analyzer for extracting stylometric features from text."""

    def __init__(self, nlp_model):
        """Initialize with a spaCy NLP model."""
        self.nlp = nlp_model

        # Common function words for analysis
        self.function_words = {
            "the",
            "of",
            "and",
            "a",
            "to",
            "in",
            "is",
            "you",
            "that",
            "it",
            "he",
            "was",
            "for",
            "on",
            "are",
            "as",
            "with",
            "his",
            "they",
            "i",
            "at",
            "be",
            "this",
            "have",
            "from",
            "or",
            "one",
            "had",
            "by",
            "word",
            "but",
            "not",
            "what",
            "all",
            "were",
            "we",
            "when",
            "your",
            "can",
            "said",
            "there",
            "each",
            "which",
            "she",
            "do",
            "how",
            "their",
            "if",
            "will",
            "up",
            "other",
            "about",
            "out",
            "many",
            "then",
            "them",
            "these",
            "so",
            "some",
            "her",
            "would",
            "make",
            "like",
            "into",
            "him",
            "has",
            "two",
            "more",
            "very",
            "after",
            "my",
            "than",
            "first",
            "been",
            "who",
            "its",
            "now",
            "people",
            "may",
            "down",
            "day",
            "get",
            "use",
            "man",
            "new",
            "way",
            "could",
            "does",
            "only",
            "where",
            "most",
            "over",
            "think",
            "also",
            "back",
            "work",
            "life",
            "why",
            "go",
            "should",
            "even",
        }

    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract all stylometric features from text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary containing all extracted stylometric features
        """
        if not text.strip():
            return self._empty_features()

        doc = self.nlp(text)
        sentences = list(doc.sents)

        if not sentences:
            return self._empty_features()

        return {
            # Sentence-level features
            "avg_sentence_len": self._avg_sentence_length(sentences),
            "sentence_len_std": self._sentence_length_std(sentences),
            "sentence_positions": self._sentence_positions(sentences),
            # Lexical diversity
            "ttr": self._type_token_ratio(doc),
            "hapax_legomena_rate": self._hapax_rate(doc),
            "avg_word_len": self._avg_word_length(doc),
            # Length-robust lexical diversity (see MTLD_THRESHOLD/MATTR_WINDOW):
            # unlike ttr/hapax_legomena_rate, these are designed not to fall
            # monotonically as a document gets longer.
            "mtld": self._mtld(doc),
            "mattr": self._mattr(doc),
            "mtld_lemma": self._mtld(doc, use_lemmas=True),
            # Vocabulary sophistication/rarity: how common the words used are, as
            # opposed to how many distinct words are used (diversity, above).
            "mean_word_frequency": self._mean_word_frequency(doc),
            "word_len_std": self._word_length_std(doc),
            "lexical_density": self._lexical_density(doc),
            # POS ratios
            "pos_ratios": self._pos_ratios(doc),
            "pos_bigram_ratios": self._pos_bigram_ratios(doc),
            # Punctuation patterns
            "punct_density": self._punctuation_density(text),
            "comma_ratio": self._comma_ratio(text),
            "semicolon_ratio": self._char_ratio_of_punct(text, ";"),
            "em_dash_ratio": self._char_ratio_of_punct(text, "—"),
            "ellipsis_ratio": self._ellipsis_ratio(text),
            "exclamation_ratio": self._char_ratio_of_punct(text, "!"),
            "parenthetical_rate": self._parenthetical_rate(text, sentences),
            # Additional features
            "function_word_ratio": self._function_word_ratio(doc),
            "function_word_freqs": self._function_word_freqs(doc),
            "hedge_rate": self._wordlist_rate(doc, HEDGE_WORDS),
            "booster_rate": self._wordlist_rate(doc, BOOSTER_WORDS),
            # Character-level orthographic profile (PAN/CLEF-style). Compared against
            # a baseline via cosine similarity, not mean/std z-scoring - see
            # calculate_char_ngram_similarity.
            "char_ngram_profile": self._char_ngram_profile(text),
            # Readability grade levels. Unlike TTR/hapax rate, these are not
            # length-confounded at the document lengths this tool sees in practice:
            # they're averages over per-sentence syllable/word counts, not a count of
            # distinct types out of a shrinking-with-length total.
            "fog": self._reading_grade(text, textstat.gunning_fog),
            "kincaid": self._reading_grade(text, textstat.flesch_kincaid_grade),
            "smog": self._reading_grade(text, textstat.smog_index),
            "coleman_liau": self._reading_grade(text, textstat.coleman_liau_index),
            "ari": self._reading_grade(text, textstat.automated_readability_index),
            "dale_chall": self._reading_grade(text, textstat.dale_chall_readability_score),
            # Syntactic complexity, read directly off the dependency parse.
            "mean_dependency_distance": self._mean_dependency_distance(sentences),
            "subordinate_clause_ratio": self._subordinate_clause_ratio(doc, sentences),
            # Repetition / word-frequency shape.
            "fourgram_repetition_rate": self._fourgram_repetition_rate(doc),
            "zipf_slope": self._zipf_slope(doc),
        }

    def _empty_features(self) -> Dict[str, Any]:
        """Return empty/default feature values for empty text."""
        return {
            "avg_sentence_len": 0.0,
            "sentence_len_std": None,
            "sentence_positions": [],
            "ttr": 0.0,
            "hapax_legomena_rate": 0.0,
            "avg_word_len": 0.0,
            "mtld": None,
            "mattr": None,
            "mtld_lemma": None,
            "mean_word_frequency": None,
            "word_len_std": None,
            "lexical_density": 0.0,
            "pos_ratios": {},
            "pos_bigram_ratios": {},
            "punct_density": 0.0,
            "comma_ratio": 0.0,
            "semicolon_ratio": 0.0,
            "em_dash_ratio": 0.0,
            "ellipsis_ratio": 0.0,
            "exclamation_ratio": 0.0,
            "parenthetical_rate": None,
            "function_word_ratio": 0.0,
            "function_word_freqs": {},
            "hedge_rate": 0.0,
            "booster_rate": 0.0,
            "char_ngram_profile": {},
            "fog": None,
            "kincaid": None,
            "smog": None,
            "coleman_liau": None,
            "ari": None,
            "dale_chall": None,
            "mean_dependency_distance": None,
            "subordinate_clause_ratio": None,
            "fourgram_repetition_rate": None,
            "zipf_slope": None,
        }

    def _avg_sentence_length(self, sentences) -> float:
        """Calculate average sentence length in words."""
        if not sentences:
            return 0.0

        lengths = []
        for sent in sentences:
            # Count non-punctuation, non-space tokens
            word_count = sum(1 for token in sent if not token.is_punct and not token.is_space)
            lengths.append(word_count)

        return statistics.mean(lengths) if lengths else 0.0

    def _sentence_length_std(self, sentences) -> Optional[float]:
        """Calculate standard deviation of sentence lengths, or None if undefined."""
        if len(sentences) < 2:
            return None

        lengths = []
        for sent in sentences:
            word_count = sum(1 for token in sent if not token.is_punct and not token.is_space)
            lengths.append(word_count)

        return statistics.stdev(lengths) if len(lengths) > 1 else None

    def _sentence_positions(self, sentences) -> List[Dict[str, Any]]:
        """Extract detailed information about each sentence."""
        positions = []

        for i, sent in enumerate(sentences):
            word_count = sum(1 for token in sent if not token.is_punct and not token.is_space)
            positions.append(
                {
                    "position": i + 1,
                    "length": word_count,
                    "text": sent.text.strip()[:100] + "..." if len(sent.text) > 100 else sent.text.strip(),
                }
            )

        return positions

    def _type_token_ratio(self, doc) -> float:
        """Calculate Type-Token Ratio (lexical diversity)."""
        # Get all words (excluding punctuation and spaces)
        words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

        if not words:
            return 0.0

        unique_words = set(words)
        return len(unique_words) / len(words)

    def _hapax_rate(self, doc) -> float:
        """Calculate Hapax Legomena rate (words appearing only once)."""
        # Get all words (excluding punctuation and spaces)
        words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

        if not words:
            return 0.0

        word_counts = Counter(words)
        hapax_words = sum(1 for count in word_counts.values() if count == 1)

        return hapax_words / len(word_counts) if word_counts else 0.0

    def _avg_word_length(self, doc) -> float:
        """Calculate average word length in characters."""
        words = [token.text for token in doc if not token.is_punct and not token.is_space]

        if not words:
            return 0.0

        return statistics.mean(len(word) for word in words)

    def _reading_grade(self, text: str, scorer) -> Optional[float]:
        """Run a textstat grade-level scorer, or None when the text is too short to score.

        SMOG in particular is only statistically validated on texts of 30+ sentences;
        textstat computes a value below that anyway, so treat it as less reliable on
        typical blog-post-length input rather than gating it out entirely here.
        """
        if not text or len(text.split()) < 3:
            return None

        return scorer(text)

    def _mtld_single_pass(self, words: List[str], threshold: float = MTLD_THRESHOLD) -> float:
        """One directional pass of MTLD: count how many times TTR drops to `threshold`."""
        factors = 0.0
        types: set = set()
        token_count = 0

        for word in words:
            token_count += 1
            types.add(word)
            ttr = len(types) / token_count

            if ttr <= threshold:
                factors += 1
                types = set()
                token_count = 0

        # The remainder after the last full factor counts as a fractional factor,
        # proportional to how far its TTR fell short of a full drop to `threshold`.
        if token_count > 0:
            ttr = len(types) / token_count
            factors += (1 - ttr) / (1 - threshold)

        return len(words) / factors if factors > 0 else float(len(words))

    def _mtld(self, doc, use_lemmas: bool = False) -> Optional[float]:
        """Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010).

        Unlike TTR, MTLD is designed to be stable across document lengths: it
        averages a forward and backward pass over the token stream rather than a
        single type-count-over-token-count ratio. Below ~50 tokens the measure is
        unreliable (too few, if any, threshold drops), so this returns None there.

        With `use_lemmas=True`, this measures diversity of root words rather than
        surface forms - i.e. whether an author favors many distinct roots or many
        inflections of few roots - without reintroducing the length confound a raw
        lemma-based TTR would have (same algorithm, same length-robustness).
        """
        if use_lemmas:
            words = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
        else:
            words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

        if len(words) < 50:
            return None

        forward = self._mtld_single_pass(words)
        backward = self._mtld_single_pass(list(reversed(words)))
        return (forward + backward) / 2

    def _mattr(self, doc, window: int = MATTR_WINDOW) -> Optional[float]:
        """Moving-Average Type-Token Ratio (Covington & McFall 2010).

        TTR computed over a fixed-size sliding window and averaged across every
        window position, which keeps it from falling as the document grows longer
        the way plain TTR does. None when there are fewer tokens than one window.
        """
        words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

        if len(words) < window:
            return None

        window_ttrs = [len(set(words[start : start + window])) / window for start in range(len(words) - window + 1)]
        return statistics.mean(window_ttrs)

    def _mean_word_frequency(self, doc) -> Optional[float]:
        """Mean Zipf frequency (log10 words-per-billion) of the vocabulary used.

        Distinct from lexical *diversity* (mtld/mattr, how many different words):
        this measures how *common* those words are, via the `wordfreq` reference
        corpus - low values mean rarer, more sophisticated vocabulary; high values
        mean plainer, more common vocabulary. An unrecognized word scores 0.0 (the
        rarest possible), which is the correct treatment for a genuinely rare or
        invented word.
        """
        words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

        if not words:
            return None

        return statistics.mean(zipf_frequency(word, "en") for word in words)

    def _word_length_std(self, doc) -> Optional[float]:
        """Standard deviation of word length in characters - the shape of the word-length
        distribution, not just its mean (`avg_word_len`)."""
        lengths = [len(token.text) for token in doc if not token.is_punct and not token.is_space]

        if len(lengths) < 2:
            return None

        return statistics.stdev(lengths)

    def _lexical_density(self, doc) -> float:
        """Content words (noun/proper noun/verb/adj/adv) as a fraction of all words.

        Distinct from `function_word_ratio`: that's a fixed closed-class wordlist
        lookup, this is a POS-tag-based open-class/closed-class split, so the two
        need not move together.
        """
        words = [token for token in doc if not token.is_punct and not token.is_space]

        if not words:
            return 0.0

        content_count = sum(1 for token in words if token.pos_ in CONTENT_POS_TAGS)
        return content_count / len(words)

    def _pos_ratios(self, doc) -> Dict[str, float]:
        """Calculate part-of-speech tag ratios."""
        # Count POS tags for non-punctuation, non-space tokens
        pos_counts = Counter(token.pos_ for token in doc if not token.is_punct and not token.is_space)
        total_tokens = sum(pos_counts.values())

        if total_tokens == 0:
            return {}

        # Calculate ratios for major POS categories
        pos_ratios = {}
        for pos_tag, count in pos_counts.items():
            pos_ratios[pos_tag] = count / total_tokens

        return pos_ratios

    def _pos_bigram_ratios(self, doc) -> Dict[str, float]:
        """Ratios of consecutive POS-tag pairs (e.g. "DET_NOUN"), within each sentence.

        A syntactic n-gram profile: published authorship-attribution work reports
        POS bigrams/trigrams discriminating authors substantially better than
        single-tag POS ratios alone. Bigrams don't cross sentence boundaries, since
        a sentence-final-to-next-sentence-initial pair isn't a real syntactic
        adjacency.
        """
        bigram_counts: Counter = Counter()
        total = 0

        for sent in doc.sents:
            tags = [token.pos_ for token in sent if not token.is_punct and not token.is_space]
            for first, second in zip(tags, tags[1:], strict=False):
                bigram_counts[f"{first}_{second}"] += 1
                total += 1

        if total == 0:
            return {}

        return {bigram: count / total for bigram, count in bigram_counts.items()}

    def _punctuation_density(self, text: str) -> float:
        """Calculate punctuation density as punctuation marks per word.

        Unit: count(punctuation marks in text) / count(words in text).
        A "word" is a non-punctuation, non-whitespace token from a spaCy parse.

        The previous implementation divided by ``len(text)`` (per-character),
        which was dimensionally inconsistent with the static baseline constant
        of 0.14. Real English prose has ~0.165 punctuation marks per *word*
        (and ~0.027 per *character*); the per-character rate against a 0.14
        baseline z-scored every human document as a >3-sigma outlier and
        carried no information. The per-word rate against the same baseline
        z-scores real prose at +0.83, which is what a baseline should look
        like. See PR/issue for the measurement that surfaced the mismatch.
        """
        if not text:
            return 0.0

        doc = self.nlp(text)
        punct_count = sum(1 for tok in doc if tok.is_punct)
        word_count = sum(1 for tok in doc if not tok.is_punct and not tok.is_space)
        if word_count == 0:
            return 0.0
        return punct_count / word_count

    def _comma_ratio(self, text: str) -> float:
        """Calculate comma usage ratio (commas / total punctuation)."""
        if not text:
            return 0.0

        comma_count = text.count(",")
        total_punct = sum(1 for char in text if char in ".,;:!?()[]{}\"'-")

        return comma_count / total_punct if total_punct > 0 else 0.0

    def _char_ratio_of_punct(self, text: str, char: str) -> float:
        """Ratio of a single punctuation character to all punctuation - same shape as
        `_comma_ratio`, generalized to any one mark (semicolon, em dash, exclamation)."""
        if not text:
            return 0.0

        char_count = text.count(char)
        total_punct = sum(1 for c in text if c in ".,;:!?()[]{}\"'-—")

        return char_count / total_punct if total_punct > 0 else 0.0

    def _ellipsis_ratio(self, text: str) -> float:
        """Ratio of ellipses (literal "..." or the single-character "…") to all punctuation."""
        if not text:
            return 0.0

        ellipsis_count = text.count("…") + text.count("...")
        total_punct = sum(1 for c in text if c in ".,;:!?()[]{}\"'-—")

        return ellipsis_count / total_punct if total_punct > 0 else 0.0

    def _parenthetical_rate(self, text: str, sentences) -> Optional[float]:
        """Parenthetical asides per sentence - real forensic-linguistics idiolect marker
        distinct from raw punctuation density (documented cases have turned on
        punctuation-mark habits like this one)."""
        if not sentences:
            return None

        return text.count("(") / len(sentences)

    def _function_word_ratio(self, doc) -> float:
        """Calculate ratio of function words to total words."""
        words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

        if not words:
            return 0.0

        function_word_count = sum(1 for word in words if word in self.function_words)
        return function_word_count / len(words)

    def _function_word_freqs(self, doc) -> Dict[str, float]:
        """Per-word relative frequency of each tracked function word.

        This is the Burrows'-Delta style feature: instead of one aggregate function-word
        ratio, keep each word's own frequency so it can be z-scored individually against
        a baseline and reduced to a single "how far off is this author's function-word
        profile" distance (see `calculate_z_scores`'s `burrows_delta`). A word absent from
        this document is a real 0.0, not a missing measurement - every tracked word gets
        an entry so baselines built across a corpus see a consistent set of keys.
        """
        words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

        if not words:
            return dict.fromkeys(self.function_words, 0.0)

        counts = Counter(words)
        total = len(words)
        return {word: counts.get(word, 0) / total for word in self.function_words}

    def _wordlist_rate(self, doc, wordlist: set) -> float:
        """Fraction of words belonging to a fixed wordlist (used for hedge/booster rate)."""
        words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

        if not words:
            return 0.0

        return sum(1 for word in words if word in wordlist) / len(words)

    def _char_ngram_profile(self, text: str, n: int = CHAR_NGRAM_SIZE) -> Dict[str, float]:
        """Relative-frequency profile of character n-grams (PAN/CLEF-style orthographic
        fingerprint), normalized within this document.

        Compared against a baseline's aggregate profile via cosine similarity (see
        `calculate_char_ngram_similarity`), not mean/std z-scoring - individual n-grams
        are too sparse per document for that (most 4-grams occur 0-2 times in an
        800-word post), but the whole-profile shape is stable enough to compare.
        """
        compact = "".join(text.split())

        if len(compact) < n:
            return {}

        ngrams = [compact[i : i + n] for i in range(len(compact) - n + 1)]
        total = len(ngrams)
        counts = Counter(ngrams)

        return {ngram: count / total for ngram, count in counts.items()}

    def _mean_dependency_distance(self, sentences) -> Optional[float]:
        """Mean linear distance (in tokens) between each word and its syntactic head.

        A cheap syntactic-complexity signal read directly off the dependency parse
        spaCy already computes - no constituency parser required. Higher values mean
        more long-distance, harder-to-process dependencies (e.g. heavy embedding);
        lower values mean shorter, more local attachments.
        """
        distances = []

        for sent in sentences:
            for token in sent:
                if token.is_punct or token.is_space or token.head is token:
                    continue
                distances.append(abs(token.i - token.head.i))

        return statistics.mean(distances) if distances else None

    def _subordinate_clause_ratio(self, doc, sentences) -> Optional[float]:
        """Subordinate clauses per sentence, via Universal Dependencies labels.

        A cheap proxy for the clauses-per-T-unit measures used in L2 syntactic
        complexity research (e.g. Lu's L2SCA), without needing that tool's
        constituency-parser pipeline: SUBORDINATE_CLAUSE_DEPS covers the UD labels
        that mark a token as heading an embedded clause.
        """
        if not sentences:
            return None

        subordinate_count = sum(1 for token in doc if token.dep_ in SUBORDINATE_CLAUSE_DEPS)
        return subordinate_count / len(sentences)

    def _fourgram_repetition_rate(self, doc, n: int = 4) -> Optional[float]:
        """Fraction of word n-grams that repeat an earlier occurrence in this document.

        Self-repetition at the n-gram level is documented in AI-text-detection
        literature as a human/AI discriminator, distinct from general stylometry.
        None when the document is too short to form more than one n-gram.
        """
        words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

        if len(words) < n + 1:
            return None

        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        repeated = sum(count - 1 for count in counts.values() if count > 1)

        return repeated / len(ngrams)

    def _zipf_slope(self, doc) -> Optional[float]:
        """Slope of the log(rank) vs. log(frequency) line for this document's words.

        Natural-language word frequencies approximate Zipf's law (slope near -1);
        the slope itself is used in AI-text-detection literature as a discriminator.
        None when there are too few distinct words to fit a meaningful line.
        """
        words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

        if len(words) < 20:
            return None

        frequencies = sorted(Counter(words).values(), reverse=True)
        if len(frequencies) < 2:
            return None

        log_ranks = [math.log(rank) for rank in range(1, len(frequencies) + 1)]
        log_freqs = [math.log(freq) for freq in frequencies]
        return self._ols_slope(log_ranks, log_freqs)

    def _ols_slope(self, xs: List[float], ys: List[float]) -> float:
        """Ordinary-least-squares slope of y on x, with no external dependency."""
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
        variance = sum((x - mean_x) ** 2 for x in xs)

        return covariance / variance if variance > 0 else 0.0
