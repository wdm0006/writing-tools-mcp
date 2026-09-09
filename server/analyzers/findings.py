"""Uniform actionable findings for the analysis tools.

This module is the single source of finding generation: the dict-shaped
analysis tools enrich their SUCCESS responses with a ``findings`` array built
here, and nowhere else. Enrichment is additive — existing response keys are
never removed or renamed — and error paths keep their exact existing shape
(they gain nothing).

A finding is one located, fixable observation about a text. ``fix_hint``
strings coach the fix ("split this 61-word sentence into two"), never restate
the flaw ("sentence too long").

Per-baseline honesty is structural: the stylometry builders read only
``z_scores`` and the detector ``flags``, and ``generate_flags`` can only fire
an indicator when its z-score exists — so a baseline that lacks a statistic
(the 9-statistic ``brown_corpus`` default, say) can never produce a finding
for the indicators that depend on it.
"""

from typing import Any, TypedDict

__all__ = [
    "Finding",
    "from_keyword_context",
    "from_keyword_density",
    "from_keyword_frequency",
    "from_perplexity",
    "from_readability",
    "from_readability_result",
    "from_stylometry",
    "from_top_keywords",
    "order_by_impact",
]


class Finding(TypedDict):
    """One located, fixable observation about a text."""

    rule: str  # stable machine name, e.g. "low_flesch", "uniform_sentences"
    location: str  # best effort: "full_text", "section:<name>", "paragraph:<n>", "s<n>", "document"
    message: str  # human-readable, grounded in the measured value
    fix_hint: str  # concrete instruction coaching the fix, not the flaw


# --- rule thresholds (each pinned by a boundary test one step either side) ---

FLESCH_MIN = 50.0  # below this, prose reads as "difficult" for general readers
KINCAID_MAX = 12.0  # above this, grade level exceeds high school
FOG_MAX = 12.0  # above this, Gunning Fog exceeds high-school level
KEYWORD_DENSITY_MAX = 5.0  # percent of words; above this reads as keyword stuffing
WORD_SHARE_MAX = 0.10  # share of counted words; above this a lemma is overused
TOP_KEYWORD_DOMINANCE = 0.25  # top keyword's share of top-keyword occurrences
MAX_OVERUSED_FINDINGS = 3  # bound the per-word findings list


def from_readability(scores: dict[str, Any], location: str = "full_text") -> list[Finding]:
    """Build findings from one ``{flesch, kincaid, fog}`` score leaf.

    Boundaries are exclusive on the healthy side: a score exactly at a cutoff
    produces no finding, one step past it does. ``None`` scores (segments too
    short to score) never fire.
    """
    findings: list[Finding] = []

    flesch = scores.get("flesch")
    if flesch is not None and flesch < FLESCH_MIN:
        findings.append(
            Finding(
                rule="low_flesch",
                location=location,
                message=f"Flesch Reading Ease {flesch:.1f} is below the {FLESCH_MIN:.0f} 'difficult' line for general readers",
                fix_hint=(
                    "split long sentences at their conjunctions and replace noun stacks with plain verbs "
                    "so a reader gets each point in one pass"
                ),
            )
        )

    kincaid = scores.get("kincaid")
    if kincaid is not None and kincaid > KINCAID_MAX:
        findings.append(
            Finding(
                rule="high_kincaid",
                location=location,
                message=f"Flesch-Kincaid grade level {kincaid:.1f} is above grade {KINCAID_MAX:.0f}",
                fix_hint=(
                    "break compound sentences into two and prefer everyday words "
                    f"to bring the grade level to {KINCAID_MAX:.0f} or under"
                ),
            )
        )

    fog = scores.get("fog")
    if fog is not None and fog > FOG_MAX:
        findings.append(
            Finding(
                rule="high_fog",
                location=location,
                message=f"Gunning Fog index {fog:.1f} is above the {FOG_MAX:.0f} high-school line",
                fix_hint=(
                    "shorten sentences and cut or define jargon — each syllable-heavy clause pushes the fog index up"
                ),
            )
        )

    return findings


def from_readability_result(result: dict[str, Any]) -> list[Finding]:
    """Build findings for a whole ``readability_score`` response.

    Dispatches on the level envelope: a bare scores dict is the ``full``
    level; ``section`` and ``paragraph`` responses are walked leaf by leaf so
    every scored region can earn a located finding. Error envelopes (which
    carry an ``error`` key) are never passed here.
    """
    if "sections" in result or "paragraphs" in result:
        findings = from_readability(result.get("full_text", {}), "full_text")
        for name, scores in result.get("sections", {}).items():
            findings.extend(from_readability(scores, f"section:{name}"))
        for paragraph in result.get("paragraphs", []):
            location = f"paragraph:{paragraph.get('paragraph_number')}"
            findings.extend(from_readability(paragraph.get("scores", {}), location))
        return findings
    return from_readability(result, "full_text")


def from_perplexity(result: dict[str, Any], location: str = "document") -> list[Finding]:
    """Build findings from a successful ``perplexity_analysis`` response.

    The document-level finding fires only on the detector's own verdict
    (``high_ai_probability``) — ambiguous reasons without the flag are not
    findings. When the flag is on, per-sentence findings locate the most
    predictable sentences so a revision has concrete targets.

    ``location`` re-anchors the findings when the same response shape is built
    for a region inside a larger document (e.g. ``section:## Notes``); with
    the default it reproduces the standalone tool's locations exactly.
    """
    flags = result.get("flags", {})
    if not flags.get("high_ai_probability"):
        return []

    thresholds = result.get("config", {}).get("thresholds", {})
    ppl_max = thresholds.get("ppl_max", 25.0)

    findings: list[Finding] = [
        Finding(
            rule="high_ai_probability",
            location=location,
            message=(
                f"GPT-2 perplexity {result.get('doc_ppl')} with burstiness {result.get('doc_burstiness')}: "
                "the document reads as statistically predictable and uniformly so"
            ),
            fix_hint=(
                "revise for human cadence — vary sentence lengths, lead with concrete specifics, "
                "and allow some irregularity — then re-run perplexity_analysis to verify the flag clears"
            ),
        )
    ]

    for index, sentence in enumerate(result.get("sentences", [])):
        ppl = sentence.get("ppl") if isinstance(sentence, dict) else None
        if ppl is not None and ppl < ppl_max:
            findings.append(
                Finding(
                    rule="low_sentence_perplexity",
                    # Default keeps the standalone tool's bare "s<n>" locations;
                    # a re-anchored region prefixes them so findings from
                    # different regions stay distinguishable when flattened.
                    location=f"s{index}" if location == "document" else f"{location} s{index}",
                    message=f"sentence perplexity {ppl} is below the {ppl_max} predictability line",
                    fix_hint=(
                        "rewrite this sentence with more varied vocabulary and a concrete detail "
                        "so its wording is less statistically predictable"
                    ),
                )
            )
    return findings


# The stylometry AI indicators: underlying z-score keys, a plain description,
# and the fix each one calls for. An indicator can only fire when its z-score
# exists in the response, and z_scores are baseline-relative — so this table
# inherits the detector's per-baseline honesty: a baseline lacking a statistic
# can never produce the indicator's finding.
_INDICATOR_PROFILE: dict[str, tuple[tuple[str, ...], str, str]] = {
    "low_ttr": (
        ("ttr",),
        "vocabulary diversity (TTR) is unusually low",
        "repeat ideas with fresh wording and trade a few repeated terms for precise synonyms",
    ),
    "low_hapax": (
        ("hapax_legomena_rate",),
        "the proportion of once-used words (hapax legomena) is unusually low",
        "introduce new vocabulary instead of recycling the same words — each new noun or verb adds a hapax",
    ),
    "uniform_sentences": (
        ("sentence_len_std",),
        "sentence lengths are unusually uniform",
        "break the rhythm: split one long sentence and merge two short ones so lengths vary",
    ),
    "unusual_sentence_length": (
        ("avg_sentence_len",),
        "average sentence length is unusual for this baseline",
        "rebalance toward the baseline register — split the longest sentences and combine fragments",
    ),
    "pos_anomalies": (
        (),
        "part-of-speech patterns deviate from the baseline",
        "rework sentences built on the same clause pattern — vary how phrases attach to the verb",
    ),
    "function_word_anomaly": (
        ("function_word_ratio",),
        "function-word usage (articles, prepositions, conjunctions) is unusual",
        (
            "swap some connective-heavy phrasing for direct verbs, or the reverse, "
            "to move function-word usage back toward the baseline"
        ),
    ),
    "unusual_reading_level": (
        ("fog",),
        "reading grade level is unusual for this baseline",
        "adjust wording complexity toward the audience — replace jargon with plain terms or add needed precision",
    ),
    "low_mtld": (
        ("mtld",),
        "length-robust lexical diversity (MTLD) is unusually low",
        "open a new subtopic with fresh vocabulary; MTLD rewards stretches of text that do not reuse words",
    ),
    "distinct_function_word_profile": (
        ("burrows_delta",),
        "the per-function-word profile differs sharply from the baseline",
        "rewrite the most formulaic transitions — your connective habits differ sharply from the baseline",
    ),
    "unusual_vocabulary_rarity": (
        ("mean_word_frequency",),
        "word commonness (mean Zipf frequency) is unusual",
        (
            "shift word choice toward the audience's register — commoner words for accessibility, "
            "rarer ones for precision"
        ),
    ),
    "unusual_hedge_rate": (
        ("hedge_rate",),
        "hedging ('perhaps', 'may') is unusual for this baseline",
        "trim or add hedging so the text's confidence level matches comparable writing",
    ),
    "unusual_booster_rate": (
        ("booster_rate",),
        "intensifiers ('clearly', 'very') are unusual for this baseline",
        "trim or add boosters so the emphasis matches the evidence",
    ),
}


def _indicator_z_keys(indicator: str, z_scores: dict[str, Any]) -> list[str]:
    """Return the z-score keys behind an indicator that are present in the response.

    ``pos_anomalies`` is driven by every ``pos_*``/``posbi_*`` key rather than a
    fixed set, mirroring how ``generate_flags`` scans those prefixes.
    """
    if indicator == "pos_anomalies":
        return sorted(key for key in z_scores if key.startswith(("pos_", "posbi_")))
    keys = _INDICATOR_PROFILE.get(indicator, ((),))[0]
    return [key for key in keys if key in z_scores]


def _indicator_message(indicator: str, z_keys: list[str], z_scores: dict[str, Any]) -> str:
    """Ground an indicator finding in the measured z-scores where they exist."""
    description = _INDICATOR_PROFILE.get(indicator, ("", f"detector flagged {indicator}", ""))[1]
    if indicator == "pos_anomalies" and z_keys:
        top = sorted(z_keys, key=lambda key: abs(float(z_scores[key])), reverse=True)[:3]
        parts = ", ".join(f"{key} {float(z_scores[key]):+.2f}" for key in top)
        return f"part-of-speech patterns deviate from the baseline ({parts})"
    if z_keys:
        return f"{description} ({float(z_scores[z_keys[0]]):+.2f} z-score vs the baseline on {z_keys[0]})"
    return description


def from_stylometry(result: dict[str, Any], location: str = "document") -> list[Finding]:
    """Build findings from a successful ``stylometric_analysis`` response.

    Driven entirely by ``z_scores`` and the detector ``flags``: indicator
    findings carry curated fix hints, the overall verdict becomes the headline
    finding, and remaining statistical outliers (features flagged far from the
    baseline that no indicator already interprets) become their own findings.

    ``location`` re-anchors the findings when the same response shape is built
    for a region inside a larger document (e.g. ``section:## Notes``); with
    the default it reproduces the standalone tool's locations exactly.
    """
    flags = result.get("flags", {})
    z_scores = result.get("z_scores", {})
    indicators = flags.get("ai_indicators", [])

    findings: list[Finding] = []

    if flags.get("high_ai_probability"):
        findings.append(
            Finding(
                rule="high_ai_probability",
                location=location,
                message=(
                    f"stylometric confidence {flags.get('confidence_score')} ({flags.get('ai_detection_confidence')}) "
                    f"from {len(indicators)} converging indicator(s)"
                ),
                fix_hint=(
                    "revise for human cadence first — sentence-length variety and concrete specifics — "
                    "then re-run stylometric_analysis to verify the confidence score drops"
                ),
            )
        )

    covered: set[str] = set()
    for indicator in indicators:
        z_keys = _indicator_z_keys(indicator, z_scores)
        covered.update(z_keys)
        findings.append(
            Finding(
                rule=str(indicator),
                location=location,
                message=_indicator_message(str(indicator), z_keys, z_scores),
                fix_hint=_INDICATOR_PROFILE.get(
                    indicator, ("", "", "compare the draft with baseline-register writing and revise the flagged habit")
                )[2],
            )
        )

    # Statistical outliers no indicator already interprets.
    for feature in list(flags.get("errors", [])) + list(flags.get("warnings", [])):
        if feature in covered:
            continue
        z = z_scores.get(feature)
        detail = (
            f"sits {abs(float(z)):.2f} standard deviations from the baseline mean"
            if z is not None
            else "flagged as a baseline outlier"
        )
        findings.append(
            Finding(
                rule=f"{feature}_outlier",
                location=location,
                message=f"{feature} {detail}",
                fix_hint=(
                    "compare this aspect of the draft with the baseline register, revise toward its range, "
                    "then re-run the analysis"
                ),
            )
        )

    return findings


def from_keyword_density(keyword: str, density: float) -> list[Finding]:
    """Build findings from a keyword-density measurement (percent of words)."""
    if density > KEYWORD_DENSITY_MAX:
        return [
            Finding(
                rule="high_keyword_density",
                location="document",
                message=f"keyword '{keyword}' fills {density:.1f}% of the text (above the {KEYWORD_DENSITY_MAX:.0f}% stuffing line)",
                fix_hint=(
                    "replace some occurrences with pronouns or precise synonyms "
                    "and let sentence structure carry the emphasis"
                ),
            )
        ]
    if density == 0.0:
        return [
            Finding(
                rule="keyword_absent",
                location="document",
                message=f"keyword '{keyword}' does not appear in the text",
                fix_hint=(
                    "if this is a target term, work it naturally into the opening "
                    "and one later paragraph instead of forcing repetitions"
                ),
            )
        ]
    return []


def from_keyword_frequency(frequencies: dict[str, int], stopwords_removed: bool = True) -> list[Finding]:
    """Build findings from a word-frequency count.

    Only the stopword-filtered counts are judged: with stopwords kept, common
    function words would dominate every share and the findings would be noise.
    """
    if not stopwords_removed:
        return []
    total = sum(frequencies.values())
    if total <= 0:
        return []

    findings: list[Finding] = []
    ranked = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)
    for word, count in ranked:
        share = count / total
        if share <= WORD_SHARE_MAX:
            break  # descending by count — nothing further can cross the share line
        findings.append(
            Finding(
                rule="overused_word",
                location="document",
                message=f"'{word}' accounts for {share * 100:.0f}% of the counted words ({count} of {total})",
                fix_hint=(
                    "vary this term with synonyms, pronouns, or a restructured sentence "
                    "wherever the repetition is not load-bearing"
                ),
            )
        )
        if len(findings) >= MAX_OVERUSED_FINDINGS:
            break
    return findings


def from_top_keywords(pairs: list[tuple[str, int]]) -> list[Finding]:
    """Build findings from the top-keyword ranking (``[(word, count), ...]``)."""
    if not pairs:
        return []
    total = sum(count for _, count in pairs)
    if total <= 0:
        return []

    top_word, top_count = pairs[0]
    share = top_count / total
    if share > TOP_KEYWORD_DOMINANCE:
        return [
            Finding(
                rule="keyword_dominance",
                location="document",
                message=(
                    f"'{top_word}' alone is {share * 100:.0f}% of the top-keyword occurrences ({top_count} of {total})"
                ),
                fix_hint=(
                    "spread the topic across more distinct terms — one lemma dominating the keyword list "
                    "usually signals a narrow repetition loop"
                ),
            )
        ]
    return []


def from_keyword_context(sentences: list[str]) -> list[Finding]:
    """Build findings from a keyword-context lookup result."""
    if not sentences:
        return [
            Finding(
                rule="keyword_absent",
                location="document",
                message="no sentence matches the keyword or its lemmas",
                fix_hint=(
                    "check the wording the draft actually uses for this concept, "
                    "or broaden the keyword to the term a reader would expect"
                ),
            )
        ]
    return []


# Impact tiers for ordering a revision brief: headline detector verdicts first,
# then stylometric signals, then readability, then keyword distribution.
_IMPACT_TIERS: dict[str, int] = {
    "high_ai_probability": 0,
    "low_sentence_perplexity": 2,
    "low_flesch": 3,
    "high_kincaid": 3,
    "high_fog": 3,
    "keyword_dominance": 4,
    "overused_word": 4,
    "high_keyword_density": 4,
    "keyword_absent": 5,
}
_INDICATOR_TIER = 1
_OUTLIER_SUFFIX = "_outlier"
_UNKNOWN_TIER = 90


def _impact_tier(rule: str) -> int:
    if rule in _IMPACT_TIERS:
        return _IMPACT_TIERS[rule]
    if rule in _INDICATOR_PROFILE:
        return _INDICATOR_TIER
    if rule.endswith(_OUTLIER_SUFFIX):
        return _INDICATOR_TIER
    return _UNKNOWN_TIER


def order_by_impact(findings: list[Finding]) -> list[Finding]:
    """Order findings highest-impact first; stable within a tier."""
    return sorted(findings, key=lambda finding: _impact_tier(finding["rule"]))
