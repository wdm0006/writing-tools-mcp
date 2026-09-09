"""Tests for the findings layer: builders, tool enrichment, and schema contracts.

The schema-invariant tests pin every pre-existing response key on ALL return
paths (spec addendum item 1): the analysis tools may add ``findings`` on
success, and may change nothing else anywhere.
"""

from unittest.mock import Mock, patch

import pytest

from server import app
from server.analyzers import AIDetectionAnalyzer
from server.analyzers.findings import (
    MAX_OVERUSED_FINDINGS,
    Finding,
    from_keyword_context,
    from_keyword_density,
    from_keyword_frequency,
    from_perplexity,
    from_readability,
    from_readability_result,
    from_stylometry,
    from_top_keywords,
    order_by_impact,
)
from server.stylometry import StylemetricAnalyzer

SAMPLE_TEXT = (
    "The committee reviewed the proposal carefully. Members raised several questions about funding. "
    "A decision was deferred until the next meeting. The chair summarized the discussion briefly. "
    "Further analysis will be conducted by the subcommittee. Recommendations are expected next month."
)

READABILITY_FULL_KEYS = {"flesch", "kincaid", "fog"}
PERPLEXITY_BASE_KEYS = {"doc_ppl", "doc_burstiness", "sentences", "config", "flags"}
STYLOMETRY_BASE_KEYS = {
    "features",
    "z_scores",
    "sentence_analysis",
    "config",
    "flags",
    "char_ngram_similarity",
    # Additive since #59: every path names the baseline actually measured against.
    "baseline_used",
}

# Indicators whose z-scores do not exist in a 9-statistic baseline like brown_corpus:
# no such baseline may ever produce findings for these rules.
UNREACHABLE_WITH_9_STAT_BASELINE = {
    "low_mtld",
    "unusual_hedge_rate",
    "unusual_booster_rate",
    "unusual_vocabulary_rarity",
    "distinct_function_word_profile",
}


def finding(**overrides) -> Finding:
    """Build a well-formed finding with defaults for the optional slots."""
    base: Finding = {
        "rule": "low_flesch",
        "location": "full_text",
        "message": "sample message",
        "fix_hint": "sample fix hint",
    }
    base.update(overrides)
    return base


def assert_finding_shape(findings) -> None:
    """Every finding carries exactly the uniform four keys."""
    for finding_item in findings:
        assert set(finding_item) == {"rule", "location", "message", "fix_hint"}


@pytest.fixture
def mock_managers():
    """Replace module-level model managers and initialization hooks.

    Patching the initializers matters: with the analyzer caches reset, a tool
    call re-enters get_analyzers(), which would otherwise feed a Mock nlp into
    the real initialize_sentence_splitter() and clobber the session global.
    """
    spacy_manager = Mock()
    gpt2_manager = Mock()
    with (
        patch.object(app, "spacy_manager", spacy_manager),
        patch.object(app, "gpt2_manager", gpt2_manager),
        patch.object(app, "initialize_preprocessor", Mock()),
        patch.object(app, "initialize_sentence_splitter", Mock()),
    ):
        yield {"spacy": spacy_manager, "gpt2": gpt2_manager}


@pytest.fixture(autouse=True)
def reset_analyzer_caches():
    """Keep the module-level analyzer caches from leaking between tests."""
    app._analyzers = None
    app._model_independent_analyzers = None
    yield
    app._analyzers = None
    app._model_independent_analyzers = None


class TestFromReadability:
    """Threshold boundaries are exclusive on the healthy side."""

    def test_flesch_at_boundary_is_clean(self):
        assert from_readability({"flesch": 50.0, "kincaid": None, "fog": None}) == []

    def test_flesch_one_step_below_fires(self):
        findings = from_readability({"flesch": 49.9, "kincaid": None, "fog": None})
        assert [f["rule"] for f in findings] == ["low_flesch"]
        assert findings[0]["location"] == "full_text"
        assert "49.9" in findings[0]["message"]

    def test_kincaid_at_boundary_is_clean(self):
        assert from_readability({"flesch": None, "kincaid": 12.0, "fog": None}) == []

    def test_kincaid_one_step_above_fires(self):
        findings = from_readability({"flesch": None, "kincaid": 12.1, "fog": None})
        assert [f["rule"] for f in findings] == ["high_kincaid"]

    def test_fog_at_boundary_is_clean(self):
        assert from_readability({"flesch": None, "kincaid": None, "fog": 12.0}) == []

    def test_fog_one_step_above_fires(self):
        findings = from_readability({"flesch": None, "kincaid": None, "fog": 12.1})
        assert [f["rule"] for f in findings] == ["high_fog"]

    def test_none_scores_never_fire(self):
        assert from_readability({"flesch": None, "kincaid": None, "fog": None}) == []

    def test_all_three_bad_fire_three_findings(self):
        findings = from_readability({"flesch": 10.0, "kincaid": 20.0, "fog": 25.0}, location="section:Intro")
        assert {f["rule"] for f in findings} == {"low_flesch", "high_kincaid", "high_fog"}
        assert all(f["location"] == "section:Intro" for f in findings)

    def test_fix_hints_coach_the_fix(self):
        findings = from_readability({"flesch": 10.0, "kincaid": 20.0, "fog": 25.0})
        for hint in (f["fix_hint"] for f in findings):
            assert hint  # non-empty and imperative
            assert hint[0].islower() or hint.startswith(("split", "break", "shorten", "adjust"))


class TestFromReadabilityResult:
    """Section and paragraph envelopes earn located findings per scored leaf."""

    def test_full_level_dispatches_to_bare_scores(self):
        result = {"flesch": 10.0, "kincaid": 20.0, "fog": 25.0}
        findings = from_readability_result(result)
        assert {f["rule"] for f in findings} == {"low_flesch", "high_kincaid", "high_fog"}
        assert all(f["location"] == "full_text" for f in findings)

    def test_section_level_locations(self):
        result = {
            "full_text": {"flesch": 10.0, "kincaid": None, "fog": None},
            "sections": {"Intro": {"flesch": 20.0, "kincaid": 30.0, "fog": None}},
        }
        findings = from_readability_result(result)
        locations = {f["location"] for f in findings}
        assert locations == {"full_text", "section:Intro"}

    def test_paragraph_level_locations(self):
        result = {
            "full_text": {"flesch": None, "kincaid": None, "fog": None},
            "paragraphs": [
                {"paragraph_number": 1, "text": "one", "scores": {"flesch": 10.0, "kincaid": None, "fog": None}},
                {"paragraph_number": 2, "text": "two", "scores": {"flesch": None, "kincaid": None, "fog": None}},
            ],
        }
        findings = from_readability_result(result)
        assert [f["location"] for f in findings] == ["paragraph:1"]


class TestFromPerplexity:
    """The headline fires on the detector verdict; sentences get located findings."""

    def _result(self, sentences, high_ai=True, thresholds=None):
        return {
            "doc_ppl": 15.0,
            "doc_burstiness": 0.0,
            "sentences": sentences,
            "config": {"thresholds": thresholds or {"ppl_max": 25.0}},
            "flags": {"high_ai_probability": high_ai},
        }

    def test_no_flag_means_no_findings(self):
        assert from_perplexity(self._result([{"text": "a", "ppl": 1.0}], high_ai=False)) == []

    def test_flag_fires_headline_with_verdict_values(self):
        findings = from_perplexity(self._result([{"text": "a", "ppl": 30.0}]))
        assert [f["rule"] for f in findings] == ["high_ai_probability"]
        assert findings[0]["location"] == "document"
        assert "15.0" in findings[0]["message"]

    def test_predictable_sentences_get_s_index_locations(self):
        sentences = [{"text": "a", "ppl": 24.9}, {"text": "b", "ppl": 30.0}, {"text": "c", "ppl": 10.0}]
        findings = from_perplexity(self._result(sentences))
        rules = [f["rule"] for f in findings]
        assert rules == ["high_ai_probability", "low_sentence_perplexity", "low_sentence_perplexity"]
        assert [f["location"] for f in findings[1:]] == ["s0", "s2"]

    def test_sentence_at_exact_threshold_is_clean(self):
        findings = from_perplexity(self._result([{"text": "a", "ppl": 25.0}]))
        assert [f["rule"] for f in findings] == ["high_ai_probability"]

    def test_threshold_read_from_config(self):
        # ppl 40.0 would not fire against the 25.0 default; the config's 50.0 line does.
        findings = from_perplexity(self._result([{"text": "a", "ppl": 40.0}], thresholds={"ppl_max": 50.0}))
        assert [f["rule"] for f in findings] == ["high_ai_probability", "low_sentence_perplexity"]


class TestFromStylometry:
    """Indicator findings are z-grounded; outliers fill the gaps indicators leave."""

    def test_flag_off_means_no_findings(self):
        result = {"flags": {}, "z_scores": {}}
        assert from_stylometry(result) == []

    def test_headline_precedes_indicators(self):
        result = {
            "flags": {
                "high_ai_probability": True,
                "confidence_score": 0.8,
                "ai_detection_confidence": "high",
                "ai_indicators": ["low_ttr"],
            },
            "z_scores": {"ttr": -2.5},
        }
        findings = from_stylometry(result)
        assert [f["rule"] for f in findings] == ["high_ai_probability", "low_ttr"]
        assert "0.8" in findings[0]["message"]
        assert "-2.50" in findings[1]["message"]

    def test_indicator_without_z_score_still_renders(self):
        result = {"flags": {"ai_indicators": ["low_ttr"]}, "z_scores": {}}
        findings = from_stylometry(result)
        assert len(findings) == 1
        assert findings[0]["rule"] == "low_ttr"
        assert findings[0]["fix_hint"]

    def test_unknown_indicator_gets_generic_guidance(self):
        result = {"flags": {"ai_indicators": ["mystery_indicator"]}, "z_scores": {}}
        findings = from_stylometry(result)
        assert findings[0]["rule"] == "mystery_indicator"
        assert findings[0]["fix_hint"]

    def test_pos_anomalies_cover_all_pos_keys(self):
        result = {
            "flags": {"ai_indicators": ["pos_anomalies"], "warnings": ["pos_NOUN", "ttr"]},
            "z_scores": {"pos_NOUN": 2.5, "posbi_DET_NOUN": -2.1, "ttr": 2.5},
        }
        findings = from_stylometry(result)
        rules = [f["rule"] for f in findings]
        assert "pos_anomalies" in rules
        assert "pos_NOUN_outlier" not in rules  # covered by the indicator
        assert "ttr_outlier" in rules  # not covered — the gap fills

    def test_uncovered_outlier_becomes_finding(self):
        result = {
            "flags": {"ai_indicators": [], "errors": ["avg_word_len"], "warnings": []},
            "z_scores": {"avg_word_len": 2.3},
        }
        findings = from_stylometry(result)
        assert [f["rule"] for f in findings] == ["avg_word_len_outlier"]
        assert "2.30" in findings[0]["message"]

    def test_outlier_without_z_value_still_renders(self):
        result = {"flags": {"ai_indicators": [], "warnings": ["mystery_feature"]}, "z_scores": {}}
        findings = from_stylometry(result)
        assert findings[0]["rule"] == "mystery_feature_outlier"

    def test_synthetic_nine_stat_baseline_is_honest(self):
        """A 9-statistic response can only produce findings its data supports."""
        z_scores = {
            "ttr": -2.5,
            "hapax_legomena_rate": -2.5,
            "sentence_len_std": -2.5,
            "avg_sentence_len": 2.5,
            "pos_NOUN": 2.5,
            "punct_density": -2.5,
            "comma_ratio": 2.5,
            "function_word_ratio": -2.5,
            "avg_word_len": 2.5,
        }
        result = {
            "flags": {
                "ai_indicators": [
                    "low_ttr",
                    "low_hapax",
                    "uniform_sentences",
                    "unusual_sentence_length",
                    "pos_anomalies",
                    "function_word_anomaly",
                ],
                "errors": ["comma_ratio"],
                "warnings": ["punct_density", "avg_word_len"],
            },
            "z_scores": z_scores,
        }
        rules = {f["rule"] for f in from_stylometry(result)}
        assert not (rules & UNREACHABLE_WITH_9_STAT_BASELINE)


class TestRealBaselineHonesty:
    """The real brown_corpus run never generates findings it lacks data for."""

    def test_nine_stat_baseline_findings_are_scoped(self, stylometry_tool):
        result = app.stylometric_analysis(SAMPLE_TEXT, "brown_corpus", "en")
        assert "error" not in result
        assert_finding_shape(result["findings"])

        rules = {f["rule"] for f in result["findings"]}
        assert not (rules & UNREACHABLE_WITH_9_STAT_BASELINE)

        for finding_item in result["findings"]:
            if finding_item["rule"].endswith("_outlier"):
                assert finding_item["rule"][: -len("_outlier")] in result["z_scores"]


class TestKeywordDensityFindings:
    def test_below_stuffing_line_is_clean(self):
        assert from_keyword_density("kw", 2.0) == []

    def test_at_boundary_is_clean(self):
        assert from_keyword_density("kw", 5.0) == []

    def test_above_boundary_fires(self):
        findings = from_keyword_density("kw", 5.1)
        assert [f["rule"] for f in findings] == ["high_keyword_density"]
        assert "5.1" in findings[0]["message"]

    def test_absent_keyword_fires(self):
        findings = from_keyword_density("kw", 0.0)
        assert [f["rule"] for f in findings] == ["keyword_absent"]


class TestKeywordFrequencyFindings:
    def test_no_stopword_filter_means_no_judgment(self):
        assert from_keyword_frequency({"the": 100, "a": 50}, stopwords_removed=False) == []

    def test_empty_counts_are_clean(self):
        assert from_keyword_frequency({}) == []

    def test_share_boundary_exclusive(self):
        # 1 of 10 = exactly 0.10 → clean; 1 of 9 ≈ 0.111 → fires (capped at MAX_OVERUSED_FINDINGS).
        assert from_keyword_frequency({f"w{i}": 1 for i in range(10)}) == []
        findings = from_keyword_frequency({f"w{i}": 1 for i in range(9)})
        assert len(findings) == MAX_OVERUSED_FINDINGS
        assert all(f["rule"] == "overused_word" for f in findings)

    def test_findings_capped_and_sorted_by_count(self):
        counts = {"big": 50, "medium": 30, "small": 15, "tiny": 5}
        findings = from_keyword_frequency(counts)
        assert len(findings) == 3  # MAX_OVERUSED_FINDINGS
        assert all(f["rule"] == "overused_word" for f in findings)
        assert "big" in findings[0]["message"]

    def test_minority_word_is_clean(self):
        # Twenty equal words: each share is 0.05, comfortably under the line.
        assert from_keyword_frequency({f"w{i}": 1 for i in range(20)}) == []


class TestTopKeywordsFindings:
    def test_empty_is_clean(self):
        assert from_top_keywords([]) == []

    def test_dominance_boundary_exclusive(self):
        # 1 of 4 = exactly 0.25 → clean.
        assert from_top_keywords([("a", 1), ("b", 1), ("c", 1), ("d", 1)]) == []
        findings = from_top_keywords([("a", 2), ("b", 2), ("c", 2), ("d", 1)])
        assert [f["rule"] for f in findings] == ["keyword_dominance"]
        assert "a" in findings[0]["message"]


class TestKeywordContextFindings:
    def test_matches_are_clean(self):
        assert from_keyword_context(["A matching sentence."]) == []

    def test_no_matches_fire_keyword_absent(self):
        findings = from_keyword_context([])
        assert [f["rule"] for f in findings] == ["keyword_absent"]


class TestOrderByImpact:
    def test_tier_ordering(self):
        findings = [
            finding(rule="keyword_absent"),
            finding(rule="low_flesch"),
            finding(rule="low_sentence_perplexity"),
            finding(rule="high_ai_probability"),
            finding(rule="uniform_sentences"),
        ]
        ordered = order_by_impact(findings)
        assert [f["rule"] for f in ordered] == [
            "high_ai_probability",
            "uniform_sentences",
            "low_sentence_perplexity",
            "low_flesch",
            "keyword_absent",
        ]

    def test_stable_within_tier(self):
        findings = [finding(rule="low_flesch"), finding(rule="high_fog")]
        assert [f["rule"] for f in order_by_impact(findings)] == ["low_flesch", "high_fog"]

    def test_outlier_rules_tier_with_indicators(self):
        findings = [finding(rule="low_flesch"), finding(rule="ttr_outlier")]
        assert [f["rule"] for f in order_by_impact(findings)] == ["ttr_outlier", "low_flesch"]

    def test_unknown_rule_sorts_last(self):
        findings = [finding(rule="brand_new_rule"), finding(rule="keyword_absent")]
        assert [f["rule"] for f in order_by_impact(findings)] == ["keyword_absent", "brand_new_rule"]


class TestReadabilityToolSchema:
    """readability_score keeps its keys on every path; findings join success only."""

    def test_full_level_adds_findings(self):
        result = app.readability_score(SAMPLE_TEXT, "full")
        assert set(result) == READABILITY_FULL_KEYS | {"findings"}
        assert_finding_shape(result["findings"])

    def test_section_level_adds_findings(self):
        result = app.readability_score(SAMPLE_TEXT, "section")
        assert set(result) == {"full_text", "sections", "findings"}
        assert_finding_shape(result["findings"])

    def test_paragraph_level_adds_findings(self):
        result = app.readability_score(SAMPLE_TEXT, "paragraph")
        assert set(result) == {"full_text", "paragraphs", "findings"}
        assert_finding_shape(result["findings"])

    def test_invalid_level_error_exact(self):
        result = app.readability_score(SAMPLE_TEXT, "bogus")
        assert set(result) == {"error"}
        assert "findings" not in result

    def test_too_short_text_still_schema_clean(self):
        result = app.readability_score("Hi.", "full")
        assert set(result) == READABILITY_FULL_KEYS | {"findings"}


class TestPerplexityToolSchema:
    """perplexity_analysis: 5 base keys + findings on success, + error on failure."""

    @pytest.fixture
    def mock_gpt2_manager(self):
        manager = Mock()
        config = {
            "model_name": "gpt2",
            "max_length": 512,
            "overlap": 50,
            "thresholds": {"ppl_max": 50, "burstiness_min": 1.0},
        }
        manager.get_model_and_tokenizer.return_value = (Mock(), Mock(), config)
        return manager

    @pytest.fixture
    def mock_config(self):
        return {"gpt2": {"model_name": "gpt2", "cache_dir": "models/gpt2", "tokenizer": "gpt2"}}

    @pytest.fixture
    def perplexity_tool(self, mock_gpt2_manager, mock_config, mock_managers, nlp):
        """Wire a mock-backed analyzer into the tool layer (real spaCy, no GPT-2)."""
        analyzer = AIDetectionAnalyzer(nlp, mock_gpt2_manager, mock_config)
        with patch.object(app, "initialize_analyzers", Mock(return_value={"ai_detection": analyzer})):
            yield analyzer

    def test_language_error_keeps_exact_shape(self, perplexity_tool):
        result = app.perplexity_analysis("Test text", language="fr")
        assert set(result) == PERPLEXITY_BASE_KEYS | {"error"}
        assert "findings" not in result
        assert result["error"] == "Only English language ('en') is currently supported"

    def test_empty_text_error_keeps_exact_shape(self, perplexity_tool):
        result = app.perplexity_analysis("", language="en")
        assert set(result) == PERPLEXITY_BASE_KEYS | {"error"}
        assert "findings" not in result

    @patch("server.analyzers.ai_detection.split_into_sentences", Mock(return_value=[]))
    def test_no_sentences_error_keeps_exact_shape(self, perplexity_tool):
        result = app.perplexity_analysis("Some text", language="en")
        assert set(result) == PERPLEXITY_BASE_KEYS | {"error"}
        assert "findings" not in result

    @patch.object(AIDetectionAnalyzer, "_chunk_text", Mock(side_effect=lambda text, *args: [text]))
    @patch.object(AIDetectionAnalyzer, "_calculate_perplexity", Mock(return_value=15.0))
    def test_success_path_adds_findings_only(self, perplexity_tool):
        result = app.perplexity_analysis(
            "The cat sat on the mat. The dog ran in the park. The bird flew over the moon."
        )
        assert set(result) == PERPLEXITY_BASE_KEYS | {"findings"}
        assert "error" not in result
        assert_finding_shape(result["findings"])
        # Identical mock perplexities => zero burstiness => the detector flags the text.
        assert result["flags"]["high_ai_probability"] is True
        assert any(f["rule"] == "high_ai_probability" for f in result["findings"])

    @patch.object(AIDetectionAnalyzer, "_calculate_perplexity", Mock(side_effect=RuntimeError("boom")))
    def test_exception_path_keeps_exact_shape(self, perplexity_tool):
        result = app.perplexity_analysis("Some text", language="en")
        assert set(result) == PERPLEXITY_BASE_KEYS | {"error"}
        assert "findings" not in result


@pytest.fixture
def stylometry_tool(ai_detection_analyzer, mock_managers):
    """Wire the real session analyzer into the tool layer; managers stay mocked."""
    with patch.object(app, "initialize_analyzers", Mock(return_value={"ai_detection": ai_detection_analyzer})):
        yield ai_detection_analyzer


class TestStylometricToolSchema:
    """stylometric_analysis: 6 base keys + findings on success, + error on failure."""

    def test_success_path_adds_findings_only(self, stylometry_tool):
        result = app.stylometric_analysis(SAMPLE_TEXT, "brown_corpus", "en")
        assert set(result) == STYLOMETRY_BASE_KEYS | {"findings"}
        assert "error" not in result
        assert_finding_shape(result["findings"])
        rules = {f["rule"] for f in result["findings"]}
        assert not (rules & UNREACHABLE_WITH_9_STAT_BASELINE)

    def test_language_error_keeps_exact_shape(self, stylometry_tool):
        result = app.stylometric_analysis(SAMPLE_TEXT, "brown_corpus", language="fr")
        assert set(result) == STYLOMETRY_BASE_KEYS | {"error"}
        assert "findings" not in result

    def test_empty_text_error_keeps_exact_shape(self, stylometry_tool):
        result = app.stylometric_analysis("", "brown_corpus", "en")
        assert set(result) == STYLOMETRY_BASE_KEYS | {"error"}
        assert "findings" not in result

    def test_unknown_baseline_error_keeps_exact_shape(self, stylometry_tool):
        result = app.stylometric_analysis(SAMPLE_TEXT, "no_such_baseline_xyz", "en")
        assert set(result) == STYLOMETRY_BASE_KEYS | {"error"}
        assert "findings" not in result

    @patch.object(StylemetricAnalyzer, "extract_features", Mock(side_effect=RuntimeError("boom")))
    def test_exception_path_keeps_exact_shape(self, stylometry_tool):
        result = app.stylometric_analysis(SAMPLE_TEXT, "brown_corpus", "en")
        assert set(result) == STYLOMETRY_BASE_KEYS | {"error"}
        assert "findings" not in result


@pytest.fixture
def mock_analyzers(mock_managers):
    """Mock analyzer set wired into the tool layer for keyword-tool tests."""
    analyzers = {name: Mock() for name in ("basic_stats", "keyword", "style")}
    with (
        patch.object(app, "initialize_analyzers", Mock(return_value=analyzers)),
        patch.object(app, "initialize_preprocessor", Mock()),
        patch.object(app, "initialize_sentence_splitter", Mock()),
    ):
        yield analyzers


class TestKeywordToolEnvelopes:
    """The four keyword tools return envelopes; analyzers keep raw shapes."""

    def test_density_envelope(self, mock_analyzers):
        mock_analyzers["keyword"].keyword_density.return_value = 7.5
        result = app.keyword_density("text with keyword", "keyword")
        assert set(result) == {"keyword", "density", "findings"}
        assert result["density"] == 7.5
        assert [f["rule"] for f in result["findings"]] == ["high_keyword_density"]

    def test_density_empty_text_skips_judgment(self, mock_analyzers):
        mock_analyzers["keyword"].keyword_density.return_value = 0.0
        result = app.keyword_density("", "kw")
        assert result == {"keyword": "kw", "density": 0.0, "findings": []}

    def test_frequency_envelope(self, mock_analyzers):
        mock_analyzers["keyword"].keyword_frequency.return_value = {"foo": 5, "bar": 1}
        result = app.keyword_frequency("some text")
        assert set(result) == {"frequencies", "findings"}
        assert result["frequencies"] == {"foo": 5, "bar": 1}
        assert all(f["rule"] == "overused_word" for f in result["findings"])

    def test_frequency_word_named_findings_cannot_collide(self, mock_analyzers):
        mock_analyzers["keyword"].keyword_frequency.return_value = {"findings": 3, "other": 1}
        result = app.keyword_frequency("some text")
        assert result["frequencies"]["findings"] == 3
        assert all(f["rule"] == "overused_word" for f in result["findings"])

    def test_top_keywords_envelope(self, mock_analyzers):
        mock_analyzers["keyword"].top_keywords.return_value = [("foo", 5), ("bar", 1)]
        result = app.top_keywords("some text", top_n=5)
        assert set(result) == {"keywords", "findings"}
        assert result["keywords"] == [("foo", 5), ("bar", 1)]
        assert [f["rule"] for f in result["findings"]] == ["keyword_dominance"]

    def test_keyword_context_envelope(self, mock_analyzers):
        mock_analyzers["keyword"].keyword_context.return_value = ["A matching sentence."]
        result = app.keyword_context("some text", "match")
        assert set(result) == {"keyword", "sentences", "findings"}
        assert result["sentences"] == ["A matching sentence."]
        assert result["findings"] == []

    def test_keyword_context_empty_fires_absent(self, mock_analyzers):
        mock_analyzers["keyword"].keyword_context.return_value = []
        result = app.keyword_context("some text", "missing")
        assert [f["rule"] for f in result["findings"]] == ["keyword_absent"]


class TestPassiveVoiceBareListContract:
    """passive_voice_detection must stay a bare list[str] — regression pin."""

    def test_tool_returns_bare_list(self, mock_managers, mock_analyzers):
        mock_analyzers["style"].passive_voice_detection.return_value = ["Sent one.", "Sent two."]
        result = app.passive_voice_detection("The ball was thrown by the boy.")
        assert isinstance(result, list)
        assert result == ["Sent one.", "Sent two."]
        assert all(isinstance(sentence, str) for sentence in result)

    def test_analyzer_returns_bare_list_of_str(self, style_analyzer):
        result = style_analyzer.passive_voice_detection("The ball was thrown by the boy. The dog barked loudly.")
        assert isinstance(result, list)
        assert all(isinstance(sentence, str) for sentence in result)
        assert "The ball was thrown by the boy." in result
