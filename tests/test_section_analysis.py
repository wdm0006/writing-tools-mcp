"""Tests for ``analyze_sections`` — per-section batch analysis plus rollup (W5).

Covers the tool-selection contract, the location re-anchoring added to the W1
finding builders, the orchestration semantics (section order, heading levels,
findings shape, degenerate documents, rollup consistency), and the registered
MCP tool over the real FastMCP client on a planted three-section document.
"""

import asyncio

import pytest
from fastmcp import Client

from server import app
from server.analyzers.findings import (
    Finding,
    from_perplexity,
    from_readability_result,
    from_stylometry,
    order_by_impact,
)
from server.analyzers.section_analysis import (
    ALL_SECTION_TOOLS,
    DEFAULT_SECTION_TOOLS,
    analyze_document_sections,
    models_for_selection,
    resolve_selection,
)
from server.text_processing import parse_markdown_sections

THREE_SECTION_DOC = (
    "A short preamble opens the document before the first heading appears.\n"
    "\n"
    "## First Section\n"
    "\n"
    "The cat sat on the mat and watched the dog with quiet, deliberate curiosity.\n"
    "\n"
    "## Second Section\n"
    "\n"
    "There were many decisions made by the committee during the long and rather contentious meeting.\n"
    "\n"
    "## Third Section\n"
    "\n"
    "Finally a conclusion arrives with enough extra words to give the readability gate something to weigh.\n"
)

EXPECTED_SECTION_KEYS = ["_leading_content", "## First Section", "## Second Section", "## Third Section"]

# Synthetic detector responses for the location-override unit tests: shaped
# exactly like successful perplexity_analysis / stylometric_analysis responses.
SAMPLE_PERPLEXITY_RESULT = {
    "doc_ppl": 210.5,
    "doc_burstiness": 180.2,
    "flags": {"high_ai_probability": True, "reasons": ["low perplexity"]},
    "sentences": [{"sentence": "the text", "ppl": 21.0}],
}

SAMPLE_STYLOMETRY_RESULT = {
    "features": {},
    "z_scores": {"type_token_ratio": -3.2},
    "flags": {
        "high_ai_probability": True,
        "confidence_score": 0.7,
        "ai_detection_confidence": "medium",
        "ai_indicators": ["low_lexical_diversity"],
        "errors": ["type_token_ratio"],
        "warnings": [],
    },
}


class TestResolveSelection:
    def test_none_selects_the_default_menu(self):
        assert resolve_selection(None) == list(DEFAULT_SECTION_TOOLS)

    def test_gpt2_tools_are_opt_in(self):
        assert "perplexity" not in DEFAULT_SECTION_TOOLS
        assert "stylometry" not in DEFAULT_SECTION_TOOLS
        assert set(ALL_SECTION_TOOLS) == set(DEFAULT_SECTION_TOOLS) | {"perplexity", "stylometry"}

    def test_unknown_tool_raises_naming_the_menu(self):
        with pytest.raises(ValueError, match="ghost_tool.*spellcheck"):
            resolve_selection(["ghost_tool"])

    def test_empty_selection_raises(self):
        with pytest.raises(ValueError, match="No analysis tools selected"):
            resolve_selection([])

    def test_duplicates_dropped_order_preserved(self):
        assert resolve_selection(["spellcheck", "readability", "spellcheck"]) == ["spellcheck", "readability"]


class TestModelsForSelection:
    def test_model_independent_selection_needs_nothing(self):
        assert models_for_selection(["readability", "word_count"]) == ()

    def test_spellcheck_needs_spacy_only(self):
        assert models_for_selection(["spellcheck"]) == ("spacy",)

    def test_perplexity_needs_both_tiers(self):
        assert models_for_selection(["perplexity"]) == ("spacy", "gpt2")

    def test_stylometry_needs_both_tiers(self):
        assert models_for_selection(["stylometry"]) == ("spacy", "gpt2")


class TestFindingsLocationOverride:
    """The ``location`` parameter re-anchors findings; the default is unchanged."""

    def test_perplexity_default_locations_unchanged(self):
        findings = from_perplexity(SAMPLE_PERPLEXITY_RESULT)
        assert [f["location"] for f in findings] == ["document", "s0"]

    def test_perplexity_section_locations(self):
        findings = from_perplexity(SAMPLE_PERPLEXITY_RESULT, "section:## Notes")
        assert [f["location"] for f in findings] == ["section:## Notes", "section:## Notes s0"]

    def test_stylometry_default_locations_unchanged(self):
        findings = from_stylometry(SAMPLE_STYLOMETRY_RESULT)
        assert findings
        assert all(f["location"] == "document" for f in findings)

    def test_stylometry_section_locations(self):
        findings = from_stylometry(SAMPLE_STYLOMETRY_RESULT, "section:## Notes")
        assert findings
        assert all(f["location"] == "section:## Notes" for f in findings)


def _sections_result(
    text,
    tools,
    basic_stats_analyzer,
    readability_analyzer,
    style_analyzer=None,
    ai_detection_analyzer=None,
    baseline=None,
):
    return analyze_document_sections(
        text,
        tools,
        basic_stats=basic_stats_analyzer,
        readability=readability_analyzer,
        style=style_analyzer,
        ai_detection=ai_detection_analyzer,
        baseline=baseline,
    )


class TestAnalyzeDocumentSections:
    """Orchestration semantics against the session analyzers."""

    def test_sections_in_document_order_with_levels(self, basic_stats_analyzer, readability_analyzer):
        result = _sections_result(
            THREE_SECTION_DOC, ["word_count", "readability"], basic_stats_analyzer, readability_analyzer
        )
        assert [s["key"] for s in result["sections"]] == EXPECTED_SECTION_KEYS
        assert [s["heading_level"] for s in result["sections"]] == [0, 2, 2, 2]
        assert result["section_count"] == len(EXPECTED_SECTION_KEYS)
        assert result["tools_used"] == ["word_count", "readability"]

    def test_per_section_text_matches_the_parser(self, basic_stats_analyzer, readability_analyzer):
        sections_data = parse_markdown_sections(THREE_SECTION_DOC)
        result = _sections_result(THREE_SECTION_DOC, ["word_count"], basic_stats_analyzer, readability_analyzer)
        for section in result["sections"]:
            assert section["text"] == sections_data[section["key"]]

    def test_per_section_results_match_standalone_calls(self, basic_stats_analyzer, readability_analyzer):
        result = _sections_result(
            THREE_SECTION_DOC, ["word_count", "readability"], basic_stats_analyzer, readability_analyzer
        )
        for section in result["sections"]:
            assert section["results"]["word_count"] == basic_stats_analyzer.word_count(section["text"])
            assert section["results"]["readability"] == readability_analyzer.readability_score(section["text"])

    def test_findings_shape_and_locations(self, basic_stats_analyzer, readability_analyzer):
        result = _sections_result(THREE_SECTION_DOC, ["readability"], basic_stats_analyzer, readability_analyzer)
        for section in result["sections"]:
            for finding in section["findings"]:
                assert set(finding) == {"rule", "location", "message", "fix_hint"}
                assert finding["location"].startswith(f"section:{section['key']}")
                assert finding["message"]
                assert finding["fix_hint"]

    def test_findings_impact_ordered_within_section(self, basic_stats_analyzer, readability_analyzer):
        result = _sections_result(THREE_SECTION_DOC, ["readability"], basic_stats_analyzer, readability_analyzer)
        for section in result["sections"]:
            findings = [Finding(**finding) for finding in section["findings"]]
            assert findings == order_by_impact(findings)

    def test_rollup_matches_standalone_responses(self, basic_stats_analyzer, readability_analyzer):
        result = _sections_result(
            THREE_SECTION_DOC, ["readability", "word_count", "reading_time"], basic_stats_analyzer, readability_analyzer
        )
        expected_scores = readability_analyzer.readability_score(THREE_SECTION_DOC)
        expected_scores["findings"] = order_by_impact(from_readability_result(expected_scores))
        assert result["rollup"]["readability"] == expected_scores
        assert result["rollup"]["word_count"] == basic_stats_analyzer.word_count(THREE_SECTION_DOC)
        assert result["rollup"]["reading_time"] == readability_analyzer.reading_time(THREE_SECTION_DOC)

    def test_empty_document_yields_no_sections_with_rollup(self, basic_stats_analyzer, readability_analyzer):
        result = _sections_result("", ["word_count", "readability"], basic_stats_analyzer, readability_analyzer)
        assert result["sections"] == []
        assert result["section_count"] == 0
        assert result["rollup"]["word_count"] == 0
        assert result["rollup"]["readability"]["flesch"] is None

    def test_whitespace_only_document_yields_no_sections(self, basic_stats_analyzer, readability_analyzer):
        result = _sections_result("   \n\t\n", ["word_count"], basic_stats_analyzer, readability_analyzer)
        assert result["sections"] == []
        assert result["section_count"] == 0

    def test_no_heading_document_single_leading_section(self, basic_stats_analyzer, readability_analyzer):
        doc = "Just some text without any heading at all.\nA second line follows it."
        result = _sections_result(doc, ["word_count"], basic_stats_analyzer, readability_analyzer)
        assert [s["key"] for s in result["sections"]] == ["_leading_content"]
        assert result["sections"][0]["heading_level"] == 0
        assert result["sections"][0]["text"] == doc

    def test_unknown_tool_raises_at_the_orchestration_layer(self, basic_stats_analyzer, readability_analyzer):
        with pytest.raises(ValueError, match="ghost_tool"):
            _sections_result(THREE_SECTION_DOC, ["ghost_tool"], basic_stats_analyzer, readability_analyzer)


class TestSectionGpt2Tools:
    """The opt-in GPT-2 tools run per section and in the rollup."""

    def test_perplexity_per_section_and_rollup(self, basic_stats_analyzer, readability_analyzer, ai_detection_analyzer):
        result = _sections_result(
            THREE_SECTION_DOC,
            ["perplexity"],
            basic_stats_analyzer,
            readability_analyzer,
            ai_detection_analyzer=ai_detection_analyzer,
        )
        for section in result["sections"]:
            assert section["results"]["perplexity"] == ai_detection_analyzer.perplexity_analysis(section["text"])
        expected_rollup = ai_detection_analyzer.perplexity_analysis(THREE_SECTION_DOC)
        expected_rollup["findings"] = order_by_impact(from_perplexity(expected_rollup))
        assert result["rollup"]["perplexity"] == expected_rollup

    def test_stylometry_baseline_passthrough_and_locations(
        self, basic_stats_analyzer, readability_analyzer, ai_detection_analyzer
    ):
        result = _sections_result(
            THREE_SECTION_DOC,
            ["stylometry"],
            basic_stats_analyzer,
            readability_analyzer,
            ai_detection_analyzer=ai_detection_analyzer,
            baseline="brown_corpus",
        )
        assert result["rollup"]["stylometry"]["baseline_used"] == "brown_corpus"
        for section in result["sections"]:
            for finding in section["findings"]:
                assert finding["location"].startswith(f"section:{section['key']}")


class TestAnalyzeSectionsMcpIntegration:
    """The registered tool over the real FastMCP client — planted 3-section document."""

    def _analyze(self, text, tools=None):
        async def scenario():
            async with Client(app.mcp) as client:
                arguments = {"text": text}
                if tools is not None:
                    arguments["tools"] = tools
                result = await client.call_tool("analyze_sections", arguments)
                return result.data

        return asyncio.run(scenario())

    def test_tool_discoverable(self):
        tools = asyncio.run(app.mcp.list_tools())
        assert "analyze_sections" in {tool.name for tool in tools}

    def test_planted_three_section_document(self):
        data = self._analyze(THREE_SECTION_DOC, tools=["readability", "word_count", "reading_time"])
        assert data["tools_used"] == ["readability", "word_count", "reading_time"]
        assert [s["key"] for s in data["sections"]] == EXPECTED_SECTION_KEYS
        assert [s["heading_level"] for s in data["sections"]] == [0, 2, 2, 2]
        for section in data["sections"]:
            assert set(section) == {"key", "heading_level", "text", "findings", "results"}
            for finding in section["findings"]:
                assert set(finding) == {"rule", "location", "message", "fix_hint"}
                assert finding["location"].startswith(f"section:{section['key']}")

    def test_rollup_consistent_with_whole_document_analysis(self):
        data = self._analyze(THREE_SECTION_DOC, tools=["readability", "word_count"])

        async def scenario():
            async with Client(app.mcp) as client:
                word_count = await client.call_tool("word_count", {"text": THREE_SECTION_DOC})
                readability = await client.call_tool("readability_score", {"text": THREE_SECTION_DOC})
                return word_count.data, readability.data

        word_count_data, readability_data = asyncio.run(scenario())
        assert data["rollup"]["word_count"] == word_count_data
        assert data["rollup"]["readability"] == readability_data

    def test_unknown_tool_selection_returns_error(self):
        data = self._analyze(THREE_SECTION_DOC, tools=["ghost_tool"])
        assert "error" in data and "ghost_tool" in data["error"]

    def test_empty_document_defined_behavior(self):
        data = self._analyze("", tools=["word_count"])
        assert data["sections"] == []
        assert data["section_count"] == 0
        assert data["rollup"]["word_count"] == 0

    def test_no_heading_document_single_section(self):
        doc = "A document with no markdown headings at all, just prose."
        data = self._analyze(doc, tools=["word_count"])
        assert [s["key"] for s in data["sections"]] == ["_leading_content"]
        assert [s["heading_level"] for s in data["sections"]] == [0]

    def test_default_selection_runs_without_models_flagged_opt_in(self):
        data = self._analyze(THREE_SECTION_DOC)
        assert data["tools_used"] == list(DEFAULT_SECTION_TOOLS)
        assert "perplexity" not in data["rollup"]
