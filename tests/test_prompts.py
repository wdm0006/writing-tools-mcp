"""Tests for the MCP prompts: discovery, rendering, and findings round-trip.

The integration tests go through the real FastMCP client so discovery and
rendering exercise the same protocol path a user's MCP session would.
"""

import asyncio
import json

import pytest
from fastmcp import Client

from server import app
from server.prompts import render_guided_revision, render_writing_checklist

DOCUMENT = "The committee reviewed the proposal carefully. A decision was deferred."

PLANTED_FINDINGS = [
    {
        "rule": "keyword_absent",
        "location": "document",
        "message": "keyword 'renewable' does not appear in the text",
        "fix_hint": "work it naturally into the opening paragraph",
    },
    {
        "rule": "high_ai_probability",
        "location": "document",
        "message": "stylometric confidence 0.8 (high) from 3 converging indicator(s)",
        "fix_hint": "revise for human cadence, then re-run the analysis",
    },
    {
        "rule": "low_flesch",
        "location": "section:Intro",
        "message": "Flesch Reading Ease 42.0 is below the 50 'difficult' line",
        "fix_hint": "split long sentences at their conjunctions",
    },
]


def prompt_text(result) -> str:
    """Concatenate the rendered text of a GetPromptResult."""
    return "\n".join(message.content.text for message in result.messages)


class TestGuidedRevisionRenderer:
    """Pure renderer units — parsing, ordering, and document inclusion."""

    def test_document_included(self):
        rendered = render_guided_revision(DOCUMENT)
        assert DOCUMENT in rendered
        assert "## Document" in rendered

    def test_planted_findings_render_rules_locations_and_hints(self):
        rendered = render_guided_revision(DOCUMENT, json.dumps(PLANTED_FINDINGS))
        for planted in PLANTED_FINDINGS:
            assert planted["rule"] in rendered
            assert planted["location"] in rendered
            assert planted["message"] in rendered
            assert planted["fix_hint"] in rendered

    def test_findings_listed_in_impact_order(self):
        rendered = render_guided_revision(DOCUMENT, json.dumps(PLANTED_FINDINGS))
        body = rendered.split("## Document")[0]  # findings section only
        positions = {name: body.index(name) for name in ("high_ai_probability", "low_flesch", "keyword_absent")}
        assert positions["high_ai_probability"] < positions["low_flesch"] < positions["keyword_absent"]

    def test_without_findings_arg_tells_you_to_run_tools(self):
        rendered = render_guided_revision(DOCUMENT)
        assert "No precomputed findings were supplied" in rendered
        assert "readability_score" in rendered
        assert "findings" in rendered

    def test_empty_findings_array_reports_clean(self):
        rendered = render_guided_revision(DOCUMENT, json.dumps([]))
        assert "no findings" in rendered

    def test_malformed_json_degrades_to_note(self):
        rendered = render_guided_revision(DOCUMENT, "not json at all")
        assert "NOTE:" in rendered
        assert DOCUMENT in rendered  # the brief still renders

    def test_non_array_json_degrades_to_note(self):
        rendered = render_guided_revision(DOCUMENT, json.dumps({"rule": "x"}))
        assert "not a JSON array" in rendered

    def test_malformed_entries_dropped_with_note(self):
        payload = json.dumps([PLANTED_FINDINGS[0], {"rule": "incomplete"}, "junk"])
        rendered = render_guided_revision(DOCUMENT, payload)
        assert "2 of 3 findings entries were malformed" in rendered
        assert "keyword_absent" in rendered
        assert "low_flesch" not in rendered.split("## Document")[0]  # dropped, not rendered

    def test_findings_order_survives_string_round_trip(self):
        payload = json.dumps(list(reversed(PLANTED_FINDINGS)))
        rendered = render_guided_revision(DOCUMENT, payload)
        keyword_pos = rendered.index("keyword_absent")
        flesch_pos = rendered.index("low_flesch")
        assert flesch_pos < keyword_pos  # tier 3 before tier 5 regardless of input order


class TestWritingChecklistRenderer:
    def test_sections_present(self):
        rendered = render_writing_checklist()
        for section in ("Structure", "Sentence variety", "Hedging", "Readability", "Keywords", "Voice"):
            assert section in rendered

    def test_points_at_analysis_tools(self):
        rendered = render_writing_checklist()
        assert "readability_score" in rendered
        assert "spellcheck" in rendered


class TestPromptRegistryIntegration:
    """Discovery and rendering through the real FastMCP client."""

    def _client(self):
        return Client(app.mcp)

    def test_prompts_discoverable(self):
        async def scenario():
            async with self._client() as client:
                prompts = await client.list_prompts()
                return {prompt.name for prompt in prompts}

        names = asyncio.run(scenario())
        assert {"guided_revision", "writing_checklist"} <= names

    def test_guided_revision_renders_planted_findings(self):
        async def scenario():
            async with self._client() as client:
                result = await client.get_prompt(
                    "guided_revision",
                    arguments={"document": DOCUMENT, "findings": json.dumps(PLANTED_FINDINGS)},
                )
                return prompt_text(result)

        rendered = asyncio.run(scenario())
        for planted in PLANTED_FINDINGS:
            assert planted["rule"] in rendered
            assert planted["location"] in rendered
            assert planted["fix_hint"] in rendered
        assert DOCUMENT in rendered

    def test_guided_revision_without_findings_renders_guidance(self):
        async def scenario():
            async with self._client() as client:
                result = await client.get_prompt("guided_revision", arguments={"document": DOCUMENT})
                return prompt_text(result)

        rendered = asyncio.run(scenario())
        assert "No precomputed findings were supplied" in rendered

    def test_writing_checklist_renders(self):
        async def scenario():
            async with self._client() as client:
                result = await client.get_prompt("writing_checklist", arguments={})
                return prompt_text(result)

        rendered = asyncio.run(scenario())
        assert "Structure" in rendered

    def test_tool_registry_still_lists_thirteen_tools(self):
        """The prompt decorators must not disturb tool registration."""
        tools = asyncio.run(app.mcp.list_tools())
        assert len(tools) == 13


@pytest.mark.parametrize(
    "renderer, args",
    [
        (render_guided_revision, (DOCUMENT, None)),
        (render_guided_revision, (DOCUMENT, "[]")),
        (render_writing_checklist, ()),
    ],
    ids=["guided-no-findings", "guided-clean", "checklist"],
)
def test_renderers_are_deterministic(renderer, args):
    assert renderer(*args) == renderer(*args)
