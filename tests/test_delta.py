"""
Tests for the stylometric_delta tool and the verify_revision prompt.

Covers the pure delta math (hand-computed), the analyzer's baseline handling
and error envelopes, the identical-inputs regression, the tool-layer schema
with findings on the revised text, and rendering through the prompt registry.
"""

import asyncio
import json
from unittest.mock import Mock, patch

import pytest
from fastmcp import Client

import server.app as app
from server.prompts import render_verify_revision
from server.stylometry import compute_statistic_deltas, compute_verdicts

DRAFT_TEXT = (
    "The committee reviewed the proposal carefully. Members raised several questions about funding. "
    "A decision was deferred until the next meeting. The chair summarized the discussion briefly. "
    "Further analysis will be conducted by the subcommittee. Recommendations are expected next month."
)

REVISED_TEXT = (
    "The committee approved the revised budget yesterday. "
    "The council rejected the amended proposal quickly. "
    "The board reviewed the updated schedule carefully."
)

DELTA_SUCCESS_KEYS = {"baseline_used", "deltas", "verdict", "text_b_analysis"}
DELTA_ERROR_KEYS = {"error", "baseline_used", "deltas", "verdict", "text_b_analysis"}


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


@pytest.fixture
def stylometry_tool(ai_detection_analyzer, mock_managers):
    """Wire the real session analyzer into the tool layer; managers stay mocked."""
    with patch.object(app, "initialize_analyzers", Mock(return_value={"ai_detection": ai_detection_analyzer})):
        yield ai_detection_analyzer


class TestComputeStatisticDeltas:
    """Hand-computed movement between two z-score dictionaries."""

    def test_shared_statistics_only(self):
        deltas = compute_statistic_deltas({"fog": -1.0, "only_a": 2.5}, {"fog": 1.0, "only_b": 3.0})
        assert [entry["statistic"] for entry in deltas] == ["fog"]

    def test_sorted_by_statistic_name(self):
        deltas = compute_statistic_deltas({"ttr": 1.0, "fog": 1.0}, {"fog": 1.0, "ttr": 1.0})
        assert [entry["statistic"] for entry in deltas] == ["fog", "ttr"]

    def test_hand_computed_movement(self):
        # z moved from -1.25 to 1.25: crosses zero, equal distance on the other side.
        (entry,) = compute_statistic_deltas({"fog": -1.25}, {"fog": 1.25})
        assert entry == {
            "statistic": "fog",
            "z_a": -1.25,
            "z_b": 1.25,
            "delta": 2.5,
            "direction": "increased",
        }

    def test_direction_decreased(self):
        (entry,) = compute_statistic_deltas({"ttr": 2.0}, {"ttr": 0.5})
        assert entry["delta"] == -1.5
        assert entry["direction"] == "decreased"

    def test_direction_none(self):
        (entry,) = compute_statistic_deltas({"ttr": 0.5}, {"ttr": 0.5})
        assert entry["delta"] == 0.0
        assert entry["direction"] == "none"

    def test_identical_inputs_yield_exact_zero(self):
        z_scores = {"fog": -0.12345, "ttr": 2.5, "punct_density": 0.0}
        deltas = compute_statistic_deltas(z_scores, dict(z_scores))
        assert all(entry["delta"] == 0.0 for entry in deltas)
        assert all(entry["direction"] == "none" for entry in deltas)

    def test_rounding_keeps_delta_equal_to_reported_difference(self):
        # z_a rounds to 1.11 and z_b to 1.12; the reported delta is the
        # difference of the two REPORTED z-scores, at the same precision.
        (entry,) = compute_statistic_deltas({"ttr": 1.114}, {"ttr": 1.116})
        assert entry["z_a"] == 1.11
        assert entry["z_b"] == 1.12
        assert entry["delta"] == round(entry["z_b"] - entry["z_a"], 2)
        assert entry["delta"] == 0.01


class TestComputeVerdicts:
    """Hand-computed verdicts: closer to zero is better, sign aside."""

    def test_toward_baseline_improves(self):
        verdicts = compute_verdicts(
            [
                {"statistic": "fog", "z_a": -2.0, "z_b": -1.0},
                {"statistic": "ttr", "z_a": 2.0, "z_b": 1.0},
            ]
        )
        assert verdicts == [
            {"statistic": "fog", "verdict": "improved"},
            {"statistic": "ttr", "verdict": "improved"},
        ]

    def test_away_from_baseline_regresses(self):
        verdicts = compute_verdicts([{"statistic": "fog", "z_a": -1.0, "z_b": -2.5}])
        assert verdicts == [{"statistic": "fog", "verdict": "regressed"}]

    def test_sign_flip_with_equal_magnitude_unchanged(self):
        # Crossing zero does not help: the distance from the baseline is the same.
        verdicts = compute_verdicts(
            [
                {"statistic": "fog", "z_a": 1.0, "z_b": -1.0},
                {"statistic": "ttr", "z_a": 0.0, "z_b": 0.0},
            ]
        )
        assert [verdict["verdict"] for verdict in verdicts] == ["unchanged", "unchanged"]

    def test_verdicts_run_parallel_to_deltas(self):
        deltas = compute_statistic_deltas({"a": 1.0, "b": 3.0}, {"a": 2.0, "b": 2.0})
        verdicts = compute_verdicts(deltas)
        assert [verdict["statistic"] for verdict in verdicts] == [entry["statistic"] for entry in deltas]


class TestStylometricDeltaAnalyzer:
    """Analyzer-level behavior: default baseline, error paths, identical inputs."""

    def test_identical_inputs_yield_zero_deltas_and_unchanged(self, ai_detection_analyzer):
        result = ai_detection_analyzer.stylometric_delta(DRAFT_TEXT, DRAFT_TEXT)

        assert "error" not in result
        assert result["deltas"], "shared statistics must be present"
        assert all(entry["delta"] == 0.0 for entry in result["deltas"])
        assert all(entry["direction"] == "none" for entry in result["deltas"])
        assert result["verdict"]
        assert all(verdict["verdict"] == "unchanged" for verdict in result["verdict"])
        assert [verdict["statistic"] for verdict in result["verdict"]] == [
            entry["statistic"] for entry in result["deltas"]
        ]

    def test_default_baseline_matches_solo_analysis(self, ai_detection_analyzer):
        result = ai_detection_analyzer.stylometric_delta(DRAFT_TEXT, REVISED_TEXT)
        solo = ai_detection_analyzer.stylometric_analysis(REVISED_TEXT)

        assert result["baseline_used"]
        assert result["baseline_used"] == solo["baseline_used"]

    def test_explicit_baseline_round_trips(self, ai_detection_analyzer):
        result = ai_detection_analyzer.stylometric_delta(DRAFT_TEXT, REVISED_TEXT, baseline="brown_corpus")

        assert result["baseline_used"] == "brown_corpus"

    def test_missing_baseline_names_it(self, ai_detection_analyzer):
        result = ai_detection_analyzer.stylometric_delta(DRAFT_TEXT, REVISED_TEXT, baseline="no_such_baseline_xyz")

        assert "no_such_baseline_xyz" in result["error"]
        assert result["baseline_used"] == "no_such_baseline_xyz"
        assert result["deltas"] == []
        assert result["verdict"] == []
        assert result["text_b_analysis"] is None

    def test_empty_text_a_error(self, ai_detection_analyzer):
        result = ai_detection_analyzer.stylometric_delta("", REVISED_TEXT)

        assert "text_a" in result["error"]
        assert result["deltas"] == []
        assert result["verdict"] == []

    def test_empty_text_b_error(self, ai_detection_analyzer):
        result = ai_detection_analyzer.stylometric_delta(DRAFT_TEXT, "   ")

        assert "text_b" in result["error"]
        assert result["deltas"] == []
        assert result["verdict"] == []

    def test_text_b_analysis_profiles_the_revised_text(self, ai_detection_analyzer):
        result = ai_detection_analyzer.stylometric_delta(DRAFT_TEXT, REVISED_TEXT)

        # The revised text is uniform; its sentence-length dispersion scores
        # far below the baseline (verified behavior of this same text in
        # tests/test_stylometry.py).
        assert result["text_b_analysis"]["z_scores"]["sentence_len_std"] < -2.0


class TestStylometricDeltaTool:
    """Tool-layer envelope: findings on the revised text, family error shape."""

    def test_success_envelope_adds_findings_only(self, stylometry_tool):
        result = app.stylometric_delta(DRAFT_TEXT, REVISED_TEXT)

        assert "error" not in result
        assert set(result) == DELTA_SUCCESS_KEYS | {"findings"}
        assert_finding_shape(result["findings"])

    def test_findings_target_the_revised_text(self, stylometry_tool):
        result = app.stylometric_delta(DRAFT_TEXT, REVISED_TEXT)

        assert "uniform_sentences" in {finding["rule"] for finding in result["findings"]}

    def test_error_envelope_keeps_family_shape(self, stylometry_tool):
        result = app.stylometric_delta("", REVISED_TEXT)

        assert set(result) == DELTA_ERROR_KEYS
        assert "findings" not in result
        assert "text_a" in result["error"]


class TestMCPClientIntegration:
    """Discovery and round-trip through the real FastMCP client."""

    def test_tool_round_trip(self, stylometry_tool):
        async def scenario():
            async with Client(app.mcp) as client:
                tools = await client.list_tools()
                result = await client.call_tool("stylometric_delta", {"text_a": DRAFT_TEXT, "text_b": REVISED_TEXT})
                return {tool.name for tool in tools}, result

        names, result = asyncio.run(scenario())
        assert "stylometric_delta" in names
        payload = result.data
        assert "error" not in payload
        assert payload["deltas"]
        assert all(verdict["verdict"] in {"improved", "regressed", "unchanged"} for verdict in payload["verdict"])

    def test_verify_revision_prompt_round_trip(self, stylometry_tool):
        async def scenario():
            async with Client(app.mcp) as client:
                delta = await client.call_tool("stylometric_delta", {"text_a": DRAFT_TEXT, "text_b": REVISED_TEXT})
                prompt = await client.get_prompt("verify_revision", arguments={"delta": json.dumps(delta.data)})
                return delta.data, prompt

        payload, prompt = asyncio.run(scenario())
        rendered = "\n".join(message.content.text for message in prompt.messages)
        assert f"## Verdict (baseline: {payload['baseline_used']})" in rendered
        assert "## Verification protocol" in rendered


def prompt_text(result) -> str:
    """Concatenate the rendered text of a GetPromptResult."""
    return "\n".join(message.content.text for message in result.messages)


FLAGGED_DELTA = {
    "baseline_used": "brown_corpus",
    "deltas": [
        {"statistic": "fog", "z_a": 1.0, "z_b": 3.0, "delta": 2.0, "direction": "increased"},
        {"statistic": "ttr", "z_a": -2.0, "z_b": -1.0, "delta": 1.0, "direction": "increased"},
        {"statistic": "comma_ratio", "z_a": 1.0, "z_b": -1.0, "delta": -2.0, "direction": "decreased"},
        {"statistic": "punct_density", "z_a": 0.5, "z_b": 0.5, "delta": 0.0, "direction": "none"},
    ],
    "verdict": [
        {"statistic": "fog", "verdict": "regressed"},
        {"statistic": "ttr", "verdict": "improved"},
        {"statistic": "comma_ratio", "verdict": "unchanged"},
        {"statistic": "punct_density", "verdict": "unchanged"},
    ],
    "text_b_analysis": {},
    "findings": [
        {
            "rule": "uniform_sentences",
            "location": "document",
            "message": "sentence lengths are uniform",
            "fix_hint": "vary sentence length",
        },
        {
            "rule": "low_flesch",
            "location": "full_text",
            "message": "flesch 10.0",
            "fix_hint": "simplify",
        },
    ],
}


class TestVerifyRevisionRenderer:
    """Pure renderer units: grouping, movement lines, degradation."""

    def test_verdict_groups_and_movement_lines(self):
        rendered = render_verify_revision(json.dumps(FLAGGED_DELTA))

        assert "## Verdict (baseline: brown_corpus)" in rendered
        assert "1 improved, 1 regressed, 2 unchanged." in rendered
        assert "Improved: ttr" in rendered
        assert "Regressed: fog" in rendered
        assert "- fog: z 1.00 -> 3.00 (delta +2.00, increased)" in rendered
        assert "- ttr: z -2.00 -> -1.00 (delta +1.00, increased)" in rendered

    def test_regressed_section_before_improved(self):
        rendered = render_verify_revision(json.dumps(FLAGGED_DELTA))

        assert rendered.index("## Regressed") < rendered.index("## Improved")

    def test_unchanged_statistics_listed(self):
        rendered = render_verify_revision(json.dumps(FLAGGED_DELTA))

        assert "## Unchanged" in rendered
        assert "comma_ratio, punct_density" in rendered

    def test_findings_rendered_with_fix_hints(self):
        rendered = render_verify_revision(json.dumps(FLAGGED_DELTA))

        assert "## Findings on the revised text" in rendered
        assert "[uniform_sentences]" in rendered
        assert "Fix: vary sentence length" in rendered
        assert "[low_flesch]" in rendered

    def test_clean_revision_renders_none(self):
        payload = {
            "baseline_used": "brown_corpus",
            "deltas": [{"statistic": "fog", "z_a": 1.0, "z_b": 1.0, "delta": 0.0, "direction": "none"}],
            "verdict": [{"statistic": "fog", "verdict": "unchanged"}],
            "text_b_analysis": {},
            "findings": [],
        }

        rendered = render_verify_revision(json.dumps(payload))

        assert "None — the revised text is clean against this baseline." in rendered

    def test_unparseable_delta_degrades_to_guidance(self):
        rendered = render_verify_revision("not json {")

        assert "## Verification could not run" in rendered
        assert "could not be parsed as JSON" in rendered
        assert "NOTE:" in rendered
        assert "## Verification protocol" in rendered

    def test_non_object_delta_degrades_to_guidance(self):
        rendered = render_verify_revision(json.dumps([1, 2]))

        assert "not a JSON object" in rendered
        assert "## Verification could not run" in rendered

    def test_error_envelope_renders_unavailable(self):
        error_payload = {
            "error": "Baseline 'gone' not found.",
            "baseline_used": "gone",
            "deltas": [],
            "verdict": [],
            "text_b_analysis": None,
        }

        rendered = render_verify_revision(json.dumps(error_payload))

        assert "## Verification could not run" in rendered
        assert "Baseline 'gone' not found." in rendered
        assert "## Verification protocol" in rendered

    def test_malformed_entries_dropped_with_note(self):
        payload = {
            "deltas": [
                {"statistic": "fog", "z_a": 1.0, "z_b": 3.0, "delta": 2.0, "direction": "increased"},
                "junk",
                {"statistic": "ttr"},
            ],
            "verdict": [
                {"statistic": "fog", "verdict": "regressed"},
                {"statistic": "ghost", "verdict": "improved"},
            ],
            "findings": [
                {"rule": "low_flesch", "location": "full_text", "message": "m", "fix_hint": "h"},
                "junk",
            ],
            "baseline_used": "brown_corpus",
        }

        rendered = render_verify_revision(json.dumps(payload))

        assert "2 deltas entries were malformed and were dropped" in rendered
        assert "1 verdict entries had no renderable delta entry and were dropped" in rendered
        assert "1 findings entries were malformed and were dropped" in rendered
        assert "- fog: z 1.00 -> 3.00 (delta +2.00, increased)" in rendered

    def test_renderer_deterministic(self):
        assert render_verify_revision(json.dumps(FLAGGED_DELTA)) == render_verify_revision(json.dumps(FLAGGED_DELTA))
