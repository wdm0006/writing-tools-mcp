"""Section-level batch analysis (W5 of the writing-quality spec).

Runs a selected subset of the analysis tools against every markdown section of
a document — via :func:`parse_markdown_sections` — and against the whole
document (the rollup). Per-section findings reuse the shared W1 builders with
``section:<key>`` locations, so a revision can target the section that needs it
instead of regenerating the document.

The orchestration here is a pure function of its inputs: the app layer supplies
the analyzer instances, so tests can drive this module with the session
fixtures and the MCP layer stays a thin wrapper that only decides which models
to load and when to release them.
"""

from collections.abc import Callable
from typing import Any, Sequence

from server.analyzers.ai_detection import AIDetectionAnalyzer
from server.analyzers.basic_stats import BasicStatsAnalyzer
from server.analyzers.findings import (
    Finding,
    from_perplexity,
    from_readability,
    from_stylometry,
    order_by_impact,
)
from server.analyzers.readability import ReadabilityAnalyzer
from server.analyzers.style_analysis import StyleAnalyzer
from server.text_processing import parse_markdown_sections

__all__ = [
    "ALL_SECTION_TOOLS",
    "DEFAULT_SECTION_TOOLS",
    "analyze_document_sections",
    "models_for_selection",
    "resolve_selection",
]

# The per-section menu. Every entry runs on a bare text string with no other
# required arguments — the keyword tools need a target keyword, which is not a
# per-section concern, so they are excluded on purpose.
SPACY_TOOLS = ("spellcheck", "passive_voice")
GPT2_TOOLS = ("perplexity", "stylometry")
MODEL_INDEPENDENT_TOOLS = ("readability", "word_count", "character_count", "reading_time")

DEFAULT_SECTION_TOOLS = MODEL_INDEPENDENT_TOOLS + SPACY_TOOLS
ALL_SECTION_TOOLS = DEFAULT_SECTION_TOOLS + GPT2_TOOLS

# Tools whose responses gain a findings array (the W1 dict-shaped tools); the
# rollup attaches one exactly as the standalone tool wrapper would.
FINDINGS_TOOLS = ("readability", "perplexity", "stylometry")


def models_for_selection(selection: Sequence[str]) -> tuple[str, ...]:
    """The model tiers a selection needs, in cleanup order.

    The GPT-2 tools pull in spaCy as well: constructing the AI-detection
    analyzer requires the spaCy model, so a perplexity or stylometry run loads
    both tiers.
    """
    models: list[str] = []
    if any(tool in SPACY_TOOLS or tool in GPT2_TOOLS for tool in selection):
        models.append("spacy")
    if any(tool in GPT2_TOOLS for tool in selection):
        models.append("gpt2")
    return tuple(models)


def resolve_selection(tools: Sequence[str] | None) -> list[str]:
    """Validate the requested tool subset; ``None`` selects the default set.

    Duplicates are dropped while preserving request order. An empty selection
    or an unknown name raises ``ValueError`` naming the valid tools.
    """
    if tools is None:
        return list(DEFAULT_SECTION_TOOLS)

    selection = list(dict.fromkeys(tools))
    if not selection:
        raise ValueError(f"No analysis tools selected. Choose from: {', '.join(ALL_SECTION_TOOLS)}")

    unknown = [tool for tool in selection if tool not in ALL_SECTION_TOOLS]
    if unknown:
        raise ValueError(f"Unknown analysis tool(s): {', '.join(unknown)}. Choose from: {', '.join(ALL_SECTION_TOOLS)}")

    return selection


def _findings_for_tool(tool_name: str, result: dict[str, Any], location: str) -> list[Finding]:
    """Findings for one raw tool result, anchored at the given region.

    Error-path responses (which carry an ``error`` key) gain nothing, matching
    the standalone tool wrappers.
    """
    if tool_name not in FINDINGS_TOOLS:
        return []
    if "error" in result:
        return []
    if tool_name == "readability":
        return from_readability(result, location)
    if tool_name == "perplexity":
        return from_perplexity(result, location)
    if tool_name == "stylometry":
        return from_stylometry(result, location)
    return []


def _require(analyzer: Any, requirement: str) -> Any:
    """Return the analyzer, or raise if the selection's dependency was not supplied."""
    if analyzer is None:
        raise ValueError(requirement)
    return analyzer


def _run_tool(
    tool_name: str,
    segment: str,
    *,
    basic_stats: BasicStatsAnalyzer,
    readability: ReadabilityAnalyzer,
    style: StyleAnalyzer | None,
    ai_detection: AIDetectionAnalyzer | None,
    baseline: str | None,
) -> Any:
    """Run one selected tool on a text segment.

    The response shape is exactly that tool's own response for the segment at
    its default settings — the same rule the rollup and per-section results
    follow, so any result can be compared 1:1 with a standalone tool call.
    """
    runners: dict[str, Callable[[], Any]] = {
        "readability": lambda: readability.readability_score(segment),
        "reading_time": lambda: readability.reading_time(segment),
        "word_count": lambda: basic_stats.word_count(segment),
        "character_count": lambda: basic_stats.character_count(segment),
        "spellcheck": lambda: basic_stats.spellcheck(segment),
        "passive_voice": lambda: _require(
            style, "tool 'passive_voice' requires the style analyzer"
        ).passive_voice_detection(segment),
        "perplexity": lambda: _require(
            ai_detection, "tool 'perplexity' requires the AI detection analyzer"
        ).perplexity_analysis(segment),
        "stylometry": lambda: _require(
            ai_detection, "tool 'stylometry' requires the AI detection analyzer"
        ).stylometric_analysis(segment, baseline),
    }
    if tool_name not in runners:  # guarded by resolve_selection
        raise ValueError(f"Unknown analysis tool: {tool_name}")
    return runners[tool_name]()


def analyze_document_sections(
    text: str,
    tools: Sequence[str],
    *,
    basic_stats: BasicStatsAnalyzer,
    readability: ReadabilityAnalyzer,
    style: StyleAnalyzer | None = None,
    ai_detection: AIDetectionAnalyzer | None = None,
    baseline: str | None = None,
) -> dict[str, Any]:
    """Run the selected tools per markdown section and roll them up document-wide.

    Sections come from :func:`parse_markdown_sections` in document order: the
    pre-first-heading content (when present) is the ``_leading_content`` section
    at heading level 0, and a subsection's body folds into its parent. Each
    section entry carries:

    - ``key``: the section's heading key (or ``_leading_content``),
    - ``heading_level``: the heading's level, 0 for leading content,
    - ``text``: the section's rendered plain-text body,
    - ``findings``: every selected tool's observations for the section,
      impact-ordered, located at ``section:<key>`` (the W1 Finding shape),
    - ``results``: each selected tool's raw response for the section text.

    The ``rollup`` carries each selected tool's response for the whole document
    — identical to the standalone tool's output, findings included — so a
    caller can verify per-section views against whole-document analysis.

    An empty document yields ``sections: []`` with the rollup still computed; a
    document with no headings yields the single ``_leading_content`` section.
    """
    selection = resolve_selection(tools)
    sections_data = parse_markdown_sections(text)
    # _section_levels maps section key -> heading level in document order; its
    # keys are exactly the real sections (heading-keyed plus leading content).
    level_by_key: dict[str, int] = sections_data.get("_section_levels", {})
    section_keys = list(level_by_key)

    sections: list[dict[str, Any]] = []
    for key in section_keys:
        section_text = sections_data[key]
        results: dict[str, Any] = {}
        findings: list[Finding] = []
        for tool_name in selection:
            result = _run_tool(
                tool_name,
                section_text,
                basic_stats=basic_stats,
                readability=readability,
                style=style,
                ai_detection=ai_detection,
                baseline=baseline,
            )
            results[tool_name] = result
            findings.extend(_findings_for_tool(tool_name, result, f"section:{key}"))
        sections.append(
            {
                "key": key,
                "heading_level": level_by_key.get(key, 0),
                "text": section_text,
                "findings": order_by_impact(findings),
                "results": results,
            }
        )

    rollup: dict[str, Any] = {}
    for tool_name in selection:
        result = _run_tool(
            tool_name,
            text,
            basic_stats=basic_stats,
            readability=readability,
            style=style,
            ai_detection=ai_detection,
            baseline=baseline,
        )
        # Attach findings exactly as the standalone tool wrapper does, so the
        # rollup equals that tool's whole-document response. Bare-shaped tools
        # (int counts, lists) have no findings and no error key.
        if tool_name in FINDINGS_TOOLS and "error" not in result:
            rollup_location = "full_text" if tool_name == "readability" else "document"
            result = {**result, "findings": order_by_impact(_findings_for_tool(tool_name, result, rollup_location))}
        rollup[tool_name] = result

    return {
        "sections": sections,
        "rollup": rollup,
        "tools_used": list(selection),
        "section_count": len(sections),
    }
