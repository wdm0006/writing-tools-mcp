"""MCP prompt renderers for the revision workflow.

Pure string builders — the ``@mcp.prompt`` wrappers in ``server.app`` delegate
here. Rendering never runs analysis: prompts are cheap templates, and the
``findings`` argument carries results an agent already computed with the
analysis tools.
"""

import json
from typing import Any

from server.analyzers.findings import Finding, order_by_impact

__all__ = ["render_guided_revision", "render_writing_checklist"]

_FINDING_KEYS = ("rule", "location", "message", "fix_hint")


def _parse_findings(findings_json: str) -> tuple[list[Finding], str | None]:
    """Parse a findings JSON array from an analysis tool response.

    Returns the well-formed findings plus an optional note describing anything
    that was dropped, so a malformed argument degrades into guidance on the
    rendered brief instead of an error.
    """
    try:
        parsed: Any = json.loads(findings_json)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return (
            [],
            f"the `findings` argument could not be parsed as JSON ({exc}); run the analysis tools and pass their `findings` array here",
        )

    if not isinstance(parsed, list):
        return [], "the `findings` argument was not a JSON array; pass the `findings` value an analysis tool returned"

    findings: list[Finding] = []
    dropped = 0
    for entry in parsed:
        if isinstance(entry, dict) and all(key in entry for key in _FINDING_KEYS):
            findings.append(
                Finding(
                    rule=str(entry["rule"]),
                    location=str(entry["location"]),
                    message=str(entry["message"]),
                    fix_hint=str(entry["fix_hint"]),
                )
            )
        else:
            dropped += 1

    note = None
    if dropped:
        note = f"{dropped} of {len(parsed)} findings entries were malformed and were dropped"
    return findings, note


def render_guided_revision(document: str, findings_json: str | None = None) -> str:
    """Render a revision brief for a document, ordered by expected impact.

    With precomputed findings the brief lists every finding's rule, location,
    message, and fix hint in impact order. Without them it tells the agent
    which tools to run and to pass their ``findings`` array back.
    """
    lines: list[str] = [
        "You are revising the document below. Work through the findings in order — highest impact first — "
        "and change only what a finding justifies.",
    ]

    findings: list[Finding] = []
    notes: list[str] = []
    if findings_json:
        findings, note = _parse_findings(findings_json)
        if note:
            notes.append(note)

    if not findings_json:
        lines.extend(
            [
                "",
                "No precomputed findings were supplied. Before revising, run the analysis tools on this document:",
                '- `readability_score` (level="full") for readability pressure, plus `stylometric_analysis` '
                "and `perplexity_analysis` for AI-detection signal",
                "- `keyword_frequency` and `keyword_density` for repetition",
                "",
                "Each tool returns a `findings` array; pass it as the `findings` argument of this prompt "
                "to get an ordered revision brief.",
            ]
        )
    elif findings:
        lines.extend(["", "## Findings (impact order)"])
        for number, finding in enumerate(order_by_impact(findings), 1):
            lines.append(f"{number}. [{finding['rule']}] at {finding['location']} — {finding['message']}")
            lines.append(f"   Fix: {finding['fix_hint']}")
    else:
        lines.extend(
            [
                "",
                "The analysis reported no findings for this document — it is clean against the current "
                "baselines. Make only the changes the author intends, then re-run the analysis to confirm.",
            ]
        )

    for note in notes:
        lines.extend(["", f"NOTE: {note}"])

    lines.extend(
        [
            "",
            "## Document",
            "",
            document,
            "",
            "## Revision protocol",
            "1. Fix each finding in order, keeping the author's meaning.",
            "2. Re-run the analysis tools on the revised text and compare against the numbers in the findings.",
            "3. Stop when the findings clear or plateau — do not polish past the evidence.",
        ]
    )
    return "\n".join(lines)


def render_writing_checklist() -> str:
    """Render the pre-flight drafting checklist.

    A lightweight entry point for agents drafting from scratch, before any
    analysis exists: the habits the analysis tools measure, stated as drafting
    moves.
    """
    return "\n".join(
        [
            "Pre-flight checklist for a draft. Work these habits in as you write, not after:",
            "",
            "## Structure",
            "- Lead each section with its point, then support it — readers should get the gist from first sentences alone.",
            "- Keep one idea per paragraph; split when a second idea arrives.",
            "",
            "## Sentence variety",
            "- Mix short and long sentences deliberately; a run of similar lengths reads as mechanical.",
            "- Vary how clauses attach to the verb — not every sentence needs the same subject-verb-object shape.",
            "",
            "## Hedging and boosters",
            "- Match hedging ('perhaps', 'may') to the actual strength of the evidence.",
            "- Cut intensifiers ('very', 'clearly') that emphasis alone has to carry; let specifics do the work.",
            "",
            "## Readability",
            "- Aim for plain verbs over noun stacks; if a sentence needs a second read, rebuild it.",
            "- Prefer everyday words unless the precise term is the point.",
            "",
            "## Keywords and repetition",
            "- Use each key term where it matters, but vary with pronouns and synonyms elsewhere.",
            "- One dominant word repeated across paragraphs usually signals a narrow loop — widen the vocabulary.",
            "",
            "## Voice",
            "- Prefer active constructions; keep passive voice for when the actor genuinely does not matter.",
            "",
            "When the draft exists, verify with the analysis tools: `readability_score`, `spellcheck`, "
            "`keyword_frequency`, and `stylometric_analysis` each report a `findings` array of concrete, "
            "located fixes.",
        ]
    )
