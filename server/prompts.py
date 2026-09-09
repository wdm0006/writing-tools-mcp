"""MCP prompt renderers for the revision workflow.

Pure string builders — the ``@mcp.prompt`` wrappers in ``server.app`` delegate
here. Rendering never runs analysis: prompts are cheap templates, and the
arguments carry results an agent already computed with the analysis tools.
"""

import json
import textwrap
from typing import Any

from server.analyzers.findings import Finding, order_by_impact

__all__ = ["render_guided_revision", "render_verify_revision", "render_writing_checklist"]

_FINDING_KEYS = ("rule", "location", "message", "fix_hint")

_WRAP_WIDTH = 88


def _well_formed_findings(entries: Any) -> tuple[list[Finding], int]:
    """Filter findings entries down to well-formed ``Finding`` dicts.

    Returns the accepted findings plus how many entries were dropped, so a
    malformed element degrades into a rendered note instead of an error.
    """
    if not isinstance(entries, list):
        return [], 0

    findings: list[Finding] = []
    dropped = 0
    for entry in entries:
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
    return findings, dropped


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

    findings, dropped = _well_formed_findings(parsed)
    note = None
    if dropped:
        note = f"{dropped} of {len(parsed)} findings entries were malformed and were dropped"
    return findings, note


def _numbered_findings(findings: list[Finding]) -> list[str]:
    """Render impact-ordered findings as the numbered brief both prompts share."""
    lines = []
    for number, finding in enumerate(order_by_impact(findings), 1):
        lines.append(f"{number}. [{finding['rule']}] at {finding['location']} — {finding['message']}")
        lines.append(f"   Fix: {finding['fix_hint']}")
    return lines


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
        lines.extend(_numbered_findings(findings))
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


def _well_formed_deltas(entries: Any) -> tuple[dict[str, dict[str, Any]], int]:
    """Map well-formed delta entries by statistic name; count dropped entries.

    A delta entry is well formed when it carries a string ``statistic``, the
    numeric ``z_a``/``z_b``/``delta`` trio, and a string ``direction`` — the
    shape ``stylometric_delta`` emits. Anything else is dropped (and counted),
    so a hand-mangled payload degrades into a note instead of a crash.
    """
    if not isinstance(entries, list):
        return {}, 0

    by_statistic: dict[str, dict[str, Any]] = {}
    dropped = 0
    for entry in entries:
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("statistic"), str)
            and isinstance(entry.get("direction"), str)
            and all(isinstance(entry.get(key), (int, float)) for key in ("z_a", "z_b", "delta"))
        ):
            by_statistic[entry["statistic"]] = entry
        else:
            dropped += 1
    return by_statistic, dropped


def _delta_movement_lines(statistics: list[str], by_statistic: dict[str, dict[str, Any]]) -> list[str]:
    """One grounded line per statistic: the z movement behind its verdict."""
    lines = []
    for name in statistics:
        entry = by_statistic[name]
        lines.append(
            f"- {name}: z {entry['z_a']:.2f} -> {entry['z_b']:.2f} (delta {entry['delta']:+.2f}, {entry['direction']})"
        )
    return lines


def _group_verdicts(verdicts: Any, deltas_by_statistic: dict[str, dict[str, Any]]) -> tuple[dict[str, list[str]], int]:
    """Bucket verdict entries by their verdict; count entries that cannot render.

    A verdict renders only when it is one of the three known values AND the
    response also carries a well-formed delta entry for the same statistic —
    the delta supplies the z movement the verdict section shows.
    """
    groups: dict[str, list[str]] = {"improved": [], "regressed": [], "unchanged": []}
    if not isinstance(verdicts, list):
        return groups, 0

    dropped = 0
    for entry in verdicts:
        statistic = entry.get("statistic") if isinstance(entry, dict) else None
        verdict = entry.get("verdict") if isinstance(entry, dict) else None
        if isinstance(statistic, str) and verdict in groups and statistic in deltas_by_statistic:
            groups[verdict].append(statistic)
        else:
            dropped += 1
    return groups, dropped


def _parse_delta_argument(delta_json: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse the ``stylometric_delta`` response argument.

    Returns the parsed dict (``None`` when the argument is unusable) plus
    notes describing anything dropped, so malformed input degrades into
    rendered guidance instead of an error.
    """
    try:
        parsed: Any = json.loads(delta_json)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return None, [f"the `delta` argument could not be parsed as JSON ({exc})"]
    if not isinstance(parsed, dict):
        return None, [
            "the `delta` argument was not a JSON object; pass the full response dict `stylometric_delta` returned"
        ]
    return parsed, []


def _verification_unavailable_lines(failure: str, remediation: str) -> list[str]:
    """The could-not-run block: what failed and what to do about it."""
    return ["", "## Verification could not run", failure, remediation]


def _verdict_lines(parsed: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Render the verdict sections of a successful delta response.

    Returns the rendered lines plus notes for anything in the response that
    was dropped as malformed.
    """
    deltas_by_statistic, dropped_deltas = _well_formed_deltas(parsed.get("deltas"))
    groups, dropped_verdicts = _group_verdicts(parsed.get("verdict"), deltas_by_statistic)

    notes: list[str] = []
    if dropped_deltas:
        notes.append(f"{dropped_deltas} deltas entries were malformed and were dropped")
    if dropped_verdicts:
        notes.append(f"{dropped_verdicts} verdict entries had no renderable delta entry and were dropped")

    improved, regressed, unchanged = (groups[name] for name in ("improved", "regressed", "unchanged"))
    baseline_used = parsed.get("baseline_used") or "the configured baseline"

    lines = [
        "",
        f"## Verdict (baseline: {baseline_used})",
        f"{len(improved)} improved, {len(regressed)} regressed, {len(unchanged)} unchanged.",
        f"Improved: {', '.join(improved) if improved else 'nothing'}",
        f"Regressed: {', '.join(regressed) if regressed else 'nothing'}",
    ]

    if regressed:
        lines.extend(["", "## Regressed — what the revision broke; fix these first"])
        lines.extend(_delta_movement_lines(regressed, deltas_by_statistic))
    if improved:
        lines.extend(["", "## Improved — the revision moved these toward the baseline"])
        lines.extend(_delta_movement_lines(improved, deltas_by_statistic))
    if unchanged:
        lines.extend(
            [
                "",
                "## Unchanged",
                textwrap.fill(", ".join(unchanged), width=_WRAP_WIDTH, initial_indent="  ", subsequent_indent="  "),
            ]
        )

    findings, dropped_findings = _well_formed_findings(parsed.get("findings"))
    if dropped_findings:
        notes.append(f"{dropped_findings} findings entries were malformed and were dropped")
    lines.extend(["", "## Findings on the revised text"])
    if findings:
        lines.extend(_numbered_findings(findings))
    else:
        lines.append(
            "None — the revised text is clean against this baseline. Make only the changes the author intends."
        )

    return lines, notes


def render_verify_revision(delta_json: str) -> str:
    """Render a ``stylometric_delta`` response as a revision verdict.

    Groups the per-statistic verdicts into improved/regressed/unchanged, shows
    the z-score movement behind each verdict, and lists the findings for the
    revised text. Malformed input degrades into guidance (a NOTE) instead of
    an error, matching the other prompt renderers.
    """
    parsed, notes = _parse_delta_argument(delta_json)

    lines: list[str] = [
        "You are verifying a revision. The `stylometric_delta` tool profiled a draft (text_a) and its revision "
        "(text_b) against the same baseline; the verdict below says what the revision actually moved.",
        "How to read it: a z-score of 0 sits exactly in the baseline's range, so a statistic improves when the "
        "revision moved its z-score closer to zero and regresses when the movement pushed it further out.",
    ]

    if parsed is None:
        lines.extend(
            _verification_unavailable_lines(
                "No delta response was available.",
                "Run the `stylometric_delta` tool on the draft and revised texts, then pass its full response "
                "as this prompt's `delta` argument.",
            )
        )
    elif "error" in parsed:
        lines.extend(
            _verification_unavailable_lines(
                f"The delta analysis failed: {parsed.get('error')}",
                "Fix the inputs the error names (usually an empty text or a missing baseline) and re-run "
                "`stylometric_delta`, then pass its response here.",
            )
        )
    else:
        verdict_lines, verdict_notes = _verdict_lines(parsed)
        lines.extend(verdict_lines)
        notes.extend(verdict_notes)

    lines.extend(
        [
            "",
            "## Verification protocol",
            "1. Fix the regressed statistics first — they are what the revision broke; leave the improved ones alone.",
            "2. Re-run `stylometric_delta` after the next pass and render this prompt again with the fresh response.",
            "3. Stop when the regressions clear or plateau — do not polish past the evidence.",
        ]
    )

    for note in notes:
        lines.extend(["", f"NOTE: {note}"])

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
