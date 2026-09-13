"""Reference writing loop: analyze -> revise -> verify, against the live stdio server.

This is the one-file proof of what writing-tools-mcp is for. It starts the real
server (stdio transport, no port), then walks a draft through the loop the
specification describes:

  1. ANALYZE  - ``stylometric_analysis`` on the draft (findings ride along)
  2. REVISE   - ``guided_revision`` renders the impact-ordered revision brief;
                the revised draft is what a reviser following that brief returns
  3. VERIFY   - ``stylometric_delta`` proves what the revision moved, and the
                ``verify_revision`` prompt renders it as a verdict

The transcript goes to stdout; all logging goes to stderr (stdout of the server
process is the MCP protocol stream and is never mixed into the transcript).

Exit codes:
  0 - the loop completed and verification is clear (no regressed statistic)
  1 - verification caught a regression the loop should have caught
  2 - the loop could not run (server did not start, tool call failed, or the
      analysis returned an error envelope)

Usage (from a checkout with uv installed; the first run resolves the locked
environment and downloads the GPT-2 weights for the detector):

  uv run python examples/writing_loop.py                # happy path, exits 0
  uv run python examples/writing_loop.py --demo-failure # replays a regression, exits 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

LOGGER = logging.getLogger("writing_loop")

REPO_ROOT = Path(__file__).resolve().parent.parent

# A deliberately weak draft: passive constructions, inflated prepositional
# padding ("during the course of the review of the process"), uniform sentence
# length, and hedged wording - exactly what stylometric_analysis should flag.
DRAFT = (
    "The quarterly report for the quarter was prepared by the analytics team. Several data gaps "
    "were identified during the course of the review of the process. It was decided by the "
    "leadership that the rollout would be postponed for the duration of the quarter. Concerns "
    "about the migration timeline were raised by the engineering staff of the organization. A "
    "plan for additional testing was proposed by the subcommittee. The findings were documented "
    "in a shared repository. Follow-up actions are being tracked by the program manager."
)

# The revision a reviser returns after working through the guided_revision brief:
# active voice, varied sentence lengths, hedging and padding removed.
REVISED = (
    "The analytics team wrote the quarterly report for the quarter, and they found several data "
    "gaps in it while reviewing the process. They raised concerns about the timeline for the "
    "migration, so leadership pushed the rollout back for the quarter. The subcommittee proposed "
    "more testing for the gaps, and they wrote them down for the team for the quarter. The "
    "program manager tracks the follow-up actions. Everyone expects a decision about them next "
    "month."
)

# A "revision" that makes the draft worse - longer, uniform, more passive - used
# by --demo-failure to show the loop catching a regression with a non-zero exit.
REGRESSED = (
    "It was determined by the analytics team of the organization that the preparation of the "
    "quarterly report for the quarter was completed by the team. The identification of the data "
    "gaps was carried out during the course of the review of the process by the staff. The "
    "postponement of the rollout for the duration of the quarter was decided by the leadership "
    "group of the organization. The raising of concerns about the timeline of the migration was "
    "done by the engineering staff of the organization for the quarter."
)

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_LOOP_FAILURE = 2


def regressed_statistics(delta: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the delta verdict entries the verification flags as regressed.

    A revision regressed a statistic when it moved the revised text's z-score
    further from the baseline's range than the draft's (|z_b| > |z_a|).
    """
    return [entry for entry in delta.get("verdict", []) if entry.get("verdict") == "regressed"]


def verdict_summary(delta: Mapping[str, Any]) -> dict[str, int]:
    """Count verdict entries by outcome (improved / regressed / unchanged)."""
    counts = {"improved": 0, "regressed": 0, "unchanged": 0}
    for entry in delta.get("verdict", []):
        outcome = entry.get("verdict")
        if outcome in counts:
            counts[outcome] += 1
    return counts


def format_finding_lines(findings: list[Mapping[str, Any]]) -> list[str]:
    """Render one line per finding: rule, location, message, fix hint."""
    lines = []
    for index, finding in enumerate(findings, start=1):
        lines.append(
            f"  {index}. [{finding.get('rule')} @ {finding.get('location')}] "
            f"{finding.get('message')} -> {finding.get('fix_hint')}"
        )
    return lines


def format_delta_lines(delta: Mapping[str, Any]) -> list[str]:
    """Render the per-statistic movement the delta tool reported."""
    verdicts = {entry.get("statistic"): entry.get("verdict") for entry in delta.get("verdict", [])}
    lines = []
    for movement in delta.get("deltas", []):
        statistic = movement.get("statistic")
        lines.append(
            f"  {statistic}: z {movement.get('z_a'):.2f} -> {movement.get('z_b'):.2f} "
            f"(delta {movement.get('delta'):+.2f}, {movement.get('direction')}) "
            f"=> {verdicts.get(statistic)}"
        )
    return lines


def banner(title: str, seconds: float) -> str:
    """Stage banner with the stage's wall-clock cost."""
    rule = "-" * 4
    return f"{rule} {title} ({seconds:.1f}s) {rule}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the loop's command line."""
    parser = argparse.ArgumentParser(
        description="Run the analyze -> revise -> verify writing loop against the live stdio server."
    )
    parser.add_argument(
        "--demo-failure",
        action="store_true",
        help="verify with a deliberately regressed revision to demonstrate the non-zero exit",
    )
    return parser.parse_args(argv)


async def run_loop(draft: str, revised: str, repo_root: Path = REPO_ROOT) -> int:
    """Walk one draft through the loop; return the process exit code."""
    command = ["uv", "run", "writing-tools-mcp"]
    LOGGER.info("starting server: %s (cwd=%s)", " ".join(command), repo_root)
    transport = StdioTransport(command=command[0], args=command[1:], cwd=str(repo_root))

    async with Client(transport) as client:
        tools = {tool.name for tool in await client.list_tools()}
        required = {"stylometric_analysis", "stylometric_delta"}
        if not required.issubset(tools):
            LOGGER.error("server is missing required tools %s (listed: %s)", sorted(required - tools), sorted(tools))
            return EXIT_LOOP_FAILURE

        print(f"WRITING LOOP - analyze -> revise -> verify (server: {' '.join(command)}, {len(tools)} tools)\n")
        print(f"DRAFT ({len(draft.split())} words):\n{draft}\n")

        # Stage 1 - ANALYZE: the draft is measured; findings ride along.
        started = time.monotonic()
        analysis = await call_tool(client, "stylometric_analysis", {"text": draft})
        analyze_seconds = time.monotonic() - started
        if "error" in analysis:
            LOGGER.error("stylometric_analysis failed: %s", analysis["error"])
            return EXIT_LOOP_FAILURE
        findings = analysis.get("findings", [])
        print(banner("STAGE 1 - ANALYZE: stylometric_analysis", analyze_seconds))
        print(f"  baseline_used: {analysis.get('baseline_used')}")
        print(f"  findings ({len(findings)}):")
        print("\n".join(format_finding_lines(findings)))

        # Stage 2 - REVISE: the guided_revision prompt turns the findings into
        # an impact-ordered brief; the revised draft is what the reviser returns.
        started = time.monotonic()
        brief = await get_prompt(client, "guided_revision", {"document": draft, "findings": json.dumps(findings)})
        revise_seconds = time.monotonic() - started
        print(banner("STAGE 2 - REVISE: guided_revision brief", revise_seconds))
        print(brief)

        # Stage 3 - VERIFY: stylometric_delta proves what the revision moved;
        # verify_revision renders the same delta as an actionable verdict.
        started = time.monotonic()
        delta = await call_tool(client, "stylometric_delta", {"text_a": draft, "text_b": revised})
        verify_seconds = time.monotonic() - started
        if "error" in delta:
            LOGGER.error("stylometric_delta failed: %s", delta["error"])
            return EXIT_LOOP_FAILURE
        print(banner("STAGE 3 - VERIFY: revised draft vs draft", verify_seconds))
        print(f"  REVISED ({len(revised.split())} words):\n{revised}\n")
        print(f"  baseline_used: {delta.get('baseline_used')}")
        counts = verdict_summary(delta)
        print("  verdicts: {improved} improved / {unchanged} unchanged / {regressed} regressed".format(**counts))
        print("\n".join(format_delta_lines(delta)))
        verdict = await get_prompt(client, "verify_revision", {"delta": json.dumps(delta)})
        print("\n  verify_revision prompt verdict:")
        for line in verdict.splitlines():
            print(f"    {line}")

    regressed = regressed_statistics(delta)
    print("\nRESULT: ", end="")
    if regressed:
        names = ", ".join(str(entry.get("statistic")) for entry in regressed)
        print(f"REGRESSION CAUGHT - the revision moved {names} away from the baseline. Exit {EXIT_REGRESSION}.")
        return EXIT_REGRESSION
    print(
        f"verification clear - {counts['improved']} statistic(s) improved, none regressed "
        f"against {delta.get('baseline_used')}. Exit {EXIT_OK}."
    )
    return EXIT_OK


async def call_tool(client: Client, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call an analysis tool and return its structured result (or error envelope)."""
    result = await client.call_tool(name, arguments)
    if result.is_error:
        LOGGER.error("%s failed: %s", name, result.content)
        raise RuntimeError(f"{name} failed")
    data = result.data
    if not isinstance(data, dict):
        raise RuntimeError(f"{name} returned {type(data).__name__}, expected a dict")
    return data


async def get_prompt(client: Client, name: str, arguments: dict[str, str]) -> str:
    """Render a registered prompt and return its single text message."""
    result = await client.get_prompt(name, arguments=arguments)
    messages = list(result.messages)
    if len(messages) != 1:
        raise RuntimeError(f"prompt {name} returned {len(messages)} messages, expected 1")
    content = messages[0].content
    if content.type != "text":
        raise RuntimeError(f"prompt {name} returned {content.type} content, expected text")
    return content.text


def main(argv: list[str] | None = None) -> int:
    """Entry point: configure stderr logging, run the loop, exit with its verdict."""
    args = parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("WRITING_LOOP_LOGLEVEL", "WARNING").upper(),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    revised = REGRESSED if args.demo_failure else REVISED
    try:
        return asyncio.run(run_loop(DRAFT, revised))
    except Exception as failure:  # the loop reports any failure (incl. task groups) as exit 2
        LOGGER.error("loop failed: %s", failure)
        return EXIT_LOOP_FAILURE


if __name__ == "__main__":
    sys.exit(main())
