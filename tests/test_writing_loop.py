"""Tests for the reference writing-loop example.

The pure helpers are tested directly; ``run_loop`` is tested against a stubbed
MCP client (no server process, no spaCy/torch load). The live end-to-end run is
proven separately by the committed transcript; unit tests keep the loop's wiring
and exit-code contract pinned without paying the model-load cost per test.
"""

import asyncio
from types import SimpleNamespace

import pytest

import examples.writing_loop as writing_loop


def make_delta(verdicts: list[dict[str, object]]) -> dict[str, object]:
    """Build a stylometric_delta response from verdict entries."""
    deltas = []
    for entry in verdicts:
        z_a = float(entry["z_a"])
        z_b = float(entry["z_b"])
        deltas.append(
            {
                "statistic": entry["statistic"],
                "z_a": z_a,
                "z_b": z_b,
                "delta": z_b - z_a,
                "direction": "decreased" if z_b < z_a else "increased",
            }
        )
    return {"verdict": verdicts, "deltas": deltas, "baseline_used": "brown_corpus"}


def improved_entry(statistic: str, z_a: float, z_b: float) -> dict[str, object]:
    return {"statistic": statistic, "verdict": "improved", "z_a": z_a, "z_b": z_b}


def regressed_entry(statistic: str, z_a: float, z_b: float) -> dict[str, object]:
    return {"statistic": statistic, "verdict": "regressed", "z_a": z_a, "z_b": z_b}


class FakeMessage:
    def __init__(self, text: str):
        self.content = SimpleNamespace(type="text", text=text)


class FakePromptResult:
    def __init__(self, text: str, count: int = 1):
        self.messages = [FakeMessage(text) for _ in range(count)]


class FakeToolResult:
    def __init__(self, data: object, is_error: bool = False):
        self.data = data
        self.is_error = is_error


class FakeClient:
    """Minimal stand-in for fastmcp.Client: canned tools, tool results, prompts."""

    def __init__(
        self,
        transport: object,
        tool_names: list[str],
        tool_results: dict[str, object],
        prompt_text: str,
    ):
        self._tool_names = tool_names
        self._tool_results = tool_results
        self._prompt_text = prompt_text
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def list_tools(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=name) for name in self._tool_names]

    async def call_tool(self, name: str, arguments: dict[str, object]) -> FakeToolResult:
        self.calls.append((name, arguments))
        return FakeToolResult(self._tool_results[name])

    async def get_prompt(self, name: str, arguments: dict[str, str]) -> FakePromptResult:
        return FakePromptResult(self._prompt_text)


def install_fake_client(monkeypatch: pytest.MonkeyPatch, fake: FakeClient) -> None:
    monkeypatch.setattr(writing_loop, "StdioTransport", lambda **_kwargs: object())
    monkeypatch.setattr(writing_loop, "Client", lambda _transport: fake)


ANALYSIS = {
    "baseline_used": "brown_corpus",
    "findings": [
        {
            "rule": "comma_ratio_outlier",
            "location": "document",
            "message": "comma_ratio sits 5.25 standard deviations from the baseline mean",
            "fix_hint": "vary the punctuation",
        }
    ],
}

ALL_IMPROVED_DELTA = make_delta(
    [
        improved_entry("ttr", 2.0, 1.0),
        improved_entry("pos_pron", -2.0, -0.5),
    ]
)

REGRESSED_DELTA = make_delta(
    [
        improved_entry("ttr", 2.0, 1.0),
        regressed_entry("pos_det", 3.0, 4.0),
    ]
)


class TestRegressedStatistics:
    def test_returns_only_regressed_entries(self):
        regressed = writing_loop.regressed_statistics(REGRESSED_DELTA)
        assert [entry["statistic"] for entry in regressed] == ["pos_det"]

    def test_empty_when_nothing_regressed(self):
        assert writing_loop.regressed_statistics(ALL_IMPROVED_DELTA) == []

    def test_missing_verdict_key_is_safe(self):
        assert writing_loop.regressed_statistics({}) == []


class TestVerdictSummary:
    def test_counts_each_outcome(self):
        counts = writing_loop.verdict_summary(REGRESSED_DELTA)
        assert counts == {"improved": 1, "regressed": 1, "unchanged": 0}

    def test_ignores_unknown_outcomes(self):
        delta = {"verdict": [{"statistic": "ttr", "verdict": "improved"}, {"statistic": "x", "verdict": "mystery"}]}
        assert writing_loop.verdict_summary(delta) == {"improved": 1, "regressed": 0, "unchanged": 0}


class TestFormatHelpers:
    def test_finding_lines_are_numbered_with_rule_and_hint(self):
        lines = writing_loop.format_finding_lines(ANALYSIS["findings"])
        assert lines == [
            "  1. [comma_ratio_outlier @ document] "
            "comma_ratio sits 5.25 standard deviations from the baseline mean -> vary the punctuation"
        ]

    def test_delta_lines_carry_scores_and_verdicts(self):
        lines = writing_loop.format_delta_lines(ALL_IMPROVED_DELTA)
        assert lines == [
            "  ttr: z 2.00 -> 1.00 (delta -1.00, decreased) => improved",
            "  pos_pron: z -2.00 -> -0.50 (delta +1.50, increased) => improved",
        ]

    def test_banner_carries_seconds(self):
        assert writing_loop.banner("STAGE 1", 1.25) == "---- STAGE 1 (1.2s) ----"


class TestParseArgs:
    def test_defaults_to_happy_path(self):
        assert writing_loop.parse_args([]).demo_failure is False

    def test_demo_failure_flag(self):
        assert writing_loop.parse_args(["--demo-failure"]).demo_failure is True


class TestRunLoop:
    def _fake(self, tool_results: dict[str, object], tool_names: list[str] | None = None) -> FakeClient:
        return FakeClient(
            transport=object(),
            tool_names=tool_names
            if tool_names is not None
            else ["stylometric_analysis", "stylometric_delta", "word_count"],
            tool_results=tool_results,
            prompt_text="REVISION BRIEF / VERDICT PROMPT",
        )

    def test_happy_path_exits_zero_and_prints_the_loop(self, monkeypatch, capsys):
        fake = self._fake({"stylometric_analysis": ANALYSIS, "stylometric_delta": ALL_IMPROVED_DELTA})
        install_fake_client(monkeypatch, fake)

        exit_code = asyncio.run(writing_loop.run_loop(writing_loop.DRAFT, writing_loop.REVISED))
        out = capsys.readouterr().out

        assert exit_code == writing_loop.EXIT_OK
        assert "STAGE 1 - ANALYZE" in out
        assert "STAGE 2 - REVISE" in out
        assert "STAGE 3 - VERIFY" in out
        assert "verification clear" in out
        assert [(name, set(arguments)) for name, arguments in fake.calls] == [
            ("stylometric_analysis", {"text"}),
            ("stylometric_delta", {"text_a", "text_b"}),
        ]

    def test_regressed_delta_exits_one(self, monkeypatch, capsys):
        fake = self._fake({"stylometric_analysis": ANALYSIS, "stylometric_delta": REGRESSED_DELTA})
        install_fake_client(monkeypatch, fake)

        exit_code = asyncio.run(writing_loop.run_loop(writing_loop.DRAFT, writing_loop.REGRESSED))
        out = capsys.readouterr().out

        assert exit_code == writing_loop.EXIT_REGRESSION
        assert "REGRESSION CAUGHT" in out
        assert "pos_det" in out

    def test_missing_required_tools_exits_two(self, monkeypatch, capsys):
        fake = self._fake({}, tool_names=["word_count"])
        install_fake_client(monkeypatch, fake)

        exit_code = asyncio.run(writing_loop.run_loop(writing_loop.DRAFT, writing_loop.REVISED))

        assert exit_code == writing_loop.EXIT_LOOP_FAILURE
        assert fake.calls == []
        capsys.readouterr()

    def test_analysis_error_envelope_exits_two(self, monkeypatch, capsys):
        fake = self._fake({"stylometric_analysis": {"error": "baseline unavailable"}})
        install_fake_client(monkeypatch, fake)

        exit_code = asyncio.run(writing_loop.run_loop(writing_loop.DRAFT, writing_loop.REVISED))

        assert exit_code == writing_loop.EXIT_LOOP_FAILURE
        capsys.readouterr()

    def test_delta_error_envelope_exits_two(self, monkeypatch, capsys):
        fake = self._fake({"stylometric_analysis": ANALYSIS, "stylometric_delta": {"error": "no baseline"}})
        install_fake_client(monkeypatch, fake)

        exit_code = asyncio.run(writing_loop.run_loop(writing_loop.DRAFT, writing_loop.REVISED))

        assert exit_code == writing_loop.EXIT_LOOP_FAILURE
        capsys.readouterr()

    def test_prompt_with_unexpected_shape_exits_two(self, monkeypatch, capsys):
        class BrokenPromptClient(FakeClient):
            async def get_prompt(self, name: str, arguments: dict[str, str]) -> FakePromptResult:
                return FakePromptResult("text", count=2)

        fake = BrokenPromptClient(
            transport=object(),
            tool_names=["stylometric_analysis", "stylometric_delta"],
            tool_results={"stylometric_analysis": ANALYSIS, "stylometric_delta": ALL_IMPROVED_DELTA},
            prompt_text="unused",
        )
        install_fake_client(monkeypatch, fake)

        # Prompt-shape failures propagate out of run_loop; main() converts them to exit 2
        # (pinned by test_loop_exception_becomes_exit_two).
        with pytest.raises(RuntimeError, match="prompt guided_revision returned"):
            asyncio.run(writing_loop.run_loop(writing_loop.DRAFT, writing_loop.REVISED))
        capsys.readouterr()


class TestMainRouting:
    def _capture_run_loop(self, monkeypatch: pytest.MonkeyPatch, exit_code: int) -> list[tuple[str, str]]:
        captured: list[tuple[str, str]] = []

        async def fake_run_loop(draft: str, revised: str, repo_root: object = writing_loop.REPO_ROOT) -> int:
            captured.append((draft, revised))
            return exit_code

        monkeypatch.setattr(writing_loop, "run_loop", fake_run_loop)
        return captured

    def test_happy_path_routes_the_verified_revision(self, monkeypatch):
        captured = self._capture_run_loop(monkeypatch, writing_loop.EXIT_OK)
        assert writing_loop.main([]) == writing_loop.EXIT_OK
        assert captured == [(writing_loop.DRAFT, writing_loop.REVISED)]

    def test_demo_failure_routes_the_regressed_revision(self, monkeypatch):
        captured = self._capture_run_loop(monkeypatch, writing_loop.EXIT_REGRESSION)
        assert writing_loop.main(["--demo-failure"]) == writing_loop.EXIT_REGRESSION
        assert captured == [(writing_loop.DRAFT, writing_loop.REGRESSED)]

    def test_loop_exception_becomes_exit_two(self, monkeypatch, capsys):
        def failing_run_loop(*_args: object, **_kwargs: object) -> int:
            raise RuntimeError("server never started")

        monkeypatch.setattr(writing_loop, "run_loop", failing_run_loop)
        assert writing_loop.main([]) == writing_loop.EXIT_LOOP_FAILURE
        capsys.readouterr()


class TestCompanionTexts:
    """The three companion texts are a verified set; pin their relationships."""

    def test_companion_texts_differ(self):
        assert writing_loop.REVISED != writing_loop.REGRESSED
        assert writing_loop.DRAFT != writing_loop.REVISED
        assert writing_loop.DRAFT != writing_loop.REGRESSED

    def test_companion_texts_are_prose(self):
        for text in (writing_loop.DRAFT, writing_loop.REVISED, writing_loop.REGRESSED):
            assert len(text.split()) > 40
            assert text.endswith(".")
