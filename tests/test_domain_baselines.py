"""Shipped domain baselines: essays, technical_docs, scientific_prose.

Guarantees for each baseline shipped under server/data/baselines/: it loads by
name through BaselineManager's search path, drives stylometric_analysis
end-to-end with per-feature z-scores, works as stylometry.default_baseline,
carries the length-robust statistics set (richer than the 9-statistic Brown
Corpus baseline), and ships inside the built wheel.

Corpus assembly lives in scripts/fetch_domain_corpora.py; per-baseline
provenance, licensing, and AI-indicator coverage are documented in README.md
("Domain baselines").
"""

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from server.analyzers.ai_detection import AIDetectionAnalyzer
from server.stylometry import BaselineManager

REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAIN_BASELINES = ("essays", "technical_docs", "scientific_prose")

#: Statistics every shipped domain baseline must carry. With pos_ratios,
#: pos_bigram_ratios, and function_word_freqs these support 9 of the 12 AI
#: indicators in server/stylometry/statistical.py; the other three
#: (low_ttr, low_hapax, function_word_anomaly) need the length-confounded
#: ttr / hapax_legomena_rate / function_word_ratio statistics that the robust
#: builder deliberately omits (see server/stylometry/corpus_baseline.py).
ROBUST_STATISTICS = frozenset(
    {
        "avg_sentence_len",
        "sentence_len_std",
        "fog",
        "kincaid",
        "mtld",
        "mattr",
        "smog",
        "coleman_liau",
        "ari",
        "dale_chall",
        "mean_dependency_distance",
        "subordinate_clause_ratio",
        "mtld_lemma",
        "mean_word_frequency",
        "word_len_std",
        "lexical_density",
        "semicolon_ratio",
        "em_dash_ratio",
        "ellipsis_ratio",
        "exclamation_ratio",
        "parenthetical_rate",
        "hedge_rate",
        "booster_rate",
        "pos_ratios",
        "pos_bigram_ratios",
        "function_word_freqs",
    }
)

#: Length-confounded statistics the robust builder excludes; the shipped
#: baselines are built with the default (robust) feature set, not --all-features.
LENGTH_CONFOUNDED_STATISTICS = ("ttr", "hapax_legomena_rate", "function_word_ratio")

ANALYSIS_TEXT = (
    "The committee reviewed the proposal twice before reaching a decision. "
    "Several members argued that the schedule was unrealistic, and one asked "
    "whether the budget had been approved at all. After a long pause, the "
    "chair suggested that everyone sleep on it. The next morning, three "
    "written objections were waiting in her inbox, each more detailed than "
    "the last. Still, the vote passed, and the work began on Monday as "
    "planned. Nobody expected the first milestone to slip, yet it did."
)


def _statistics_of(name: str) -> dict:
    return BaselineManager({}).load_baseline(name)["statistics"]


class TestShippedDomainBaselines:
    """Every shipped domain baseline loads by name with sane corpus_info."""

    @pytest.mark.parametrize("name", DOMAIN_BASELINES)
    def test_loads_by_name(self, name):
        baseline = BaselineManager({}).load_baseline(name)

        assert baseline["corpus_info"]["name"] == name
        assert baseline["corpus_info"]["sample_size"] >= 2
        # Provenance summary (sources, license posture) is stored on the baseline.
        assert "description" in baseline["corpus_info"]
        assert "Project Gutenberg" in baseline["corpus_info"]["description"]

    @pytest.mark.parametrize("name", DOMAIN_BASELINES)
    def test_statistics_cover_the_robust_feature_set(self, name):
        missing = ROBUST_STATISTICS - _statistics_of(name).keys()
        assert not missing, f"{name} is missing robust statistics: {sorted(missing)}"

    @pytest.mark.parametrize("name", DOMAIN_BASELINES)
    def test_robust_baselines_omit_length_confounded_statistics(self, name):
        stats = _statistics_of(name)
        for key in LENGTH_CONFOUNDED_STATISTICS:
            assert key not in stats, (
                f"{name} carries {key}; the shipped baselines use the robust "
                "builder defaults, so a change here is a rebuild, not an edit"
            )

    def test_domain_baselines_are_richer_than_brown_corpus(self):
        brown = BaselineManager({}).load_baseline("brown_corpus")["statistics"]

        for name in DOMAIN_BASELINES:
            assert len(_statistics_of(name)) > len(brown), (
                f"{name} should exceed brown_corpus's {len(brown)} statistics"
            )


class TestDomainBaselineAnalysis:
    """stylometric_analysis returns z-scores against each domain baseline."""

    @pytest.mark.parametrize("name", DOMAIN_BASELINES)
    def test_z_scores_against_each_baseline(self, nlp, gpt2_manager, name):
        analyzer = AIDetectionAnalyzer(nlp, gpt2_manager, {})
        result = analyzer.stylometric_analysis(ANALYSIS_TEXT, baseline=name)

        assert "error" not in result
        assert result["baseline_used"] == name
        assert result["config"]["baseline_info"]["name"] == name
        numeric_z = [v for v in result["z_scores"].values() if isinstance(v, (int, float))]
        assert len(numeric_z) >= 10  # the robust set drives the z-score space
        assert result["sentence_analysis"]

    @pytest.mark.parametrize("name", DOMAIN_BASELINES)
    def test_works_as_configured_default_baseline(self, nlp, gpt2_manager, name):
        analyzer = AIDetectionAnalyzer(nlp, gpt2_manager, {"stylometry": {"default_baseline": name}})
        result = analyzer.stylometric_analysis(ANALYSIS_TEXT)

        assert "error" not in result
        assert result["baseline_used"] == name

    def test_indicator_machinery_runs_on_domain_baseline(self, nlp, gpt2_manager):
        analyzer = AIDetectionAnalyzer(nlp, gpt2_manager, {})
        result = analyzer.stylometric_analysis(ANALYSIS_TEXT, baseline="essays")

        assert "error" not in result
        flags = result["flags"]
        assert isinstance(flags["ai_indicators"], list)
        # The three length-confounded indicators cannot fire on robust statistics.
        assert not set(flags["ai_indicators"]) & {"low_ttr", "low_hapax", "function_word_anomaly"}


class TestWheelBundling:
    """The built wheel ships every domain baseline JSON."""

    def _build_wheel(self, tmp_path: Path) -> Path:
        hatchling = Path(sys.executable).parent / "hatchling"
        if not hatchling.exists():  # pragma: no cover - dev extra installs it
            pytest.skip("hatchling not installed; add the dev extra")
        result = subprocess.run(
            [str(hatchling), "build", "-t", "wheel", "-d", str(tmp_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"hatchling build failed:\n{result.stderr}"
        wheels = list(tmp_path.glob("*.whl"))
        assert len(wheels) == 1
        return wheels[0]

    def test_domain_baselines_ship_in_wheel(self, tmp_path):
        wheel = self._build_wheel(tmp_path)
        bundled = set(zipfile.ZipFile(wheel).namelist())

        for name in DOMAIN_BASELINES:
            assert f"server/data/baselines/{name}.json" in bundled
        assert "server/data/baselines/brown_corpus.json" in bundled
