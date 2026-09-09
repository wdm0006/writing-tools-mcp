"""
Tests for the live baseline configuration.

These pin the W3 wiring: ``stylometry.custom_baselines_dir`` is the actual
save/load root for custom baselines, ``stylometry.default_baseline`` selects
the baseline ``stylometric_analysis`` measures against when a call omits one,
every response honestly names the baseline used, and legacy configs carrying
the removed ``stylometry.features`` key degrade to defaults with a warning.
"""

import json

import pytest

from server.analyzers import AIDetectionAnalyzer
from server.stylometry import BaselineManager, resolve_baseline_name
from server.stylometry import baselines as baselines_module

TEST_TEXT = (
    "The committee approved the revised budget yesterday. "
    "The council rejected the amended proposal quickly. "
    "The board reviewed the updated schedule carefully."
)

CUSTOM_BASELINE = {
    "corpus_info": {"name": "Config Test Corpus", "description": "Baseline built for config tests", "language": "en"},
    "statistics": {
        "avg_sentence_len": {"mean": 8.0, "std": 2.0},
        "ttr": {"mean": 0.7, "std": 0.05},
        "hapax_legomena_rate": {"mean": 0.6, "std": 0.05},
        "avg_word_len": {"mean": 4.0, "std": 0.5},
    },
}


@pytest.fixture
def package_baselines_dir(tmp_path, monkeypatch):
    """Point the module BASELINES_DIR at a temp tree so tests never write into the repo."""
    data_dir = tmp_path / "data" / "baselines"
    (data_dir / "custom_baselines").mkdir(parents=True)
    monkeypatch.setattr(baselines_module, "BASELINES_DIR", data_dir)
    return data_dir


class TestConfigDrivenCustomDir:
    """custom_baselines_dir is the save/load root when it is set."""

    def test_save_load_roundtrip_under_configured_dir(self, nlp, gpt2_manager, tmp_path):
        """A saved baseline lands in the configured dir, loads by name, and the analyzer uses it."""
        configured_dir = tmp_path / "my_baselines"
        config = {"stylometry": {"custom_baselines_dir": str(configured_dir)}}
        manager = BaselineManager(config)

        assert manager.save_baseline("config_roundtrip", CUSTOM_BASELINE) is True
        assert (configured_dir / "config_roundtrip.json").is_file()
        assert not (
            baselines_module.BASELINES_DIR / baselines_module.CUSTOM_BASELINES_SUBDIR / "config_roundtrip.json"
        ).exists()

        # A fresh manager (empty in-memory cache) still finds it by name alone.
        fresh = BaselineManager(config)
        assert fresh.load_baseline("config_roundtrip") == CUSTOM_BASELINE

        # The config-wired analyzer measures against it when asked.
        analyzer = AIDetectionAnalyzer(nlp, gpt2_manager, config)
        result = analyzer.stylometric_analysis(TEST_TEXT, baseline="config_roundtrip")

        assert "error" not in result
        assert result["baseline_used"] == "config_roundtrip"
        assert result["config"]["baseline"] == "config_roundtrip"
        assert result["features"]

    def test_relative_dir_resolves_against_working_directory(self, tmp_path, monkeypatch):
        """A relative custom_baselines_dir anchors to the cwd, like .mcp-config.yaml itself."""
        monkeypatch.chdir(tmp_path)

        manager = BaselineManager({"stylometry": {"custom_baselines_dir": "relative_baselines"}})

        assert manager.custom_baselines_dir == tmp_path / "relative_baselines"

    def test_unset_dir_keeps_package_root(self, package_baselines_dir):
        """Without a configured dir, saves still go to the package custom subdir."""
        manager = BaselineManager()

        assert manager.custom_baselines_dir is None
        assert manager.save_baseline("legacy_spot", CUSTOM_BASELINE) is True
        assert (package_baselines_dir / baselines_module.CUSTOM_BASELINES_SUBDIR / "legacy_spot.json").is_file()

    @pytest.mark.parametrize("bad_value", [None, "", "   ", 123])
    def test_unusable_dir_values_fall_back_to_package_root(self, bad_value):
        """A missing, empty, or non-string dir keeps the built-in package root."""
        manager = BaselineManager({"stylometry": {"custom_baselines_dir": bad_value}})

        assert manager.custom_baselines_dir is None

    def test_configured_dir_shadows_same_named_package_baseline(self, package_baselines_dir, tmp_path):
        """A configured baseline wins over a same-named file in the package custom subdir."""
        shadowed = {**CUSTOM_BASELINE, "corpus_info": {"name": "Shadowed"}}
        (package_baselines_dir / baselines_module.CUSTOM_BASELINES_SUBDIR / "shadow_me.json").write_text(
            json.dumps(shadowed), encoding="utf-8"
        )

        config_dir = tmp_path / "user_baselines"
        override = {**CUSTOM_BASELINE, "corpus_info": {"name": "Override"}}
        config = {"stylometry": {"custom_baselines_dir": str(config_dir)}}
        BaselineManager(config).save_baseline("shadow_me", override)

        loaded = BaselineManager(config).load_baseline("shadow_me")
        assert loaded["corpus_info"]["name"] == "Override"

    def test_listing_includes_configured_dir_baselines(self, tmp_path):
        """list_available_baselines reports names from the configured dir alongside built-ins."""
        config = {"stylometry": {"custom_baselines_dir": str(tmp_path)}}
        BaselineManager(config).save_baseline("listed_custom", CUSTOM_BASELINE)

        available = BaselineManager(config).list_available_baselines()

        assert available["listed_custom"] == "Custom baseline"
        assert available["brown_corpus"].startswith("Human writing baseline")


class TestConfiguredDefaultBaseline:
    """stylometry.default_baseline selects the baseline when a call omits one."""

    def test_argument_less_call_uses_configured_default(self, nlp, gpt2_manager, tmp_path):
        """With default_baseline set, a call without a baseline measures against it."""
        config = {
            "stylometry": {"default_baseline": "my_default", "custom_baselines_dir": str(tmp_path / "defs")},
        }
        BaselineManager(config).save_baseline("my_default", CUSTOM_BASELINE)

        analyzer = AIDetectionAnalyzer(nlp, gpt2_manager, config)
        result = analyzer.stylometric_analysis(TEST_TEXT)

        assert "error" not in result
        assert result["baseline_used"] == "my_default"
        assert result["config"]["baseline"] == "my_default"
        assert result["config"]["baseline_info"]["name"] == "Config Test Corpus"

    def test_explicit_per_call_baseline_still_wins(self, nlp, gpt2_manager, tmp_path):
        """An explicit baseline argument overrides the configured default."""
        config = {
            "stylometry": {"default_baseline": "my_default", "custom_baselines_dir": str(tmp_path / "defs")},
        }
        BaselineManager(config).save_baseline("my_default", CUSTOM_BASELINE)

        analyzer = AIDetectionAnalyzer(nlp, gpt2_manager, config)
        result = analyzer.stylometric_analysis(TEST_TEXT, baseline="brown_corpus")

        assert result["baseline_used"] == "brown_corpus"
        assert result["config"]["baseline"] == "brown_corpus"

    def test_fallback_without_config_is_brown_corpus(self, nlp, gpt2_manager):
        """A configless analyzer keeps measuring against brown_corpus."""
        analyzer = AIDetectionAnalyzer(nlp, gpt2_manager, {})
        result = analyzer.stylometric_analysis(TEST_TEXT)

        assert result["baseline_used"] == "brown_corpus"


class TestResolveBaselineName:
    """Unit tests for the pure baseline-resolution rule."""

    def test_explicit_argument_wins_over_config(self):
        config = {"stylometry": {"default_baseline": "configured"}}

        assert resolve_baseline_name("explicit", config) == "explicit"

    def test_configured_default_used_when_argument_omitted(self):
        config = {"stylometry": {"default_baseline": "configured"}}

        assert resolve_baseline_name(None, config) == "configured"

    def test_built_in_default_without_usable_config(self):
        assert resolve_baseline_name(None, None) == "brown_corpus"
        assert resolve_baseline_name(None, {}) == "brown_corpus"
        assert resolve_baseline_name(None, {"stylometry": "not a mapping"}) == "brown_corpus"

    def test_empty_argument_falls_through_to_config(self):
        config = {"stylometry": {"default_baseline": "configured"}}

        assert resolve_baseline_name("", config) == "configured"
        assert resolve_baseline_name("   ", config) == "configured"

    def test_empty_configured_default_falls_back_to_built_in(self):
        assert resolve_baseline_name(None, {"stylometry": {"default_baseline": ""}}) == "brown_corpus"
        assert resolve_baseline_name(None, {"stylometry": {"default_baseline": "  "}}) == "brown_corpus"
        assert resolve_baseline_name(None, {"stylometry": {"default_baseline": 7}}) == "brown_corpus"


class TestResponseShapeParity:
    """baseline_used is additive and uniform across every return path."""

    def test_baseline_used_present_on_every_return_path(self, nlp, gpt2_manager):
        """Success and all failure modes carry the same keys, with baseline_used naming the baseline."""
        analyzer = AIDetectionAnalyzer(nlp, gpt2_manager, {})

        success = analyzer.stylometric_analysis(TEST_TEXT)
        wrong_language = analyzer.stylometric_analysis(TEST_TEXT, language="fr")
        empty = analyzer.stylometric_analysis("")
        missing = analyzer.stylometric_analysis(TEST_TEXT, baseline="no_such_baseline")

        for path_name, result in (("language", wrong_language), ("empty", empty), ("missing", missing)):
            assert "error" in result, f"{path_name} path must be an error response"
            assert set(result) - {"error"} == set(success), f"{path_name} path key drift"

        assert success["baseline_used"] == "brown_corpus"
        assert wrong_language["baseline_used"] == "brown_corpus"
        assert empty["baseline_used"] == "brown_corpus"
        assert missing["baseline_used"] == "no_such_baseline"
        assert "no_such_baseline" in missing["error"]
        assert missing["config"]["baseline"] == "no_such_baseline"
