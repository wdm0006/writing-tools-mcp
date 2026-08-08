"""Test module for configuration management."""

import logging
from pathlib import Path
from unittest.mock import patch

import yaml

from server.config import load_config
from server.config.defaults import DEFAULT_CONFIG
from server.config.loader import _validate_config
from server.config.schema import CONFIG_SCHEMA


class TestConfigLoading:
    """Test the config loading functionality."""

    def test_default_config(self):
        """Test loading default configuration."""
        with patch("pathlib.Path.exists", return_value=False):
            config = load_config()

        # Check that we get a valid config dict
        assert isinstance(config, dict)
        assert len(config) > 0  # Should have some default values

    @patch("builtins.open")
    @patch("pathlib.Path.exists")
    @patch("yaml.safe_load")
    def test_custom_config(self, mock_yaml, mock_exists, mock_open):
        """Test loading custom configuration."""
        mock_exists.return_value = True
        mock_yaml.return_value = {"perplexity": {"model_name": "distilgpt2"}}

        config = load_config()

        # Should contain the custom value
        assert config["perplexity"]["model_name"] == "distilgpt2"

    @patch("builtins.open")
    @patch("pathlib.Path.exists")
    @patch("yaml.safe_load")
    def test_config_file_error(self, mock_yaml, mock_exists, mock_open):
        """Test handling of config file errors."""
        mock_exists.return_value = True
        mock_yaml.side_effect = Exception("YAML error")

        # Should not raise exception, just use defaults
        config = load_config()
        assert isinstance(config, dict)


class TestConfigStructure:
    """Test the structure and content of configuration."""

    def test_required_sections_present(self):
        """Test that all required configuration sections are present."""
        config = load_config()

        required_sections = ["perplexity", "stylometry", "logging"]
        for section in required_sections:
            assert section in config, f"Required section '{section}' missing from config"

    def test_perplexity_config_structure(self):
        """Test perplexity configuration structure."""
        config = load_config()
        perplexity_config = config["perplexity"]

        required_keys = ["model_name", "max_length", "overlap", "thresholds", "device", "language"]
        for key in required_keys:
            assert key in perplexity_config, f"Required perplexity config key '{key}' missing"

    def test_stylometry_config_structure(self):
        """Test stylometry configuration structure."""
        config = load_config()
        stylometry_config = config["stylometry"]

        assert "thresholds" in stylometry_config
        thresholds = stylometry_config["thresholds"]

        required_thresholds = ["warning_z", "error_z", "ai_confidence_threshold"]
        for threshold in required_thresholds:
            assert threshold in thresholds, f"Required threshold '{threshold}' missing"
            assert isinstance(thresholds[threshold], (int, float))

    def test_logging_config_structure(self):
        """Test logging configuration structure."""
        config = load_config()
        logging_config = config["logging"]

        assert "level" in logging_config
        assert "format" in logging_config
        assert isinstance(logging_config["level"], str)
        assert isinstance(logging_config["format"], str)

    def test_config_values_types(self):
        """Test that configuration values have correct types."""
        config = load_config()

        # String values
        assert isinstance(config["perplexity"]["model_name"], str)
        assert isinstance(config["perplexity"]["device"], str)
        assert isinstance(config["perplexity"]["language"], str)

        # Numeric values
        assert isinstance(config["perplexity"]["max_length"], int)
        assert isinstance(config["perplexity"]["overlap"], int)
        assert isinstance(config["perplexity"]["thresholds"]["ppl_max"], (int, float))
        assert isinstance(config["perplexity"]["thresholds"]["burstiness_min"], (int, float))
        assert isinstance(config["stylometry"]["thresholds"]["warning_z"], (int, float))
        assert isinstance(config["stylometry"]["thresholds"]["error_z"], (int, float))
        assert isinstance(config["stylometry"]["thresholds"]["ai_confidence_threshold"], (int, float))


def _write_config(tmp_path, text):
    """Write a `.mcp-config.yaml` into a temporary directory and return its path."""
    config_file = tmp_path / ".mcp-config.yaml"
    config_file.write_text(text)
    return str(config_file)


def _leaf_paths(mapping, prefix=""):
    """Yield the dotted path of every non-mapping leaf in a nested dictionary."""
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _leaf_paths(value, path)
        else:
            yield path


class TestUnknownKeys:
    """Unknown keys are rejected with a warning naming the full path."""

    def test_misspelled_consumed_key_is_dropped(self, tmp_path, caplog):
        """A typo such as `max_lenght` warns and leaves the real setting at its default."""
        path = _write_config(tmp_path, "perplexity:\n  max_lenght: 99\n")

        with caplog.at_level(logging.WARNING):
            config = load_config(path)

        assert "max_lenght" not in config["perplexity"]
        assert config["perplexity"]["max_length"] == 512
        assert "perplexity.max_lenght" in caplog.text

    def test_unknown_top_level_section_is_dropped(self, tmp_path, caplog):
        """An unknown top-level section warns and does not enter the configuration."""
        path = _write_config(tmp_path, "test_key: test_value\n")

        with caplog.at_level(logging.WARNING):
            config = load_config(path)

        assert "test_key" not in config
        assert "'test_key'" in caplog.text

    def test_unknown_threshold_is_dropped(self, tmp_path, caplog):
        """Unknown keys are rejected at every nesting level, not just the top one."""
        path = _write_config(tmp_path, "stylometry:\n  thresholds:\n    warning_zz: 4.0\n")

        with caplog.at_level(logging.WARNING):
            config = load_config(path)

        assert "warning_zz" not in config["stylometry"]["thresholds"]
        assert config["stylometry"]["thresholds"]["warning_z"] == 2.0
        assert "stylometry.thresholds.warning_zz" in caplog.text

    def test_known_siblings_of_an_unknown_key_survive(self, tmp_path, caplog):
        """Rejecting one key does not discard valid keys beside it."""
        path = _write_config(tmp_path, "perplexity:\n  max_lenght: 99\n  overlap: 12\n")

        with caplog.at_level(logging.WARNING):
            config = load_config(path)

        assert config["perplexity"]["overlap"] == 12
        assert config["perplexity"]["max_length"] == 512


class TestWrongTypes:
    """Wrongly typed values are rejected with a path-specific warning."""

    def test_scalar_where_a_mapping_is_required(self, tmp_path, caplog):
        """`perplexity: gpt2` is rejected instead of reaching the model managers."""
        path = _write_config(tmp_path, "perplexity: gpt2\n")

        with caplog.at_level(logging.WARNING):
            config = load_config(path)

        assert config["perplexity"] == DEFAULT_CONFIG["perplexity"]
        assert "'perplexity' must be a mapping, got str" in caplog.text

    def test_scalar_where_a_nested_mapping_is_required(self, tmp_path, caplog):
        """A scalar in place of a nested section falls back to the default section."""
        path = _write_config(tmp_path, "perplexity:\n  thresholds: 25.0\n")

        with caplog.at_level(logging.WARNING):
            config = load_config(path)

        assert config["perplexity"]["thresholds"] == DEFAULT_CONFIG["perplexity"]["thresholds"]
        assert "'perplexity.thresholds' must be a mapping, got float" in caplog.text

    def test_non_numeric_threshold(self, tmp_path, caplog):
        """A non-numeric threshold warns and falls back to the default."""
        path = _write_config(tmp_path, 'perplexity:\n  thresholds:\n    ppl_max: "high"\n')

        with caplog.at_level(logging.WARNING):
            config = load_config(path)

        assert config["perplexity"]["thresholds"]["ppl_max"] == 25.0
        assert "'perplexity.thresholds.ppl_max' must be a number, got str" in caplog.text

    def test_non_integer_max_length(self, tmp_path, caplog):
        """A float where an integer is required warns and falls back to the default."""
        path = _write_config(tmp_path, "perplexity:\n  max_length: 512.5\n")

        with caplog.at_level(logging.WARNING):
            config = load_config(path)

        assert config["perplexity"]["max_length"] == 512
        assert "'perplexity.max_length' must be an integer, got float" in caplog.text

    def test_boolean_is_not_accepted_as_a_number(self, tmp_path, caplog):
        """YAML booleans are rejected even though `bool` subclasses `int`."""
        path = _write_config(tmp_path, "perplexity:\n  overlap: true\n")

        with caplog.at_level(logging.WARNING):
            config = load_config(path)

        assert config["perplexity"]["overlap"] == 50
        assert "'perplexity.overlap' must be an integer, got bool" in caplog.text

    def test_non_string_model_name(self, tmp_path, caplog):
        """A numeric model name warns and falls back to the default."""
        path = _write_config(tmp_path, "perplexity:\n  model_name: 2\n")

        with caplog.at_level(logging.WARNING):
            config = load_config(path)

        assert config["perplexity"]["model_name"] == "gpt2"
        assert "'perplexity.model_name' must be a string, got int" in caplog.text

    def test_feature_list_with_non_string_entries(self, tmp_path, caplog):
        """A list value is checked element by element."""
        path = _write_config(tmp_path, "stylometry:\n  features:\n    pos_tags: [NOUN, 3]\n")

        with caplog.at_level(logging.WARNING):
            config = load_config(path)

        assert "pos_tags" not in config["stylometry"].get("features", {})
        assert "'stylometry.features.pos_tags' must be a list of strings, got list" in caplog.text

    def test_config_file_containing_a_scalar(self, tmp_path, caplog):
        """A YAML document that is not a mapping falls back to the defaults."""
        path = _write_config(tmp_path, "just a string\n")

        with caplog.at_level(logging.WARNING):
            config = load_config(path)

        assert config == DEFAULT_CONFIG
        assert "must contain a mapping, got str" in caplog.text


class TestValidOverrides:
    """Valid configuration still merges recursively."""

    def test_partial_override_preserves_unspecified_defaults(self, tmp_path, caplog):
        """Overriding one threshold leaves its siblings and other sections untouched."""
        path = _write_config(
            tmp_path,
            "perplexity:\n"
            "  model_name: distilgpt2\n"
            "  thresholds:\n"
            "    ppl_max: 40\n"
            "stylometry:\n"
            "  thresholds:\n"
            "    error_z: 4.5\n",
        )

        with caplog.at_level(logging.WARNING):
            config = load_config(path)

        assert caplog.text == ""
        assert config["perplexity"]["model_name"] == "distilgpt2"
        assert config["perplexity"]["thresholds"]["ppl_max"] == 40
        assert config["perplexity"]["thresholds"]["burstiness_min"] == 2.5
        assert config["perplexity"]["max_length"] == 512
        assert config["stylometry"]["thresholds"]["error_z"] == 4.5
        assert config["stylometry"]["thresholds"]["warning_z"] == 2.0
        assert config["logging"] == DEFAULT_CONFIG["logging"]

    def test_integers_are_accepted_where_numbers_are_expected(self, tmp_path, caplog):
        """An integer is a valid number, so `warning_z: 3` is not a type error."""
        path = _write_config(tmp_path, "stylometry:\n  thresholds:\n    warning_z: 3\n")

        with caplog.at_level(logging.WARNING):
            config = load_config(path)

        assert caplog.text == ""
        assert config["stylometry"]["thresholds"]["warning_z"] == 3


class TestSchemaCoverage:
    """The schema covers everything the server actually reads."""

    # Settings read by `initialize_models` or by the analysis path, as dotted paths.
    CONSUMED_SETTINGS = [
        "perplexity.model_name",
        "perplexity.max_length",
        "perplexity.overlap",
        "perplexity.device",
        "perplexity.thresholds.ppl_max",
        "perplexity.thresholds.burstiness_min",
        "stylometry.thresholds.warning_z",
        "stylometry.thresholds.error_z",
        "stylometry.thresholds.ai_confidence_threshold",
    ]

    def test_every_consumed_setting_is_declared(self):
        """Every setting the server reads has a schema entry."""
        declared = set(_leaf_paths(CONFIG_SCHEMA))
        for setting in self.CONSUMED_SETTINGS:
            assert setting in declared, f"Consumed setting '{setting}' is not validated"

    def test_every_default_is_declared_and_valid(self):
        """The shipped defaults themselves satisfy the schema."""
        assert set(_leaf_paths(DEFAULT_CONFIG)) <= set(_leaf_paths(CONFIG_SCHEMA))

        config = _validate_config(DEFAULT_CONFIG, ".mcp-config.yaml")
        assert config == DEFAULT_CONFIG

    def test_shipped_config_file_validates_cleanly(self, caplog):
        """The `.mcp-config.yaml` committed to the repository produces no warnings."""
        shipped = Path(__file__).parent.parent / ".mcp-config.yaml"
        user_config = yaml.safe_load(shipped.read_text())

        with caplog.at_level(logging.WARNING):
            validated = _validate_config(user_config, str(shipped))

        assert caplog.text == ""
        assert validated == user_config
