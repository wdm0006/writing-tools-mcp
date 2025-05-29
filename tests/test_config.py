"""Test module for configuration management."""

from unittest.mock import patch

from server.config import load_config


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
        mock_yaml.return_value = {"test_key": "test_value"}

        config = load_config()

        # Should contain the custom value
        assert "test_key" in config
        assert config["test_key"] == "test_value"

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
