"""Configuration loading functionality."""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from server.config.defaults import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def load_config(config_path: str = ".mcp-config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file with defaults.

    Args:
        config_path: Path to configuration file

    Returns:
        Merged configuration dictionary
    """
    config = _deep_copy_dict(DEFAULT_CONFIG)
    config_file = Path(config_path)

    if config_file.exists():
        try:
            with open(config_file) as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    config = _merge_config(config, user_config)
        except Exception as e:
            logger.warning(f"Error loading config file: {e}. Using defaults.")

    return config


def _merge_config(default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge user config with defaults."""
    result = _deep_copy_dict(default)

    for key, value in user.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value

    return result


def _deep_copy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Deep copy a dictionary to avoid mutation issues."""
    if isinstance(d, dict):
        return {k: _deep_copy_dict(v) for k, v in d.items()}
    return d
