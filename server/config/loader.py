"""Configuration loading functionality."""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from server.config.defaults import DEFAULT_CONFIG
from server.config.schema import CONFIG_SCHEMA, INTEGER, NUMBER, STRING, STRING_LIST

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
                    config = _merge_config(config, _validate_config(user_config, config_path))
        except Exception as e:
            logger.warning(f"Error loading config file: {e}. Using defaults.")

    return config


def _validate_config(user_config: Any, config_path: str) -> Dict[str, Any]:
    """Drop unsupported keys and wrongly typed values from a user configuration."""
    if not isinstance(user_config, dict):
        logger.warning(f"{config_path} must contain a mapping, got {_type_name(user_config)}. Using defaults.")
        return {}

    return _validate_section(user_config, CONFIG_SCHEMA, "", config_path)


def _validate_section(user: Dict[str, Any], schema: Dict[str, Any], path: str, config_path: str) -> Dict[str, Any]:
    """Validate one mapping level against its schema, warning about every rejected entry."""
    validated: Dict[str, Any] = {}

    for key, value in user.items():
        full_path = f"{path}.{key}" if path else str(key)

        if key not in schema:
            logger.warning(f"Unknown configuration key '{full_path}' in {config_path}. Ignoring it.")
            continue

        expected = schema[key]
        if isinstance(expected, dict):
            if not isinstance(value, dict):
                logger.warning(
                    f"Configuration key '{full_path}' must be a mapping, got {_type_name(value)}. Using defaults."
                )
                continue
            validated[key] = _validate_section(value, expected, full_path, config_path)
        elif _matches_kind(value, expected):
            validated[key] = value
        else:
            logger.warning(
                f"Configuration key '{full_path}' must be {expected}, got {_type_name(value)}. Using the default."
            )

    return validated


def _matches_kind(value: Any, kind: str) -> bool:
    """Check a scalar (or list) value against a schema leaf."""
    # bool is a subclass of int, but a YAML boolean is never a valid number here.
    if isinstance(value, bool):
        return False
    if kind == STRING:
        return isinstance(value, str)
    if kind == INTEGER:
        return isinstance(value, int)
    if kind == NUMBER:
        return isinstance(value, (int, float))
    if kind == STRING_LIST:
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    raise ValueError(f"Unknown schema kind: {kind!r}")


def _type_name(value: Any) -> str:
    """Human-readable type name for warning messages."""
    return type(value).__name__


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
