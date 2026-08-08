"""Schema describing the supported `.mcp-config.yaml` surface."""

STRING = "a string"
INTEGER = "an integer"
NUMBER = "a number"
STRING_LIST = "a list of strings"

# Every key a user may set. Nested mappings are described by nested dictionaries;
# leaves carry the expected value kind. Keys outside this schema are rejected by
# `server.config.loader`, so anything added to `.mcp-config.yaml` or to
# `DEFAULT_CONFIG` has to be declared here too.
CONFIG_SCHEMA = {
    "perplexity": {
        "model_name": STRING,
        "max_length": INTEGER,
        "overlap": INTEGER,
        "device": STRING,
        "language": STRING,
        "thresholds": {
            "ppl_max": NUMBER,
            "burstiness_min": NUMBER,
        },
    },
    "stylometry": {
        "default_baseline": STRING,
        "custom_baselines_dir": STRING,
        "thresholds": {
            "warning_z": NUMBER,
            "error_z": NUMBER,
            "ai_confidence_threshold": NUMBER,
        },
        "features": {
            "enabled": STRING_LIST,
            "pos_tags": STRING_LIST,
        },
    },
    "logging": {
        "level": STRING,
        "format": STRING,
    },
}
