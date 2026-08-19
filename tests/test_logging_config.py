"""Tests for applying the `logging` section of `.mcp-config.yaml`."""

import contextlib
import io
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import server.app as app
from server.config import load_config

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture
def restore_root_logging():
    """Restore the root logger, which `configure_logging` reconfigures with `force=True`."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


@contextlib.contextmanager
def redirected_streams():
    """Capture both streams while the logging handler is built and used.

    The redirection has to happen inside the test body: the handler binds whatever
    `sys.stderr` is when `configure_logging` runs, and pytest reinstates its own capture
    objects between the fixture and call phases.
    """
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        yield stdout, stderr


def _load(tmp_path, text):
    """Write a `.mcp-config.yaml` and return the configuration it produces."""
    config_file = tmp_path / ".mcp-config.yaml"
    config_file.write_text(textwrap.dedent(text))
    return load_config(str(config_file))


class TestConfiguredLevel:
    """The configured level reaches the logger."""

    def test_debug_level_is_applied(self, tmp_path, restore_root_logging):
        config = _load(tmp_path, "logging:\n  level: DEBUG\n")

        app.configure_logging(config)

        assert logging.getLogger().getEffectiveLevel() == logging.DEBUG
        assert logging.getLogger("server.app").isEnabledFor(logging.DEBUG)

    def test_debug_record_is_emitted(self, tmp_path, restore_root_logging):
        config = _load(tmp_path, "logging:\n  level: DEBUG\n")

        with redirected_streams() as (_, stderr):
            app.configure_logging(config)
            logging.getLogger("probe").debug("debug-record")

        assert "debug-record" in stderr.getvalue()

    def test_default_level_still_suppresses_debug_records(self, tmp_path, restore_root_logging):
        """With no `logging` override the behaviour is unchanged: INFO, no debug records."""
        config = _load(tmp_path, "perplexity:\n  model_name: gpt2\n")

        with redirected_streams() as (_, stderr):
            app.configure_logging(config)
            logging.getLogger("probe").debug("debug-record")
            logging.getLogger("probe").info("info-record")

        assert logging.getLogger().getEffectiveLevel() == logging.INFO
        assert "debug-record" not in stderr.getvalue()
        assert "info-record" in stderr.getvalue()

    def test_level_is_case_insensitive(self, tmp_path, restore_root_logging):
        config = _load(tmp_path, "logging:\n  level: debug\n")

        app.configure_logging(config)

        assert logging.getLogger().getEffectiveLevel() == logging.DEBUG


class TestConfiguredFormat:
    """The configured format reaches emitted records."""

    def test_custom_format_is_used(self, tmp_path, restore_root_logging):
        config = _load(
            tmp_path,
            """
            logging:
              level: WARNING
              format: "CUSTOM|%(levelname)s|%(message)s"
            """,
        )

        with redirected_streams() as (_, stderr):
            app.configure_logging(config)
            logging.getLogger("probe").warning("formatted-record")

        assert "CUSTOM|WARNING|formatted-record" in stderr.getvalue()

    def test_default_format_is_used_when_unset(self, tmp_path, restore_root_logging):
        config = _load(tmp_path, "logging:\n  level: WARNING\n")

        with redirected_streams() as (_, stderr):
            app.configure_logging(config)
            logging.getLogger("probe").warning("default-format-record")

        # The shipped format is "<timestamp> - <level> - <message>".
        assert " - WARNING - default-format-record" in stderr.getvalue()


class TestInvalidLevel:
    """An unrecognized level is survivable."""

    def test_unknown_level_falls_back_and_warns(self, tmp_path, restore_root_logging):
        config = _load(tmp_path, 'logging:\n  level: "LOUD"\n')

        with redirected_streams() as (_, stderr):
            app.configure_logging(config)  # must not raise

        assert logging.getLogger().getEffectiveLevel() == logging.INFO
        assert "Unknown logging level" in stderr.getvalue()
        assert "LOUD" in stderr.getvalue()

    def test_unknown_level_still_applies_the_configured_format(self, tmp_path, restore_root_logging):
        config = _load(
            tmp_path,
            """
            logging:
              level: "LOUD"
              format: "FALLBACK|%(message)s"
            """,
        )

        with redirected_streams() as (_, stderr):
            app.configure_logging(config)

        assert "FALLBACK|Unknown logging level" in stderr.getvalue()


class TestStreamIsPinnedToStderr:
    """Logs must never reach stdout, which carries the MCP JSON-RPC stream."""

    @pytest.mark.parametrize("level_name", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_records_go_to_stderr_only(self, level_name, tmp_path, restore_root_logging):
        config = _load(tmp_path, f"logging:\n  level: {level_name}\n")

        with redirected_streams() as (stdout, stderr):
            app.configure_logging(config)
            logging.getLogger("probe").log(getattr(logging, level_name), "record-%s", level_name)

        assert f"record-{level_name}" in stderr.getvalue()
        assert stdout.getvalue() == ""


class TestServerStartup:
    """The real import path: bootstrap logging, then load and apply the configuration."""

    def test_loader_warnings_are_visible_and_stdout_stays_clean(self, tmp_path):
        (tmp_path / ".mcp-config.yaml").write_text(
            textwrap.dedent(
                """
                logging:
                  level: "DEBUG"
                  format: "CFG|%(levelname)s|%(message)s"
                not_a_real_section: 1
                """
            )
        )
        script = textwrap.dedent(
            """
            import logging

            import server.app  # noqa: F401 - importing runs bootstrap + configure_logging

            logging.getLogger("probe").debug("probe-debug-record")
            print("stdout-marker")
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )

        assert result.returncode == 0, result.stderr
        # The bootstrap configuration runs before `load_config`, so the loader's own warnings are
        # visible — and carry the bootstrap format, not the one the file being loaded asks for.
        assert " - WARNING - Unknown configuration key 'not_a_real_section'" in result.stderr
        # The configured level and format then take effect.
        assert "CFG|DEBUG|probe-debug-record" in result.stderr
        # Nothing but the script's own output reached stdout.
        assert result.stdout.strip() == "stdout-marker"
