import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Adjust sys.path to allow importing from the 'app' directory
# This assumes the tests are run from the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_DIR = os.path.join(PROJECT_ROOT, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from main import LogStreamRelay, ServerWorker  # noqa: E402
from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal  # noqa: E402
from PySide6.QtTest import QSignalSpy  # Corrected import for QSignalSpy


# Helper class to receive signals
class SignalReceiver(QObject):
    signal = Signal(str)


@pytest.fixture
def log_relay(qapp_instance):  # Ensure QApplication instance exists
    """Fixture to create a LogStreamRelay instance."""
    return LogStreamRelay()


def test_log_stream_relay_isatty(log_relay):
    """Test that LogStreamRelay.isatty() returns False."""
    assert not log_relay.isatty()


def test_log_stream_relay_single_line(log_relay):
    """Test writing a single complete line."""
    receiver = SignalReceiver()
    log_relay.messageWritten.connect(receiver.signal)
    spy = QSignalSpy(receiver.signal)

    test_message = "Hello World\n"
    log_relay.write(test_message)

    # Create an event loop and timer to process events
    loop = QEventLoop()
    QTimer.singleShot(100, loop.quit)  # Quit after 100ms
    loop.exec()

    assert spy.count() == 1
    assert spy.at(0)[0] == "Hello World"  # Get first argument of first signal
    assert log_relay._buffer == ""  # Buffer should be empty


def test_log_stream_relay_multiple_lines(log_relay):
    """Test writing multiple complete lines in one go."""
    receiver = SignalReceiver()
    log_relay.messageWritten.connect(receiver.signal)
    spy = QSignalSpy(receiver.signal)

    log_relay.write("First line\nSecond line\nThird line\n")

    # Create an event loop and timer to process events
    loop = QEventLoop()
    QTimer.singleShot(100, loop.quit)  # Quit after 100ms
    loop.exec()

    assert spy.count() == 3
    assert spy.at(0)[0] == "First line"
    assert spy.at(1)[0] == "Second line"
    assert spy.at(2)[0] == "Third line"
    assert log_relay._buffer == ""  # Buffer should be empty


def test_log_stream_relay_partial_line_then_complete(log_relay):
    """Test writing a partial line, then completing it."""
    receiver = SignalReceiver()
    log_relay.messageWritten.connect(receiver.signal)
    spy = QSignalSpy(receiver.signal)

    log_relay.write("Partial message...")
    assert spy.count() == 0  # No signal yet
    assert log_relay._buffer == "Partial message..."

    log_relay.write("completed.\n")

    # Create an event loop and timer to process events
    loop = QEventLoop()
    QTimer.singleShot(100, loop.quit)  # Quit after 100ms
    loop.exec()

    assert spy.count() == 1
    assert spy.at(0)[0] == "Partial message...completed."
    assert log_relay._buffer == ""  # Buffer should be empty


def test_log_stream_relay_partial_line_remains_in_buffer(log_relay):
    """Test that a partial line without newline remains in buffer after write."""
    receiver = SignalReceiver()
    log_relay.messageWritten.connect(receiver.signal)
    spy = QSignalSpy(receiver.signal)

    log_relay.write("This has no newline")
    assert spy.count() == 0
    assert log_relay._buffer == "This has no newline"


def test_log_stream_relay_flush_emits_remaining(log_relay):
    """Test that flush() emits any remaining content in the buffer."""
    receiver = SignalReceiver()
    log_relay.messageWritten.connect(receiver.signal)
    spy = QSignalSpy(receiver.signal)

    log_relay.write("Buffered content")
    assert spy.count() == 0
    assert log_relay._buffer == "Buffered content"

    log_relay.flush()

    # Create an event loop and timer to process events
    loop = QEventLoop()
    QTimer.singleShot(100, loop.quit)  # Quit after 100ms
    loop.exec()

    assert spy.count() == 1
    assert spy.at(0)[0] == "Buffered content"
    assert log_relay._buffer == ""  # Buffer should be empty after flush


def test_log_stream_relay_flush_empty_buffer(log_relay):
    """Test that flush() does nothing if the buffer is empty."""
    receiver = SignalReceiver()
    log_relay.messageWritten.connect(receiver.signal)
    spy = QSignalSpy(receiver.signal)

    assert log_relay._buffer == ""
    log_relay.flush()
    assert spy.count() == 0
    assert log_relay._buffer == ""


def test_log_stream_relay_write_multiple_new_lines_at_once(log_relay):
    """Test writing a string with multiple newlines, ensuring each line is emitted."""
    receiver = SignalReceiver()
    log_relay.messageWritten.connect(receiver.signal)
    spy = QSignalSpy(receiver.signal)

    log_relay.write("Line1\nLine2\n\nLine4\n")  # Empty line between Line2 and Line4

    # Create an event loop and timer to process events
    loop = QEventLoop()
    QTimer.singleShot(100, loop.quit)  # Quit after 100ms
    loop.exec()

    assert spy.count() == 4  # Emits "Line1", "Line2", "" (empty string for the blank line), "Line4"
    assert spy.at(0)[0] == "Line1"
    assert spy.at(1)[0] == "Line2"
    assert spy.at(2)[0] == ""  # Empty line
    assert spy.at(3)[0] == "Line4"
    assert log_relay._buffer == ""


@pytest.fixture
def mock_log_relays():
    """Fixture to create mock LogStreamRelay instances."""
    stdout_relay = MagicMock(spec=LogStreamRelay)
    stderr_relay = MagicMock(spec=LogStreamRelay)
    return stdout_relay, stderr_relay


@pytest.fixture
def server_worker(qapp_instance, mock_log_relays):
    """Fixture to create a ServerWorker instance with mock relays."""
    stdout_relay, stderr_relay = mock_log_relays
    worker = ServerWorker(stdout_relay=stdout_relay, stderr_relay=stderr_relay)
    return worker


@patch("main.uvicorn")
def test_server_worker_run_server_success(mock_uvicorn, server_worker, mock_log_relays):
    """Test ServerWorker.run_server() successful execution path."""
    mock_stdout_relay, mock_stderr_relay = mock_log_relays

    # Configure mocks
    mock_uv_server_instance = MagicMock()
    mock_uvicorn.Server.return_value = mock_uv_server_instance
    mock_uvicorn.Config.return_value = MagicMock()  # Config object

    spy_finished = QSignalSpy(server_worker.finished)

    # sys.stdout/stderr redirection occurs within run_server
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    server_worker.run_server()

    # Create an event loop and timer to process events
    loop = QEventLoop()
    QTimer.singleShot(100, loop.quit)  # Quit after 100ms
    loop.exec()

    mock_uvicorn.Config.assert_called_once()
    mock_uvicorn.Server.assert_called_once()
    mock_uv_server_instance.run.assert_called_once()

    # Check if stdout/stderr were redirected to relays
    mock_stdout_relay.write.assert_called()
    mock_stderr_relay.write.assert_called()

    assert spy_finished.count() == 1  # finished signal should have been emitted

    # Check that original stdio is restored
    assert sys.stdout == original_stdout
    assert sys.stderr == original_stderr

    # Test stop_server
    server_worker._uvicorn_server = mock_uv_server_instance
    server_worker.stop_server()
    assert mock_uv_server_instance.should_exit is True


@patch("main.uvicorn")
def test_server_worker_run_server_import_error(mock_uvicorn, server_worker, mock_log_relays):
    """Test ServerWorker.run_server() when server module import fails."""
    mock_stdout_relay, mock_stderr_relay = mock_log_relays
    spy_finished = QSignalSpy(server_worker.finished)

    # Simulate ImportError by modifying sys.modules
    import sys

    original_server = sys.modules.get("server")
    sys.modules["server"] = None

    try:
        server_worker.run_server()
    except ModuleNotFoundError:
        pass  # Expected error

    # Create an event loop and timer to process events
    loop = QEventLoop()
    QTimer.singleShot(100, loop.quit)  # Quit after 100ms
    loop.exec()

    # Restore original module
    if original_server:
        sys.modules["server"] = original_server
    else:
        del sys.modules["server"]

    # Check that the error was logged
    mock_stdout_relay.write.assert_called()
    assert spy_finished.count() == 1


@patch("main.uvicorn")
def test_server_worker_run_server_attribute_error(mock_uvicorn, server_worker, mock_log_relays):
    """Test ServerWorker.run_server() when server module lacks required attribute."""
    mock_stdout_relay, mock_stderr_relay = mock_log_relays
    spy_finished = QSignalSpy(server_worker.finished)

    # Simulate AttributeError by modifying sys.modules
    import sys
    from types import ModuleType

    mock_server = ModuleType("server")
    mock_mcp = ModuleType("mcp")
    mock_mcp.name = "TestMCP"
    # Intentionally NOT setting sse_app to trigger AttributeError
    mock_server.mcp = mock_mcp
    original_server = sys.modules.get("server")
    sys.modules["server"] = mock_server

    try:
        server_worker.run_server()
    except AttributeError:
        pass  # Expected error

    # Create an event loop and timer to process events
    loop = QEventLoop()
    QTimer.singleShot(100, loop.quit)  # Quit after 100ms
    loop.exec()

    # Restore original module
    if original_server:
        sys.modules["server"] = original_server
    else:
        del sys.modules["server"]

    mock_stdout_relay.write.assert_called()
    assert spy_finished.count() == 1


@patch("main.uvicorn")
def test_server_worker_run_server_uvicorn_run_exception(mock_uvicorn, server_worker, mock_log_relays):
    """Test ServerWorker.run_server() when uvicorn.run() raises an exception."""
    mock_stdout_relay, mock_stderr_relay = mock_log_relays

    # Configure mocks
    mock_uv_server_instance = MagicMock()
    mock_uv_server_instance.run.side_effect = Exception("Test Uvicorn Exception")
    mock_uvicorn.Server.return_value = mock_uv_server_instance
    mock_uvicorn.Config.return_value = MagicMock()

    spy_finished = QSignalSpy(server_worker.finished)

    # Simulate a working server module
    import sys
    from types import ModuleType

    mock_server = ModuleType("server")
    mock_mcp = ModuleType("mcp")
    mock_mcp.name = "TestMCP"
    mock_mcp.sse_app = lambda: MagicMock()
    mock_server.mcp = mock_mcp
    original_server = sys.modules.get("server")
    sys.modules["server"] = mock_server

    try:
        server_worker.run_server()
    except Exception:
        pass  # Expected error

    # Create an event loop and timer to process events
    loop = QEventLoop()
    QTimer.singleShot(100, loop.quit)  # Quit after 100ms
    loop.exec()

    # Restore original module
    if original_server:
        sys.modules["server"] = original_server
    else:
        del sys.modules["server"]

    mock_stdout_relay.write.assert_called()
    assert spy_finished.count() == 1


def test_server_worker_stop_server_no_instance(server_worker, mock_log_relays):
    """Test stop_server when uvicorn_server instance is None."""
    mock_stdout_relay, _ = mock_log_relays
    server_worker._uvicorn_server = None  # Ensure no server instance
    server_worker.stop_server()

    # Check logs for "No Uvicorn server instance to stop"
    found_message = False
    for call_args in mock_stdout_relay.write.call_args_list:
        if isinstance(call_args[0][0], str) and "No Uvicorn server instance to stop" in call_args[0][0]:
            found_message = True
            break
    assert found_message, "Message about no server to stop was not logged"


@pytest.mark.skip(reason="MainWindow tests require more setup with qtbot or extensive mocking")
class TestMainWindow:
    def test_main_window_placeholder(self):
        """Placeholder for MainWindow tests."""
        pass


# More tests for ServerWorker and MainWindow will follow
