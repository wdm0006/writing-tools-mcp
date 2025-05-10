import asyncio
import logging
import os
import sys
import threading

import uvicorn
from mcp import ClientSession
from mcp.client.sse import sse_client
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Configure root logger first thing
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# Add the parent directory of 'app' to sys.path to allow importing 'server'
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
SERVER_DIR = os.path.join(PROJECT_ROOT, "server")

# Prepend SERVER_DIR first, then PROJECT_ROOT to prioritize modules within server/
# when just 'import server' is used.
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class QtLogHandler(logging.Handler):
    def __init__(self, append_log_func):
        super().__init__()
        self.append_log_func = append_log_func
        self.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"))

    def emit(self, record):
        msg = self.format(record)
        self.append_log_func(msg)


class LogStreamRelay(QObject):
    """
    A QObject that emulates a stream-like interface (write, flush, isatty)
    and emits a Qt signal for each complete line written to it.

    This class is designed to capture stdout/stderr and relay messages to the
    Qt application's main thread for display in a UI element (e.g., QTextEdit).
    It buffers incoming messages until a newline character is encountered,
    then emits the `messageWritten` signal with the complete line.

    Attributes:
        messageWritten (Signal): Qt signal that emits a string (a line of log).
    """

    messageWritten = Signal(str)

    def __init__(self, parent=None):
        """
        Initializes the LogStreamRelay.

        Args:
            parent (QObject, optional): The parent QObject. Defaults to None.
        """
        super().__init__(parent)
        self._buffer = ""

    def write(self, text: str) -> None:
        """
        Writes text to the relay, emitting messageWritten signal for each complete line.

        Args:
            text (str): The text to write.
        """
        try:
            # Split text into lines, keeping the newline characters
            lines = text.splitlines(keepends=True)

            for line in lines:
                self._buffer += line
                if line.endswith("\n"):
                    # We have a complete line, emit it without the newline
                    try:
                        self.messageWritten.emit(self._buffer.rstrip("\n"))
                    except RuntimeError as e:
                        logging.error(f"LogStreamRelay Error: Cannot emit signal - {e}\nMessage: {self._buffer}")
                    self._buffer = ""
        except Exception as e:
            # If any other error occurs, write to original stderr
            logging.error(f"LogStreamRelay Error: {e}\nMessage: {text}")

    def flush(self) -> None:
        """
        Flushes the buffer, emitting any remaining content.
        """
        if self._buffer:
            try:
                self.messageWritten.emit(self._buffer)
            except RuntimeError as e:
                logging.error(f"LogStreamRelay Error: Cannot emit signal - {e}\nMessage: {self._buffer}")
            self._buffer = ""

    def isatty(self) -> bool:
        """
        Returns False to indicate this is not a TTY.

        Returns:
            bool: Always False.
        """
        return False


class ServerWorker(QObject):
    """
    Manages the execution of the Uvicorn server in a separate QThread.

    This worker is responsible for:
    - Setting up and starting the Uvicorn server with the MCP application.
    - Redirecting the server's stdout and stderr to LogStreamRelay instances.
    - Providing a slot (`stop_server`) to gracefully shut down Uvicorn.
    - Emitting a `finished` signal when the server has stopped.

    Attributes:
        finished (Signal): Emitted when the Uvicorn server run finishes or an error occurs.
        _uvicorn_server (uvicorn.Server | None): The Uvicorn server instance.
        _stdout_relay (LogStreamRelay): Relay for stdout messages.
        _stderr_relay (LogStreamRelay): Relay for stderr messages.
    """

    finished = Signal()

    def __init__(self, stdout_relay: LogStreamRelay, stderr_relay: LogStreamRelay, parent=None):
        """
        Initializes the ServerWorker.

        Args:
            stdout_relay (LogStreamRelay): The relay for capturing stdout.
            stderr_relay (LogStreamRelay): The relay for capturing stderr.
            parent (QObject, optional): The parent QObject. Defaults to None.
        """
        super().__init__(parent)
        self._uvicorn_server = None
        self._stdout_relay = stdout_relay
        self._stderr_relay = stderr_relay
        self._stdout_relay.write("ServerWorker: Initialized.\n")

    @Slot()
    def run_server(self):
        """
        Starts the Uvicorn server.

        This method performs the following steps:
        1. Captures original stdout/stderr.
        2. Redirects sys.stdout and sys.stderr to the provided LogStreamRelay instances.
        3. Imports the MCP server application instance.
        4. Configures and starts the Uvicorn server.
        5. Restores original stdout/stderr and emits `finished` signal upon completion or error.
        """
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        original_handlers = logging.root.handlers[:]

        try:
            # Set up logging redirection first
            sys.stdout = self._stdout_relay
            sys.stderr = self._stderr_relay

            # Clear existing handlers and add our relay handler
            logging.root.handlers = []
            relay_handler = logging.StreamHandler(self._stdout_relay)
            relay_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"))
            logging.root.addHandler(relay_handler)

            self._stderr_relay.write("ServerWorker: stdout/stderr successfully redirected.\n")

            try:
                # Import and configure server
                from server import mcp as server_module

                mcp_instance = server_module

                if not hasattr(mcp_instance, "sse_app"):
                    error_msg = "MCP instance does not have 'sse_app' method required for ASGI."
                    self._stderr_relay.write(f"ServerWorker Error: {error_msg}\n")
                    raise AttributeError(error_msg)

                # Configure and start server
                config = uvicorn.Config(
                    app=mcp_instance.sse_app(),
                    host="127.0.0.1",
                    port=8001,
                    log_config=None,
                )
                self._uvicorn_server = uvicorn.Server(config=config)
                self._stdout_relay.write("ServerWorker: Starting Uvicorn server...\n")
                self._uvicorn_server.run()

            except Exception as e:
                import traceback

                tb = traceback.format_exc()
                self._stderr_relay.write(
                    f"ServerWorker Error: Server execution failed - {str(e)}\nFull traceback:\n{tb}\n"
                )
                raise

        except Exception as e:
            if not isinstance(e, SystemExit):
                self._stderr_relay.write(f"ServerWorker Critical Error: {str(e)}\n")
        finally:
            # Restore original stdout/stderr and logging
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            logging.root.handlers = original_handlers

            self.finished.emit()

    @Slot()
    def stop_server(self):
        """
        Requests the Uvicorn server to stop.

        This method sets the `should_exit` flag on the Uvicorn server instance,
        which Uvicorn checks to initiate a graceful shutdown.
        """
        if self._uvicorn_server:
            self._stdout_relay.write("ServerWorker: Stopping Uvicorn server...\n")
            self._uvicorn_server.should_exit = True
        else:
            self._stdout_relay.write("ServerWorker: No Uvicorn server instance to stop.\n")
            self._stdout_relay.flush()


class MainWindow(QMainWindow):
    """
    Main application window that manages the server control UI and log display.

    This window provides:
    - A text area for displaying server logs
    - Buttons for starting/stopping/restarting the server
    - Management of the ServerWorker and its thread
    """

    def __init__(self):
        """
        Initializes the MainWindow, setting up the UI and server management components.
        """
        super().__init__()
        self.setWindowTitle("MCP Server Control")
        self.resize(800, 600)

        # Initialize server management attributes
        self._server_worker = None
        self._server_thread = None
        self._restart_requested = False

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        layout.addWidget(self.log_display)

        # Create button container and layout
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        layout.addWidget(button_container)

        # Create control buttons
        self.start_button = QPushButton("Start Server")
        self.start_button.clicked.connect(self.start_server)
        button_layout.addWidget(self.start_button)

        self.restart_button = QPushButton("Restart Server")
        self.restart_button.clicked.connect(self.restart_server)
        self.restart_button.setEnabled(False)  # Initially disabled
        button_layout.addWidget(self.restart_button)

        self.test_button = QPushButton("Test MCP")
        self.test_button.clicked.connect(self.test_mcp_server)
        button_layout.addWidget(self.test_button)

        # Add Connect to Apps button
        self.connect_apps_button = QPushButton("Connect to Apps")
        self.connect_apps_button.clicked.connect(self.show_connect_apps_dialog)
        button_layout.addWidget(self.connect_apps_button)

        # Create LogStreamRelay instances for stdout/stderr
        self._stdout_relay = LogStreamRelay(self)
        self._stderr_relay = LogStreamRelay(self)

        # Connect the relays' messageWritten signals to append_log
        self._stdout_relay.messageWritten.connect(self.append_log)
        self._stderr_relay.messageWritten.connect(self.append_log)

        # Set up logging
        LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
        self.logger = logging.getLogger(__name__)
        handler = QtLogHandler(self.append_log)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logging.getLogger().addHandler(handler)
        # Remove duplicate logs if root logger already has handlers
        if len(logging.getLogger().handlers) > 1:
            logging.getLogger().handlers = [handler]
        self.logger.info("MainWindow: Initialized and ready.")

    @Slot(str)
    def append_log(self, message: str):
        """
        Slot that appends a message to the log display.

        Args:
            message (str): The message to append.
        """
        self.log_display.append(message)

    def start_server(self):
        """
        Starts the server in a new thread.

        This method:
        1. Creates a new QThread
        2. Creates a new ServerWorker
        3. Moves the worker to the thread
        4. Connects necessary signals
        5. Starts the thread
        """
        self.logger.info("MainWindow: start_server() called.")

        if self._server_thread and self._server_thread.isRunning():
            self.logger.info("MainWindow: Server thread already running.")
            return

        # Create and configure the thread
        self._server_thread = QThread()
        self._server_thread.setObjectName("ServerThread")

        # Create the worker with our LogStreamRelay instances
        self._server_worker = ServerWorker(stdout_relay=self._stdout_relay, stderr_relay=self._stderr_relay)

        # Move worker to thread and set up connections
        self._server_worker.moveToThread(self._server_thread)
        self._server_thread.started.connect(self._server_worker.run_server)
        self._server_worker.finished.connect(self.on_server_worker_finished)

        # Set up cleanup connections
        self._server_worker.finished.connect(self._server_thread.quit)
        self._server_thread.finished.connect(self._server_thread.deleteLater)
        self._server_worker.finished.connect(self._server_worker.deleteLater)

        # Start the thread
        self.logger.info("MainWindow: Starting server thread...")
        self._server_thread.start()

        # Update UI
        self.start_button.setEnabled(False)
        self.restart_button.setEnabled(True)

    def restart_server(self):
        """
        Initiates a server restart sequence.

        This method:
        1. Sets the restart flag
        2. Requests the current server to stop
        3. Waits for the finished signal
        4. start_server will be called by the finished handler
        """
        self.logger.info("MainWindow: restart_server() called.")
        if not self._server_worker or not self._server_thread or not self._server_thread.isRunning():
            self.logger.info("MainWindow: No running server to restart. Starting fresh...")
            self.start_server()
            return

        self.logger.info("MainWindow: Setting restart flag and stopping current server...")
        self._restart_requested = True
        self.restart_button.setEnabled(False)  # Prevent multiple restarts
        self._server_worker.stop_server()

    def test_mcp_server(self):
        """
        Tests the MCP server by connecting via the MCP SDK (SSE) and listing tools.
        """
        self.logger.info("MainWindow: Testing MCP server using MCP SDK (SSE)...")

        def run_test():
            async def async_test():
                try:
                    # Connect to the SSE server
                    async with sse_client("http://127.0.0.1:8001/sse") as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            tools = await session.list_tools()
                            self.logger.info(f"MCP SDK Test Success: {tools}")
                except Exception as e:
                    self.logger.error(f"MCP SDK Test Error: {type(e).__name__}: {e}")

            asyncio.run(async_test())

        threading.Thread(target=run_test, daemon=True).start()

    def show_connect_apps_dialog(self):
        """
        Show a modal dialog with documentation on connecting external apps to this MCP.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Connect to Apps")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        from PySide6.QtWidgets import QPushButton, QTextEdit

        doc_text = (
            "<b>How to Connect External Apps to MCP</b><br><br>"
            "<b>Cursor:</b><br>"
            "1. Open Cursor settings.<br>"
            "2. Go to the 'AI Providers' section.<br>"
            "3. Add a new provider with the following endpoint:<br>"
            "<code>http://127.0.0.1:8001/sse</code><br>"
            "4. Set the API type to 'FastMCP' or 'OpenAI-compatible' if available.<br>"
            "5. Save and test the connection.<br><br>"
            "<b>Claude.ai (if supported):</b><br>"
            "1. Go to Claude's integrations or custom API section.<br>"
            "2. Enter the endpoint: <code>http://127.0.0.1:8001/sse</code><br>"
            "3. Use your MCP API key if required (see your MCP settings).<br>"
            "4. Save and test the connection.<br><br>"
            "<b>General Notes:</b><br>"
            "- Ensure this MCP server is running and accessible from the app.<br>"
            "- For more details, see the documentation or ask in the support channel."
        )
        text = QTextEdit()
        text.setReadOnly(True)
        text.setHtml(doc_text)
        text.setStyleSheet(
            "background-color: #101010; color: #00FF41; font-family: 'Fira Mono', monospace; border: none;"
        )
        layout.addWidget(text)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.setStyleSheet("background-color: #181A1B; border: 2px solid #00FF41;")
        dialog.setFixedSize(500, 400)
        dialog.exec()

    @Slot()
    def on_server_worker_finished(self):
        """
        Slot called when the ServerWorker's `finished` signal is emitted.
        This indicates the Uvicorn server has stopped.
        It handles cleanup of the thread and worker references and, if a restart
        was requested, initiates it.
        """
        self.logger.info("MainWindow: on_server_worker_finished() slot triggered.")

        # The ServerWorker and QThread instances are scheduled for deletion via deleteLater
        # connected to their respective 'finished' signals. We should nullify MainWindow's
        # references to them to prevent using dangling pointers.

        current_thread_ref = self._server_thread  # Store ref for logging and potential ops

        self.logger.debug(
            f"MainWindow: Nullifying references to ServerWorker (was {type(self._server_worker)}) and QThread (was {type(self._server_thread)})."
        )
        self._server_worker = None
        self._server_thread = None

        if current_thread_ref:
            if current_thread_ref.isRunning():
                self.logger.debug(
                    f"MainWindow: QThread {current_thread_ref.objectName() or 'Unnamed'} is still marked as running after worker finished. Requesting quit()."
                )
                current_thread_ref.quit()  # Ask the QThread's event loop to stop
                if not current_thread_ref.wait(2000):  # Wait up to 2 seconds
                    self.logger.warning(
                        f"MainWindow: Warning: QThread {current_thread_ref.objectName() or 'Unnamed'} did not finish cleanly after quit() and wait(). It might be forcefully terminated or still cleaning up."
                    )
                else:
                    self.logger.debug(
                        f"MainWindow: QThread {current_thread_ref.objectName() or 'Unnamed'} finished after quit() and wait()."
                    )
            else:
                self.logger.debug(
                    f"MainWindow: QThread {current_thread_ref.objectName() or 'Unnamed'} was already not running when worker finished."
                )
        else:
            self.logger.debug("MainWindow: _server_thread was already None when worker finished.")

        if self._restart_requested:
            self.logger.info("MainWindow: Server stop confirmed by worker finishing. Proceeding with restart...")
            self._restart_requested = False
            self.start_server()
        else:
            self.logger.info("MainWindow: Server has stopped (no restart requested).")
            self.restart_button.setEnabled(True)  # Re-enable button as server is now stopped

    def closeEvent(self, event):
        """
        Handles the main window's close event.
        Attempts to gracefully stop the server thread before closing the application.

        Args:
            event (QCloseEvent): The close event.
        """
        # Show a modal dialog indicating closing
        closing_dialog = QDialog(self)
        closing_dialog.setWindowTitle("Closing...")
        closing_dialog.setModal(True)
        layout = QVBoxLayout(closing_dialog)
        from PySide6.QtWidgets import QLabel

        label = QLabel("Closing...")
        label.setStyleSheet("color: #00FF41; font-size: 18px; padding: 20px;")
        layout.addWidget(label)
        closing_dialog.setStyleSheet("background-color: #181A1B; border: 2px solid #00FF41;")
        closing_dialog.setFixedSize(300, 100)
        closing_dialog.show()
        QApplication.processEvents()  # Ensure dialog is shown immediately

        self.logger.info("MainWindow: closeEvent() triggered. Application is closing.")
        self._restart_requested = False  # Ensure no restart happens during shutdown

        if self._server_thread and self._server_thread.isRunning():
            self.logger.info("MainWindow: Server thread is running. Attempting to stop it for application shutdown...")
            if self._server_worker:
                self.logger.info("MainWindow: Calling ServerWorker.stop_server() for shutdown.")
                self._server_worker.stop_server()
            else:
                self.logger.warning("MainWindow: Server thread running, but no worker to signal stop. This is unusual.")

            # Wait for the thread to finish. The on_server_worker_finished logic will handle
            # some cleanup, but we need to ensure the thread actually stops here.
            current_thread_for_shutdown = self._server_thread
            self.logger.info(
                f"MainWindow: Waiting up to 5 seconds for server thread ({current_thread_for_shutdown.objectName() or 'Unnamed'}) to finish..."
            )
            if not current_thread_for_shutdown.wait(5000):
                self.logger.warning(
                    f"MainWindow: Warning: Server thread ({current_thread_for_shutdown.objectName() or 'Unnamed'}) did not stop cleanly within 5 seconds during closeEvent. It might be forcefully terminated."
                )
            else:
                self.logger.info(
                    f"MainWindow: Server thread ({current_thread_for_shutdown.objectName() or 'Unnamed'}) stopped successfully during closeEvent."
                )
        else:
            self.logger.info("MainWindow: Server thread was not running or already stopped at closeEvent.")

        self.logger.info("MainWindow: Proceeding with super().closeEvent().")
        closing_dialog.close()  # Hide the dialog just before closing
        super().closeEvent(event)
        self.logger.info("MainWindow: Application closed.")  # This log might not appear if event loop is gone


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Apply global dark Matrix theme with monospaced font
    app.setStyleSheet("""
        QWidget {
            background-color: #181A1B;
            color: #00FF41;
            font-family: "Fira Mono", "Consolas", "Courier New", monospace;
        }
        QPushButton {
            background-color: #222;
            color: #00FF41;
            border: 1px solid #00FF41;
            border-radius: 4px;
            padding: 6px 12px;
        }
        QPushButton:hover {
            background-color: #1a2b1a;
        }
        QTextEdit, QLineEdit {
            background-color: #101010;
            color: #00FF41;
            font-family: "Fira Mono", "Consolas", "Courier New", monospace;
            border: 1px solid #00FF41;
        }
    """)
    window = MainWindow()
    window.setWindowTitle("mcp@mcw")
    window.show()
    exit_code = app.exec()
    sys.exit(exit_code)
