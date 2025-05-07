import sys

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp_instance():
    """Create a QApplication instance for the test session, if one doesn't exist."""
    app = QApplication.instance()
    if app is None:
        # sys.argv is needed for QApplication constructor, use a default if not running from a context that provides it.
        app = QApplication(sys.argv if hasattr(sys, "argv") else [])
    return app
