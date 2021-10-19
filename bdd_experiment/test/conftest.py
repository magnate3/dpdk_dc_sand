"""Stub fixture for writing extra details to a report."""

import pytest


class Reporter:
    """Provides mechanisms to log steps taken in a test."""

    def __init__(self, data: list) -> None:
        self._data = data

    def detail(self, message: str) -> None:
        """Report a low-level detail, associated with the current :meth:`step`."""
        self._data.append({"$msg_type": "detail", "message": message})


@pytest.fixture
def pdf_report(request) -> Reporter:
    """Fixture for logging steps in a test."""
    data = [{"$msg_type": "test_info", "blurb": request.node.function.__doc__}]
    request.node.user_properties.append(("pdf_report_data", data))
    return Reporter(data)
