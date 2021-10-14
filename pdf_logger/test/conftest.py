"""Fixtures for capturing Hooks and fixtures for logging Pytest's output to a PDF.

Commented out functions are there to remind me that the hooks exist, in case I
should need them in future tinkerings.
"""

from typing import Optional

import pytest


class Reporter:
    """Provides mechanisms to log steps taken in a test."""

    def __init__(self, data: list) -> None:
        self._data = data
        self._cur_step: Optional[list] = None

    def step(self, message: str) -> None:
        """Report the start of a high-level step."""
        self._cur_step = []
        self._data.append({"$msg_type": "step", "message": message, "details": self._cur_step})

    def detail(self, message: str) -> None:
        """Report a low-level detail, associated with the previous call to :meth:`step`."""
        if self._cur_step is None:
            raise ValueError("Cannot have detail without a current step")
        self._cur_step.append({"$msg_type": "detail", "message": message})


@pytest.fixture
def pdf_report(request) -> Reporter:
    """Fixture for logging steps in a test."""
    data = [{"$msg_type": "test_info", "blurb": request.node.function.__doc__}]
    request.node.user_properties.append(("pdf_report_data", data))
    return Reporter(data)


@pytest.fixture
def skarab_dsim_fixture(pdf_report):
    """SKARAB DSim."""
    pdf_report.step("Setting up SKARAP Dsim")
    pdf_report.detail("skarab020406, firmware 0.1")
    yield "SKARAB DSIM"
    pdf_report.step("Tearing down SKARAB dsim.")
    pdf_report.detail("DSim teardown finished.")
