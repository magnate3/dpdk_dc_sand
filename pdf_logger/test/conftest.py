"""Fixtures for capturing Hooks and fixtures for logging Pytest's output to a PDF.

Commented out functions are there to remind me that the hooks exist, in case I
should need them in future tinkerings.
"""
import time
from typing import List, Optional, Union

import numpy as np
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
        self._cur_step.append({"$msg_type": "detail", "message": message, "timestamp": time.time()})

    def plot(
        self,
        x: np.ndarray,
        y: Union[np.ndarray, List[np.ndarray]],
        caption: str,
        xlabel: str = "",
        ylabel: str = "",
        legend_labels: Union[str, List[str]] = "",
    ) -> None:
        """Stick a plot in the report."""
        if self._cur_step is None:
            raise ValueError("Cannot have a plot without a current step")
        if isinstance(y, list):
            y = [array.tolist() for array in y]
        else:
            y = y.tolist()
        self._cur_step.append(
            {
                "$msg_type": "plot",
                "y": y,
                "x": x.tolist(),
                "caption": caption,
                "xlabel": xlabel,
                "ylabel": ylabel,
                "legend_labels": legend_labels,
            }
        )


@pytest.fixture
def pdf_report(request) -> Reporter:
    """Fixture for logging steps in a test."""
    data = [{"$msg_type": "test_info", "blurb": request.node.function.__doc__, "test_start": time.time()}]
    request.node.user_properties.append(("pdf_report_data", data))
    return Reporter(data)


@pytest.fixture
def skarab_dsim_fixture(pdf_report):
    """SKARAB DSim."""
    pdf_report.step("Setting up SKARAB Dsim")
    time.sleep(0.7)
    pdf_report.detail("skarab020406, firmware 0.1")
    yield "SKARAB DSIM"
    pdf_report.step("Tearing down SKARAB dsim.")
    time.sleep(0.3)
    pdf_report.detail("DSim teardown finished.")
