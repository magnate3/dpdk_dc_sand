"""Fixtures for capturing Hooks and fixtures for logging Pytest's output to a PDF.

Commented out functions are there to remind me that the hooks exist, in case I
should need them in future tinkerings.
"""
import time
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
import pytest


class Reporter:
    """Provides mechanisms to log steps taken in a test."""

    def __init__(self, data: list) -> None:
        self._data = data
        self._cur_step: Optional[list] = None

    def config(self, **kwargs) -> None:
        """Report the test cconfiguration."""
        test_config = {"$msg_type": "config"}
        test_config.update(kwargs)
        self._data.append(test_config)

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
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        caption: Optional[str] = "",
        xlabel: Optional[str] = "",
        ylabel: Optional[str] = "",
        legend_labels: Optional[Union[str, List[str]]] = "",
    ) -> None:
        """Capture numerical data for plotting.

        Parameters
        ----------
        x
            X-data for plotting. Must be one-dimensional.
        y
            Y-data for plotting. Can be up to two-dimensional, but the length
            of the second dimension must agree with the length of `x`.
        caption
            Title for the graph.
        xlabel
            Label for the X-axis.
        ylabel
            Label for the Y-axis.
        legend_labels
            Legend labels for the various sets of data plotted. Optional only
            in single-dimension plots, if a 2D `y` is given, a list of labels
            must be passed.

        Raises
        ------
        ValueError
            If called before :func:`Report.step`, as the plot must be associated
            with a step in the test procedure.
        """
        # Coerce to np.ndarray for data validation.
        x = np.asarray(x)
        y = np.asarray(y)

        # I must admit that I'm nervous about using `assert` for this but I
        # guess that we're unlikely ever to run a test suite with `-O`.
        assert x.ndim == 1, f"x has {x.ndim} dimensions, expected 1!"
        assert y.ndim <= 2, "Can't have y with more than 2 dimensions!"
        assert x.size == y.shape[-1], "x and y must have same length for plotting!"
        if y.ndim > 1:
            assert len(legend_labels) == y.shape[0], "If y is 2-dimensional, we need legend labels."

        # Moving swiftly along.
        if self._cur_step is None:
            raise ValueError("Cannot have a plot without a current step")

        self._cur_step.append(
            {
                "$msg_type": "plot",
                "y": y.tolist(),
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
