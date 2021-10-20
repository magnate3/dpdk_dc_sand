# noqa: D100
from time import sleep

import numpy as np
from numpy.random import default_rng


def test_channelisation(pdf_report, skarab_dsim_fixture):
    """Channelisation Test.

    Fulfils requirement ABC.
    """
    pdf_report.step("Select a random channel.")
    pdf_report.detail("Random channel selected: 42")

    pdf_report.step("Sweep frequencies across that channel and capture heaps.")
    pdf_report.detail("Setting Dsim frequency to 1")
    sleep(1)
    pdf_report.detail("Heap captured.")
    pdf_report.detail("Setting Dsim frequency to 2")
    sleep(1)
    pdf_report.detail("Heap captured.")
    pdf_report.detail("Setting Dsim frequency to 3")
    sleep(1)
    pdf_report.detail("Heap captured.")

    pdf_report.step("Draw a plot to illustrate some of the data.")
    pdf_report.detail("Making up some numners.")
    xaxis = np.arange(10, dtype=np.float)
    desired = np.ones_like(xaxis)
    rng = default_rng(1)
    actual = desired - rng.standard_normal(size=desired.shape) * 0.05
    caption = "A plot to show you what we mean."
    pdf_report.plot(xaxis, [desired, actual], caption, "Frequency [MHz]", "Magnitude [linear]", ["desired", "actual"])

    assert 1 in [1, 2, 3], "Assertion not true!"
    pdf_report.detail("Plot shown.")


def test_delay_tracking(pdf_report, skarab_dsim_fixture):
    """Delay tracking test.

    Fulfils requirement FGD.
    """
    pdf_report.step("Set delay model on Antenna 1")
    pdf_report.detail("Setting delay model on antenna 1")
    sleep(1)
    assert 1 <= 1.2, "Delay model not correctly configured."
    pdf_report.detail("Antenna 1 delay model confirmed as 345.")

    pdf_report.step("Capture 3 heaps, compare the phase, check for within spec.")
    sleep(1)
    pdf_report.detail("Captured heap 1")
    sleep(1)
    pdf_report.detail("Captured heap 2")
    sleep(1)
    pdf_report.detail("Captured heap 3")
    a = 2
    b = 1
    assert a < b, "Phase not within spec!"


def test_baselines(pdf_report, skarab_dsim_fixture):
    """Baseline test.

    Blurb.

    Fulfils another requirement.
    """
    pdf_report.step("Set something.")
    sleep(1)
    pdf_report.detail("Checking that something is set.")
    pdf_report.detail("Confirmed, something is set properly.")
    # Test a bare assert without a message
    assert 9 < 8
