# noqa: D100
from pdf_logger.convenience import check, detail, step


def test_channelisation(skarab_dsim_fixture):
    """Channelisation Test.

    Fulfils requirement ABC.
    """
    step("Select a random channel.")
    detail("Random channel selected: 42")

    step("Sweep frequencies across that channel and capture heaps.")
    detail("Setting Dsim frequency to 1")
    detail("Heap captured.")
    detail("Setting Dsim frequency to 2")
    detail("Heap captured.")
    detail("Setting Dsim frequency to 3")
    detail("Heap captured.")

    step("Check peak is in centre of the channel.")
    check(1 in [1, 2, 3], "Peak not in centre of channel!")
    detail("Peak located in centre of channel.")

    step("Check response far away is below -62 dB")
    check(0 < 1, "Rejection not good enough!")
    detail("Out of channel rejection is within spec.")


def test_delay_tracking(skarab_dsim_fixture):
    """Delay tracking test.

    Fulfils requirement FGD.
    """
    step("Set delay model on Antenna 1")
    detail("Setting delay model on antenna 1")
    check(1 <= 1.2, "Delay model not correctly configured.")
    detail("Antenna 1 delay model confirmed as 345.")

    step("Capture 3 heaps, compare the phase, check for within spec.")
    detail("Captured heap 1")
    detail("Captured heap 2")
    detail("Captured heap 3")
    check(2 < 1, "Phase not within spec!")


def test_baselines(skarab_dsim_fixture):
    """Baseline test.

    Blurb.

    Fulfils another requirement.
    """
    step("Set something.")
    detail("Checking that something is set.")
    detail("Confirmed, something is set properly.")
    check(7 < 8, "Seven not less than 8.")
