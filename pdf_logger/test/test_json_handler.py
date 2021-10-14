# noqa: D100


def test_channelisation(pdf_report, skarab_dsim_fixture):
    """Channelisation Test.

    Fulfils requirement ABC.
    """
    pdf_report.step("Select a random channel.")
    pdf_report.detail("Random channel selected: 42")

    pdf_report.step("Sweep frequencies across that channel and capture heaps.")
    pdf_report.detail("Setting Dsim frequency to 1")
    pdf_report.detail("Heap captured.")
    pdf_report.detail("Setting Dsim frequency to 2")
    pdf_report.detail("Heap captured.")
    pdf_report.detail("Setting Dsim frequency to 3")
    pdf_report.detail("Heap captured.")

    pdf_report.step("Check peak is in centre of the channel.")
    assert 1 in [1, 2, 3], "Peak not in centre of channel!"
    pdf_report.detail("Peak located in centre of channel.")

    pdf_report.step("Check response far away is below -62 dB")
    assert 0 < 1, "Rejection not good enough!"
    pdf_report.detail("Out of channel rejection is within spec.")


def test_delay_tracking(pdf_report, skarab_dsim_fixture):
    """Delay tracking test.

    Fulfils requirement FGD.
    """
    pdf_report.step("Set delay model on Antenna 1")
    pdf_report.detail("Setting delay model on antenna 1")
    assert 1 <= 1.2, "Delay model not correctly configured."
    pdf_report.detail("Antenna 1 delay model confirmed as 345.")

    pdf_report.step("Capture 3 heaps, compare the phase, check for within spec.")
    pdf_report.detail("Captured heap 1")
    pdf_report.detail("Captured heap 2")
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
    pdf_report.detail("Checking that something is set.")
    pdf_report.detail("Confirmed, something is set properly.")
    # Test a bare assert without a message
    assert 9 < 8
