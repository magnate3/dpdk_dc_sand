# noqa: D100

from pytest_bdd import given, parsers, scenarios, then, when

scenarios("../features/channelisation.feature")


@given("a random channel", target_fixture="channel")
def random_channel(pdf_report):  # noqa: D103
    channel = 42
    pdf_report.detail(f"Random channel selected: {channel}")
    return channel


@when("sweeping frequencies across a channel and capturing heaps", target_fixture="heaps")
def sweep_channel(pdf_report, channel):  # noqa: D103
    pdf_report.detail(f"Setting Dsim frequency to {channel}.1")
    pdf_report.detail("Heap captured.")
    pdf_report.detail(f"Setting Dsim frequency to {channel}.2")
    pdf_report.detail("Heap captured.")
    pdf_report.detail(f"Setting Dsim frequency to {channel}.3")
    pdf_report.detail("Heap captured.")
    return [1, 10, 2]


@then("the peak is in the centre of the channel")
def peak_centre(pdf_report, heaps):  # noqa: D103
    assert max(heaps) == heaps[1]


@then(parsers.parse("response outside the channel is below {db:g} dB"))
def no_leakage(pdf_report, heaps, db):  # noqa: D103
    assert heaps[0] < 10 ** (0.1 * db) * heaps[1]
