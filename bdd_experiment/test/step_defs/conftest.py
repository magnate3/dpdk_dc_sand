# noqa: D100

from pytest_bdd import given


@given("a SKARAB dsim")
def skarab_dsim_fixture(pdf_report):
    """SKARAB DSim."""
    pdf_report.detail("skarab020406, firmware 0.1")
    yield "SKARAB DSIM"
    pdf_report.detail("DSim teardown finished.")
