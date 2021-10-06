# noqa: D100
import logging

import pytest

from pdf_logger.json_handler import JsonHandler

# Take charge of the logging.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Make sure we're using the Json Handler created for the purpose.
my_handler = JsonHandler("report", "report")
logger.addHandler(my_handler)

# Because the Json handler keeps its own internal state, to help with debugging
# that, we add a normal file handler as well, which will just log everything. It
# makes picking through the JSON output easier.
file_handler = logging.FileHandler("report/raw_output.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"))
logger.addHandler(file_handler)


@pytest.fixture
def skarab_dsim_fixture():
    """SKARAB DSim."""
    logging.info("Setting up SKARAP Dsim")
    logging.debug("skarab020406, firmware 0.1")
    yield "SKARAB DSIM"
    logging.info("Tearing down SKARAB dsim.")
    logging.debug("DSim teardown finished.")


def pytest_runtest_logstart(nodeid, location):
    """Indicate that we are starting the test."""
    logging.warning(f"START {nodeid} {location}")


def pytest_runtest_logfinish(nodeid, location):
    """Emit a warning when a test is finished."""
    logging.warning(f"STOP {nodeid} {location}")


# def pytest_report_collectionfinish(config, startdir, items: Sequence[pytest.Item]):
#     """We can get a list of tests that are actually going to run."""
#     logging.error("Actually running the hook.")
#     for test in items:
#         logging.error(test)
#     logging.error("Finished with the for loop.")


# def pytest_assertion_pass(item, lineno, orig, expl):
#     logging.debug(f"Assertion passed: {expl}")

# def pytest_assertrepr_compare(config, op, left, right):
#     logging.error(f"Assert failed: {left} {op} {right}")


def pytest_fixture_setup(fixturedef, request):
    """Log the fixture's docstring."""
    extra = {"fixture_docstring": fixturedef.func.__doc__}
    logging.debug(f"FIXTURE_SETUP {extra}", extra=extra)


# def pytest_fixture_post_finalizer(fixturedef, request):
#     logging.debug("In the post-fixture-finalizer hook.")


# def pytest_runtest_setup(item: pytest.Item):
#     logging.debug(f"TEST SETUP {item.function.__name__}")


def pytest_runtest_call(item: pytest.Item):
    """Log the docstring of the function before you actually run it."""
    extra = {"test_name": item.function.__name__, "test_docstring": item.function.__doc__}
    logging.debug(f"BLURB {extra}", extra=extra)


# def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item):
#     logging.debug(f"TEST TEARDOWN {item.function.__name__}")


def pytest_runtest_logreport(report):
    """Log the outcome of the test.

    We only do the actual test itself, not setup or teardown.
    """
    if report.when == "call":  # As opposed to setup and teardown. This is where to get the actual outcome of the test.
        extra = {"outcome": report.outcome, "longrepr": str(report.longrepr)}
        logging.debug(f"OUTCOME {extra}", extra=extra)


def pytest_sessionfinish():
    """Close out the files.

    Later on we'll be having latex in this. But not yet, just json for now.
    """
    logging.shutdown()  # If we don't do this, the output files aren't closed yet.
