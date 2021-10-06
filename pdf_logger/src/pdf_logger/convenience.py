"""Convenience functions.

I don't have a strong opinion about this, it just seems like less typing, though
at the cost of slightly obscuring the fact that it's just the plain Python
logging system operating in the tests.
"""
import logging

step = logging.info
detail = logging.debug


def check(expression: bool, argument: str):
    """Perform an assertion, log if it fails.

    This function came to be because pytest doesn't make it very easy to log the
    argument of the assertion statements.

    This lets the argument end up in the logs as an error, but it messes up
    pytest's cli output. Since this is intended to be run on a CI/CD server,
    this may not be a problem, as the report is what a human will actually
    read.
    """
    try:
        assert expression, argument
    except AssertionError:
        logging.error(argument)
        raise
