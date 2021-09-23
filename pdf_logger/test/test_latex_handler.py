# noqa: D100
import logging

from pdf_logger.latex_handler import LatexHandler


def test_latex_handler():
    """Test :class:`LatexHandler` by emitting a few simple log messages."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    my_handler = LatexHandler()
    logger.addHandler(my_handler)

    logger.debug("Debug")
    logger.info("Info")
    logger.warning("Warning")
    logger.error("Error")
    logger.critical("Critical")

    assert True
