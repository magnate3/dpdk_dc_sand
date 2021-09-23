"""Module declaring LatexHandler Class."""

import logging
from typing import Union


class LatexHandler(logging.Handler):
    """Output latex-formatted log files."""
    def __init__(
        self, output_filename: str = "log", destination_dir: str = ".", level: Union[int, str] = logging.INFO
    ) -> None:
        super().__init__(level=level)
        # TODO: Check whether directory exists first, if not, create it.
        self._logfile = open(f"{destination_dir}/{output_filename}.tex", mode="w")
        self._logfile.write(
            r"\documentclass{article}"
            r"\begin{document}"
        )

    def close(self):
        """Finish off and safely close the output file."""
        self._logfile.write(r"\end{document}")
        self._logfile.write("\n")
        self._logfile.close()
        super().close()

    def emit(self, record: logging.LogRecord):
        """Write a log message to the output file."""
        self._logfile.write(f"{record.getMessage()}\n\n")  # 2x`\n` because latex needs a blank line for a new paragraph


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    my_handler = LatexHandler()
    logger.addHandler(my_handler)

    logger.debug("Debug")
    logger.info("Info")
    logger.warning("Warning")
    logger.error("Error")
    logger.critical("Critical")
