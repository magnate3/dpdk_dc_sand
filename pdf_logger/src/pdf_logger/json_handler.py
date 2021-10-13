"""Module declaring LatexHandler Class."""

import datetime
import json
import logging
import os
import shutil
from enum import Enum, auto
from typing import Union

import pkg_resources
from dotenv import load_dotenv
from mako.template import Template

from .latex_dump import dump_latex_from_json

# Get information from the .env file, such as tester's name, which shouldn't
# really be in git.
load_dotenv()


class State(Enum):
    """Enumeration of states for the state machine."""

    IDLE = auto()
    FIXTURE_SETUP = auto()
    PROCEDURE = auto()
    FIXTURE_TEARDOWN = auto()


def blank_test_result():
    """Generate a dictionary with the necessary keywords for a blank test to slot into."""
    return {"name": "", "blurb": "", "fixture": "", "procedure": [], "outcome": ""}


class JsonHandler(logging.Handler):
    """Output json-serialised log files."""

    def __init__(
        self, output_filename: str = "log", destination_dir: str = ".", level: Union[int, str] = logging.DEBUG
    ) -> None:
        super().__init__(level=level)
        if not os.path.exists(destination_dir):
            os.mkdir(destination_dir)
        self.output_path = os.path.join(destination_dir, output_filename)  # Won't have an extension, this is fine.
        self.destination_dir = destination_dir
        self.result_list = []
        self.current_test = blank_test_result()
        self.current_step = []
        self.state: State = State.IDLE

    def close(self):
        """Dump the result list as a json file and try to make a latex file too.

        The latex part is tentative for now, the json is there so that the
        latex process can be re-run if necessary because something is messed up.
        """
        # TODO: maybe check that the last thing in the result list is the current result?
        # Append if not an empty dictionary.
        with open(f"{self.output_path}.json", "w") as fp:
            json.dump(self.result_list, fp, indent=2)

        # TODO: I'm sure there's a better way to do this context manager.
        # It may also end up getting replaced with a mako template render, so
        # perhaps not worth worrying just yet.
        template_path = os.path.join(pkg_resources.resource_filename(__name__, ""), "preamble_template.tex")
        in_template = Template(filename=template_path)
        with open(self.output_path + ".tex", "w") as out_file:
            date = datetime.date.today()
            out_file.write(in_template.render(TESTER_NAME=os.getenv("TESTER_NAME"), DATE=date.strftime("%d %B %Y")))
            out_file.write(dump_latex_from_json(self.result_list))
            out_file.write("\\end{document}\n")

        # These things are really part of the template and you need them to render
        # the actual document. Not sure of a better way to get them there.
        if not os.path.exists(os.path.join(self.destination_dir, "katdoc.sty")):
            shutil.copy(os.path.join(pkg_resources.resource_filename(__name__, ""), "katdoc.sty"), self.destination_dir)
        if not os.path.exists(os.path.join(self.destination_dir, "sarao_logo.png")):
            shutil.copy(
                os.path.join(pkg_resources.resource_filename(__name__, ""), "sarao_logo.png"), self.destination_dir
            )
        super().close()

    def emit(self, record: logging.LogRecord):  # noqa: C901
        """Record log message in the result list.

        Each log message is a 'tick' for the state machine, the current state
        and the contents of the message determine where the message will go,
        or whether it will just tick the machine over into the next state.

        This isn't a very robust state machine, and assumes that there will
        never be any unexpected log messages. Strange behaviour may result if
        pytest changes things in a subsequent version (though it seems to be
        fairly stable between versions in this regard).
        """
        msg = record.getMessage()

        if self.state == State.IDLE:
            if msg.startswith("START"):
                self.state = State.FIXTURE_SETUP
        elif self.state == State.FIXTURE_SETUP:
            if msg.startswith("FIXTURE_SETUP"):
                self.current_test["fixture"] = record.fixture_docstring
            elif record.levelno == logging.INFO:
                if len(self.current_step):
                    self.current_test["procedure"].append(self.current_step)
                self.current_step = [msg]
            elif msg.startswith("BLURB"):
                if len(self.current_step):
                    self.current_test["procedure"].append(self.current_step)
                    self.current_step = []
                self.state = State.PROCEDURE
                self.current_test["name"] = record.test_name
                self.current_test["blurb"] = record.test_docstring
            else:
                self.current_step.append(msg)
        elif self.state == State.PROCEDURE:
            if msg.startswith("OUTCOME"):
                if len(self.current_step):
                    self.current_test["procedure"].append(self.current_step)
                    self.current_step = []
                self.current_test["outcome"] = record.outcome
                self.current_test["procedure"].append([f"Test {record.outcome.upper()}."])
                if record.outcome == "failed":
                    self.current_test["detailed_outcome"] = record.longrepr
                self.state = State.FIXTURE_TEARDOWN
            elif record.levelno == logging.INFO:
                if len(self.current_step):
                    self.current_test["procedure"].append(self.current_step)
                self.current_step = [msg]
            else:
                self.current_step.append(msg)
        elif self.state == State.FIXTURE_TEARDOWN:
            if record.levelno == logging.INFO:
                if len(self.current_step):
                    self.current_test["procedure"].append(self.current_step)
                self.current_step = [msg]
            elif msg.startswith("STOP"):
                if len(self.current_step):
                    self.current_test["procedure"].append(self.current_step)
                    self.current_step = []
                    self.result_list.append(self.current_test)
                    self.current_test = blank_test_result()
                self.state = State.IDLE
            else:
                self.current_step.append(msg)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    my_handler = JsonHandler()
    logger.addHandler(my_handler)

    logger.debug("Debug", extra={"foo": "bar"})
    logger.info("Info")
    logger.warning("Warning")
    logger.error("Error")
    logger.critical("Critical")
