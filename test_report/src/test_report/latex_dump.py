"""Generate a PDF based on the intermediate JSON output."""
import argparse
import importlib.resources
import inspect
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from docutils.core import publish_parts
from dotenv import dotenv_values
from pylatex import (
    Command,
    Document,
    FlushLeft,
    LongTable,
    MiniPage,
    MultiColumn,
    NoEscape,
    Section,
    Subsection,
    Subsubsection,
    TextColor,
)
from pylatex.labelref import Hyperref
from pylatex.utils import bold


def readable_duration(duration: float) -> str:
    """Change a duration in seconds to something human-readable."""
    if duration < 60:
        return f"{duration:.3f} s"
    else:
        seconds = duration % 60
        duration /= 60
        if duration < 60:
            return f"{int(duration)} m {seconds:.3f} s"
        else:
            minutes = int(duration % 60)
            duration /= 60
            return f"{int(duration)} h {minutes} m {seconds:.3f} s"


def docstring2latex(text: str) -> str:
    """Turn a section of ReStructured Text (like a docstring) into LaTeX."""
    return publish_parts(source=inspect.cleandoc(text), writer_name="latex")["body"]


@dataclass
class Detail:
    """A message logged by ``pdf_report.detail``."""

    message: str
    timestamp: float


@dataclass
class Plot:
    """A plot requested by ``pdf_report.plot``.

    This class does the drawing, given the details.
    """

    #: The X-axis.
    x: np.ndarray
    y: np.ndarray
    caption: str
    xlabel: str
    ylabel: str
    legend_labels: Union[str, List[str]]

    def get_pgf_str(self) -> str:
        """Output the PGF-plots string for the data represented here."""
        with plt.style.context("ggplot"):
            fig, ax = plt.subplots()
            if self.y.ndim > 1:
                for y, legend_label in zip(self.y, self.legend_labels):
                    ax.plot(self.x, y, label=legend_label)
            else:
                ax.plot(self.x, self.y, label=self.legend_labels)
            ax.set(
                title=self.caption,
                xlabel=self.xlabel,
                ylabel=self.ylabel,
            )
            ax.legend()
            ax.grid()
        tikzplotlib.clean_figure(fig)
        return tikzplotlib.get_tikz_code(fig, table_row_sep=r"\\")


@dataclass
class Step:
    """A step created by ``pdf_report.step``."""

    message: str
    details: List[Union[Detail, Plot]] = field(default_factory=list)


@dataclass
class Result:
    """A single test execution.

    This combines the setup, call teardown phases, which appear as separate
    lines in the report.json file.
    """

    nodeid: str
    name: str
    blurb: str
    steps: List[Step] = field(default_factory=list)
    outcome: Literal["passed", "failed", "skipped"] = "failed"
    failure_message: Optional[str] = None
    start_time: Optional[float] = None
    duration: float = 0.0
    config: dict = field(default_factory=dict)


@dataclass
class ConfigParam:
    """A configuration parameter describing software / hardware used in the test run."""

    name: str
    value: str


def parse(input_data: list) -> Tuple[List[ConfigParam], List[Result]]:
    """Parse the data written by pytest-reportlog.

    .. todo::

        flake8 complains that this function is too complex. It may have a point.
        It may be worthwhile breaking it up into smaller functions.

    Parameters
    ----------
    input_data
        A Python list, which should have been loaded from pytest-reportlog's
        JSON output.

    Returns
    -------
    Lists of :class:`ConfigParam` and :class:`Result` objects representing the
    test configuration and the results of all the tests logged in the JSON input.
    """
    test_configuration: list[ConfigParam] = []
    results: list[Result] = []
    for line in input_data:
        if line["$report_type"] == "TestConfiguration":
            # Tabulate this somehow. It'll need to modify the return.
            line.pop("$report_type")
            for config_param_name, config_param_value in line.items():
                test_configuration.append(ConfigParam(config_param_name, config_param_value))
            continue
        if line["$report_type"] != "TestReport":
            continue
        nodeid = line["nodeid"]
        if not results or results[-1].nodeid != nodeid:
            # It's the first time we've seen this nodeid (otherwise we merge
            # with the existing Result).
            results.append(Result(nodeid, line["location"][2], ""))
        result = results[-1]
        # The teardown phase has all the log messages, so we ignore the setup and call phases.
        if line["when"] == "teardown":
            for prop in line["user_properties"]:
                if prop[0] == "pdf_report_data":
                    for msg in prop[1][:]:
                        assert isinstance(msg, dict)
                        msg_type = msg["$msg_type"]
                        if msg_type == "step":
                            # I realise that the extended conditional expression in the list
                            # comprehension here is starting to verge on inelegant, but it
                            # works well enough without having to have much more verbose code.
                            details = [
                                Detail(detail["message"], detail["timestamp"])
                                if detail["$msg_type"] == "detail"
                                else Plot(
                                    np.array(detail["x"]),
                                    np.array(detail["y"]),
                                    detail["caption"],
                                    detail["xlabel"],
                                    detail["ylabel"],
                                    detail["legend_labels"],
                                )
                                if detail["$msg_type"] == "plot"
                                else Detail("", 0)  # Blank catch-all.
                                for detail in msg["details"]
                            ]
                            result.steps.append(Step(msg["message"], details))
                        elif msg_type == "test_info":
                            if not result.blurb:
                                result.blurb = msg["blurb"]
                            if result.start_time is None:
                                result.start_time = float(msg["test_start"])
                        elif msg_type == "config":
                            msg.pop("$msg_type")  # We don't need this.
                            result.config.update(msg)
                        else:
                            raise ValueError(f"Do not know how to parse $msg_type of {msg_type!r}")
        # If teardown fails, the whole test should be seen as failing
        if line["outcome"] != "passed" or line["when"] == "call":
            result.outcome = line["outcome"]
        # The test duration will be the sum of setup, call and teardown.
        result.duration += line["duration"]
        try:
            failure_message = line["longrepr"]["reprcrash"]["message"]
        except (KeyError, TypeError):
            pass
        else:
            # TODO: if multiple phases have failure messages, we probably want to
            # collect them all. We could also collect the more detailed messages.
            result.failure_message = failure_message
    return test_configuration, results


def fix_test_name(test_name: str) -> str:
    """Change a test's name from a pytest one to a more human-friendly one."""
    return " ".join([word.capitalize() for word in test_name.split("_") if word != "test"])


def document_from_json(input_data: Union[str, list]) -> Document:
    """Take a test result and generate a :class:`pylatex.Document` for a report.

    Parameters
    ----------
    input_data
        Either the list of parsed JSON entries, or a path to the report file
        generated by :samp:`pytest --report-log={filename}`.

    Returns
    -------
    doc
        A document
    """
    try:
        result_list = []
        with open(input_data) as fp:
            for line in fp:
                result_list.append(json.loads(line))
    except TypeError:
        result_list = input_data

    test_configuration, results = parse(result_list)

    # Get information from the .env file, such as tester's name, which shouldn't
    # really be in git. Allow environment to override
    config = {**dotenv_values(), **os.environ}

    doc = Document(
        document_options=["11pt", "english", "twoside"],
        inputenc=None,  # katdoc inputs inputenc with specific options, so prevent a clash
    )
    today = date.today()  # TODO: should store inside the JSON
    doc.set_variable("theAuthor", config.get("TESTER_NAME", "Unknown"))
    doc.set_variable("docDate", today.strftime("%d %B %Y"))
    doc.preamble.append(NoEscape(importlib.resources.read_text("test_report", "preamble.tex")))
    doc.append(Command("title", "Integration Test Report"))
    doc.append(Command("makekatdocbeginning"))

    with doc.create(Section("Test Configuration")) as config_section:
        with config_section.create(LongTable(r"|r|l|")) as config_table:
            config_table.add_hline()
            config_table.add_row([bold("Configuration Parameter"), bold("Value")])
            config_table.add_hline()
            for config_param in test_configuration:
                config_table.add_row([config_param.name, config_param.value])
                config_table.add_hline()
            config_table.add_row(["Some other software version", "x.y.ZZ"])
            config_table.add_hline()

    with doc.create(Section("Result Summary")) as summary_section:
        with summary_section.create(LongTable(r"|r|l|r|")) as summary_table:
            total_duration: float = 0.0
            passed = 0
            failed = 0
            summary_table.add_hline()
            summary_table.add_row(
                [
                    bold("Test"),
                    bold("Result"),
                    bold("Duration"),
                ]
            )
            summary_table.add_hline()
            for result in results:
                summary_table.add_row(
                    # PyLatex strips (among other things) underscores from label names, apparently they're dangerous.
                    # See: https://jeltef.github.io/PyLaTeX/latest/pylatex/pylatex.labelref.html#pylatex.labelref.Marker
                    [
                        Hyperref(f"subsec:{result.name.replace('_', '')}", fix_test_name(result.name)),
                        TextColor("green" if result.outcome == "passed" else "red", result.outcome),
                        readable_duration(result.duration),
                    ]
                )
                summary_table.add_hline()
                total_duration += result.duration
                if result.outcome == "passed":
                    passed += 1
                else:
                    failed += 1
        summary_section.append(f"{len(results)} tests run, with {passed} passing and {failed} failing.\n")
        summary_section.append(f"Total test duration: {readable_duration(total_duration)}")

    with doc.create(Section("Detailed Test Results")) as section:
        for result in results:
            with section.create(Subsection(fix_test_name(result.name), label=result.name)):
                section.append(NoEscape(docstring2latex(result.blurb) + "\n\n"))
                section.append("Outcome: ")
                section.append(TextColor("green" if result.outcome == "passed" else "red", result.outcome.upper()))
                section.append(Command("hspace", "1cm"))
                section.append(f"Test start time: {datetime.fromtimestamp(float(result.start_time)).strftime('%T')}")
                section.append(Command("hspace", "1cm"))
                section.append(f"Duration: {readable_duration(result.duration)} seconds\n")

                with section.create(LongTable(r"|l|p{0.4\linewidth}|")) as config_table:
                    config_table.add_hline()
                    config_table.add_row((MultiColumn(2, align="|c|", data=bold("Test Configuration")),))
                    config_table.add_hline()
                    for key, value in result.config.items():
                        config_table.add_row([key.capitalize(), value])
                        config_table.add_hline()

                with section.create(Subsubsection("Procedure", label=False)) as procedure:
                    with section.create(LongTable(r"|l|p{0.7\linewidth}|")) as procedure_table:
                        for step in result.steps:
                            procedure_table.add_hline()
                            procedure_table.add_row((MultiColumn(2, align="|l|", data=bold(step.message)),))
                            procedure_table.add_hline()
                            for detail in step.details:
                                if isinstance(detail, Detail):
                                    procedure_table.add_row(
                                        [
                                            f"{readable_duration(detail.timestamp - result.start_time)}",
                                            detail.message,
                                        ]
                                    )
                                elif isinstance(detail, Plot):
                                    mp = MiniPage(width=NoEscape(r"0.6\textwidth"))
                                    mp.append(NoEscape(detail.get_pgf_str()))
                                    procedure_table.add_row((MultiColumn(2, align="|c|", data=mp),))
                                procedure_table.add_hline()

                    if result.failure_message:
                        with procedure.create(FlushLeft()) as failure_message:
                            failure_message.append(result.failure_message)

    return doc


def main():
    """Convert a JSON report to a PDF."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Output of pytest --report-log=...")
    parser.add_argument("pdf", help="PDF file to write")
    args = parser.parse_args()
    doc = document_from_json(args.input)
    if args.pdf.endswith(".pdf"):
        args.pdf = args.pdf[:-4]  # Strip .pdf suffix, because generate_pdf appends it
    with tempfile.NamedTemporaryFile(mode="w", prefix="latexmkrc") as latexmkrc:
        with importlib.resources.path("test_report", "katdoc.sty") as katdoc_sty_path:
            # TODO: latexmk uses Perl, which has different string parsing to
            # Python. If the path contains both a single quote and a special
            # symbol it will not produce a valid Perl string.
            parent_dir = str(katdoc_sty_path.parent)
            latexmkrc.write(f"ensure_path('TEXINPUTS', {parent_dir!r})\n")
            latexmkrc.flush()
            doc.generate_pdf(args.pdf, compiler="latexmk", compiler_args=["--pdf", "-r", latexmkrc.name])
