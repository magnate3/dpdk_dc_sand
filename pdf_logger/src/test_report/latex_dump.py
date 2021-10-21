"""Generate a PDF based on the intermediate json output."""
import argparse
import importlib.resources
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
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
)
from pylatex.utils import bold


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
    xaxis: np.ndarray
    yaxis: np.ndarray
    caption: str
    xlabel: str
    ylabel: str
    legend_labels: Union[str, List[str]]

    def get_pgf_str(self) -> str:
        """Output the PGF-plots string for the data represented here."""
        plt.style.use("ggplot")
        if self.yaxis.shape[0] > 1:
            for yaxis, legend_label in zip(self.yaxis, self.legend_labels):
                plt.plot(self.xaxis, yaxis, label=legend_label)
        else:
            plt.plot(self.xaxis, self.yaxis, label=self.legend_labels)
        plt.title(self.caption)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()
        plt.grid(True)
        tikzplotlib.clean_figure()
        return tikzplotlib.get_tikz_code(table_row_sep=r"\\")


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
    duration: float = 0.0


def parse(input_data: list) -> List[Result]:
    """Parse the data written by pytest-reportlog.

    Parameters
    ----------
    input_data
        A Python list, which should have been loaded from pytest-reportlog's
        JSON output.

    Returns
    -------
    A list of :class:`Result` objects representing the results of all the tests
    logged in the JSON input.
    """
    results = []
    for line in input_data:
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
                        msg_type = msg["$msg_type"]
                        if msg_type == "step":
                            details = [
                                Detail(detail["message"], detail["timestamp"])
                                if detail["$msg_type"] == "detail"
                                else Plot(
                                    np.array(detail["xaxis"]),
                                    np.array(detail["yaxis"]),
                                    detail["caption"],
                                    detail["xlabel"],
                                    detail["ylabel"],
                                    detail["legend_labels"],
                                )
                                for detail in msg["details"]
                            ]
                            result.steps.append(Step(msg["message"], details))
                        elif msg_type == "test_info":
                            if not result.blurb:
                                result.blurb = msg["blurb"]
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
    return results


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
    results = parse(result_list)

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

    with doc.create(Section("Result Summary")) as summary_section:
        with summary_section.create(LongTable(r"|r|l|")) as summary_table:
            summary_table.add_hline()
            for result in results:
                summary_table.add_row([fix_test_name(result.name), result.outcome])
                summary_table.add_hline()

    with doc.create(Section("Detailed Test Results")) as section:
        for result in results:
            with section.create(Subsection(fix_test_name(result.name))):
                section.append(result.blurb)
                with section.create(Subsubsection("Summary", label=False)) as summary:
                    summary.append(bold(f"Test {result.outcome}\n\n"))
                    summary.append(f"Test duration: {result.duration:.3f} seconds\n")  # TODO: handle minutes / hours
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
                                            datetime.fromtimestamp(float(detail.timestamp)).strftime("%T.%f"),
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
