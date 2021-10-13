"""Generate a PDF based on the intermediate json output."""
import json
from typing import Union

from pylatex import Document, LongTable, MultiColumn, Section, Subsection, Subsubsection
from pylatex.utils import bold


def dump_latex_from_json(input_data: Union[str, list]) -> str:
    """Take a test result and generate latex for a report.

    This function can be used 'live', on the nested list / dictionary thing,
    or it can be given a path to a json file and it'll load from that.

    Parameters
    ----------
    input_data
        Either the list of test result dictionaries, or a path to the json
        file containing the dump of said list.

    Returns
    -------
    str
        A string with the latex output. Not a complete document, just the
        section which has been known as 'Detailed Results' or some such. This
        will need to be fitted into a template.
    """
    try:
        with open(input_data) as fp:
            result_list = json.load(fp)
    except TypeError:
        result_list = input_data

    doc = Document()
    # TODO: Add a summary table.
    with doc.create(Section("Test Results")) as section:
        for result in result_list:
            with section.create(Subsection(result["name"])):
                section.append(result["blurb"])
                with section.create(Subsubsection("Procedure", label=False)):
                    with section.create(LongTable(r"|l|p{0.7\linewidth}|")) as procedure_table:
                        for step in result["procedure"]:
                            procedure_table.add_hline()
                            procedure_table.add_row((MultiColumn(2, align="|l|", data=bold(step[0])),))
                            procedure_table.add_hline()
                            try:
                                for detail in step[1:]:
                                    # TODO: timestamps for the actual steps.
                                    procedure_table.add_row(["timestamp", detail])
                                    procedure_table.add_hline()
                            except IndexError:
                                pass

        return section.dumps()
