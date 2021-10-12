"""Generate a PDF based on the intermediate json output."""
import json

import pylatex

filepath = "../../report/report.json"

with open(filepath) as fp:
    result_list = json.load(fp)

doc = pylatex.Document()

for result in result_list:
    with doc.create(pylatex.Section(result["name"])):
        doc.append(result["blurb"])
        with doc.create(pylatex.Subsection("Procedure")):
            with doc.create(pylatex.lists.Itemize()) as procedure_list:
                for step in result["procedure"]:
                    procedure_list.add_item(step[0])
                    try:
                        with doc.create(pylatex.lists.Itemize()) as detail_list:
                            for detail in step[1:]:
                                detail_list.add_item(detail)
                    except IndexError:
                        pass

doc.generate_pdf("../../report/report", clean_tex=False)
