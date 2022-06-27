"""Sphinx configuration."""
# -- Project information -----------------------------------------------------

project = "CBF MK+ design dock mock-up"
copyright = "2022, NRF / SARAO"
author = "James Smith"

releasename = "Revision"
release = "A"


# These variable names pre-pended with `sarao_` just to avoid clashes with Sphinx ones.
sarao_project = "MeerKAT Extension"
sarao_doctype = "Design Document"
sarao_docnumber = "E1200-765-4321"
sarao_doc_classification = "Commercial in Confidence"

# Tuple of tuples with (Role, Name, Designation, Affiliation, Date)
sarao_contributors = (
    ("Submitted by", "J.N. Smith", "Digital Engineer", "SARAO", ""),
    ("Perused by", "A. van der Byl", "DSP Specialist", "SARAO", ""),
    ("Savaged by", "B. Merry", "Senior DSP Engineer", "SARAO", ""),
    ("Rejected by", "T. van Balla", "Functional Manager: DSP", "SARAO", ""),
)

releasename = "Revision"
release = "B"

# (Revision, Date, ECN Number, Comments)
sarao_doc_history = (
    ("A", "01 January 1970", "N/A", "Initial release for internal review."),
    ("B", "23 June 2022", "N/A", "Add Python logic, bring document into modern era."),
)

# -- General configuration ---------------------------------------------------

today_fmt = "%d %b %Y"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinxcontrib.bibtex"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# This lets us get numbered figures.
numfig = True

# --- Latex output options

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ("index", "index.tex", r"MK+ CBF Design Document mock-up", "James Smith", "howto"),
]

latex_additional_files = ["katdoc.sty", "sarao_logo.png"]

# Latex configuration in latest versions of Sphinx makes use of a dictionary of
# stuff rather than lots of individual things. I'm going to build it up here:
latex_elements = {}
latex_elements["papersize"] = "a4"
latex_elements["pointsize"] = "11pt"
latex_elements["maketitle"] = "\\makekatdocbeginning"

# This is going to take a bit of building up.
preamble = r"""\usepackage{katdoc}
\usepackage{mathtools}
\usepackage{graphicx}

\usepackage{tablefootnote}
\usepackage{color}
\usepackage{colortbl}
\usepackage{listings}
\lstset{breaklines, breakatwhitespace, basicstyle=\scriptsize\ttfamily}
% Magic incantation based on
% http://www.bollchen.de/blog/2011/04/good-looking-line-breaks-with-the-listings-package/
\lstset{prebreak=\raisebox{0ex}[0ex][0ex]{\ensuremath{\hookleftarrow}}}
\lstset{postbreak=\raisebox{0ex}[0ex][0ex]{\ensuremath{\hookrightarrow\space}}}

\newcommand{\docClient}{NRF (National Research Foundation)}
\newcommand{\docFacility}{South African Radio Astronomy Observatory (SARAO)}
\newcommand{\docFunction}{Engineering / Digital Signal Processing}
"""

# These values come from stuff that was determined up at the top of the file.
preamble += f"\\newcommand{{\\docProject}}{{{sarao_project}}}\n"
preamble += f"\\newcommand{{\\docType}}{{{sarao_doctype}}}\n"
preamble += f"\\newcommand{{\\docId}}{{{sarao_docnumber}}}\n"
preamble += f"\\newcommand{{\\docReleaseName}}{{{releasename}}}\n"
preamble += f"\\newcommand{{\\docRelease}}{{{release}}}\n"
preamble += f"\\newcommand{{\\docClassification}}{{{sarao_doc_classification}}}\n"
preamble += f"\\newcommand{{\\docCopyright}}{{\\copyright {copyright}}}\n"

# Contributors to the document in their various forms.
preamble += r"""%% Format: \addcontributor{Role}{Name}{Designation}{Affiliation}{Date}
\newcommand{\docApproval}{
"""
for role, name, designation, affiliation, date in sarao_contributors:
    preamble += f"\t\\addcontributor{{{role}}}{{{name}}}{{{designation}}}{{{affiliation}}}{{{date}}}\n"

# Document revision history.
preamble += r"""}
%% Format: \addchange{Revision}{Date}{ECN Number}{Comments}
\newcommand{\docHistory}{
"""
for revision, date, ecn_number, comments in sarao_doc_history:
    preamble += f"\\addchange{{{revision}}}{{{date}}}{{{ecn_number}}}{{{comments}}}"

# Software used in generating the document.
# TODO: this is going to need to be updated properly at some point.
# The info can probably be obtained procedurally somehow but I'm lazy right now.
preamble += r"""}
%% Format: \addprogram{Role}{Package}{Version}
\newcommand{\docSoftware}{
    \addprogram{Text processor}{pdf\LaTeX}{3.14159265-2.6-1.40.20 (TeX Live 2019/Debian)}{}
}

\renewcommand{\floatpagefraction}{0.8}
"""

latex_elements["preamble"] = preamble

# Bibliographies.
bibtex_bibfiles = [
    "applicable_docs.bib",
    "reference_docs.bib",
]
