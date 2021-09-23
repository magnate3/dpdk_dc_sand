# noqa: D100
import logging
import subprocess


def pytest_sessionfinish():
    """Run pdflatex to generate a PDF of the logged latex."""
    logging.shutdown()  # If we don't do this, the output files aren't closed yet.
    subprocess.run(["pdflatex", "report.tex"], cwd="report")
