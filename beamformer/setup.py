#!/usr/bin/env python
"""Setup File for installing as a module."""

from setuptools import setup, find_packages

setup(
    name="Beamformer Reorder",
    version="0.1",
    description="Beamformer Reorder",
    author="Avdbyl",
    author_email="avanderbyl@ska.ac.za",
    url="",  # Blank until we have a website for this.
    packages=find_packages(),
    package_data={
        "": ["kernels/*.mako"]
    },  # This line does not work as expected when using pip install - see MANIFEST.in file for fix.
    include_package_data=True,
    classifiers=[
        # "License :: OSI Approved :: GNU General Public License v2 (GPLv2)", # TBD before taking this repo public
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.6",
)
