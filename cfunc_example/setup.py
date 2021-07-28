from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup(
    ext_package="cfunc_example",
    ext_modules=[
        Pybind11Extension(
            "consumer",
            sources=["src/cfunc_example/consumer.cpp"],
            cxx_std=17,
        )
    ],
)
