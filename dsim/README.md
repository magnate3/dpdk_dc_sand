# DSim control

This module aims to provide a platform-agnostic interface in controlling the two options of DSim for [katgpucbf](https://github.com/ska-sa/katgpucbf)'s integration testing. These options are namely:
1. The traditional SKARAB-based DSim *(link AvdB's doc here)*, and
2. The [software-based DSim](https://github.com/ska-sa/katgpucbf/blob/main/src/tools/dsim.cpp).

## SKARAB DSim

As per the linked document above, the SKARAB-based DSim has its roots in the age of MeerKAT development. Therefore the methods required to interact with it have been parsed from existing packages.
- Its top-level interface and control has been extracted from [corr2](https://github.com/ska-sa/corr2).
- This, in turn, requires the [casperfpga](https://github.com/ska-sa/casperfpga) Python package to facilitate communications with the SKARAB.

## Software DSim

This DSim was initially created when the GPU F-Engine was being developed. It is fairly simple in its operation and will likely require development to bring its feature set up to parity with its SKARAB counterpart.

## For usage and developing

As mentioned above, the `requirements.txt` does indicate the entire [casperfpga package](https://github.com/ska-sa/casperfpga/tree/python38_updates) as a requirement. This is to facilitate the underlying register reads and writes required to interface with the SKARAB platform. I would recommend creating a virtual environment to work in when using/developing with this package.

Once you've done so, you can `pip install -e .` in this directory (with the `setup.py`) and connect to a DSim *of your choice.

*Currently only the SKARAB DSim is supported, and this can be tested by running the `control_dsim.py` script.

---

## TODO
* Add a `dev-setup.sh` file similar to that in [katgpucbf](https://github.com/ska-sa/katgpucbf/blob/main/dev-setup.sh).
